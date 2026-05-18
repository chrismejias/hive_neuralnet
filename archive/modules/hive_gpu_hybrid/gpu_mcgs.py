"""
Monte Carlo Graph Search (MCGS) for GPU-accelerated Hive training.

Based on Czech et al. (2021) "Monte-Carlo Graph Search for AlphaZero" and
KataGo's graph search documentation.  Key differences from standard MCTS:

1. **DAG instead of tree** – nodes represent unique game states (by hash).
   When two move orderings reach the same position, the node is shared.
2. **Edge-based statistics** – each parent–child edge carries its own visit
   count and Q-value (from the parent's perspective).  PUCT uses per-edge
   counts so each parent explores independently of other parents.
3. **Transposition sharing** – identical states found via different move
   sequences share a single node, saving NN evaluations and propagating
   tactical discoveries across all paths to that state.

Architecture designed for future GPU porting:
- Data structures use integer hashes and could map to GPU hash tables
- Node/edge data in flat arrays (currently Python dicts, portable to CUDA)
- Batch state hashing via CPU numpy (portable to GPU hash kernels)

Usage:
    from hive_gpu.gpu_mcgs import MCGSOrchestrator
    orchestrator = MCGSOrchestrator(net, config)
    examples = orchestrator.self_play_batch()
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass

import numpy as np
import torch

from hive_gnn.graph_types import HiveGraph
from hive_gpu.gpu_mcts import (
    GPUMCTSConfig,
    GPUMCTSOrchestrator,
    GPUTrainingExample,
)
from hive_common.token_types import HiveTokenSequence


# ── MCGS Data Structures ──────────────────────────────────────────────


class MCGSEdge:
    """Edge in the MCGS DAG: a (parent, action) → child relationship.

    Stores per-edge visit count and Q-value *from the parent's perspective*.
    This is the core MCGS innovation: each parent maintains its own
    exploration statistics for shared child nodes, preventing the
    information-leak problem of naive graph search.

    GPU-portable: could be stored as flat arrays indexed by edge_id.
    """

    __slots__ = (
        "action",
        "move_bytes",
        "prior",
        "visit_count",
        "total_value",
        "child_hash",
    )

    def __init__(
        self,
        action: int,
        move_bytes: np.ndarray,
        prior: float,
        child_hash: int | None = None,
    ) -> None:
        self.action = action
        self.move_bytes = move_bytes
        self.prior = prior
        self.visit_count: int = 0
        self.total_value: float = 0.0  # Q from parent's perspective
        self.child_hash: int | None = child_hash

    @property
    def mean_value(self) -> float:
        """Q(s, a) from the parent's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCGSNode:
    """Node in the MCGS DAG: a unique game state.

    Unlike GPUMCTSNode (single parent pointer), MCGSNode can have
    multiple parents via edges from different nodes in the DAG.

    GPU-portable: could be stored as flat arrays indexed by node_id.
    """

    __slots__ = (
        "state_hash",
        "edges",
        "visit_count",
        "total_value",
        "nn_value",
        "is_expanded",
        "is_terminal",
        "terminal_value",
    )

    def __init__(self, state_hash: int) -> None:
        self.state_hash = state_hash
        self.edges: dict[int, MCGSEdge] = {}  # action_idx → MCGSEdge
        self.visit_count: int = 0
        self.total_value: float = 0.0  # V from this node's perspective
        self.nn_value: float = 0.0  # Raw NN value (for transposition reuse)
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0

    @property
    def mean_value(self) -> float:
        """V(s) from the current player's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCGSDAG:
    """Per-search DAG mapping state hashes to nodes.

    Each MCTS search (one position) gets its own DAG.  Transpositions
    within the search share nodes; across searches the DAG is discarded.

    GPU-portable: could be implemented as a GPU hash table (cuckoo
    hashing) with flat node/edge arrays.
    """

    __slots__ = ("nodes", "transposition_hits", "nn_evals", "max_nodes")

    def __init__(self, max_nodes: int = 50_000) -> None:
        self.nodes: dict[int, MCGSNode] = {}
        self.transposition_hits: int = 0
        self.nn_evals: int = 0
        self.max_nodes: int = max_nodes

    @property
    def is_full(self) -> bool:
        return len(self.nodes) >= self.max_nodes

    def get_or_create(self, state_hash: int) -> tuple[MCGSNode, bool]:
        """Get existing node or create a new one.  Returns (node, is_new).

        If the DAG is at capacity and the hash is not already present,
        returns (None, False) — the caller should treat this as a leaf
        without creating a new node.
        """
        if state_hash in self.nodes:
            return self.nodes[state_hash], False
        if self.is_full:
            return None, False
        node = MCGSNode(state_hash)
        self.nodes[state_hash] = node
        return node, True

    def __contains__(self, state_hash: int) -> bool:
        return state_hash in self.nodes

    def __getitem__(self, state_hash: int) -> MCGSNode:
        return self.nodes[state_hash]

    def __len__(self) -> int:
        return len(self.nodes)


# ── MCGS Orchestrator ──────────────────────────────────────────────────


class MCGSOrchestrator(GPUMCTSOrchestrator):
    """GPU-accelerated self-play using Monte Carlo Graph Search.

    Subclasses ``GPUMCTSOrchestrator`` and overrides the search-related
    methods (select, expand, backprop, policy extraction) to use a DAG
    with edge-based statistics.  All GPU encoding, move mapping, and
    auxiliary-target computation is inherited unchanged.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        config: GPUMCTSConfig | None = None,
    ) -> None:
        super().__init__(net, config)
        # Accumulated DAG stats across all searches in this batch
        self.total_transposition_hits: int = 0
        self.total_nn_evals: int = 0
        self.total_dag_nodes: int = 0

    # ── Public API (override) ──────────────────────────────────────────

    def self_play_batch(self) -> list[list[GPUTrainingExample]]:
        """Play B games in parallel with MCGS (DAG-based search).

        All B games run together from start to finish.  No slot recycling:
        games that finish early simply become inactive while others continue.
        Playout cap randomization is applied per-iteration (all games in a
        batch use the same sim count) so games stay roughly synchronized.
        """
        B = self.config.batch_size
        cfg = self.config

        root_states = self.ext.create_initial_states(B, cfg.expansion_mask)

        move_numbers = [0] * B
        histories: list[
            list[
                tuple[
                    HiveGraph,
                    np.ndarray,
                    int,
                    np.ndarray,
                    HiveTokenSequence | None,
                    np.ndarray | None,
                ]
            ]
        ] = [[] for _ in range(B)]

        # Reset per-batch stats
        self.total_transposition_hits = 0
        self.total_nn_evals = 0
        self.total_dag_nodes = 0

        # All slots start active
        active = [True] * B

        # Per-iteration playout cap: all games use the same sim count
        fast_sims = cfg.playout_cap_fast_sims or max(
            1, cfg.num_simulations // 8
        )
        if cfg.playout_cap_randomize:
            use_full_playout = (
                np.random.random() < cfg.playout_cap_randomize_prob
            )
            num_sims = cfg.num_simulations if use_full_playout else fast_sims
        else:
            use_full_playout = True
            num_sims = cfg.num_simulations

        # Per-slot result storage (filled when each game ends)
        final_results = np.zeros(B, dtype=np.int32)

        while any(active):
            current_turns = self._get_turns(root_states, B)

            policies, graphs, seqs, roots, nn_priors = self._batched_mcgs(
                root_states, B, active, move_numbers, num_sims=num_sims
            )

            # Mobility for current positions
            mob_tensor, mob_counts = self.ext.compute_mobility_batch(
                root_states, B, False
            )
            mob_np = mob_tensor.cpu().numpy()
            mob_board_counts = mob_counts.cpu().numpy()

            # Record history and select actions
            action_move_bytes = np.zeros(
                (B, self._move_size), dtype=np.uint8
            )

            for i in range(B):
                if not active[i]:
                    continue

                policy = policies[i]
                turn = current_turns[i]

                mob_i = mob_np[i, : mob_board_counts[i]].copy()
                histories[i].append(
                    (graphs[i], policy.copy(), turn, mob_i, seqs[i], nn_priors[i])
                )

                # Select action
                if move_numbers[i] >= cfg.temperature_drop_move:
                    action = int(np.argmax(policy))
                else:
                    action = int(np.random.choice(len(policy), p=policy))

                # Look up move bytes from MCGS root edges
                root = roots[i]
                if root is not None and action in root.edges:
                    action_move_bytes[i] = root.edges[action].move_bytes
                else:
                    centroids = self.ext.compute_centroids_batch(
                        root_states[i : i + 1], 1
                    ).cpu().numpy()
                    cq, cr = int(centroids[0, 0]), int(centroids[0, 1])
                    move_bytes = self._action_to_gpu_move(
                        root_states, i, action, None, cq, cr
                    )
                    if move_bytes is not None:
                        action_move_bytes[i] = move_bytes

            # Release MCGS DAG memory before applying moves
            del roots
            gc.collect()

            # Apply moves on GPU
            moves_t = torch.from_numpy(action_move_bytes).cuda()
            self.ext.apply_moves_batch(root_states, moves_t, B)

            for i in range(B):
                if active[i]:
                    move_numbers[i] += 1

            # Check game over
            results = self.ext.check_results_batch(
                root_states, B
            ).cpu().numpy()

            for i in range(B):
                if not active[i]:
                    continue
                game_over = (
                    results[i] != 0
                    or move_numbers[i] >= cfg.max_game_length
                )
                if game_over:
                    final_results[i] = int(results[i])
                    active[i] = False

        # ── Build training examples for all games at once ──
        # Compute final mobility and queen surround for all games
        fm_tensor, fm_counts = self.ext.compute_mobility_batch(
            root_states, B, True
        )
        fm_np = fm_tensor.cpu().numpy()
        fm_board_counts = fm_counts.cpu().numpy()
        qs_data = self._compute_queen_surround_batch(root_states, B)

        all_examples = self._build_examples(
            histories,
            final_results,
            fm_np,
            fm_board_counts,
            qs_data,
        )

        # Only include non-empty games; filter fast-playout games
        # (they have no history entries since use_full_playout was False)
        return [exs for exs in all_examples if exs]

    # ── MCGS Search ────────────────────────────────────────────────────

    def _batched_mcgs(
        self,
        root_states: torch.Tensor,
        B: int,
        active: list[bool],
        move_numbers: list[int],
        num_sims: int | None = None,
    ) -> tuple[
        list[np.ndarray],
        list[HiveGraph],
        list[HiveTokenSequence | None],
        list[MCGSNode | None],
        list[np.ndarray | None],
    ]:
        """Run MCGS for all active games.

        Returns same format as _batched_mcts:
        (policies, root_graphs, root_seqs, roots, nn_priors)
        """
        cfg = self.config
        if num_sims is None:
            num_sims = cfg.num_simulations

        # Per-game DAGs
        dags: list[MCGSDAG | None] = [
            MCGSDAG(max_nodes=cfg.max_dag_nodes) if active[i] else None
            for i in range(B)
        ]
        roots: list[MCGSNode | None] = [None] * B

        # Hash root states and create root nodes
        root_bytes = root_states.cpu().numpy()
        for i in range(B):
            if dags[i] is None:
                continue
            root_hash = hash(root_bytes[i].tobytes())
            root_node, _ = dags[i].get_or_create(root_hash)
            roots[i] = root_node

        # Expand roots (NN eval + create edges)
        root_graphs, root_seqs = self._expand_roots_mcgs(
            root_states, B, dags, roots, active
        )

        # Extract NN priors before temp/noise
        nn_priors: list[np.ndarray | None] = [None] * B
        for i in range(B):
            if roots[i] is not None and roots[i].edges:
                prior = np.zeros(self._action_space_size, dtype=np.float32)
                for action, edge in roots[i].edges.items():
                    prior[action] = edge.prior
                nn_priors[i] = prior

        # Root policy temperature + Dirichlet noise
        for i in range(B):
            if roots[i] is not None:
                self._apply_root_policy_temp_mcgs(roots[i])
                self._add_dirichlet_noise_mcgs(roots[i])

        # Run wave-parallel MCGS simulations
        self._run_wave_parallel_mcgs(
            root_states, B, dags, roots, num_sims=num_sims
        )

        # Extract policies from root edge visit counts
        policies = []
        for i in range(B):
            if roots[i] is not None:
                policies.append(
                    self._get_policy_mcgs(roots[i], move_numbers[i])
                )
            else:
                policies.append(
                    np.zeros(self._action_space_size, dtype=np.float32)
                )

        # Accumulate DAG stats
        for dag in dags:
            if dag is not None:
                self.total_transposition_hits += dag.transposition_hits
                self.total_nn_evals += dag.nn_evals
                self.total_dag_nodes += len(dag)

        return policies, root_graphs, root_seqs, roots, nn_priors

    # ── Root Expansion ─────────────────────────────────────────────────

    def _expand_roots_mcgs(
        self,
        root_states: torch.Tensor,
        B: int,
        dags: list[MCGSDAG | None],
        roots: list[MCGSNode | None],
        active: list[bool],
    ) -> tuple[list[HiveGraph], list[HiveTokenSequence | None]]:
        """Expand root nodes: encode, NN eval, create edges."""
        # Check terminals
        results = self.ext.check_results_batch(root_states, B).cpu().numpy()
        for i in range(B):
            if roots[i] is not None and results[i] != 0:
                roots[i].is_terminal = True
                roots[i].terminal_value = self._result_to_value(
                    results[i], self._get_turn_single(root_states, i)
                )
                roots[i].is_expanded = True

        # Encode all root states
        root_seqs: list[HiveTokenSequence | None] = [None] * B
        if hasattr(self.encoder, "encode_batch_with_graphs_and_seqs"):
            encoded, root_graphs, root_seqs = (
                self.encoder.encode_batch_with_graphs_and_seqs(root_states, B)
            )
        elif hasattr(self.encoder, "encode_batch_with_graphs"):
            encoded, root_graphs = self.encoder.encode_batch_with_graphs(
                root_states, B
            )
        else:
            encoded = self.encoder.encode_batch(root_states, B)
            root_graphs = [None] * B

        masks, _ = self.ext.generate_legal_mask_batch(root_states, B)

        with torch.no_grad():
            policy_logits, values, *_ = self.net(encoded)

        masks_bool = masks == 0
        policy_logits[masks_bool] = float("-inf")
        action_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        values_np = values.cpu().numpy().flatten()

        # Legal moves for edge creation
        all_legal_moves, all_num_legal = self.ext.generate_legal_moves_batch(
            root_states, B
        )
        all_legal_np = all_legal_moves.cpu().numpy()
        all_num_legal_np = all_num_legal.cpu().numpy()
        centroids = self.ext.compute_centroids_batch(
            root_states, B
        ).cpu().numpy()

        for i in range(B):
            root = roots[i]
            if root is None or root.is_terminal:
                continue

            root.nn_value = float(values_np[i])
            n_legal = all_num_legal_np[i]
            legal_moves_raw = all_legal_np[i]
            probs = action_probs[i]
            cq, cr = int(centroids[i, 0]), int(centroids[i, 1])

            for mi in range(n_legal):
                move_raw = legal_moves_raw[mi]
                action_idx = self._gpu_move_to_action(move_raw, cq, cr)
                if action_idx is None or action_idx < 0:
                    continue
                if action_idx not in root.edges:
                    edge = MCGSEdge(
                        action=action_idx,
                        move_bytes=move_raw.copy(),
                        prior=float(probs[action_idx]),
                    )
                    root.edges[action_idx] = edge

            root.is_expanded = True
            if dags[i] is not None:
                dags[i].nn_evals += 1

        return root_graphs, root_seqs

    # ── Wave-Parallel MCGS Simulations ─────────────────────────────────

    def _run_wave_parallel_mcgs(
        self,
        root_states: torch.Tensor,
        B: int,
        dags: list[MCGSDAG | None],
        roots: list[MCGSNode | None],
        num_sims: int | None = None,
    ) -> None:
        """Run MCGS simulations in waves of W parallel sims per game.

        Each wave:
        1. SELECT W leaves per game using edge-based PUCT + virtual loss
        2. REPLAY move paths on GPU to obtain leaf states
        3. CHECK TERMINALS on GPU
        4. HASH leaf states and check DAG for transpositions
        5. NN EVAL only genuinely new states (sub-batch)
        6. EXPAND new nodes with edges
        7. BACKPROP values along edges
        """
        cfg = self.config
        if num_sims is None:
            num_sims = cfg.num_simulations
        W = cfg.wave_size
        num_waves = math.ceil(num_sims / W)
        WB = W * B
        state_size = root_states.shape[1]

        # Pre-allocate leaf state buffer
        leaf_states_wb = root_states.new_zeros(WB, state_size)

        def _cuda_sync_check():
            """Synchronize and check for CUDA errors early."""
            if root_states.is_cuda:
                torch.cuda.synchronize()

        for wave in range(num_waves):
            actual_w = min(W, num_sims - wave * W)
            total = actual_w * B

            # ═══ PHASE 1: SELECT W leaves per game ═══
            # path: list of (MCGSNode, MCGSEdge) from root toward leaf
            all_paths: list[list[list[tuple[MCGSNode, MCGSEdge]]]] = []
            all_vl_edges: list[list[list[MCGSEdge]]] = []
            all_move_paths: list[list[list[np.ndarray]]] = []
            all_leaf_nodes: list[list[MCGSNode | None]] = []

            for w in range(actual_w):
                paths_w: list[list[tuple[MCGSNode, MCGSEdge]]] = [
                    [] for _ in range(B)
                ]
                vl_edges_w: list[list[MCGSEdge]] = [[] for _ in range(B)]
                move_paths_w: list[list[np.ndarray]] = [[] for _ in range(B)]
                leaf_nodes_w: list[MCGSNode | None] = [None] * B

                for i in range(B):
                    if roots[i] is None:
                        continue
                    leaf, path, vl_edges, move_path = self._select_mcgs(
                        roots[i], dags[i]
                    )
                    paths_w[i] = path
                    vl_edges_w[i] = vl_edges
                    move_paths_w[i] = move_path
                    leaf_nodes_w[i] = leaf

                all_paths.append(paths_w)
                all_vl_edges.append(vl_edges_w)
                all_move_paths.append(move_paths_w)
                all_leaf_nodes.append(leaf_nodes_w)

            # ═══ PHASE 2: REPLAY LEAF STATES ON GPU ═══
            for w in range(actual_w):
                leaf_states_wb[w * B : (w + 1) * B].copy_(root_states)

            max_depth = 0
            for w in range(actual_w):
                for i in range(B):
                    md = len(all_move_paths[w][i])
                    if md > max_depth:
                        max_depth = md

            for d in range(max_depth):
                # Collect only slots that have a move at this depth
                active_indices = []
                active_moves = []
                for w in range(actual_w):
                    for i in range(B):
                        flat_idx = w * B + i
                        if d < len(all_move_paths[w][i]):
                            active_indices.append(flat_idx)
                            active_moves.append(all_move_paths[w][i][d])

                if not active_indices:
                    continue

                # Build contiguous sub-batch of only active states
                idx_tensor = torch.tensor(
                    active_indices, dtype=torch.long,
                    device=leaf_states_wb.device,
                )
                sub_states = leaf_states_wb[idx_tensor].clone()
                n_active = len(active_indices)

                move_arr = np.zeros(
                    (n_active, self._move_size), dtype=np.uint8
                )
                for j, mv in enumerate(active_moves):
                    move_arr[j] = mv

                dm_tensor = torch.from_numpy(move_arr).cuda()
                self.ext.apply_moves_batch(sub_states, dm_tensor, n_active)
                _cuda_sync_check()  # catch apply_moves errors immediately

                # Write back to leaf_states_wb
                leaf_states_wb[idx_tensor] = sub_states

            # ═══ PHASE 3: CHECK TERMINALS ═══
            leaf_results = self.ext.check_results_batch(
                leaf_states_wb[:total], total
            )
            _cuda_sync_check()  # catch check_results errors immediately
            leaf_results = leaf_results.cpu().numpy()

            # Classify: which leaves need hash resolution?
            needs_hash_check: list[tuple[int, int, int]] = []  # (w, i, flat)

            for w in range(actual_w):
                for i in range(B):
                    if roots[i] is None:
                        continue
                    flat_idx = w * B + i
                    leaf = all_leaf_nodes[w][i]

                    if leaf is not None and (
                        leaf.is_terminal or leaf.is_expanded
                    ):
                        # Already resolved: terminal or re-selected expanded
                        if leaf.is_terminal:
                            pass  # value known
                        # else: already expanded, use mean_value
                        continue

                    # Unresolved: either leaf is None (edge to unknown child)
                    # or leaf is a known but unexpanded node
                    needs_hash_check.append((w, i, flat_idx))

            # ═══ PHASE 4: HASH + TRANSPOSITION CHECK ═══
            needs_nn: list[tuple[int, int, int, MCGSNode]] = []
            transposition_values: dict[tuple[int, int], float] = {}

            if needs_hash_check:
                # Gather only needed states for hashing
                hash_flat_indices = [fi for _, _, fi in needs_hash_check]
                hash_idx_tensor = torch.tensor(
                    hash_flat_indices,
                    dtype=torch.long,
                    device=leaf_states_wb.device,
                )
                hash_states_cpu = (
                    leaf_states_wb[hash_idx_tensor].cpu().numpy()
                )

                for batch_pos, (w, i, flat_idx) in enumerate(
                    needs_hash_check
                ):
                    state_hash = hash(hash_states_cpu[batch_pos].tobytes())

                    # Link parent edge to this child hash
                    path = all_paths[w][i]
                    if path:
                        _, last_edge = path[-1]
                        last_edge.child_hash = state_hash

                    # Check DAG for transposition
                    dag = dags[i]
                    if state_hash in dag and dag[state_hash].is_expanded:
                        # Transposition to an already-expanded node
                        existing = dag[state_hash]
                        all_leaf_nodes[w][i] = existing
                        dag.transposition_hits += 1
                        tv = (
                            existing.mean_value
                            if existing.visit_count > 0
                            else existing.nn_value
                        )
                        transposition_values[(w, i)] = tv
                    else:
                        # New state (or unexpanded from same wave)
                        if state_hash in dag:
                            new_node = dag[state_hash]
                        elif dag.is_full:
                            # DAG at capacity — skip this leaf entirely
                            continue
                        else:
                            new_node = MCGSNode(state_hash)
                            dag.nodes[state_hash] = new_node
                        all_leaf_nodes[w][i] = new_node

                        if leaf_results[flat_idx] != 0:
                            # Terminal
                            new_node.is_terminal = True
                            new_node.terminal_value = (
                                self._result_to_value(
                                    leaf_results[flat_idx],
                                    self._get_turn_single(
                                        leaf_states_wb, flat_idx
                                    ),
                                )
                            )
                        elif not new_node.is_expanded:
                            needs_nn.append((w, i, flat_idx, new_node))

            # ═══ PHASE 5: NN EVAL (only new states) ═══
            nn_values: dict[tuple[int, int], float] = {}

            if needs_nn:
                nn_flat_indices = [fi for _, _, fi, _ in needs_nn]
                nn_idx_tensor = torch.tensor(
                    nn_flat_indices,
                    dtype=torch.long,
                    device=leaf_states_wb.device,
                )
                sub_states = leaf_states_wb[nn_idx_tensor]
                N = len(nn_flat_indices)

                encoded = self.encoder.encode_batch(sub_states, N)
                _cuda_sync_check()  # catch encode errors immediately
                masks, _ = self.ext.generate_legal_mask_batch(sub_states, N)
                _cuda_sync_check()  # catch legal mask errors immediately

                with torch.no_grad():
                    if (
                        self.config.nn_max_batch > 0
                        and N > self.config.nn_max_batch
                    ):
                        policy_logits, values = self._nn_forward_subbatched(
                            encoded, N
                        )
                    else:
                        policy_logits, values, *_ = self.net(encoded)

                masks_bool = masks == 0
                policy_logits[masks_bool] = float("-inf")
                action_probs = (
                    torch.softmax(policy_logits, dim=-1).cpu().numpy()
                )
                values_np = values.cpu().numpy().flatten()

                # Legal moves for edge creation
                all_legal_moves, all_num_legal = (
                    self.ext.generate_legal_moves_batch(sub_states, N)
                )
                all_legal_np = all_legal_moves.cpu().numpy()
                all_num_legal_np = all_num_legal.cpu().numpy()
                leaf_centroids = self.ext.compute_centroids_batch(
                    sub_states, N
                ).cpu().numpy()

                # ═══ PHASE 6: EXPAND ═══
                for idx_in_batch, (w, i, flat_idx, new_node) in enumerate(
                    needs_nn
                ):
                    new_node.nn_value = float(values_np[idx_in_batch])
                    nn_values[(w, i)] = new_node.nn_value

                    n_legal = all_num_legal_np[idx_in_batch]
                    legal_raw = all_legal_np[idx_in_batch]
                    probs = action_probs[idx_in_batch]
                    cq = int(leaf_centroids[idx_in_batch, 0])
                    cr = int(leaf_centroids[idx_in_batch, 1])

                    for mi in range(n_legal):
                        move_raw = legal_raw[mi]
                        action_idx = self._gpu_move_to_action(
                            move_raw, cq, cr
                        )
                        if action_idx is None or action_idx < 0:
                            continue
                        if action_idx not in new_node.edges:
                            edge = MCGSEdge(
                                action=action_idx,
                                move_bytes=move_raw.copy(),
                                prior=float(probs[action_idx]),
                            )
                            new_node.edges[action_idx] = edge

                    new_node.is_expanded = True
                    if dags[i] is not None:
                        dags[i].nn_evals += 1

            # ═══ PHASE 7: BACKPROP ═══
            for w in range(actual_w):
                for i in range(B):
                    if roots[i] is None:
                        continue

                    leaf = all_leaf_nodes[w][i]
                    path = all_paths[w][i]
                    vl_edges = all_vl_edges[w][i]

                    if leaf is None:
                        value = 0.0
                    elif leaf.is_terminal:
                        value = leaf.terminal_value
                    elif (w, i) in transposition_values:
                        value = transposition_values[(w, i)]
                    elif (w, i) in nn_values:
                        value = nn_values[(w, i)]
                    else:
                        # Re-selected an already-expanded node
                        value = (
                            leaf.mean_value
                            if leaf.visit_count > 0
                            else 0.0
                        )

                    self._backpropagate_mcgs(path, vl_edges, leaf, value)

    # ── Selection (edge-based PUCT) ────────────────────────────────────

    def _select_mcgs(
        self,
        root: MCGSNode,
        dag: MCGSDAG,
    ) -> tuple[
        MCGSNode | None,
        list[tuple[MCGSNode, MCGSEdge]],
        list[MCGSEdge],
        list[np.ndarray],
    ]:
        """Select a leaf using edge-based PUCT in the DAG.

        Returns:
            (leaf_node, path, vl_edges, move_path)

            leaf_node: the leaf MCGSNode, or None if the last edge
                       points to an unresolved child (needs hash check).
            path: list of (parent_node, edge) from root toward leaf.
            vl_edges: edges with virtual loss applied (for undo).
            move_path: move_bytes for GPU state replay.
        """
        path: list[tuple[MCGSNode, MCGSEdge]] = []
        vl_edges: list[MCGSEdge] = []
        move_path: list[np.ndarray] = []

        node = root
        # Track visited node hashes to detect cycles in the DAG.
        # Hive positions can repeat (pieces moving back and forth),
        # creating genuine cycles that would loop selection forever.
        visited: set[int] = {root.state_hash}
        # Limit selection depth to prevent excessively deep move-replay
        # paths that stress the GPU (each depth = one apply_moves_batch).
        max_select_depth = 50
        depth = 0
        while depth < max_select_depth:
            if node.is_terminal:
                break
            if not node.is_expanded or not node.edges:
                break

            # Compute parent visit sum once for PUCT
            n_parent = sum(e.visit_count for e in node.edges.values())

            # Pick edge with highest PUCT score, skipping edges that
            # would revisit an already-visited node (cycle avoidance)
            best_edge: MCGSEdge | None = None
            best_score = float("-inf")
            for edge in node.edges.values():
                if (
                    edge.child_hash is not None
                    and edge.child_hash in visited
                ):
                    continue
                score = self._puct_score_edge(n_parent, edge)
                if score > best_score:
                    best_score = score
                    best_edge = edge

            if best_edge is None:
                break

            # Virtual loss on edge: makes Q more negative (discourages
            # parallel waves from selecting the same edge)
            best_edge.visit_count += 1
            best_edge.total_value -= 1.0
            vl_edges.append(best_edge)

            path.append((node, best_edge))

            if best_edge.move_bytes is not None:
                move_path.append(best_edge.move_bytes)

            # Follow to child node
            if (
                best_edge.child_hash is not None
                and best_edge.child_hash in dag
            ):
                node = dag[best_edge.child_hash]
                visited.add(node.state_hash)
                depth += 1
                # Continue selection through the child
            else:
                # Unresolved edge → need GPU replay + hash check
                node = None
                break

        return node, path, vl_edges, move_path

    def _puct_score_edge(
        self, n_parent: int, edge: MCGSEdge
    ) -> float:
        """PUCT score using edge statistics.

        Q is already stored from the parent's perspective on the edge,
        so no negation is needed (unlike tree MCTS where child values
        must be negated).
        """
        q = edge.mean_value if edge.visit_count > 0 else 0.0
        exploration = (
            self.config.c_puct
            * edge.prior
            * math.sqrt(n_parent)
            / (1 + edge.visit_count)
        )
        return q + exploration

    # ── Backpropagation (edge-based) ───────────────────────────────────

    def _backpropagate_mcgs(
        self,
        path: list[tuple[MCGSNode, MCGSEdge]],
        vl_edges: list[MCGSEdge],
        leaf_node: MCGSNode | None,
        value: float,
    ) -> None:
        """Undo virtual loss and backpropagate value along edges.

        Value convention:
        - ``value`` is from the leaf's current player's perspective.
        - Edge total_value stores Q from the parent's perspective.
        - Node total_value stores V from the node's own perspective.

        We walk the path from leaf to root, alternating the sign.
        """
        # Undo virtual loss
        for edge in vl_edges:
            edge.visit_count -= 1
            edge.total_value += 1.0

        # Update leaf node
        if leaf_node is not None:
            leaf_node.visit_count += 1
            leaf_node.total_value += value

        # Walk from leaf's parent up to root
        # path = [(root, edge0), ..., (parent_of_leaf, edge_to_leaf)]
        v = value
        for parent_node, edge in reversed(path):
            v = -v  # Flip: now from parent's perspective
            edge.visit_count += 1
            edge.total_value += v
            parent_node.visit_count += 1
            parent_node.total_value += v

    # ── Root Policy Temperature ────────────────────────────────────────

    def _apply_root_policy_temp_mcgs(self, node: MCGSNode) -> None:
        """Soften root edge priors with temperature > 1.0."""
        temp = self.config.root_policy_temp
        if temp <= 1.0 or not node.edges:
            return

        actions = list(node.edges.keys())
        priors = np.array(
            [node.edges[a].prior for a in actions], dtype=np.float64
        )
        priors = priors ** (1.0 / temp)
        total = priors.sum()
        if total > 0:
            priors /= total
        else:
            priors[:] = 1.0 / len(priors)

        for i, action in enumerate(actions):
            node.edges[action].prior = float(priors[i])

    # ── Dirichlet Noise ────────────────────────────────────────────────

    def _add_dirichlet_noise_mcgs(self, node: MCGSNode) -> None:
        """Add Dirichlet noise to root edge priors."""
        if not node.edges:
            return

        actions = list(node.edges.keys())
        n_legal = len(actions)

        if self.config.shaped_dirichlet:
            alpha = max(0.03, 10.0 / n_legal)
        else:
            alpha = self.config.dirichlet_alpha

        noise = np.random.dirichlet([alpha] * n_legal)
        eps = self.config.dirichlet_epsilon

        for i, action in enumerate(actions):
            edge = node.edges[action]
            edge.prior = (1 - eps) * edge.prior + eps * noise[i]

    # ── Policy Extraction ──────────────────────────────────────────────

    def _get_policy_mcgs(
        self, root: MCGSNode, move_number: int
    ) -> np.ndarray:
        """Build policy vector from root edge visit counts.

        Applies policy target pruning (same as tree MCTS).
        """
        policy = np.zeros(self._action_space_size, dtype=np.float32)
        if not root.edges:
            return policy

        temp = self.config.temperature
        if move_number >= self.config.temperature_drop_move:
            temp = 0.0

        if temp == 0.0:
            best_action = max(
                root.edges,
                key=lambda a: root.edges[a].visit_count,
            )
            policy[best_action] = 1.0
        else:
            actions = sorted(root.edges.keys())
            visits = np.array(
                [root.edges[a].visit_count for a in actions],
                dtype=np.float64,
            )

            # Policy target pruning
            pruning_threshold = self.config.policy_target_pruning
            if pruning_threshold > 0 and visits.max() > 0:
                cutoff = visits.max() * pruning_threshold
                visits[visits < cutoff] = 0.0

            if temp != 1.0:
                visits = visits ** (1.0 / temp)

            total = visits.sum()
            if total > 0:
                visits /= total
                for i, action in enumerate(actions):
                    policy[action] = visits[i]
            else:
                for action in actions:
                    policy[action] = 1.0 / len(actions)

        return policy
