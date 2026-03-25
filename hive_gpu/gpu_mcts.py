"""
GPU-accelerated MCTS with batched neural network inference.

Architecture: CPU-side MCTS trees + GPU batched inference.
All B games run in lockstep: each simulation step selects leaves in all
trees, encodes them in one GPU batch, runs one NN forward pass, then
expands and backpropagates on CPU.

Usage:
    from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig

    orchestrator = GPUMCTSOrchestrator(net, config)
    examples = orchestrator.self_play_batch()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

import hive_gpu
try:
    from hive_gnn.graph_types import HiveGraph
except ImportError:
    HiveGraph = None  # type: ignore[assignment,misc]
from hive_gpu.gpu_encoder import GPUGNNEncoder, GPUTransformerEncoder
from hive_transformer.token_types import HiveTokenSequence


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class GPUMCTSConfig:
    """Configuration for GPU-accelerated MCTS."""

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_drop_move: int = 20
    batch_size: int = 64       # B: number of parallel GPU game slots
    games_per_batch: int = 0   # target games per self-play call (0 = same as batch_size)
    max_game_length: int = 300
    encoder_type: str = "gnn"  # "gnn" or "transformer"
    expansion_mask: int = 0    # 3-bit: bit 0=Mosquito, 1=Ladybug, 2=Pillbug
    wave_size: int = 8         # W: parallel sims per wave (1 = sequential)
    nn_max_batch: int = 0      # max NN batch size (0 = no limit)

    # Playout cap randomization (KataGo-style)
    playout_cap_randomize: bool = True
    playout_cap_randomize_prob: float = 0.25   # prob of full playouts; else fast
    playout_cap_fast_sims: int = 0             # 0 = auto (num_simulations // 8)

    # Policy target pruning (zero out visits below threshold fraction of max)
    policy_target_pruning: float = 0.02

    # Root policy temperature (soften NN prior before Dirichlet noise)
    root_policy_temp: float = 1.1

    # Shaped Dirichlet noise (scale alpha inversely with legal move count)
    shaped_dirichlet: bool = True

    # MCGS DAG node cap per game (prevents OOM from unbounded DAG growth)
    max_dag_nodes: int = 50_000


# ── Training Example ──────────────────────────────────────────────────


class GPUTrainingExample(NamedTuple):
    """A training example produced by GPU self-play."""

    graph: HiveGraph              # GNN graph representation of the state
    policy_target: np.ndarray     # (29914,) MCTS visit distribution
    value_target: float           # +1 (won), -1 (lost), 0 (draw)
    mobility_target: np.ndarray          # (num_board_nodes,) float32
    queen_surround_target: np.ndarray    # (num_board_nodes, 2) float32
    queen_surround_mask: np.ndarray      # (2,) float32
    final_mobility_target: np.ndarray    # (num_board_nodes,) float32
    sequence: HiveTokenSequence | None = None  # Transformer token sequence (None for GNN path)
    nn_prior: np.ndarray | None = None   # NN policy prior for surprise weighting


# ── MCTS Node (CPU-side, no GameState) ─────────────────────────────────


class GPUMCTSNode:
    """
    Lightweight MCTS tree node.

    Unlike the CPU MCTSNode, this does not store a GameState. Game states
    are replayed on GPU from the root state using the move path from root
    to leaf.
    """

    __slots__ = (
        "children",
        "visit_count",
        "total_value",
        "prior",
        "is_expanded",
        "parent",
        "parent_action",
        "move_bytes",
        "is_terminal",
        "terminal_value",
    )

    def __init__(
        self,
        parent: GPUMCTSNode | None = None,
        parent_action: int | None = None,
        prior: float = 0.0,
        move_bytes: np.ndarray | None = None,
    ) -> None:
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior
        self.move_bytes = move_bytes  # 6-byte GPUMove for GPU replay
        self.children: dict[int, GPUMCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


# ── GPU MCTS Orchestrator ──────────────────────────────────────────────


class GPUMCTSOrchestrator:
    """
    Runs batched MCTS self-play on GPU.

    All B games are played in lockstep. Each MCTS simulation:
    1. CPU: Select leaf node in each tree (PUCT)
    2. GPU: Replay move paths to get leaf states
    3. GPU: Encode leaf states + generate legal masks
    4. GPU: NN forward pass (batched)
    5. CPU: Expand leaves + backpropagate values
    """

    def __init__(
        self,
        net: torch.nn.Module,
        config: GPUMCTSConfig | None = None,
    ) -> None:
        self.ext = hive_gpu.load_extension()
        self.config = config or GPUMCTSConfig()

        # Use torch.compile for faster NN forward passes if available.
        # Falls back to eager mode on Windows (no Triton) or other failures.
        self.net = net
        try:
            import triton  # noqa: F401
            self.net = torch.compile(net)
        except (ImportError, Exception):
            pass

        if self.config.encoder_type == "gnn":
            self.encoder = GPUGNNEncoder()
        else:
            self.encoder = GPUTransformerEncoder()

        self._action_space_size = self.ext.ACTION_SPACE_SIZE
        self._move_size = self.ext.SIZEOF_GPU_MOVE

    # ── Public API ─────────────────────────────────────────────────────

    def self_play_batch(self) -> list[list[GPUTrainingExample]]:
        """
        Play B games in parallel with batched MCTS.

        Returns:
            List of B lists of training examples (one per game).
        """
        B = self.config.batch_size
        cfg = self.config

        # Initialize B game states on GPU
        root_states = self.ext.create_initial_states(B, cfg.expansion_mask)

        # Game tracking
        active = [True] * B
        move_numbers = [0] * B
        # Per-game history: list of (graph, policy, turn, mobility, seq, nn_prior) tuples
        histories: list[list[tuple[HiveGraph, np.ndarray, int, np.ndarray, HiveTokenSequence | None, np.ndarray | None]]] = [
            [] for _ in range(B)
        ]

        while any(active):
            # Get current player turns for recording
            current_turns = self._get_turns(root_states, B)

            # Playout cap randomization (KataGo-style): each move uses either
            # full simulations (prob = playout_cap_randomize_prob) or fast sims.
            # Only full-playout moves are recorded as training examples.
            if cfg.playout_cap_randomize:
                use_full = np.random.random() < cfg.playout_cap_randomize_prob
                fast_sims = cfg.playout_cap_fast_sims or max(1, cfg.num_simulations // 8)
                num_sims = cfg.num_simulations if use_full else fast_sims
            else:
                use_full = True
                num_sims = cfg.num_simulations

            # Run MCTS for all active games.
            # _batched_mcts returns root graphs (from the _expand_roots encode call),
            # optional token sequences (transformer path only), and trees (for cached
            # action→move_bytes lookup).
            policies, graphs, seqs, trees, nn_priors = self._batched_mcts(
                root_states, B, active, move_numbers, num_sims=num_sims
            )

            # Compute per-piece mobility for current positions
            mob_tensor, mob_counts = self.ext.compute_mobility_batch(
                root_states, B, False
            )
            mob_np = mob_tensor.cpu().numpy()       # [B, MAX_ENC_NODES]
            mob_board_counts = mob_counts.cpu().numpy()  # [B]

            # Record history and select actions
            action_move_bytes = np.zeros((B, self._move_size), dtype=np.uint8)

            for i in range(B):
                if not active[i]:
                    continue

                policy = policies[i]
                turn = current_turns[i]

                # Record (graph, policy, turn, mobility, seq, nn_prior).
                # With playout cap randomization, only record full-playout moves.
                if use_full:
                    mob_i = mob_np[i, :mob_board_counts[i]].copy()
                    histories[i].append((graphs[i], policy.copy(), turn, mob_i, seqs[i], nn_priors[i]))

                # Select action
                if move_numbers[i] >= cfg.temperature_drop_move:
                    action = int(np.argmax(policy))
                else:
                    action = int(np.random.choice(len(policy), p=policy))

                # Look up move bytes from MCTS tree (O(1) dict lookup)
                tree = trees[i]
                if tree is not None and action in tree.children:
                    action_move_bytes[i] = tree.children[action].move_bytes
                else:
                    # Fallback: generate legal moves (shouldn't normally happen)
                    centroids = self.ext.compute_centroids_batch(
                        root_states[i:i+1], 1
                    ).cpu().numpy()
                    cq, cr = int(centroids[0, 0]), int(centroids[0, 1])
                    move_bytes = self._action_to_gpu_move(
                        root_states, i, action, None, cq, cr
                    )
                    if move_bytes is not None:
                        action_move_bytes[i] = move_bytes

            # Apply moves on GPU (all games at once)
            moves_t = torch.from_numpy(action_move_bytes).cuda()
            self.ext.apply_moves_batch(root_states, moves_t, B)

            # Update move numbers
            for i in range(B):
                if active[i]:
                    move_numbers[i] += 1

            # Check for game over
            results = self.ext.check_results_batch(root_states, B).cpu().numpy()
            for i in range(B):
                if not active[i]:
                    continue
                if results[i] != 0:  # 0 = IN_PROGRESS
                    active[i] = False
                elif move_numbers[i] >= cfg.max_game_length:
                    active[i] = False

        # Build training examples from histories
        final_results = self.ext.check_results_batch(root_states, B).cpu().numpy()

        # Compute game-end auxiliary targets
        # Final mobility: check both players' pieces
        final_mob_tensor, final_mob_counts = self.ext.compute_mobility_batch(
            root_states, B, True  # both_players=True
        )
        final_mob_np = final_mob_tensor.cpu().numpy()
        final_mob_board_counts = final_mob_counts.cpu().numpy()

        # Queen surround: CPU-side parsing of final states
        final_qs_data = self._compute_queen_surround_batch(root_states, B)

        return self._build_examples(
            histories, final_results,
            final_mob_np, final_mob_board_counts,
            final_qs_data,
        )

    # ── Batched MCTS ───────────────────────────────────────────────────

    def _batched_mcts(
        self,
        root_states: torch.Tensor,
        B: int,
        active: list[bool],
        move_numbers: list[int],
        num_sims: int | None = None,
    ) -> tuple[list[np.ndarray], list[HiveGraph], list[HiveTokenSequence | None], list[GPUMCTSNode | None], list[np.ndarray | None]]:
        """
        Run num_simulations of batched MCTS.

        Args:
            root_states: [B, sizeof(HiveState)] GPU tensor
            B: batch size
            active: which games are still in progress
            move_numbers: current move number per game

        Returns:
            (policies, root_graphs, root_seqs, trees, nn_priors): List of B
            policy vectors, HiveGraph objects, HiveTokenSequence (or None for
            GNN path), MCTS trees, and NN policy priors (for surprise weighting).
        """
        cfg = self.config
        if num_sims is None:
            num_sims = cfg.num_simulations

        # Create fresh trees for each position
        trees = [GPUMCTSNode() if active[i] else None for i in range(B)]

        # Expand all roots first; returns root-state graphs (and sequences for transformer)
        root_graphs, root_seqs = self._expand_roots(root_states, B, trees, active)

        # Extract NN priors BEFORE temperature/noise modification (for surprise weighting)
        nn_priors: list[np.ndarray | None] = [None] * B
        for i in range(B):
            if trees[i] is not None and trees[i].children:
                prior = np.zeros(self._action_space_size, dtype=np.float32)
                for action, child in trees[i].children.items():
                    prior[action] = child.prior
                nn_priors[i] = prior

        # Soften root priors with temperature, then add Dirichlet noise
        for i in range(B):
            if trees[i] is not None:
                self._apply_root_policy_temp(trees[i])
                self._add_dirichlet_noise(trees[i])

        # Run simulations — wave-parallel or sequential
        if cfg.wave_size > 1:
            self._run_wave_parallel_sims(root_states, B, trees, num_sims=num_sims)
        else:
            self._run_sequential_sims(root_states, B, trees, num_sims=num_sims)

        # Extract policies from visit counts
        policies = []
        for i in range(B):
            if trees[i] is not None:
                policies.append(self._get_policy(trees[i], move_numbers[i]))
            else:
                policies.append(np.zeros(self._action_space_size, dtype=np.float32))

        return policies, root_graphs, root_seqs, trees, nn_priors

    # ── Simulation loops ────────────────────────────────────────────────

    def _run_sequential_sims(
        self,
        root_states: torch.Tensor,
        B: int,
        trees: list[GPUMCTSNode | None],
        num_sims: int | None = None,
    ) -> None:
        """Run MCTS simulations sequentially (one sim per round-trip)."""
        cfg = self.config
        if num_sims is None:
            num_sims = cfg.num_simulations
        leaf_states = root_states.clone()

        for sim in range(num_sims):
            # 1. SELECT: find leaf in each tree
            leaves: list[GPUMCTSNode | None] = [None] * B
            move_paths: list[list[np.ndarray]] = [[] for _ in range(B)]
            vl_paths: list[list[GPUMCTSNode]] = [[] for _ in range(B)]

            for i in range(B):
                if trees[i] is None:
                    continue
                leaf, vl_path, path_moves = self._select(trees[i])
                leaves[i] = leaf
                vl_paths[i] = vl_path
                move_paths[i] = path_moves

            # 2. PREPARE LEAF STATES: replay move paths on GPU
            leaf_states.copy_(root_states)
            max_depth = max((len(p) for p in move_paths), default=0)
            for d in range(max_depth):
                active_indices = []
                active_moves = []
                for i in range(B):
                    if d < len(move_paths[i]):
                        active_indices.append(i)
                        active_moves.append(move_paths[i][d])

                if not active_indices:
                    continue

                idx_tensor = torch.tensor(
                    active_indices, dtype=torch.long,
                    device=leaf_states.device,
                )
                sub_states = leaf_states[idx_tensor].clone()
                n_active = len(active_indices)

                move_arr = np.zeros(
                    (n_active, self._move_size), dtype=np.uint8
                )
                for j, mv in enumerate(active_moves):
                    move_arr[j] = mv

                dm_tensor = torch.from_numpy(move_arr).cuda()
                self.ext.apply_moves_batch(sub_states, dm_tensor, n_active)
                leaf_states[idx_tensor] = sub_states

            # Check which leaves are terminal
            leaf_results = self.ext.check_results_batch(leaf_states, B).cpu().numpy()

            needs_eval: list[int] = []
            for i in range(B):
                if leaves[i] is None:
                    continue
                if leaf_results[i] != 0:
                    leaves[i].is_terminal = True
                    leaves[i].terminal_value = self._result_to_value(
                        leaf_results[i], self._get_turn_single(leaf_states, i)
                    )
                elif not leaves[i].is_expanded:
                    needs_eval.append(i)

            # 3 & 4. ENCODE + NN FORWARD for non-terminal leaves
            if needs_eval:
                self._evaluate_and_expand(leaf_states, B, leaves, needs_eval)

            # 5. BACKPROPAGATE
            for i in range(B):
                leaf = leaves[i]
                if leaf is None:
                    continue

                if leaf.is_terminal:
                    value = leaf.terminal_value
                elif i in needs_eval:
                    value = float(self._last_values[i])
                else:
                    value = 0.0

                self._backpropagate(leaf, value, vl_paths[i])

    def _run_wave_parallel_sims(
        self,
        root_states: torch.Tensor,
        B: int,
        trees: list[GPUMCTSNode | None],
        num_sims: int | None = None,
    ) -> None:
        """Run MCTS simulations in waves of W parallel sims per game."""
        cfg = self.config
        if num_sims is None:
            num_sims = cfg.num_simulations
        W = cfg.wave_size
        num_waves = math.ceil(num_sims / W)
        WB = W * B
        state_size = root_states.shape[1]

        # Pre-allocate leaf state buffer for W×B states
        leaf_states_wb = root_states.new_zeros(WB, state_size)

        for wave in range(num_waves):
            actual_w = min(W, num_sims - wave * W)
            total = actual_w * B

            # === PHASE 1: SELECT W leaves per game ===
            all_leaves: list[list[GPUMCTSNode | None]] = []
            all_vl_paths: list[list[list[GPUMCTSNode]]] = []
            all_move_paths: list[list[list[np.ndarray]]] = []

            for w in range(actual_w):
                leaves_w: list[GPUMCTSNode | None] = [None] * B
                vl_paths_w: list[list[GPUMCTSNode]] = [[] for _ in range(B)]
                move_paths_w: list[list[np.ndarray]] = [[] for _ in range(B)]

                for i in range(B):
                    if trees[i] is None:
                        continue
                    leaf, vl_path, path_moves = self._select(trees[i])
                    leaves_w[i] = leaf
                    vl_paths_w[i] = vl_path
                    move_paths_w[i] = path_moves

                all_leaves.append(leaves_w)
                all_vl_paths.append(vl_paths_w)
                all_move_paths.append(move_paths_w)

            # === PHASE 2: PREPARE W×B LEAF STATES ===
            # Copy root states W times into flat buffer
            for w in range(actual_w):
                leaf_states_wb[w * B:(w + 1) * B].copy_(root_states)

            # Replay move paths depth-by-depth on all W×B states
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

                # Write back to leaf_states_wb
                leaf_states_wb[idx_tensor] = sub_states

            # === PHASE 3: CHECK TERMINALS ===
            leaf_results = self.ext.check_results_batch(
                leaf_states_wb[:total], total
            ).cpu().numpy()

            needs_eval: list[tuple[int, int, int]] = []  # (w, i, flat_idx)
            for w in range(actual_w):
                for i in range(B):
                    flat_idx = w * B + i
                    leaf = all_leaves[w][i]
                    if leaf is None:
                        continue
                    if leaf_results[flat_idx] != 0:
                        leaf.is_terminal = True
                        leaf.terminal_value = self._result_to_value(
                            leaf_results[flat_idx],
                            self._get_turn_single(leaf_states_wb, flat_idx),
                        )
                    elif not leaf.is_expanded:
                        needs_eval.append((w, i, flat_idx))

            # === PHASE 4: ENCODE + NN + EXPAND ===
            values_np = None
            if needs_eval:
                self._evaluate_and_expand(
                    leaf_states_wb[:total], total,
                    # Build flat leaves list for the expand helper
                    None, None,
                    # Use wave-specific expand path
                    wave_needs_eval=needs_eval,
                    wave_leaves=all_leaves,
                )
                values_np = self._last_values

            # === PHASE 5: BACKPROPAGATE ===
            needs_eval_set = {(w, i) for w, i, _ in needs_eval} if needs_eval else set()
            for w in range(actual_w):
                for i in range(B):
                    leaf = all_leaves[w][i]
                    if leaf is None:
                        continue

                    flat_idx = w * B + i
                    if leaf.is_terminal:
                        value = leaf.terminal_value
                    elif (w, i) in needs_eval_set and values_np is not None:
                        value = float(values_np[flat_idx])
                    else:
                        value = 0.0

                    self._backpropagate(leaf, value, all_vl_paths[w][i])

    def _evaluate_and_expand(
        self,
        leaf_states: torch.Tensor,
        total: int,
        flat_leaves: list[GPUMCTSNode | None] | None,
        needs_eval_flat: list[int] | None,
        *,
        wave_needs_eval: list[tuple[int, int, int]] | None = None,
        wave_leaves: list[list[GPUMCTSNode | None]] | None = None,
    ) -> None:
        """Encode leaf states, run NN, expand non-terminal leaves.

        Supports two calling conventions:
        - Sequential: flat_leaves[i], needs_eval_flat=[i, ...]
        - Wave-parallel: wave_leaves[w][i], wave_needs_eval=[(w,i,flat), ...]

        Stores values in self._last_values for caller to read.
        """
        # Encode
        encoded = self.encoder.encode_batch(leaf_states, total)
        masks, _ = self.ext.generate_legal_mask_batch(leaf_states, total)

        # NN forward (possibly sub-batched)
        with torch.no_grad():
            if self.config.nn_max_batch > 0 and total > self.config.nn_max_batch:
                policy_logits, values = self._nn_forward_subbatched(
                    encoded, total
                )
            else:
                policy_logits, values, *_ = self.net(encoded)

        # Apply mask + softmax
        masks_bool = masks == 0
        policy_logits[masks_bool] = float("-inf")
        action_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        self._last_values = values.cpu().numpy().flatten()

        # Legal moves for child creation
        all_legal_moves, all_num_legal = self.ext.generate_legal_moves_batch(
            leaf_states, total
        )
        all_legal_np = all_legal_moves.cpu().numpy()
        all_num_legal_np = all_num_legal.cpu().numpy()

        leaf_centroids = self.ext.compute_centroids_batch(
            leaf_states, total
        ).cpu().numpy()

        # Build iteration list
        if wave_needs_eval is not None and wave_leaves is not None:
            eval_iter = [
                (wave_leaves[w][i], flat_idx)
                for w, i, flat_idx in wave_needs_eval
            ]
        else:
            eval_iter = [
                (flat_leaves[idx], idx) for idx in (needs_eval_flat or [])
            ]

        # Expand
        for leaf, flat_idx in eval_iter:
            if leaf is None or leaf.is_expanded:
                continue

            n_legal = all_num_legal_np[flat_idx]
            legal_moves_raw = all_legal_np[flat_idx]
            probs = action_probs[flat_idx]
            cq = int(leaf_centroids[flat_idx, 0])
            cr = int(leaf_centroids[flat_idx, 1])

            for mi in range(n_legal):
                move_raw = legal_moves_raw[mi]
                action_idx = self._gpu_move_to_action(move_raw, cq, cr)
                if action_idx is None or action_idx < 0:
                    continue

                if action_idx not in leaf.children:
                    child = GPUMCTSNode(
                        parent=leaf,
                        parent_action=action_idx,
                        prior=float(probs[action_idx]),
                        move_bytes=move_raw.copy(),
                    )
                    leaf.children[action_idx] = child

            leaf.is_expanded = True

    def _nn_forward_subbatched(
        self,
        encoded,
        total: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NN forward in sub-batches, concatenating results."""
        max_batch = self.config.nn_max_batch
        all_policy = []
        all_values = []

        for start in range(0, total, max_batch):
            end = min(start + max_batch, total)
            # Slice the encoded batch — works for HiveTokenBatch (dim 0 slicing)
            chunk = encoded.slice_batch(start, end) if hasattr(encoded, 'slice_batch') else encoded
            policy_logits, values, *_ = self.net(chunk)
            all_policy.append(policy_logits)
            all_values.append(values)

        return torch.cat(all_policy, dim=0), torch.cat(all_values, dim=0)

    # ── Root expansion ─────────────────────────────────────────────────

    def _expand_roots(
        self,
        root_states: torch.Tensor,
        B: int,
        trees: list[GPUMCTSNode | None],
        active: list[bool],
    ) -> tuple[list[HiveGraph], list[HiveTokenSequence | None]]:
        """Expand all root nodes using a single batched NN call.

        Returns:
            (list[HiveGraph], list[HiveTokenSequence | None]) for the root positions.
            Sequences are populated for the transformer encoder path, None otherwise.
        """
        # Check which roots are non-terminal
        results = self.ext.check_results_batch(root_states, B).cpu().numpy()
        for i in range(B):
            if trees[i] is not None and results[i] != 0:
                trees[i].is_terminal = True
                trees[i].terminal_value = self._result_to_value(
                    results[i], self._get_turn_single(root_states, i)
                )
                trees[i].is_expanded = True

        # Encode once — get HiveGraphBatch/HiveTokenBatch (for NN) and
        # list[HiveGraph] (for storage). For transformer, also get token sequences.
        root_seqs: list[HiveTokenSequence | None] = [None] * B
        if hasattr(self.encoder, "encode_batch_with_graphs_and_seqs"):
            encoded, root_graphs, root_seqs = self.encoder.encode_batch_with_graphs_and_seqs(root_states, B)
        elif hasattr(self.encoder, "encode_batch_with_graphs"):
            encoded, root_graphs = self.encoder.encode_batch_with_graphs(root_states, B)
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

        # Get legal moves for child creation
        all_legal_moves, all_num_legal = self.ext.generate_legal_moves_batch(
            root_states, B
        )
        all_legal_np = all_legal_moves.cpu().numpy()
        all_num_legal_np = all_num_legal.cpu().numpy()

        # Compute centroids on GPU (one batch call instead of per-game CPU copies)
        centroids = self.ext.compute_centroids_batch(root_states, B).cpu().numpy()

        for i in range(B):
            tree = trees[i]
            if tree is None or tree.is_terminal:
                continue

            n_legal = all_num_legal_np[i]
            legal_moves_raw = all_legal_np[i]
            probs = action_probs[i]
            cq, cr = int(centroids[i, 0]), int(centroids[i, 1])

            for mi in range(n_legal):
                move_raw = legal_moves_raw[mi]
                action_idx = self._gpu_move_to_action(move_raw, cq, cr)
                if action_idx is None or action_idx < 0:
                    continue

                if action_idx not in tree.children:
                    child = GPUMCTSNode(
                        parent=tree,
                        parent_action=action_idx,
                        prior=float(probs[action_idx]),
                        move_bytes=move_raw.copy(),
                    )
                    tree.children[action_idx] = child

            tree.is_expanded = True

        return root_graphs, root_seqs

    # ── Tree operations ────────────────────────────────────────────────

    def _select(
        self, node: GPUMCTSNode
    ) -> tuple[GPUMCTSNode, list[GPUMCTSNode], list[np.ndarray]]:
        """
        Select a leaf node using PUCT. Returns (leaf, vl_path, move_path).

        move_path: list of move_bytes from root to leaf (for GPU state replay).
        """
        vl_path: list[GPUMCTSNode] = []
        move_path: list[np.ndarray] = []

        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break

            # Pick child with highest PUCT score
            best_action = -1
            best_score = float("-inf")
            for action, child in node.children.items():
                score = self._puct_score(node, child)
                if score > best_score:
                    best_score = score
                    best_action = action

            node = node.children[best_action]

            # Virtual loss
            node.visit_count += 1
            node.total_value -= 1.0
            vl_path.append(node)

            # Record move for GPU replay
            if node.move_bytes is not None:
                move_path.append(node.move_bytes)

        return node, vl_path, move_path

    def _puct_score(self, parent: GPUMCTSNode, child: GPUMCTSNode) -> float:
        """PUCT = -Q_child + c_puct * P * sqrt(N_parent) / (1 + N_child)."""
        q = -child.mean_value if child.visit_count > 0 else 0.0
        exploration = (
            self.config.c_puct
            * child.prior
            * math.sqrt(parent.visit_count)
            / (1 + child.visit_count)
        )
        return q + exploration

    def _backpropagate(
        self,
        node: GPUMCTSNode,
        value: float,
        vl_path: list[GPUMCTSNode],
    ) -> None:
        """Undo virtual loss and backpropagate value with alternating signs."""
        # Undo virtual loss
        for vl_node in vl_path:
            vl_node.visit_count -= 1
            vl_node.total_value += 1.0

        # Standard backpropagation
        current = node
        v = value
        while current is not None:
            current.visit_count += 1
            current.total_value += v
            v = -v
            current = current.parent

    def _apply_root_policy_temp(self, node: GPUMCTSNode) -> None:
        """Soften root priors with temperature > 1.0 before Dirichlet noise.

        Raises priors to power 1/T and renormalizes to prevent the NN's
        policy from being too peaky early in training.
        """
        temp = self.config.root_policy_temp
        if temp <= 1.0 or not node.children:
            return

        actions = list(node.children.keys())
        priors = np.array([node.children[a].prior for a in actions], dtype=np.float64)

        # Apply temperature: p^(1/T), then renormalize
        priors = priors ** (1.0 / temp)
        total = priors.sum()
        if total > 0:
            priors /= total
        else:
            priors[:] = 1.0 / len(priors)

        for i, action in enumerate(actions):
            node.children[action].prior = float(priors[i])

    def _add_dirichlet_noise(self, node: GPUMCTSNode) -> None:
        """Add Dirichlet noise to root priors for exploration.

        With shaped_dirichlet=True (default), alpha is scaled inversely
        with the number of legal moves: alpha = max(0.03, 10/N_legal).
        This concentrates noise on fewer moves in high-branching-factor
        positions, similar to KataGo's approach for Go (alpha=0.03 for
        N=361 legal moves ≈ 10/361).
        """
        if not node.children:
            return

        actions = list(node.children.keys())
        n_legal = len(actions)

        # Shaped alpha: scale inversely with number of legal moves
        if self.config.shaped_dirichlet:
            alpha = max(0.03, 10.0 / n_legal)
        else:
            alpha = self.config.dirichlet_alpha

        noise = np.random.dirichlet([alpha] * n_legal)

        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _get_policy(self, root: GPUMCTSNode, move_number: int) -> np.ndarray:
        """Build policy vector from root's child visit counts.

        Applies policy target pruning: children with visits below
        ``policy_target_pruning`` fraction of the max are zeroed out
        to reduce noise in the training signal.
        """
        policy = np.zeros(self._action_space_size, dtype=np.float32)

        if not root.children:
            return policy

        temp = self.config.temperature
        if move_number >= self.config.temperature_drop_move:
            temp = 0.0

        if temp == 0.0:
            best_action = max(
                root.children,
                key=lambda a: root.children[a].visit_count,
            )
            policy[best_action] = 1.0
        else:
            actions = sorted(root.children.keys())
            visits = np.array(
                [root.children[a].visit_count for a in actions],
                dtype=np.float64,
            )

            # Policy target pruning: zero out low-visit children
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

    # ── GPU move ↔ action index mapping ────────────────────────────────

    def _gpu_move_to_action(
        self,
        move_raw: np.ndarray,
        center_q: int,
        center_r: int,
    ) -> int | None:
        """
        Map a raw GPU move (6 bytes) to an action index.

        Uses the provided centroid for coordinate mapping.
        Returns None if the move maps outside the encoder grid.
        """
        # Parse GPUMove bytes: [type:u8, pt:u8, from:u16LE, to:u16LE]
        m_type = int(move_raw[0])
        m_pt = int(move_raw[1])
        m_from = int(move_raw[2]) | (int(move_raw[3]) << 8)
        m_to = int(move_raw[4]) | (int(move_raw[5]) << 8)

        if m_type == 2:  # PASS
            return self.ext.PASS_ACTION_INDEX

        BOARD_SIZE = self.ext.BOARD_SIZE
        HALF = BOARD_SIZE // 2
        ENC_GRID = self.ext.ENC_GRID
        ENC_HALF = ENC_GRID // 2
        NUM_ENC_GRID_CELLS = ENC_GRID * ENC_GRID

        if m_type == 0:  # PLACE
            to_q = m_to % BOARD_SIZE - HALF
            to_r = m_to // BOARD_SIZE - HALF
            enc_col = to_q - center_q + ENC_HALF
            enc_row = to_r - center_r + ENC_HALF
            if enc_col < 0 or enc_col >= ENC_GRID or enc_row < 0 or enc_row >= ENC_GRID:
                return None
            pt_0idx = m_pt - 1
            pos_idx = enc_row * ENC_GRID + enc_col
            return pt_0idx * NUM_ENC_GRID_CELLS + pos_idx
        else:  # MOVE
            from_q = m_from % BOARD_SIZE - HALF
            from_r = m_from // BOARD_SIZE - HALF
            to_q = m_to % BOARD_SIZE - HALF
            to_r = m_to // BOARD_SIZE - HALF
            src_enc_col = from_q - center_q + ENC_HALF
            src_enc_row = from_r - center_r + ENC_HALF
            dst_enc_col = to_q - center_q + ENC_HALF
            dst_enc_row = to_r - center_r + ENC_HALF
            if (src_enc_col < 0 or src_enc_col >= ENC_GRID or
                src_enc_row < 0 or src_enc_row >= ENC_GRID or
                dst_enc_col < 0 or dst_enc_col >= ENC_GRID or
                dst_enc_row < 0 or dst_enc_row >= ENC_GRID):
                return None
            src_idx = src_enc_row * ENC_GRID + src_enc_col
            dst_idx = dst_enc_row * ENC_GRID + dst_enc_col
            return self.ext.MOVEMENT_OFFSET + src_idx * NUM_ENC_GRID_CELLS + dst_idx

    def _action_to_gpu_move(
        self,
        states: torch.Tensor,
        game_idx: int,
        action: int,
        mask: np.ndarray,
        center_q: int,
        center_r: int,
    ) -> np.ndarray | None:
        """
        Find the GPU move bytes for a given action index.

        Generates legal moves and finds the one matching the action.
        """
        # Generate legal moves for this single game
        single_state = states[game_idx:game_idx + 1]
        moves_tensor, num_legal = self.ext.generate_legal_moves_batch(single_state, 1)
        n_legal = num_legal[0].item()
        raw_moves = moves_tensor[0].cpu().numpy()

        # Find the move that maps to this action
        for mi in range(n_legal):
            move_raw = raw_moves[mi]
            move_action = self._gpu_move_to_action(move_raw, center_q, center_r)
            if move_action == action:
                return move_raw.copy()

        # Fallback: if exact match not found, pick any legal action
        # (can happen if action was from softmax noise hitting wrong index)
        if n_legal > 0:
            return raw_moves[0].copy()

        return None

    # ── Helper methods ─────────────────────────────────────────────────

    def _get_turns(self, states: torch.Tensor, B: int) -> list[int]:
        """Get current turn number for each game (needed for value sign).

        Uses a single batch GPU→CPU copy instead of B individual copies.
        """
        # Single batch transfer
        all_bytes = states.cpu().numpy()  # [B, sizeof(HiveState)]
        turn_offset = 1876  # empirically verified offset
        turns = []
        for i in range(B):
            turn = int(all_bytes[i, turn_offset]) | (int(all_bytes[i, turn_offset + 1]) << 8)
            turns.append(turn)
        return turns

    def _get_turn_single(self, states: torch.Tensor, game_idx: int) -> int:
        """Get turn number for a single game from raw state bytes."""
        state_bytes = states[game_idx].cpu().numpy()
        BOARD_SIZE = self.ext.BOARD_SIZE
        NUM_CELLS = BOARD_SIZE * BOARD_SIZE

        # HiveState layout: turn at offset 1876 (empirically verified;
        # 2 bytes padding between height[289] and occupied Bitboard for alignment)
        turn_offset = 1876
        turn = int(state_bytes[turn_offset]) | (int(state_bytes[turn_offset + 1]) << 8)
        return turn

    def _result_to_value(self, result: int, turn: int) -> float:
        """
        Convert game result to value from current player's perspective.

        result: 0=in_progress, 1=white_wins, 2=black_wins, 3=draw
        turn: current turn number (turn % 2 == 0 means it's white's move)
        """
        if result == 3:  # DRAW
            return 0.0
        current_is_white = (turn % 2 == 0)
        if result == 1:  # WHITE_WINS
            return 1.0 if current_is_white else -1.0
        if result == 2:  # BLACK_WINS
            return -1.0 if current_is_white else 1.0
        return 0.0

    # ── HiveState byte parsing constants ───────────────────────────────

    # Hex grid: 17×17, direction offsets (dcol, drow)
    _BOARD_SIZE = 17
    _NUM_CELLS = _BOARD_SIZE * _BOARD_SIZE  # 289
    _DIR_DCOL = [+1, +1,  0, -1, -1,  0]
    _DIR_DROW = [ 0, -1, -1,  0, +1, +1]
    _MAX_STACK = 5

    # HiveState struct byte offsets (empirically verified).
    # Bitboard (uint64_t[5]) requires 8-byte alignment, adding 2 bytes padding
    # after height[289] at offset 1734 → occupied starts at 1736.
    _OFF_PIECES = 0                        # uint8[5][289] = 1445
    _OFF_HEIGHT = _MAX_STACK * _NUM_CELLS  # 1445
    _BB_SIZE = 5 * 8                       # 40 bytes per Bitboard
    _OFF_OCCUPIED = 1736                   # Bitboard, 40 bytes
    _OFF_WHITE_TOP = 1776                  # Bitboard, 40 bytes
    _OFF_BLACK_TOP = 1816                  # Bitboard, 40 bytes
    _OFF_QUEEN_CELL = 1856                 # uint16[2] = 4 bytes
    _OFF_HANDS = 1860                      # uint8[2][8] = 16 bytes
    _OFF_TURN = 1876                       # uint16

    def _hex_neighbors(self, cell: int) -> list[int]:
        """Get valid hex neighbor cells (matching GPU hex_grid.cuh layout)."""
        row = cell // self._BOARD_SIZE
        col = cell % self._BOARD_SIZE
        neighbors = []
        for d in range(6):
            nr = row + self._DIR_DROW[d]
            nc = col + self._DIR_DCOL[d]
            if 0 <= nr < self._BOARD_SIZE and 0 <= nc < self._BOARD_SIZE:
                neighbors.append(nr * self._BOARD_SIZE + nc)
            else:
                neighbors.append(-1)
        return neighbors

    def _compute_queen_surround_batch(
        self,
        states_tensor: torch.Tensor,
        B: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Compute queen surround targets from final state bytes on CPU.

        Returns list of (qs_target, qs_mask) per game where:
          qs_target: (num_board_nodes, 2) float32
          qs_mask: (2,) float32
        """
        states_np = states_tensor.cpu().numpy()  # [B, sizeof(HiveState)]
        results = []

        for i in range(B):
            sb = states_np[i]

            # Parse queen cells
            qc_off = self._OFF_QUEEN_CELL
            queen_cells = [
                int(sb[qc_off]) | (int(sb[qc_off + 1]) << 8),
                int(sb[qc_off + 2]) | (int(sb[qc_off + 3]) << 8),
            ]

            # Parse heights
            heights = sb[self._OFF_HEIGHT:self._OFF_HEIGHT + self._NUM_CELLS]

            # Build cell → top node index map (same iteration as encoder)
            top_node_at: dict[int, int] = {}
            node_count = 0
            for cell in range(self._NUM_CELLS):
                h = int(heights[cell])
                if h == 0:
                    continue
                first_node = node_count
                for level in range(h):
                    if level == h - 1:
                        top_node_at[cell] = node_count
                    node_count += 1
            num_board_nodes = node_count

            # Queen surround mask
            qs_mask = np.zeros(2, dtype=np.float32)
            qs_target = np.zeros((num_board_nodes, 2), dtype=np.float32)

            for c in range(2):
                qc = queen_cells[c]
                if qc == 0xFFFF:
                    continue
                qs_mask[c] = 1.0

                # Find occupied neighbors of this queen
                neighbors = self._hex_neighbors(qc)
                for nb in neighbors:
                    if nb < 0:
                        continue
                    if int(heights[nb]) > 0 and nb in top_node_at:
                        node_idx = top_node_at[nb]
                        qs_target[node_idx, c] = 1.0

            results.append((qs_target, qs_mask))

        return results

    def _build_examples(
        self,
        histories: list[list[tuple[HiveGraph, np.ndarray, int, np.ndarray, HiveTokenSequence | None, np.ndarray | None]]],
        final_results: np.ndarray,
        final_mob_np: np.ndarray,
        final_mob_board_counts: np.ndarray,
        final_qs_data: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[list[GPUTrainingExample]]:
        """Build training examples from game histories and final results."""
        all_examples = []

        for i, history in enumerate(histories):
            result = int(final_results[i])
            examples = []
            num_steps = len(history)

            for step_idx, (graph, policy, turn, mobility, seq, nn_prior) in enumerate(history):
                # Value from this player's perspective
                if result == 0 or result == 3:  # IN_PROGRESS or DRAW
                    value = 0.0
                else:
                    player_is_white = (turn % 2 == 0)
                    if result == 1:  # WHITE_WINS
                        value = 1.0 if player_is_white else -1.0
                    else:  # BLACK_WINS
                        value = -1.0 if player_is_white else 1.0

                # Use seq.num_board_tokens for transformer; fall back to graph for GNN
                if seq is not None:
                    n = seq.num_board_tokens
                elif graph is not None:
                    n = graph.num_piece_nodes
                else:
                    n = 0
                is_final = (step_idx == num_steps - 1)

                if is_final:
                    # Use computed game-end targets for the final position
                    final_n = int(final_mob_board_counts[i])
                    fm = final_mob_np[i, :final_n].copy()
                    qs_target, qs_mask = final_qs_data[i]
                    # Sizes may differ if pieces were placed on the final move
                    # Pad or trim to match the final position's board node count
                    if len(fm) != n:
                        fm = np.zeros(n, dtype=np.float32)
                    if qs_target.shape[0] != n:
                        qs_target = np.zeros((n, 2), dtype=np.float32)
                else:
                    # Non-final positions: zero game-end targets
                    fm = np.zeros(n, dtype=np.float32)
                    qs_target = np.zeros((n, 2), dtype=np.float32)
                    qs_mask = np.zeros(2, dtype=np.float32)

                examples.append(GPUTrainingExample(
                    graph=graph,
                    policy_target=policy,
                    value_target=value,
                    mobility_target=mobility,
                    queen_surround_target=qs_target,
                    queen_surround_mask=qs_mask,
                    final_mobility_target=fm,
                    sequence=seq,
                    nn_prior=nn_prior,
                ))

            all_examples.append(examples)

        return all_examples
