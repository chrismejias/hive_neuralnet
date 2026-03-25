"""
GPU-native MCTS with batched neural network inference.

Architecture: GPU-resident MCTS trees + GPU kernels for select/expand/backprop.
Eliminates Python-loop overhead for tree operations by running them as CUDA kernels.
The NN forward pass, encoding, and legal move generation remain on GPU as before.

Usage:
    from hive_gpu.gpu_native_mcts import GPUNativeMCTSOrchestrator
    from hive_gpu.gpu_mcts import GPUMCTSConfig

    orchestrator = GPUNativeMCTSOrchestrator(net, config)
    examples = orchestrator.self_play_batch()
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

import hive_gpu

try:
    from hive_gnn.graph_types import HiveGraph
except ImportError:
    HiveGraph = None  # type: ignore[assignment,misc]
from hive_gpu.gpu_encoder import GPUGNNEncoder, GPUTransformerEncoder
from hive_gpu.gpu_mcts import GPUMCTSConfig, GPUTrainingExample
from hive_transformer.token_types import HiveTokenSequence


class GPUNativeMCTSOrchestrator:
    """
    Runs batched MCTS self-play with GPU-native tree operations.

    The tree structure lives on GPU as flat tensors. Selection, expansion,
    and backpropagation are CUDA kernels. Only the NN forward pass and
    training-example assembly touch Python.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        config: GPUMCTSConfig | None = None,
    ) -> None:
        self.ext = hive_gpu.load_extension()
        self.config = config or GPUMCTSConfig()

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
        self._max_nodes = getattr(self.ext, 'DEFAULT_MAX_TREE_NODES', 32768)
        self._max_depth = getattr(self.ext, 'MAX_TREE_DEPTH', 128)

    # ── Tree tensor management ────────────────────────────────────

    def _alloc_tree(self, B: int) -> dict[str, torch.Tensor]:
        """Allocate GPU tree tensors for B games."""
        M = self._max_nodes
        dev = "cuda"
        return {
            "visit_count":    torch.zeros(B, M, dtype=torch.int32, device=dev),
            "total_value":    torch.zeros(B, M, dtype=torch.float32, device=dev),
            "prior":          torch.zeros(B, M, dtype=torch.float32, device=dev),
            "parent_idx":     torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "move_bytes":     torch.zeros(B, M, self._move_size, dtype=torch.uint8, device=dev),
            "action_idx":     torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "first_child":    torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "num_children":   torch.zeros(B, M, dtype=torch.int32, device=dev),
            "is_terminal":    torch.zeros(B, M, dtype=torch.int8, device=dev),
            "terminal_value": torch.zeros(B, M, dtype=torch.float32, device=dev),
            "node_count":     torch.ones(B, dtype=torch.int32, device=dev),  # root=node 0
        }

    def _reset_tree(self, tree: dict[str, torch.Tensor]) -> None:
        """Reset tree to root-only state for a new move decision."""
        tree["visit_count"].zero_()
        tree["total_value"].zero_()
        tree["prior"].zero_()
        tree["parent_idx"].fill_(-1)
        tree["move_bytes"].zero_()
        tree["action_idx"].fill_(-1)
        tree["first_child"].fill_(-1)
        tree["num_children"].zero_()
        tree["is_terminal"].zero_()
        tree["terminal_value"].zero_()
        tree["node_count"].fill_(1)  # node 0 = root

    def _tree_args(self, tree: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Return tree tensors in the order expected by CUDA host wrappers."""
        return [
            tree["visit_count"], tree["total_value"], tree["prior"],
            tree["parent_idx"], tree["move_bytes"], tree["action_idx"],
            tree["first_child"], tree["num_children"],
            tree["is_terminal"], tree["terminal_value"], tree["node_count"],
        ]

    # ── Public API ────────────────────────────────────────────────

    def self_play_batch(self) -> list[list[GPUTrainingExample]]:
        """Play B games in parallel with GPU-native MCTS."""
        B = self.config.batch_size
        cfg = self.config

        root_states = self.ext.create_initial_states(B, cfg.expansion_mask)
        tree = self._alloc_tree(B)

        active = [True] * B
        move_numbers = [0] * B
        histories: list[list[tuple]] = [[] for _ in range(B)]

        while any(active):
            current_turns = self._get_turns(root_states, B)

            # Playout cap randomisation
            if cfg.playout_cap_randomize:
                use_full = np.random.random() < cfg.playout_cap_randomize_prob
                fast_sims = cfg.playout_cap_fast_sims or max(1, cfg.num_simulations // 8)
                num_sims = cfg.num_simulations if use_full else fast_sims
            else:
                use_full = True
                num_sims = cfg.num_simulations

            # Run GPU-native MCTS
            policies, graphs, seqs, nn_priors = self._batched_mcts(
                root_states, B, active, move_numbers, tree, num_sims=num_sims,
            )

            # Compute per-piece mobility
            mob_tensor, mob_counts = self.ext.compute_mobility_batch(root_states, B, False)
            mob_np = mob_tensor.cpu().numpy()
            mob_board_counts = mob_counts.cpu().numpy()

            # Record history & select actions
            action_move_bytes = np.zeros((B, self._move_size), dtype=np.uint8)

            for i in range(B):
                if not active[i]:
                    continue

                policy = policies[i]
                turn = current_turns[i]

                if use_full:
                    mob_i = mob_np[i, :mob_board_counts[i]].copy()
                    histories[i].append(
                        (graphs[i], policy.copy(), turn, mob_i, seqs[i], nn_priors[i])
                    )

                # Action selection
                if move_numbers[i] >= cfg.temperature_drop_move:
                    action = int(np.argmax(policy))
                else:
                    action = int(np.random.choice(len(policy), p=policy))

                # Look up move bytes from tree
                move_bytes = self._action_to_move_bytes_from_tree(tree, i, action)
                if move_bytes is not None:
                    action_move_bytes[i] = move_bytes
                else:
                    # Fallback: generate legal moves
                    centroids = self.ext.compute_centroids_batch(
                        root_states[i:i+1], 1
                    ).cpu().numpy()
                    cq, cr = int(centroids[0, 0]), int(centroids[0, 1])
                    fb = self._action_to_gpu_move_fallback(
                        root_states, i, action, cq, cr,
                    )
                    if fb is not None:
                        action_move_bytes[i] = fb

            # Apply moves
            moves_t = torch.from_numpy(action_move_bytes).cuda()
            self.ext.apply_moves_batch(root_states, moves_t, B)

            for i in range(B):
                if active[i]:
                    move_numbers[i] += 1

            # Check game over
            results = self.ext.check_results_batch(root_states, B).cpu().numpy()
            for i in range(B):
                if not active[i]:
                    continue
                if results[i] != 0 or move_numbers[i] >= cfg.max_game_length:
                    active[i] = False

        # Build training examples
        final_results = self.ext.check_results_batch(root_states, B).cpu().numpy()
        final_mob_tensor, final_mob_counts = self.ext.compute_mobility_batch(
            root_states, B, True
        )
        final_mob_np = final_mob_tensor.cpu().numpy()
        final_mob_board_counts = final_mob_counts.cpu().numpy()
        final_qs_data = self._compute_queen_surround_batch(root_states, B)

        return self._build_examples(
            histories, final_results,
            final_mob_np, final_mob_board_counts,
            final_qs_data,
        )

    # ── GPU-native MCTS core ──────────────────────────────────────

    def _batched_mcts(
        self,
        root_states: torch.Tensor,
        B: int,
        active: list[bool],
        move_numbers: list[int],
        tree: dict[str, torch.Tensor],
        num_sims: int | None = None,
    ) -> tuple[
        list[np.ndarray],
        list[HiveGraph],
        list[HiveTokenSequence | None],
        list[np.ndarray | None],
    ]:
        """Run GPU-native MCTS.  Returns (policies, graphs, seqs, nn_priors)."""
        cfg = self.config
        if num_sims is None:
            num_sims = cfg.num_simulations

        # Reset tree
        self._reset_tree(tree)

        # Mark terminal roots
        game_active = torch.tensor(
            [1 if active[i] else 0 for i in range(B)],
            dtype=torch.int8, device="cuda",
        )

        # Expand roots: encode + NN + expand kernel
        root_graphs, root_seqs, nn_priors = self._expand_roots(
            root_states, B, tree, active,
        )

        # Apply root policy temperature + Dirichlet noise
        self._apply_root_noise(tree, B, active)

        # Run simulation waves
        W = cfg.wave_size
        num_waves = math.ceil(num_sims / W)
        state_size = root_states.shape[1]
        leaf_states = torch.zeros(W * B, state_size, dtype=torch.uint8, device="cuda")

        for wave in range(num_waves):
            actual_w = min(W, num_sims - wave * W)
            total = actual_w * B

            # SELECT
            leaf_idx, move_paths, path_lens, vl_paths, vl_lens = self.ext.mcts_select_batch(
                *self._tree_args(tree),
                game_active, cfg.c_puct, B, actual_w, self._max_nodes,
            )

            # REPLAY
            self.ext.mcts_replay_batch(
                root_states, leaf_states[:total],
                move_paths[:total], path_lens[:total], leaf_idx[:total],
                B, total,
            )

            # TERMINAL CHECK
            results = self.ext.check_results_batch(leaf_states[:total], total)

            # ENCODE + NN
            encoded = self.encoder.encode_batch(leaf_states[:total], total)
            masks, _ = self.ext.generate_legal_mask_batch(leaf_states[:total], total)

            with torch.no_grad():
                if cfg.nn_max_batch > 0 and total > cfg.nn_max_batch:
                    policy_logits, values = self._nn_forward_subbatched(encoded, total)
                else:
                    policy_logits, values, *_ = self.net(encoded)

            masks_bool = masks == 0
            policy_logits[masks_bool] = float("-inf")
            action_probs = torch.softmax(policy_logits, dim=-1)
            values_flat = values.squeeze(-1)

            # LEGAL MOVES
            legal_moves, num_legal = self.ext.generate_legal_moves_batch(
                leaf_states[:total], total,
            )

            # EXPAND
            was_expanded = self.ext.mcts_expand_batch(
                *self._tree_args(tree),
                leaf_idx[:total], leaf_states[:total],
                legal_moves, num_legal, action_probs, results,
                B, total, self._max_nodes,
            )

            # BACKPROP
            self.ext.mcts_backprop_batch(
                *self._tree_args(tree),
                leaf_idx[:total], values_flat,
                vl_paths[:total], vl_lens[:total], was_expanded,
                B, total, self._max_nodes,
            )

        # Extract policies
        move_nums_t = torch.tensor(move_numbers, dtype=torch.int32, device="cuda")
        policies_t = self.ext.mcts_extract_policy_batch(
            *self._tree_args(tree),
            move_nums_t, cfg.temperature, cfg.temperature_drop_move,
            cfg.policy_target_pruning, B, self._max_nodes,
        )
        policies_np = policies_t.cpu().numpy()  # [B, ACTION_SPACE_SIZE]

        policies = []
        for i in range(B):
            policies.append(policies_np[i].copy())

        return policies, root_graphs, root_seqs, nn_priors

    # ── Root expansion ────────────────────────────────────────────

    def _expand_roots(
        self,
        root_states: torch.Tensor,
        B: int,
        tree: dict[str, torch.Tensor],
        active: list[bool],
    ) -> tuple[list, list, list]:
        """Expand all root nodes via encode + NN + GPU expand kernel.

        Returns (root_graphs, root_seqs, nn_priors).
        """
        # Check terminal roots
        results = self.ext.check_results_batch(root_states, B)

        # Encode
        root_seqs: list[HiveTokenSequence | None] = [None] * B
        if hasattr(self.encoder, "encode_batch_with_graphs_and_seqs"):
            encoded, root_graphs, root_seqs_list = (
                self.encoder.encode_batch_with_graphs_and_seqs(root_states, B)
            )
            root_seqs = root_seqs_list
        elif hasattr(self.encoder, "encode_batch_with_graphs"):
            encoded, root_graphs = self.encoder.encode_batch_with_graphs(root_states, B)
        else:
            encoded = self.encoder.encode_batch(root_states, B)
            root_graphs = [None] * B

        # NN forward
        masks, _ = self.ext.generate_legal_mask_batch(root_states, B)
        with torch.no_grad():
            policy_logits, values, *_ = self.net(encoded)

        masks_bool = masks == 0
        policy_logits[masks_bool] = float("-inf")
        action_probs = torch.softmax(policy_logits, dim=-1)

        # Legal moves
        legal_moves, num_legal = self.ext.generate_legal_moves_batch(root_states, B)

        # Set up leaf_indices (all roots = node 0) for expand kernel
        leaf_indices = torch.zeros(B, dtype=torch.int32, device="cuda")

        # Expand roots via GPU kernel
        was_expanded = self.ext.mcts_expand_batch(
            *self._tree_args(tree),
            leaf_indices, root_states,
            legal_moves, num_legal, action_probs, results,
            B, B, self._max_nodes,
        )

        # Extract NN priors BEFORE noise (for surprise weighting)
        nn_priors: list[np.ndarray | None] = [None] * B
        action_probs_np = action_probs.cpu().numpy()
        fc_cpu = tree["first_child"][:, 0].cpu().numpy()
        nc_cpu = tree["num_children"][:, 0].cpu().numpy()
        ai_cpu = tree["action_idx"].cpu().numpy()

        for i in range(B):
            if not active[i] or fc_cpu[i] < 0 or nc_cpu[i] == 0:
                continue
            prior = np.zeros(self._action_space_size, dtype=np.float32)
            fc_i = fc_cpu[i]
            for c in range(nc_cpu[i]):
                action = ai_cpu[i, fc_i + c]
                if 0 <= action < self._action_space_size:
                    prior[action] = action_probs_np[i, action]
            nn_priors[i] = prior

        return root_graphs, root_seqs, nn_priors

    def _apply_root_noise(
        self,
        tree: dict[str, torch.Tensor],
        B: int,
        active: list[bool],
    ) -> None:
        """Apply root policy temperature + Dirichlet noise via GPU kernel."""
        cfg = self.config
        # Get num_children per game to generate appropriately-sized Dirichlet noise
        nc_cpu = tree["num_children"][:, 0].cpu().numpy()  # root's num_children

        max_nc = int(nc_cpu.max()) if nc_cpu.max() > 0 else 1

        # Generate Dirichlet noise on CPU, copy to GPU
        noise = np.zeros((B, max_nc), dtype=np.float32)
        for i in range(B):
            nc = int(nc_cpu[i])
            if nc == 0 or not active[i]:
                continue
            if cfg.shaped_dirichlet:
                alpha = max(0.03, 10.0 / nc)
            else:
                alpha = cfg.dirichlet_alpha
            noise[i, :nc] = np.random.dirichlet([alpha] * nc)

        noise_t = torch.from_numpy(noise).cuda()

        self.ext.mcts_apply_root_noise(
            *self._tree_args(tree),
            noise_t, max_nc,
            cfg.dirichlet_epsilon, cfg.root_policy_temp,
            B, self._max_nodes,
        )

    # ── Action → move bytes lookup ────────────────────────────────

    def _action_to_move_bytes_from_tree(
        self,
        tree: dict[str, torch.Tensor],
        game_idx: int,
        action: int,
    ) -> np.ndarray | None:
        """Look up move bytes for a given action from root's children in the GPU tree."""
        fc = tree["first_child"][game_idx, 0].item()
        nc = tree["num_children"][game_idx, 0].item()
        if fc < 0 or nc == 0:
            return None

        # Search root's children for matching action
        ai_slice = tree["action_idx"][game_idx, fc:fc + nc].cpu().numpy()
        mb_slice = tree["move_bytes"][game_idx, fc:fc + nc].cpu().numpy()

        for c in range(nc):
            if ai_slice[c] == action:
                return mb_slice[c].copy()
        return None

    def _action_to_gpu_move_fallback(
        self,
        states: torch.Tensor,
        game_idx: int,
        action: int,
        center_q: int,
        center_r: int,
    ) -> np.ndarray | None:
        """Fallback: generate legal moves to find one matching the action."""
        single_state = states[game_idx:game_idx + 1]
        moves_tensor, num_legal = self.ext.generate_legal_moves_batch(single_state, 1)
        n_legal = num_legal[0].item()
        raw_moves = moves_tensor[0].cpu().numpy()

        for mi in range(n_legal):
            move_raw = raw_moves[mi]
            move_action = self._gpu_move_to_action(move_raw, center_q, center_r)
            if move_action == action:
                return move_raw.copy()

        if n_legal > 0:
            return raw_moves[0].copy()
        return None

    def _gpu_move_to_action(
        self, move_raw: np.ndarray, center_q: int, center_r: int,
    ) -> int | None:
        """Map raw GPU move bytes to action index (CPU fallback)."""
        m_type = int(move_raw[0])
        m_pt = int(move_raw[1])
        m_from = int(move_raw[2]) | (int(move_raw[3]) << 8)
        m_to = int(move_raw[4]) | (int(move_raw[5]) << 8)

        if m_type == 2:
            return self.ext.PASS_ACTION_INDEX

        BOARD_SIZE = self.ext.BOARD_SIZE
        HALF = BOARD_SIZE // 2
        ENC_GRID = self.ext.ENC_GRID
        ENC_HALF = ENC_GRID // 2
        NUM_ENC = ENC_GRID * ENC_GRID

        if m_type == 0:  # PLACE
            to_q = m_to % BOARD_SIZE - HALF
            to_r = m_to // BOARD_SIZE - HALF
            ec = to_q - center_q + ENC_HALF
            er = to_r - center_r + ENC_HALF
            if ec < 0 or ec >= ENC_GRID or er < 0 or er >= ENC_GRID:
                return None
            return (m_pt - 1) * NUM_ENC + er * ENC_GRID + ec
        else:  # MOVE
            fq = m_from % BOARD_SIZE - HALF
            fr = m_from // BOARD_SIZE - HALF
            tq = m_to % BOARD_SIZE - HALF
            tr = m_to // BOARD_SIZE - HALF
            sc = fq - center_q + ENC_HALF
            sr = fr - center_r + ENC_HALF
            dc = tq - center_q + ENC_HALF
            dr = tr - center_r + ENC_HALF
            if (sc < 0 or sc >= ENC_GRID or sr < 0 or sr >= ENC_GRID or
                dc < 0 or dc >= ENC_GRID or dr < 0 or dr >= ENC_GRID):
                return None
            return self.ext.MOVEMENT_OFFSET + (sr * ENC_GRID + sc) * NUM_ENC + (dr * ENC_GRID + dc)

    # ── NN sub-batching ───────────────────────────────────────────

    def _nn_forward_subbatched(
        self, encoded, total: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NN forward in sub-batches, concatenating results."""
        max_batch = self.config.nn_max_batch
        all_policy, all_values = [], []
        for start in range(0, total, max_batch):
            end = min(start + max_batch, total)
            chunk = encoded.slice_batch(start, end) if hasattr(encoded, 'slice_batch') else encoded
            policy_logits, values, *_ = self.net(chunk)
            all_policy.append(policy_logits)
            all_values.append(values)
        return torch.cat(all_policy, dim=0), torch.cat(all_values, dim=0)

    # ── Utility methods (shared with GPUMCTSOrchestrator) ─────────

    _BOARD_SIZE = 17
    _NUM_CELLS = _BOARD_SIZE * _BOARD_SIZE
    _DIR_DCOL = [+1, +1, 0, -1, -1, 0]
    _DIR_DROW = [0, -1, -1, 0, +1, +1]
    _MAX_STACK = 5
    _OFF_HEIGHT = _MAX_STACK * _NUM_CELLS
    _OFF_QUEEN_CELL = 1856
    _OFF_TURN = 1876

    def _get_turns(self, states: torch.Tensor, B: int) -> list[int]:
        all_bytes = states.cpu().numpy()
        off = self._OFF_TURN
        return [
            int(all_bytes[i, off]) | (int(all_bytes[i, off + 1]) << 8)
            for i in range(B)
        ]

    def _hex_neighbors(self, cell: int) -> list[int]:
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
        self, states_tensor: torch.Tensor, B: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        states_np = states_tensor.cpu().numpy()
        results = []
        for i in range(B):
            sb = states_np[i]
            qc_off = self._OFF_QUEEN_CELL
            queen_cells = [
                int(sb[qc_off]) | (int(sb[qc_off + 1]) << 8),
                int(sb[qc_off + 2]) | (int(sb[qc_off + 3]) << 8),
            ]
            heights = sb[self._OFF_HEIGHT:self._OFF_HEIGHT + self._NUM_CELLS]
            top_node_at: dict[int, int] = {}
            node_count = 0
            for cell in range(self._NUM_CELLS):
                h = int(heights[cell])
                if h == 0:
                    continue
                for level in range(h):
                    if level == h - 1:
                        top_node_at[cell] = node_count
                    node_count += 1
            num_board_nodes = node_count
            qs_mask = np.zeros(2, dtype=np.float32)
            qs_target = np.zeros((num_board_nodes, 2), dtype=np.float32)
            for c in range(2):
                qc = queen_cells[c]
                if qc == 0xFFFF:
                    continue
                qs_mask[c] = 1.0
                neighbors = self._hex_neighbors(qc)
                for nb in neighbors:
                    if nb < 0:
                        continue
                    if int(heights[nb]) > 0 and nb in top_node_at:
                        qs_target[top_node_at[nb], c] = 1.0
            results.append((qs_target, qs_mask))
        return results

    def _build_examples(
        self,
        histories,
        final_results: np.ndarray,
        final_mob_np: np.ndarray,
        final_mob_board_counts: np.ndarray,
        final_qs_data: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[list[GPUTrainingExample]]:
        all_examples = []
        for i, history in enumerate(histories):
            result = int(final_results[i])
            examples = []
            num_steps = len(history)
            for step_idx, (graph, policy, turn, mobility, seq, nn_prior) in enumerate(history):
                if result == 0 or result == 3:
                    value = 0.0
                else:
                    player_is_white = (turn % 2 == 0)
                    if result == 1:
                        value = 1.0 if player_is_white else -1.0
                    else:
                        value = -1.0 if player_is_white else 1.0
                if seq is not None:
                    n = seq.num_board_tokens
                elif graph is not None:
                    n = graph.num_piece_nodes
                else:
                    n = 0
                is_final = (step_idx == num_steps - 1)
                if is_final:
                    final_n = int(final_mob_board_counts[i])
                    fm = final_mob_np[i, :final_n].copy()
                    qs_target, qs_mask = final_qs_data[i]
                    if len(fm) != n:
                        fm = np.zeros(n, dtype=np.float32)
                    if qs_target.shape[0] != n:
                        qs_target = np.zeros((n, 2), dtype=np.float32)
                else:
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
