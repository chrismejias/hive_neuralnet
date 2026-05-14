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
from archive.modules.hive_gpu_hybrid.gpu_mcts import GPUMCTSConfig, GPUTrainingExample
from hive_common.token_types import HiveTokenSequence


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

        # Don't use torch.compile with GPU-native MCTS — compiled kernels
        # use non-default CUDA streams that race with custom CUDA kernels,
        # causing intermittent hangs in the MCTS expand/backprop pipeline.
        self.net = net

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
            "root_node":      torch.zeros(B, dtype=torch.int32, device=dev),  # current root per game
        }

    def _reset_tree(self, tree: dict[str, torch.Tensor]) -> None:
        """Full reset to a fresh single-root tree (called once per self_play_batch)."""
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
        tree["root_node"].zero_()    # all games start at node 0

    def _tree_args(self, tree: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Return tree tensors in the order expected by CUDA host wrappers."""
        return [
            tree["visit_count"], tree["total_value"], tree["prior"],
            tree["parent_idx"], tree["move_bytes"], tree["action_idx"],
            tree["first_child"], tree["num_children"],
            tree["is_terminal"], tree["terminal_value"], tree["node_count"],
        ]

    # ── Public API ────────────────────────────────────────────────

    def self_play_batch(
        self,
        start_states: "torch.Tensor | None" = None,
    ) -> list[list[GPUTrainingExample]]:
        """Play B games in parallel with GPU-native MCTS.

        Args:
            start_states: Optional [B, SIZEOF_HIVE_STATE] uint8 tensor of
                pre-built starting positions (e.g. endgame positions for
                curriculum learning).  If None, fresh initial states are
                created via create_initial_states().
        """
        B = self.config.batch_size
        cfg = self.config

        if start_states is not None:
            root_states = start_states.cuda()
        else:
            root_states = self.ext.create_initial_states(B, cfg.expansion_mask)
        tree = self._alloc_tree(B)

        # Full reset once per game batch (not per move — tree reuse handles moves)
        self._reset_tree(tree)

        active = [True] * B
        move_numbers = [0] * B
        histories: list[list[tuple]] = [[] for _ in range(B)]
        first_move = True

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

            # Run GPU-native MCTS (tree is reused across moves)
            policies, graphs, seqs, nn_priors = self._batched_mcts(
                root_states, B, active, move_numbers, tree,
                num_sims=num_sims, first_move=first_move,
            )
            first_move = False

            # Compute per-piece mobility
            mob_tensor, mob_counts = self.ext.compute_mobility_batch(root_states, B, False)
            mob_np = mob_tensor.cpu().numpy()
            mob_board_counts = mob_counts.cpu().numpy()

            # Override policies for any game with an immediate winning move
            self._check_immediate_wins(
                root_states, B, active, current_turns, tree, policies,
            )

            # Record history & select actions
            action_move_bytes = np.zeros((B, self._move_size), dtype=np.uint8)
            chosen_child_nodes = [-1] * B  # child node index chosen per game (for tree reuse)

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
                    psum = policy.sum()
                    if psum > 0 and np.isfinite(psum):
                        policy = policy / psum
                    else:
                        # Fallback: uniform over legal actions
                        mask = policy > 0
                        policy = np.zeros_like(policy)
                        if mask.any():
                            policy[mask] = 1.0 / mask.sum()
                        else:
                            policy[:] = 1.0 / len(policy)
                    action = int(np.random.choice(len(policy), p=policy))

                # Look up move bytes from tree; also record the child node for reuse
                move_bytes, child_node = self._action_to_move_bytes_from_tree(tree, i, action)
                chosen_child_nodes[i] = child_node
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

            # Reroot tree at chosen child for each active game
            self._reroot_tree(tree, B, active, chosen_child_nodes)

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
        first_move: bool = False,
    ) -> tuple[
        list[np.ndarray],
        list[HiveGraph],
        list[HiveTokenSequence | None],
        list[np.ndarray | None],
    ]:
        """Run GPU-native MCTS.  Returns (policies, graphs, seqs, nn_priors).

        Tree reuse: the tree is NOT reset here — it persists across moves within
        a game batch.  `first_move=True` triggers root expansion + noise;
        subsequent moves only re-apply noise to the already-expanded new root.
        """
        cfg = self.config
        if num_sims is None:
            num_sims = cfg.num_simulations

        game_active = torch.tensor(
            [1 if active[i] else 0 for i in range(B)],
            dtype=torch.int8, device="cuda",
        )

        if first_move:
            # First move of the batch: expand roots via NN, then apply noise.
            root_graphs, root_seqs, nn_priors = self._expand_roots(
                root_states, B, tree, active,
            )
            self._apply_root_noise(tree, B, active)
        else:
            # Subsequent moves: root is the chosen child from the previous move.
            # Its children are already expanded with NN priors — just encode for
            # history and re-apply fresh Dirichlet noise.
            root_graphs, root_seqs = self._encode_roots_for_history(root_states, B)
            nn_priors = self._extract_nn_priors_from_tree(tree, B, active)
            self._apply_root_noise(tree, B, active)

        # Run simulation waves
        W = cfg.wave_size
        num_waves = math.ceil(num_sims / W)
        state_size = root_states.shape[1]
        leaf_states = torch.zeros(W * B, state_size, dtype=torch.uint8, device="cuda")

        for wave in range(num_waves):
            actual_w = min(W, num_sims - wave * W)
            total = actual_w * B

            # SELECT — starts from root_node[game] (supports tree reuse)
            leaf_idx, move_paths, path_lens, vl_paths, vl_lens = self.ext.mcts_select_batch(
                *self._tree_args(tree),
                game_active, tree["root_node"], cfg.c_puct, B, actual_w, self._max_nodes,
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

            # Fused legal-moves + mask: one move-gen kernel instead of two
            legal_moves, num_legal, masks = self.ext.generate_legal_moves_and_mask_batch(
                leaf_states[:total], total,
            )

            with torch.no_grad():
                if cfg.nn_max_batch > 0 and total > cfg.nn_max_batch:
                    policy_logits, values = self._nn_forward_subbatched(encoded, total)
                else:
                    policy_logits, values, *_ = self.net(encoded)

            masks_bool = masks == 0
            policy_logits[masks_bool] = float("-inf")
            action_probs = torch.softmax(policy_logits, dim=-1)
            values_flat = values.squeeze(-1)

            # Fused EXPAND + BACKPROP: was_expanded stays in registers
            self.ext.mcts_expand_and_backprop_batch(
                *self._tree_args(tree),
                leaf_idx[:total], leaf_states[:total],
                legal_moves, num_legal, action_probs, results,
                values_flat, vl_paths[:total], vl_lens[:total],
                B, total, self._max_nodes,
            )

        # Extract policies from current root_node per game
        move_nums_t = torch.tensor(move_numbers, dtype=torch.int32, device="cuda")
        policies_t = self.ext.mcts_extract_policy_batch(
            *self._tree_args(tree),
            move_nums_t, tree["root_node"], cfg.temperature, cfg.temperature_drop_move,
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

        # Sync PyTorch async ops before custom kernels
        torch.cuda.synchronize()

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
        # On first move root_node is always 0 so indexing [:, 0] is equivalent,
        # but use root_node for consistency.
        nn_priors: list[np.ndarray | None] = [None] * B
        action_probs_np = action_probs.cpu().numpy()
        _row = torch.arange(B, device="cuda")
        fc_cpu = tree["first_child"][_row, tree["root_node"]].cpu().numpy()
        nc_cpu = tree["num_children"][_row, tree["root_node"]].cpu().numpy()
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
        # Gather num_children for the current root node per game
        row_idx = torch.arange(B, device="cuda")
        nc_tensor = tree["num_children"][row_idx, tree["root_node"]]  # [B]
        nc_cpu = nc_tensor.cpu().numpy()

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
            noise_t, tree["root_node"], max_nc,
            cfg.dirichlet_epsilon, cfg.root_policy_temp,
            B, self._max_nodes,
        )

    def _encode_roots_for_history(
        self,
        root_states: torch.Tensor,
        B: int,
    ) -> tuple[list, list]:
        """Encode current root states to get graphs/seqs for history recording.

        Unlike _expand_roots, this does NOT run the NN or touch the tree.
        Used on moves 2+ when the tree is reused.
        """
        if hasattr(self.encoder, "encode_batch_with_graphs_and_seqs"):
            _, root_graphs, root_seqs = (
                self.encoder.encode_batch_with_graphs_and_seqs(root_states, B)
            )
            return root_graphs, root_seqs
        elif hasattr(self.encoder, "encode_batch_with_graphs"):
            _, root_graphs = self.encoder.encode_batch_with_graphs(root_states, B)
            return root_graphs, [None] * B
        else:
            return [None] * B, [None] * B

    def _extract_nn_priors_from_tree(
        self,
        tree: dict[str, torch.Tensor],
        B: int,
        active: list[bool],
    ) -> list[np.ndarray | None]:
        """Read NN priors for the current root's children from tree storage.

        The tree stores clean NN priors (before noise) for all expanded nodes.
        Called before _apply_root_noise so the priors are still noise-free.
        Reads only the root's children (O(nc) per game), not the full tree.
        """
        row_idx = torch.arange(B, device="cuda")
        fc_cpu = tree["first_child"][row_idx, tree["root_node"]].cpu().numpy()
        nc_cpu = tree["num_children"][row_idx, tree["root_node"]].cpu().numpy()

        nn_priors: list[np.ndarray | None] = [None] * B
        for i in range(B):
            fc_i = int(fc_cpu[i])
            nc_i = int(nc_cpu[i])
            if not active[i] or fc_i < 0 or nc_i == 0:
                continue
            # Slice read: only nc_i entries, not the full max_nodes tensor
            ai_slice = tree["action_idx"][i, fc_i:fc_i + nc_i].cpu().numpy()
            prior_slice = tree["prior"][i, fc_i:fc_i + nc_i].cpu().numpy()
            prior = np.zeros(self._action_space_size, dtype=np.float32)
            for c in range(nc_i):
                action = int(ai_slice[c])
                if 0 <= action < self._action_space_size:
                    prior[action] = float(prior_slice[c])
            nn_priors[i] = prior
        return nn_priors

    def _reroot_tree(
        self,
        tree: dict[str, torch.Tensor],
        B: int,
        active: list[bool],
        chosen_child_nodes: list[int],
    ) -> None:
        """Update root_node to the chosen child for each active game.

        The chosen child's subtree is preserved in the flat node array.
        We only update the logical root pointer and clear the child's
        parent link so backprop stops there correctly.
        """
        new_roots = tree["root_node"].clone()
        for i in range(B):
            if active[i] and chosen_child_nodes[i] >= 0:
                new_roots[i] = chosen_child_nodes[i]
        tree["root_node"].copy_(new_roots)

        # Clear parent_idx for all new roots so backprop terminates correctly.
        # scatter_: tree["parent_idx"][i, new_roots[i]] = -1 for all i.
        tree["parent_idx"].scatter_(
            1,
            new_roots.long().unsqueeze(1),
            torch.full((B, 1), -1, dtype=torch.int32, device="cuda"),
        )

    # ── Win-in-one override ───────────────────────────────────────

    def _check_immediate_wins(
        self,
        root_states: torch.Tensor,
        B: int,
        active: list[bool],
        current_turns: list[int],
        tree: dict[str, torch.Tensor],
        policies: list[np.ndarray],
    ) -> None:
        """Override policies with probability 1 for any game with an immediate win.

        Applies every root child's move to a cloned batch of GPU states and
        checks results in one vectorised call.  This catches winning moves even
        when their policy prior is too low for MCTS to have explored them.

        Sign convention: terminal_value at a root child is from the *child's*
        player perspective (the opponent).  From the PUCT formula
            Q = -total_value[child] / visits
        a terminal_value < 0 at the child means Q > 0 from the root player's
        perspective → winning for the root player.  We confirm by checking the
        raw GPU result code:  WHITE_WINS=1, BLACK_WINS=2.
        """
        # Gather per-game root-child metadata — use sliced reads to avoid
        # pulling the full [B, max_nodes] tensors to CPU every move.
        row_idx = torch.arange(B, device="cuda")
        fc_np  = tree["first_child"][row_idx, tree["root_node"]].cpu().numpy()   # [B]
        nc_np  = tree["num_children"][row_idx, tree["root_node"]].cpu().numpy()  # [B]

        # Build a flat batch: one entry per (game, root-child) pair
        game_indices:    list[int] = []
        child_indices:   list[int] = []
        move_bytes_list: list[np.ndarray] = []
        action_idx_list: list[int] = []

        for i in range(B):
            if not active[i]:
                continue
            fc = int(fc_np[i])
            nc = int(nc_np[i])
            if fc < 0 or nc == 0:
                continue
            # Slice reads: only nc entries for this game's root children
            mb_slice = tree["move_bytes"][i, fc:fc + nc].cpu().numpy()
            ai_slice = tree["action_idx"][i, fc:fc + nc].cpu().numpy()
            for c in range(nc):
                game_indices.append(i)
                child_indices.append(fc + c)
                move_bytes_list.append(mb_slice[c])
                action_idx_list.append(int(ai_slice[c]))

        if not game_indices:
            return

        total = len(game_indices)
        gi_t  = torch.tensor(game_indices, dtype=torch.int64, device="cuda")

        # Clone root states (fancy-index creates a copy, not a view)
        states_test = root_states[gi_t].clone()

        # Build corresponding move bytes tensor from already-fetched slices
        moves_np = np.stack(move_bytes_list, axis=0).astype(np.uint8)
        moves_t = torch.from_numpy(moves_np).cuda()

        # Apply all moves and check results in one GPU call
        self.ext.apply_moves_batch(states_test, moves_t, total)
        results_np = self.ext.check_results_batch(states_test, total).cpu().numpy()

        # Map results back; WHITE_WINS=1, BLACK_WINS=2
        # win_result for current player: turn%2==0 → White → 1, else → 2
        already_overridden: set[int] = set()
        for k in range(total):
            if results_np[k] == 0:
                continue  # IN_PROGRESS — not a win
            i = game_indices[k]
            if i in already_overridden:
                continue  # already found a win for this game
            win_result = 1 if (current_turns[i] % 2 == 0) else 2
            if results_np[k] == win_result:
                win_action = action_idx_list[k]
                if 0 <= win_action < len(policies[i]):
                    policies[i][:] = 0.0
                    policies[i][win_action] = 1.0
                    already_overridden.add(i)

    # ── Action → move bytes lookup ────────────────────────────────

    def _action_to_move_bytes_from_tree(
        self,
        tree: dict[str, torch.Tensor],
        game_idx: int,
        action: int,
    ) -> tuple[np.ndarray | None, int]:
        """Look up move bytes for a given action from the current root's children.

        Returns (move_bytes, child_node_index).  child_node_index is -1 if not found.
        """
        root_node = tree["root_node"][game_idx].item()
        fc = tree["first_child"][game_idx, root_node].item()
        nc = tree["num_children"][game_idx, root_node].item()
        if fc < 0 or nc == 0:
            return None, -1

        # Search root's children for matching action
        ai_slice = tree["action_idx"][game_idx, fc:fc + nc].cpu().numpy()
        mb_slice = tree["move_bytes"][game_idx, fc:fc + nc].cpu().numpy()

        for c in range(nc):
            if ai_slice[c] == action:
                return mb_slice[c].copy(), fc + c
        return None, -1

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

    _BOARD_SIZE = 23
    _NUM_CELLS = _BOARD_SIZE * _BOARD_SIZE  # 529
    _DIR_DCOL = [+1, +1, 0, -1, -1, 0]
    _DIR_DROW = [0, -1, -1, 0, +1, +1]
    _MAX_STACK = 5
    _OFF_HEIGHT = _MAX_STACK * _NUM_CELLS   # 2645
    _OFF_QUEEN_CELL = 3392
    _OFF_TURN = 3412

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
