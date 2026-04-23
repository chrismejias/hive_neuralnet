"""
Gumbel AlphaZero + MCTS self-play orchestrator for the FNN.

Uses the existing GPU MCTS tree kernels (`mcts_select_with_root_mask_batch`,
`mcts_expand_and_backprop_dense_priors_batch`) with per-legal-move dense
priors — no ACTION_SPACE indirection.

Tree priors come from FNN action scoring: encode root, encode each legal
successor state, score each successor relative to the root, softmax over
legal slots. Leaf value comes from the value head applied to the leaf
root embedding.

This orchestrator performs true multi-ply MCTS: each simulation extends
the tree by one node via PUCT selection, and Sequential Halving at the
root is implemented via an alive-mask over root children.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import hive_gpu
from hive_fnn.fnn_network import HiveFNN
from hive_fnn.fnn_replay_buffer import FNNTrainingExample

_OFF_TURN = 3412
_GUMBEL_WAVE_SCHEDULE = (2, 4, 8, 16)


@dataclass
class FNNMCTSConfig:
    num_simulations:             int   = 128
    max_num_considered_actions:  int   = 16
    c_puct:                      float = 1.25
    c_visit:                     float = 50.0
    c_scale:                     float = 1.0
    temperature:                 float = 1.0
    temperature_drop_move:       int   = 20
    batch_size:                  int   = 128
    max_game_length:             int   = 300
    expansion_mask:              int   = 0
    wave_parallel:               bool  = True
    wave_size:                   int   = 4
    dirichlet_alpha:             float = 0.3
    dirichlet_epsilon:           float = 0.25
    max_tree_nodes:              int   = 65536
    # Rebase each game's tree to a fresh root after applying a move.
    # This prevents node-id growth across plies from exhausting the fixed
    # tree node pool in long games.
    rebase_tree_each_move:       bool  = True


class FNNMCTSOrchestrator:
    """Gumbel AlphaZero with true MCTS tree search, FNN backbone."""

    def __init__(
        self,
        net: HiveFNN,
        config: FNNMCTSConfig | None = None,
    ) -> None:
        self.ext        = hive_gpu.load_extension()
        self.config     = config or FNNMCTSConfig()
        self.net        = net
        self._move_size = self.ext.SIZEOF_GPU_MOVE
        self._max_nodes = int(self.config.max_tree_nodes)
        self._max_legal = int(self.ext.MAX_LEGAL_MOVES)
        self._slot_idx = torch.arange(
            self._max_legal, device="cuda", dtype=torch.int64,
        ).unsqueeze(0)
        self._row_idx_cache: dict[int, torch.Tensor] = {}

    def _row_indices(self, n: int) -> torch.Tensor:
        cached = self._row_idx_cache.get(n)
        if cached is None:
            cached = torch.arange(n, device="cuda", dtype=torch.int64).unsqueeze(1)
            self._row_idx_cache[n] = cached
        return cached

    # ── Tree tensor management ────────────────────────────────────────

    def _alloc_tree(self, B: int) -> dict[str, torch.Tensor]:
        M = self._max_nodes
        dev = "cuda"
        return {
            "visit_count":    torch.zeros(B, M, dtype=torch.int32,  device=dev),
            "total_value":    torch.zeros(B, M, dtype=torch.float32, device=dev),
            "prior":          torch.zeros(B, M, dtype=torch.float32, device=dev),
            "parent_idx":     torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "move_bytes":     torch.zeros(B, M, self._move_size, dtype=torch.uint8, device=dev),
            "action_idx":     torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "first_child":    torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "num_children":   torch.zeros(B, M, dtype=torch.int32, device=dev),
            "is_terminal":    torch.zeros(B, M, dtype=torch.int8, device=dev),
            "terminal_value": torch.zeros(B, M, dtype=torch.float32, device=dev),
            "node_count":     torch.ones(B, dtype=torch.int32, device=dev),
            "root_node":      torch.zeros(B, dtype=torch.int32, device=dev),
        }

    def _reset_tree(self, tree: dict[str, torch.Tensor]) -> None:
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
        tree["node_count"].fill_(1)
        tree["root_node"].zero_()

    def _tree_args(self, tree: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [
            tree["visit_count"], tree["total_value"], tree["prior"],
            tree["parent_idx"], tree["move_bytes"], tree["action_idx"],
            tree["first_child"], tree["num_children"],
            tree["is_terminal"], tree["terminal_value"], tree["node_count"],
        ]

    # ── FNN evaluation helper ─────────────────────────────────────────

    def _eval_states(
        self,
        states: torch.Tensor,
        legal_moves: torch.Tensor,
        num_legal: torch.Tensor,
        total: int,
        root_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode states + score every legal successor.

        Returns:
            priors_per_legal: (total, MAX_LEGAL) softmax over legal slots,
                zeros elsewhere.
            root_values:      (total,) scalar value estimate per state.
        """
        dev = "cuda"
        MAX_L = self._max_legal
        n64 = num_legal.to(torch.int64)

        if root_features is None:
            root_features = self.ext.extract_fnn_features_batch(
                states, legal_moves, num_legal, total,
            )   # (total, feat_dim)

        if total == 0:
            empty = torch.zeros(0, MAX_L, dtype=torch.float32, device=dev)
            return empty, empty.new_zeros((0,))

        slot_idx = self._slot_idx
        valid = slot_idx < n64.unsqueeze(1)          # (total, MAX_L)
        action_to_root = self._row_indices(total).expand_as(valid)[valid]
        move_indices = slot_idx.expand_as(valid)[valid]
        N_total = int(action_to_root.shape[0])

        if N_total == 0:
            with torch.inference_mode():
                root_emb = self.net.encode(root_features)
                root_values = self.net.value_head(root_emb).squeeze(-1).float()
            priors_per_legal = torch.zeros(
                total, MAX_L, dtype=torch.float32, device=dev,
            )
            return priors_per_legal, root_values

        succ_features = self.ext.fnn_successor_features_batch(
            states, legal_moves, action_to_root, move_indices, N_total,
        )

        with torch.inference_mode():
            combined = torch.cat([root_features, succ_features], dim=0)
            all_emb = self.net.encode(combined)
            root_emb = all_emb[:total]
            succ_emb = all_emb[total:]
            root_values = self.net.value_head(root_emb).squeeze(-1).float()
            gathered_root = root_emb[action_to_root]
            action_logits = self.net.score_actions(
                gathered_root, succ_emb,
            ).float()                                  # (N_total,)

        # Scatter logits back into the fixed-width legal tensor. Keeping
        # MAX_L avoids a per-eval CPU sync on num_legal.max().item().
        legal_logits = torch.full(
            (total, MAX_L), -1e30, dtype=torch.float32, device=dev,
        )
        legal_logits[valid] = action_logits

        # Softmax over valid slots (padded slots stay effectively -inf).
        prior_pad = torch.softmax(legal_logits, dim=1)
        prior_pad = prior_pad.masked_fill(~valid, 0.0)
        return prior_pad, root_values

    def _find_immediate_wins(
        self,
        states: torch.Tensor,
        legal_moves: torch.Tensor,
        num_legal: torch.Tensor,
        total: int,
    ) -> torch.Tensor:
        """Return winning legal-move index per game, or -1 if none exists."""
        dev = states.device
        winners = torch.full((total,), -1, dtype=torch.int64, device=dev)
        n64 = num_legal.to(torch.int64)
        if total == 0 or not bool((n64 > 0).any().item()):
            return winners

        max_legal = legal_moves.shape[1]
        slot_idx = torch.arange(max_legal, device=dev, dtype=torch.int64).unsqueeze(0)
        valid = slot_idx < n64.unsqueeze(1)
        action_to_root = torch.arange(total, device=dev, dtype=torch.int64).unsqueeze(1).expand_as(valid)[valid]
        move_indices = slot_idx.expand_as(valid)[valid]
        n_total = int(action_to_root.shape[0])
        if n_total == 0:
            return winners

        child_states = states[action_to_root].clone()
        child_moves = legal_moves[action_to_root, move_indices]
        self.ext.apply_moves_batch(child_states, child_moves, n_total)
        results = self.ext.check_results_batch(child_states, n_total)

        root_turn = states[action_to_root, _OFF_TURN].to(torch.int32)
        is_white = (root_turn % 2 == 0)
        win_mask = ((results == 1) & is_white) | ((results == 2) & ~is_white)
        if not bool(win_mask.any().item()):
            return winners

        win_roots = action_to_root[win_mask]
        win_moves = move_indices[win_mask]
        for i in range(total):
            mask_i = win_roots == i
            if bool(mask_i.any().item()):
                winners[i] = win_moves[mask_i][0]
        return winners

    # ── Public API ────────────────────────────────────────────────────

    def self_play_batch(
        self,
        start_states: torch.Tensor | None = None,
    ) -> list[list[FNNTrainingExample]]:
        B   = self.config.batch_size
        cfg = self.config
        dev = "cuda"

        states = (
            start_states.cuda()
            if start_states is not None
            else self.ext.create_initial_states(B, cfg.expansion_mask)
        )

        tree = self._alloc_tree(B)
        self._reset_tree(tree)

        active_mask = torch.ones((B,), dtype=torch.bool, device=dev)
        move_numbers = torch.zeros((B,), dtype=torch.int64, device=dev)
        histories: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(B)]

        while bool(active_mask.any().item()):
            legal_moves, num_legal, root_features = (
                self.ext.generate_legal_moves_and_fnn_features_batch(states, B)
            )
            n_per_game = num_legal.to(torch.int64)
            nlegal_np = num_legal.cpu().numpy()

            has_actions = active_mask & (n_per_game > 0)
            newly_finished = active_mask & (n_per_game == 0)
            if newly_finished.any():
                active_mask = active_mask & ~newly_finished

            immediate_wins = self._find_immediate_wins(states, legal_moves, num_legal, B)
            has_immediate_win = immediate_wins >= 0

            max_n = int(n_per_game.max().item()) if B > 0 else 0
            max_k = min(cfg.max_num_considered_actions, max_n) if max_n > 0 else 0
            if max_k == 0:
                break

            priors_per_legal, _root_vals = self._eval_states(
                states, legal_moves, num_legal, B, root_features,
            )
            if bool(has_immediate_win.any().item()):
                priors_per_legal.zero_()
                priors_per_legal[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0

            # Padded logits over MAX_LEGAL slots (log of prior, with -inf on invalid).
            MAX_L = self._max_legal
            slot_idx = torch.arange(MAX_L, device=dev).unsqueeze(0)
            valid_slot = slot_idx < n_per_game.unsqueeze(1)
            safe_prior = priors_per_legal.clamp(min=1e-20)
            legal_logits = torch.where(
                valid_slot, safe_prior.log(),
                torch.full_like(priors_per_legal, -1e30),
            )

            # ── Gumbel top-k over legal slots ─────────────────────────
            u = torch.rand(B, MAX_L, device=dev).clamp(1e-4, 1 - 1e-4)
            gumbel = -torch.log(-torch.log(u))
            gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
            perturbed = gumbel + legal_logits
            _, topk_slots = torch.topk(perturbed, max_k, dim=1)

            # ── Expand root if needed ────────────────────────────────
            game_active_t = has_actions.to(torch.int8)
            self._expand_root_if_needed(
                tree, states, legal_moves, num_legal,
                priors_per_legal, game_active_t, B,
            )
            self._apply_root_dirichlet(tree, B, has_actions)

            if bool(has_immediate_win.any().item()):
                slot_visits = torch.zeros(B, MAX_L, dtype=torch.int32, device=dev)
                slot_visits[has_immediate_win, immediate_wins[has_immediate_win]] = cfg.num_simulations
                probs_pad = torch.zeros(B, MAX_L, dtype=torch.float32, device=dev)
                probs_pad[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0
                chosen_slot = torch.where(
                    has_immediate_win,
                    immediate_wins,
                    torch.zeros_like(immediate_wins),
                )
            else:
                alive_mask = torch.zeros(B, MAX_L, dtype=torch.int8, device=dev)
                alive_mask.scatter_(1, topk_slots, 1)
                alive_mask = alive_mask * valid_slot.to(torch.int8)

                # ── Sequential Halving ────────────────────────────────────
                n_rounds = max(1, math.ceil(math.log2(max(max_k, 2))))
                sims_per_round = max(1, cfg.num_simulations // n_rounds)

                for round_i in range(n_rounds):
                    round_wave_size = (
                        _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)]
                        if cfg.wave_parallel else 1
                    )
                    self._run_simulations(
                        tree, states, game_active_t, alive_mask, B, sims_per_round,
                        wave_size=round_wave_size,
                    )

                    alive_counts = alive_mask.sum(dim=1)
                    cur_alive_max = int(alive_counts.max().item())
                    if cur_alive_max <= 1:
                        continue
                    num_keep = max(1, cur_alive_max // 2)

                    slot_visits, slot_q = self._gather_root_child_stats(tree, B)
                    max_nv = int(slot_visits.max().item())
                    sigma_norm = (cfg.c_visit + max_nv) * cfg.c_scale
                    sigma_score = (gumbel + legal_logits + sigma_norm * slot_q).float()
                    sigma_score = torch.where(
                        alive_mask.bool(), sigma_score,
                        torch.full_like(sigma_score, -1e30),
                    )

                    _, keep_slots = torch.topk(sigma_score, num_keep, dim=1)
                    new_alive = torch.zeros_like(alive_mask)
                    new_alive.scatter_(1, keep_slots, 1)
                    new_alive = new_alive * valid_slot.to(torch.int8)
                    alive_mask = new_alive

                # ── Final action selection ────────────────────────────────
                slot_visits, slot_q = self._gather_root_child_stats(tree, B)
                max_nv = int(slot_visits.max().item())
                sigma_norm = (cfg.c_visit + max_nv) * cfg.c_scale
                final_sigma = (gumbel + legal_logits + sigma_norm * slot_q).float()
                final_sigma = torch.where(
                    alive_mask.bool(), final_sigma,
                    torch.full_like(final_sigma, -1e30),
                )
                chosen_slot = torch.argmax(final_sigma, dim=1)

                # ── Policy target = visit distribution over legal moves ───
                visits_f = slot_visits.float()
                vsum = visits_f.sum(dim=1, keepdim=True)
                probs_pad = torch.where(
                    vsum > 0,
                    visits_f / vsum.clamp(min=1.0),
                    torch.zeros_like(visits_f),
                )

                uniform = valid_slot.float()
                ucount = uniform.sum(dim=1, keepdim=True).clamp(min=1.0)
                uniform = uniform / ucount
                probs_pad = torch.where(vsum > 0, probs_pad, uniform)

            # ── Record history ───────────────────────────────────────
            has_actions_cpu = has_actions.cpu().numpy()
            states_cpu = states.cpu().numpy()
            probs_cpu  = probs_pad.cpu().numpy()
            for i in range(B):
                if not bool(has_actions_cpu[i]):
                    continue
                n = int(nlegal_np[i])
                histories[i].append((
                    states_cpu[i].copy(),
                    probs_cpu[i, :n].astype(np.float32, copy=True),
                ))

            # ── Apply chosen moves + re-root ──────────────────────────
            move_bytes = torch.zeros(
                (B, self._move_size), dtype=torch.uint8, device=dev,
            )
            active_idx = torch.nonzero(has_actions, as_tuple=False).squeeze(1)
            if active_idx.numel() > 0:
                slot_sel = chosen_slot[active_idx].clamp(max=MAX_L - 1)
                move_bytes[active_idx] = legal_moves[active_idx, slot_sel]

            self.ext.apply_moves_batch(states, move_bytes, B)

            chosen_child_nodes = [-1] * B
            fc_col = tree["first_child"][
                torch.arange(B, device=dev), tree["root_node"],
            ].cpu().numpy()
            chosen_cpu = chosen_slot.cpu().numpy()
            for i in range(B):
                if not bool(has_actions_cpu[i]):
                    continue
                fci = int(fc_col[i])
                if fci >= 0:
                    chosen_child_nodes[i] = fci + int(chosen_cpu[i])
            self._reroot_tree(tree, B, has_actions_cpu, chosen_child_nodes)

            move_numbers = move_numbers + has_actions.to(torch.int64)

            results = self.ext.check_results_batch(states, B)
            active_mask = active_mask & (results == 0) & (
                move_numbers < cfg.max_game_length
            )

        # ── Build training examples ──────────────────────────────────
        final_results = self.ext.check_results_batch(states, B).cpu().numpy()
        examples: list[list[FNNTrainingExample]] = [[] for _ in range(B)]
        for i in range(B):
            winner = final_results[i]
            use_for_value = (winner != 0)
            for state_bytes, target_pi in histories[i]:
                turn = int(state_bytes[_OFF_TURN]) | (
                    int(state_bytes[_OFF_TURN + 1]) << 8
                )
                value = self._result_to_value(winner, turn)
                examples[i].append(FNNTrainingExample(
                    state_bytes=state_bytes,
                    policy_target=target_pi.astype(np.float32),
                    value_target=float(value),
                    use_for_value=use_for_value,
                ))
        return examples

    # ── Helpers ───────────────────────────────────────────────────────

    def _expand_root_if_needed(
        self,
        tree, states, legal_moves, num_legal,
        priors_per_legal, game_active_t, B,
    ):
        row = torch.arange(B, device="cuda")
        root_node = tree["root_node"]
        fc = tree["first_child"][row, root_node]
        needs = (fc < 0) & game_active_t.to(torch.bool)
        if not torch.any(needs):
            return

        leaf_indices = root_node.to(torch.int32).clone()
        leaf_indices = torch.where(
            needs, leaf_indices, torch.full_like(leaf_indices, -1),
        )
        results = self.ext.check_results_batch(states, B)
        self.ext.mcts_expand_dense_priors_batch(
            *self._tree_args(tree),
            leaf_indices, states,
            legal_moves, num_legal, priors_per_legal, results,
            B, B, self._max_nodes,
        )

    def _apply_root_dirichlet(self, tree, B, active_mask):
        cfg = self.config
        row = torch.arange(B, device="cuda")
        nc_t = tree["num_children"][row, tree["root_node"]]
        nc_cpu = nc_t.cpu().numpy()
        active_cpu = active_mask.cpu().numpy()
        max_nc = int(nc_cpu.max()) if nc_cpu.size > 0 and nc_cpu.max() > 0 else 1

        noise = np.zeros((B, max_nc), dtype=np.float32)
        for i in range(B):
            nc = int(nc_cpu[i])
            if nc == 0 or not bool(active_cpu[i]):
                continue
            alpha = max(0.03, min(cfg.dirichlet_alpha, 10.0 / nc))
            noise[i, :nc] = np.random.dirichlet([alpha] * nc)
        noise_t = torch.from_numpy(noise).cuda()
        self.ext.mcts_apply_root_noise(
            *self._tree_args(tree),
            noise_t, tree["root_node"], max_nc,
            cfg.dirichlet_epsilon, 1.0,
            B, self._max_nodes,
        )

    def _run_simulations(
        self, tree, root_states, game_active_t, alive_mask, B, num_sims,
        wave_size=None,
    ):
        cfg = self.config
        W = max(1, int(cfg.wave_size if wave_size is None else wave_size))
        num_waves = math.ceil(num_sims / W)
        state_size = root_states.shape[1]
        leaf_states = torch.zeros(W * B, state_size, dtype=torch.uint8, device="cuda")

        for wave in range(num_waves):
            actual_w = min(W, num_sims - wave * W)
            total = actual_w * B

            leaf_idx, move_paths, path_lens, vl_paths, vl_lens = (
                self.ext.mcts_select_with_root_mask_batch(
                    *self._tree_args(tree),
                    game_active_t, tree["root_node"],
                    alive_mask, self._max_legal,
                    cfg.c_puct, B, actual_w, self._max_nodes,
                )
            )

            self.ext.mcts_replay_batch(
                root_states, leaf_states[:total],
                move_paths[:total], path_lens[:total], leaf_idx[:total],
                B, total,
            )

            results = self.ext.check_results_batch(leaf_states[:total], total)
            legal_moves, num_legal, leaf_features = (
                self.ext.generate_legal_moves_and_fnn_features_batch(
                    leaf_states[:total], total,
                )
            )

            priors_leaf, leaf_vals = self._eval_states(
                leaf_states[:total], legal_moves, num_legal, total, leaf_features,
            )

            self.ext.mcts_expand_and_backprop_dense_priors_batch(
                *self._tree_args(tree),
                leaf_idx[:total], leaf_states[:total],
                legal_moves, num_legal, priors_leaf, results,
                leaf_vals, vl_paths[:total], vl_lens[:total],
                B, total, self._max_nodes,
            )

    def _gather_root_child_stats(
        self, tree, B,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = torch.arange(B, device="cuda", dtype=torch.long)
        root = tree["root_node"]
        fc = tree["first_child"][row, root].to(torch.long)
        nc = tree["num_children"][row, root].to(torch.long)

        slots = self._slot_idx.expand(B, -1)
        valid = (fc.unsqueeze(1) >= 0) & (slots < nc.unsqueeze(1))
        child_idx = (fc.unsqueeze(1) + slots).clamp(min=0, max=self._max_nodes - 1)
        row2 = row.unsqueeze(1).expand_as(child_idx)

        visits_raw = tree["visit_count"][row2, child_idx]
        total_raw = tree["total_value"][row2, child_idx]
        visits = torch.where(valid, visits_raw, torch.zeros_like(visits_raw))
        q = torch.where(
            valid & (visits_raw > 0),
            -total_raw / visits_raw.clamp(min=1).float(),
            torch.zeros_like(total_raw),
        )
        return visits, q

    def _reroot_tree(self, tree, B, active_cpu, chosen_child_nodes):
        if self.config.rebase_tree_each_move:
            # Hard rebase: drop previous ply's tree for moved games and start
            # the next ply from a fresh root node at index 0.
            for i in range(B):
                if not (bool(active_cpu[i]) and chosen_child_nodes[i] >= 0):
                    continue
                tree["visit_count"][i].zero_()
                tree["total_value"][i].zero_()
                tree["prior"][i].zero_()
                tree["parent_idx"][i].fill_(-1)
                tree["move_bytes"][i].zero_()
                tree["action_idx"][i].fill_(-1)
                tree["first_child"][i].fill_(-1)
                tree["num_children"][i].zero_()
                tree["is_terminal"][i].zero_()
                tree["terminal_value"][i].zero_()
                tree["node_count"][i] = 1
                tree["root_node"][i] = 0
            return

        # Legacy behavior: keep subtree rooted at chosen child.
        new_roots = tree["root_node"].clone()
        for i in range(B):
            if bool(active_cpu[i]) and chosen_child_nodes[i] >= 0:
                new_roots[i] = chosen_child_nodes[i]
        tree["root_node"].copy_(new_roots)
        tree["parent_idx"].scatter_(
            1, new_roots.long().unsqueeze(1),
            torch.full((B, 1), -1, dtype=torch.int32, device="cuda"),
        )

    @staticmethod
    def _result_to_value(result: int, turn: int) -> float:
        if result == 0 or result == 3:
            return 0.0
        is_white = (turn % 2 == 0)
        root_won = (result == 1 and is_white) or (result == 2 and not is_white)
        return 1.0 if root_won else -1.0
