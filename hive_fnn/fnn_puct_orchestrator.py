"""
Plain PUCT MCTS self-play orchestrator for the HiveGo-style FNN.

This keeps the same tree expansion / backprop kernels as the Gumbel-root path,
but uses ordinary PUCT selection at the root instead of Gumbel sequential
halving. Priors still come from the FNN successor-state scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import hive_gpu
from hive_fnn.fnn_mcts_orchestrator import FNNMCTSConfig, FNNMCTSOrchestrator
from hive_fnn.fnn_network import HiveFNN
from hive_fnn.fnn_replay_buffer import FNNTrainingExample

_OFF_TURN = 3412


@dataclass
class FNNPUCTConfig:
    num_simulations: int = 128
    c_puct: float = 1.25
    temperature: float = 1.0
    temperature_drop_move: int = 20
    batch_size: int = 128
    max_game_length: int = 300
    expansion_mask: int = 0
    wave_size: int = 16
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_tree_nodes: int = 65536
    # Rebase each game's tree to a fresh root after applying a move.
    # This prevents node-id growth across plies from exhausting the fixed
    # tree node pool in long games.
    rebase_tree_each_move: bool = True


class FNNPUCTOrchestrator:
    """Plain PUCT MCTS for the FNN backbone."""

    def __init__(
        self,
        net: HiveFNN,
        config: FNNPUCTConfig | None = None,
    ) -> None:
        self.ext = hive_gpu.load_extension()
        self.config = config or FNNPUCTConfig()
        self.net = net
        self._move_size = self.ext.SIZEOF_GPU_MOVE
        self._max_nodes = int(self.config.max_tree_nodes)
        self._max_legal = int(self.ext.MAX_LEGAL_MOVES)
        self._eval_helper = FNNMCTSOrchestrator(
            net,
            FNNMCTSConfig(
                num_simulations=self.config.num_simulations,
                batch_size=self.config.batch_size,
                max_game_length=self.config.max_game_length,
                expansion_mask=self.config.expansion_mask,
                wave_size=self.config.wave_size,
                dirichlet_alpha=self.config.dirichlet_alpha,
                dirichlet_epsilon=self.config.dirichlet_epsilon,
                max_tree_nodes=self.config.max_tree_nodes,
                rebase_tree_each_move=self.config.rebase_tree_each_move,
            ),
        )

    def _alloc_tree(self, B: int) -> dict[str, torch.Tensor]:
        M = self._max_nodes
        dev = "cuda"
        return {
            "visit_count": torch.zeros(B, M, dtype=torch.int32, device=dev),
            "total_value": torch.zeros(B, M, dtype=torch.float32, device=dev),
            "prior": torch.zeros(B, M, dtype=torch.float32, device=dev),
            "parent_idx": torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "move_bytes": torch.zeros(B, M, self._move_size, dtype=torch.uint8, device=dev),
            "action_idx": torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "first_child": torch.full((B, M), -1, dtype=torch.int32, device=dev),
            "num_children": torch.zeros(B, M, dtype=torch.int32, device=dev),
            "is_terminal": torch.zeros(B, M, dtype=torch.int8, device=dev),
            "terminal_value": torch.zeros(B, M, dtype=torch.float32, device=dev),
            "node_count": torch.ones(B, dtype=torch.int32, device=dev),
            "root_node": torch.zeros(B, dtype=torch.int32, device=dev),
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

    def _eval_states(
        self,
        states: torch.Tensor,
        legal_moves: torch.Tensor,
        num_legal: torch.Tensor,
        total: int,
        root_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._eval_helper._eval_states(
            states, legal_moves, num_legal, total, root_features,
        )

    def _find_immediate_wins(
        self,
        states: torch.Tensor,
        legal_moves: torch.Tensor,
        num_legal: torch.Tensor,
        total: int,
    ) -> torch.Tensor:
        return self._eval_helper._find_immediate_wins(states, legal_moves, num_legal, total)

    def self_play_batch(
        self,
        start_states: torch.Tensor | None = None,
    ) -> list[list[FNNTrainingExample]]:
        B = self.config.batch_size
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
            if max_n == 0:
                break

            priors_per_legal, _root_vals = self._eval_states(
                states, legal_moves, num_legal, B, root_features,
            )
            if bool(has_immediate_win.any().item()):
                priors_per_legal.zero_()
                priors_per_legal[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0

            game_active_t = has_actions.to(torch.int8)
            self._expand_root_if_needed(
                tree, states, legal_moves, num_legal,
                priors_per_legal, game_active_t, B,
            )
            self._apply_root_dirichlet(tree, B, has_actions)

            if bool(has_immediate_win.any().item()):
                slot_visits = torch.zeros(B, self._max_legal, dtype=torch.int32, device=dev)
                slot_visits[has_immediate_win, immediate_wins[has_immediate_win]] = cfg.num_simulations
                probs_pad = torch.zeros(B, self._max_legal, dtype=torch.float32, device=dev)
                probs_pad[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0
            else:
                valid_slot = (
                    torch.arange(self._max_legal, device=dev).unsqueeze(0)
                    < n_per_game.unsqueeze(1)
                )
                alive_mask = valid_slot.to(torch.int8)

                self._run_simulations(
                    tree, states, game_active_t, alive_mask, B, cfg.num_simulations,
                )

                slot_visits, _slot_q = self._gather_root_child_stats(tree, B)
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

            has_actions_cpu = has_actions.cpu().numpy()
            states_cpu = states.cpu().numpy()
            probs_cpu = probs_pad.cpu().numpy()
            for i in range(B):
                if not bool(has_actions_cpu[i]):
                    continue
                n = int(nlegal_np[i])
                histories[i].append((
                    states_cpu[i].copy(),
                    probs_cpu[i, :n].astype(np.float32, copy=True),
                ))

            chosen_slot = torch.zeros((B,), dtype=torch.int64, device=dev)
            active_idx = torch.nonzero(has_actions, as_tuple=False).squeeze(1)
            if bool(has_immediate_win.any().item()):
                chosen_slot[has_immediate_win] = immediate_wins[has_immediate_win]
            elif active_idx.numel() > 0:
                greedy_mask = move_numbers[active_idx] >= cfg.temperature_drop_move
                greedy_rows = active_idx[greedy_mask]
                sample_rows = active_idx[~greedy_mask]

                if greedy_rows.numel() > 0:
                    chosen_slot[greedy_rows] = probs_pad[greedy_rows].argmax(dim=1)
                if sample_rows.numel() > 0:
                    sample_policy = probs_pad[sample_rows]
                    sample_policy = sample_policy / sample_policy.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    chosen_slot[sample_rows] = torch.multinomial(sample_policy, 1).squeeze(1)

            move_bytes = torch.zeros((B, self._move_size), dtype=torch.uint8, device=dev)
            if active_idx.numel() > 0:
                slot_sel = chosen_slot[active_idx].clamp(max=self._max_legal - 1)
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
            active_mask = active_mask & (results == 0) & (move_numbers < cfg.max_game_length)

        final_results = self.ext.check_results_batch(states, B).cpu().numpy()
        examples: list[list[FNNTrainingExample]] = [[] for _ in range(B)]
        for i in range(B):
            winner = final_results[i]
            use_for_value = (winner != 0)
            for state_bytes, target_pi in histories[i]:
                turn = int(state_bytes[_OFF_TURN]) | (int(state_bytes[_OFF_TURN + 1]) << 8)
                value = self._result_to_value(winner, turn)
                examples[i].append(FNNTrainingExample(
                    state_bytes=state_bytes,
                    policy_target=target_pi.astype(np.float32),
                    value_target=float(value),
                    use_for_value=use_for_value,
                ))
        return examples

    def _expand_root_if_needed(self, tree, states, legal_moves, num_legal, priors_per_legal, game_active_t, B):
        row = torch.arange(B, device="cuda")
        root_node = tree["root_node"]
        fc = tree["first_child"][row, root_node]
        needs = (fc < 0) & game_active_t.to(torch.bool)
        if not torch.any(needs):
            return
        leaf_indices = root_node.to(torch.int32).clone()
        leaf_indices = torch.where(needs, leaf_indices, torch.full_like(leaf_indices, -1))
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

    def _run_simulations(self, tree, root_states, game_active_t, alive_mask, B, num_sims):
        cfg = self.config
        W = cfg.wave_size
        num_waves = (num_sims + W - 1) // W
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

    def _gather_root_child_stats(self, tree, B) -> tuple[torch.Tensor, torch.Tensor]:
        row = torch.arange(B, device="cuda")
        root = tree["root_node"]
        fc = tree["first_child"][row, root].cpu().numpy()
        nc = tree["num_children"][row, root].cpu().numpy()

        visits = torch.zeros(B, self._max_legal, dtype=torch.int32, device="cuda")
        q = torch.zeros(B, self._max_legal, dtype=torch.float32, device="cuda")

        for i in range(B):
            fci = int(fc[i])
            nci = int(nc[i])
            if fci < 0 or nci == 0:
                continue
            lim = min(nci, self._max_legal)
            vc = tree["visit_count"][i, fci:fci + lim]
            tv = tree["total_value"][i, fci:fci + lim]
            visits[i, :lim] = vc
            q[i, :lim] = torch.where(
                vc > 0, -tv / vc.clamp(min=1).float(),
                torch.zeros_like(tv),
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
