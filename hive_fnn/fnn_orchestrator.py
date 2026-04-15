"""
Gumbel AlphaZero self-play for the HiveGo-style FNN.

Unlike the two-stage MC approach, the FNN is cheap enough to score ALL legal
successor states upfront. The Gumbel search then selects candidates and runs
sequential halving using precomputed Q-values from the value head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import hive_gpu
from hive_fnn.fnn_features import FNNFeatureEncoder
from hive_fnn.fnn_network import HiveFNN
from hive_fnn.fnn_replay_buffer import FNNTrainingExample


@dataclass
class FNNGumbelConfig:
    num_simulations: int = 128
    max_num_considered_actions: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    temperature: float = 1.0
    temperature_drop_move: int = 20
    batch_size: int = 128
    max_game_length: int = 300
    expansion_mask: int = 0


class FNNGumbelOrchestrator:
    """Gumbel self-play scoring ALL successor states with the FNN."""

    _OFF_TURN = 3412

    def __init__(
        self, net: HiveFNN, config: FNNGumbelConfig | None = None,
    ) -> None:
        self.ext = hive_gpu.load_extension()
        self.encoder = FNNFeatureEncoder()
        self.net = net
        self.config = config or FNNGumbelConfig()
        self._move_size = self.ext.SIZEOF_GPU_MOVE

    def self_play_batch(
        self,
        start_states: torch.Tensor | None = None,
    ) -> list[list[FNNTrainingExample]]:
        B = self.config.batch_size
        cfg = self.config
        device = torch.device("cuda")
        states = (
            start_states.cuda()
            if start_states is not None
            else self.ext.create_initial_states(B, cfg.expansion_mask)
        )

        move_numbers = torch.zeros((B,), dtype=torch.int64, device=device)
        histories: list[list[tuple[np.ndarray, np.ndarray, float]]] = [
            [] for _ in range(B)
        ]
        active_mask = torch.ones((B,), dtype=torch.bool, device=device)

        # Pre-allocate reusable buffers
        game_indices = torch.arange(B, device=device, dtype=torch.int64)
        move_bytes_buf = torch.zeros(
            (B, self._move_size), dtype=torch.uint8, device=device,
        )
        chosen_buf = torch.zeros((B,), dtype=torch.int64, device=device)
        neg_inf = torch.tensor(float("-inf"), device=device)
        num_rounds = max(1, math.ceil(math.log2(max(cfg.max_num_considered_actions, 1))))

        while bool(active_mask.any().item()):
            # ---- Generate legal moves + encode root ----
            legal_moves, num_legal = self.ext.generate_legal_moves_batch(states, B)
            num_actions = num_legal.to(torch.int64)

            has_actions = active_mask & (num_actions > 0)
            newly_finished = active_mask & (num_actions == 0)

            full_policy_pad = None

            if has_actions.any():
                # ---- Build flat successor states ----
                max_legal = legal_moves.shape[1]
                slot_idx = torch.arange(
                    max_legal, device=device, dtype=torch.int64,
                ).unsqueeze(0)
                valid = slot_idx < num_actions.unsqueeze(1)
                action_to_root = game_indices.unsqueeze(1).expand_as(valid)[valid]
                move_indices = slot_idx.expand_as(valid)[valid]

                total_actions = action_to_root.shape[0]

                if total_actions > 0:
                    child_states = states[action_to_root].clone()
                    child_moves = legal_moves[action_to_root, move_indices]
                    self.ext.apply_moves_batch(child_states, child_moves, total_actions)

                    # ---- CUDA feature extraction for root + successors ----
                    root_features = self.ext.extract_fnn_features_batch(
                        states, legal_moves, num_legal, B,
                    )
                    # Child states need their own legal moves for feature extraction
                    child_legal, child_nlegal = self.ext.generate_legal_moves_batch(
                        child_states, total_actions,
                    )
                    succ_features = self.ext.extract_fnn_features_batch(
                        child_states, child_legal, child_nlegal, total_actions,
                    )

                    with torch.no_grad():
                        # Encode root + successors in one batch through the network
                        combined_features = torch.cat([root_features, succ_features], dim=0)
                        all_emb = self.net.encode(combined_features)
                        root_emb = all_emb[:B]
                        succ_emb = all_emb[B:]

                        root_values = self.net.value_head(root_emb)
                        succ_values = self.net.value_head(succ_emb).squeeze(-1)

                        gathered_root = root_emb[action_to_root]
                        action_logits = self.net.score_actions(
                            gathered_root, succ_emb,
                        )

                    # ---- Pad to (B, max_count) with single call ----
                    max_count = int(num_actions.max().item())
                    col_idx = torch.arange(max_count, device=device, dtype=torch.int64).unsqueeze(0)
                    legal_mask = col_idx < num_actions.unsqueeze(1)

                    action_pad = neg_inf.expand(B, max_count).clone()
                    action_pad[legal_mask] = action_logits

                    value_pad = torch.zeros((B, max_count), device=device)
                    value_pad[legal_mask] = succ_values

                    # ---- Gumbel candidate selection ----
                    candidate_count = min(
                        cfg.max_num_considered_actions, max_count,
                    )
                    noise = -torch.log(
                        -torch.log(
                            torch.rand(
                                (B, max_count), device=device, dtype=torch.float32,
                            ).clamp_(1e-4, 1.0 - 1e-4)
                        )
                    )
                    perturbed = torch.where(legal_mask, action_pad + noise, neg_inf)
                    cand_scores, cand_idx = torch.topk(
                        perturbed, k=candidate_count, dim=1,
                    )

                    candidate_slots = torch.arange(
                        candidate_count, device=device, dtype=torch.int64,
                    ).unsqueeze(0)
                    cand_num = num_actions.unsqueeze(1).clamp(max=candidate_count)
                    candidate_valid = candidate_slots < cand_num
                    candidate_alive = candidate_valid & has_actions.unsqueeze(1)

                    # Gather precomputed Q-values for candidates
                    cand_values = value_pad.gather(1, cand_idx)

                    # ---- Sequential halving ----
                    q_sums = torch.zeros((B, candidate_count), device=device)
                    visits = torch.zeros(
                        (B, candidate_count), dtype=torch.int32, device=device,
                    )
                    remaining = torch.full(
                        (B,), cfg.num_simulations, dtype=torch.int32, device=device,
                    )

                    for round_idx in range(num_rounds):
                        alive_counts = candidate_alive.sum(dim=1)
                        if not alive_counts.any():
                            break

                        sims_each = torch.where(
                            alive_counts > 0,
                            (remaining // alive_counts.clamp_min(1)).clamp_min(1),
                            torch.zeros_like(remaining),
                        )
                        sims_expanded = sims_each.unsqueeze(1)
                        q_sums.addcmul_(
                            candidate_alive.float(),
                            (-cand_values) * sims_expanded,
                        )
                        visits += candidate_alive * sims_expanded
                        remaining.sub_(sims_each * alive_counts).clamp_min_(0)

                        if round_idx < num_rounds - 1:
                            visits_f = visits.float()
                            q_mean = q_sums / visits_f.clamp_min(1.0)
                            q_mean.masked_fill_(visits == 0, 0.0)
                            max_visit = visits_f.max(dim=1).values.clamp_min(1.0)
                            sigma = cand_scores + (
                                cfg.c_visit + max_visit.unsqueeze(1)
                            ) * cfg.c_scale * q_mean
                            sigma.masked_fill_(~candidate_alive, float("-inf"))
                            keep_n = ((alive_counts + 1) // 2).clamp_min(1)
                            rank = sigma.argsort(dim=1, descending=True).argsort(dim=1)
                            candidate_alive = candidate_alive & (
                                rank < keep_n.unsqueeze(1)
                            )

                    # ---- Build policy from visit counts ----
                    visit_probs = visits.float()
                    visit_sums = visit_probs.sum(dim=1, keepdim=True)
                    fallback = candidate_valid.float()
                    fallback = fallback / fallback.sum(
                        dim=1, keepdim=True,
                    ).clamp_min(1.0)
                    visit_probs = torch.where(
                        visit_sums > 0,
                        visit_probs / visit_sums.clamp_min(1.0),
                        fallback,
                    )

                    full_policy_pad = torch.zeros(
                        (B, max_count), device=device,
                    )
                    full_policy_pad.scatter_(1, cand_idx, visit_probs)
                    full_policy_pad.mul_(legal_mask)

                    # ---- Record training examples (batch GPU->CPU) ----
                    finished_indices = has_actions.nonzero(as_tuple=False).squeeze(1)
                    if finished_indices.numel() > 0:
                        state_bytes_cpu = (
                            states[finished_indices].detach().cpu().numpy()
                        )
                        root_values_cpu = (
                            root_values[finished_indices, 0].detach().cpu().numpy()
                        )
                        policy_cpu = (
                            full_policy_pad[finished_indices].detach().cpu().numpy()
                        )
                        num_actions_cpu = (
                            num_actions[finished_indices].detach().cpu().numpy()
                        )
                        finished_list = finished_indices.tolist()
                        for row, game_idx in enumerate(finished_list):
                            n = int(num_actions_cpu[row])
                            histories[game_idx].append((
                                state_bytes_cpu[row].copy(),
                                policy_cpu[row, :n].astype(np.float32, copy=True),
                                float(root_values_cpu[row]),
                            ))

            if newly_finished.any():
                active_mask &= ~newly_finished

            # ---- Select and apply move ----
            if has_actions.any() and full_policy_pad is not None:
                finished_indices = has_actions.nonzero(as_tuple=False).squeeze(1)
                greedy_mask = move_numbers[finished_indices] >= cfg.temperature_drop_move
                greedy_rows = finished_indices[greedy_mask]
                sample_rows = finished_indices[~greedy_mask]

                chosen_buf.zero_()
                if greedy_rows.numel() > 0:
                    chosen_buf[greedy_rows] = full_policy_pad[greedy_rows].argmax(dim=1)
                if sample_rows.numel() > 0:
                    sample_policy = full_policy_pad[sample_rows]
                    sample_policy = sample_policy / sample_policy.sum(
                        dim=1, keepdim=True,
                    ).clamp_min(1e-8)
                    chosen_buf[sample_rows] = torch.multinomial(
                        sample_policy, 1,
                    ).squeeze(1)

                move_bytes_buf.zero_()
                move_bytes_buf[finished_indices] = legal_moves[
                    finished_indices, chosen_buf[finished_indices]
                ]
                self.ext.apply_moves_batch(states, move_bytes_buf, B)

            move_numbers += has_actions.to(torch.int64)

            results = self.ext.check_results_batch(states, B)
            active_mask &= (results == 0) & (move_numbers < cfg.max_game_length)

        # ---- Build final training examples ----
        final_results = self.ext.check_results_batch(states, B).cpu().numpy()
        examples: list[list[FNNTrainingExample]] = [[] for _ in range(B)]
        for i in range(B):
            winner = final_results[i]
            for state_bytes, target_pi, _root_v in histories[i]:
                turn = int(state_bytes[self._OFF_TURN]) | (
                    int(state_bytes[self._OFF_TURN + 1]) << 8
                )
                value = self._result_to_value(winner, turn)
                examples[i].append(
                    FNNTrainingExample(
                        state_bytes=state_bytes,
                        policy_target=target_pi.astype(np.float32),
                        value_target=float(value),
                    )
                )
        return examples

    @staticmethod
    def _result_to_value(result: int, turn: int) -> float:
        if result == 3 or result == 0:
            return 0.0
        is_white = turn % 2 == 0
        root_won = (result == 1 and is_white) or (result == 2 and not is_white)
        return 1.0 if root_won else -1.0


class FNNCudaOrchestrator:
    """GPU-native Gumbel self-play — entire game loop runs in one kernel."""

    _OFF_TURN = 3412

    def __init__(
        self, net: HiveFNN, config: FNNGumbelConfig | None = None,
    ) -> None:
        self.ext = hive_gpu.load_extension()
        self.net = net
        self.config = config or FNNGumbelConfig()

    def _flatten_weights(self) -> torch.Tensor:
        """Flatten FNN parameters into the layout expected by the CUDA kernel."""
        n = self.net
        return torch.cat([
            n.fc1.weight.data.flatten(),
            n.fc1.bias.data.flatten(),
            n.ln1.weight.data.flatten(),
            n.ln1.bias.data.flatten(),
            n.fc2.weight.data.flatten(),
            n.fc2.bias.data.flatten(),
            n.value_fc.weight.data.flatten(),
            n.value_fc.bias.data.flatten(),
            n.action_fc1.weight.data.flatten(),
            n.action_fc1.bias.data.flatten(),
            n.action_fc2.weight.data.flatten(),
            n.action_fc2.bias.data.flatten(),
        ]).contiguous()

    def self_play_batch(
        self,
        start_states: torch.Tensor | None = None,
    ) -> list[list[FNNTrainingExample]]:
        cfg = self.config
        B = cfg.batch_size
        self.net.eval()

        weights = self._flatten_weights()
        seed = int(torch.randint(0, 2**62, (1,)).item())

        (states, policy_probs, policy_indices, num_legal,
         num_candidates, lengths, results) = self.ext.fnn_selfplay_batch(
            weights,
            self.net.config.hidden_dim,
            self.net.config.embed_dim,
            self.net.config.action_hidden,
            B, cfg.max_game_length,
            cfg.num_simulations, cfg.max_num_considered_actions,
            cfg.c_visit, cfg.c_scale,
            cfg.temperature_drop_move, cfg.expansion_mask,
            seed,
        )

        return self._build_examples(
            states, policy_probs, policy_indices,
            num_legal, num_candidates, lengths, results,
        )

    def _build_examples(
        self,
        states: torch.Tensor,
        policy_probs: torch.Tensor,
        policy_indices: torch.Tensor,
        num_legal: torch.Tensor,
        num_candidates: torch.Tensor,
        lengths: torch.Tensor,
        results: torch.Tensor,
    ) -> list[list[FNNTrainingExample]]:
        B = states.shape[0]
        s_cpu = states.cpu().numpy()
        pp_cpu = policy_probs.cpu().numpy()
        pi_cpu = policy_indices.cpu().numpy()
        nl_cpu = num_legal.cpu().numpy()
        nc_cpu = num_candidates.cpu().numpy()
        len_cpu = lengths.cpu().numpy()
        res_cpu = results.cpu().numpy()

        examples: list[list[FNNTrainingExample]] = []
        for i in range(B):
            game_ex: list[FNNTrainingExample] = []
            gl = int(len_cpu[i])
            winner = int(res_cpu[i])

            for step in range(gl):
                sb = s_cpu[i, step].copy()
                n = int(nl_cpu[i, step])
                ncand = int(nc_cpu[i, step])

                # Reconstruct full policy (over num_legal moves)
                full_policy = np.zeros(n, dtype=np.float32)
                for k in range(ncand):
                    idx = int(pi_cpu[i, step, k])
                    if 0 <= idx < n:
                        full_policy[idx] = float(pp_cpu[i, step, k])

                ps = full_policy.sum()
                if ps > 0:
                    full_policy /= ps
                elif n > 0:
                    full_policy[:] = 1.0 / n

                turn = int(sb[self._OFF_TURN]) | (
                    int(sb[self._OFF_TURN + 1]) << 8
                )
                value = self._result_to_value(winner, turn)
                game_ex.append(FNNTrainingExample(
                    state_bytes=sb,
                    policy_target=full_policy,
                    value_target=float(value),
                ))

            examples.append(game_ex)
        return examples

    @staticmethod
    def _result_to_value(result: int, turn: int) -> float:
        if result == 3 or result == 0:
            return 0.0
        is_white = turn % 2 == 0
        root_won = (result == 1 and is_white) or (result == 2 and not is_white)
        return 1.0 if root_won else -1.0


def _flat_to_padded(
    flat_values: torch.Tensor,
    counts: torch.Tensor,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Convert a flat ragged vector into a padded [B, max_count] tensor."""
    B = counts.shape[0]
    if B == 0:
        return flat_values.new_full((0, 0), pad_value)
    max_count = int(counts.max().item())
    if max_count == 0 or flat_values.numel() == 0:
        return flat_values.new_full((B, max_count), pad_value)
    col_idx = torch.arange(
        max_count, device=counts.device, dtype=torch.int64,
    ).unsqueeze(0)
    mask = col_idx < counts.unsqueeze(1)
    out = flat_values.new_full((B, max_count), pad_value)
    out[mask] = flat_values
    return out
