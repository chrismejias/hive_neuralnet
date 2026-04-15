from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import hive_gpu
from hive_gpu.gpu_encoder import GPUTransformerEncoder
from hive_mc.mc_replay_buffer import MCTrainingExample
from hive_mc.mc_transformer import HiveMoveTransformer
from hive_mc.mc_utils import (
    build_successor_batch,
    flat_to_padded,
    parse_move_features,
)


@dataclass
class MCGumbelConfig:
    num_simulations: int = 128
    max_num_considered_actions: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    temperature: float = 1.0
    temperature_drop_move: int = 20
    batch_size: int = 128
    max_game_length: int = 300
    expansion_mask: int = 0
    nn_max_batch: int = 0


class MCGumbelOrchestrator:
    """Two-stage move-conditioned Gumbel self-play.

    Stage 1: encode root once, screen all legal moves with lightweight head.
    Stage 2: encode only top-k successor states, score with value head.
    Sequential halving uses real Q-values from the value head on successors.
    """

    _OFF_TURN = 3412

    def __init__(self, net: HiveMoveTransformer, config: MCGumbelConfig | None = None) -> None:
        self.ext = hive_gpu.load_extension()
        self.encoder = GPUTransformerEncoder()
        self.net = net
        self.config = config or MCGumbelConfig()
        self._move_size = self.ext.SIZEOF_GPU_MOVE

    def self_play_batch(
        self,
        start_states: torch.Tensor | None = None,
    ) -> list[list[MCTrainingExample]]:
        B = self.config.batch_size
        cfg = self.config
        states = (
            start_states.cuda()
            if start_states is not None
            else self.ext.create_initial_states(B, cfg.expansion_mask)
        )

        move_numbers = torch.zeros((B,), dtype=torch.int64, device=states.device)
        histories: list[list[tuple[np.ndarray, np.ndarray, float]]] = [[] for _ in range(B)]
        active_mask = torch.ones((B,), dtype=torch.bool, device=states.device)

        while bool(active_mask.any().item()):
            # ── Stage 1: encode root + screen all legal moves ──────────
            root_batch = self.encoder.encode_batch(states, B)
            legal_moves, num_legal = self.ext.generate_legal_moves_batch(states, B)
            num_actions = num_legal.to(torch.int64)
            move_feats = parse_move_features(legal_moves, num_legal, states.device)

            with torch.no_grad():
                _, root_cls = self.net.encode(root_batch)
                root_values = self.net.value_head(root_cls, root_batch.global_features)
                screening_logits = self.net.screen(
                    root_cls, root_batch.global_features, move_feats,
                )

            n_per_game = num_actions
            has_actions = active_mask & (n_per_game > 0)
            newly_finished = active_mask & (n_per_game == 0)

            screening_pad = flat_to_padded(
                screening_logits, num_actions, pad_value=float("-inf"),
            )
            max_count = screening_pad.shape[1]
            candidate_count = (
                min(cfg.max_num_considered_actions, max_count)
                if max_count > 0 else 0
            )
            full_policy_pad = torch.zeros_like(screening_pad)

            if candidate_count > 0 and has_actions.any():
                legal_cols = torch.arange(
                    max_count, device=states.device, dtype=torch.int64,
                ).unsqueeze(0)
                legal_mask = legal_cols < n_per_game.unsqueeze(1)

                # Gumbel top-k from screening scores
                noise = -torch.log(
                    -torch.log(
                        torch.rand_like(screening_pad, dtype=torch.float32)
                        .clamp(1e-4, 1.0 - 1e-4)
                    )
                )
                perturbed = torch.where(
                    legal_mask, screening_pad + noise, float("-inf"),
                )
                cand_scores, cand_idx = torch.topk(
                    perturbed, k=candidate_count, dim=1,
                )

                candidate_slots = torch.arange(
                    candidate_count, device=states.device, dtype=torch.int64,
                ).unsqueeze(0)
                candidate_valid = candidate_slots < n_per_game.unsqueeze(1).clamp(
                    max=candidate_count,
                )
                candidate_alive = candidate_valid & has_actions.unsqueeze(1)

                # ── Stage 2: encode only top-k successor states ────────
                flat_game = torch.arange(
                    B, device=states.device,
                ).unsqueeze(1).expand(B, candidate_count).reshape(-1)
                flat_move_col = cand_idx.reshape(-1)
                flat_valid = candidate_alive.reshape(-1)

                n_valid = int(flat_valid.sum().item())
                cand_values = torch.zeros(
                    (B, candidate_count), device=states.device,
                )

                if n_valid > 0:
                    valid_game = flat_game[flat_valid]
                    valid_move_col = flat_move_col[flat_valid]

                    with torch.no_grad():
                        action_batch = build_successor_batch(
                            states, legal_moves, valid_game, valid_move_col,
                            encoder=self.encoder, ext=self.ext,
                        )
                        _, action_cls = self.net.encode(action_batch)
                        q_vals = self.net.value_head(
                            action_cls, action_batch.global_features,
                        ).squeeze(-1)

                    # Scatter Q-values back to (B, candidate_count)
                    flat_indices = torch.arange(
                        B * candidate_count, device=states.device,
                    )
                    valid_flat_idx = flat_indices[flat_valid]
                    cand_values.view(-1)[valid_flat_idx] = q_vals

                # ── Sequential halving ─────────────────────────────────
                q_sums = torch.zeros(
                    (B, candidate_count), device=states.device,
                )
                visits = torch.zeros(
                    (B, candidate_count), dtype=torch.int32, device=states.device,
                )
                remaining = torch.full(
                    (B,),
                    int(cfg.num_simulations),
                    dtype=torch.int32,
                    device=states.device,
                )

                num_rounds = max(1, math.ceil(math.log2(max(candidate_count, 1))))
                for round_idx in range(num_rounds):
                    alive_counts = candidate_alive.sum(dim=1)
                    if alive_counts.max().item() == 0:
                        break

                    sims_each = torch.where(
                        alive_counts > 0,
                        torch.clamp(
                            remaining // alive_counts.clamp_min(1), min=1,
                        ),
                        torch.zeros_like(remaining),
                    )
                    q_sums += candidate_alive * (
                        (-cand_values) * sims_each.unsqueeze(1)
                    )
                    visits += candidate_alive * sims_each.unsqueeze(1)
                    remaining = torch.clamp(
                        remaining - sims_each * alive_counts, min=0,
                    )

                    if round_idx < num_rounds - 1:
                        q_mean = torch.where(
                            visits > 0,
                            q_sums / visits.float(),
                            torch.zeros_like(q_sums),
                        )
                        max_visit = visits.max(dim=1).values.float().clamp(min=1.0)
                        sigma = cand_scores + (
                            cfg.c_visit + max_visit.unsqueeze(1)
                        ) * cfg.c_scale * q_mean
                        sigma = sigma.masked_fill(~candidate_alive, float("-inf"))
                        keep_n = torch.clamp((alive_counts + 1) // 2, min=1)
                        rank = sigma.argsort(dim=1, descending=True).argsort(dim=1)
                        candidate_alive = candidate_alive & (
                            rank < keep_n.unsqueeze(1)
                        )

                # ── Build policy from visit counts ─────────────────────
                visit_probs = visits.float()
                visit_sums = visit_probs.sum(dim=1, keepdim=True)
                fallback = candidate_valid.float()
                fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
                visit_probs = torch.where(
                    visit_sums > 0,
                    visit_probs / visit_sums.clamp_min(1.0),
                    fallback,
                )
                full_policy_pad.scatter_(1, cand_idx, visit_probs)
                full_policy_pad = full_policy_pad * legal_mask

                # ── Record training examples ───────────────────────────
                finished_indices = torch.nonzero(
                    has_actions, as_tuple=False,
                ).squeeze(1)
                if finished_indices.numel() > 0:
                    state_bytes_cpu = states[finished_indices].detach().cpu().numpy()
                    root_values_cpu = root_values[finished_indices, 0].detach().cpu().numpy()
                    full_policy_cpu = full_policy_pad[finished_indices].detach().cpu().numpy()
                    num_actions_cpu = n_per_game[finished_indices].detach().cpu().numpy()
                    finished_list = finished_indices.tolist()
                    for row, game_idx in enumerate(finished_list):
                        n = int(num_actions_cpu[row])
                        histories[game_idx].append(
                            (
                                state_bytes_cpu[row].copy(),
                                full_policy_cpu[row, :n].astype(
                                    np.float32, copy=True,
                                ),
                                float(root_values_cpu[row]),
                            )
                        )

            if newly_finished.any():
                active_mask = active_mask & ~newly_finished

            # ── Select and apply move ──────────────────────────────────
            chosen = torch.zeros((B,), dtype=torch.int64, device=states.device)
            if has_actions.any():
                finished_indices = torch.nonzero(
                    has_actions, as_tuple=False,
                ).squeeze(1)
                greedy_rows = finished_indices[
                    move_numbers[finished_indices] >= cfg.temperature_drop_move
                ]
                sample_rows = finished_indices[
                    move_numbers[finished_indices] < cfg.temperature_drop_move
                ]

                if greedy_rows.numel() > 0:
                    chosen[greedy_rows] = full_policy_pad[greedy_rows].argmax(dim=1)
                if sample_rows.numel() > 0:
                    sample_policy = full_policy_pad[sample_rows]
                    sample_policy = sample_policy / sample_policy.sum(
                        dim=1, keepdim=True,
                    ).clamp_min(1e-8)
                    chosen[sample_rows] = torch.multinomial(
                        sample_policy, 1,
                    ).squeeze(1)

                move_bytes = torch.zeros(
                    (B, self._move_size), dtype=torch.uint8, device=states.device,
                )
                move_bytes[finished_indices] = legal_moves[
                    finished_indices, chosen[finished_indices]
                ]
                self.ext.apply_moves_batch(states, move_bytes, B)

            move_numbers = move_numbers + has_actions.to(torch.int64)

            results = self.ext.check_results_batch(states, B)
            active_mask = active_mask & (results == 0) & (
                move_numbers < cfg.max_game_length
            )

        # ── Build final training examples ──────────────────────────────
        final_results = self.ext.check_results_batch(states, B).cpu().numpy()
        examples: list[list[MCTrainingExample]] = [[] for _ in range(B)]
        for i in range(B):
            winner = final_results[i]
            for _ply, (state_bytes, target_pi, _root_v) in enumerate(histories[i]):
                turn = int(state_bytes[self._OFF_TURN]) | (
                    int(state_bytes[self._OFF_TURN + 1]) << 8
                )
                value = self._result_to_value(winner, turn)
                examples[i].append(
                    MCTrainingExample(
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
