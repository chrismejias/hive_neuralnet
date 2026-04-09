"""
Gumbel AlphaZero self-play orchestrator for the PRS model.

Action space: 6,841 piece-relative indices (instead of engine's ~85K).
Legal masks are built by converting GPUMove bytes → PRS indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import hive_gpu
from hive_prs.action_space import (
    ACTION_SPACE_SIZE, moves_to_action_indices, batch_moves_to_action_indices,
)
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.prs_replay_buffer import PRSTrainingExample

# Byte offset of the turn counter in the raw HiveState tensor.
_OFF_TURN = 3412


@dataclass
class PRSGumbelConfig:
    num_simulations:            int   = 128
    max_num_considered_actions: int   = 16
    c_visit:                    float = 50.0
    c_scale:                    float = 1.0
    temperature:                float = 1.0
    temperature_drop_move:      int   = 20
    batch_size:                 int   = 128
    max_game_length:            int   = 300
    expansion_mask:             int   = 0
    nn_max_batch:               int   = 0


class PRSGumbelOrchestrator:
    """Gumbel AlphaZero self-play using the PRS action space."""

    def __init__(self, net: torch.nn.Module, config: PRSGumbelConfig | None = None) -> None:
        self.ext     = hive_gpu.load_extension()
        self.config  = config or PRSGumbelConfig()
        self.net     = net
        self.encoder = PRSEncoder()
        self._move_size = self.ext.SIZEOF_GPU_MOVE

    # ── Public ──────────────────────────────────────────────────────────

    def self_play_batch(
        self,
        start_states: torch.Tensor | None = None,
    ) -> list[list[PRSTrainingExample]]:
        B   = self.config.batch_size
        cfg = self.config

        states = (
            start_states.cuda()
            if start_states is not None
            else self.ext.create_initial_states(B, cfg.expansion_mask)
        )

        active       = [True] * B
        move_numbers = [0]    * B
        histories: list[list[dict]] = [[] for _ in range(B)]

        while any(active):
            # ── 1. Encode ──
            prs_batch = self.encoder.encode_batch(states, B)
            occ_cpu   = prs_batch.occupied_cells.cpu().numpy()   # (B, MAX_BOARD)
            nocc_cpu  = prs_batch.num_occupied.cpu().numpy()     # (B,)

            # ── 2. Legal moves ──
            legal_t, nlegal_t = self.ext.generate_legal_moves_batch(states, B)
            legal_np  = legal_t.cpu().numpy()       # (B, MAX_L, move_size)
            nlegal_np = nlegal_t.cpu().numpy()      # (B,)

            # ── 3. PRS legal mask + inverse lookup (vectorised) ──
            prs_from_move: list[np.ndarray]     = batch_moves_to_action_indices(
                legal_np, nlegal_np, occ_cpu, nocc_cpu)
            move_from_prs: list[dict[int, int]] = []
            prs_mask = torch.zeros(B, ACTION_SPACE_SIZE, dtype=torch.bool, device="cuda")

            for i in range(B):
                idx = prs_from_move[i]
                valid_mask = (idx >= 0) & (idx < ACTION_SPACE_SIZE)
                valid_prs  = idx[valid_mask]
                valid_j    = np.where(valid_mask)[0]
                # keep first occurrence (lowest j) for duplicate PRS indices
                _, first   = np.unique(valid_prs, return_index=True)
                inv        = dict(zip(valid_prs[first].tolist(), valid_j[first].tolist()))
                move_from_prs.append(inv)
                if len(valid_prs):
                    prs_mask[i, valid_prs] = True

            # ── 4. Root NN evaluation ──
            with torch.no_grad():
                policy_logits, root_values = self._net_forward(prs_batch, B)
            root_values = root_values.squeeze(-1)   # (B,)
            policy_logits = policy_logits.float()
            policy_logits[~prs_mask] = float("-inf")

            nn_prior_np = torch.softmax(policy_logits, dim=-1).cpu().numpy()  # (B, A)

            # ── 5. Gumbel + top-k (sparse: work only on legal actions) ──
            num_legal_prs = prs_mask.sum(dim=1)   # (B,)
            k      = cfg.max_num_considered_actions
            eff_k  = num_legal_prs.clamp(max=k)
            max_k  = min(k, int(num_legal_prs.max().item()))

            if max_k == 0:
                for i in range(B):
                    active[i] = False
                break

            # Sparse gumbel: gather legal logits into (B, max_legal) tensor,
            # do topk there, then scatter selected indices back to global PRS space.
            max_legal = int(num_legal_prs.max().item())

            # Build padded (B, max_legal) tensors of legal PRS indices and logits
            legal_idx_padded  = torch.zeros(B, max_legal, dtype=torch.int64, device="cuda")
            legal_log_padded  = torch.full((B, max_legal), float("-inf"), device="cuda")
            for i in range(B):
                nl = int(num_legal_prs[i].item())
                li = prs_mask[i].nonzero(as_tuple=False).squeeze(1)[:nl]  # (nl,)
                legal_idx_padded[i, :nl] = li
                legal_log_padded[i, :nl] = policy_logits[i, li]

            u       = torch.rand(B, max_legal, device="cuda").clamp(1e-10, 1 - 1e-7)
            gumbel  = -torch.log(-torch.log(u))
            gumbel[legal_log_padded == float("-inf")] = float("-inf")
            perturbed_sparse = gumbel + legal_log_padded    # (B, max_legal)

            _, topk_local = torch.topk(perturbed_sparse, max_k, dim=1)  # (B, max_k)
            topk_actions  = legal_idx_padded.gather(1, topk_local)       # (B, max_k) global PRS idx
            # perturbed at topk positions (for score computation later)
            perturbed_topk = perturbed_sparse.gather(1, topk_local)      # (B, max_k)

            # ── 6. Sequential halving ──
            q_sums        = torch.zeros(B, max_k, device="cuda")
            visit_counts  = torch.zeros(B, max_k, dtype=torch.int32, device="cuda")
            cand_mask     = torch.ones(B, max_k, dtype=torch.bool, device="cuda")

            for i in range(B):
                ek = int(eff_k[i].item())
                if ek < max_k:
                    cand_mask[i, ek:] = False

            # Sequential halving: evaluate each surviving candidate once per round,
            # using ONE batched NN call (B × num_surviving states) per round.
            # Each round halves the candidate set by score.
            # Total NN calls = n_rounds (4 for k=16), not num_simulations.
            n_rounds = max(1, math.ceil(math.log2(max_k + 1e-9)))
            evals_per_round = max(1, cfg.num_simulations // max(n_rounds * max_k, 1))

            for _ in range(n_rounds):
                for _rep in range(evals_per_round):
                    q_sums, visit_counts = self._halving_round_batched(
                        states, topk_actions, cand_mask, q_sums, visit_counts,
                        legal_np, nlegal_np, occ_cpu, nocc_cpu, move_from_prs,
                    )

                # Halve: keep top half by score
                num_cands = int(cand_mask.sum(dim=1).max().item())
                if num_cands <= 1:
                    break
                num_keep = max(1, num_cands // 2)

                max_n = int(visit_counts.max().item())
                sigma = (cfg.c_visit + max_n) * cfg.c_scale
                q_mean = q_sums / visit_counts.clamp(min=1).float()
                scores = perturbed_topk + q_mean / (sigma + 1e-8)
                scores[~cand_mask] = float("-inf")

                _, keep = torch.topk(scores, num_keep, dim=1)
                new_mask = torch.zeros_like(cand_mask)
                new_mask.scatter_(1, keep, True)
                cand_mask = new_mask

            # ── 7. Select action ──
            max_n  = int(visit_counts.max().item())
            sigma  = (cfg.c_visit + max_n) * cfg.c_scale
            q_mean = q_sums / visit_counts.clamp(min=1).float()
            final_scores = perturbed_topk + q_mean / (sigma + 1e-8)
            final_scores[~cand_mask] = float("-inf")

            sel_local  = torch.argmax(final_scores, dim=1)                           # (B,)
            sel_global = topk_actions.gather(1, sel_local.unsqueeze(1)).squeeze(1)   # (B,) PRS idx
            sel_np     = sel_global.cpu().numpy()

            # Policy target from visit counts
            visits_float = visit_counts.float()
            topk_np      = topk_actions.cpu().numpy()
            visits_np    = visits_float.cpu().numpy()

            # ── 8. Record history ──
            turns_lo = states[:B, _OFF_TURN].to(torch.int32)
            turns_hi = states[:B, _OFF_TURN + 1].to(torch.int32)
            turns    = (turns_lo | (turns_hi << 8)).cpu().tolist()

            for i in range(B):
                if not active[i]:
                    continue
                vs = visits_np[i]
                vsum = vs.sum()
                if vsum > 0:
                    probs = vs / vsum
                else:
                    probs = np.ones(max_k, dtype=np.float32) / max_k

                histories[i].append({
                    "token_features":   prs_batch.token_features[i].cpu().numpy().copy(),
                    "token_positions":  prs_batch.token_positions[i].cpu().numpy().astype(np.int32),
                    "token_types":      prs_batch.token_types[i].cpu().numpy().astype(np.int8),
                    "num_board_tokens": int(prs_batch.num_board_tokens[i].item()),
                    "global_features":  prs_batch.global_features[i].cpu().numpy().copy(),
                    "seq_length":       int(prs_batch.seq_lengths[i].item()),
                    "occupied_cells":   occ_cpu[i].copy(),
                    "num_occupied":     int(nocc_cpu[i]),
                    "topk_actions":     topk_np[i].copy(),    # (max_k,) PRS indices
                    "probs":            probs.copy(),          # (max_k,) float32
                    "nn_prior":         nn_prior_np[i].copy(), # (A,) float32
                    "turn":             int(turns[i]),
                    "prs_from_move":    prs_from_move[i].copy(),  # (n_legal,) int32
                })

            # ── 9. Apply selected moves ──
            move_bytes_np = np.zeros((B, self._move_size), dtype=np.uint8)
            for i in range(B):
                if not active[i]:
                    continue
                sel_prs = int(sel_np[i])
                j = move_from_prs[i].get(sel_prs, -1)
                if j >= 0:
                    move_bytes_np[i] = legal_np[i, j]
                elif int(nlegal_np[i]) > 0:
                    move_bytes_np[i] = legal_np[i, 0]   # fallback

            move_bytes_t = torch.from_numpy(move_bytes_np).cuda()
            self.ext.apply_moves_batch(states, move_bytes_t, B)

            for i in range(B):
                if active[i]:
                    move_numbers[i] += 1

            results = self.ext.check_results_batch(states, B).cpu().numpy()
            for i in range(B):
                if not active[i]:
                    continue
                if results[i] != 0 or move_numbers[i] >= cfg.max_game_length:
                    active[i] = False

        final_results = self.ext.check_results_batch(states, B).cpu().numpy()
        return self._build_examples(histories, final_results)

    # ── Batched halving round ────────────────────────────────────────────
    #
    # Evaluates ALL surviving (game, candidate) pairs in ONE NN call.
    # For round 1 with B=128 games and k=16 candidates: batch = 128×16 = 2048 states.
    # This replaces the old sequential loop of num_evals × B separate calls.

    def _halving_round_batched(
        self,
        root_states:   torch.Tensor,         # (B, state_size)
        topk_actions:  torch.Tensor,         # (B, max_k) PRS indices
        cand_mask:     torch.Tensor,         # (B, max_k) bool
        q_sums:        torch.Tensor,
        visit_counts:  torch.Tensor,
        legal_np:      np.ndarray,           # (B, MAX_L, move_size) from root
        nlegal_np:     np.ndarray,           # (B,) int32
        occ_cpu:       np.ndarray,           # (B, MAX_BOARD) int32
        nocc_cpu:      np.ndarray,           # (B,) int32
        move_from_prs: list[dict[int, int]], # B-length list of PRS→move_j dicts
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate every surviving candidate for every game in one NN batch."""

        # Collect surviving (game_i, local_k_j) pairs
        game_idx, k_idx = torch.where(cand_mask)   # both (N,) int64  (on CUDA)
        N = game_idx.shape[0]
        if N == 0:
            return q_sums, visit_counts

        game_idx_np = game_idx.cpu().numpy()
        cand_prs_np = topk_actions[game_idx, k_idx].cpu().numpy()   # (N,) PRS indices

        # Build N child states: root_states[game_idx[n]] + apply cand action
        # Keep indexing on GPU — avoids D2H + H2D roundtrip
        child_states = root_states[game_idx].clone()    # (N, state_size) on CUDA

        move_bytes_np = np.zeros((N, self._move_size), dtype=np.uint8)
        for n in range(N):
            g   = int(game_idx_np[n])
            sel = int(cand_prs_np[n])
            j   = move_from_prs[g].get(sel, -1)
            if j >= 0:
                move_bytes_np[n] = legal_np[g, j]
            elif int(nlegal_np[g]) > 0:
                move_bytes_np[n] = legal_np[g, 0]

        move_bytes_t = torch.from_numpy(move_bytes_np).cuda()
        self.ext.apply_moves_batch(child_states, move_bytes_t, N)

        # ONE encode + ONE NN forward for all N child states
        child_prs = self.encoder.encode_batch(child_states, N)
        with torch.no_grad():
            _, child_values = self.net(child_prs)
        child_q = -child_values.squeeze(-1).float()   # (N,) — negated for root perspective

        # Scatter back
        q_sums.index_put_((game_idx, k_idx), child_q, accumulate=True)
        ones = torch.ones(N, dtype=torch.int32, device="cuda")
        visit_counts.index_put_((game_idx, k_idx), ones, accumulate=True)

        return q_sums, visit_counts

    # ── NN forward (with optional sub-batching) ───────────────────────

    def _net_forward(
        self, prs_batch: "PRSTokenBatch", B: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mb = self.config.nn_max_batch
        if mb <= 0 or B <= mb:
            with torch.no_grad():
                return self.net(prs_batch)

        all_logits, all_values = [], []
        for s in range(0, B, mb):
            e = min(s + mb, B)
            with torch.no_grad():
                lo, v = self.net(prs_batch.slice_batch(s, e))
            all_logits.append(lo)
            all_values.append(v)
        return torch.cat(all_logits, 0), torch.cat(all_values, 0)

    # ── Build training examples ────────────────────────────────────────

    def _build_examples(
        self,
        histories:     list[list[dict]],
        final_results: np.ndarray,
    ) -> list[list[PRSTrainingExample]]:
        all_examples = []

        for i, history in enumerate(histories):
            result = int(final_results[i])
            game_exs = []

            for step in history:
                turn = step["turn"]
                player_is_white = (turn % 2 == 0)

                if result == 0 or result == 3:
                    value = 0.0
                elif result == 1:
                    value = 1.0 if player_is_white else -1.0
                else:
                    value = -1.0 if player_is_white else 1.0

                # Dense policy from sparse (topk_actions, probs)
                topk = step["topk_actions"]
                prob = step["probs"]
                policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                for ai, p in zip(topk, prob):
                    ai = int(ai)
                    if 0 <= ai < ACTION_SPACE_SIZE:
                        policy[ai] = float(p)

                # Mark all legal moves with epsilon floor so loss masks correctly
                prs_lm = step["prs_from_move"]  # (n_legal,) int32
                valid_legal = np.array(
                    [p for p in prs_lm if 0 <= p < ACTION_SPACE_SIZE], dtype=np.int32
                )
                if len(valid_legal) > 0:
                    policy[valid_legal] = np.maximum(policy[valid_legal], 1e-4)
                psum = policy.sum()
                if psum > 0:
                    policy /= psum

                # Surprise weight
                nn_prior = step["nn_prior"]
                mask = policy > 0
                sw = 1.0
                if mask.any():
                    p_pos = policy[mask]
                    q_pos = np.clip(nn_prior[mask], 1e-8, 1.0)
                    kl = float(np.sum(p_pos * np.log(p_pos / q_pos)))
                    sw = max(0.1, 1.0 + kl)

                game_exs.append(PRSTrainingExample(
                    token_features   = step["token_features"],
                    token_positions  = step["token_positions"],
                    token_types      = step["token_types"],
                    num_board_tokens = step["num_board_tokens"],
                    global_features  = step["global_features"],
                    seq_length       = step["seq_length"],
                    occupied_cells   = step["occupied_cells"],
                    num_occupied     = step["num_occupied"],
                    policy_target    = policy,
                    value_target     = value,
                    surprise_weight  = sw,
                ))

            all_examples.append(game_exs)

        return all_examples
