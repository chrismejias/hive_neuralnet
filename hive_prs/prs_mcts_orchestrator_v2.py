"""Gumbel AlphaZero + MCTS self-play orchestrator for PRS v2 (813-slot head).

Differences vs. `PRSMCTSOrchestrator` (v1):
  * Priors come from the 813-slot structured head, not the 6,841-dim bilinear.
  * Each legal move's prior = softmax_slot[slot_of_legal[k]] / count_shared,
    where `count_shared` is the number of legal moves sharing that slot.
    This keeps Σ priors = 1 and treats engine-emitted byte-duplicate throws
    (e.g. P + M-as-P) as sharing probability mass.
  * Root action selection uses the final Gumbel sigma score among surviving
    candidates; training targets still aggregate visits in slot space.
  * Training examples record slot-space visit targets + the legal mask,
    and keep raw HiveState bytes so the trainer can rebuild head inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import hive_gpu
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.prs_replay_buffer_v2 import PRSTrainingExampleV2
from hive_prs.prs_v2_bridge import (
    build_head_inputs_from_states,
    build_head_inputs_from_kernel,
)
from hive_prs.slot_map import N_SLOTS, PASS_SLOT, map_legal_moves

_OFF_TURN = 3412  # byte offset of turn counter in HiveState
_GUMBEL_WAVE_SCHEDULE = (1, 2, 4, 8)


@dataclass
class PRSMCTSConfigV2:
    num_simulations:             int   = 128
    max_num_considered_actions:  int   = 16
    c_puct:                      float = 1.25
    c_visit:                     float = 50.0
    c_scale:                     float = 1.0
    temperature:                 float = 1.0
    temperature_drop_move:       int   = 20
    batch_size:                  int   = 128
    max_game_length:             int   = 300
    expansion_mask:              int   = 7
    nn_max_batch:                int   = 0
    wave_parallel:               bool  = True
    wave_size:                   int   = 16
    dirichlet_alpha:             float = 0.3
    dirichlet_epsilon:           float = 0.25
    max_tree_nodes:              int   = 65536
    # Rebase each game's tree to a fresh root after applying a move.
    # This prevents monotonic node-id growth across plies from exhausting the
    # fixed node pool in long games.
    rebase_tree_each_move:       bool  = True
    # Diagnostics for root-visit starvation. If enabled, logs when the root
    # has zero legal visits after all simulation rounds.
    debug_zero_visit_logging:    bool  = False
    debug_zero_visit_limit:      int   = 20
    # Additional diagnostics: trace root expansion success/failure and first
    # simulation-wave selection behavior.
    debug_expand_logging:        bool  = False
    debug_expand_limit:          int   = 20
    debug_select_wave_logging:   bool  = False
    debug_select_wave_limit:     int   = 4


class PRSMCTSOrchestratorV2:
    """Gumbel AlphaZero with tree-search MCTS, PRS v2 (813-slot) head."""

    def __init__(
        self,
        net: torch.nn.Module,           # HivePRSTransformerV2
        config: PRSMCTSConfigV2 | None = None,
    ) -> None:
        self.ext        = hive_gpu.load_extension()
        self.config     = config or PRSMCTSConfigV2()
        self.net        = net
        self.encoder    = PRSEncoder()
        self._move_size = self.ext.SIZEOF_GPU_MOVE
        self._max_nodes = int(self.config.max_tree_nodes)
        self._max_depth = int(self.ext.MAX_TREE_DEPTH)
        self._max_legal = int(self.ext.MAX_LEGAL_MOVES)
        self._dbg_expand_logs_emitted = 0
        self._dbg_select_logs_emitted = 0

    # ── Tree tensor management (SoA) ──────────────────────────────────

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

    # ── Slot-mapping + legal-prior construction ──────────────────────

    def _classify_kernel(
        self,
        states:   torch.Tensor,       # (B, SIZEOF_HIVE_STATE) uint8 on CUDA
        legal_t:  torch.Tensor,       # (B, MAX_L, 6) uint8 on CUDA
        nlegal_t: torch.Tensor,       # (B,) int32 on CUDA
        B: int,
    ) -> tuple:
        """Run the CUDA prs_v2 classify kernel. Returns the 9-tuple of GPU
        tensors: (dir_piece_idx, throw_piece_idx, long_piece_idx,
        move_nbrs, place_nbrs, move_mask, place_mask, current_color,
        slot_of_legal)."""
        return self.ext.prs_v2_classify_batch(
            states, legal_t, nlegal_t, B, self._max_legal,
        )

    def _build_legal_priors_v2(
        self,
        policy_logits_813: torch.Tensor,   # (B, 813)
        slot_of_legal_t:   torch.Tensor,   # (B, MAX_L) int32 on CUDA, -1 = invalid
        B: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (priors_per_legal [B,MAX_L], logit_per_legal [B,MAX_L]).

        Per-state procedure:
          * legal_mask[s] = True for any s that appears in slot_of_legal
          * softmax over legal slots only
          * prior_per_legal[k] = softmax[slot_of_legal[k]] / count_shared_slot
          * logit_per_legal[k] = raw logit at slot_of_legal[k] (for Gumbel)
        Padding slots get 0 prior and -inf logit.
        """
        device = policy_logits_813.device
        slot_t = slot_of_legal_t.to(device=device, dtype=torch.long)      # (B, MAX_L)
        valid  = slot_t >= 0                                              # (B, MAX_L)

        # Build legal_mask (B, 813)
        legal_mask = torch.zeros(B, N_SLOTS, dtype=torch.bool, device=device)
        safe = slot_t.clamp(min=0)
        legal_mask.scatter_(1, safe, valid)

        # Softmax over legal slots only
        masked = policy_logits_813.masked_fill(~legal_mask, float("-inf"))
        # Guard empty-legal rows: softmax of all -inf yields NaN — replace with 0
        any_legal = legal_mask.any(dim=1, keepdim=True)
        masked = torch.where(any_legal, masked, torch.zeros_like(masked))
        probs  = torch.softmax(masked, dim=-1)
        probs  = probs * any_legal.float()

        # Count shared slots per legal index so we can divide shared mass
        counts = torch.zeros(B, N_SLOTS, dtype=torch.float32, device=device)
        counts.scatter_add_(1, safe, valid.float())
        counts_per_legal = counts.gather(1, safe).clamp(min=1.0)          # (B, MAX_L)

        prob_per_slot = probs.gather(1, safe)                             # (B, MAX_L)
        priors_per_legal = torch.where(
            valid, prob_per_slot / counts_per_legal, torch.zeros_like(prob_per_slot),
        )

        logit_per_slot = policy_logits_813.gather(1, safe)
        logit_per_legal = torch.where(
            valid, logit_per_slot, torch.full_like(logit_per_slot, -1e30),
        )

        return priors_per_legal.contiguous(), logit_per_legal.contiguous()

    # ── Net forward: trunk + v2 head ─────────────────────────────────

    def _net_forward(
        self,
        prs_batch,
        kernel_out: tuple,           # 15-tuple from prs_v2_classify_batch
        B: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits_813 [B, 813], value [B, 1]).

        Uses precomputed CUDA kernel outputs for head inputs + slot mapping.
        """
        mb = self.config.nn_max_batch
        if mb <= 0 or B <= mb:
            with torch.no_grad():
                board_h, cls_h, full_h, value = self.net.forward_trunk(prs_batch)
                inp, _ = build_head_inputs_from_kernel(
                    board_h, cls_h, full_h, kernel_out,
                )
                logits = self.net.head(inp)
            return logits, value
        # Sub-batching: slice each kernel-output tensor along dim 0.
        all_l, all_v = [], []
        for s in range(0, B, mb):
            e = min(s + mb, B)
            sub_kernel = tuple(t[s:e] for t in kernel_out)
            with torch.no_grad():
                sub = prs_batch.slice_batch(s, e)
                board_h, cls_h, full_h, value = self.net.forward_trunk(sub)
                inp, _ = build_head_inputs_from_kernel(
                    board_h, cls_h, full_h, sub_kernel,
                )
                lo = self.net.head(inp)
            all_l.append(lo); all_v.append(value)
        return torch.cat(all_l, 0), torch.cat(all_v, 0)

    # ── Public API ───────────────────────────────────────────────────

    def self_play_batch(
        self,
        start_states: torch.Tensor | None = None,
    ) -> list[list[PRSTrainingExampleV2]]:
        B   = self.config.batch_size
        cfg = self.config

        states = (
            start_states.cuda()
            if start_states is not None
            else self.ext.create_initial_states(B, cfg.expansion_mask)
        )

        tree = self._alloc_tree(B)
        self._reset_tree(tree)

        active       = [True] * B
        move_numbers = [0]    * B
        histories: list[list[dict]] = [[] for _ in range(B)]
        zero_visit_logs_emitted = 0
        self._dbg_expand_logs_emitted = 0
        self._dbg_select_logs_emitted = 0

        while any(active):
            # ── Encode root + legal moves ─────────────────────────────
            prs_batch = self.encoder.encode_batch(states, B)
            occ_cpu   = prs_batch.occupied_cells.cpu().numpy()
            nocc_cpu  = prs_batch.num_occupied.cpu().numpy()

            legal_t, nlegal_t = self.ext.generate_legal_moves_batch(states, B)
            nlegal_np = nlegal_t.cpu().numpy()

            # ── Immediate-win detection ────────────────────────────────
            # Try every legal move; if any wins for the current player,
            # record it now so we can override action + one-hot training
            # target after MCTS completes.
            turns_np_early = self._get_turns(states, B)
            win_overrides = self._check_immediate_wins_v2(
                states, legal_t, nlegal_np, active, turns_np_early, B,
            )

            # ── CUDA slot-classify + head-input bridge (one kernel call) ──
            kernel_out = self._classify_kernel(states, legal_t, nlegal_t, B)
            slot_of_legal_t = kernel_out[8]                  # (B, MAX_L) int32

            # ── Root NN forward (trunk + v2 head) ─────────────────────
            policy_logits_813, root_values = self._net_forward(
                prs_batch, kernel_out, B,
            )
            policy_logits_813 = policy_logits_813.float()

            priors_per_legal, root_logit_per_legal = self._build_legal_priors_v2(
                policy_logits_813, slot_of_legal_t, B,
            )

            # ── Gumbel noise + top-k considered candidates ────────────
            max_k = min(cfg.max_num_considered_actions, int(nlegal_np.max())) \
                if int(nlegal_np.max()) > 0 else 1
            if max_k == 0:
                for i in range(B):
                    active[i] = False
                break

            u = torch.rand(B, self._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
            gumbel = -torch.log(-torch.log(u))
            nlegal_t_gpu = nlegal_t.to(torch.int64)
            slot_idx_t = torch.arange(self._max_legal, device="cuda").unsqueeze(0)
            valid_slot = slot_idx_t < nlegal_t_gpu.unsqueeze(1)
            gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
            perturbed = gumbel + root_logit_per_legal
            topk_scores, topk_slots = torch.topk(perturbed, max_k, dim=1)

            active_t = torch.tensor(
                [1 if a else 0 for a in active], dtype=torch.int8, device="cuda",
            )
            self._expand_root_if_needed(
                tree, states, legal_t, nlegal_t,
                priors_per_legal, active_t, B,
            )
            self._apply_root_dirichlet(tree, B, active)

            alive_mask = torch.zeros(B, self._max_legal, dtype=torch.int8, device="cuda")
            alive_mask.scatter_(1, topk_slots, 1)

            # ── Sequential Halving loop ──────────────────────────────
            n_rounds = max(1, math.ceil(math.log2(max(max_k, 2))))
            sims_per_round = max(1, cfg.num_simulations // n_rounds)

            for round_i in range(n_rounds):
                round_wave_size = (
                    _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)]
                    if cfg.wave_parallel else 1
                )
                self._run_simulations(
                    tree, states, active_t, alive_mask, B, sims_per_round,
                    wave_size=round_wave_size,
                )
                alive_counts = alive_mask.sum(dim=1)
                cur_alive_max = int(alive_counts.max().item())
                if cur_alive_max <= 1:
                    continue
                num_keep = max(1, cur_alive_max // 2)

                slot_visits, slot_q = self._gather_root_child_stats(tree, B)
                max_n = int(slot_visits.max().item())
                sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
                sigma_score = (gumbel + root_logit_per_legal
                               + sigma_norm * slot_q).float()
                sigma_score = torch.where(
                    alive_mask.bool(), sigma_score,
                    torch.full_like(sigma_score, -1e30),
                )
                _, keep_slots = torch.topk(sigma_score, num_keep, dim=1)
                new_alive = torch.zeros_like(alive_mask)
                new_alive.scatter_(1, keep_slots, 1)
                new_alive = new_alive * valid_slot.to(torch.int8)
                alive_mask = new_alive

            # ── Final action selection (Gumbel sigma score) ──────────
            leg_visits, leg_q = self._gather_root_child_stats(tree, B)
            # leg_visits: (B, MAX_L) int32 — visits per legal-move index.
            if cfg.debug_zero_visit_logging and zero_visit_logs_emitted < cfg.debug_zero_visit_limit:
                leg_visit_sums = leg_visits.sum(dim=1).cpu().numpy()
                row = torch.arange(B, device="cuda")
                roots = tree["root_node"]
                root_nc = tree["num_children"][row, roots].cpu().numpy()
                alive_counts = alive_mask.sum(dim=1).cpu().numpy()
                for i in range(B):
                    if zero_visit_logs_emitted >= cfg.debug_zero_visit_limit:
                        break
                    if not active[i]:
                        continue
                    n_i = int(nlegal_np[i])
                    if n_i <= 0:
                        continue
                    if int(leg_visit_sums[i]) != 0:
                        continue
                    root_fc = int(tree["first_child"][i, roots[i]].item())
                    print(
                        "[PRS_V2_ZERO_ROOT_VISITS] "
                        f"game={i} ply={move_numbers[i]} nlegal={n_i} "
                        f"root_fc={root_fc} root_children={int(root_nc[i])} "
                        f"alive_candidates={int(alive_counts[i])} "
                        f"n_rounds={n_rounds} sims_per_round={sims_per_round} "
                        f"num_simulations={cfg.num_simulations}",
                    )
                    zero_visit_logs_emitted += 1

            slot_t = slot_of_legal_t.to(dtype=torch.long)                 # (B, MAX_L)
            valid_legal = slot_t >= 0

            max_n = int(leg_visits.max().item())
            sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
            final_sigma = (gumbel + root_logit_per_legal + sigma_norm * leg_q).float()
            final_sigma = torch.where(
                alive_mask.bool() & valid_legal,
                final_sigma,
                torch.full_like(final_sigma, -1e30),
            )
            chosen_slot_idx = torch.argmax(final_sigma, dim=1)            # (B,)

            # ── Record history (single big CPU transfer batch) ──────
            slot_of_legal_np = slot_of_legal_t.cpu().numpy().astype(np.int64)
            states_cpu       = states.cpu().numpy()
            legal_np         = legal_t.cpu().numpy()
            leg_visits_np    = leg_visits.cpu().numpy()
            chosen_np        = chosen_slot_idx.cpu().numpy().copy()
            turns_np         = turns_np_early  # already computed above

            # Apply win overrides to action selection
            for _i, _k in win_overrides.items():
                chosen_np[_i] = _k

            tf_cpu = prs_batch.token_features.cpu().numpy()
            tp_cpu = prs_batch.token_positions.cpu().numpy()
            tt_cpu = prs_batch.token_types.cpu().numpy()
            nb_cpu = prs_batch.num_board_tokens.cpu().numpy()
            gf_cpu = prs_batch.global_features.cpu().numpy()
            sl_cpu = prs_batch.seq_lengths.cpu().numpy()

            for i in range(B):
                if not active[i]:
                    continue
                n_i = int(nlegal_np[i])
                if n_i == 0:
                    continue
                # Slot-space visit target (aggregates duplicates)
                slot_target = np.zeros(N_SLOTS, dtype=np.float32)
                legal_mask_np = np.zeros(N_SLOTS, dtype=bool)
                for k in range(n_i):
                    s = int(slot_of_legal_np[i, k])
                    if s < 0:
                        continue
                    slot_target[s] += float(leg_visits_np[i, k])
                    legal_mask_np[s] = True
                ssum = slot_target.sum()
                if ssum > 0:
                    slot_target /= ssum
                else:
                    # Uniform over legal slots as a safety fallback
                    nleg = int(legal_mask_np.sum())
                    if nleg > 0:
                        slot_target[legal_mask_np] = 1.0 / nleg

                # Immediate-win override: replace visit-count target with
                # a one-hot pointing at the winning move's slot.
                if i in win_overrides:
                    win_k = win_overrides[i]
                    win_slot = int(slot_of_legal_np[i, win_k])
                    if win_slot >= 0:
                        slot_target[:] = 0.0
                        slot_target[win_slot] = 1.0

                # Value target filled in at _build_examples from game result
                S = int(sl_cpu[i])
                # Store raw moves + visit counts (truncated to nlegal) so the
                # trainer can C6-rotate at augmentation time.
                raw_moves = legal_np[i, :n_i].copy()                  # (n_i, move_sz) uint8
                raw_visits = leg_visits_np[i, :n_i].astype(np.float32)
                histories[i].append({
                    "token_features":   tf_cpu[i, :S].copy(),
                    "token_positions":  tp_cpu[i, :S].astype(np.int32),
                    "token_types":      tt_cpu[i, :S].astype(np.int8),
                    "num_board_tokens": int(nb_cpu[i]),
                    "global_features":  gf_cpu[i].copy(),
                    "seq_length":       S,
                    "occupied_cells":   occ_cpu[i].copy(),
                    "num_occupied":     int(nocc_cpu[i]),
                    "state_bytes":      states_cpu[i].copy(),
                    "legal_moves":      raw_moves,
                    "visit_counts":     raw_visits,
                    "nlegal":           n_i,
                    "slot_target":      slot_target,
                    "legal_mask":       legal_mask_np,
                    "turn":             int(turns_np[i]),
                })

            # ── Apply chosen moves + re-root tree ────────────────────
            move_bytes_np = np.zeros((B, self._move_size), dtype=np.uint8)
            chosen_child_nodes = [-1] * B
            for i in range(B):
                if not active[i]:
                    continue
                k = int(chosen_np[i])
                n_i = int(nlegal_np[i])
                if n_i == 0:
                    continue
                if k >= n_i:
                    k = 0
                move_bytes_np[i] = legal_np[i, k]
                old_root = int(tree["root_node"][i].item())
                fc = int(tree["first_child"][i, old_root].item())
                if fc >= 0:
                    chosen_child_nodes[i] = fc + k

            move_bytes_t = torch.from_numpy(move_bytes_np).cuda()
            self.ext.apply_moves_batch(states, move_bytes_t, B)
            self._reroot_tree(tree, B, active, chosen_child_nodes)

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

    # ── Tree expansion / simulations (same as v1) ────────────────────

    def _expand_root_if_needed(
        self,
        tree, states, legal_moves, num_legal,
        priors_per_legal, game_active, B: int,
    ) -> None:
        row = torch.arange(B, device="cuda")
        root_node = tree["root_node"]
        fc = tree["first_child"][row, root_node]
        nc = tree["num_children"][row, root_node]
        cfg = self.config
        if cfg.debug_expand_logging and self._dbg_expand_logs_emitted < cfg.debug_expand_limit:
            fc_cpu = fc.cpu().numpy()
            nc_cpu = nc.cpu().numpy()
            active_cpu = game_active.to(torch.bool).cpu().numpy()
            nlegal_cpu_pre = num_legal.cpu().numpy()
            root_cpu = root_node.cpu().numpy()
            for i in range(B):
                if self._dbg_expand_logs_emitted >= cfg.debug_expand_limit:
                    break
                if not bool(active_cpu[i]):
                    continue
                if int(nlegal_cpu_pre[i]) <= 0:
                    continue
                # Inconsistent metadata: expansion guard (fc<0) won't trigger,
                # but the root has no children to select from.
                if int(fc_cpu[i]) >= 0 and int(nc_cpu[i]) == 0:
                    print(
                        "[PRS_V2_ROOT_META_INCONSISTENT] "
                        f"game={i} root={int(root_cpu[i])} nlegal={int(nlegal_cpu_pre[i])} "
                        f"fc={int(fc_cpu[i])} nc={int(nc_cpu[i])}",
                    )
                    self._dbg_expand_logs_emitted += 1
        needs_expand = (fc < 0) & (game_active.to(torch.bool))
        if not torch.any(needs_expand):
            return
        root_node_cpu = root_node.cpu().numpy()
        needs_cpu = needs_expand.cpu().numpy()
        nlegal_cpu = num_legal.cpu().numpy()

        pre_fc_cpu = fc.cpu().numpy()
        leaf_indices = root_node.to(torch.int32).clone()
        leaf_indices = torch.where(
            needs_expand, leaf_indices,
            torch.full_like(leaf_indices, -1),
        )
        results = self.ext.check_results_batch(states, B)
        self.ext.mcts_expand_dense_priors_batch(
            *self._tree_args(tree),
            leaf_indices, states,
            legal_moves, num_legal, priors_per_legal, results,
            B, B, self._max_nodes,
        )

        if cfg.debug_expand_logging and self._dbg_expand_logs_emitted < cfg.debug_expand_limit:
            row = torch.arange(B, device="cuda")
            post_fc = tree["first_child"][row, root_node]
            post_nc = tree["num_children"][row, root_node]
            post_fc_cpu = post_fc.cpu().numpy()
            post_nc_cpu = post_nc.cpu().numpy()
            for i in range(B):
                if self._dbg_expand_logs_emitted >= cfg.debug_expand_limit:
                    break
                if not needs_cpu[i]:
                    continue
                nl = int(nlegal_cpu[i])
                if nl <= 0:
                    continue
                pre_fc_i = int(pre_fc_cpu[i])
                post_fc_i = int(post_fc_cpu[i])
                post_nc_i = int(post_nc_cpu[i])
                if post_fc_i < 0 or post_nc_i <= 0:
                    print(
                        "[PRS_V2_EXPAND_FAIL] "
                        f"game={i} root={int(root_node_cpu[i])} nlegal={nl} "
                        f"pre_fc={pre_fc_i} post_fc={post_fc_i} post_nc={post_nc_i}",
                    )
                    self._dbg_expand_logs_emitted += 1

    def _apply_root_dirichlet(self, tree, B: int, active: list[bool]) -> None:
        cfg = self.config
        row = torch.arange(B, device="cuda")
        nc_t = tree["num_children"][row, tree["root_node"]]
        nc_cpu = nc_t.cpu().numpy()
        max_nc = int(nc_cpu.max()) if nc_cpu.size > 0 and nc_cpu.max() > 0 else 1

        noise = np.zeros((B, max_nc), dtype=np.float32)
        for i in range(B):
            nc = int(nc_cpu[i])
            if nc == 0 or not active[i]:
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
        self, tree, root_states, game_active, alive_mask, B: int,
        num_sims: int, wave_size: int | None = None,
    ) -> None:
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
                    game_active, tree["root_node"],
                    alive_mask, self._max_legal,
                    cfg.c_puct, B, actual_w, self._max_nodes,
                )
            )
            if (
                wave == 0 and cfg.debug_select_wave_logging
                and self._dbg_select_logs_emitted < cfg.debug_select_wave_limit
            ):
                leaf_cpu = leaf_idx[:total].cpu().numpy()
                valid_leaf = int((leaf_cpu >= 0).sum())
                row = torch.arange(B, device="cuda")
                roots = tree["root_node"]
                root_nc = tree["num_children"][row, roots].cpu().numpy()
                alive_counts = alive_mask.sum(dim=1).cpu().numpy()
                print(
                    "[PRS_V2_SELECT_WAVE0] "
                    f"total_leaf={total} valid_leaf={valid_leaf} "
                    f"mean_root_children={float(np.mean(root_nc)):.2f} "
                    f"mean_alive={float(np.mean(alive_counts)):.2f} "
                    f"num_sims={num_sims} wave_size={W}",
                )
                self._dbg_select_logs_emitted += 1

            self.ext.mcts_replay_batch(
                root_states, leaf_states[:total],
                move_paths[:total], path_lens[:total], leaf_idx[:total],
                B, total,
            )

            results = self.ext.check_results_batch(leaf_states[:total], total)
            legal_moves, num_legal = self.ext.generate_legal_moves_batch(
                leaf_states[:total], total,
            )

            prs_leaf = self.encoder.encode_batch(leaf_states[:total], total)

            # CUDA classify on leaves (one kernel call instead of CPU loop)
            kernel_leaf = self._classify_kernel(
                leaf_states[:total], legal_moves, num_legal, total,
            )
            slot_of_leaf_t = kernel_leaf[8]

            leaf_logits, leaf_values = self._net_forward(
                prs_leaf, kernel_leaf, total,
            )
            leaf_logits = leaf_logits.float()
            leaf_values = leaf_values.squeeze(-1).float()

            priors_leaf, _ = self._build_legal_priors_v2(
                leaf_logits, slot_of_leaf_t, total,
            )

            self.ext.mcts_expand_and_backprop_dense_priors_batch(
                *self._tree_args(tree),
                leaf_idx[:total], leaf_states[:total],
                legal_moves, num_legal, priors_leaf, results,
                leaf_values, vl_paths[:total], vl_lens[:total],
                B, total, self._max_nodes,
            )

    # ── Gather per-legal-slot root-child stats (same as v1) ──────────

    def _gather_root_child_stats(
        self, tree, B: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = torch.arange(B, device="cuda")
        root = tree["root_node"]
        fc   = tree["first_child"][row, root]
        nc   = tree["num_children"][row, root]

        visits = torch.zeros(B, self._max_legal, dtype=torch.int32,  device="cuda")
        q      = torch.zeros(B, self._max_legal, dtype=torch.float32, device="cuda")

        fc_cpu = fc.cpu().numpy()
        nc_cpu = nc.cpu().numpy()
        for i in range(B):
            fci = int(fc_cpu[i])
            nci = int(nc_cpu[i])
            if fci < 0 or nci == 0:
                continue
            lim = min(nci, self._max_legal)
            vc_slice = tree["visit_count"][i, fci:fci + lim]
            tv_slice = tree["total_value"][i, fci:fci + lim]
            visits[i, :lim] = vc_slice
            with torch.no_grad():
                q[i, :lim] = torch.where(
                    vc_slice > 0,
                    -tv_slice / vc_slice.clamp(min=1).float(),
                    torch.zeros_like(tv_slice),
                )
        return visits, q

    def _reroot_tree(
        self, tree, B: int, active: list[bool], chosen_child_nodes: list[int],
    ) -> None:
        if self.config.rebase_tree_each_move:
            # Hard rebase: drop previous ply's tree for moved games and start
            # the next ply from a fresh root node at index 0.
            for i in range(B):
                if not (active[i] and chosen_child_nodes[i] >= 0):
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
            if active[i] and chosen_child_nodes[i] >= 0:
                new_roots[i] = chosen_child_nodes[i]
        tree["root_node"].copy_(new_roots)
        tree["parent_idx"].scatter_(
            1, new_roots.long().unsqueeze(1),
            torch.full((B, 1), -1, dtype=torch.int32, device="cuda"),
        )

    def _get_turns(self, states: torch.Tensor, B: int) -> np.ndarray:
        lo = states[:B, _OFF_TURN].to(torch.int32)
        hi = states[:B, _OFF_TURN + 1].to(torch.int32)
        return (lo | (hi << 8)).cpu().numpy()

    # ── Immediate-win detection ──────────────────────────────────────

    def _check_immediate_wins_v2(
        self,
        states: torch.Tensor,
        legal_t: torch.Tensor,          # (B, MAX_L, move_size)
        nlegal_np: np.ndarray,          # (B,) int
        active: list[bool],
        current_turns: np.ndarray,      # (B,) int — current turn counter
        B: int,
    ) -> dict[int, int]:
        """Try every legal move; return {game_idx: winning_legal_move_index}.

        Fully batched: one apply_moves_batch + one check_results_batch.
        Cost is one extra CUDA kernel per ply when any game has legal moves
        (cheap relative to MCTS simulations).
        """
        # Build flat (game, move) index arrays for all active games
        gi_flat: list[int] = []
        mi_flat: list[int] = []
        game_ranges: list[tuple[int, int]] = []   # (start, count) per game

        flat_start = 0
        for i in range(B):
            nl = int(nlegal_np[i]) if active[i] else 0
            game_ranges.append((flat_start, nl))
            for m in range(nl):
                gi_flat.append(i)
                mi_flat.append(m)
            flat_start += nl

        total = len(gi_flat)
        if total == 0:
            return {}

        gi_t = torch.tensor(gi_flat, dtype=torch.int64, device="cuda")
        mi_t = torch.tensor(mi_flat, dtype=torch.int64, device="cuda")

        # Single batched apply + check
        test_states = states[gi_t].clone()
        flat_moves  = legal_t[gi_t, mi_t]
        self.ext.apply_moves_batch(test_states, flat_moves, total)
        results_np = self.ext.check_results_batch(test_states, total).cpu().numpy()

        # Find first winning move per game (result==1 for white, 2 for black)
        win_overrides: dict[int, int] = {}
        for i in range(B):
            start, nl = game_ranges[i]
            if nl == 0:
                continue
            win_result = 1 if (int(current_turns[i]) % 2 == 0) else 2
            for m in range(nl):
                if results_np[start + m] == win_result:
                    win_overrides[i] = m
                    break

        return win_overrides

    # ── Build training examples ──────────────────────────────────────

    def _build_examples(
        self,
        histories:     list[list[dict]],
        final_results: np.ndarray,
    ) -> list[list[PRSTrainingExampleV2]]:
        all_examples: list[list[PRSTrainingExampleV2]] = []
        for i, history in enumerate(histories):
            result = int(final_results[i])
            use_for_value = (result != 0)
            game_exs: list[PRSTrainingExampleV2] = []
            for step in history:
                turn = step["turn"]
                player_is_white = (turn % 2 == 0)
                if result == 0 or result == 3:
                    value = 0.0
                elif result == 1:
                    value = 1.0 if player_is_white else -1.0
                else:
                    value = -1.0 if player_is_white else 1.0

                game_exs.append(PRSTrainingExampleV2(
                    token_features   = step["token_features"],
                    token_positions  = step["token_positions"],
                    token_types      = step["token_types"],
                    num_board_tokens = step["num_board_tokens"],
                    global_features  = step["global_features"],
                    seq_length       = step["seq_length"],
                    state_bytes      = step["state_bytes"],
                    legal_moves      = step["legal_moves"],
                    visit_counts     = step["visit_counts"],
                    nlegal           = step["nlegal"],
                    occupied_cells   = step["occupied_cells"],
                    num_occupied     = step["num_occupied"],
                    slot_target      = step["slot_target"],
                    legal_mask       = step["legal_mask"],
                    value_target     = value,
                    use_for_value    = use_for_value,
                ))
            all_examples.append(game_exs)
        return all_examples
