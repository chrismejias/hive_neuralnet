"""
Gumbel AlphaZero + MCTS self-play orchestrator for the PRS model.

Unlike the flat `PRSGumbelOrchestrator` (which evaluates candidates once with
the NN and re-weights via Sequential Halving), this orchestrator runs a
true multi-ply tree search: each simulation is a PUCT tree traversal that
extends the tree by one node.  With N simulations the tree can reach depth
~log2(N) on promising lines, giving real tactical signal that the flat
version cannot produce.

Key differences from the flat version:
  - Uses the GPU MCTS tree kernels (`mcts_select_with_root_mask_batch`,
    `mcts_expand_and_backprop_dense_priors_batch`).
  - Priors feed the tree per-legal-move (dense priors) — no 29,407-dim
    engine action space indirection.
  - Root children are restricted via an alive-mask during Sequential
    Halving; interior nodes use unmodified PUCT.
  - Tree is reused across moves within a game via root re-rooting.

The outer training-example interface (`PRSTrainingExample`) is unchanged so
the trainer and replay buffer work without modification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import hive_gpu
from hive_prs.action_space import (
    ACTION_SPACE_SIZE,
    batch_moves_to_action_indices,
    batch_moves_to_all_reps,
)
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.prs_replay_buffer import PRSTrainingExample

_OFF_TURN = 3412  # byte offset of turn counter in HiveState


@dataclass
class PRSMCTSConfig:
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
    nn_max_batch:                int   = 0
    wave_size:                   int   = 16
    dirichlet_alpha:             float = 0.3
    dirichlet_epsilon:           float = 0.25
    max_tree_nodes:              int   = 8192   # per-game tree capacity


class PRSMCTSOrchestrator:
    """Gumbel AlphaZero with true MCTS tree search, PRS action space."""

    def __init__(
        self,
        net: torch.nn.Module,
        config: PRSMCTSConfig | None = None,
    ) -> None:
        self.ext        = hive_gpu.load_extension()
        self.config     = config or PRSMCTSConfig()
        self.net        = net
        self.encoder    = PRSEncoder()
        self._move_size = self.ext.SIZEOF_GPU_MOVE
        self._max_nodes = int(self.config.max_tree_nodes)
        self._max_depth = int(self.ext.MAX_TREE_DEPTH)
        self._max_legal = int(self.ext.MAX_LEGAL_MOVES)

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

    # ── Public API ────────────────────────────────────────────────────

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

        tree = self._alloc_tree(B)
        self._reset_tree(tree)

        active       = [True] * B
        move_numbers = [0]    * B
        histories: list[list[dict]] = [[] for _ in range(B)]

        while any(active):
            # ── Encode root + legal moves ─────────────────────────────
            prs_batch = self.encoder.encode_batch(states, B)
            occ_cpu   = prs_batch.occupied_cells.cpu().numpy()
            nocc_cpu  = prs_batch.num_occupied.cpu().numpy()

            legal_t, nlegal_t = self.ext.generate_legal_moves_batch(states, B)
            legal_np  = legal_t.cpu().numpy()
            nlegal_np = nlegal_t.cpu().numpy()

            # PRS action index for each legal move (per game)
            prs_from_move = batch_moves_to_action_indices(
                legal_np, nlegal_np, occ_cpu, nocc_cpu,
            )
            # All rotational reps per legal move (for policy-target spreading)
            all_reps_per_game = batch_moves_to_all_reps(
                legal_np, nlegal_np, occ_cpu, nocc_cpu, prs_from_move,
            )

            # ── Root NN forward ───────────────────────────────────────
            with torch.no_grad():
                policy_logits, root_values = self._net_forward(prs_batch, B)
            policy_logits = policy_logits.float()   # (B, 6841)

            # Build per-legal priors tensor on GPU (softmax over legal moves)
            priors_per_legal, root_logit_per_legal = self._build_legal_priors(
                policy_logits, prs_from_move, nlegal_np, B,
            )

            # ── Gumbel noise + top-k considered candidates (per game) ─
            max_k = min(cfg.max_num_considered_actions, int(nlegal_np.max())) \
                if int(nlegal_np.max()) > 0 else 1

            if max_k == 0:
                for i in range(B):
                    active[i] = False
                break

            # Gumbel over legal slots (inf for padded slots)
            u = torch.rand(B, self._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
            gumbel = -torch.log(-torch.log(u))
            nlegal_t_gpu = nlegal_t.to(torch.int64)
            slot_idx = torch.arange(self._max_legal, device="cuda").unsqueeze(0)
            valid_slot = slot_idx < nlegal_t_gpu.unsqueeze(1)    # (B, MAX_L)
            gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
            perturbed = gumbel + root_logit_per_legal            # (B, MAX_L)
            # Top-k over legal slots
            topk_scores, topk_slots = torch.topk(perturbed, max_k, dim=1)  # (B, max_k)

            # ── Expand root via dense-priors kernel ──────────────────
            # Use empty/zero tensors for vl_paths since root expansion has no
            # prior virtual-loss path.  Game-active mask controls which games
            # get expanded.
            active_t = torch.tensor(
                [1 if a else 0 for a in active], dtype=torch.int8, device="cuda",
            )
            self._expand_root_if_needed(
                tree, states, legal_t, nlegal_t,
                priors_per_legal, active_t, B,
            )

            # Apply Dirichlet noise to root child priors (in-tree)
            self._apply_root_dirichlet(tree, B, active)

            # ── Alive mask (over legal slots) ─────────────────────────
            # alive_mask[b, slot] = 1 if slot in topk_slots[b, :max_k]
            alive_mask = torch.zeros(B, self._max_legal, dtype=torch.int8, device="cuda")
            alive_mask.scatter_(1, topk_slots, 1)
            # Mask to only games that have real legal moves
            for _ in range(1):   # keep as a tensor op for clarity
                pass

            # ── Sequential Halving: run sims with alive restriction ──
            q_sums       = torch.zeros(B, self._max_legal, dtype=torch.float32, device="cuda")
            visit_counts = torch.zeros(B, self._max_legal, dtype=torch.int32,  device="cuda")

            n_rounds = max(1, math.ceil(math.log2(max(max_k, 2))))
            total_sims = cfg.num_simulations
            # Per-round sim budget
            sims_per_round = max(1, total_sims // n_rounds)

            for round_i in range(n_rounds):
                # Run sims_per_round simulations (wave-parallelised)
                self._run_simulations(
                    tree, states, active_t, alive_mask, B, sims_per_round,
                )

                # Halve alive set: keep top-half by sigma score
                alive_counts = alive_mask.sum(dim=1)                  # (B,)
                cur_alive_max = int(alive_counts.max().item())
                if cur_alive_max <= 1:
                    continue
                num_keep = max(1, cur_alive_max // 2)

                # Gather Q for each legal slot from tree's root-child visits/values
                slot_visits, slot_q = self._gather_root_child_stats(tree, B)
                # slot_visits, slot_q both (B, MAX_L)
                max_n = int(slot_visits.max().item())
                sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
                # sigma_score for each slot (using current Q estimate)
                sigma_score = root_logit_per_legal + (
                    torch.empty(B, self._max_legal, device="cuda").copy_(
                        gumbel  # gumbel noise saved above (g)
                    )
                )
                # Re-compute using the saved gumbel tensor (not temp copy — avoid alias)
                sigma_score = (gumbel + root_logit_per_legal
                               + sigma_norm * slot_q).float()
                # Mask out non-alive / padded
                sigma_score = torch.where(
                    alive_mask.bool(),
                    sigma_score,
                    torch.full_like(sigma_score, -1e30),
                )

                # Keep top num_keep per game
                _, keep_slots = torch.topk(sigma_score, num_keep, dim=1)
                new_alive = torch.zeros_like(alive_mask)
                new_alive.scatter_(1, keep_slots, 1)
                # Preserve padding invariant: only valid slots may remain alive
                new_alive = new_alive * valid_slot.to(torch.int8)
                alive_mask = new_alive

            # ── Final action selection ────────────────────────────────
            slot_visits, slot_q = self._gather_root_child_stats(tree, B)
            max_n = int(slot_visits.max().item())
            sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
            final_sigma = (gumbel + root_logit_per_legal
                           + sigma_norm * slot_q).float()
            final_sigma = torch.where(
                alive_mask.bool(), final_sigma,
                torch.full_like(final_sigma, -1e30),
            )
            # argmax per game (greedy among survivors)
            chosen_slot = torch.argmax(final_sigma, dim=1)   # (B,) legal slot

            # ── Record history + build policy from visit counts ──────
            # Policy target (per-legal visits, normalised) — the (slot, PRS idx)
            # mapping is already in prs_from_move; we record them for build_examples.
            visits_np  = slot_visits.cpu().numpy()     # (B, MAX_L) int32
            chosen_np  = chosen_slot.cpu().numpy()
            turns_np   = self._get_turns(states, B)

            tf_cpu = prs_batch.token_features.cpu().numpy()
            tp_cpu = prs_batch.token_positions.cpu().numpy()
            tt_cpu = prs_batch.token_types.cpu().numpy()
            nb_cpu = prs_batch.num_board_tokens.cpu().numpy()
            gf_cpu = prs_batch.global_features.cpu().numpy()
            sl_cpu = prs_batch.seq_lengths.cpu().numpy()
            nn_prior_cpu = torch.softmax(policy_logits, dim=-1).cpu().numpy()

            for i in range(B):
                if not active[i]:
                    continue
                n_i = int(nlegal_np[i])
                vs  = visits_np[i, :n_i].astype(np.float32)
                vsum = vs.sum()
                if vsum > 0:
                    probs = vs / vsum
                else:
                    probs = np.ones(n_i, dtype=np.float32) / max(n_i, 1)
                S = int(sl_cpu[i])
                histories[i].append({
                    "token_features":   tf_cpu[i, :S].copy(),
                    "token_positions":  tp_cpu[i, :S].astype(np.int32),
                    "token_types":      tt_cpu[i, :S].astype(np.int8),
                    "num_board_tokens": int(nb_cpu[i]),
                    "global_features":  gf_cpu[i].copy(),
                    "seq_length":       S,
                    "occupied_cells":   occ_cpu[i].copy(),
                    "num_occupied":     int(nocc_cpu[i]),
                    "slot_probs":       probs.copy(),
                    "prs_from_move":    prs_from_move[i].copy(),
                    "all_reps":         all_reps_per_game[i].copy(),
                    "nn_prior":         nn_prior_cpu[i].copy(),
                    "turn":             int(turns_np[i]),
                })

            # ── Apply chosen moves + re-root tree ────────────────────
            move_bytes_np = np.zeros((B, self._move_size), dtype=np.uint8)
            chosen_child_nodes = [-1] * B
            for i in range(B):
                if not active[i]:
                    continue
                slot = int(chosen_np[i])
                n_i  = int(nlegal_np[i])
                if slot >= n_i:
                    # Fallback to slot 0
                    slot = 0
                move_bytes_np[i] = legal_np[i, slot]
                # Child node for re-rooting: first_child[b, old_root] + slot
                old_root = int(tree["root_node"][i].item())
                fc = int(tree["first_child"][i, old_root].item())
                if fc >= 0:
                    chosen_child_nodes[i] = fc + slot

            move_bytes_t = torch.from_numpy(move_bytes_np).cuda()
            self.ext.apply_moves_batch(states, move_bytes_t, B)

            # Re-root
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

    # ── NN forward (with optional sub-batching) ──────────────────────

    def _net_forward(
        self, prs_batch, B: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mb = self.config.nn_max_batch
        if mb <= 0 or B <= mb:
            with torch.no_grad():
                return self.net(prs_batch)
        all_l, all_v = [], []
        for s in range(0, B, mb):
            e = min(s + mb, B)
            with torch.no_grad():
                lo, v = self.net(prs_batch.slice_batch(s, e))
            all_l.append(lo)
            all_v.append(v)
        return torch.cat(all_l, 0), torch.cat(all_v, 0)

    # ── Build per-legal priors + per-legal logit (padded to MAX_LEGAL) ─

    def _build_legal_priors(
        self,
        policy_logits: torch.Tensor,   # (B, ACTION_SPACE_SIZE)
        prs_from_move: list[np.ndarray],
        nlegal_np:     np.ndarray,
        B: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (priors_per_legal [B,MAX_L], logit_per_legal [B,MAX_L]).

        Slots >= n_legal are filled with -inf logit and 0 prior.
        """
        # Assemble a (B, MAX_L) int64 tensor of PRS action indices per legal slot.
        legal_prs = np.full((B, self._max_legal), -1, dtype=np.int64)
        for i in range(B):
            arr = prs_from_move[i]
            n   = min(len(arr), self._max_legal)
            legal_prs[i, :n] = arr[:n]
        legal_prs_t = torch.from_numpy(legal_prs).cuda()

        # Gather logits per-legal; invalid PRS indices (-1) get -inf
        valid_t = legal_prs_t >= 0
        safe_idx = legal_prs_t.clamp(min=0)
        gathered = torch.gather(policy_logits, 1, safe_idx)   # (B, MAX_L)
        legal_logits = torch.where(
            valid_t, gathered, torch.full_like(gathered, -1e30),
        )

        # Mask padded slots (beyond n_legal) to -inf
        slot_idx = torch.arange(self._max_legal, device="cuda").unsqueeze(0)
        nlegal_t = torch.from_numpy(nlegal_np.astype(np.int64)).cuda().unsqueeze(1)
        valid_slot = slot_idx < nlegal_t                      # (B, MAX_L)
        legal_logits = torch.where(
            valid_slot, legal_logits, torch.full_like(legal_logits, -1e30),
        )

        # Softmax over valid slots
        priors = torch.softmax(legal_logits, dim=-1)
        # Zero out padded priors (softmax of -inf is 0 but guard against NaN)
        priors = torch.where(valid_slot, priors, torch.zeros_like(priors))

        return priors.contiguous(), legal_logits.contiguous()

    # ── Expand root: call dense_priors_expand for games with an unexpanded root ─

    def _expand_root_if_needed(
        self,
        tree: dict[str, torch.Tensor],
        states: torch.Tensor,
        legal_moves: torch.Tensor,
        num_legal:   torch.Tensor,
        priors_per_legal: torch.Tensor,
        game_active: torch.Tensor,
        B: int,
    ) -> None:
        """Expand the current root per game if its children haven't been
        created yet.  If the root was rerooted from a prior move, its
        children are already in the tree and we skip re-expansion.
        """
        # Gather first_child for the current root (per game)
        row = torch.arange(B, device="cuda")
        root_node = tree["root_node"]
        fc = tree["first_child"][row, root_node]                 # (B,)
        needs_expand = (fc < 0) & (game_active.to(torch.bool))
        if not torch.any(needs_expand):
            return

        # mcts_expand_dense_priors_kernel expands leaves given leaf_indices.
        # We want to expand root-per-game for games in need_expand.
        # Use leaf_indices = root_node for all B; no-op for already-expanded.
        leaf_indices = root_node.to(torch.int32).clone()
        # Games that do not need expansion: set leaf_idx = -1 (kernel skips)
        leaf_indices = torch.where(
            needs_expand, leaf_indices,
            torch.full_like(leaf_indices, -1),
        )

        # Terminal detection at root
        results = self.ext.check_results_batch(states, B)

        self.ext.mcts_expand_dense_priors_batch(
            *self._tree_args(tree),
            leaf_indices, states,
            legal_moves, num_legal, priors_per_legal, results,
            B, B, self._max_nodes,
        )

    # ── Root Dirichlet noise via GPU kernel (reused) ──────────────────

    def _apply_root_dirichlet(
        self,
        tree: dict[str, torch.Tensor],
        B: int,
        active: list[bool],
    ) -> None:
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

    # ── Run `num_sims` simulations (wave-parallel) ───────────────────

    def _run_simulations(
        self,
        tree: dict[str, torch.Tensor],
        root_states: torch.Tensor,
        game_active: torch.Tensor,
        alive_mask:  torch.Tensor,   # (B, MAX_L) int8
        B: int,
        num_sims: int,
    ) -> None:
        cfg = self.config
        W = cfg.wave_size
        num_waves = math.ceil(num_sims / W)

        state_size = root_states.shape[1]
        leaf_states = torch.zeros(W * B, state_size, dtype=torch.uint8, device="cuda")

        for wave in range(num_waves):
            actual_w = min(W, num_sims - wave * W)
            total = actual_w * B

            # SELECT with root mask
            leaf_idx, move_paths, path_lens, vl_paths, vl_lens = (
                self.ext.mcts_select_with_root_mask_batch(
                    *self._tree_args(tree),
                    game_active, tree["root_node"],
                    alive_mask, self._max_legal,
                    cfg.c_puct, B, actual_w, self._max_nodes,
                )
            )

            # REPLAY
            self.ext.mcts_replay_batch(
                root_states, leaf_states[:total],
                move_paths[:total], path_lens[:total], leaf_idx[:total],
                B, total,
            )

            # TERMINAL CHECK
            results = self.ext.check_results_batch(leaf_states[:total], total)

            # Legal moves at leaves
            legal_moves, num_legal = self.ext.generate_legal_moves_batch(
                leaf_states[:total], total,
            )

            # ENCODE leaves + NN forward
            prs_leaf = self.encoder.encode_batch(leaf_states[:total], total)
            with torch.no_grad():
                if cfg.nn_max_batch > 0 and total > cfg.nn_max_batch:
                    leaf_logits, leaf_values = self._nn_sub(prs_leaf, total)
                else:
                    leaf_logits, leaf_values = self.net(prs_leaf)
            leaf_logits = leaf_logits.float()
            leaf_values = leaf_values.squeeze(-1).float()

            # Build per-legal priors at leaves
            occ = prs_leaf.occupied_cells.cpu().numpy()
            nocc = prs_leaf.num_occupied.cpu().numpy()
            legal_np  = legal_moves.cpu().numpy()
            nlegal_np = num_legal.cpu().numpy()
            prs_from_leaf = batch_moves_to_action_indices(
                legal_np, nlegal_np, occ, nocc,
            )
            priors_leaf, _ = self._build_legal_priors(
                leaf_logits, prs_from_leaf, nlegal_np, total,
            )

            # EXPAND + BACKPROP (fused, dense priors)
            self.ext.mcts_expand_and_backprop_dense_priors_batch(
                *self._tree_args(tree),
                leaf_idx[:total], leaf_states[:total],
                legal_moves, num_legal, priors_leaf, results,
                leaf_values, vl_paths[:total], vl_lens[:total],
                B, total, self._max_nodes,
            )

    def _nn_sub(self, batch, total: int):
        mb = self.config.nn_max_batch
        all_l, all_v = [], []
        for s in range(0, total, mb):
            e = min(s + mb, total)
            with torch.no_grad():
                lo, v = self.net(batch.slice_batch(s, e))
            all_l.append(lo); all_v.append(v)
        return torch.cat(all_l, 0), torch.cat(all_v, 0)

    # ── Gather per-legal-slot root-child stats ───────────────────────

    def _gather_root_child_stats(
        self, tree: dict[str, torch.Tensor], B: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (visits [B,MAX_L], q [B,MAX_L]).

        q is the root-player-perspective mean value of the child
        (= -total_value/visits, because child total_value is stored from
        the opponent's perspective).  Slot index = legal-move index.
        """
        row = torch.arange(B, device="cuda")
        root = tree["root_node"]
        fc   = tree["first_child"][row, root]     # (B,)
        nc   = tree["num_children"][row, root]    # (B,)

        visits = torch.zeros(B, self._max_legal, dtype=torch.int32,  device="cuda")
        q      = torch.zeros(B, self._max_legal, dtype=torch.float32, device="cuda")

        # For each game, read visits/total for children slots 0..nc-1
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

    # ── Tree re-rooting ──────────────────────────────────────────────

    def _reroot_tree(
        self,
        tree: dict[str, torch.Tensor],
        B: int,
        active: list[bool],
        chosen_child_nodes: list[int],
    ) -> None:
        new_roots = tree["root_node"].clone()
        for i in range(B):
            if active[i] and chosen_child_nodes[i] >= 0:
                new_roots[i] = chosen_child_nodes[i]
        tree["root_node"].copy_(new_roots)
        # Clear parent link at the new root so backprop terminates there
        tree["parent_idx"].scatter_(
            1, new_roots.long().unsqueeze(1),
            torch.full((B, 1), -1, dtype=torch.int32, device="cuda"),
        )

    # ── Turn extraction (for history) ────────────────────────────────

    def _get_turns(self, states: torch.Tensor, B: int) -> np.ndarray:
        lo = states[:B, _OFF_TURN].to(torch.int32)
        hi = states[:B, _OFF_TURN + 1].to(torch.int32)
        return (lo | (hi << 8)).cpu().numpy()

    # ── Build training examples (reuses spreading logic from flat) ───

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

                # Spread per-slot MCTS visits → PRS policy, with rep spreading
                slot_probs = step["slot_probs"]          # (n_legal,)
                prs_idxs   = step["prs_from_move"]       # (n_legal,) int32
                all_reps   = step["all_reps"]            # (n_legal, 6) int32
                nn_prior   = step["nn_prior"]            # (A,) float32

                policy           = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                canonical_policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

                for slot_i in range(len(slot_probs)):
                    p = float(slot_probs[slot_i])
                    if p <= 0.0:
                        continue
                    can = int(prs_idxs[slot_i]) if slot_i < len(prs_idxs) else -1
                    if 0 <= can < ACTION_SPACE_SIZE:
                        canonical_policy[can] += p
                    # Spread across reps of this physical move
                    if slot_i < all_reps.shape[0]:
                        reps = all_reps[slot_i]
                        reps = reps[(reps >= 0) & (reps < ACTION_SPACE_SIZE)]
                        if len(reps) > 0:
                            policy[reps] += p / len(reps)
                        elif 0 <= can < ACTION_SPACE_SIZE:
                            policy[can] += p
                    elif 0 <= can < ACTION_SPACE_SIZE:
                        policy[can] += p

                # Epsilon floor on all legal reps
                all_legal = all_reps.ravel()
                all_legal = all_legal[(all_legal >= 0) & (all_legal < ACTION_SPACE_SIZE)]
                if len(all_legal) > 0:
                    all_legal = np.unique(all_legal)
                    policy[all_legal] = np.maximum(policy[all_legal], 1e-4)
                psum = policy.sum()
                if psum > 0:
                    policy /= psum

                # Surprise weight via KL(canonical_policy || nn_prior)
                csum = canonical_policy.sum()
                if csum > 0:
                    canonical_policy /= csum
                mask = canonical_policy > 0
                sw = 1.0
                if mask.any():
                    p_pos = canonical_policy[mask]
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
