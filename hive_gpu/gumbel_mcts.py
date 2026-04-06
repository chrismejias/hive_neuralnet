"""
Gumbel AlphaZero search with batched GPU inference.

Replaces iterative MCTS tree traversals with Sequential Halving + Gumbel
noise (Danihelka et al., 2022).  All NN evaluations within a halving round
are independent and batched together, eliminating the serial tree-traversal
bottleneck of standard MCTS.

Key differences from GPU-native MCTS:
  - No tree data structure — flat tensors for Q-values and visit counts
  - Fixed number of sequential rounds (log2 of considered actions)
  - Much better GPU utilization at equivalent simulation budgets
  - Designed for large-VRAM GPUs (24–80 GB)

Usage:
    from hive_gpu.gumbel_mcts import GumbelAlphaZeroOrchestrator, GumbelConfig

    orchestrator = GumbelAlphaZeroOrchestrator(net, config)
    examples = orchestrator.self_play_batch()
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

import hive_gpu

try:
    from hive_gnn.graph_types import HiveGraph
except ImportError:
    HiveGraph = None  # type: ignore[assignment,misc]
from hive_gpu.gpu_encoder import GPUGNNEncoder, GPUTransformerEncoder
from archive.modules.hive_gpu_hybrid.gpu_mcts import GPUMCTSConfig, GPUTrainingExample
from hive_transformer.token_types import HiveTokenSequence


# ── Configuration ─────────────────────────────────────────────────────


@dataclass
class GumbelConfig:
    """Configuration for Gumbel AlphaZero search."""

    # Simulation budget (total NN evals per move, excluding root eval)
    num_simulations: int = 128

    # Maximum actions to consider after initial top-k selection.
    # Paper uses 16 or 32.  Lower = fewer rounds, faster; higher = broader.
    max_num_considered_actions: int = 32

    # Q-transform parameters: sigma(q) = (c_visit + max_n) * c_scale * q
    c_visit: float = 50.0
    c_scale: float = 1.0

    # Game parameters
    temperature: float = 1.0
    temperature_drop_move: int = 20
    batch_size: int = 256
    max_game_length: int = 300
    encoder_type: str = "transformer"
    expansion_mask: int = 0

    # NN batching
    nn_max_batch: int = 0  # 0 = no limit

    # Queen pressure value shaping for draws
    queen_pressure_scale: float = 0.4


# ── Orchestrator ──────────────────────────────────────────────────────


class GumbelAlphaZeroOrchestrator:
    """
    GPU-native Gumbel AlphaZero search for batched self-play.

    No MCTS tree.  Sequential halving with Gumbel noise selects actions.
    All NN evaluations within a halving round are independent and batched.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        config: GumbelConfig | None = None,
    ) -> None:
        self.ext = hive_gpu.load_extension()
        self.config = config or GumbelConfig()
        self.net = net

        if self.config.encoder_type == "gnn":
            self.encoder = GPUGNNEncoder()
        else:
            self.encoder = GPUTransformerEncoder()

        self._action_space_size = self.ext.ACTION_SPACE_SIZE
        self._move_size = self.ext.SIZEOF_GPU_MOVE

    # ── Public API ────────────────────────────────────────────────

    def self_play_batch(
        self,
        start_states: "torch.Tensor | None" = None,
    ) -> list[list[GPUTrainingExample]]:
        """Play B games in parallel using Gumbel AlphaZero search."""
        B = self.config.batch_size
        cfg = self.config

        if start_states is not None:
            states = start_states.cuda()
        else:
            states = self.ext.create_initial_states(B, cfg.expansion_mask)

        active = [True] * B
        move_numbers = [0] * B
        histories: list[list[tuple]] = [[] for _ in range(B)]

        while any(active):
            current_turns = self._get_turns(states, B)

            # ── Gumbel search for all active games ──
            # policies: list of (action_indices [max_k], probs [max_k]) sparse tuples
            # nn_priors: list of (action_indices [max_k], nn_prior_vals [max_k]) or None
            policies, graphs, seqs, nn_priors, root_legal_moves, root_legal_action_indices, root_num_legal = self._gumbel_search(
                states, B, active, move_numbers,
            )

            # Compute per-piece mobility for history.
            # Start a non-blocking transfer so the DMA can run while
            # _check_immediate_wins dispatches its GPU kernels.
            mob_tensor, mob_counts = self.ext.compute_mobility_batch(states, B, False)
            mob_cpu        = mob_tensor.to("cpu", non_blocking=True)
            mob_counts_cpu = mob_counts.to("cpu", non_blocking=True)

            # Override policies for immediate wins — returns {game_idx: win_action}
            win_overrides = self._check_immediate_wins(states, B, active, current_turns)

            # mob transfers are complete by now (GPU kernels above forced sync).
            mob_np          = mob_cpu.numpy()
            mob_board_counts = mob_counts_cpu.numpy()

            # Record history and select actions
            # selected_actions[i] = chosen action index; -1 for inactive games.
            selected_actions = np.full(B, -1, dtype=np.int64)

            for i in range(B):
                if not active[i]:
                    continue

                act_indices, probs = policies[i]   # [max_k] each (numpy)
                turn = current_turns[i]

                mob_i = mob_np[i, :mob_board_counts[i]].copy()
                histories[i].append(
                    (graphs[i], (act_indices, probs), turn, mob_i, seqs[i], nn_priors[i])
                )

                # Apply win override: force action and store one-hot sparse policy
                if i in win_overrides:
                    action = win_overrides[i]
                    # Replace history tail with one-hot sparse policy
                    histories[i][-1] = (
                        graphs[i],
                        (np.array([action], dtype=act_indices.dtype),
                         np.array([1.0], dtype=probs.dtype)),
                        turn, mob_i, seqs[i], nn_priors[i],
                    )
                    selected_actions[i] = action
                    continue

                # Action selection: greedy after temp_drop, else sample from sparse dist
                if move_numbers[i] >= cfg.temperature_drop_move:
                    selected_actions[i] = int(act_indices[int(np.argmax(probs))])
                else:
                    psum = probs.sum()
                    if psum > 0 and np.isfinite(psum):
                        p = probs / psum
                    else:
                        p = np.ones(len(probs), dtype=np.float32) / len(probs)
                    local_idx = int(np.random.choice(len(p), p=p))
                    selected_actions[i] = int(act_indices[local_idx])

            # Vectorised move lookup — one gather replaces B per-game kernel launches.
            actions_t = torch.from_numpy(selected_actions).to(device="cuda").unsqueeze(1)  # [B, 1]
            matched = self._match_actions_to_legal_moves(
                actions_t, root_legal_action_indices, root_num_legal,
            ).squeeze(1)  # [B]
            has_match = matched >= 0
            safe_idx = matched.clamp(min=0).long()
            batch_idx = torch.arange(B, device="cuda")
            move_bytes_t = root_legal_moves[batch_idx, safe_idx]  # [B, move_size]
            move_bytes_t[~has_match] = 0

            # Apply moves directly from the GPU tensor — no CPU roundtrip needed.
            self.ext.apply_moves_batch(states, move_bytes_t, B)

            for i in range(B):
                if active[i]:
                    move_numbers[i] += 1

            # Check game over
            results = self.ext.check_results_batch(states, B).cpu().numpy()
            for i in range(B):
                if not active[i]:
                    continue
                if results[i] != 0 or move_numbers[i] >= cfg.max_game_length:
                    active[i] = False

        # Build training examples
        final_results = self.ext.check_results_batch(states, B).cpu().numpy()
        final_mob_tensor, final_mob_counts = self.ext.compute_mobility_batch(
            states, B, True,
        )
        final_mob_np = final_mob_tensor.cpu().numpy()
        final_mob_board_counts = final_mob_counts.cpu().numpy()
        final_qs_data = self._compute_queen_surround_batch(states, B)

        return self._build_examples(
            histories, final_results,
            final_mob_np, final_mob_board_counts,
            final_qs_data,
        )

    # ── Gumbel search core ────────────────────────────────────────

    def _gumbel_search(
        self,
        root_states: torch.Tensor,
        B: int,
        active: list[bool],
        move_numbers: list[int],
    ) -> tuple[
        list[np.ndarray],
        list,
        list[HiveTokenSequence | None],
        list[np.ndarray | None],
        torch.Tensor,  # legal_moves  [B, MAX_LEGAL_MOVES, move_size]
        torch.Tensor,  # legal_action_indices  [B, MAX_LEGAL_MOVES]
        torch.Tensor,  # num_legal_moves  [B]
    ]:
        """Run Gumbel AlphaZero search.  Returns (policies, graphs, seqs, nn_priors)."""
        cfg = self.config
        A = self._action_space_size

        # ── Step 1: Root NN evaluation ──
        # Encode root states for NN + history
        root_seqs: list[HiveTokenSequence | None] = [None] * B
        if hasattr(self.encoder, "encode_batch_with_graphs_and_seqs"):
            encoded, root_graphs, root_seqs = (
                self.encoder.encode_batch_with_graphs_and_seqs(root_states, B)
            )
        elif hasattr(self.encoder, "encode_batch_with_graphs"):
            encoded, root_graphs = self.encoder.encode_batch_with_graphs(root_states, B)
        else:
            encoded = self.encoder.encode_batch(root_states, B)
            root_graphs = [None] * B

        # Legal mask
        legal_mask_int, _ = self.ext.generate_legal_mask_batch(root_states, B)
        legal_mask = legal_mask_int.bool()  # [B, A]  True = legal

        # NN forward pass on roots
        with torch.no_grad():
            if cfg.nn_max_batch > 0 and B > cfg.nn_max_batch:
                policy_logits, root_values = self._nn_forward_subbatched(encoded, B)
            else:
                policy_logits, root_values, *_ = self.net(encoded)

        root_values = root_values.squeeze(-1)  # [B]

        # Mask illegal actions
        policy_logits[~legal_mask] = float("-inf")

        # Store NN priors (for surprise weighting)
        nn_prior_probs = torch.softmax(policy_logits, dim=-1)  # [B, A]

        # ── Step 2: Gumbel noise ──
        # Sample Gumbel(0) noise: g = -log(-log(u)), u ~ Uniform(0,1)
        u = torch.rand(B, A, device="cuda").clamp(1e-10, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(u))
        gumbel[~legal_mask] = float("-inf")

        # Perturbed logits
        perturbed = gumbel + policy_logits  # [B, A]

        # ── Step 3: Top-k action selection ──
        num_legal = legal_mask.sum(dim=1)  # [B]
        k = cfg.max_num_considered_actions
        # For games with fewer legal moves than k, we consider all of them
        effective_k = torch.clamp(num_legal, max=k)  # [B]
        max_k = min(k, int(num_legal.max().item()))

        if max_k == 0:
            # No legal moves at all — return uniform policies
            policies = [np.zeros(A, dtype=np.float32) for _ in range(B)]
            nn_priors = [nn_prior_probs[i].cpu().numpy() for i in range(B)]
            return policies, root_graphs, root_seqs, nn_priors

        # Select top max_k actions per game
        topk_scores, topk_actions = torch.topk(perturbed, max_k, dim=1)  # [B, max_k]

        # ── Step 4: Sequential halving ──
        q_sums = torch.zeros(B, max_k, device="cuda")   # accumulated child values
        visit_counts = torch.zeros(B, max_k, dtype=torch.int32, device="cuda")
        candidate_mask = torch.ones(B, max_k, dtype=torch.bool, device="cuda")

        # Mask out padding for games with fewer than max_k legal moves
        for i in range(B):
            ek = int(effective_k[i].item())
            if ek < max_k:
                candidate_mask[i, ek:] = False

        # Number of halving rounds
        num_candidates_initial = int(candidate_mask.sum(dim=1).max().item())
        num_rounds = max(1, math.ceil(math.log2(num_candidates_initial)))

        # Get legal moves upfront for child state construction
        legal_moves, num_legal_moves = self.ext.generate_legal_moves_batch(root_states, B)
        legal_action_indices = self.ext.legal_moves_to_actions_batch(
            root_states, legal_moves, num_legal_moves, B,
        )

        remaining_budget = torch.full(
            (B,), cfg.num_simulations, dtype=torch.int32, device="cuda"
        )

        for round_idx in range(num_rounds):
            n_remaining = candidate_mask.sum(dim=1).float()  # [B]
            max_remaining = int(n_remaining.max().item())
            if max_remaining == 0:
                break

            # Allocate the remaining per-game simulation budget across the
            # still-alive candidates. This keeps later rounds meaningful
            # instead of duplicating the same one-ply value estimate.
            n_remaining_i = candidate_mask.sum(dim=1).int()
            visits_this_round = torch.where(
                n_remaining_i > 0,
                torch.clamp(remaining_budget // n_remaining_i.clamp(min=1), min=1),
                torch.zeros_like(remaining_budget),
            )
            if int(visits_this_round.max().item()) == 0:
                break

            # ── Build child states for all candidates × visits ──
            child_values = self._evaluate_candidates(
                root_states, B, topk_actions, candidate_mask,
                visits_this_round, legal_moves, num_legal_moves,
                legal_action_indices,
            )  # [B, max_k] mean child values (negated = from root player perspective)

            # Update Q-values
            visits_per_candidate = visits_this_round.unsqueeze(1).float()
            q_sums += child_values * visits_per_candidate * candidate_mask.float()
            visit_counts += (
                visits_this_round.unsqueeze(1) * candidate_mask.int()
            )
            remaining_budget = torch.clamp(
                remaining_budget - visits_this_round * n_remaining_i,
                min=0,
            )

            # ── Score candidates and halve ──
            if round_idx < num_rounds - 1:
                max_n = visit_counts.max(dim=1, keepdim=True).values.float().clamp(min=1)
                q_mean = torch.where(
                    visit_counts > 0,
                    q_sums / visit_counts.float(),
                    torch.zeros_like(q_sums),
                )
                qtransform = (cfg.c_visit + max_n) * cfg.c_scale * q_mean

                # Gumbel scores: g(a) + logit(a) + qtransform(Q(a))
                # Gather the original logits for the topk actions
                topk_logits = policy_logits.gather(1, topk_actions)  # [B, max_k]
                topk_gumbel = gumbel.gather(1, topk_actions)  # [B, max_k]
                sigma = topk_gumbel + topk_logits + qtransform
                sigma[~candidate_mask] = float("-inf")

                # Keep top half
                n_keep = (n_remaining / 2).ceil().int().clamp(min=1)
                max_keep = int(n_keep.max().item())

                # Vectorised halving: rank every candidate slot, keep rank < n_keep.
                # sorted_idx[i, r] = index of the r-th best slot for game i.
                # inv_rank[i, j] = rank of slot j (0 = best).
                sorted_idx = sigma.argsort(dim=1, descending=True)  # [B, max_k]
                inv_rank = torch.empty_like(sorted_idx)
                inv_rank.scatter_(
                    1, sorted_idx,
                    torch.arange(max_k, device="cuda").unsqueeze(0).expand(B, -1),
                )
                candidate_mask = (inv_rank < n_keep.unsqueeze(1)) & candidate_mask

        # ── Step 5: Compute improved policy ──
        policies_np, nn_priors_list = self._compute_improved_policy(
            policy_logits, topk_actions, q_sums, visit_counts,
            legal_mask, root_values, nn_prior_probs, B, max_k, active,
        )

        return policies_np, root_graphs, root_seqs, nn_priors_list, legal_moves, legal_action_indices, num_legal_moves

    # ── Child state evaluation ────────────────────────────────────

    def _evaluate_candidates(
        self,
        root_states: torch.Tensor,
        B: int,
        topk_actions: torch.Tensor,   # [B, max_k]
        candidate_mask: torch.Tensor,  # [B, max_k]
        visits_per_action: torch.Tensor,
        legal_moves: torch.Tensor,
        num_legal_moves: torch.Tensor,
        legal_action_indices: torch.Tensor,  # [B, MAX_LEGAL_MOVES]
    ) -> torch.Tensor:
        """Evaluate candidate actions by applying them and running NN on child states.

        For each active candidate, clone root → apply action → NN eval.
        Returns [B, max_k] tensor of mean child values (negated for root perspective).

        When more than one simulation is allocated to a surviving candidate, we
        probe multiple likely replies from the child state and aggregate them as
        a pessimistic best reply for the opponent. That gives later halving
        rounds additional search signal instead of duplicating the same one-ply
        value.
        """
        max_k = topk_actions.shape[1]
        state_size = root_states.shape[1]

        # Collect all (game_idx, candidate_idx) pairs that need evaluation
        matched_moves = self._match_actions_to_legal_moves(
            topk_actions, legal_action_indices, num_legal_moves,
        )
        active_candidates = candidate_mask & (matched_moves >= 0)
        candidate_pairs = torch.nonzero(active_candidates, as_tuple=False)

        if candidate_pairs.numel() == 0:
            return torch.zeros(B, max_k, device="cuda")

        gi_t = candidate_pairs[:, 0].to(dtype=torch.int64)
        cj_t = candidate_pairs[:, 1].to(dtype=torch.int64)
        mi_t = matched_moves[gi_t, cj_t].to(dtype=torch.int64)
        total = int(candidate_pairs.shape[0])

        # Clone root states for each candidate
        child_states = root_states[gi_t].clone()  # [total, state_size]

        # Gather move bytes
        moves = legal_moves[gi_t, mi_t]  # [total, move_size]

        # Apply moves
        self.ext.apply_moves_batch(child_states, moves, total)

        # Check for terminal states (keep on GPU for vectorised override below)
        results_t = self.ext.check_results_batch(child_states, total)
        results = results_t.cpu().numpy()  # still needed by _probe_child_replies

        # Encode child states and run NN
        encoded = self.encoder.encode_batch(child_states, total)
        with torch.no_grad():
            cfg = self.config
            if cfg.nn_max_batch > 0 and total > cfg.nn_max_batch:
                child_logits, child_values = self._nn_forward_subbatched(encoded, total)
            else:
                child_logits, child_values, *_ = self.net(encoded)

        child_values = child_values.squeeze(-1)  # [total]

        # Vectorised terminal override — no Python loop over results.
        # Results: 0=in_progress, 1=white_wins, 2=black_wins, 3=draw
        root_turn_bytes = root_states[gi_t, self._OFF_TURN]  # [total]
        root_is_white = (root_turn_bytes % 2 == 0)
        root_won = ((results_t == 1) & root_is_white) | ((results_t == 2) & ~root_is_white)
        is_terminal = results_t != 0
        is_draw = results_t == 3
        terminal_values = torch.where(
            is_draw,
            torch.zeros_like(child_values),
            torch.where(root_won,
                        torch.full_like(child_values, -1.0),
                        torch.full_like(child_values,  1.0)),
        )
        child_values = torch.where(is_terminal, terminal_values, child_values)

        # Negate: Q(a) from root's perspective = -V(child)
        neg_child_values = -child_values

        # Spend extra budget on likely replies from the child state.
        # Pre-compute child legal info and reuse child_logits already in hand so
        # that _probe_child_replies does not need to re-encode / re-run the NN on
        # the same child states a second time.
        max_reply_budget = int(visits_per_action.max().item()) if total > 0 else 0
        if max_reply_budget > 1:
            child_legal_moves, child_num_legal = self.ext.generate_legal_moves_batch(
                child_states, total
            )
            child_legal_mask_int, _ = self.ext.generate_legal_mask_batch(child_states, total)
            child_legal_mask = child_legal_mask_int.bool()
            child_legal_action_indices = self.ext.legal_moves_to_actions_batch(
                child_states, child_legal_moves, child_num_legal, total,
            )
            # Apply legal mask to the logits we already have; clone to avoid
            # mutating the tensor that may be used elsewhere.
            masked_child_logits = child_logits.clone()
            masked_child_logits[~child_legal_mask] = float("-inf")

            child_reply_values = self._probe_child_replies(
                child_states,
                neg_child_values,
                results,
                gi_t,
                visits_per_action,
                child_legal_moves=child_legal_moves,
                child_num_legal=child_num_legal,
                child_legal_action_indices=child_legal_action_indices,
                child_policy_logits=masked_child_logits,
            )
        else:
            child_reply_values = neg_child_values

        # Scatter back to [B, max_k]
        result_tensor = torch.zeros(B, max_k, device="cuda")
        result_tensor[gi_t, cj_t] = child_reply_values

        return result_tensor

    def _probe_child_replies(
        self,
        child_states: torch.Tensor,
        default_root_values: torch.Tensor,
        child_results: np.ndarray,
        root_game_indices: torch.Tensor,
        visits_per_action: torch.Tensor,
        child_legal_moves: "torch.Tensor | None" = None,
        child_num_legal: "torch.Tensor | None" = None,
        child_legal_action_indices: "torch.Tensor | None" = None,
        child_policy_logits: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Probe likely replies from each non-terminal child state.

        Returns a root-perspective value per child candidate. Terminal children
        or children with no extra budget keep their default one-ply estimate.

        When pre-computed child legal info and policy logits are supplied by
        _evaluate_candidates, this method skips re-encoding and re-running the
        NN on child states (they were just evaluated one call earlier).
        """
        total = child_states.shape[0]
        root_values = default_root_values.clone()
        if total == 0:
            return root_values

        if child_legal_moves is None:
            child_legal_moves, child_num_legal = self.ext.generate_legal_moves_batch(
                child_states, total
            )
        if child_legal_action_indices is None:
            child_legal_mask_int, _ = self.ext.generate_legal_mask_batch(child_states, total)
            child_legal_mask = child_legal_mask_int.bool()
            child_legal_action_indices = self.ext.legal_moves_to_actions_batch(
                child_states, child_legal_moves, child_num_legal, total,
            )

        if child_policy_logits is None:
            child_legal_mask_int, _ = self.ext.generate_legal_mask_batch(child_states, total)
            child_legal_mask = child_legal_mask_int.bool()
            encoded = self.encoder.encode_batch(child_states, total)
            with torch.no_grad():
                cfg = self.config
                if cfg.nn_max_batch > 0 and total > cfg.nn_max_batch:
                    child_policy_logits, _ = self._nn_forward_subbatched(encoded, total)
                else:
                    child_policy_logits, _, *_ = self.net(encoded)
            child_policy_logits[~child_legal_mask] = float("-inf")

        # child_policy_logits is already masked (either passed in or computed above)
        legal_moves = child_legal_moves
        num_legal = child_num_legal
        legal_action_indices = child_legal_action_indices
        child_policy = torch.softmax(child_policy_logits, dim=-1)

        move_scores = self._gather_legal_action_scores(child_policy, legal_action_indices)
        best_move_idx = move_scores.argmax(dim=1)  # [total] — top-1 reply per parent

        # Vectorised eligibility: non-terminal, has reply budget, has legal moves,
        # and the best move has a finite score (i.e. is actually legal).
        child_results_t = torch.from_numpy(child_results).to("cuda")
        reply_budgets = (visits_per_action[root_game_indices] - 1).clamp(min=0)
        best_scores = move_scores.gather(1, best_move_idx.unsqueeze(1)).squeeze(1)
        eligible = (
            (child_results_t == 0) &
            (reply_budgets > 0) &
            (num_legal > 0) &
            best_scores.isfinite()
        )

        reply_parent_t = eligible.nonzero(as_tuple=True)[0]   # [n_replies]
        reply_move_t = best_move_idx[reply_parent_t]           # [n_replies]
        reply_parent = reply_parent_t.cpu().tolist()
        reply_move_indices = reply_move_t.cpu().tolist()

        if not reply_parent:
            return root_values

        reply_total = len(reply_parent)
        parent_idx_t = torch.tensor(reply_parent, dtype=torch.int64, device="cuda")
        grandchild_states = child_states[parent_idx_t].clone()
        move_idx_t = torch.tensor(reply_move_indices, dtype=torch.int64, device="cuda")
        moves_t = legal_moves[parent_idx_t, move_idx_t]
        self.ext.apply_moves_batch(grandchild_states, moves_t, reply_total)

        grandchild_results = self.ext.check_results_batch(
            grandchild_states, reply_total
        ).cpu().numpy()
        encoded_gc = self.encoder.encode_batch(grandchild_states, reply_total)
        with torch.no_grad():
            cfg = self.config
            if cfg.nn_max_batch > 0 and reply_total > cfg.nn_max_batch:
                _, grandchild_values = self._nn_forward_subbatched(encoded_gc, reply_total)
            else:
                _, grandchild_values, *_ = self.net(encoded_gc)
        grandchild_values = grandchild_values.squeeze(-1)

        # After the opponent reply, it is the root player's turn again.
        # Vectorised terminal override for grandchild states.
        gc_results_t = torch.from_numpy(grandchild_results).to("cuda")
        parent_idx_for_turn = torch.tensor(reply_parent, dtype=torch.int64, device="cuda")
        gc_root_turn_bytes = child_states[parent_idx_for_turn, self._OFF_TURN]
        gc_root_is_white = (gc_root_turn_bytes % 2 == 1)
        gc_root_won = (
            ((gc_results_t == 1) & gc_root_is_white) |
            ((gc_results_t == 2) & ~gc_root_is_white)
        )
        gc_is_terminal = gc_results_t != 0
        gc_is_draw = gc_results_t == 3
        gc_terminal = torch.where(
            gc_is_draw,
            torch.zeros_like(grandchild_values),
            torch.where(gc_root_won,
                        torch.full_like(grandchild_values, 1.0),
                        torch.full_like(grandchild_values, -1.0)),
        )
        grandchild_values = torch.where(gc_is_terminal, gc_terminal, grandchild_values)

        # Each parent has exactly one reply (top-1), so scatter directly.
        root_values[parent_idx_for_turn] = grandchild_values

        return root_values

    # ── Action-to-move helpers ────────────────────────────────────

    def _match_actions_to_legal_moves(
        self,
        actions: torch.Tensor,            # [B, N]
        legal_action_indices: torch.Tensor,  # [B, MAX_LEGAL_MOVES]
        num_legal: torch.Tensor,          # [B]
    ) -> torch.Tensor:
        """Return legal move indices for each action, or -1 when absent."""
        matches = legal_action_indices.unsqueeze(1).eq(actions.unsqueeze(-1))
        legal_slots = (
            torch.arange(
                legal_action_indices.shape[1], device=legal_action_indices.device,
            ).unsqueeze(0) < num_legal.unsqueeze(1)
        )
        matches &= legal_slots.unsqueeze(1)
        has_match = matches.any(dim=-1)
        move_indices = matches.float().argmax(dim=-1).to(dtype=torch.int64)
        return torch.where(
            has_match,
            move_indices,
            torch.full_like(move_indices, -1),
        )

    def _gather_legal_action_scores(
        self,
        action_scores: torch.Tensor,      # [B, A]
        legal_action_indices: torch.Tensor,  # [B, MAX_LEGAL_MOVES]
    ) -> torch.Tensor:
        """Gather full-action scores onto the legal move list."""
        safe_actions = legal_action_indices.clamp(min=0).long()
        gathered = action_scores.gather(1, safe_actions)
        gathered[legal_action_indices < 0] = float("-inf")
        return gathered

    # ── Improved policy computation ───────────────────────────────

    def _compute_improved_policy(
        self,
        logits: torch.Tensor,         # [B, A]
        topk_actions: torch.Tensor,    # [B, max_k]
        q_sums: torch.Tensor,         # [B, max_k]
        visit_counts: torch.Tensor,   # [B, max_k]
        legal_mask: torch.Tensor,     # [B, A] bool  (unused now, kept for API compat)
        root_values: torch.Tensor,    # [B]
        nn_prior_probs: torch.Tensor, # [B, A]
        B: int,
        max_k: int,
        active: list[bool],
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray] | None]]:
        """Compute improved policy targets from completed Q-values.

        Works entirely in [B, max_k] space — no [B, A] allocations.
        Returns sparse (action_indices, probs) pairs instead of dense arrays;
        _build_examples scatters them into full [A] vectors when needed.

        pi_improved(a) = softmax(logit(a) + qtransform(Q_completed(a)))

        For actions that were never visited, Q_completed = -root_value
        (the root value from the opponent's perspective = default expectation).
        """
        cfg = self.config

        # Compute Q-mean for visited candidates
        q_mean = torch.where(
            visit_counts > 0,
            q_sums / visit_counts.float(),
            torch.zeros_like(q_sums),
        )  # [B, max_k]

        max_n = visit_counts.max(dim=1, keepdim=True).values.float().clamp(min=1)

        topk_q = torch.where(
            visit_counts > 0,
            q_mean,
            (-root_values).unsqueeze(1).expand_as(q_mean),
        )
        topk_logits = logits.gather(1, topk_actions)          # [B, max_k]
        topk_qtransform = (cfg.c_visit + max_n) * cfg.c_scale * topk_q
        improved_topk = topk_logits + topk_qtransform         # [B, max_k]

        # Softmax over max_k candidates only — avoids the [B, A] allocation,
        # scatter_, legal_mask broadcast, and large softmax that dominated runtime.
        # Padding slots (topk_logits = -inf) get 0 probability automatically.
        improved_probs = torch.softmax(improved_topk, dim=-1)  # [B, max_k]

        # Gather nn_prior at topk positions only — [B, max_k] transfer vs [B, A]
        nn_prior_topk = nn_prior_probs.gather(1, topk_actions)  # [B, max_k]

        # Single small transfer: [B, max_k] × 3 tensors  (vs [B, A] × 2 before)
        actions_np    = topk_actions.cpu().numpy()    # [B, max_k] int64
        probs_np      = improved_probs.cpu().numpy()  # [B, max_k] float32
        nn_prior_np   = nn_prior_topk.cpu().numpy()   # [B, max_k] float32

        policies: list[tuple[np.ndarray, np.ndarray]] = []
        nn_priors: list[tuple[np.ndarray, np.ndarray] | None] = []
        for i in range(B):
            act_i = actions_np[i]   # [max_k] — action indices
            prob_i = probs_np[i]    # [max_k] — probabilities
            policies.append((act_i, prob_i))
            if active[i]:
                nn_priors.append((act_i, nn_prior_np[i]))
            else:
                nn_priors.append(None)

        return policies, nn_priors

    # ── Win-in-one override ───────────────────────────────────────

    def _check_immediate_wins(
        self,
        root_states: torch.Tensor,
        B: int,
        active: list[bool],
        current_turns: list[int],
    ) -> dict[int, int]:
        """Find any immediate wins and return {game_idx: win_action_index}.

        Fully batched: one apply_moves_batch + one check_results_batch across
        all games, replacing the previous per-game kernel dispatch loop.
        Callers apply the override to whichever policy representation they use.
        """
        legal_moves, num_legal = self.ext.generate_legal_moves_batch(root_states, B)
        legal_action_indices = self.ext.legal_moves_to_actions_batch(
            root_states, legal_moves, num_legal, B,
        )
        nlegal_np = num_legal.cpu().numpy()

        # Build flat index tensors across all active games in one pass
        gi_flat: list[int] = []
        mi_flat: list[int] = []
        game_nlegal: list[int] = []    # number of moves per game (0 if inactive)

        for i in range(B):
            nl = int(nlegal_np[i]) if active[i] else 0
            game_nlegal.append(nl)
            for m in range(nl):
                gi_flat.append(i)
                mi_flat.append(m)

        total = len(gi_flat)
        if total == 0:
            return {}

        gi_t = torch.tensor(gi_flat, dtype=torch.int64, device="cuda")
        mi_t = torch.tensor(mi_flat, dtype=torch.int64, device="cuda")

        # Single batched apply + check — two kernel launches, one CPU sync
        test_states = root_states[gi_t].clone()
        flat_moves = legal_moves[gi_t, mi_t]
        self.ext.apply_moves_batch(test_states, flat_moves, total)
        results_np = self.ext.check_results_batch(test_states, total).cpu().numpy()

        # Scatter-reduce: find first winning move per game
        win_overrides: dict[int, int] = {}
        flat_idx = 0
        for i in range(B):
            nl = game_nlegal[i]
            if nl > 0:
                win_result = 1 if (current_turns[i] % 2 == 0) else 2
                for mi in range(nl):
                    if results_np[flat_idx + mi] == win_result:
                        win_overrides[i] = int(legal_action_indices[i, mi].item())
                        break
            flat_idx += nl

        return win_overrides

    # ── Action → move bytes lookup ────────────────────────────────

    def _action_to_gpu_move(
        self,
        states: torch.Tensor,
        game_idx: int,
        action: int,
    ) -> np.ndarray | None:
        """Find the GPU move bytes for a given action index."""
        single = states[game_idx:game_idx + 1]
        moves_t, num_legal = self.ext.generate_legal_moves_batch(single, 1)
        legal_action_indices = self.ext.legal_moves_to_actions_batch(
            single, moves_t, num_legal, 1,
        )
        move_indices = (legal_action_indices[0] == action).nonzero(as_tuple=False)
        nl = int(num_legal[0].item())
        if move_indices.numel() > 0:
            return moves_t[0, int(move_indices[0, 0].item())].cpu().numpy().copy()
        if nl > 0:
            return moves_t[0, 0].cpu().numpy().copy()
        return None

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

    # ── Utility methods ───────────────────────────────────────────

    _BOARD_SIZE = 23
    _NUM_CELLS = _BOARD_SIZE * _BOARD_SIZE  # 529
    _DIR_DCOL = [+1, +1, 0, -1, -1, 0]
    _DIR_DROW = [0, -1, -1, 0, +1, +1]
    _MAX_STACK = 5
    _OFF_HEIGHT = _MAX_STACK * _NUM_CELLS   # 2645
    _OFF_QUEEN_CELL = 3392
    _OFF_TURN = 3412

    def _get_turns(self, states: torch.Tensor, B: int) -> list[int]:
        off = self._OFF_TURN
        lo = states[:B, off].to(dtype=torch.int32)
        hi = states[:B, off + 1].to(dtype=torch.int32)
        return (lo | (hi << 8)).cpu().tolist()

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
        qp_scale = self.config.queen_pressure_scale
        A = self._action_space_size
        all_examples = []
        for i, history in enumerate(histories):
            result = int(final_results[i])
            examples = []
            num_steps = len(history)

            draw_value_white = 0.0
            if (result == 0 or result == 3) and qp_scale > 0.0:
                qs_target_final, qs_mask_final = final_qs_data[i]
                white_q_surrounded = float(qs_target_final[:, 0].sum()) if qs_mask_final[0] > 0 else 0.0
                black_q_surrounded = float(qs_target_final[:, 1].sum()) if qs_mask_final[1] > 0 else 0.0
                draw_value_white = qp_scale * (black_q_surrounded - white_q_surrounded) / 6.0

            for step_idx, (graph, policy_sparse, turn, mobility, seq, nn_prior_sparse) in enumerate(history):
                if result == 0 or result == 3:
                    if draw_value_white != 0.0:
                        player_is_white = (turn % 2 == 0)
                        value = draw_value_white if player_is_white else -draw_value_white
                    else:
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

                # Reconstruct dense [A] policy from sparse (action_indices, probs).
                # Only the max_k (≤16) non-zero entries need scattering.
                pol_actions, pol_probs = policy_sparse
                policy = np.zeros(A, dtype=np.float32)
                policy[pol_actions] = pol_probs

                # Reconstruct sparse nn_prior into full [A] vector.
                # _compute_surprise_weight only reads positions where policy > 0,
                # which are exactly pol_actions — so only those slots need values.
                nn_prior: np.ndarray | None = None
                if nn_prior_sparse is not None:
                    nn_prior_actions, nn_prior_vals = nn_prior_sparse
                    nn_prior = np.zeros(A, dtype=np.float32)
                    nn_prior[nn_prior_actions] = nn_prior_vals

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
