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

    # Policy target pruning
    policy_target_pruning: float = 0.02

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
            policies, graphs, seqs, nn_priors = self._gumbel_search(
                states, B, active, move_numbers,
            )

            # Compute per-piece mobility for history
            mob_tensor, mob_counts = self.ext.compute_mobility_batch(states, B, False)
            mob_np = mob_tensor.cpu().numpy()
            mob_board_counts = mob_counts.cpu().numpy()

            # Override policies for immediate wins
            self._check_immediate_wins(states, B, active, current_turns, policies)

            # Record history and select actions
            action_move_bytes = np.zeros((B, self._move_size), dtype=np.uint8)

            for i in range(B):
                if not active[i]:
                    continue

                policy = policies[i]
                turn = current_turns[i]

                mob_i = mob_np[i, :mob_board_counts[i]].copy()
                histories[i].append(
                    (graphs[i], policy.copy(), turn, mob_i, seqs[i], nn_priors[i])
                )

                # Action selection: greedy after temp_drop, else sample
                if move_numbers[i] >= cfg.temperature_drop_move:
                    action = int(np.argmax(policy))
                else:
                    psum = policy.sum()
                    if psum > 0 and np.isfinite(psum):
                        p = policy / psum
                    else:
                        mask = policy > 0
                        p = np.zeros_like(policy)
                        if mask.any():
                            p[mask] = 1.0 / mask.sum()
                        else:
                            p[:] = 1.0 / len(p)
                    action = int(np.random.choice(len(p), p=p))

                # Look up move bytes for selected action
                fb = self._action_to_gpu_move(states, i, action)
                if fb is not None:
                    action_move_bytes[i] = fb

            # Apply moves
            moves_t = torch.from_numpy(action_move_bytes).cuda()
            self.ext.apply_moves_batch(states, moves_t, B)

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

                # For each game, keep the top n_keep[i] candidates
                sorted_idx = sigma.argsort(dim=1, descending=True)  # [B, max_k]
                new_candidate_mask = torch.zeros_like(candidate_mask)
                for i in range(B):
                    if not active[i]:
                        continue
                    nk = int(n_keep[i].item())
                    keep_indices = sorted_idx[i, :nk]
                    new_candidate_mask[i, keep_indices] = True
                candidate_mask = new_candidate_mask

        # ── Step 5: Compute improved policy ──
        policies_np, nn_priors_list = self._compute_improved_policy(
            policy_logits, topk_actions, q_sums, visit_counts,
            legal_mask, root_values, nn_prior_probs, B, max_k, active,
        )

        return policies_np, root_graphs, root_seqs, nn_priors_list

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

        # Check for terminal states
        results = self.ext.check_results_batch(child_states, total).cpu().numpy()

        # Encode child states and run NN
        encoded = self.encoder.encode_batch(child_states, total)
        with torch.no_grad():
            cfg = self.config
            if cfg.nn_max_batch > 0 and total > cfg.nn_max_batch:
                child_logits, child_values = self._nn_forward_subbatched(encoded, total)
            else:
                child_logits, child_values, *_ = self.net(encoded)

        child_values = child_values.squeeze(-1)  # [total]

        # For terminal states, use the game result instead of NN value
        # Results: 0=in_progress, 1=white_wins, 2=black_wins, 3=draw
        for k in range(total):
            r = results[k]
            if r == 0:
                continue
            elif r == 3:
                child_values[k] = 0.0
            else:
                # Terminal value from child's perspective:
                # The child state is after root player moved.
                # We need to read whose turn it is in the child state.
                # Simpler: r==1 means white won, r==2 means black won.
                # The root player just moved, so if it's now opponent's turn,
                # result favoring root player → child_value = -1 (bad for child).
                gi = int(gi_t[k].item())
                root_turn_byte = root_states[gi, self._OFF_TURN].item()
                root_is_white = (root_turn_byte % 2 == 0)
                root_won = (r == 1 and root_is_white) or (r == 2 and not root_is_white)
                child_values[k] = -1.0 if root_won else 1.0

        # Negate: Q(a) from root's perspective = -V(child)
        neg_child_values = -child_values

        # Spend extra budget on likely replies from the child state.
        max_reply_budget = int(visits_per_action.max().item()) if total > 0 else 0
        if max_reply_budget > 1:
            child_reply_values = self._probe_child_replies(
                child_states,
                neg_child_values,
                results,
                gi_t,
                visits_per_action,
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
    ) -> torch.Tensor:
        """Probe likely replies from each non-terminal child state.

        Returns a root-perspective value per child candidate. Terminal children
        or children with no extra budget keep their default one-ply estimate.
        """
        total = child_states.shape[0]
        root_values = default_root_values.clone()
        if total == 0:
            return root_values

        legal_moves, num_legal = self.ext.generate_legal_moves_batch(child_states, total)
        legal_mask_int, _ = self.ext.generate_legal_mask_batch(child_states, total)
        legal_mask = legal_mask_int.bool()
        legal_action_indices = self.ext.legal_moves_to_actions_batch(
            child_states, legal_moves, num_legal, total,
        )

        encoded = self.encoder.encode_batch(child_states, total)
        with torch.no_grad():
            cfg = self.config
            if cfg.nn_max_batch > 0 and total > cfg.nn_max_batch:
                child_policy_logits, _ = self._nn_forward_subbatched(encoded, total)
            else:
                child_policy_logits, _, *_ = self.net(encoded)
        child_policy_logits[~legal_mask] = float("-inf")
        child_policy = torch.softmax(child_policy_logits, dim=-1)

        num_legal_cpu = num_legal.cpu()
        root_game_indices_cpu = root_game_indices.cpu()
        reply_parent: list[int] = []
        reply_move_indices: list[int] = []
        move_scores = self._gather_legal_action_scores(child_policy, legal_action_indices)
        ranked_moves = torch.argsort(move_scores, dim=1, descending=True)

        for parent_idx in range(total):
            if child_results[parent_idx] != 0:
                continue
            root_game_idx = int(root_game_indices_cpu[parent_idx].item())
            reply_budget = max(0, int(visits_per_action[root_game_idx].item()) - 1)
            nl = int(num_legal_cpu[parent_idx].item())
            if reply_budget <= 0 or nl == 0:
                continue
            keep = min(reply_budget, nl)
            for k in range(keep):
                move_idx = int(ranked_moves[parent_idx, k].item())
                if move_idx >= nl or move_scores[parent_idx, move_idx].item() == float("-inf"):
                    break
                reply_parent.append(parent_idx)
                reply_move_indices.append(move_idx)

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
        for k in range(reply_total):
            r = grandchild_results[k]
            if r == 0:
                continue
            elif r == 3:
                grandchild_values[k] = 0.0
            else:
                root_turn_byte = child_states[
                    reply_parent[k], self._OFF_TURN
                ].item()
                root_is_white = (root_turn_byte % 2 == 1)
                root_won = (r == 1 and root_is_white) or (r == 2 and not root_is_white)
                grandchild_values[k] = 1.0 if root_won else -1.0

        # Opponent selects the reply that is worst for the root player.
        for parent_idx in set(reply_parent):
            vals = grandchild_values[
                torch.tensor(
                    [i for i, p in enumerate(reply_parent) if p == parent_idx],
                    dtype=torch.int64,
                    device=grandchild_values.device,
                )
            ]
            root_values[parent_idx] = torch.min(vals)

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
        legal_mask: torch.Tensor,     # [B, A] bool
        root_values: torch.Tensor,    # [B]
        nn_prior_probs: torch.Tensor, # [B, A]
        B: int,
        max_k: int,
        active: list[bool],
    ) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        """Compute improved policy targets from completed Q-values.

        pi_improved(a) = softmax(logit(a) + qtransform(Q_completed(a)))

        For actions that were never visited, Q_completed = -root_value
        (the root value from the opponent's perspective = default expectation).
        """
        cfg = self.config
        A = self._action_space_size

        # Compute Q-mean for visited candidates
        q_mean = torch.where(
            visit_counts > 0,
            q_sums / visit_counts.float(),
            torch.zeros_like(q_sums),
        )  # [B, max_k]

        max_n = visit_counts.max(dim=1, keepdim=True).values.float().clamp(min=1)

        # Restrict policy improvement to the searched candidate set rather than
        # leaking probability mass back onto the full legal action space.
        improved_logits = torch.full_like(logits, float("-inf"))
        topk_q = torch.where(
            visit_counts > 0,
            q_mean,
            (-root_values).unsqueeze(1).expand_as(q_mean),
        )
        topk_logits = logits.gather(1, topk_actions)
        topk_qtransform = (cfg.c_visit + max_n) * cfg.c_scale * topk_q
        improved_topk = topk_logits + topk_qtransform
        improved_logits.scatter_(1, topk_actions, improved_topk)
        improved_logits[~legal_mask] = float("-inf")

        # Softmax → improved policy
        improved_policy = torch.softmax(improved_logits, dim=-1)  # [B, A]

        # Apply policy target pruning
        if cfg.policy_target_pruning > 0:
            max_prob = improved_policy.max(dim=1, keepdim=True).values
            threshold = cfg.policy_target_pruning * max_prob
            improved_policy[improved_policy < threshold] = 0.0
            # Renormalize
            sums = improved_policy.sum(dim=1, keepdim=True).clamp(min=1e-8)
            improved_policy = improved_policy / sums

        # Convert to numpy
        improved_np = improved_policy.cpu().numpy()
        nn_prior_np = nn_prior_probs.cpu().numpy()

        policies = []
        nn_priors = []
        for i in range(B):
            policies.append(improved_np[i].copy())
            if active[i]:
                nn_priors.append(nn_prior_np[i].copy())
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
        policies: list[np.ndarray],
    ) -> None:
        """Override policies with probability 1 for any immediate win.

        Fully batched: one apply_moves_batch + one check_results_batch across
        all games, replacing the previous per-game kernel dispatch loop.
        """
        legal_moves, num_legal = self.ext.generate_legal_moves_batch(root_states, B)
        legal_action_indices = self.ext.legal_moves_to_actions_batch(
            root_states, legal_moves, num_legal, B,
        )
        nlegal_np = num_legal.cpu().numpy()

        # Build flat index tensors across all active games in one pass
        gi_flat: list[int] = []
        mi_flat: list[int] = []
        game_offsets: list[int] = []   # start index in flat array for each game
        game_nlegal: list[int] = []    # number of moves per game (0 if inactive)

        for i in range(B):
            game_offsets.append(len(gi_flat))
            nl = int(nlegal_np[i]) if active[i] else 0
            game_nlegal.append(nl)
            for m in range(nl):
                gi_flat.append(i)
                mi_flat.append(m)

        total = len(gi_flat)
        if total == 0:
            return

        gi_t = torch.tensor(gi_flat, dtype=torch.int64, device="cuda")
        mi_t = torch.tensor(mi_flat, dtype=torch.int64, device="cuda")

        # Single batched apply + check — two kernel launches, one CPU sync
        test_states = root_states[gi_t].clone()
        flat_moves = legal_moves[gi_t, mi_t]
        self.ext.apply_moves_batch(test_states, flat_moves, total)
        results_np = self.ext.check_results_batch(test_states, total).cpu().numpy()

        # Scatter-reduce: find first winning move per game
        flat_idx = 0
        for i in range(B):
            nl = game_nlegal[i]
            if nl == 0:
                flat_idx += nl
                continue
            win_result = 1 if (current_turns[i] % 2 == 0) else 2
            for mi in range(nl):
                if results_np[flat_idx + mi] == win_result:
                    action = int(legal_action_indices[i, mi].item())
                    if 0 <= action < len(policies[i]):
                        policies[i][:] = 0.0
                        policies[i][action] = 1.0
                    break
            flat_idx += nl

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

            for step_idx, (graph, policy, turn, mobility, seq, nn_prior) in enumerate(history):
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
