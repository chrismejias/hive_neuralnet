"""
Benchmark a single training iteration with per-step timing.

Usage:
    python benchmark_iteration.py

Parameters: 256 games, 128 simulations, 32 Gumbel considered, base game only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np

# ── Timing helpers ────────────────────────────────────────────────────────────

def cuda_sync_time() -> float:
    """Return wall time after a CUDA sync so timings are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@dataclass
class Timings:
    """Accumulator for named durations."""
    buckets: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, dt: float) -> None:
        self.buckets[key] = self.buckets.get(key, 0.0) + dt
        self.counts[key] = self.counts.get(key, 0) + 1

    def report(self, title: str = "Timing Report") -> None:
        if not self.buckets:
            print(f"\n[{title}] — no data collected\n")
            return
        total = sum(self.buckets.values())
        w = max(len(k) for k in self.buckets)
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
        print(f"  {'Step':<{w}}   {'Total':>9}   {'Calls':>6}   {'Avg':>9}   {'Share':>7}")
        print(f"  {'-'*w}   {'-'*9}   {'-'*6}   {'-'*9}   {'-'*7}")
        for k, v in self.buckets.items():
            n = self.counts[k]
            share = 100.0 * v / total if total > 0 else 0.0
            print(f"  {k:<{w}}   {v:>8.2f}s   {n:>6}   {v/n:>8.3f}s   {share:>6.1f}%")
        print(f"  {'-'*w}   {'-'*9}")
        print(f"  {'TOTAL':<{w}}   {total:>8.2f}s")
        print(f"{'='*65}\n")


TIMINGS = Timings()

# ── Patch GumbelAlphaZeroOrchestrator to instrument hot paths ─────────────────

import hive_gpu
from hive_gpu.gumbel_mcts import GumbelAlphaZeroOrchestrator, GumbelConfig

_orig_gumbel_search = GumbelAlphaZeroOrchestrator._gumbel_search
_orig_evaluate_candidates = GumbelAlphaZeroOrchestrator._evaluate_candidates
_orig_self_play_batch = GumbelAlphaZeroOrchestrator.self_play_batch


def _timed_self_play_batch(self, start_states=None):
    t0 = cuda_sync_time()
    result = _orig_self_play_batch(self, start_states=start_states)
    TIMINGS.add("self_play / total", cuda_sync_time() - t0)
    return result


def _timed_gumbel_search(self, root_states, B, active, move_numbers):
    cfg = self.config
    A = self._action_space_size

    # --- encode ---
    t0 = cuda_sync_time()
    if hasattr(self.encoder, "encode_batch_with_graphs_and_seqs"):
        encoded, root_graphs, root_seqs = self.encoder.encode_batch_with_graphs_and_seqs(root_states, B)
    elif hasattr(self.encoder, "encode_batch_with_graphs"):
        encoded, root_graphs = self.encoder.encode_batch_with_graphs(root_states, B)
        root_seqs = [None] * B
    else:
        encoded = self.encoder.encode_batch(root_states, B)
        root_graphs = [None] * B
        root_seqs = [None] * B
    TIMINGS.add("gumbel / encode_root", cuda_sync_time() - t0)

    # --- legal mask ---
    t0 = cuda_sync_time()
    legal_mask_int, _ = self.ext.generate_legal_mask_batch(root_states, B)
    legal_mask = legal_mask_int.bool()
    TIMINGS.add("gumbel / legal_mask", cuda_sync_time() - t0)

    # --- root NN forward ---
    t0 = cuda_sync_time()
    with torch.no_grad():
        if cfg.nn_max_batch > 0 and B > cfg.nn_max_batch:
            policy_logits, root_values = self._nn_forward_subbatched(encoded, B)
        else:
            policy_logits, root_values, *_ = self.net(encoded)
    TIMINGS.add("gumbel / root_nn_forward", cuda_sync_time() - t0)

    root_values = root_values.squeeze(-1)
    policy_logits[~legal_mask] = float("-inf")
    nn_prior_probs = torch.softmax(policy_logits, dim=-1)

    # --- Gumbel noise + top-k ---
    t0 = cuda_sync_time()
    u = torch.rand(B, A, device="cuda").clamp(1e-10, 1.0 - 1e-7)
    gumbel = -torch.log(-torch.log(u))
    gumbel[~legal_mask] = float("-inf")
    perturbed = gumbel + policy_logits
    num_legal = legal_mask.sum(dim=1)
    k = cfg.max_num_considered_actions
    effective_k = torch.clamp(num_legal, max=k)
    max_k = min(k, int(num_legal.max().item()))
    TIMINGS.add("gumbel / gumbel_noise_topk_prep", cuda_sync_time() - t0)

    if max_k == 0:
        policies = [np.zeros(A, dtype=np.float32) for _ in range(B)]
        nn_priors = [nn_prior_probs[i].cpu().numpy() for i in range(B)]
        return policies, root_graphs, root_seqs, nn_priors

    t0 = cuda_sync_time()
    topk_scores, topk_actions = torch.topk(perturbed, max_k, dim=1)
    TIMINGS.add("gumbel / topk_select", cuda_sync_time() - t0)

    # --- sequential halving setup ---
    q_sums = torch.zeros(B, max_k, device="cuda")
    visit_counts = torch.zeros(B, max_k, dtype=torch.int32, device="cuda")
    candidate_mask = torch.ones(B, max_k, dtype=torch.bool, device="cuda")

    for i in range(B):
        ek = int(effective_k[i].item())
        if ek < max_k:
            candidate_mask[i, ek:] = False

    num_candidates_initial = int(candidate_mask.sum(dim=1).max().item())
    num_rounds = max(1, math.ceil(math.log2(num_candidates_initial)))

    t0 = cuda_sync_time()
    legal_moves, num_legal_moves = self.ext.generate_legal_moves_batch(root_states, B)
    legal_action_indices = self.ext.legal_moves_to_actions_batch(
        root_states, legal_moves, num_legal_moves, B,
    )
    TIMINGS.add("gumbel / gen_legal_moves", cuda_sync_time() - t0)

    remaining_budget = torch.full((B,), cfg.num_simulations, dtype=torch.int32, device="cuda")

    # --- halving rounds ---
    for round_idx in range(num_rounds):
        n_remaining = candidate_mask.sum(dim=1).float()
        max_remaining = int(n_remaining.max().item())
        if max_remaining == 0:
            break

        n_remaining_i = candidate_mask.sum(dim=1).int()
        visits_this_round = torch.where(
            n_remaining_i > 0,
            torch.clamp(remaining_budget // n_remaining_i.clamp(min=1), min=1),
            torch.zeros_like(remaining_budget),
        )
        if int(visits_this_round.max().item()) == 0:
            break

        t0 = cuda_sync_time()
        child_values = _timed_evaluate_candidates(
            self, root_states, B, topk_actions, candidate_mask,
            visits_this_round, legal_moves, num_legal_moves, legal_action_indices,
        )
        TIMINGS.add(f"gumbel / halving_round[{round_idx}]_eval", cuda_sync_time() - t0)

        visits_per_candidate = visits_this_round.unsqueeze(1).float()
        q_sums += child_values * visits_per_candidate * candidate_mask.float()
        visit_counts += visits_this_round.unsqueeze(1) * candidate_mask.int()
        remaining_budget = torch.clamp(
            remaining_budget - visits_this_round * n_remaining_i, min=0,
        )

        if round_idx < num_rounds - 1:
            t0 = cuda_sync_time()
            max_n = visit_counts.max(dim=1, keepdim=True).values.float().clamp(min=1)
            q_mean = torch.where(
                visit_counts > 0,
                q_sums / visit_counts.float(),
                torch.zeros_like(q_sums),
            )
            qtransform = (cfg.c_visit + max_n) * cfg.c_scale * q_mean
            topk_logits = policy_logits.gather(1, topk_actions)
            topk_gumbel = gumbel.gather(1, topk_actions)
            sigma = topk_gumbel + topk_logits + qtransform
            sigma[~candidate_mask] = float("-inf")
            n_keep = (n_remaining / 2).ceil().int().clamp(min=1)
            sorted_idx = sigma.argsort(dim=1, descending=True)
            inv_rank = torch.empty_like(sorted_idx)
            inv_rank.scatter_(
                1, sorted_idx,
                torch.arange(max_k, device="cuda").unsqueeze(0).expand(B, -1),
            )
            candidate_mask = (inv_rank < n_keep.unsqueeze(1)) & candidate_mask
            TIMINGS.add(f"gumbel / halving_round[{round_idx}]_prune", cuda_sync_time() - t0)

    # --- improved policy ---
    t0 = cuda_sync_time()
    policies_np, nn_priors_list = self._compute_improved_policy(
        policy_logits, topk_actions, q_sums, visit_counts,
        legal_mask, root_values, nn_prior_probs, B, max_k, active,
    )
    TIMINGS.add("gumbel / improved_policy", cuda_sync_time() - t0)

    return policies_np, root_graphs, root_seqs, nn_priors_list, legal_moves, legal_action_indices, num_legal_moves


def _timed_evaluate_candidates(
    self, root_states, B, topk_actions, candidate_mask,
    visits_per_action, legal_moves, num_legal_moves, legal_action_indices,
):
    max_k = topk_actions.shape[1]

    t0 = cuda_sync_time()
    matched_moves = self._match_actions_to_legal_moves(
        topk_actions, legal_action_indices, num_legal_moves,
    )
    active_candidates = candidate_mask & (matched_moves >= 0)
    candidate_pairs = torch.nonzero(active_candidates, as_tuple=False)
    TIMINGS.add("eval_cands / match_actions", cuda_sync_time() - t0)

    if candidate_pairs.numel() == 0:
        return torch.zeros(B, max_k, device="cuda")

    gi_t = candidate_pairs[:, 0].to(dtype=torch.int64)
    cj_t = candidate_pairs[:, 1].to(dtype=torch.int64)
    mi_t = matched_moves[gi_t, cj_t].to(dtype=torch.int64)
    total = int(candidate_pairs.shape[0])

    t0 = cuda_sync_time()
    child_states = root_states[gi_t].clone()
    moves = legal_moves[gi_t, mi_t]
    self.ext.apply_moves_batch(child_states, moves, total)
    TIMINGS.add("eval_cands / clone_apply_moves", cuda_sync_time() - t0)

    t0 = cuda_sync_time()
    results_t = self.ext.check_results_batch(child_states, total)
    results = results_t.cpu().numpy()
    TIMINGS.add("eval_cands / check_results", cuda_sync_time() - t0)

    t0 = cuda_sync_time()
    encoded = self.encoder.encode_batch(child_states, total)
    TIMINGS.add("eval_cands / encode_children", cuda_sync_time() - t0)

    t0 = cuda_sync_time()
    with torch.no_grad():
        cfg = self.config
        if cfg.nn_max_batch > 0 and total > cfg.nn_max_batch:
            child_logits, child_values = self._nn_forward_subbatched(encoded, total)
        else:
            child_logits, child_values, *_ = self.net(encoded)
    TIMINGS.add("eval_cands / child_nn_forward", cuda_sync_time() - t0)

    child_values = child_values.squeeze(-1)

    root_turn_bytes = root_states[gi_t, self._OFF_TURN]
    root_is_white = (root_turn_bytes % 2 == 0)
    root_won = ((results_t == 1) & root_is_white) | ((results_t == 2) & ~root_is_white)
    is_terminal = results_t != 0
    is_draw = results_t == 3
    terminal_values = torch.where(
        is_draw,
        torch.zeros_like(child_values),
        torch.where(root_won,
                    torch.full_like(child_values, -1.0),
                    torch.full_like(child_values, 1.0)),
    )
    child_values = torch.where(is_terminal, terminal_values, child_values)
    neg_child_values = -child_values

    max_reply_budget = int(visits_per_action.max().item()) if total > 0 else 0
    if max_reply_budget > 1:
        t0 = cuda_sync_time()
        child_legal_moves, child_num_legal = self.ext.generate_legal_moves_batch(
            child_states, total
        )
        child_legal_mask_int, _ = self.ext.generate_legal_mask_batch(child_states, total)
        child_legal_mask = child_legal_mask_int.bool()
        child_legal_action_indices = self.ext.legal_moves_to_actions_batch(
            child_states, child_legal_moves, child_num_legal, total,
        )
        masked_child_logits = child_logits.clone()
        masked_child_logits[~child_legal_mask] = float("-inf")
        TIMINGS.add("eval_cands / probe_precompute", cuda_sync_time() - t0)

        t0 = cuda_sync_time()
        child_reply_values = self._probe_child_replies(
            child_states, neg_child_values, results, gi_t, visits_per_action,
            child_legal_moves=child_legal_moves,
            child_num_legal=child_num_legal,
            child_legal_action_indices=child_legal_action_indices,
            child_policy_logits=masked_child_logits,
        )
        TIMINGS.add("eval_cands / probe_replies", cuda_sync_time() - t0)
    else:
        child_reply_values = neg_child_values

    result_tensor = torch.zeros(B, max_k, device="cuda")
    result_tensor[gi_t, cj_t] = child_reply_values
    return result_tensor


# Install patches
GumbelAlphaZeroOrchestrator.self_play_batch = _timed_self_play_batch
GumbelAlphaZeroOrchestrator._gumbel_search = _timed_gumbel_search
GumbelAlphaZeroOrchestrator._evaluate_candidates = _timed_evaluate_candidates


# ── Main benchmark ─────────────────────────────────────────────────────────────

def main() -> None:
    from hive_transformer.transformer_net import TransformerConfig, HiveTransformer
    from hive_gpu.gpu_trainer import GPUTrainer, GPUTrainConfig

    print("Setting up trainer...")
    train_config = GPUTrainConfig(
        num_iterations=1,
        games_per_batch=256,
        mcts_simulations=128,
        gumbel_max_considered=32,
        use_gumbel=True,
        expansion_mask=0,       # base game only
        encoder_type="transformer",
        num_epochs=5,
        batch_size=64,
        checkpoint_dir="/tmp/bench_checkpoints",
        skip_arena=True,
    )
    net_config = TransformerConfig.small()
    trainer = GPUTrainer(config=train_config, net_config=net_config, net_class=HiveTransformer)

    # Warm up CUDA
    print("Warming up CUDA...")
    dummy = torch.zeros(1, 1, device=trainer.device)
    del dummy
    torch.cuda.synchronize()

    print("\nRunning benchmark iteration (256 games, 128 sims, k=32, base game)...\n")

    # ── Self-play ──
    t_sp0 = cuda_sync_time()
    trainer.best_net.eval()
    new_examples, sp_stats = trainer._self_play_phase(iteration=1)
    t_sp1 = cuda_sync_time()
    sp_time = t_sp1 - t_sp0

    print(f"Self-play done: {len(new_examples)} examples from "
          f"{sp_stats['num_games']} games "
          f"(W:{sp_stats['white_wins']} B:{sp_stats['black_wins']} "
          f"D:{sp_stats['draws']}) in {sp_time:.2f}s")

    trainer.buffer.add_examples(new_examples)
    torch.cuda.empty_cache()

    # ── Training ──
    t_tr0 = cuda_sync_time()
    new_net, train_loss, loss_components = trainer._train_phase(iteration=1)
    t_tr1 = cuda_sync_time()
    train_time = t_tr1 - t_tr0

    print(f"Training done: loss={train_loss:.4f} in {train_time:.2f}s")

    # ── Checkpoint ──
    t_ck0 = cuda_sync_time()
    trainer._save_checkpoint(new_net, iteration=1)
    t_ck1 = cuda_sync_time()
    ck_time = t_ck1 - t_ck0

    print(f"Checkpoint saved in {ck_time:.3f}s")

    # ── Top-level summary ──────────────────────────────────────────────────────
    total_time = sp_time + train_time + ck_time
    top_level = Timings()
    top_level.add("1. self_play", sp_time)
    top_level.add("2. training", train_time)
    top_level.add("3. checkpoint", ck_time)
    top_level.report("Top-level phase timing")

    # ── Detailed gumbel timing ─────────────────────────────────────────────────
    # Separate gumbel internals from eval_cands internals
    gumbel_keys = [k for k in TIMINGS.buckets if k.startswith("gumbel /") or k == "self_play / total"]
    cand_keys   = [k for k in TIMINGS.buckets if k.startswith("eval_cands /")]

    gumbel_t = Timings()
    for k in gumbel_keys:
        gumbel_t.buckets[k] = TIMINGS.buckets[k]
        gumbel_t.counts[k]  = TIMINGS.counts[k]
    gumbel_t.report("Gumbel search internals (cumulative across all moves)")

    cand_t = Timings()
    for k in cand_keys:
        cand_t.buckets[k] = TIMINGS.buckets[k]
        cand_t.counts[k]  = TIMINGS.counts[k]
    cand_t.report("Candidate evaluation internals (cumulative across all rounds)")

    print(f"Total iteration wall time: {total_time:.2f}s")
    print(f"Throughput: {sp_stats['num_games'] / sp_time:.1f} games/s  |  "
          f"{len(new_examples) / sp_time:.0f} examples/s (self-play)")


if __name__ == "__main__":
    main()
