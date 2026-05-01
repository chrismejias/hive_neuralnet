"""
Verify the prior-anchored policy target: measure improved policy mass
distribution with the new approach vs the old v_pi approach.

Usage:
  python3.10 fnn_mass_new_policy.py \
      --checkpoint checkpoints_fnn/hive_fnn_checkpoint_0200.pt \
      --positions 50
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")
import hive_gpu
from hive_fnn.fnn_mcts_orchestrator import (
    FNNMCTSConfig, FNNMCTSOrchestrator,
    _GUMBEL_K, _GUMBEL_ROUNDS, _GUMBEL_WAVE_SCHEDULE,
)
from hive_fnn.fnn_network import FNNConfig, HiveFNN


def load_net(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = HiveFNN(FNNConfig.large())
    net.load_state_dict(ckpt["model_state_dict"])
    return net.cuda().eval()


def build_orch(net, num_sims):
    cfg = FNNMCTSConfig(
        num_simulations=num_sims,
        batch_size=1,
        wave_parallel=True,
        wave_size=4,
        deterministic_non_root=True,
        expansion_mask=0,
        rebase_tree_each_move=False,
    )
    return FNNMCTSOrchestrator(net, cfg)


def make_position(ext, warmup_moves):
    states = ext.create_initial_states(1, 0)
    for _ in range(warmup_moves):
        lm, nl = ext.generate_legal_moves_batch(states, 1)
        n = int(nl[0].item())
        if n == 0:
            break
        mv = torch.zeros(1, ext.SIZEOF_GPU_MOVE, dtype=torch.uint8, device="cuda")
        mv[0] = lm[0, np.random.randint(0, n)]
        ext.apply_moves_batch(states, mv, 1)
    return states


def run_and_measure(orch, state, num_sims):
    dev = "cuda"
    B = 1
    cfg = orch.config
    ext = orch.ext
    MAX_L = orch._max_legal

    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    legal_moves, num_legal, root_features = (
        ext.generate_legal_moves_and_fnn_features_batch(state, B)
    )
    n_legal = int(num_legal[0].item())
    if n_legal == 0:
        return None

    priors, root_vals, child_q_init = orch._eval_states(
        state, legal_moves, num_legal, B, root_features
    )
    v_pi = float(root_vals[0].item())
    prior_np = priors[0, :n_legal].float().cpu().numpy()

    game_active = torch.ones(B, dtype=torch.int8, device=dev)
    orch._expand_root_if_needed(
        tree, state, legal_moves, num_legal, priors, child_q_init, game_active, B
    )

    valid_slot = (
        torch.arange(MAX_L, device=dev).unsqueeze(0)
        < num_legal.to(torch.int64).unsqueeze(1)
    )
    legal_logits = torch.where(
        valid_slot, priors.clamp(min=1e-20).log(), torch.full_like(priors, -1e30)
    )

    u = torch.rand(B, MAX_L, device=dev).clamp(1e-4, 1 - 1e-4)
    gumbel = -torch.log(-torch.log(u))
    gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
    perturbed = gumbel + legal_logits

    actual_k = min(_GUMBEL_K, n_legal)
    _, topk_slots = torch.topk(perturbed, actual_k, dim=1)
    candidate_slots = topk_slots.to(torch.int32)
    cand_valid_mask = torch.gather(valid_slot, 1, candidate_slots.long())
    candidate_slots = torch.where(
        cand_valid_mask, candidate_slots, torch.full_like(candidate_slots, -1)
    )

    sims_per_round = max(1, num_sims // _GUMBEL_ROUNDS)
    for round_i in range(_GUMBEL_ROUNDS):
        n_candidates = candidate_slots.shape[1]
        sims_per_cand = max(1, sims_per_round // n_candidates)
        wave_size = _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)]
        orch._run_simulations_for_root_slots(
            tree, state, game_active, candidate_slots, B, sims_per_cand,
            wave_size=wave_size,
        )
        if round_i == _GUMBEL_ROUNDS - 1:
            break
        cand_valid = candidate_slots >= 0
        cand_visits, cand_q = orch._gather_root_candidate_stats(tree, B, candidate_slots)
        per_game_keep = (cand_valid.sum(dim=1) // 2).clamp(min=1)
        max_keep = n_candidates // 2
        sigma_norm = (cfg.c_visit + cand_visits.float().max()) * cfg.c_scale
        cand_idx = candidate_slots.long().clamp(min=0)
        cand_score = (torch.gather(gumbel + legal_logits, 1, cand_idx) + sigma_norm * cand_q).float()
        cand_score = torch.where(cand_valid, cand_score, torch.full_like(cand_score, -1e30))
        _, keep_pos = torch.topk(cand_score, max_keep, dim=1)
        keep_rank = orch._keep_rank(max_keep)
        keep_valid = keep_rank < per_game_keep.unsqueeze(1)
        new_slots = torch.gather(candidate_slots, 1, keep_pos)
        candidate_slots = torch.where(keep_valid, new_slots, torch.full_like(new_slots, -1))

    slot_visits, slot_q = orch._gather_root_child_stats(tree, B)
    visits_np = slot_visits[0, :n_legal].float().cpu().numpy()
    q_mcts_np = slot_q[0, :n_legal].float().cpu().numpy()
    visited_mask = visits_np > 0

    max_n = float(visits_np.max())
    sigma = (cfg.c_visit + max_n) * cfg.c_scale
    log_prior = np.log(np.maximum(prior_np, 1e-20))

    # ── OLD approach: v_pi fallback ─────────────────────────────────────────
    completed_q_old = np.where(visited_mask, q_mcts_np, v_pi)
    logits_old = log_prior + sigma * completed_q_old
    logits_old -= logits_old.max()
    probs_old = np.exp(logits_old)
    probs_old /= probs_old.sum()

    # ── NEW approach: prior-anchored ─────────────────────────────────────────
    # Sampled moves: softmax(log_prior + sigma*Q_mcts) scaled to prior mass budget
    # Unsampled moves: keep prior
    sampled_logits = np.where(visited_mask, log_prior + sigma * q_mcts_np, -np.inf)
    sampled_logits_shifted = sampled_logits - np.max(sampled_logits[visited_mask]) if visited_mask.any() else sampled_logits
    sampled_exp = np.where(visited_mask, np.exp(sampled_logits_shifted), 0.0)
    sampled_dist = sampled_exp / sampled_exp.sum()  # sums to 1 over sampled

    prior_mass_sampled = prior_np[visited_mask].sum()
    probs_new = np.where(visited_mask, sampled_dist * prior_mass_sampled, prior_np)

    # Metrics
    mass_old_sampled = float(probs_old[visited_mask].sum())
    mass_new_sampled = float(probs_new[visited_mask].sum())

    # Top-1 move agreement between old and new
    top1_old = int(np.argmax(probs_old))
    top1_new = int(np.argmax(probs_new))
    top1_agree = (top1_old == top1_new)

    # Entropy of the improved policies (lower = more concentrated)
    def ent(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    # How concentrated is sampled mass in old vs new?
    # For sampled moves: what fraction goes to the best sampled move?
    sampled_probs_old = probs_old[visited_mask]
    sampled_probs_new = probs_new[visited_mask]

    return {
        "n_legal": n_legal,
        "n_visited": int(visited_mask.sum()),
        "v_pi": v_pi,
        "q_mcts_mean": float(q_mcts_np[visited_mask].mean()) if visited_mask.any() else 0.0,
        "prior_mass_sampled": float(prior_mass_sampled),
        # Mass
        "old_mass_sampled": mass_old_sampled,
        "new_mass_sampled": mass_new_sampled,
        # Entropy
        "old_entropy": ent(probs_old),
        "new_entropy": ent(probs_new),
        # Top-1
        "top1_agree": int(top1_agree),
        # Within-sampled concentration
        "old_sampled_top1_frac": float(sampled_probs_old.max() / sampled_probs_old.sum()) if len(sampled_probs_old) else 0.0,
        "new_sampled_top1_frac": float(sampled_probs_new.max() / sampled_probs_new.sum()) if len(sampled_probs_new) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=50)
    ap.add_argument("--sims", type=int, default=2048)
    ap.add_argument("--warmup-moves", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ext = hive_gpu.load_extension()
    net = load_net(args.checkpoint)
    orch = build_orch(net, args.sims)

    results = []
    for i in range(args.positions):
        state = make_position(ext, args.warmup_moves)
        r = run_and_measure(orch, state, args.sims)
        if r:
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.positions}...", flush=True)

    def avg(k):
        return np.mean([r[k] for r in results])
    def pct(k):
        return avg(k) * 100

    print(f"\n{'='*60}")
    print(f"  Policy target comparison  ({len(results)} positions, {args.sims} sims)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    print(f"\n  {'Metric':<45}  {'OLD (v_pi)':>10}  {'NEW (prior-anchored)':>20}")
    print(f"  {'-'*80}")
    print(f"  {'Mass in sampled (k) moves':<45}  {pct('old_mass_sampled'):>9.1f}%  {pct('new_mass_sampled'):>19.1f}%")
    print(f"  {'Mass in unsampled moves':<45}  {100-pct('old_mass_sampled'):>9.1f}%  {100-pct('new_mass_sampled'):>19.1f}%")
    print(f"  {'Prior mass in sampled moves (budget)':<45}  {'n/a':>10}  {pct('prior_mass_sampled'):>19.1f}%")
    print(f"  {'Policy entropy (nats)':<45}  {avg('old_entropy'):>10.3f}  {avg('new_entropy'):>20.3f}")
    print(f"  {'Top-1 agreement old vs new':<45}  {'—':>10}  {pct('top1_agree'):>19.1f}%")
    print(f"  {'Within-sampled: top-1 frac of sampled mass':<45}  {pct('old_sampled_top1_frac'):>9.1f}%  {pct('new_sampled_top1_frac'):>19.1f}%")

    print(f"\n  v_pi mean:          {avg('v_pi'):.4f}")
    print(f"  Q_mcts mean:        {avg('q_mcts_mean'):.4f}")
    print(f"  Legal moves:        {avg('n_legal'):.1f}")
    print(f"  Visited moves (k):  {avg('n_visited'):.1f}")
    print()


if __name__ == "__main__":
    main()
