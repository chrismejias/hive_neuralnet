"""
Diagnose the improved-policy mass distribution in FNN Gumbel MCTS.

Core question: why does ~73% of improved-policy mass land on unsampled moves?

Measures:
  1. v_pi (root value) distribution
  2. Q_mcts distribution for sampled (k=16) moves
  3. How many sampled moves beat v_pi — and by how much
  4. sigma value and its effect on softmax sharpness
  5. Mass in top-k vs rest, varying sims (128, 256, 512, 1024, 2048)
  6. Correlation: v_pi vs mean Q_mcts across positions (calibration check)

Usage:
  python3.10 fnn_mass_diagnostic.py \
      --checkpoint checkpoints_fnn/hive_fnn_checkpoint_0200.pt \
      --positions 30 --warmup-moves 20
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")
import hive_gpu
from hive_fnn.fnn_mcts_orchestrator import (
    FNNMCTSConfig, FNNMCTSOrchestrator,
    _GUMBEL_K, _GUMBEL_ROUNDS, _GUMBEL_WAVE_SCHEDULE,
)
from hive_fnn.fnn_network import FNNConfig, HiveFNN


def load_net(path: str) -> HiveFNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = HiveFNN(FNNConfig.large())
    net.load_state_dict(ckpt["model_state_dict"])
    return net.cuda().eval()


def build_orch(net, num_sims: int, det: bool = True) -> FNNMCTSOrchestrator:
    cfg = FNNMCTSConfig(
        num_simulations=num_sims,
        batch_size=1,
        wave_parallel=True,
        wave_size=4,
        deterministic_non_root=det,
        virtual_q_penalty=0.25,
        non_root_sigma=4.0,
        expansion_mask=0,
        rebase_tree_each_move=False,
    )
    return FNNMCTSOrchestrator(net, cfg)


def make_position(ext, warmup_moves: int) -> torch.Tensor:
    states = ext.create_initial_states(1, 0)
    for _ in range(warmup_moves):
        lm, nl = ext.generate_legal_moves_batch(states, 1)
        n = int(nl[0].item())
        if n == 0:
            break
        idx = np.random.randint(0, n)
        mv = torch.zeros(1, ext.SIZEOF_GPU_MOVE, dtype=torch.uint8, device="cuda")
        mv[0] = lm[0, idx]
        ext.apply_moves_batch(states, mv, 1)
    return states


def run_gumbel_and_measure(orch, state, num_sims: int, k: int = 16) -> dict | None:
    """Run full Gumbel sequential halving and measure improved-policy mass."""
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

    priors, root_vals, child_q = orch._eval_states(
        state, legal_moves, num_legal, B, root_features
    )
    v_pi = float(root_vals[0].item())
    prior_np = priors[0, :n_legal].float().cpu().numpy()

    game_active = torch.ones(B, dtype=torch.int8, device=dev)
    orch._expand_root_if_needed(
        tree, state, legal_moves, num_legal, priors, child_q, game_active, B
    )

    valid_slot = (
        torch.arange(MAX_L, device=dev).unsqueeze(0)
        < num_legal.to(torch.int64).unsqueeze(1)
    )
    safe_prior = priors.clamp(min=1e-20)
    legal_logits = torch.where(
        valid_slot, safe_prior.log(), torch.full_like(priors, -1e30)
    )

    u = torch.rand(B, MAX_L, device=dev).clamp(1e-4, 1 - 1e-4)
    gumbel = -torch.log(-torch.log(u))
    gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
    perturbed = gumbel + legal_logits

    actual_k = min(k, n_legal)
    _, topk_slots = torch.topk(perturbed, actual_k, dim=1)
    candidate_slots = topk_slots.to(torch.int32)
    cand_valid_mask = torch.gather(valid_slot, 1, candidate_slots.long())
    candidate_slots = torch.where(
        cand_valid_mask, candidate_slots, torch.full_like(candidate_slots, -1)
    )

    sims_per_round = max(1, num_sims // _GUMBEL_ROUNDS)

    for round_i in range(_GUMBEL_ROUNDS):
        n_candidates = int(candidate_slots.shape[1])
        sims_per_cand = max(1, sims_per_round // n_candidates)
        wave_size = (
            _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)]
        )
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
        cand_score = (
            torch.gather(gumbel + legal_logits, 1, cand_idx) + sigma_norm * cand_q
        ).float()
        cand_score = torch.where(
            cand_valid, cand_score, torch.full_like(cand_score, -1e30)
        )
        _, keep_pos = torch.topk(cand_score, max_keep, dim=1)
        keep_rank = orch._keep_rank(max_keep)
        keep_valid = keep_rank < per_game_keep.unsqueeze(1)
        new_slots = torch.gather(candidate_slots, 1, keep_pos)
        candidate_slots = torch.where(
            keep_valid, new_slots, torch.full_like(new_slots, -1)
        )

    # ── Gather final stats ──────────────────────────────────────────────────────
    slot_visits, slot_q = orch._gather_root_child_stats(tree, B)

    visits_np = slot_visits[0, :n_legal].float().cpu().numpy()
    q_np = slot_q[0, :n_legal].float().cpu().numpy()

    visited_mask = visits_np > 0
    n_visited = int(visited_mask.sum())
    max_n = float(visits_np.max())
    sigma = (cfg.c_visit + max_n) * cfg.c_scale

    # completed Q
    completed_q = np.where(visited_mask, q_np, v_pi)

    # improved policy
    improved_logits = np.log(np.maximum(prior_np, 1e-20)) + sigma * completed_q
    improved_logits -= improved_logits.max()
    improved_probs = np.exp(improved_logits)
    improved_probs /= improved_probs.sum()

    # which slots are in the sampled k?
    sampled_slots = set()
    cand_np = candidate_slots[0].cpu().numpy()
    for s in cand_np:
        if 0 <= s < n_legal:
            sampled_slots.add(int(s))

    # also include any visited slot (might differ slightly from sampled due to halving)
    visited_slots = set(int(i) for i in range(n_legal) if visited_mask[i])

    mass_sampled = float(improved_probs[list(sampled_slots)].sum()) if sampled_slots else 0.0
    mass_visited = float(improved_probs[list(visited_slots)].sum()) if visited_slots else 0.0
    mass_rest = float(improved_probs.sum()) - mass_visited  # unsampled

    # Q stats for sampled moves
    q_sampled = q_np[list(visited_slots)] if visited_slots else np.array([])
    n_above_vpi = int((q_sampled > v_pi).sum()) if len(q_sampled) else 0

    return {
        "n_legal": n_legal,
        "n_visited": n_visited,
        "v_pi": v_pi,
        "q_mean": float(q_sampled.mean()) if len(q_sampled) else float("nan"),
        "q_max": float(q_sampled.max()) if len(q_sampled) else float("nan"),
        "q_min": float(q_sampled.min()) if len(q_sampled) else float("nan"),
        "q_std": float(q_sampled.std()) if len(q_sampled) > 1 else 0.0,
        "n_above_vpi": n_above_vpi,
        "mass_sampled": mass_sampled,
        "mass_visited": mass_visited,
        "mass_rest": mass_rest,
        "sigma": sigma,
        "max_n": max_n,
        "vpi_minus_qmean": v_pi - float(q_sampled.mean()) if len(q_sampled) else float("nan"),
    }


def print_results(results: list[dict], label: str) -> None:
    def avg(k):
        vals = [r[k] for r in results if not np.isnan(r.get(k, float("nan")))]
        return np.mean(vals) if vals else float("nan")

    print(f"\n{'='*60}")
    print(f"  {label}  ({len(results)} positions)")
    print(f"{'='*60}")

    print(f"\n  VALUE / Q CALIBRATION")
    print(f"  {'v_pi (root value estimate)':<40} {avg('v_pi'):>8.4f}")
    print(f"  {'Q_mcts mean (sampled moves)':<40} {avg('q_mean'):>8.4f}")
    print(f"  {'Q_mcts max  (sampled moves)':<40} {avg('q_max'):>8.4f}")
    print(f"  {'Q_mcts min  (sampled moves)':<40} {avg('q_min'):>8.4f}")
    print(f"  {'v_pi - Q_mcts_mean (gap)':<40} {avg('vpi_minus_qmean'):>8.4f}")
    print(f"  {'Sampled moves with Q > v_pi':<40} {avg('n_above_vpi'):>8.2f} / {avg('n_visited'):.1f}")

    print(f"\n  IMPROVED POLICY MASS")
    print(f"  {'Mass in sampled (k) moves':<40} {avg('mass_sampled'):>8.3f}  ({100*avg('mass_sampled'):.1f}%)")
    print(f"  {'Mass in visited moves':<40} {avg('mass_visited'):>8.3f}  ({100*avg('mass_visited'):.1f}%)")
    print(f"  {'Mass in unsampled moves':<40} {avg('mass_rest'):>8.3f}  ({100*avg('mass_rest'):.1f}%)")

    print(f"\n  SEARCH PARAMS")
    print(f"  {'sigma = (c_visit + max_n) * c_scale':<40} {avg('sigma'):>8.1f}")
    print(f"  {'max visit count (max_n)':<40} {avg('max_n'):>8.1f}")
    print(f"  {'Legal moves':<40} {avg('n_legal'):>8.1f}")
    print(f"  {'Visited moves':<40} {avg('n_visited'):>8.1f}")

    # Calibration interpretation
    gap = avg("vpi_minus_qmean")
    sigma_val = avg("sigma")
    if not np.isnan(gap) and not np.isnan(sigma_val):
        factor = np.exp(sigma_val * abs(gap))
        print(f"\n  CALIBRATION DIAGNOSIS")
        print(f"  sigma * gap = {sigma_val:.1f} * {gap:.4f} = {sigma_val * gap:.2f}")
        print(f"  => unsampled moves get exp({sigma_val * gap:.2f}) = {factor:.1f}x boost over avg sampled")
        if gap > 0.01:
            print(f"  => VALUE HEAD IS OVER-OPTIMISTIC: v_pi >> Q_mcts by {gap:.3f}")
            print(f"     Unsampled moves inherit v_pi, dominating the improved policy.")
        elif gap < -0.01:
            print(f"  => value head slightly pessimistic: Q_mcts > v_pi by {-gap:.3f}")
            print(f"     Sampled moves dominate the improved policy (good).")
        else:
            print(f"  => Value head well-calibrated (gap < 0.01).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=30)
    ap.add_argument("--warmup-moves", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--det", action="store_true", default=False,
                    help="Use deterministic non-root selection")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ext = hive_gpu.load_extension()
    net = load_net(args.checkpoint)

    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Positions:   {args.positions}  |  Warmup: {args.warmup_moves}  |  DET: {args.det}")

    # Compare across different sim counts
    for num_sims in [256, 512, 1024, 2048]:
        orch = build_orch(net, num_sims, det=args.det)
        results = []
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        for i in range(args.positions):
            state = make_position(ext, args.warmup_moves)
            r = run_gumbel_and_measure(orch, state, num_sims)
            if r:
                results.append(r)
        label = f"sims={num_sims}  k={_GUMBEL_K}  {'DET' if args.det else 'PUCT'}"
        print_results(results, label)


if __name__ == "__main__":
    main()
