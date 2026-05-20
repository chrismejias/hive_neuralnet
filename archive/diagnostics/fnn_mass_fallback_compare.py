"""
Compare three fallback strategies for unsampled moves in Gumbel improved policy:

  A) v_pi         — current approach; root value estimate
  B) child_init_q — per-move initial Q estimate (-v(successor)); consistent w/ MCTS
  C) 0.0          — neutral; don't assume anything about unsampled moves

For each strategy, measures improved-policy mass in sampled vs unsampled moves.

Usage:
  python3.10 fnn_mass_fallback_compare.py \
      --checkpoint checkpoints_fnn/hive_fnn_checkpoint_0200.pt \
      --positions 50 --warmup-moves 20
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


def build_orch(net, num_sims, det=True):
    cfg = FNNMCTSConfig(
        num_simulations=num_sims,
        batch_size=1,
        wave_parallel=True,
        wave_size=4,
        deterministic_non_root=det,
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
    """Run full Gumbel search, return per-move arrays."""
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
    child_init_q_np = child_q_init[0, :n_legal].float().cpu().numpy()

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
        cand_score = (
            torch.gather(gumbel + legal_logits, 1, cand_idx) + sigma_norm * cand_q
        ).float()
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

    def improved_mass(fallback_q_unvisited):
        completed_q = np.where(visited_mask, q_mcts_np, fallback_q_unvisited)
        logits = log_prior + sigma * completed_q
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        mass_visited = float(probs[visited_mask].sum())
        mass_unvisited = float(probs[~visited_mask].sum())
        return mass_visited, mass_unvisited

    # Strategy A: v_pi (current approach)
    mass_A = improved_mass(v_pi)

    # Strategy B: child_init_q (per-move value head estimate from root's perspective)
    mass_B = improved_mass(child_init_q_np)

    # Strategy C: 0.0 (neutral fallback)
    mass_C = improved_mass(0.0)

    return {
        "n_legal": n_legal,
        "n_visited": int(visited_mask.sum()),
        "v_pi": v_pi,
        "q_mcts_mean": float(q_mcts_np[visited_mask].mean()) if visited_mask.any() else 0.0,
        "q_mcts_max": float(q_mcts_np[visited_mask].max()) if visited_mask.any() else 0.0,
        "child_init_q_mean": float(child_init_q_np.mean()),
        "child_init_q_max": float(child_init_q_np.max()),
        "sigma": sigma,
        "A_visited": mass_A[0], "A_unvisited": mass_A[1],
        "B_visited": mass_B[0], "B_unvisited": mass_B[1],
        "C_visited": mass_C[0], "C_unvisited": mass_C[1],
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

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sims: {args.sims}  k={_GUMBEL_K}  Positions: {args.positions}")

    results = []
    for i in range(args.positions):
        state = make_position(ext, args.warmup_moves)
        r = run_and_measure(orch, state, args.sims)
        if r:
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.positions} done...", flush=True)

    def avg(k):
        return np.mean([r[k] for r in results])

    print(f"\n{'='*60}")
    print(f"  RESULTS ({len(results)} positions, {args.sims} sims, k={_GUMBEL_K})")
    print(f"{'='*60}")

    print(f"\n  Q VALUES")
    print(f"  {'v_pi (root value head)':<40} {avg('v_pi'):>8.4f}")
    print(f"  {'Q_mcts mean (sampled)':<40} {avg('q_mcts_mean'):>8.4f}")
    print(f"  {'Q_mcts max  (sampled)':<40} {avg('q_mcts_max'):>8.4f}")
    print(f"  {'child_init_q mean (all moves)':<40} {avg('child_init_q_mean'):>8.4f}")
    print(f"  {'child_init_q max  (all moves)':<40} {avg('child_init_q_max'):>8.4f}")
    print(f"  {'sigma':<40} {avg('sigma'):>8.1f}")

    print(f"\n  IMPROVED POLICY MASS (sampled/unsampled)  — 3 fallback strategies")
    print(f"  {'Strategy':<50}  {'Sampled':>8}  {'Unsampled':>10}")
    print(f"  {'-'*72}")
    print(f"  {'A: v_pi   (current — root value)':<50}  {avg('A_visited'):>7.1%}  {avg('A_unvisited'):>9.1%}")
    print(f"  {'B: child_init_q (per-move -v(succ))':<50}  {avg('B_visited'):>7.1%}  {avg('B_unvisited'):>9.1%}")
    print(f"  {'C: 0.0    (neutral)':<50}  {avg('C_visited'):>7.1%}  {avg('C_unvisited'):>9.1%}")

    print(f"\n  DIAGNOSIS")
    vpi = avg("v_pi")
    qm = avg("q_mcts_mean")
    ciq = avg("child_init_q_mean")
    sigma = avg("sigma")
    print(f"  v_pi vs Q_mcts gap:     {vpi:.4f} - {qm:.4f} = {vpi - qm:.4f}")
    print(f"  child_init_q vs Q_mcts: {ciq:.4f} vs {qm:.4f} (gap {qm - ciq:.4f})")
    print(f"  Strategy B effective:   exp(sigma * (Q_mcts - child_init_q)) = exp({sigma:.1f} * {qm - ciq:.4f})")
    mass_ratio = np.exp(sigma * (qm - ciq))
    print(f"    = {mass_ratio:.2e}x boost for sampled over unsampled")
    print()


if __name__ == "__main__":
    main()
