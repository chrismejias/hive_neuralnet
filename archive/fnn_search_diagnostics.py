"""Diagnose whether FNN Gumbel search depth is limited by:
  (A) Flat policy  — prior doesn't concentrate probability on a few moves
  (B) Flat value   — value head can't distinguish top-k candidates
  (C) c_puct too high — PUCT exploration keeps spreading visits laterally

Measurements:
  1. Policy: prior entropy, top-k concentration, KL from uniform
  2. Value discrimination: Q-range per round, Q vs prior agreement,
     fraction of rounds where Q ranking agrees with prior ranking
  3. Sigma decomposition: how much of the halving score comes from
     (Gumbel + log_prior) vs (sigma_norm * Q) at each round
  4. c_puct sweep (0.25, 0.5, 1.25, 2.5, 5.0): visit-depth metrics
     at each setting — does lower c_puct produce deeper trees?

Usage:
    python fnn_search_diagnostics.py [--positions N] [--sims N]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import hive_gpu
from hive_fnn.fnn_network import HiveFNN
from hive_fnn.fnn_mcts_orchestrator import (
    FNNMCTSConfig, FNNMCTSOrchestrator,
    _GUMBEL_K, _GUMBEL_ROUNDS, _GUMBEL_WAVE_SCHEDULE,
)

CHECKPOINT  = "checkpoints_fnn_deterministic/hive_fnn_checkpoint_0250.pt"
N_GEN_GAMES = 200
PLY_TARGET  = 30

CPUCT_SWEEP = [0.25, 0.5, 1.25, 2.5, 5.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def gini(v: np.ndarray) -> float:
    """Gini coefficient of a non-negative array: 0=uniform, 1=concentrated."""
    v = np.sort(np.abs(v))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() / (n * v.sum())) - (n + 1) / n)

def compute_node_depths(parent_idx_np: np.ndarray, node_count: int) -> np.ndarray:
    depths = np.empty(node_count, dtype=np.int32)
    depths[0] = 0
    for i in range(1, node_count):
        p = int(parent_idx_np[i])
        depths[i] = depths[p] + 1 if p >= 0 else 0
    return depths

def generate_positions(ext, n_games: int, ply_target: int) -> list[np.ndarray]:
    dev = "cuda"
    states = ext.create_initial_states(n_games, 0)
    for ply in range(ply_target + 1):
        legal_moves, num_legal = ext.generate_legal_moves_batch(states, n_games)
        nl = num_legal.cpu().numpy()
        if ply == ply_target:
            s_np = states.cpu().numpy()
            return [s_np[i].copy() for i in range(n_games) if nl[i] > 1]
        if (nl == 0).all():
            break
        move_bytes = torch.zeros(n_games, ext.SIZEOF_GPU_MOVE,
                                  dtype=torch.uint8, device=dev)
        for i in range(n_games):
            n = int(nl[i])
            if n > 0:
                move_bytes[i] = legal_moves[i, np.random.randint(0, n)]
        ext.apply_moves_batch(states, move_bytes, n_games)
        results = ext.check_results_batch(states, n_games).cpu().numpy()
        finished = np.where(results != 0)[0]
        if len(finished):
            fresh = ext.create_initial_states(len(finished), 0)
            for j, idx in enumerate(finished):
                states[idx] = fresh[j]
    return []


# ── Core instrumented search ──────────────────────────────────────────────────

def run_instrumented(orch: FNNMCTSOrchestrator,
                     state_np: np.ndarray,
                     num_sims: int) -> dict | None:
    """Run one Gumbel search; return detailed per-round statistics."""
    dev   = "cuda"
    B     = 1
    cfg   = orch.config
    ext   = orch.ext
    MAX_L = orch._max_legal

    state = torch.from_numpy(state_np).unsqueeze(0).to(dev)
    tree  = orch._alloc_tree(B)
    orch._reset_tree(tree)

    legal_moves, num_legal, root_features = \
        ext.generate_legal_moves_and_fnn_features_batch(state, B)
    n_legal = int(num_legal[0].item())
    if n_legal < 2:
        return None

    priors, root_val = orch._eval_states(state, legal_moves, num_legal, B, root_features)
    prior_np = priors[0, :n_legal].float().cpu().numpy()
    prior_np = prior_np / prior_np.sum().clip(min=1e-10)

    valid_slot = (
        torch.arange(MAX_L, device=dev).unsqueeze(0)
        < num_legal.to(torch.int64).unsqueeze(1)
    )
    safe_prior   = priors.clamp(min=1e-20)
    legal_logits = torch.where(valid_slot, safe_prior.log(),
                                torch.full_like(priors, -1e30))

    game_active = torch.ones(B, dtype=torch.int8, device=dev)
    orch._expand_root_if_needed(tree, state, legal_moves, num_legal,
                                 priors, game_active, B)
    orch._apply_root_dirichlet(tree, B, game_active.bool())

    u = torch.rand(B, MAX_L, device=dev).clamp(1e-4, 1 - 1e-4)
    gumbel = -torch.log(-torch.log(u))
    gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
    perturbed = gumbel + legal_logits

    k = min(_GUMBEL_K, n_legal)
    _, topk_slots = torch.topk(perturbed, k, dim=1)
    candidate_slots = topk_slots.to(torch.int32)
    cand_valid_init = torch.gather(valid_slot, 1, candidate_slots.long())
    candidate_slots = torch.where(cand_valid_init, candidate_slots,
                                   torch.full_like(candidate_slots, -1))

    sims_per_round = max(1, num_sims // _GUMBEL_ROUNDS)
    round_data     = []

    for round_i in range(_GUMBEL_ROUNDS):
        cand_valid    = candidate_slots >= 0
        n_alive       = int(cand_valid[0].sum().item())
        n_candidates  = int(candidate_slots.shape[1])
        sims_per_cand = max(1, sims_per_round // n_candidates)

        wave_size = (
            _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)]
            if cfg.wave_parallel else 1
        )
        orch._run_simulations_for_root_slots(
            tree, state, game_active, candidate_slots, B, sims_per_cand,
            wave_size=wave_size,
        )
        cand_visits, cand_q = orch._gather_root_candidate_stats(
            tree, B, candidate_slots,
        )

        alive_mask  = cand_valid[0].cpu().numpy()
        alive_slots = candidate_slots[0][cand_valid[0]].cpu().numpy()
        alive_q     = cand_q[0][cand_valid[0]].float().cpu().numpy()
        alive_prior_logit = legal_logits[0][alive_slots.astype(np.int64)].float().cpu().numpy()
        alive_gumbel      = gumbel[0][alive_slots.astype(np.int64)].float().cpu().numpy()

        # Sigma score decomposition
        sigma_norm     = (cfg.c_visit + float(cand_visits.max())) * cfg.c_scale
        prior_term     = alive_gumbel + alive_prior_logit   # Gumbel + log_prior
        q_term         = sigma_norm * alive_q
        total_sigma    = prior_term + q_term

        # How much variance in the sigma score comes from Q vs prior term?
        var_prior = float(prior_term.var()) if len(prior_term) > 1 else 0.0
        var_q     = float(q_term.var())     if len(q_term)     > 1 else 0.0
        var_total = float(total_sigma.var()) if len(total_sigma) > 1 else 1e-10
        q_frac    = var_q / (var_prior + var_q + 1e-10)  # fraction of variance from Q

        # Does Q ranking agree with prior ranking?
        prior_rank = np.argsort(-alive_prior_logit)
        q_rank     = np.argsort(-alive_q)
        if n_alive > 2:
            spearman = float(np.corrcoef(
                np.argsort(prior_rank), np.argsort(q_rank)
            )[0, 1])
        elif n_alive == 2:
            spearman = float(np.sign(
                (alive_prior_logit[0] - alive_prior_logit[1]) *
                (alive_q[0] - alive_q[1])
            ))
        else:
            spearman = 1.0

        # Visit concentration (Gini on visit counts)
        alive_visits = cand_visits[0][cand_valid[0]].cpu().numpy().astype(np.float32)
        visit_gini   = gini(alive_visits)

        round_data.append({
            "n_alive":       n_alive,
            "sims_per_cand": sims_per_cand,
            "q_range":       float(alive_q.max() - alive_q.min()) if n_alive > 1 else 0.0,
            "sigma_norm":    sigma_norm,
            "q_frac":        q_frac,        # fraction of sigma variance from Q
            "prior_q_spearman": spearman,   # rank correlation prior vs Q
            "visit_gini":    visit_gini,    # 0=uniform visits, 1=concentrated
        })

        if n_alive <= 1:
            continue
        per_game_keep = (cand_valid.sum(dim=1) // 2).clamp(min=1)
        max_keep      = n_candidates // 2
        cand_idx      = candidate_slots.long().clamp(min=0)
        cand_score    = (
            torch.gather(gumbel + legal_logits, 1, cand_idx)
            + sigma_norm * cand_q
        ).float()
        cand_score    = torch.where(cand_valid, cand_score,
                                     torch.full_like(cand_score, -1e30))
        _, keep_pos   = torch.topk(cand_score, max_keep, dim=1)
        keep_rank     = orch._keep_rank(max_keep)
        keep_valid    = keep_rank < per_game_keep.unsqueeze(1)
        new_slots     = torch.gather(candidate_slots, 1, keep_pos)
        candidate_slots = torch.where(keep_valid, new_slots,
                                       torch.full_like(new_slots, -1))

    # Tree depth
    node_count = int(tree["node_count"][0].item())
    parent_np  = tree["parent_idx"][0, :node_count].cpu().numpy()
    depths     = compute_node_depths(parent_np, node_count)

    return {
        "n_legal":       n_legal,
        "prior_entropy": entropy(prior_np),
        "prior_top1":    float(prior_np.max()),
        "prior_gini":    gini(prior_np),
        "root_val":      float(root_val[0].item()),
        "node_count":    node_count,
        "avg_depth":     float(depths.mean()),
        "max_depth":     int(depths.max()),
        "round_data":    round_data,
    }


# ── c_puct sweep ─────────────────────────────────────────────────────────────

def cpuct_sweep(net, ext, positions, num_sims, cpuct_values):
    """For each c_puct, run the search and record depth/visit metrics."""
    results = {}
    for cpuct in cpuct_values:
        cfg  = FNNMCTSConfig(
            num_simulations=num_sims, batch_size=1,
            c_puct=cpuct, wave_parallel=True, expansion_mask=0,
        )
        orch = FNNMCTSOrchestrator(net, cfg)
        node_counts, avg_depths, max_depths = [], [], []
        top1_changes = []
        prior_top1s  = []

        for pos in positions:
            r = run_instrumented(orch, pos, num_sims)
            if r is None:
                continue
            node_counts.append(r["node_count"])
            avg_depths.append(r["avg_depth"])
            max_depths.append(r["max_depth"])
            prior_top1s.append(r["prior_top1"])

        results[cpuct] = {
            "avg_nodes":   float(np.mean(node_counts)),
            "avg_depth":   float(np.mean(avg_depths)),
            "max_depth":   float(np.mean(max_depths)),
            "nodes_per_sim": float(np.mean(node_counts)) / num_sims,
        }
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def report(results: list[dict], sweep: dict, num_sims: int) -> None:
    def avg(key):
        return np.mean([r[key] for r in results])

    print(f"\n{'='*66}")
    print(f"  FNN Search Diagnostics  ({len(results)} positions, {num_sims} sims)")
    print(f"{'='*66}")

    # ── (A) Policy flatness ──────────────────────────────────────────────
    print(f"\n  (A) POLICY SHARPNESS")
    print(f"  {'Metric':<44}  {'Value':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Prior entropy (nats)':<44}  {avg('prior_entropy'):>8.3f}")
    log_uniform = math.log(avg('n_legal'))
    print(f"  {'Uniform entropy for avg legal moves (nats)':<44}  {log_uniform:>8.3f}")
    print(f"  {'Relative entropy = prior / uniform':<44}  {avg('prior_entropy')/log_uniform:>8.3f}")
    print(f"  {'Prior Gini (0=flat, 1=spike)':<44}  {avg('prior_gini'):>8.3f}")
    print(f"  {'Top-1 prior probability':<44}  {avg('prior_top1'):>8.3f}")
    # top-k concentration
    for topk, label in [(1, "top-1"), (3, "top-3"), (5, "top-5")]:
        pass  # handled in round data section

    # ── (B) Value discrimination ──────────────────────────────────────────
    print(f"\n  (B) VALUE HEAD DISCRIMINATION  (per halving round)")
    print(f"  {'Round':<7}  {'Q-range':>9}  {'sigma_norm':>11}  "
          f"{'Q contrib':>11}  {'Q frac':>8}  {'Prior-Q rho':>12}  {'Visit Gini':>11}")
    print(f"  {'-'*78}")
    for ri in range(_GUMBEL_ROUNDS):
        rd_list = [r["round_data"][ri] for r in results if len(r["round_data"]) > ri]
        if not rd_list:
            continue
        q_range  = np.mean([d["q_range"]       for d in rd_list])
        snorm    = np.mean([d["sigma_norm"]     for d in rd_list])
        q_frac   = np.mean([d["q_frac"]        for d in rd_list])
        spearman = np.mean([d["prior_q_spearman"] for d in rd_list])
        v_gini   = np.mean([d["visit_gini"]    for d in rd_list])
        q_contrib = snorm * q_range  # total Q contribution in nats
        print(f"  {ri+1:<7}  {q_range:>9.4f}  {snorm:>11.1f}  "
              f"{q_contrib:>11.3f}  {q_frac:>8.3f}  {spearman:>12.3f}  {v_gini:>11.3f}")

    print(f"\n  Interpretation:")
    rd0 = [r["round_data"][0] for r in results if r["round_data"]]
    rd3 = [r["round_data"][-1] for r in results if r["round_data"]]
    mean_q_frac_r1 = np.mean([d["q_frac"]  for d in rd0])
    mean_q_frac_r4 = np.mean([d["q_frac"]  for d in rd3])
    mean_rho_r1    = np.mean([d["prior_q_spearman"] for d in rd0])
    mean_rho_r4    = np.mean([d["prior_q_spearman"] for d in rd3])
    print(f"  Q variance fraction: round 1 = {mean_q_frac_r1:.3f}, "
          f"round 4 = {mean_q_frac_r4:.3f}")
    print(f"  Prior-Q rank corr:   round 1 = {mean_rho_r1:.3f}, "
          f"round 4 = {mean_rho_r4:.3f}")
    if mean_q_frac_r4 < 0.2:
        print(f"  -> Q term contributes < 20% of sigma variance: VALUE HEAD is the bottleneck.")
    else:
        print(f"  -> Q term contributes >= 20% of sigma variance: value is informative.")
    if mean_rho_r4 > 0.7:
        print(f"  -> High prior-Q correlation: search CONFIRMS the prior (consistent network).")
    elif mean_rho_r4 < 0.3:
        print(f"  -> Low prior-Q correlation: search DIVERGES from prior (policy may be misleading).")
    else:
        print(f"  -> Moderate prior-Q correlation: search partially overrides the prior.")

    # ── (C) c_puct sweep ──────────────────────────────────────────────────
    print(f"\n  (C) c_puct SWEEP  (effect on tree depth and breadth)")
    print(f"  {'c_puct':<10}  {'Avg nodes':>11}  {'Avg depth':>10}  "
          f"{'Max depth':>10}  {'Nodes/sim':>10}")
    print(f"  {'-'*58}")
    for cpuct, s in sweep.items():
        print(f"  {cpuct:<10.2f}  {s['avg_nodes']:>11.1f}  {s['avg_depth']:>10.2f}  "
              f"{s['max_depth']:>10.1f}  {s['nodes_per_sim']:>10.2f}")

    cpuct_vals   = list(sweep.keys())
    depth_vals   = [sweep[c]["avg_depth"]     for c in cpuct_vals]
    nodes_vals   = [sweep[c]["nodes_per_sim"] for c in cpuct_vals]
    depth_range  = max(depth_vals) - min(depth_vals)
    nodes_range  = max(nodes_vals) - min(nodes_vals)

    print(f"\n  Interpretation:")
    print(f"  Depth range across c_puct sweep: {depth_range:.2f} "
          f"({'sensitive' if depth_range > 0.5 else 'insensitive'} to c_puct)")
    if depth_range < 0.5:
        print(f"  -> Depth barely changes with c_puct: the value head's flatness, not")
        print(f"     over-exploration, is the root cause. Tuning c_puct won't help.")
    else:
        print(f"  -> Depth changes significantly with c_puct: exploration tuning matters.")
        best_depth_cpuct = cpuct_vals[np.argmax(depth_vals)]
        print(f"     Deepest search at c_puct={best_depth_cpuct}.")

    print(f"\n  SYSTEMATIC c_puct SELECTION:")
    mean_q_range = np.mean([r["round_data"][0]["q_range"] for r in results])
    mean_n_legal = avg("n_legal")
    mean_prior   = avg("prior_top1")
    # At k=16 candidates, N_root ~ sims, N_cand ~ sims/16
    # PUCT exploration term ~ c_puct * mean_prior * sqrt(N_root) / (1 + N_cand)
    # We want this ~ Q_range for balanced exploration vs exploitation
    n_root_approx = float(num_sims)
    n_cand_approx = float(num_sims) / _GUMBEL_K
    suggested_cpuct = mean_q_range * (1 + n_cand_approx) / (mean_prior * math.sqrt(n_root_approx))
    print(f"  Q-range (round 1): {mean_q_range:.4f}")
    print(f"  Avg prior top-1:   {mean_prior:.4f}")
    print(f"  At {num_sims} sims, N_root~{n_root_approx:.0f}, N_cand~{n_cand_approx:.0f}")
    print(f"  => c_puct that balances exploration/exploitation: ~{suggested_cpuct:.3f}")
    print(f"     (current: {results[0]['round_data'][0].get('sims_per_cand', '?')})")
    print(f"  Note: for Gumbel MCTS, c_visit ({results[0]['round_data'][0]['sigma_norm']:.0f}*Q)")
    print(f"  dominates the halving decision; c_puct mainly controls per-sim tree traversal.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=100)
    ap.add_argument("--sims",      type=int, default=1024)
    ap.add_argument("--ply",       type=int, default=30)
    args = ap.parse_args()

    print("Loading extension and model...")
    ext  = hive_gpu.load_extension()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net  = HiveFNN(ckpt["net_config"]).cuda().eval()
    net.load_state_dict(ckpt["model_state_dict"])

    print(f"Generating positions...")
    positions = generate_positions(ext, N_GEN_GAMES, PLY_TARGET)
    if len(positions) > args.positions:
        np.random.shuffle(positions)
        positions = positions[:args.positions]
    print(f"  Got {len(positions)} positions.")

    # Main diagnostic run at default c_puct
    cfg  = FNNMCTSConfig(num_simulations=args.sims, batch_size=1,
                          wave_parallel=True, expansion_mask=0)
    orch = FNNMCTSOrchestrator(net, cfg)

    print(f"Running main diagnostic ({args.sims} sims)...")
    results = []
    for i, pos in enumerate(positions):
        if i % 20 == 0:
            print(f"  {i}/{len(positions)}...", flush=True)
        r = run_instrumented(orch, pos, args.sims)
        if r:
            results.append(r)

    # c_puct sweep on a smaller subset
    sweep_positions = positions[:30]
    print(f"\nRunning c_puct sweep ({len(sweep_positions)} positions each)...")
    for cpuct in CPUCT_SWEEP:
        print(f"  c_puct={cpuct}...", flush=True)
    sweep = cpuct_sweep(net, ext, sweep_positions, args.sims, CPUCT_SWEEP)

    report(results, sweep, args.sims)


if __name__ == "__main__":
    main()
