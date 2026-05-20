"""Profile the depth/breadth structure of FNN Gumbel MCTS at 1024 simulations.

For each of N midgame positions, runs the full Gumbel sequential-halving search
with instrumentation inserted at each halving round.  Reports:

  Per-round (4 rounds of halving):
    - Candidates alive at start of round
    - sims_per_candidate this round
    - Q-value range and std of surviving candidates

  Per-position aggregate:
    - Prior entropy / visit entropy (sharpening from search)
    - Effective breadth = exp(visit entropy)  — number of "live" options
    - Top-1 change rate  (does MCTS override the raw policy pick?)
    - Tree size (nodes created)
    - Tree depth distribution (from parent_idx reconstruction)
    - Effective branching factor = tree_size / tree_depth

Usage:
    python fnn_gumbel_profile.py [--positions N] [--sims N]
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

CHECKPOINT  = "checkpoints_fnn/hive_fnn_checkpoint_0050.pt"
N_GEN_GAMES = 200
PLY_TARGET  = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p)))


def compute_node_depths(parent_idx_np: np.ndarray, node_count: int) -> np.ndarray:
    """Return depth of every live node. O(node_count); works because nodes are
    allocated parent-before-child so a single forward pass suffices."""
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
        move_bytes = torch.zeros(n_games, ext.SIZEOF_GPU_MOVE, dtype=torch.uint8, device=dev)
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


# ── Instrumented Gumbel search ────────────────────────────────────────────────

def gumbel_search_instrumented(
    orch: FNNMCTSOrchestrator,
    state_np: np.ndarray,
    num_sims: int,
) -> dict:
    """Run one Gumbel search with per-round statistics collected."""
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
    if n_legal == 0:
        return {}

    priors, root_val = orch._eval_states(state, legal_moves, num_legal, B, root_features)
    prior_np = priors[0, :n_legal].float().cpu().numpy()
    prior_entropy = entropy(prior_np)
    prior_top1    = int(np.argmax(prior_np))

    valid_slot = (
        torch.arange(MAX_L, device=dev).unsqueeze(0)
        < num_legal.to(torch.int64).unsqueeze(1)
    )
    safe_prior  = priors.clamp(min=1e-20)
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
    candidate_valid = torch.gather(valid_slot, 1, candidate_slots.long())
    candidate_slots = torch.where(candidate_valid, candidate_slots,
                                   torch.full_like(candidate_slots, -1))

    sims_per_round = max(1, num_sims // _GUMBEL_ROUNDS)

    round_stats = []
    total_sims_run = 0

    for round_i in range(_GUMBEL_ROUNDS):
        cand_valid   = candidate_slots >= 0
        n_alive      = int(cand_valid[0].sum().item())
        n_candidates = int(candidate_slots.shape[1])
        sims_per_cand = max(1, sims_per_round // n_candidates)
        actual_sims   = sims_per_cand * n_alive
        total_sims_run += actual_sims

        wave_size = (
            _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)]
            if cfg.wave_parallel else 1
        )
        orch._run_simulations_for_root_slots(
            tree, state, game_active, candidate_slots, B, sims_per_cand,
            wave_size=wave_size,
        )

        # Collect Q-values for alive candidates
        cand_visits, cand_q = orch._gather_root_candidate_stats(
            tree, B, candidate_slots,
        )
        alive_q = cand_q[0][cand_valid[0]].float().cpu().numpy()
        q_range = float(alive_q.max() - alive_q.min()) if len(alive_q) > 1 else 0.0
        q_std   = float(alive_q.std())                  if len(alive_q) > 1 else 0.0

        round_stats.append({
            "round":         round_i + 1,
            "n_alive":       n_alive,
            "sims_per_cand": sims_per_cand,
            "actual_sims":   actual_sims,
            "q_range":       q_range,
            "q_std":         q_std,
        })

        # Halve — same logic as select_move_gpu
        if n_alive <= 1:
            continue
        per_game_keep = (cand_valid.sum(dim=1) // 2).clamp(min=1)
        max_keep      = n_candidates // 2

        sigma_norm = (cfg.c_visit + cand_visits.max()) * cfg.c_scale
        cand_idx   = candidate_slots.long().clamp(min=0)
        cand_score = (
            torch.gather(gumbel + legal_logits, 1, cand_idx)
            + sigma_norm * cand_q
        ).float()
        cand_score = torch.where(cand_valid, cand_score,
                                  torch.full_like(cand_score, -1e30))
        _, keep_pos = torch.topk(cand_score, max_keep, dim=1)
        keep_rank   = orch._keep_rank(max_keep)
        keep_valid  = keep_rank < per_game_keep.unsqueeze(1)
        new_slots   = torch.gather(candidate_slots, 1, keep_pos)
        candidate_slots = torch.where(keep_valid, new_slots,
                                       torch.full_like(new_slots, -1))

    # ── Post-search stats ─────────────────────────────────────────────────────

    # Visit distribution (root children only, slot index = legal move index)
    root_node = int(tree["root_node"][0].item())
    vc = tree["visit_count"][0].cpu().numpy()
    ai = tree["action_idx"][0].cpu().numpy()
    node_count = int(tree["node_count"][0].item())

    # Build visit array over legal slots
    visit_slots = np.zeros(n_legal, dtype=np.float32)
    for node in range(node_count):
        p = tree["parent_idx"][0, node].item()
        if p == root_node:
            slot = int(ai[node])
            if 0 <= slot < n_legal:
                visit_slots[slot] += int(vc[node])

    # Also include root's own child visit counts via first_child chain
    # (simpler: just use action_idx to map node→legal slot)
    total_visits = visit_slots.sum()
    if total_visits > 0:
        visit_probs = visit_slots / total_visits
    else:
        visit_probs = prior_np.copy()

    visit_entropy  = entropy(visit_probs)
    visit_top1     = int(np.argmax(visit_probs))
    top1_changed   = (visit_top1 != prior_top1)
    eff_breadth    = math.exp(visit_entropy)

    # Tree depth distribution
    parent_np = tree["parent_idx"][0, :node_count].cpu().numpy()
    depths    = compute_node_depths(parent_np, node_count)
    avg_depth = float(depths.mean())
    max_depth = int(depths.max())

    # Effective branching factor: nodes at each depth level
    depth_counts = np.bincount(depths)
    # Geometric mean of child/parent ratio across non-root levels
    ratios = []
    for d in range(1, len(depth_counts)):
        if depth_counts[d - 1] > 0 and depth_counts[d] > 0:
            ratios.append(depth_counts[d] / depth_counts[d - 1])
    eff_branching = float(np.mean(ratios)) if ratios else 1.0

    return {
        "n_legal":        n_legal,
        "root_value":     float(root_val[0].item()),
        "prior_entropy":  prior_entropy,
        "visit_entropy":  visit_entropy,
        "eff_breadth":    eff_breadth,
        "top1_changed":   top1_changed,
        "prior_top1_prob": float(prior_np[prior_top1]),
        "visit_top1_prob": float(visit_probs[visit_top1]),
        "total_sims_run": total_sims_run,
        "node_count":     node_count,
        "avg_depth":      avg_depth,
        "max_depth":      max_depth,
        "eff_branching":  eff_branching,
        "depth_counts":   depth_counts.tolist(),
        "round_stats":    round_stats,
    }


# ── Aggregation and reporting ─────────────────────────────────────────────────

def report(results: list[dict], num_sims: int) -> None:
    if not results:
        print("No results.")
        return

    def avg(key):
        return np.mean([r[key] for r in results])

    def pct(key):
        return np.mean([r[key] for r in results]) * 100

    print(f"\n{'='*64}")
    print(f"  FNN Gumbel Profile  ({len(results)} positions, {num_sims} sims)")
    print(f"{'='*64}")

    print(f"\n  POSITION CONTEXT")
    print(f"  {'Metric':<40}  {'Value':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Avg legal moves':<40}  {avg('n_legal'):>10.1f}")
    print(f"  {'Avg root value (win prob)':<40}  {avg('root_value'):>10.3f}")

    print(f"\n  POLICY SHARPNESS")
    print(f"  {'Metric':<40}  {'Value':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Prior entropy (nats)':<40}  {avg('prior_entropy'):>10.3f}")
    print(f"  {'Visit entropy after search (nats)':<40}  {avg('visit_entropy'):>10.3f}")
    prior_eff_breadth = np.mean([math.exp(r['prior_entropy']) for r in results])
    print(f"  {'Effective breadth before (exp H)':<40}  {prior_eff_breadth:>10.2f}")
    print(f"  {'Effective breadth after  (exp H)':<40}  {avg('eff_breadth'):>10.2f}")
    print(f"  {'Prior  top-1 prob':<40}  {avg('prior_top1_prob'):>10.3f}")
    print(f"  {'Visit  top-1 prob':<40}  {avg('visit_top1_prob'):>10.3f}")
    print(f"  {'Top-1 changed by search':<40}  {pct('top1_changed'):>9.1f}%")

    print(f"\n  SEARCH STRUCTURE")
    print(f"  {'Metric':<40}  {'Value':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Total sims actually run':<40}  {avg('total_sims_run'):>10.1f}")
    print(f"  {'Tree nodes created':<40}  {avg('node_count'):>10.1f}")
    print(f"  {'Average tree depth':<40}  {avg('avg_depth'):>10.2f}")
    print(f"  {'Max tree depth':<40}  {avg('max_depth'):>10.1f}")
    print(f"  {'Effective branching factor':<40}  {avg('eff_branching'):>10.2f}")
    nodes_per_sim = np.mean([r['node_count'] / max(r['total_sims_run'], 1) for r in results])
    print(f"  {'Nodes per sim (depth proxy)':<40}  {nodes_per_sim:>10.2f}")

    print(f"\n  HALVING ROUNDS  (averages across positions)")
    print(f"  {'Round':<8}  {'Alive':>6}  {'Sims/cand':>10}  {'Q-range':>9}  {'Q-std':>7}")
    print(f"  {'-'*50}")
    for ri in range(_GUMBEL_ROUNDS):
        rs_list = [r["round_stats"][ri] for r in results if len(r["round_stats"]) > ri]
        if not rs_list:
            continue
        alive      = np.mean([rs["n_alive"]       for rs in rs_list])
        s_per_c    = np.mean([rs["sims_per_cand"]  for rs in rs_list])
        q_range    = np.mean([rs["q_range"]        for rs in rs_list])
        q_std      = np.mean([rs["q_std"]          for rs in rs_list])
        print(f"  {ri+1:<8}  {alive:>6.1f}  {s_per_c:>10.1f}  {q_range:>9.3f}  {q_std:>7.3f}")

    # Depth histogram (aggregate across all positions)
    max_d = max(len(r["depth_counts"]) for r in results)
    agg_depths = np.zeros(max_d, dtype=np.float64)
    for r in results:
        dc = r["depth_counts"]
        agg_depths[:len(dc)] += dc
    total_nodes = agg_depths.sum()

    print(f"\n  TREE DEPTH DISTRIBUTION  (fraction of all nodes)")
    for d in range(min(max_d, 20)):
        frac = agg_depths[d] / total_nodes if total_nodes > 0 else 0
        bar  = "#" * int(round(frac * 40))
        print(f"    depth {d:2d}:  {frac:5.3f}  {bar}")
    if max_d > 20:
        deep_frac = agg_depths[20:].sum() / total_nodes
        print(f"    depth 20+: {deep_frac:5.3f}")

    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--sims",      type=int, default=1024)
    ap.add_argument("--ply",       type=int, default=30,
                    help="Ply at which to sample midgame positions (default 30)")
    args = ap.parse_args()

    print("Loading extension and model...")
    ext  = hive_gpu.load_extension()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net  = HiveFNN(ckpt["net_config"]).cuda().eval()
    net.load_state_dict(ckpt["model_state_dict"])
    cfg  = FNNMCTSConfig(
        num_simulations=args.sims,
        batch_size=1,
        wave_parallel=True,
        expansion_mask=0,
    )
    orch = FNNMCTSOrchestrator(net, cfg)

    print(f"Generating positions (ply={args.ply}, {N_GEN_GAMES} games)...")
    positions = generate_positions(ext, N_GEN_GAMES, args.ply)
    if len(positions) > args.positions:
        np.random.shuffle(positions)
        positions = positions[:args.positions]
    print(f"  Got {len(positions)} positions.  Profiling Gumbel search...")

    results = []
    for i, pos in enumerate(positions):
        if i % 20 == 0:
            print(f"  {i}/{len(positions)}...", flush=True)
        r = gumbel_search_instrumented(orch, pos, args.sims)
        if r:
            results.append(r)

    report(results, args.sims)


if __name__ == "__main__":
    main()
