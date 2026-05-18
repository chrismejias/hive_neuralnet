"""Check how well parent Q predicts top-k children Q values.

For each sampled midgame position:
  - Evaluate the root: get value v_root and policy priors over legal moves
  - Identify top-K legal moves by prior
  - Apply each move to get the child state
  - Evaluate each child state: get v_child (from child player's perspective)
  - The parent's prediction for child k is: -v_root  (two-player flip)
  - Compare -v_root vs. v_child for each top-K child

Reports Pearson r, Spearman rho, MAE, bias, sign-agreement across positions,
broken down by child rank (top-1 vs top-2 etc.), plus within-position rank
correlation (does policy order agree with parent-value order?).
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from scipy.stats import spearmanr

import hive_gpu
from hive_fnn.fnn_network import HiveFNN
from hive_fnn.fnn_mcts_orchestrator import FNNMCTSOrchestrator, FNNMCTSConfig

CHECKPOINT  = "checkpoints_fnn_deterministic/hive_fnn_checkpoint_0250.pt"
N_GEN_GAMES = 300
PLY_TARGET  = 30
TOP_K       = 5


def generate_positions(ext, n_games: int, ply_target: int) -> list[np.ndarray]:
    dev    = "cuda"
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


def eval_batch(orch, states_np: list[np.ndarray], ext):
    """Evaluate a batch of states; return (priors_list, values_np)."""
    dev = "cuda"
    B   = len(states_np)
    states = torch.from_numpy(np.stack(states_np)).to(dev)
    legal_moves, num_legal, root_features = \
        ext.generate_legal_moves_and_fnn_features_batch(states, B)
    with torch.no_grad():
        priors, values = orch._eval_states(states, legal_moves, num_legal, B, root_features)
    nl_np     = num_legal.cpu().numpy()
    priors_np = priors.cpu().numpy()
    vals_np   = values.cpu().numpy()
    legal_np  = legal_moves.cpu().numpy()
    return priors_np, vals_np, nl_np, legal_np


def pearson(x, y):
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spr(x, y):
    if len(x) < 2:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions",  type=int, default=N_GEN_GAMES)
    parser.add_argument("--ply",        type=int, default=PLY_TARGET)
    parser.add_argument("--topk",       type=int, default=TOP_K)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    print("Loading extension and model...")
    ext  = hive_gpu.load_extension()
    ckpt = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg_dict = ckpt.get("config", {})
    net  = HiveFNN(**{k: v for k, v in cfg_dict.items()
                      if k in HiveFNN.__init__.__code__.co_varnames})
    net.load_state_dict(ckpt["model_state_dict"])
    net.cuda().eval()
    orch = FNNMCTSOrchestrator(net, config=FNNMCTSConfig())

    K = args.topk
    print(f"Generating {args.positions} positions (ply={args.ply})...")
    root_states_np = generate_positions(ext, args.positions, args.ply)
    print(f"  Got {len(root_states_np)} positions.")

    # ── Evaluate all root positions in one batch ──────────────────────
    root_priors, root_vals, root_nl, root_legal = eval_batch(orch, root_states_np, ext)
    N = len(root_states_np)

    # For each rank, apply the rank-th best move (by prior) and evaluate the child
    # child_vals[rank][i] = value of child from child's POV, or NaN if unavailable
    child_vals = np.full((K, N), np.nan)

    for rank in range(K):
        child_states_np = []
        valid_idx       = []

        for i in range(N):
            n = int(root_nl[i])
            if n <= rank:
                continue
            order = np.argsort(-root_priors[i, :n])
            slot  = int(order[rank])
            child_states_np.append(root_states_np[i].copy())
            valid_idx.append((i, slot, root_legal[i, slot]))

        if not child_states_np:
            continue

        # Apply moves in a GPU batch
        M      = len(child_states_np)
        dev    = "cuda"
        cstates = torch.from_numpy(np.stack(child_states_np)).to(dev)
        moves   = torch.zeros(M, ext.SIZEOF_GPU_MOVE, dtype=torch.uint8, device=dev)
        for j, (i, slot, mv_bytes) in enumerate(valid_idx):
            moves[j] = torch.from_numpy(mv_bytes.copy())
        ext.apply_moves_batch(cstates, moves, M)

        clegal, cnlegal, cfeatures = \
            ext.generate_legal_moves_and_fnn_features_batch(cstates, M)
        with torch.no_grad():
            _, cvals = orch._eval_states(cstates, clegal, cnlegal, M, cfeatures)
        cv_np = cvals.cpu().numpy()

        for j, (i, slot, _) in enumerate(valid_idx):
            child_vals[rank, i] = float(cv_np[j])

        print(f"  rank {rank+1}: evaluated {M} children.")

    # ── Build flat arrays ─────────────────────────────────────────────
    pred_all, act_all, rank_all = [], [], []
    for rank in range(K):
        for i in range(N):
            if np.isnan(child_vals[rank, i]):
                continue
            pred_all.append(-root_vals[i])       # parent prediction (negated flip)
            act_all.append(child_vals[rank, i])  # child actual value
            rank_all.append(rank)

    pred_all = np.array(pred_all)
    act_all  = np.array(act_all)
    rank_all = np.array(rank_all)

    # ── Print report ──────────────────────────────────────────────────
    print()
    print("=" * 68)
    print(f"  Parent Q → Child Q Correlation  ({N} positions, top-{K} children)")
    print("=" * 68)
    print()
    print("  Parent prediction = -v_root  (negated for two-player flip)")
    print("  Child actual      =  v_child (value head at child position)")
    print()
    n_total = len(pred_all)
    print(f"  {'Metric':<36}  {'Value':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Samples (all ranks combined)':<36}  {n_total:>10d}")
    print(f"  {'Pearson r':<36}  {pearson(pred_all, act_all):>10.4f}")
    print(f"  {'Spearman rho':<36}  {spr(pred_all, act_all):>10.4f}")
    mae_v = float(np.mean(np.abs(pred_all - act_all)))
    bias  = float(np.mean(pred_all - act_all))
    print(f"  {'MAE':<36}  {mae_v:>10.4f}")
    print(f"  {'Bias  (pred - actual)':<36}  {bias:>10.4f}")
    sign_ag = float(np.mean(np.sign(pred_all) == np.sign(act_all)))
    print(f"  {'Sign agreement':<36}  {sign_ag:>10.1%}")

    print()
    errs = np.abs(pred_all - act_all)
    print(f"  |error| percentiles:")
    for p in [25, 50, 75, 90, 95]:
        print(f"    p{p:<3}: {np.percentile(errs, p):.4f}")

    print()
    print(f"  Per-rank breakdown  (rank = child order by policy prior):")
    print(f"  {'Rank':<7}  {'N':>5}  {'Pearson r':>10}  {'Spearman':>10}  "
          f"{'MAE':>8}  {'Bias':>8}  {'Sign%':>7}")
    print(f"  {'-'*67}")
    for rank in range(K):
        m = rank_all == rank
        x, y = pred_all[m], act_all[m]
        s_ag = float(np.mean(np.sign(x) == np.sign(y))) if len(x) else float("nan")
        print(f"  #{rank+1:<6}  {m.sum():>5}  {pearson(x,y):>10.4f}  "
              f"{spr(x,y):>10.4f}  {float(np.mean(np.abs(x-y))):>8.4f}  "
              f"{float(np.mean(x-y)):>8.4f}  {s_ag:>7.1%}")

    # ── Within-position: does prior rank → value rank? ────────────────
    within_rhos = []
    for i in range(N):
        cv = [child_vals[r, i] for r in range(K) if not np.isnan(child_vals[r, i])]
        if len(cv) < 2:
            continue
        k_use = len(cv)
        # prior rank: 0 = best prior; value rank from parent's POV = lower child value = better
        prior_ranks = list(range(k_use))
        value_ranks = list(np.argsort(cv))   # ascending child value = parent prefers
        r, _ = spearmanr(prior_ranks, value_ranks)
        within_rhos.append(float(r))

    if within_rhos:
        arr = np.array(within_rhos)
        print()
        print(f"  Within-position: prior rank vs. parent-preferred value rank")
        print(f"  (rho=+1 → policy correctly orders children by parent value)")
        print(f"  Mean Spearman rho : {arr.mean():+.4f}")
        print(f"  Std               : {arr.std():.4f}")
        print(f"  Fraction rho > 0  : {(arr > 0).mean():.1%}")

    print()


if __name__ == "__main__":
    main()
