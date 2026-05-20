"""
Diagnostic: compare deterministic vs PUCT non-root exploration.

Runs both selection modes from the same positions using the same model
checkpoint and measures:
  1. Unique nodes visited at each tree depth
  2. Visit distribution entropy and Gini at each depth
  3. Simulation path lengths
  4. Root-child Q estimate spread and coverage
  5. Fraction of simulations that reach a previously-unseen node ("new" sims)

Usage:
  python3.10 fnn_search_diagnostic_nonroot.py \
      --checkpoint checkpoints_fnn_deterministic/hive_fnn_checkpoint_0300.pt \
      --sims 512 --positions 8 --warmup-moves 12
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")
import hive_gpu
from hive_fnn.fnn_mcts_orchestrator import FNNMCTSConfig, FNNMCTSOrchestrator
from hive_fnn.fnn_network import FNNConfig, HiveFNN


# ── helpers ────────────────────────────────────────────────────────────────────

def load_net(path: str) -> HiveFNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = HiveFNN(FNNConfig.large())
    net.load_state_dict(ckpt["model_state_dict"])
    return net.cuda().eval()


def build_orch(net: HiveFNN, num_sims: int, det: bool) -> FNNMCTSOrchestrator:
    cfg = FNNMCTSConfig(
        num_simulations=num_sims,
        batch_size=1,
        wave_parallel=True,
        wave_size=4,
        deterministic_non_root=det,
        virtual_q_penalty=0.25,
        non_root_sigma=4.0,
        expansion_mask=0,
        rebase_tree_each_move=False,  # keep full tree for inspection
    )
    return FNNMCTSOrchestrator(net, cfg)


def gini(arr: np.ndarray) -> float:
    """Gini coefficient — 0 = perfectly equal, 1 = all mass on one node."""
    a = np.sort(arr.astype(float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def entropy(arr: np.ndarray) -> float:
    a = arr[arr > 0].astype(float)
    if len(a) == 0:
        return 0.0
    p = a / a.sum()
    return float(-np.sum(p * np.log(p)))


def extract_depth_visits(tree: dict, game: int = 0) -> dict[int, np.ndarray]:
    """BFS from root; return {depth: array of visit counts for nodes at that depth}."""
    root = int(tree["root_node"][game].item())
    fc = tree["first_child"][game].cpu().numpy()
    nc = tree["num_children"][game].cpu().numpy()
    vc = tree["visit_count"][game].cpu().numpy()

    depth_visits: dict[int, list[int]] = defaultdict(list)
    queue = [(root, 0)]
    seen: set[int] = set()

    while queue:
        node, depth = queue.pop(0)
        if node < 0 or node in seen:
            continue
        seen.add(node)
        depth_visits[depth].append(int(vc[node]))

        first = int(fc[node])
        num   = int(nc[node])
        if first >= 0 and num > 0:
            for child in range(first, min(first + num, len(fc))):
                if child not in seen:
                    queue.append((child, depth + 1))

    return {d: np.array(v) for d, v in depth_visits.items()}


def run_simulations_collect_paths(
    orch: FNNMCTSOrchestrator,
    states: torch.Tensor,
    num_sims: int,
    det: bool,
) -> tuple[dict, list[int]]:
    """
    Run num_sims simulations and return (final tree dict, list of path lengths).
    Also returns the count of simulations that reached a node not previously seen
    (i.e., expanded a genuinely new leaf vs re-entering an existing node).
    """
    ext  = orch.ext
    cfg  = orch.config
    B    = 1
    dev  = "cuda"

    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    legal_moves, num_legal, root_features = (
        ext.generate_legal_moves_and_fnn_features_batch(states, B)
    )
    priors, root_vals, child_q = orch._eval_states(
        states, legal_moves, num_legal, B, root_features
    )

    game_active = torch.ones(B, dtype=torch.int8, device=dev)
    orch._expand_root_if_needed(
        tree, states, legal_moves, num_legal, priors, child_q, game_active, B
    )

    n_per = num_legal.to(torch.int64)
    valid = torch.arange(orch._max_legal, device=dev).unsqueeze(0) < n_per.unsqueeze(1)
    alive_mask = valid.to(torch.int8)

    W = cfg.wave_size
    num_waves = (num_sims + W - 1) // W

    all_path_lens: list[int] = []
    new_leaf_sims = 0           # sims that expanded a brand-new node

    node_count_before = int(tree["node_count"][0].item())

    state_size = states.shape[1]
    leaf_states = torch.zeros(W * B, state_size, dtype=torch.uint8, device=dev)

    for wave in range(num_waves):
        actual_w = min(W, num_sims - wave * W)
        total    = actual_w * B

        leaf_idx, move_paths, path_lens, vl_paths, vl_lens = (
            ext.mcts_select_with_root_mask_batch(
                *orch._tree_args(tree),
                tree["child_init_q"],
                game_active, tree["root_node"],
                alive_mask, orch._max_legal,
                cfg.c_puct, B, actual_w, orch._max_nodes,
                det, cfg.non_root_sigma if det else 0.0, 0.0,
            )
        )

        # Record path lengths for this wave
        plens = path_lens[:total].cpu().numpy()
        all_path_lens.extend(plens.tolist())

        # Count sims that went deeper than depth 0 (hit an actual leaf)
        fc_root = int(tree["first_child"][0, int(tree["root_node"][0].item())].item())

        # Replay + eval + expand
        ext.mcts_replay_batch(
            states, leaf_states[:total],
            move_paths[:total], path_lens[:total], leaf_idx[:total],
            B, total,
        )
        results = ext.check_results_batch(leaf_states[:total], total)
        lm2, nl2, lf2 = ext.generate_legal_moves_and_fnn_features_batch(
            leaf_states[:total], total
        )
        priors_leaf, leaf_vals, child_q_leaf = orch._eval_states(
            leaf_states[:total], lm2, nl2, total, lf2
        )

        nc_before = int(tree["node_count"][0].item())
        ext.mcts_expand_and_backprop_dense_priors_batch(
            *orch._tree_args(tree),
            tree["child_init_q"], child_q_leaf,
            leaf_idx[:total], leaf_states[:total],
            lm2, nl2, priors_leaf, results,
            leaf_vals, vl_paths[:total], vl_lens[:total],
            B, total, orch._max_nodes, 0.0,
        )
        nc_after = int(tree["node_count"][0].item())
        new_leaf_sims += (nc_after - nc_before)   # nodes created = genuinely new leaves

    return tree, all_path_lens, new_leaf_sims


def print_stats(label: str, depth_visits: dict, path_lens: list[int], new_leaves: int, num_sims: int):
    print(f"\n  ── {label} ──")

    # depth table
    print(f"  {'depth':>5}  {'nodes':>6}  {'entropy':>8}  {'max-entropy':>11}  "
          f"{'ent%':>5}  {'gini':>6}  {'max-frac':>8}")
    for depth in sorted(depth_visits.keys()):
        v   = depth_visits[depth]
        tot = v.sum()
        if tot == 0:
            continue
        n   = len(v)
        ent = entropy(v)
        max_ent = np.log(n) if n > 1 else 0.0
        g   = gini(v)
        mf  = v.max() / tot
        pct = 100 * ent / max_ent if max_ent > 0 else 0.0
        print(f"  {depth:>5}  {n:>6}  {ent:>8.3f}  {max_ent:>11.3f}  "
              f"{pct:>5.1f}%  {g:>6.3f}  {mf:>8.3f}")

    # path length summary
    pl = np.array(path_lens)
    print(f"\n  Path depth:  mean={pl.mean():.2f}  std={pl.std():.2f}  "
          f"min={pl.min()}  max={pl.max()}")

    # new-leaf rate
    print(f"  New nodes created: {new_leaves}/{num_sims} sims "
          f"({100*new_leaves/num_sims:.1f}%)")


def root_child_stats(tree: dict) -> None:
    game = 0
    root = int(tree["root_node"][game].item())
    fc   = int(tree["first_child"][game, root].item())
    nc   = int(tree["num_children"][game, root].item())
    if fc < 0 or nc == 0:
        print("  (root not expanded)")
        return

    vc = tree["visit_count"][game, fc:fc + nc].cpu().numpy().astype(float)
    tv = tree["total_value"][game, fc:fc + nc].cpu().numpy()
    pr = tree["prior"][game, fc:fc + nc].cpu().numpy()

    visited  = vc > 0
    q_mcts   = np.where(visited, -tv / np.maximum(vc, 1), np.nan)
    q_init   = tree["child_init_q"][game, fc:fc + nc].cpu().numpy()

    print(f"\n  Root children: {nc} total, "
          f"{visited.sum()} visited ({100*visited.mean():.0f}%)")
    print(f"  Visit dist:  max={int(vc.max())}  min={int(vc[visited].min()) if visited.any() else 0}  "
          f"mean={vc[visited].mean():.1f}  entropy={entropy(vc):.3f}  gini={gini(vc):.3f}")
    if visited.sum() > 1:
        q_v = q_mcts[visited]
        print(f"  Q(MCTS):     mean={np.nanmean(q_v):.4f}  std={np.nanstd(q_v):.4f}  "
              f"range=[{np.nanmin(q_v):.4f}, {np.nanmax(q_v):.4f}]")
        corr = np.corrcoef(pr[visited], q_v)[0, 1]
        print(f"  prior vs Q corr (visited children): {corr:.3f}")
    print(f"  Q_init:      mean={q_init.mean():.4f}  std={q_init.std():.4f}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sims",           type=int, default=512)
    ap.add_argument("--positions",      type=int, default=6)
    ap.add_argument("--warmup-moves",   type=int, default=12)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ext = hive_gpu.load_extension()
    net = load_net(args.checkpoint)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Simulations: {args.sims}  |  Positions: {args.positions}  "
          f"|  Warmup moves: {args.warmup_moves}")

    # accumulators for aggregate stats
    agg: dict[str, dict[str, list]] = {
        "PUCT": defaultdict(list),
        "DET":  defaultdict(list),
    }

    for pos_idx in range(args.positions):
        print(f"\n{'='*60}")
        print(f"Position {pos_idx + 1}")

        # Build a mid-game position with random moves
        states = ext.create_initial_states(1, 0)
        for _ in range(args.warmup_moves):
            lm, nl = ext.generate_legal_moves_batch(states, 1)
            n = int(nl[0].item())
            if n == 0:
                break
            idx = np.random.randint(0, n)
            mv  = torch.zeros(1, ext.SIZEOF_GPU_MOVE, dtype=torch.uint8, device="cuda")
            mv[0] = lm[0, idx]
            ext.apply_moves_batch(states, mv, 1)

        for det, label, key in [(False, "PUCT (non-det)", "PUCT"), (True, "Deterministic", "DET")]:
            orch = build_orch(net, args.sims, det)
            tree, path_lens, new_leaves = run_simulations_collect_paths(
                orch, states.clone(), args.sims, det
            )
            dv = extract_depth_visits(tree)
            print_stats(label, dv, path_lens, new_leaves, args.sims)
            root_child_stats(tree)

            # Accumulate aggregate stats
            agg[key]["new_leaf_rate"].append(new_leaves / args.sims)
            agg[key]["mean_depth"].append(np.mean(path_lens))
            if 2 in dv:
                agg[key]["depth2_nodes"].append(len(dv[2]))
                agg[key]["depth2_gini"].append(gini(dv[2]))
                agg[key]["depth2_ent_pct"].append(
                    100 * entropy(dv[2]) / np.log(len(dv[2])) if len(dv[2]) > 1 else 0
                )

    # Print aggregate summary
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY (mean across positions)")
    print(f"{'Metric':<35}  {'PUCT':>10}  {'DET':>10}")
    metrics = [
        ("new_leaf_rate",   "New-leaf rate"),
        ("mean_depth",      "Mean sim depth"),
        ("depth2_nodes",    "Unique nodes @ depth 2"),
        ("depth2_gini",     "Gini @ depth 2"),
        ("depth2_ent_pct",  "Entropy % of max @ depth 2"),
    ]
    for key, name in metrics:
        puct_vals = agg["PUCT"].get(key, [])
        det_vals  = agg["DET"].get(key, [])
        pv = f"{np.mean(puct_vals):.3f}" if puct_vals else "n/a"
        dv = f"{np.mean(det_vals):.3f}"  if det_vals  else "n/a"
        print(f"  {name:<33}  {pv:>10}  {dv:>10}")


if __name__ == "__main__":
    main()
