"""
Diagnostic: compare deterministic vs PUCT non-root exploration for PRS v2.

Runs both selection modes from the same positions using the same PRS v2
checkpoint and measures:
  1. Unique nodes visited at each tree depth
  2. Visit distribution entropy and Gini at each depth
  3. Simulation path lengths
  4. Root-child Q estimate spread and coverage
  5. Fraction of simulations that reach a previously-unseen node ("new" sims)

Usage:
  python3.11 prs_search_diagnostic_nonroot.py \
      --checkpoint checkpoints_prs_v2/prs_v2_iter_0600.pt \
      --sims 512 --positions 6 --warmup-moves 12
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")
import hive_gpu
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2


def load_net(path: str) -> tuple[HivePRSTransformerV2, PRSConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net_config = ckpt.get("net_config", PRSConfig.small())
    net = HivePRSTransformerV2(net_config)
    net.load_state_dict(ckpt["model_state"])
    return net.cuda().eval(), net_config


def build_orch(net: HivePRSTransformerV2, num_sims: int, det: bool) -> PRSMCTSOrchestratorV2:
    cfg = PRSMCTSConfigV2(
        num_simulations=num_sims,
        batch_size=1,
        wave_parallel=True,
        deterministic_non_root=det,
        virtual_q_penalty=0.25,
        non_root_sigma=1.0,
        expansion_mask=0,
        rebase_tree_each_move=False,
    )
    return PRSMCTSOrchestratorV2(net, cfg)


def gini(arr: np.ndarray) -> float:
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
        num = int(nc[node])
        if first >= 0 and num > 0:
            for child in range(first, min(first + num, len(fc))):
                if child not in seen:
                    queue.append((child, depth + 1))

    return {d: np.array(v) for d, v in depth_visits.items()}


def run_simulations_collect_paths(
    orch: PRSMCTSOrchestratorV2,
    states: torch.Tensor,
    num_sims: int,
    det: bool,
) -> tuple[dict, list[int], int]:
    """Run the same PRS v2 root-slot search used in self-play and collect stats."""
    ext = orch.ext
    cfg = orch.config
    B = 1
    dev = "cuda"

    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    prs_batch = orch.encoder.encode_batch(states, B)
    legal_t, nlegal_t = ext.generate_legal_moves_batch(states, B)
    if int(nlegal_t[0].item()) <= 0:
        return tree, [], 0

    kernel_out = orch._classify_kernel(states, legal_t, nlegal_t, B)
    slot_of_legal_t = kernel_out[8]
    policy_logits_813, root_values = orch._net_forward(prs_batch, kernel_out, B)
    policy_logits_813 = policy_logits_813.float()

    priors_per_legal, root_logit_per_legal = orch._build_legal_priors_v2(
        policy_logits_813, slot_of_legal_t, B,
    )

    active = [True]
    active_t = torch.tensor([1], dtype=torch.int8, device=dev)
    nlegal_np = nlegal_t.cpu().numpy()
    nlegal_t_gpu = nlegal_t.to(torch.int64)
    slot_idx_t = torch.arange(orch._max_legal, device=dev).unsqueeze(0)
    valid_slot = slot_idx_t < nlegal_t_gpu.unsqueeze(1)

    u = torch.rand(B, orch._max_legal, device=dev).clamp(1e-4, 1 - 1e-4)
    gumbel = -torch.log(-torch.log(u))
    gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
    perturbed = gumbel + root_logit_per_legal
    _, topk_slots = torch.topk(perturbed, min(16, int(nlegal_np[0])), dim=1)

    orch._expand_root_if_needed(
        tree, states, legal_t, nlegal_t, priors_per_legal, active_t, B,
    )
    orch._apply_root_dirichlet(tree, B, active)

    candidate_slots = topk_slots.to(torch.int32)
    candidate_valid = torch.gather(valid_slot, 1, candidate_slots.long())
    candidate_slots = torch.where(
        candidate_valid,
        candidate_slots,
        torch.full_like(candidate_slots, -1),
    )

    sims_per_round = max(1, cfg.num_simulations // 4)
    all_path_lens: list[int] = []
    new_leaf_sims = 0
    state_size = states.shape[1]
    # Root-slot selection can replay up to 16 candidates across up to 8 waves.
    leaf_states = torch.zeros(16 * 16 * B, state_size, dtype=torch.uint8, device=dev)

    for round_i in range(4):
        root_slots = candidate_slots
        candidate_valid = root_slots >= 0
        num_candidates = int(root_slots.shape[1])
        sims_per_candidate = max(1, sims_per_round // num_candidates)
        round_wave_size = (
            (1, 2, 4, 8)[round_i] if cfg.wave_parallel else 1
        )

        num_waves = int(np.ceil(sims_per_candidate / round_wave_size))
        q_penalty = (
            float(cfg.virtual_q_penalty)
            if cfg.deterministic_non_root and round_wave_size > 1
            else 0.0
        )
        for wave in range(num_waves):
            actual_w = min(round_wave_size, sims_per_candidate - wave * round_wave_size)
            total = actual_w * num_candidates * B

            pre_first_child = tree["first_child"].clone()
            leaf_idx, move_paths, path_lens, vl_paths, vl_lens = (
                ext.mcts_select_with_root_slots_batch(
                    *orch._tree_args(tree),
                    tree["child_init_q"],
                    active_t, tree["root_node"],
                    root_slots, num_candidates, orch._max_legal,
                    cfg.c_puct, B, actual_w, orch._max_nodes,
                    cfg.deterministic_non_root, cfg.non_root_sigma, q_penalty,
                )
            )
            all_path_lens.extend(path_lens[:total].cpu().numpy().tolist())
            leaf_idx_cpu = leaf_idx[:total].long().clamp(min=0)
            new_leaf_sims += int(
                (pre_first_child[0, leaf_idx_cpu] < 0).sum().item()
            )

            ext.mcts_replay_batch(
                states, leaf_states[:total],
                move_paths[:total], path_lens[:total], leaf_idx[:total],
                B, total,
            )
            results = ext.check_results_batch(leaf_states[:total], total)
            legal_moves, num_legal = ext.generate_legal_moves_batch(
                leaf_states[:total], total,
            )
            prs_leaf = orch.encoder.encode_batch(leaf_states[:total], total)
            kernel_leaf = orch._classify_kernel(
                leaf_states[:total], legal_moves, num_legal, total,
            )
            slot_of_leaf_t = kernel_leaf[8]
            leaf_logits, leaf_values = orch._net_forward(prs_leaf, kernel_leaf, total)
            leaf_logits = leaf_logits.float()
            leaf_values = leaf_values.squeeze(-1).float()
            priors_leaf, _ = orch._build_legal_priors_v2(
                leaf_logits, slot_of_leaf_t, total,
            )
            child_q_leaf = leaf_values.unsqueeze(1).expand(total, orch._max_legal).contiguous()
            ext.mcts_expand_and_backprop_dense_priors_batch(
                *orch._tree_args(tree),
                tree["child_init_q"], child_q_leaf,
                leaf_idx[:total], leaf_states[:total],
                legal_moves, num_legal, priors_leaf, results,
                leaf_values, vl_paths[:total], vl_lens[:total],
                B, total, orch._max_nodes, q_penalty,
            )

        if num_candidates <= 1:
            continue
        per_game_keep = (candidate_valid.sum(dim=1) // 2).clamp(min=1)
        max_keep = num_candidates // 2

        cand_visits, cand_q = orch._gather_root_candidate_stats(tree, B, root_slots)
        sigma_norm = (cfg.c_visit + cand_visits.max()) * cfg.c_scale
        cand_idx = root_slots.long().clamp(min=0)
        cand_score = (
            torch.gather(gumbel + root_logit_per_legal, 1, cand_idx)
            + sigma_norm * cand_q
        ).float()
        cand_score = torch.where(
            candidate_valid, cand_score,
            torch.full_like(cand_score, -1e30),
        )
        _, keep_pos = torch.topk(cand_score, max_keep, dim=1)
        keep_rank = orch._keep_rank(max_keep)
        keep_valid = keep_rank < per_game_keep.unsqueeze(1)
        new_slots = torch.gather(root_slots, 1, keep_pos)
        candidate_slots = torch.where(
            keep_valid,
            new_slots,
            torch.full_like(new_slots, -1),
        )

    return tree, all_path_lens, new_leaf_sims


def print_stats(label: str, depth_visits: dict, path_lens: list[int], new_leaves: int, num_sims: int) -> None:
    print(f"\n  ── {label} ──")
    print(f"  {'depth':>5}  {'nodes':>6}  {'entropy':>8}  {'max-entropy':>11}  "
          f"{'ent%':>5}  {'gini':>6}  {'max-frac':>8}")
    for depth in sorted(depth_visits.keys()):
        v = depth_visits[depth]
        tot = v.sum()
        if tot == 0:
            continue
        n = len(v)
        ent = entropy(v)
        max_ent = np.log(n) if n > 1 else 0.0
        g = gini(v)
        mf = v.max() / tot
        pct = 100 * ent / max_ent if max_ent > 0 else 0.0
        print(f"  {depth:>5}  {n:>6}  {ent:>8.3f}  {max_ent:>11.3f}  "
              f"{pct:>5.1f}%  {g:>6.3f}  {mf:>8.3f}")

    pl = np.array(path_lens) if path_lens else np.array([0])
    print(f"\n  Path depth:  mean={pl.mean():.2f}  std={pl.std():.2f}  "
          f"min={pl.min()}  max={pl.max()}")
    print(f"  New leaves reached: {new_leaves}/{num_sims} sims "
          f"({100*new_leaves/num_sims:.1f}%)")


def root_child_stats(tree: dict) -> None:
    game = 0
    root = int(tree["root_node"][game].item())
    fc = int(tree["first_child"][game, root].item())
    nc = int(tree["num_children"][game, root].item())
    if fc < 0 or nc == 0:
        print("  (root not expanded)")
        return

    vc = tree["visit_count"][game, fc:fc + nc].cpu().numpy().astype(float)
    tv = tree["total_value"][game, fc:fc + nc].cpu().numpy()
    pr = tree["prior"][game, fc:fc + nc].cpu().numpy()

    visited = vc > 0
    q_mcts = np.where(visited, -tv / np.maximum(vc, 1), np.nan)
    q_init = tree["child_init_q"][game, fc:fc + nc].cpu().numpy()

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


def nonroot_override_stats(tree: dict, sigma: float) -> dict[str, float]:
    """Quantify how often deterministic non-root selection departs from prior-only ordering."""
    game = 0
    root = int(tree["root_node"][game].item())
    fc = tree["first_child"][game].cpu().numpy()
    nc = tree["num_children"][game].cpu().numpy()
    vc = tree["visit_count"][game].cpu().numpy().astype(float)
    tv = tree["total_value"][game].cpu().numpy()
    pri = tree["prior"][game].cpu().numpy().astype(float)
    vq = tree["virtual_q_penalty"][game].cpu().numpy().astype(float)
    init_q = tree["child_init_q"][game].cpu().numpy().astype(float)

    total_weight = 0.0
    override_weight = 0.0
    rank_weighted = 0.0
    prior_ratio_weighted = 0.0
    q_gain_weighted = 0.0
    score_gain_weighted = 0.0
    nodes = 0
    overrides = 0

    for node in range(len(fc)):
        if node == root:
            continue
        parent_visits = float(vc[node])
        if parent_visits <= 0:
            continue
        first = int(fc[node])
        num = int(nc[node])
        if first < 0 or num <= 0:
            continue

        child_end = min(first + num, len(fc))
        child_idx = np.arange(first, child_end)
        child_vis = vc[child_idx]
        child_prior = np.maximum(pri[child_idx], 1e-20)
        child_q = np.where(
            child_vis > 0,
            -tv[child_idx] / np.maximum(child_vis, 1.0),
            init_q[child_idx],
        ) - vq[child_idx]
        child_score = np.log(child_prior) + sigma * child_q

        prior_best = int(np.argmax(child_prior))
        score_best = int(np.argmax(child_score))
        selected_prior = float(child_prior[score_best])
        best_prior = float(child_prior[prior_best])
        selected_q = float(child_q[score_best])
        best_q = float(child_q[prior_best])

        rank_of_selected = 1 + int(np.sum(child_prior > selected_prior))
        q_gain = selected_q - best_q
        score_gain = float(child_score[score_best] - child_score[prior_best])

        total_weight += parent_visits
        rank_weighted += parent_visits * rank_of_selected
        prior_ratio_weighted += parent_visits * (selected_prior / best_prior if best_prior > 0 else 0.0)
        q_gain_weighted += parent_visits * q_gain
        score_gain_weighted += parent_visits * score_gain
        nodes += 1
        if score_best != prior_best:
            override_weight += parent_visits
            overrides += 1

    if total_weight == 0:
        return {
            "nodes": 0.0,
            "override_rate_weighted": 0.0,
            "override_rate_nodes": 0.0,
            "mean_prior_rank": 0.0,
            "mean_prior_ratio": 0.0,
            "mean_q_gain": 0.0,
            "mean_score_gain": 0.0,
        }

    return {
        "nodes": float(nodes),
        "override_rate_weighted": override_weight / total_weight,
        "override_rate_nodes": overrides / nodes if nodes else 0.0,
        "mean_prior_rank": rank_weighted / total_weight,
        "mean_prior_ratio": prior_ratio_weighted / total_weight,
        "mean_q_gain": q_gain_weighted / total_weight,
        "mean_score_gain": score_gain_weighted / total_weight,
    }


def print_override_stats(stats: dict[str, float]) -> None:
    print(f"\n  NON-ROOT PRIOR OVERRIDE (deterministic only)")
    print(f"  {'Non-root nodes analyzed':<40} {stats['nodes']:>8.0f}")
    print(f"  {'Weighted override rate':<40} {100*stats['override_rate_weighted']:>8.2f}%")
    print(f"  {'Node-count override rate':<40} {100*stats['override_rate_nodes']:>8.2f}%")
    print(f"  {'Mean prior rank of selected child':<40} {stats['mean_prior_rank']:>8.2f}")
    print(f"  {'Mean selected/best prior ratio':<40} {stats['mean_prior_ratio']:>8.3f}")
    print(f"  {'Mean Q gain over prior-best child':<40} {stats['mean_q_gain']:>8.4f}")
    print(f"  {'Mean combined score gain':<40} {stats['mean_score_gain']:>8.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sims", type=int, default=512)
    ap.add_argument(
        "--sims-sweep",
        type=int,
        nargs="*",
        default=None,
        help="Simulation counts to sweep; overrides --sims.",
    )
    ap.add_argument("--positions", type=int, default=6)
    ap.add_argument("--warmup-moves", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ext = hive_gpu.load_extension()
    net, net_cfg = load_net(args.checkpoint)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Net config : d_model={net_cfg.d_model} layers={net_cfg.num_layers} heads={net_cfg.num_heads}")
    print(f"Simulations: {args.sims}  |  Positions: {args.positions}  "
          f"|  Warmup moves: {args.warmup_moves}")

    sweep = args.sims_sweep if args.sims_sweep else [256, 512, 1024, 2048]
    agg: dict[str, dict[str, list]] = {
        "PUCT": defaultdict(list),
        "DET": defaultdict(list),
    }

    for pos_idx in range(args.positions):
        print(f"\n{'='*60}")
        print(f"Position {pos_idx + 1}")

        states = ext.create_initial_states(1, 0)
        for _ in range(args.warmup_moves):
            lm, nl = ext.generate_legal_moves_batch(states, 1)
            n = int(nl[0].item())
            if n == 0:
                break
            idx = np.random.randint(0, n)
            mv = torch.zeros(1, ext.SIZEOF_GPU_MOVE, dtype=torch.uint8, device="cuda")
            mv[0] = lm[0, idx]
            ext.apply_moves_batch(states, mv, 1)

        for det, label, key in [(False, "PUCT (non-det)", "PUCT"), (True, "Deterministic", "DET")]:
            orch = build_orch(net, args.sims, det)
            tree, path_lens, new_leaves = run_simulations_collect_paths(
                orch, states.clone(), args.sims, det,
            )
            dv = extract_depth_visits(tree)
            print_stats(label, dv, path_lens, new_leaves, args.sims)
            root_child_stats(tree)
            if det:
                print_override_stats(nonroot_override_stats(tree, orch.config.non_root_sigma))

            agg[key]["new_leaf_rate"].append(new_leaves / args.sims if args.sims else 0.0)
            agg[key]["mean_depth"].append(np.mean(path_lens) if path_lens else 0.0)
            if 2 in dv:
                agg[key]["depth2_nodes"].append(len(dv[2]))
                agg[key]["depth2_gini"].append(gini(dv[2]))
                agg[key]["depth2_ent_pct"].append(
                    100 * entropy(dv[2]) / np.log(len(dv[2])) if len(dv[2]) > 1 else 0,
                )

    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY (mean across positions)")
    print(f"{'Metric':<35}  {'PUCT':>10}  {'DET':>10}")
    metrics = [
        ("new_leaf_rate", "New-leaf rate"),
        ("mean_depth", "Mean sim depth"),
        ("depth2_nodes", "Unique nodes @ depth 2"),
        ("depth2_gini", "Gini @ depth 2"),
        ("depth2_ent_pct", "Entropy % of max @ depth 2"),
    ]
    for key, name in metrics:
        puct_vals = agg["PUCT"].get(key, [])
        det_vals = agg["DET"].get(key, [])
        pv = f"{np.mean(puct_vals):.3f}" if puct_vals else "n/a"
        dv = f"{np.mean(det_vals):.3f}" if det_vals else "n/a"
        print(f"  {name:<33}  {pv:>10}  {dv:>10}")


if __name__ == "__main__":
    main()
