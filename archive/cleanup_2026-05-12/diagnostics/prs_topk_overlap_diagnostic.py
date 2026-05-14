from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")

import hive_gpu
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2


def load_net(path: str) -> HivePRSTransformerV2:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = HivePRSTransformerV2(ckpt["net_config"]).cuda().eval()
    current = net.state_dict()
    filtered = {}
    for key, tensor in ckpt["model_state"].items():
        if key not in current:
            continue
        if current[key].shape != tensor.shape:
            continue
        filtered[key] = tensor
    net.load_state_dict(filtered, strict=False)
    return net


def analyze_root(
    orch: PRSMCTSOrchestratorV2,
    states: torch.Tensor,
    considered_k: int,
) -> dict:
    B = int(states.shape[0])
    cfg = orch.config
    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    prs_batch = orch.encoder.encode_batch(states, B)
    legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(states, B)
    nlegal_np = nlegal_t.cpu().numpy()

    kernel_out = orch._classify_kernel(states, legal_t, nlegal_t, B)
    slot_of_legal_t = kernel_out[8]
    policy_logits_813, _ = orch._net_forward(prs_batch, kernel_out, B)
    policy_logits_813 = policy_logits_813.float()
    priors_per_legal, root_logit_per_legal = orch._build_legal_priors_v2(
        policy_logits_813, slot_of_legal_t, B,
    )

    max_legal = int(nlegal_np.max()) if nlegal_np.size > 0 else 0
    if max_legal <= 0:
        return {
            "top1_in16": 0,
            "top2_in16": 0,
            "top4_in16": 0,
            "top1_in32": 0,
            "top2_in32": 0,
            "top4_in32": 0,
            "best_in16_rank_sum": 0,
            "best_in16_rank_le_1": 0,
            "best_in16_rank_le_2": 0,
            "best_in16_rank_le_4": 0,
            "best_in16_rank_le_8": 0,
            "best_in16_rank_le_16": 0,
            "best_in16_rank_le_32": 0,
            "best_in16_rank_values": [],
            "count": 0,
        }
    max_k = min(considered_k, max_legal)

    score_basis = root_logit_per_legal
    _, topk_slots = torch.topk(score_basis, max_k, dim=1)
    active_t = torch.ones(B, dtype=torch.int8, device="cuda")

    orch._expand_root_if_needed(
        tree, states, legal_t, nlegal_t, priors_per_legal, active_t, B,
    )

    alive_mask = torch.zeros(B, orch._max_legal, dtype=torch.int8, device="cuda")
    alive_mask.scatter_(1, topk_slots, 1)

    n_rounds = max(1, int(np.ceil(np.log2(max(max_k, 2)))))
    sims_per_round = max(1, cfg.num_simulations // n_rounds)

    for round_i in range(n_rounds):
        round_wave_size = (
            [1, 2, 4, 8][min(round_i, 3)]
            if cfg.wave_parallel else 1
        )
        orch._run_simulations(tree, states, active_t, alive_mask, B, sims_per_round)
        alive_counts = alive_mask.sum(dim=1)
        cur_alive_max = int(alive_counts.max().item())
        if cur_alive_max <= 1:
            continue
        num_keep = max(1, cur_alive_max // 2)
        slot_visits, slot_q = orch._gather_root_child_stats(tree, B)
        max_n = int(slot_visits.max().item())
        sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
        sigma_score = (score_basis + sigma_norm * slot_q).float()
        sigma_score = torch.where(
            alive_mask.bool(), sigma_score, torch.full_like(sigma_score, -1e30),
        )
        _, keep_slots = torch.topk(sigma_score, num_keep, dim=1)
        new_alive = torch.zeros_like(alive_mask)
        new_alive.scatter_(1, keep_slots, 1)
        slot_idx_t = torch.arange(orch._max_legal, device="cuda").unsqueeze(0)
        valid_slot = slot_idx_t < nlegal_t.to(torch.int64).unsqueeze(1)
        alive_mask = new_alive * valid_slot.to(torch.int8)

    leg_visits, leg_q = orch._gather_root_child_stats(tree, B)
    max_n = int(leg_visits.max().item())
    sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
    final_score = (root_logit_per_legal + sigma_norm * leg_q).float()

    result = {
        "top1_in16": 0,
        "top2_in16": 0,
        "top4_in16": 0,
        "top1_in32": 0,
        "top2_in32": 0,
        "top4_in32": 0,
        "best_in16_rank_sum": 0,
        "best_in16_rank_le_1": 0,
        "best_in16_rank_le_2": 0,
        "best_in16_rank_le_4": 0,
        "best_in16_rank_le_8": 0,
        "best_in16_rank_le_16": 0,
        "best_in16_rank_le_32": 0,
        "best_in16_rank_values": [],
        "count": 0,
    }
    prior_np = priors_per_legal.detach().cpu().numpy()
    final_np = final_score.detach().cpu().numpy()
    for i in range(B):
        n_i = int(nlegal_np[i])
        if n_i <= 0:
            continue
        prior_rank = np.argsort(-prior_np[i, :n_i], kind="stable")
        prior_top16 = set(int(x) for x in prior_rank[: min(16, n_i)])
        prior_top32 = set(int(x) for x in prior_rank[: min(32, n_i)])
        final_rank = np.argsort(-final_np[i, :n_i], kind="stable")
        final_rank_map = {int(slot): rank + 1 for rank, slot in enumerate(final_rank)}
        result["top1_in16"] += sum(int(x in prior_top16) for x in final_rank[:1])
        result["top2_in16"] += sum(int(x in prior_top16) for x in final_rank[: min(2, n_i)])
        result["top4_in16"] += sum(int(x in prior_top16) for x in final_rank[: min(4, n_i)])
        result["top1_in32"] += sum(int(x in prior_top32) for x in final_rank[:1])
        result["top2_in32"] += sum(int(x in prior_top32) for x in final_rank[: min(2, n_i)])
        result["top4_in32"] += sum(int(x in prior_top32) for x in final_rank[: min(4, n_i)])
        best_slot_in16 = max(prior_top16, key=lambda slot: final_np[i, slot])
        best_rank = int(final_rank_map[best_slot_in16])
        result["best_in16_rank_sum"] += best_rank
        result["best_in16_rank_values"].append(best_rank)
        result["best_in16_rank_le_1"] += int(best_rank <= 1)
        result["best_in16_rank_le_2"] += int(best_rank <= 2)
        result["best_in16_rank_le_4"] += int(best_rank <= 4)
        result["best_in16_rank_le_8"] += int(best_rank <= 8)
        result["best_in16_rank_le_16"] += int(best_rank <= 16)
        result["best_in16_rank_le_32"] += int(best_rank <= 32)
        result["count"] += 1
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--simulations", type=int, default=1536)
    ap.add_argument("--considered", type=int, default=64)
    ap.add_argument("--expansion-mask", type=int, default=7)
    ap.add_argument("--max-game-length", type=int, default=300)
    args = ap.parse_args()

    net = load_net(args.checkpoint)
    cfg = PRSMCTSConfigV2(
        num_simulations=args.simulations,
        max_num_considered_actions=args.considered,
        batch_size=args.games,
        max_game_length=args.max_game_length,
        expansion_mask=args.expansion_mask,
        wave_parallel=True,
    )
    orch = PRSMCTSOrchestratorV2(net, cfg)
    ext = hive_gpu.load_extension()

    B = args.games
    states = ext.create_initial_states(B, args.expansion_mask)
    active = np.ones(B, dtype=bool)
    move_numbers = np.zeros(B, dtype=np.int32)

    totals = {
        "top1_in16": 0,
        "top2_in16": 0,
        "top4_in16": 0,
        "top1_in32": 0,
        "top2_in32": 0,
        "top4_in32": 0,
        "best_in16_rank_sum": 0,
        "best_in16_rank_le_1": 0,
        "best_in16_rank_le_2": 0,
        "best_in16_rank_le_4": 0,
        "best_in16_rank_le_8": 0,
        "best_in16_rank_le_16": 0,
        "best_in16_rank_le_32": 0,
        "best_in16_rank_values": [],
        "count": 0,
    }

    while bool(active.any()):
        idx = np.flatnonzero(active)
        sub_states = states[idx].clone()
        diag = analyze_root(orch, sub_states, args.considered)
        for k in totals:
            if k == "best_in16_rank_values":
                totals[k].extend(diag[k])
            else:
                totals[k] += diag[k]

        chosen = orch.__class__.__mro__[0]  # silence lint-style complaints
        del chosen
        chosen_moves = []
        # reuse arena-style chooser for the actual move application
        tree = orch._alloc_tree(idx.size)
        orch._reset_tree(tree)
        prs_batch = orch.encoder.encode_batch(sub_states, idx.size)
        legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(sub_states, idx.size)
        nlegal_np = nlegal_t.cpu().numpy()
        turns_np = orch._get_turns(sub_states, idx.size)
        active_sub = [int(n) > 0 for n in nlegal_np]
        win_overrides = orch._check_immediate_wins_v2(sub_states, legal_t, nlegal_np, active_sub, turns_np, idx.size)
        kernel_out = orch._classify_kernel(sub_states, legal_t, nlegal_t, idx.size)
        slot_of_legal_t = kernel_out[8]
        policy_logits_813, _ = orch._net_forward(prs_batch, kernel_out, idx.size)
        policy_logits_813 = policy_logits_813.float()
        priors_per_legal, root_logit_per_legal = orch._build_legal_priors_v2(
            policy_logits_813, slot_of_legal_t, idx.size,
        )
        max_legal = int(nlegal_np.max()) if nlegal_np.size > 0 else 0
        max_k = min(args.considered, max_legal) if max_legal > 0 else 0
        score_basis = root_logit_per_legal
        if max_k > 0:
            _, topk_slots = torch.topk(score_basis, max_k, dim=1)
            active_t = torch.tensor([1 if a else 0 for a in active_sub], dtype=torch.int8, device="cuda")
            orch._expand_root_if_needed(tree, sub_states, legal_t, nlegal_t, priors_per_legal, active_t, idx.size)
            alive_mask = torch.zeros(idx.size, orch._max_legal, dtype=torch.int8, device="cuda")
            alive_mask.scatter_(1, topk_slots, 1)
            n_rounds = max(1, int(np.ceil(np.log2(max(max_k, 2)))))
            sims_per_round = max(1, cfg.num_simulations // n_rounds)
            for round_i in range(n_rounds):
                round_wave_size = [1, 2, 4, 8][min(round_i, 3)]
                orch._run_simulations(tree, sub_states, active_t, alive_mask, idx.size, sims_per_round)
                alive_counts = alive_mask.sum(dim=1)
                cur_alive_max = int(alive_counts.max().item())
                if cur_alive_max <= 1:
                    continue
                num_keep = max(1, cur_alive_max // 2)
                slot_visits, slot_q = orch._gather_root_child_stats(tree, idx.size)
                max_n = int(slot_visits.max().item())
                sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
                sigma_score = (score_basis + sigma_norm * slot_q).float()
                sigma_score = torch.where(alive_mask.bool(), sigma_score, torch.full_like(sigma_score, -1e30))
                _, keep_slots = torch.topk(sigma_score, num_keep, dim=1)
                new_alive = torch.zeros_like(alive_mask)
                new_alive.scatter_(1, keep_slots, 1)
                slot_idx_t = torch.arange(orch._max_legal, device="cuda").unsqueeze(0)
                valid_slot = slot_idx_t < nlegal_t.to(torch.int64).unsqueeze(1)
                alive_mask = new_alive * valid_slot.to(torch.int8)
            leg_visits, _ = orch._gather_root_child_stats(tree, idx.size)
            slot_t = slot_of_legal_t.to(dtype=torch.long)
            valid_legal = slot_t >= 0
            slot_safe = slot_t.clamp(min=0)
            slot_sum = torch.zeros(idx.size, 813, dtype=torch.float32, device="cuda")
            slot_sum.scatter_add_(1, slot_safe, leg_visits.float() * valid_legal.float())
            agg_per_legal = slot_sum.gather(1, slot_safe)
            agg_per_legal = torch.where(valid_legal, agg_per_legal, torch.zeros_like(agg_per_legal))
            agg_per_legal = agg_per_legal * alive_mask.float()
            agg_np = agg_per_legal.cpu().numpy()
        else:
            agg_np = np.zeros((idx.size, 0), dtype=np.float32)

        move_bytes = np.zeros((idx.size, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
        legal_np = legal_t.cpu().numpy()
        for j in range(idx.size):
            n_i = int(nlegal_np[j])
            if n_i <= 0:
                continue
            if j in win_overrides:
                pick = int(win_overrides[j])
            else:
                scores = agg_np[j, :n_i].astype(np.float64, copy=True)
                pick = int(np.argmax(scores)) if scores.size else 0
            move_bytes[j] = legal_np[j, pick]

        ext.apply_moves_batch(sub_states, torch.from_numpy(move_bytes).cuda(), idx.size)
        states[idx] = sub_states
        move_numbers[idx] += 1
        final_results = ext.check_results_batch(states, B).cpu().numpy()
        active = active & (final_results == 0) & (move_numbers < args.max_game_length)

    print(f"positions analyzed: {totals['count']}")
    if totals["count"] == 0:
        return
    top1_total = totals["count"] * 1
    top2_total = totals["count"] * 2
    top4_total = totals["count"] * 4
    print(f"top1 by final score in prior top16: {totals['top1_in16']}/{top1_total} = {totals['top1_in16']/top1_total:.3f}")
    print(f"top1 by final score outside prior top16: {top1_total - totals['top1_in16']}/{top1_total} = {(top1_total - totals['top1_in16'])/top1_total:.3f}")
    print(f"top2 by final score in prior top16: {totals['top2_in16']}/{top2_total} = {totals['top2_in16']/top2_total:.3f}")
    print(f"top2 by final score outside prior top16: {top2_total - totals['top2_in16']}/{top2_total} = {(top2_total - totals['top2_in16'])/top2_total:.3f}")
    print(f"top4 by final score in prior top16: {totals['top4_in16']}/{top4_total} = {totals['top4_in16']/top4_total:.3f}")
    print(f"top4 by final score outside prior top16: {top4_total - totals['top4_in16']}/{top4_total} = {(top4_total - totals['top4_in16'])/top4_total:.3f}")
    print(f"top1 by final score in prior top32: {totals['top1_in32']}/{top1_total} = {totals['top1_in32']/top1_total:.3f}")
    print(f"top1 by final score outside prior top32: {top1_total - totals['top1_in32']}/{top1_total} = {(top1_total - totals['top1_in32'])/top1_total:.3f}")
    print(f"top2 by final score in prior top32: {totals['top2_in32']}/{top2_total} = {totals['top2_in32']/top2_total:.3f}")
    print(f"top2 by final score outside prior top32: {top2_total - totals['top2_in32']}/{top2_total} = {(top2_total - totals['top2_in32'])/top2_total:.3f}")
    print(f"top4 by final score in prior top32: {totals['top4_in32']}/{top4_total} = {totals['top4_in32']/top4_total:.3f}")
    print(f"top4 by final score outside prior top32: {top4_total - totals['top4_in32']}/{top4_total} = {(top4_total - totals['top4_in32'])/top4_total:.3f}")
    rank_values = np.asarray(totals["best_in16_rank_values"], dtype=np.int32)
    print(
        "best move inside prior top16 final-rank stats: "
        f"mean={totals['best_in16_rank_sum']/totals['count']:.2f}, "
        f"median={float(np.median(rank_values)):.1f}"
    )
    print(f"best move inside prior top16 ranked top1 overall: {totals['best_in16_rank_le_1']}/{totals['count']} = {totals['best_in16_rank_le_1']/totals['count']:.3f}")
    print(f"best move inside prior top16 ranked top2 overall: {totals['best_in16_rank_le_2']}/{totals['count']} = {totals['best_in16_rank_le_2']/totals['count']:.3f}")
    print(f"best move inside prior top16 ranked top4 overall: {totals['best_in16_rank_le_4']}/{totals['count']} = {totals['best_in16_rank_le_4']/totals['count']:.3f}")
    print(f"best move inside prior top16 ranked top8 overall: {totals['best_in16_rank_le_8']}/{totals['count']} = {totals['best_in16_rank_le_8']/totals['count']:.3f}")
    print(f"best move inside prior top16 ranked top16 overall: {totals['best_in16_rank_le_16']}/{totals['count']} = {totals['best_in16_rank_le_16']/totals['count']:.3f}")
    print(f"best move inside prior top16 ranked top32 overall: {totals['best_in16_rank_le_32']}/{totals['count']} = {totals['best_in16_rank_le_32']/totals['count']:.3f}")


if __name__ == "__main__":
    main()
