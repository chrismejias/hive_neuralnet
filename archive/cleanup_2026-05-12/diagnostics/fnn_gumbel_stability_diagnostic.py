from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")

import hive_gpu
from hive_fnn.fnn_mcts_orchestrator import (
    FNNMCTSConfig,
    FNNMCTSOrchestrator,
    _GUMBEL_ROUNDS,
    _GUMBEL_WAVE_SCHEDULE,
)
from hive_fnn.fnn_network import HiveFNN


def load_net(path: str) -> HiveFNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = HiveFNN(ckpt["net_config"]).cuda().eval()
    state = ckpt.get("model_state_dict", ckpt.get("model_state"))
    if state is None:
        raise KeyError("checkpoint missing model state")
    net.load_state_dict(state, strict=True)
    return net


def run_root_search(
    orch: FNNMCTSOrchestrator,
    states: torch.Tensor,
    considered_k: int,
    gumbel_seed: int,
) -> dict:
    B = int(states.shape[0])
    cfg = orch.config
    dev = states.device
    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    legal_moves, num_legal, root_features = (
        orch.ext.generate_legal_moves_and_fnn_features_batch(states, B)
    )
    num_legal_np = num_legal.cpu().numpy()
    if int(num_legal_np.max()) <= 0:
        return {"chosen": [], "final_rank": [], "nlegal": num_legal_np, "legal_moves": legal_moves.detach().cpu().numpy()}

    priors_per_legal, _root_values, child_q_per_legal = orch._eval_states(
        states, legal_moves, num_legal, B, root_features,
    )
    del _root_values

    MAX_L = orch._max_legal
    valid_slot = orch._slot_idx < num_legal.to(torch.int64).unsqueeze(1)
    safe_prior = priors_per_legal.clamp_min(1e-30)
    legal_logits = torch.where(valid_slot, safe_prior.log(), torch.full_like(priors_per_legal, -1e30))

    gen = torch.Generator(device=dev)
    gen.manual_seed(gumbel_seed)
    u = torch.rand(B, MAX_L, device=dev, generator=gen).clamp(1e-4, 1 - 1e-4)
    gumbel = -torch.log(-torch.log(u))
    gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
    max_k = min(considered_k, int(num_legal_np.max()))
    perturbed = gumbel + legal_logits
    _, topk_slots = torch.topk(perturbed, max_k, dim=1)

    has_actions = num_legal > 0
    game_active_t = has_actions.to(torch.int8)
    orch._expand_root_if_needed(
        tree, states, legal_moves, num_legal,
        priors_per_legal, child_q_per_legal, game_active_t, B,
    )

    candidate_slots = topk_slots.to(torch.int32)
    candidate_valid = torch.gather(valid_slot, 1, candidate_slots.long())
    candidate_slots = torch.where(
        candidate_valid,
        candidate_slots,
        torch.full_like(candidate_slots, -1),
    )

    sims_per_round = max(1, cfg.num_simulations // _GUMBEL_ROUNDS)
    for round_i in range(_GUMBEL_ROUNDS):
        root_slots = candidate_slots
        candidate_valid = root_slots >= 0
        num_candidates = int(root_slots.shape[1])
        sims_per_candidate = max(1, sims_per_round // max(num_candidates, 1))
        round_wave_size = (
            _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)]
            if cfg.wave_parallel else 1
        )
        orch._run_simulations_for_root_slots(
            tree, states, game_active_t, root_slots, B, sims_per_candidate,
            wave_size=round_wave_size,
        )
        if num_candidates <= 1:
            continue
        per_game_keep = (candidate_valid.sum(dim=1) // 2).clamp(min=1)
        max_keep = max(1, num_candidates // 2)

        cand_visits, cand_q = orch._gather_root_candidate_stats(tree, B, root_slots)
        sigma_norm = (cfg.c_visit + cand_visits.max()) * cfg.c_scale
        cand_idx = root_slots.long().clamp(min=0)
        cand_score = (
            torch.gather(gumbel + legal_logits, 1, cand_idx)
            + sigma_norm * cand_q
        ).float()
        cand_score = torch.where(candidate_valid, cand_score, torch.full_like(cand_score, -1e30))
        _, keep_pos = torch.topk(cand_score, max_keep, dim=1)
        keep_rank = orch._keep_rank(max_keep)
        keep_valid = keep_rank < per_game_keep.unsqueeze(1)
        new_slots = torch.gather(root_slots, 1, keep_pos)
        candidate_slots = torch.where(keep_valid, new_slots, torch.full_like(new_slots, -1))

    leg_visits, leg_q = orch._gather_root_child_stats(tree, B)
    cand_visits, cand_q = orch._gather_root_candidate_stats(tree, B, candidate_slots)
    sigma_norm = (cfg.c_visit + cand_visits.max()) * cfg.c_scale
    final_score = (gumbel + legal_logits + sigma_norm * leg_q).float()
    final_score = torch.where(valid_slot, final_score, torch.full_like(final_score, -1e30))

    candidate_valid = candidate_slots >= 0
    cand_idx = candidate_slots.long().clamp(min=0)
    final_cand_sigma = (
        torch.gather(gumbel + legal_logits, 1, cand_idx)
        + sigma_norm * cand_q
    ).float()
    final_cand_sigma = torch.where(candidate_valid, final_cand_sigma, torch.full_like(final_cand_sigma, -1e30))
    chosen_pos = torch.argmax(final_cand_sigma, dim=1)
    chosen_slot = torch.gather(candidate_slots, 1, chosen_pos.unsqueeze(1)).squeeze(1)

    chosen_np = chosen_slot.detach().cpu().numpy()
    final_np = final_score.detach().cpu().numpy()
    ranks = []
    for i in range(B):
        n_i = int(num_legal_np[i])
        if n_i <= 0:
            ranks.append(np.empty((0,), dtype=np.int32))
            continue
        order = np.argsort(-final_np[i, :n_i], kind="stable")
        rank_map = np.empty(n_i, dtype=np.int32)
        rank_map[order] = np.arange(1, n_i + 1, dtype=np.int32)
        ranks.append(rank_map)

    return {
        "chosen": chosen_np,
        "final_rank": ranks,
        "nlegal": num_legal_np,
        "legal_moves": legal_moves.detach().cpu().numpy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--simulations", type=int, default=4096)
    ap.add_argument("--considered", type=int, default=16)
    ap.add_argument("--expansion-mask", type=int, default=7)
    ap.add_argument("--max-game-length", type=int, default=300)
    ap.add_argument("--seed-a", type=int, default=12345)
    ap.add_argument("--seed-b", type=int, default=67890)
    args = ap.parse_args()

    net = load_net(args.checkpoint)
    cfg = FNNMCTSConfig(
        num_simulations=args.simulations,
        max_num_considered_actions=args.considered,
        batch_size=args.games,
        max_game_length=args.max_game_length,
        expansion_mask=args.expansion_mask,
        wave_parallel=True,
        dirichlet_epsilon=0.0,
    )
    orch = FNNMCTSOrchestrator(net, cfg)
    ext = hive_gpu.load_extension()

    B = args.games
    states = ext.create_initial_states(B, args.expansion_mask)
    active = np.ones(B, dtype=bool)
    move_numbers = np.zeros(B, dtype=np.int32)

    total_positions = 0
    top1_match = 0
    a_in_b_top4 = 0
    a_in_b_top8 = 0
    b_in_a_top4 = 0
    b_in_a_top8 = 0
    a_rank_in_b_sum = 0
    b_rank_in_a_sum = 0
    step_index = 0

    while bool(active.any()):
        idx = np.flatnonzero(active)
        sub_states = states[idx].clone()

        out_a = run_root_search(orch, sub_states, args.considered, args.seed_a + step_index)
        out_b = run_root_search(orch, sub_states, args.considered, args.seed_b + step_index)

        chosen_a = out_a["chosen"]
        chosen_b = out_b["chosen"]
        ranks_a = out_a["final_rank"]
        ranks_b = out_b["final_rank"]
        nlegal_np = out_a["nlegal"]

        for j in range(idx.size):
            n_i = int(nlegal_np[j])
            if n_i <= 0:
                continue
            a = int(chosen_a[j])
            b = int(chosen_b[j])
            rank_a_in_b = int(ranks_b[j][a]) if 0 <= a < n_i else n_i + 1
            rank_b_in_a = int(ranks_a[j][b]) if 0 <= b < n_i else n_i + 1
            top1_match += int(a == b)
            a_in_b_top4 += int(rank_a_in_b <= 4)
            a_in_b_top8 += int(rank_a_in_b <= 8)
            b_in_a_top4 += int(rank_b_in_a <= 4)
            b_in_a_top8 += int(rank_b_in_a <= 8)
            a_rank_in_b_sum += rank_a_in_b
            b_rank_in_a_sum += rank_b_in_a
            total_positions += 1

        move_bytes = np.zeros((idx.size, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
        legal_np = out_a["legal_moves"]
        for j in range(idx.size):
            n_i = int(nlegal_np[j])
            if n_i <= 0:
                continue
            move_bytes[j] = legal_np[j, int(chosen_a[j])]

        ext.apply_moves_batch(sub_states, torch.from_numpy(move_bytes).cuda(), idx.size)
        states[idx] = sub_states
        move_numbers[idx] += 1
        final_results = ext.check_results_batch(states, B).cpu().numpy()
        active = active & (final_results == 0) & (move_numbers < args.max_game_length)
        step_index += 1

    print(f"positions analyzed: {total_positions}")
    if total_positions == 0:
        return
    print(f"top1 exact agreement: {top1_match}/{total_positions} = {top1_match/total_positions:.3f}")
    print(f"run-A top1 inside run-B top4: {a_in_b_top4}/{total_positions} = {a_in_b_top4/total_positions:.3f}")
    print(f"run-A top1 inside run-B top8: {a_in_b_top8}/{total_positions} = {a_in_b_top8/total_positions:.3f}")
    print(f"run-B top1 inside run-A top4: {b_in_a_top4}/{total_positions} = {b_in_a_top4/total_positions:.3f}")
    print(f"run-B top1 inside run-A top8: {b_in_a_top8}/{total_positions} = {b_in_a_top8/total_positions:.3f}")
    print(f"mean rank of run-A top1 inside run-B: {a_rank_in_b_sum/total_positions:.2f}")
    print(f"mean rank of run-B top1 inside run-A: {b_rank_in_a_sum/total_positions:.2f}")


if __name__ == "__main__":
    main()
