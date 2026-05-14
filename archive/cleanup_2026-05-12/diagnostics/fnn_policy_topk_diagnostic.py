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


def sample_policy_positions(
    orch: FNNMCTSOrchestrator,
    num_positions: int,
    min_ply: int,
    max_ply: int,
    expansion_mask: int,
    seed: int,
) -> torch.Tensor:
    """Sample legal midgame positions by rolling out the FNN policy stochastically."""
    rng = np.random.default_rng(seed)
    target_ply = rng.integers(min_ply, max_ply + 1, size=num_positions, dtype=np.int32)
    states = orch.ext.create_initial_states(num_positions, expansion_mask)
    active = np.ones(num_positions, dtype=bool)
    ply = np.zeros(num_positions, dtype=np.int32)

    while bool(active.any()):
        unfinished = active & (ply < target_ply)
        if not bool(unfinished.any()):
            break
        idx = np.flatnonzero(unfinished)
        sub = states[idx].clone()
        B = int(idx.size)
        legal, nlegal, root_features = orch.ext.generate_legal_moves_and_fnn_features_batch(sub, B)
        nlegal_np = nlegal.cpu().numpy()
        priors, _values, _child_q = orch._eval_states(sub, legal, nlegal, B, root_features)
        priors_np = priors.cpu().numpy()
        legal_np = legal.cpu().numpy()
        move_bytes = np.zeros((B, orch.ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
        for j in range(B):
            n = int(nlegal_np[j])
            if n <= 0:
                active[idx[j]] = False
                continue
            probs = priors_np[j, :n].astype(np.float64, copy=True)
            psum = float(probs.sum())
            if not np.isfinite(psum) or psum <= 0.0:
                probs = np.full(n, 1.0 / n, dtype=np.float64)
            else:
                probs /= psum
            k = int(rng.choice(n, p=probs))
            move_bytes[j] = legal_np[j, k]
        orch.ext.apply_moves_batch(sub, torch.from_numpy(move_bytes).cuda(), B)
        states[idx] = sub
        ply[idx] += 1
        results = orch.ext.check_results_batch(states, num_positions).cpu().numpy()
        active = active & (results == 0)

    return states


def analyze_root(
    orch: FNNMCTSOrchestrator,
    states: torch.Tensor,
    considered_k: int,
    gumbel_seed: int,
) -> dict[str, object]:
    B = int(states.shape[0])
    cfg = orch.config
    dev = states.device
    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    legal_moves, num_legal, root_features = (
        orch.ext.generate_legal_moves_and_fnn_features_batch(states, B)
    )
    nlegal_np = num_legal.cpu().numpy()
    if int(nlegal_np.max()) <= 0:
        return {"count": 0}

    priors_per_legal, _root_values, child_q_per_legal = orch._eval_states(
        states, legal_moves, num_legal, B, root_features,
    )

    valid_slot = orch._slot_idx < num_legal.to(torch.int64).unsqueeze(1)
    safe_prior = priors_per_legal.clamp_min(1e-30)
    legal_logits = torch.where(
        valid_slot,
        safe_prior.log(),
        torch.full_like(priors_per_legal, -1e30),
    )

    gen = torch.Generator(device=dev)
    gen.manual_seed(gumbel_seed)
    u = torch.rand(B, orch._max_legal, device=dev, generator=gen).clamp(1e-4, 1 - 1e-4)
    gumbel = -torch.log(-torch.log(u))
    gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))

    max_k = min(considered_k, int(nlegal_np.max()))
    _, topk_slots = torch.topk(gumbel + legal_logits, max_k, dim=1)

    has_actions = num_legal > 0
    game_active_t = has_actions.to(torch.int8)
    orch._expand_root_if_needed(
        tree,
        states,
        legal_moves,
        num_legal,
        priors_per_legal,
        child_q_per_legal,
        game_active_t,
        B,
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
            tree,
            states,
            game_active_t,
            root_slots,
            B,
            sims_per_candidate,
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
        cand_score = torch.where(
            candidate_valid,
            cand_score,
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

    cand_visits, cand_q = orch._gather_root_candidate_stats(tree, B, candidate_slots)
    sigma_norm = (cfg.c_visit + cand_visits.max()) * cfg.c_scale
    cand_idx = candidate_slots.long().clamp(min=0)
    final_cand_sigma = (
        torch.gather(gumbel + legal_logits, 1, cand_idx)
        + sigma_norm * cand_q
    ).float()
    final_cand_sigma = torch.where(
        candidate_slots >= 0,
        final_cand_sigma,
        torch.full_like(final_cand_sigma, -1e30),
    )
    chosen_pos = torch.argmax(final_cand_sigma, dim=1)
    chosen_slot = torch.gather(candidate_slots, 1, chosen_pos.unsqueeze(1)).squeeze(1)

    prior_np = priors_per_legal.detach().cpu().numpy()
    chosen_np = chosen_slot.detach().cpu().numpy()

    ranks: list[int] = []
    top4 = top8 = top16 = top32 = 0
    count = 0
    nlegal_sum = 0
    for i in range(B):
        n = int(nlegal_np[i])
        chosen = int(chosen_np[i])
        if n <= 0 or chosen < 0 or chosen >= n:
            continue
        prior_order = np.argsort(-prior_np[i, :n], kind="stable")
        rank_map = np.empty(n, dtype=np.int32)
        rank_map[prior_order] = np.arange(1, n + 1, dtype=np.int32)
        rank = int(rank_map[chosen])
        ranks.append(rank)
        top4 += int(rank <= 4)
        top8 += int(rank <= 8)
        top16 += int(rank <= 16)
        top32 += int(rank <= 32)
        count += 1
        nlegal_sum += n

    return {
        "count": count,
        "top4": top4,
        "top8": top8,
        "top16": top16,
        "top32": top32,
        "ranks": ranks,
        "nlegal_sum": nlegal_sum,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--sample-checkpoint",
        default=None,
        help="Optional checkpoint used only to sample positions before analyzing --checkpoint.",
    )
    ap.add_argument("--positions", type=int, default=16)
    ap.add_argument("--simulations", type=int, default=24576)
    ap.add_argument("--considered", type=int, default=64)
    ap.add_argument("--min-ply", type=int, default=20)
    ap.add_argument("--max-ply", type=int, default=120)
    ap.add_argument("--expansion-mask", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260509)
    args = ap.parse_args()

    net = load_net(args.checkpoint)
    sample_net = load_net(args.sample_checkpoint) if args.sample_checkpoint else net
    cfg = FNNMCTSConfig(
        num_simulations=args.simulations,
        max_num_considered_actions=args.considered,
        batch_size=args.positions,
        expansion_mask=args.expansion_mask,
        wave_parallel=True,
        dirichlet_epsilon=0.0,
    )
    orch = FNNMCTSOrchestrator(net, cfg)
    sample_orch = FNNMCTSOrchestrator(sample_net, cfg) if sample_net is not net else orch
    _ext = hive_gpu.load_extension()

    states = sample_policy_positions(
        sample_orch,
        args.positions,
        args.min_ply,
        args.max_ply,
        args.expansion_mask,
        args.seed,
    )
    result = analyze_root(orch, states, args.considered, args.seed + 999)
    count = int(result["count"])
    print(f"checkpoint: {args.checkpoint}")
    if args.sample_checkpoint:
        print(f"sample_checkpoint: {args.sample_checkpoint}")
    print(
        f"sample: positions={args.positions}, analyzed={count}, "
        f"ply_range=[{args.min_ply},{args.max_ply}], expansion_mask={args.expansion_mask}"
    )
    print(f"search: k={args.considered}, simulations={args.simulations}")
    if count == 0:
        return
    for k in (32, 16, 8, 4):
        hits = int(result[f"top{k}"])
        print(f"search-selected move in policy top {k}: {hits}/{count} = {hits / count:.3f}")
    ranks = np.asarray(result["ranks"], dtype=np.int32)
    print(f"mean policy rank of selected move: {float(ranks.mean()):.2f}")
    print(f"median policy rank of selected move: {float(np.median(ranks)):.1f}")
    print(f"max policy rank of selected move: {int(ranks.max())}")
    print(f"mean legal moves: {float(result['nlegal_sum']) / count:.1f}")


if __name__ == "__main__":
    main()
