from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
import hive_gpu


def latest_checkpoint() -> str | None:
    paths = sorted(glob.glob("checkpoints_prs_v2/prs_v2_iter_*.pt"))
    return paths[-1] if paths else None


def choose_moves(
    orch: PRSMCTSOrchestratorV2,
    states: torch.Tensor,   # (B, SIZEOF_HIVE_STATE) uint8 on CUDA
    move_numbers: np.ndarray | None = None,
    use_noise: bool = False,
    stochastic: bool = False,
) -> np.ndarray:
    B = int(states.shape[0])
    cfg = orch.config
    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    prs_batch = orch.encoder.encode_batch(states, B)
    legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(states, B)
    nlegal_np = nlegal_t.cpu().numpy()

    turns_np = orch._get_turns(states, B)
    active = [int(n) > 0 for n in nlegal_np]
    win_overrides = orch._check_immediate_wins_v2(
        states, legal_t, nlegal_np, active, turns_np, B,
    )

    kernel_out = orch._classify_kernel(states, legal_t, nlegal_t, B)
    slot_of_legal_t = kernel_out[8]
    policy_logits_813, _ = orch._net_forward(prs_batch, kernel_out, B)
    policy_logits_813 = policy_logits_813.float()
    priors_per_legal, root_logit_per_legal = orch._build_legal_priors_v2(
        policy_logits_813, slot_of_legal_t, B,
    )

    max_legal = int(nlegal_np.max()) if nlegal_np.size > 0 else 0
    if max_legal <= 0:
        return np.zeros(B, dtype=np.int64)
    max_k = min(cfg.max_num_considered_actions, max_legal)

    if use_noise:
        u = torch.rand(B, orch._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
        score_basis = -torch.log(-torch.log(u)) + root_logit_per_legal
    else:
        score_basis = root_logit_per_legal

    _, topk_slots = torch.topk(score_basis, max_k, dim=1)
    active_t = torch.tensor([1 if a else 0 for a in active], dtype=torch.int8, device="cuda")

    orch._expand_root_if_needed(
        tree, states, legal_t, nlegal_t, priors_per_legal, active_t, B,
    )
    if use_noise and cfg.dirichlet_epsilon > 0:
        orch._apply_root_dirichlet(tree, B, active)

    alive_mask = torch.zeros(B, orch._max_legal, dtype=torch.int8, device="cuda")
    alive_mask.scatter_(1, topk_slots, 1)

    n_rounds = max(1, int(np.ceil(np.log2(max(max_k, 2)))))
    sims_per_round = max(1, cfg.num_simulations // n_rounds)

    for _ in range(n_rounds):
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

    leg_visits, _ = orch._gather_root_child_stats(tree, B)
    slot_t = slot_of_legal_t.to(dtype=torch.long)
    valid_legal = slot_t >= 0
    slot_safe = slot_t.clamp(min=0)
    from hive_prs.slot_map import N_SLOTS
    slot_sum = torch.zeros(B, N_SLOTS, dtype=torch.float32, device="cuda")
    slot_sum.scatter_add_(1, slot_safe, leg_visits.float() * valid_legal.float())
    agg_per_legal = slot_sum.gather(1, slot_safe)
    agg_per_legal = torch.where(valid_legal, agg_per_legal, torch.zeros_like(agg_per_legal))
    agg_per_legal = agg_per_legal * alive_mask.float()
    agg_np = agg_per_legal.cpu().numpy()
    chosen = np.zeros(B, dtype=np.int64)
    for i in range(B):
        n_i = int(nlegal_np[i])
        if n_i <= 0:
            chosen[i] = 0
            continue
        if i in win_overrides:
            chosen[i] = int(win_overrides[i])
            continue
        scores = agg_np[i, :n_i].astype(np.float64, copy=True)
        if stochastic and move_numbers is not None and int(move_numbers[i]) < cfg.temperature_drop_move:
            ssum = float(scores.sum())
            if ssum > 0 and np.isfinite(ssum):
                probs = scores / ssum
            else:
                probs = np.full(n_i, 1.0 / n_i, dtype=np.float64)
            chosen[i] = int(np.random.choice(n_i, p=probs))
        else:
            chosen[i] = int(np.argmax(scores))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description="Arena: latest PRS checkpoint vs random untrained PRS net.")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--simulations", type=int, default=8)
    ap.add_argument("--max-considered", type=int, default=8)
    ap.add_argument("--expansion-mask", type=int, default=7)
    ap.add_argument("--max-game-length", type=int, default=300)
    ap.add_argument("--temperature-drop-move", type=int, default=20)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = args.checkpoint or latest_checkpoint()
    if not ckpt_path:
        raise SystemExit("No checkpoint found")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_config = ckpt["net_config"]

    latest_net = HivePRSTransformerV2(net_config).cuda().eval()
    latest_net.load_state_dict(ckpt["model_state"])

    random_net = HivePRSTransformerV2(net_config).cuda().eval()

    cfg = PRSMCTSConfigV2(
        num_simulations=args.simulations,
        max_num_considered_actions=args.max_considered,
        batch_size=args.games,
        max_game_length=args.max_game_length,
        temperature_drop_move=args.temperature_drop_move,
        expansion_mask=args.expansion_mask,
    )
    latest_orch = PRSMCTSOrchestratorV2(latest_net, cfg)
    random_orch = PRSMCTSOrchestratorV2(random_net, cfg)
    ext = hive_gpu.load_extension()

    B = args.games
    states = ext.create_initial_states(B, args.expansion_mask)
    active = np.ones(B, dtype=bool)
    move_numbers = np.zeros(B, dtype=np.int32)
    # First half latest plays White, second half latest plays Black.
    latest_is_white = np.zeros(B, dtype=bool)
    latest_is_white[: B // 2] = True
    results = np.zeros(B, dtype=np.int32)

    while bool(active.any()):
        turns = latest_orch._get_turns(states, B)
        white_to_move = (turns % 2 == 0)
        latest_turn = active & (white_to_move == latest_is_white)
        random_turn = active & ~latest_turn

        move_bytes = np.zeros((B, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)

        for mask, orch in ((latest_turn, latest_orch), (random_turn, random_orch)):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            sub_states = states[idx].clone()
            chosen = choose_moves(
                orch,
                sub_states,
                move_numbers=move_numbers[idx],
                use_noise=args.stochastic,
                stochastic=args.stochastic,
            )
            legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(sub_states, idx.size)
            legal_np = legal_t.cpu().numpy()
            nlegal_np = nlegal_t.cpu().numpy()
            for j, game_idx in enumerate(idx):
                n_i = int(nlegal_np[j])
                if n_i <= 0:
                    continue
                k = int(chosen[j])
                if k >= n_i:
                    k = 0
                move_bytes[game_idx] = legal_np[j, k]

        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            break
        sub_states = states[active_idx].clone()
        sub_moves = torch.from_numpy(move_bytes[active_idx]).cuda()
        ext.apply_moves_batch(sub_states, sub_moves, int(active_idx.size))
        states[active_idx] = sub_states
        move_numbers[active_idx] += 1

        results = ext.check_results_batch(states, B).cpu().numpy()
        active = active & (results == 0) & (move_numbers < args.max_game_length)

    latest_score = 0.0
    latest_wins = 0
    random_wins = 0
    draws = 0
    capped = 0
    final_results = ext.check_results_batch(states, B).cpu().numpy()
    for i in range(B):
        r = int(final_results[i])
        if r == 0:
            capped += 1
            draws += 1
            latest_score += 0.5
        elif r == 3:
            draws += 1
            latest_score += 0.5
        elif (r == 1 and latest_is_white[i]) or (r == 2 and not latest_is_white[i]):
            latest_wins += 1
            latest_score += 1.0
        else:
            random_wins += 1

    print(f"Checkpoint: {ckpt_path}")
    print(
        f"Arena config: games={B}, sims={args.simulations}, "
        f"k={args.max_considered}, expansions={args.expansion_mask}, "
        f"max_game_length={args.max_game_length}, stochastic={args.stochastic}"
    )
    print(f"Latest net wins:  {latest_wins}")
    print(f"Random net wins:  {random_wins}")
    print(f"Draws:            {draws}  (capped={capped})")
    print(f"Latest score:     {latest_score:.1f}/{B} = {latest_score / B:.3f}")


if __name__ == "__main__":
    main()
