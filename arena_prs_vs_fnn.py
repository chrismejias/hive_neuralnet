from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import hive_gpu
from hive_fnn.fnn_mcts_orchestrator import FNNMCTSConfig, FNNMCTSOrchestrator
from hive_fnn.fnn_network import HiveFNN
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.slot_map import N_SLOTS


def latest_prs_checkpoint() -> str | None:
    paths = sorted(glob.glob("checkpoints_prs_v2/prs_v2_iter_*.pt"))
    return paths[-1] if paths else None


def latest_fnn_checkpoint() -> str | None:
    paths = sorted(glob.glob("checkpoints_fnn/hive_fnn_checkpoint_*.pt"))
    return paths[-1] if paths else None


def choose_prs_moves(
    orch: PRSMCTSOrchestratorV2,
    states: torch.Tensor,
    move_numbers: np.ndarray | None = None,
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

    u = torch.rand(B, orch._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
    score_basis = -torch.log(-torch.log(u)) + root_logit_per_legal

    _, topk_slots = torch.topk(score_basis, max_k, dim=1)
    active_t = torch.tensor([1 if a else 0 for a in active], dtype=torch.int8, device="cuda")

    orch._expand_root_if_needed(
        tree, states, legal_t, nlegal_t, priors_per_legal, active_t, B,
    )
    if cfg.dirichlet_epsilon > 0:
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


def choose_fnn_moves(
    orch: FNNMCTSOrchestrator,
    states: torch.Tensor,
    move_numbers: np.ndarray | None = None,
    stochastic: bool = False,
) -> np.ndarray:
    B = int(states.shape[0])
    cfg = orch.config
    tree = orch._alloc_tree(B)
    orch._reset_tree(tree)

    legal_moves, num_legal, root_features = orch.ext.generate_legal_moves_and_fnn_features_batch(states, B)
    n_per_game = num_legal.to(torch.int64)
    nlegal_np = num_legal.cpu().numpy()

    has_actions = n_per_game > 0
    immediate_wins = orch._find_immediate_wins(states, legal_moves, num_legal, B)
    has_immediate_win = immediate_wins >= 0

    max_n = int(n_per_game.max().item()) if B > 0 else 0
    if max_n == 0:
        return np.zeros(B, dtype=np.int64)
    max_k = min(cfg.max_num_considered_actions, max_n)

    priors_per_legal, _root_vals = orch._eval_states(
        states, legal_moves, num_legal, B, root_features,
    )
    if bool(has_immediate_win.any().item()):
        priors_per_legal.zero_()
        priors_per_legal[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0

    valid_slot = (
        torch.arange(orch._max_legal, device="cuda").unsqueeze(0)
        < n_per_game.unsqueeze(1)
    )
    safe_prior = priors_per_legal.clamp(min=1e-20)
    legal_logits = torch.where(
        valid_slot, safe_prior.log(),
        torch.full_like(priors_per_legal, -1e30),
    )

    if bool(has_immediate_win.any().item()):
        probs_pad = torch.zeros(B, orch._max_legal, dtype=torch.float32, device="cuda")
        probs_pad[has_immediate_win, immediate_wins[has_immediate_win]] = 1.0
    else:
        u = torch.rand(B, orch._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
        gumbel = -torch.log(-torch.log(u))
        gumbel = torch.where(valid_slot, gumbel, torch.full_like(gumbel, -1e30))
        perturbed = gumbel + legal_logits
        _, topk_slots = torch.topk(perturbed, max_k, dim=1)

        active_t = has_actions.to(torch.int8)
        orch._expand_root_if_needed(
            tree, states, legal_moves, num_legal,
            priors_per_legal, active_t, B,
        )
        orch._apply_root_dirichlet(tree, B, has_actions)

        alive_mask = torch.zeros(B, orch._max_legal, dtype=torch.int8, device="cuda")
        alive_mask.scatter_(1, topk_slots, 1)

        n_rounds = max(1, int(np.ceil(np.log2(max(max_k, 2)))))
        sims_per_round = max(1, cfg.num_simulations // n_rounds)

        for round_i in range(n_rounds):
            round_wave_size = (
                orch.config.wave_size if cfg.wave_parallel else 1
            )
            orch._run_simulations(
                tree, states, active_t, alive_mask, B, sims_per_round,
                wave_size=round_wave_size,
            )
            alive_counts = alive_mask.sum(dim=1)
            cur_alive_max = int(alive_counts.max().item())
            if cur_alive_max <= 1:
                continue
            num_keep = max(1, cur_alive_max // 2)
            slot_visits, slot_q = orch._gather_root_child_stats(tree, B)
            max_n = int(slot_visits.max().item())
            sigma_norm = (cfg.c_visit + max_n) * cfg.c_scale
            sigma_score = (gumbel + legal_logits + sigma_norm * slot_q).float()
            sigma_score = torch.where(
                alive_mask.bool(), sigma_score,
                torch.full_like(sigma_score, -1e30),
            )
            _, keep_slots = torch.topk(sigma_score, num_keep, dim=1)
            new_alive = torch.zeros_like(alive_mask)
            new_alive.scatter_(1, keep_slots, 1)
            alive_mask = new_alive * valid_slot.to(torch.int8)

        slot_visits, _ = orch._gather_root_child_stats(tree, B)
        visits_f = slot_visits.float()
        vsum = visits_f.sum(dim=1, keepdim=True)
        probs_pad = torch.where(
            vsum > 0,
            visits_f / vsum.clamp(min=1.0),
            torch.zeros_like(visits_f),
        )
        uniform = valid_slot.float()
        ucount = uniform.sum(dim=1, keepdim=True).clamp(min=1.0)
        uniform = uniform / ucount
        probs_pad = torch.where(vsum > 0, probs_pad, uniform)

    chosen = torch.zeros((B,), dtype=torch.int64, device="cuda")
    if bool(has_immediate_win.any().item()):
        chosen[has_immediate_win] = immediate_wins[has_immediate_win]
    else:
        active_idx = torch.nonzero(has_actions, as_tuple=False).squeeze(1)
        if active_idx.numel() > 0:
            if move_numbers is not None:
                move_numbers_np = np.asarray(move_numbers)
                active_moves = move_numbers_np[active_idx.cpu().numpy()]
                greedy_mask = active_moves >= cfg.temperature_drop_move
                greedy_rows = active_idx[greedy_mask]
                sample_rows = active_idx[~greedy_mask]
            else:
                greedy_rows = active_idx
                sample_rows = active_idx[:0]

            if greedy_rows.numel() > 0:
                chosen[greedy_rows] = probs_pad[greedy_rows].argmax(dim=1)
            if sample_rows.numel() > 0:
                sample_policy = probs_pad[sample_rows]
                sample_policy = sample_policy / sample_policy.sum(dim=1, keepdim=True).clamp_min(1e-8)
                chosen[sample_rows] = torch.multinomial(sample_policy, 1).squeeze(1)

    chosen_np = chosen.cpu().numpy()
    out = np.zeros(B, dtype=np.int64)
    for i in range(B):
        n_i = int(nlegal_np[i])
        if n_i <= 0:
            out[i] = 0
        else:
            k = int(chosen_np[i])
            out[i] = 0 if k >= n_i else k
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Arena: latest PRS v2 checkpoint vs latest FNN checkpoint.")
    ap.add_argument("--prs-checkpoint", type=str, default=None)
    ap.add_argument("--fnn-checkpoint", type=str, default=None)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--prs-sims", type=int, default=256)
    ap.add_argument("--prs-k", type=int, default=16)
    ap.add_argument("--fnn-sims", type=int, default=1024)
    ap.add_argument("--fnn-k", type=int, default=16)
    ap.add_argument("--expansion-mask", type=int, default=7)
    ap.add_argument("--max-game-length", type=int, default=300)
    ap.add_argument("--temperature-drop-move", type=int, default=20)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prs_ckpt_path = args.prs_checkpoint or latest_prs_checkpoint()
    fnn_ckpt_path = args.fnn_checkpoint or latest_fnn_checkpoint()
    if not prs_ckpt_path:
        raise SystemExit("No PRS checkpoint found")
    if not fnn_ckpt_path:
        raise SystemExit("No FNN checkpoint found")

    prs_ckpt = torch.load(prs_ckpt_path, map_location="cpu", weights_only=False)
    fnn_ckpt = torch.load(fnn_ckpt_path, map_location="cpu", weights_only=False)

    prs_net = HivePRSTransformerV2(prs_ckpt["net_config"]).cuda().eval()
    prs_net.load_state_dict(prs_ckpt["model_state"])
    fnn_net = HiveFNN(fnn_ckpt["net_config"]).cuda().eval()
    fnn_net.load_state_dict(fnn_ckpt["model_state_dict"])

    prs_cfg = PRSMCTSConfigV2(
        num_simulations=args.prs_sims,
        max_num_considered_actions=args.prs_k,
        batch_size=args.games,
        max_game_length=args.max_game_length,
        temperature_drop_move=args.temperature_drop_move,
        expansion_mask=args.expansion_mask,
    )
    fnn_cfg = FNNMCTSConfig(
        num_simulations=args.fnn_sims,
        max_num_considered_actions=args.fnn_k,
        batch_size=args.games,
        max_game_length=args.max_game_length,
        temperature_drop_move=args.temperature_drop_move,
        expansion_mask=args.expansion_mask,
    )

    prs_orch = PRSMCTSOrchestratorV2(prs_net, prs_cfg)
    fnn_orch = FNNMCTSOrchestrator(fnn_net, fnn_cfg)
    ext = hive_gpu.load_extension()

    B = args.games
    states = ext.create_initial_states(B, args.expansion_mask)
    active = np.ones(B, dtype=bool)
    move_numbers = np.zeros(B, dtype=np.int32)
    prs_is_white = np.zeros(B, dtype=bool)
    prs_is_white[: B // 2] = True
    final_results = np.zeros(B, dtype=np.int32)

    while bool(active.any()):
        turns = prs_orch._get_turns(states, B)
        white_to_move = (turns % 2 == 0)
        prs_turn = active & (white_to_move == prs_is_white)
        fnn_turn = active & ~prs_turn

        move_bytes = np.zeros((B, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)

        for mask, orch, chooser in (
            (prs_turn, prs_orch, choose_prs_moves),
            (fnn_turn, fnn_orch, choose_fnn_moves),
        ):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            sub_states = states[idx].clone()
            chosen = chooser(
                orch,
                sub_states,
                move_numbers=move_numbers[idx],
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

        final_results = ext.check_results_batch(states, B).cpu().numpy()
        active = active & (final_results == 0) & (move_numbers < args.max_game_length)

    prs_wins = 0
    fnn_wins = 0
    draws = 0
    capped = 0
    prs_score = 0.0
    final_results = ext.check_results_batch(states, B).cpu().numpy()
    for i in range(B):
        r = int(final_results[i])
        if r == 0:
            capped += 1
            draws += 1
            prs_score += 0.5
        elif r == 3:
            draws += 1
            prs_score += 0.5
        elif (r == 1 and prs_is_white[i]) or (r == 2 and not prs_is_white[i]):
            prs_wins += 1
            prs_score += 1.0
        else:
            fnn_wins += 1

    print(f"PRS checkpoint: {prs_ckpt_path}")
    print(f"FNN checkpoint: {fnn_ckpt_path}")
    print(
        f"Arena config: games={B}, prs_sims={args.prs_sims}, prs_k={args.prs_k}, "
        f"fnn_sims={args.fnn_sims}, fnn_k={args.fnn_k}, "
        f"expansions={args.expansion_mask}, max_game_length={args.max_game_length}, "
        f"stochastic={args.stochastic}"
    )
    print(f"PRS wins:       {prs_wins}")
    print(f"FNN wins:       {fnn_wins}")
    print(f"Draws:          {draws}  (capped={capped})")
    print(f"PRS score:      {prs_score:.1f}/{B} = {prs_score / B:.3f}")


if __name__ == "__main__":
    main()
