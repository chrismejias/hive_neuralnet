"""GPU-parallel arena: cooperative alpha-beta versus batched Gumbel MCTS."""

from __future__ import annotations

import argparse

import numpy as np
import torch

import arena
import hive_gpu
from hive_fnn.fnn_native_alphabeta import search_batch
from hive_fnn.fnn_alphabeta_tuning import load_search_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints_fnn/hive_fnn_checkpoint_1080.pt")
    parser.add_argument("--mode", choices=("versus-gumbel", "self-play"), default="versus-gumbel")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--alpha-beta-nodes", type=int, default=10_000)
    parser.add_argument("--alpha-beta-depth", type=int, default=32)
    parser.add_argument("--alpha-beta-search-config")
    parser.add_argument("--gumbel-sims", type=int, default=8_192)
    parser.add_argument("--gumbel-k", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--expansion-mask", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    alpha_beta_search_config = (
        load_search_config(args.alpha_beta_search_config)
        if args.alpha_beta_search_config
        else None
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    net = arena.load_checkpoint("fnn", args.checkpoint)
    gumbel, _ = arena.build_orchestrator(
        "fnn", net, args.games, args.gumbel_sims, args.gumbel_k, True, 4,
        args.expansion_mask, args.max_plies, 1.0, 0, False, eval_mode=True,
    )
    ext = hive_gpu.load_extension()
    states = ext.create_initial_states(args.games, args.expansion_mask)
    gumbel_tree = gumbel._alloc_tree(args.games)
    gumbel._reset_tree(gumbel_tree)
    active = np.ones(args.games, dtype=bool)
    plies = np.zeros(args.games, dtype=np.int32)
    alpha_is_white = np.zeros(args.games, dtype=bool)
    alpha_is_white[::2] = True

    while bool(active.any()):
        turns = states[:, 3412].cpu().numpy().astype(np.int32)
        turns |= states[:, 3413].cpu().numpy().astype(np.int32) << 8
        if args.mode == "self-play":
            alpha_turn = active.copy()
            gumbel_turn = np.zeros(args.games, dtype=bool)
        else:
            alpha_turn = active & (((turns & 1) == 0) == alpha_is_white)
            gumbel_turn = active & ~alpha_turn

        alpha_rows = np.flatnonzero(alpha_turn)
        if alpha_rows.size:
            row_t = torch.from_numpy(alpha_rows.astype(np.int64, copy=False)).cuda()
            sub_states = states.index_select(0, row_t).contiguous()
            moves, _values, _stats = search_batch(
                net, sub_states, node_budget=args.alpha_beta_nodes,
                max_depth=args.alpha_beta_depth,
                search_config=alpha_beta_search_config,
            )
            ext.apply_moves_batch(sub_states, moves, int(alpha_rows.size))
            states.index_copy_(0, row_t, sub_states)
            plies[alpha_rows] += 1

        gumbel_rows = np.flatnonzero(gumbel_turn)
        arena._run_side_rows(
            model_type="fnn", orch=gumbel, tree=gumbel_tree, states=states,
            rows=gumbel_rows, move_numbers=plies, stochastic=args.noise > 0.0,
            gumbel_noise_scale=args.noise, policy_only=False,
        )

        moved = np.concatenate((alpha_rows, gumbel_rows))
        if not moved.size:
            break
        row_t = torch.from_numpy(moved.astype(np.int64, copy=False)).cuda()
        result = ext.check_results_batch(states.index_select(0, row_t), int(moved.size)).cpu().numpy()
        active[moved] = (result == 0) & (plies[moved] < args.max_plies)

    results = ext.check_results_batch(states, args.games).cpu().numpy()
    if args.mode == "self-play":
        decisive = int(np.count_nonzero((results == 1) | (results == 2)))
        print(f"GPU alpha-beta self-play: games={args.games}, decisive={decisive}, draws={args.games - decisive}")
    else:
        alpha_score = 0.0
        for i, result in enumerate(results):
            if result in (0, 3):
                alpha_score += 0.5
            elif (result == 1 and alpha_is_white[i]) or (result == 2 and not alpha_is_white[i]):
                alpha_score += 1.0
        print(f"GPU alpha-beta vs Gumbel: {alpha_score:.1f}/{args.games} = {alpha_score / args.games:.3f}")
    print(f"alpha-beta nodes={args.alpha_beta_nodes}, depth={args.alpha_beta_depth}")
    print(f"Gumbel simulations={args.gumbel_sims}, k={args.gumbel_k}, noise={args.noise}")


if __name__ == "__main__":
    main()
