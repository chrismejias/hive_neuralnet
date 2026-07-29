"""Paired GPU alpha-beta arena between two FNN checkpoints."""

from __future__ import annotations

import argparse
import time

import torch

import arena
import hive_gpu
from hive_fnn.fnn_native_alphabeta import AlphaBetaGPUContext, search_batch
from tune_fnn_alphabeta import _load_network


def _random_paired_openings(
    states: torch.Tensor,
    *,
    pairs: int,
    plies: int,
    seed: int,
) -> None:
    """Create paired openings without synchronizing move counts to the CPU."""

    ext = hive_gpu.load_extension()
    games = pairs * 2
    generator = torch.Generator(device=states.device)
    generator.manual_seed(seed)
    pair_rows = torch.arange(pairs, device=states.device, dtype=torch.int64) * 2
    for _ in range(plies):
        legal_moves, num_legal = ext.generate_legal_moves_batch(states, games)
        pair_counts = num_legal.index_select(0, pair_rows).to(torch.int64)
        slots = torch.floor(
            torch.rand(pairs, device=states.device, generator=generator)
            * pair_counts.clamp_min(1)
        ).to(torch.int64)
        pair_moves = legal_moves[pair_rows, slots]
        selected = torch.repeat_interleave(pair_moves, 2, dim=0).contiguous()
        ext.apply_moves_batch(states, selected, games)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")

    loader = _load_network if args.weights == "ema" else (
        lambda path: arena.load_checkpoint("fnn", path)
    )
    challenger = loader(args.challenger)
    baseline = loader(args.baseline)
    ext = hive_gpu.load_extension()
    games = args.pairs * 2
    states = ext.create_initial_states(games)
    _random_paired_openings(
        states,
        pairs=args.pairs,
        plies=args.opening_plies,
        seed=args.seed,
    )

    move_counts = {"challenger": 0, "baseline": 0}
    node_counts = {
        "challenger": torch.zeros((), dtype=torch.int64, device=states.device),
        "baseline": torch.zeros((), dtype=torch.int64, device=states.device),
    }
    plies = torch.full(
        (games,), args.opening_plies, dtype=torch.int32, device=states.device,
    )
    active = ext.check_results_batch(states, games) == 0
    challenger_is_white = torch.zeros(
        games, dtype=torch.bool, device=states.device,
    )
    challenger_is_white[::2] = True
    contexts = {
        "challenger": AlphaBetaGPUContext(challenger, capacity=games),
        "baseline": AlphaBetaGPUContext(baseline, capacity=games),
    }
    started = time.perf_counter()
    rounds = 0

    while bool(active.any().item()):
        turns = states[:, 3412].to(torch.int32)
        turns |= states[:, 3413].to(torch.int32) << 8
        challenger_turn = active & (((turns & 1) == 0) == challenger_is_white)
        for name, net, mask in (
            ("challenger", challenger, challenger_turn),
            ("baseline", baseline, active & ~challenger_turn),
        ):
            rows = torch.nonzero(mask, as_tuple=False).flatten()
            num_rows = rows.numel()
            if not num_rows:
                continue
            sub_states = states.index_select(0, rows).contiguous()
            moves, _values, stats = search_batch(
                net,
                sub_states,
                node_budget=args.nodes,
                max_depth=args.max_depth,
                context=contexts[name],
            )
            ext.apply_moves_batch(sub_states, moves, num_rows)
            states.index_copy_(0, rows, sub_states)
            plies.index_add_(
                0, rows, torch.ones_like(rows, dtype=plies.dtype),
            )
            move_counts[name] += num_rows
            node_counts[name] += stats[:, 1].sum(dtype=torch.int64)

        results_gpu = ext.check_results_batch(states, games)
        active &= (results_gpu == 0) & (plies < args.max_plies)
        rounds += 1
        if args.progress_every > 0 and rounds % args.progress_every == 0:
            print(
                f"round={rounds} active={int(active.sum().item())}/{games} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    results = ext.check_results_batch(states, games).cpu().numpy()
    challenger_wins = baseline_wins = draws = 0
    for game, result in enumerate(results):
        if (
            (result == 1 and challenger_is_white[game])
            or (result == 2 and not challenger_is_white[game])
        ):
            challenger_wins += 1
        elif result in (1, 2):
            baseline_wins += 1
        else:
            draws += 1

    score = (challenger_wins + 0.5 * draws) / games
    print(
        f"challenger={challenger_wins} baseline={baseline_wins} "
        f"draws={draws} games={games}"
    )
    print(f"challenger_score={score:.4f}")
    for name in ("challenger", "baseline"):
        mean_nodes = int(node_counts[name].item()) / max(1, move_counts[name])
        print(f"{name}_mean_nodes={mean_nodes:.1f}")


if __name__ == "__main__":
    main()
