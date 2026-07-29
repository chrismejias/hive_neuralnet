"""Benchmark fresh versus reusable CUDA alpha-beta across batch widths."""

from __future__ import annotations

import argparse
import time

import torch

import arena
import hive_gpu
from hive_fnn.fnn_native_alphabeta import AlphaBetaGPUContext, search_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", default="8,16,32,64,128,256")
    parser.add_argument("--nodes", type=int, default=1_000)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--position-plies", default="0,12,24")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def make_position(ext, batch: int, target_plies: int) -> tuple[torch.Tensor, int]:
    """Build identical deterministic states to isolate batch-width effects."""

    states = ext.create_initial_states(batch)
    completed = 0
    for ply in range(target_plies):
        moves, counts = ext.generate_legal_moves_batch(states, batch)
        if int(counts[0].item()) <= 0:
            break
        slot = (ply * 7 + 3) % int(counts[0].item())
        selected = moves[:, slot].contiguous()
        ext.apply_moves_batch(states, selected, batch)
        completed += 1
        if int(ext.check_results_batch(states[:1], 1)[0].item()) != 0:
            break
    return states, completed


def timed_searches(net, states, args, *, reusable: bool) -> tuple[float, int]:
    context = AlphaBetaGPUContext(net, capacity=states.shape[0]) if reusable else None
    for _ in range(args.warmup):
        search_batch(
            net, states, node_budget=args.nodes, max_depth=args.max_depth,
            context=context,
        )
    torch.cuda.synchronize()
    total_nodes = 0
    started = time.perf_counter()
    for _ in range(args.repeats):
        _moves, _values, stats = search_batch(
            net, states, node_budget=args.nodes, max_depth=args.max_depth,
            context=context,
        )
        total_nodes += int(stats[:, 1].sum().item())
    torch.cuda.synchronize()
    return time.perf_counter() - started, total_nodes


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    batches = [int(value) for value in args.batches.split(",") if value]
    position_plies = [
        int(value) for value in args.position_plies.split(",") if value
    ]
    if not batches or min(batches) <= 0:
        raise ValueError("batch sizes must be positive")
    net = arena.load_checkpoint("fnn", args.checkpoint)
    ext = hive_gpu.load_extension()
    print("mode,batch,position_plies,seconds,nodes,nodes_per_second,moves_per_second")
    for requested_plies in position_plies:
        for batch in batches:
            states, actual_plies = make_position(ext, batch, requested_plies)
            fresh = timed_searches(net, states, args, reusable=False)
            reused = timed_searches(net, states, args, reusable=True)
            for mode, (seconds, nodes) in (("fresh", fresh), ("reuse", reused)):
                moves = batch * args.repeats
                print(
                    f"{mode},{batch},{actual_plies},{seconds:.6f},{nodes},"
                    f"{nodes / seconds:.1f},{moves / seconds:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
