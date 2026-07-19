"""Tune native FNN alpha-beta search parameters with paired SPSA arenas."""

from __future__ import annotations

import argparse

import torch

from hive_fnn.fnn_alphabeta_tuning import (
    AlphaBetaSPSATuner,
    SPSAConfig,
    decode_search_config,
    run_paired_alpha_beta_arena,
)
from hive_fnn.fnn_network import FNNConfig, HiveFNN


def _load_network(path: str) -> HiveFNN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("net_config")
    if not isinstance(config, FNNConfig):
        config = FNNConfig.large()
    state = (
        checkpoint.get("ema_state_dict")
        or checkpoint.get("model_state_dict")
        or checkpoint.get("champion_state_dict")
    )
    if state is None:
        raise KeyError("checkpoint does not contain FNN weights")
    net = HiveFNN(config)
    net.load_state_dict(state)
    return net.cuda().eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=2_000)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--expansion-mask", type=int, default=0)
    parser.add_argument("--output-dir", default="alphabeta_spsa")
    parser.add_argument("--resume")
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--perturbation", type=float, default=0.20)
    parser.add_argument("--node-cost-penalty", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    net = _load_network(args.checkpoint)
    tuner = AlphaBetaSPSATuner(
        args.output_dir,
        config=SPSAConfig(
            learning_rate=args.learning_rate,
            perturbation=args.perturbation,
            node_cost_penalty=args.node_cost_penalty,
        ),
        seed=args.seed,
    )
    if args.resume:
        tuner.load(args.resume)
        print(f"resumed={args.resume} iteration={tuner.iteration}", flush=True)

    while tuner.iteration < args.iterations:
        plus, minus, delta, a_k, c_k = tuner.ask()
        arena = run_paired_alpha_beta_arena(
            net,
            decode_search_config(plus),
            decode_search_config(minus),
            pairs=args.pairs,
            node_budget=args.nodes,
            max_depth=args.max_depth,
            max_plies=args.max_plies,
            opening_plies=args.opening_plies,
            expansion_mask=args.expansion_mask,
            seed=args.seed + tuner.iteration,
        )
        record = tuner.tell(plus, minus, delta, a_k, c_k, arena)
        print(
            f"iteration={record['iteration']} "
            f"score={record['plus_score']:.3f} "
            f"W={record['plus_wins']} L={record['minus_wins']} "
            f"D={record['draws']} "
            f"nodes=({record['plus_nodes_per_move']:.1f},"
            f"{record['minus_nodes_per_move']:.1f}) "
            f"objective={record['objective_difference']:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
