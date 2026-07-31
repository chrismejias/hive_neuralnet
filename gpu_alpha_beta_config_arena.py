"""Paired GPU arena comparing tuned and default alpha-beta search settings."""

from __future__ import annotations

import argparse

from hive_fnn.fnn_alphabeta_tuning import (
    load_search_config,
    run_paired_alpha_beta_arena,
)
from hive_fnn.fnn_native_alphabeta import AlphaBetaSearchConfig
from tune_fnn_alphabeta import _load_network


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tuned-config")
    parser.add_argument(
        "--plus-profile",
        choices=("baseline", "quiescence", "threat", "proof", "ordering", "full"),
        default="full",
    )
    parser.add_argument(
        "--minus-profile",
        choices=("baseline", "quiescence", "threat", "proof", "ordering", "full"),
        default="baseline",
    )
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--min-batch", type=int, default=0,
        help="Pad tail searches with duplicate states for GPU utilization",
    )
    args = parser.parse_args()

    plus_config = (
        load_search_config(args.tuned_config) if args.tuned_config
        else AlphaBetaSearchConfig.from_profile(args.plus_profile)
    )
    minus_config = AlphaBetaSearchConfig.from_profile(args.minus_profile)
    result = run_paired_alpha_beta_arena(
        _load_network(args.checkpoint),
        plus_config,
        minus_config,
        pairs=args.pairs,
        node_budget=args.nodes,
        max_depth=args.max_depth,
        max_plies=args.max_plies,
        opening_plies=args.opening_plies,
        expansion_mask=0,
        seed=args.seed,
        min_search_batch=args.min_batch,
    )
    print(
        f"plus={result.plus_wins} minus={result.minus_wins} "
        f"draws={result.draws} games={result.games}"
    )
    print(f"plus_score={result.plus_score:.4f}")
    print(f"plus_mean_nodes={result.plus_nodes_per_move:.1f}")
    print(f"minus_mean_nodes={result.minus_nodes_per_move:.1f}")


if __name__ == "__main__":
    main()
