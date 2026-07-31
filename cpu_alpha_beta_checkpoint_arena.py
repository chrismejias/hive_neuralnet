"""Paired native-CPU alpha-beta arena between two FNN checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import random
import time

from hive_engine.game_state import GameResult, GameState, Move
from hive_fnn.fnn_alphabeta_player import AlphaBetaConfig, FNNAlphaBetaPlayer


def _matching_move(state: GameState, packed: bytes) -> Move:
    for move in state.legal_moves():
        if FNNAlphaBetaPlayer._cpu_move_bytes(move) == packed:
            return move
    raise RuntimeError("paired opening move is not legal")


def _play_one(task: tuple) -> dict:
    (
        game_index, challenger_path, baseline_path, nodes, max_depth,
        max_plies, opening_plies, seed,
    ) = task
    cfg = AlphaBetaConfig(
        node_budget=nodes, max_depth=max_depth, torch_threads=1,
        native_tree=True,
    )
    challenger = FNNAlphaBetaPlayer.from_checkpoint(challenger_path, config=cfg)
    baseline = FNNAlphaBetaPlayer.from_checkpoint(baseline_path, config=cfg)
    state = GameState()
    rng = random.Random(seed + game_index // 2)
    opening: list[bytes] = []
    for _ in range(opening_plies):
        if state.result != GameResult.IN_PROGRESS:
            break
        move = rng.choice(state.legal_moves())
        opening.append(FNNAlphaBetaPlayer._cpu_move_bytes(move))
        state.apply_move(move)

    challenger_white = game_index % 2 == 0
    elapsed = {"challenger": 0.0, "baseline": 0.0}
    move_counts = {"challenger": 0, "baseline": 0}
    node_counts = {"challenger": 0, "baseline": 0}
    while state.result == GameResult.IN_PROGRESS and state.turn < max_plies:
        challenger_turn = (state.turn % 2 == 0) == challenger_white
        name = "challenger" if challenger_turn else "baseline"
        player = challenger if challenger_turn else baseline
        start = time.perf_counter()
        move = player.choose_move(state)
        elapsed[name] += time.perf_counter() - start
        move_counts[name] += 1
        node_counts[name] += player.last_stats.nodes
        state.apply_move(move)

    if state.result == GameResult.WHITE_WINS:
        winner = "challenger" if challenger_white else "baseline"
    elif state.result == GameResult.BLACK_WINS:
        winner = "baseline" if challenger_white else "challenger"
    else:
        winner = "draw"
    return {
        "index": game_index,
        "winner": winner,
        "result": state.result.name,
        "plies": state.turn,
        "challenger_white": challenger_white,
        "elapsed": elapsed,
        "move_counts": move_counts,
        "node_counts": node_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=50_000)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    games = args.pairs * 2
    tasks = [(
        index, args.challenger, args.baseline, args.nodes, args.max_depth,
        args.max_plies, args.opening_plies, args.seed,
    ) for index in range(games)]
    results = []
    arena_started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_play_one, tasks):
            results.append(result)
            print(
                f"game {result['index'] + 1}/{games}: "
                f"challenger={'white' if result['challenger_white'] else 'black'} "
                f"result={result['result']} winner={result['winner']} "
                f"plies={result['plies']}",
                flush=True,
            )
    arena_elapsed = time.perf_counter() - arena_started

    scores = {"challenger": 0, "baseline": 0, "draw": 0}
    elapsed = {"challenger": 0.0, "baseline": 0.0}
    move_counts = {"challenger": 0, "baseline": 0}
    node_counts = {"challenger": 0, "baseline": 0}
    for result in results:
        scores[result["winner"]] += 1
        for name in ("challenger", "baseline"):
            elapsed[name] += result["elapsed"][name]
            move_counts[name] += result["move_counts"][name]
            node_counts[name] += result["node_counts"][name]
    score = (scores["challenger"] + 0.5 * scores["draw"]) / games
    print("\nArena summary")
    print(
        f"challenger={scores['challenger']} baseline={scores['baseline']} "
        f"draws={scores['draw']} games={games}"
    )
    print(f"challenger_score={score:.4f}")
    total_nodes = sum(node_counts.values())
    total_search_seconds = sum(elapsed.values())
    print(f"wall_time={arena_elapsed:.3f}s")
    print(f"total_nodes={total_nodes}")
    print(f"aggregate_nodes_per_second={total_nodes / max(arena_elapsed, 1e-9):.1f}")
    print(
        "search_only_worker_normalized_nodes_per_second="
        f"{total_nodes * args.workers / max(total_search_seconds, 1e-9):.1f}"
    )
    for name in ("challenger", "baseline"):
        print(
            f"{name}_mean_time={elapsed[name] / max(1, move_counts[name]):.4f}s "
            f"{name}_mean_nodes={node_counts[name] / max(1, move_counts[name]):.1f}"
        )


if __name__ == "__main__":
    main()
