"""Color-balanced Base Hive arena: local FNN alpha-beta versus Nokamute."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from hive_engine.game_state import GameResult, GameState
from hive_engine.uhp import UHPClient
from hive_fnn.fnn_alphabeta_player import AlphaBetaConfig, FNNAlphaBetaPlayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints_fnn_alphabeta_50k_256/hive_fnn_alphabeta_0139.pt")
    parser.add_argument("--nokamute", default="external/nokamute/target/release/nokamute")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--our-nodes", type=int, default=10_000)
    parser.add_argument("--our-max-depth", type=int, default=16)
    parser.add_argument(
        "--our-native-tree",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--nokamute-seconds", type=float, default=1.0)
    budget.add_argument("--nokamute-depth", type=int)
    parser.add_argument("--nokamute-threads", type=int, default=1)
    parser.add_argument("--nokamute-table-mb", type=int, default=64)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--trace-moves", action="store_true")
    parser.add_argument("--ours-black-first", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0 or args.opening_plies < 0:
        raise ValueError("games must be positive and opening plies non-negative")
    for path in (args.checkpoint, args.nokamute):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    search = (
        f"depth {args.nokamute_depth}" if args.nokamute_depth is not None
        else f"seconds {args.nokamute_seconds}"
    )
    ours = FNNAlphaBetaPlayer.from_checkpoint(
        args.checkpoint,
        config=AlphaBetaConfig(
            node_budget=args.our_nodes,
            max_depth=args.our_max_depth,
            torch_threads=1,
            native_tree=args.our_native_tree,
        ),
    )
    rng = random.Random(args.seed)
    scores = {"ours": 0, "nokamute": 0, "draw": 0}
    elapsed = {"ours": 0.0, "nokamute": 0.0}
    moves = {"ours": 0, "nokamute": 0}
    our_nodes = 0

    with UHPClient(args.nokamute) as nokamute:
        print("Nokamute:", " | ".join(nokamute.info))
        for game_index in range(args.games):
            state = GameState()
            nokamute.new_game(threads=args.nokamute_threads, table_mb=args.nokamute_table_mb)
            for _ in range(args.opening_plies):
                if state.result != GameResult.IN_PROGRESS:
                    break
                legal = state.legal_moves()
                opening_move = legal[rng.randrange(len(legal))]
                nokamute.play(state, opening_move)
                state.apply_move(opening_move)

            ours_white = (game_index % 2 == 0) != args.ours_black_first
            while state.result == GameResult.IN_PROGRESS and state.turn < args.max_plies:
                our_turn = (state.current_player.value == 0) == ours_white
                start = time.perf_counter()
                if our_turn:
                    move = ours.choose_move(state)
                    elapsed["ours"] += time.perf_counter() - start
                    moves["ours"] += 1
                    our_nodes += ours.last_stats.nodes
                    nokamute.play(state, move)
                else:
                    move = nokamute.best_move(state, search)
                    elapsed["nokamute"] += time.perf_counter() - start
                    moves["nokamute"] += 1
                    # UHP bestmove is a query; echo the selected move back so
                    # Nokamute advances its internal board in lockstep.
                    nokamute.play(state, move)
                if args.trace_moves:
                    from hive_engine.uhp import move_to_uhp
                    print(f"game={game_index + 1} ply={state.turn + 1} player={'ours' if our_turn else 'nokamute'} move={move_to_uhp(state, move)}")
                state.apply_move(move)

            if state.result == GameResult.WHITE_WINS:
                winner = "ours" if ours_white else "nokamute"
            elif state.result == GameResult.BLACK_WINS:
                winner = "nokamute" if ours_white else "ours"
            else:
                winner = "draw"
            scores[winner] += 1
            print(
                f"game {game_index + 1}/{args.games}: ours={'white' if ours_white else 'black'} "
                f"result={state.result.name} winner={winner} plies={state.turn}",
                flush=True,
            )

    total = args.games
    score = (scores["ours"] + 0.5 * scores["draw"]) / total
    print("\nArena summary")
    print(f"ours={scores['ours']} nokamute={scores['nokamute']} draws={scores['draw']}")
    print(f"ours_score={score:.4f}")
    print(f"ours_mean_time={elapsed['ours'] / max(1, moves['ours']):.3f}s")
    print(f"ours_mean_nodes={our_nodes / max(1, moves['ours']):.1f}")
    print(f"nokamute_mean_time={elapsed['nokamute'] / max(1, moves['nokamute']):.3f}s")


if __name__ == "__main__":
    main()
