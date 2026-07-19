"""Small CPU arena for iterative alpha-beta versus native Gumbel MCTS."""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from hive_engine.game_state import GameResult, GameState, Move
from hive_fnn.fnn_alphabeta_player import AlphaBetaConfig, FNNAlphaBetaPlayer
from hive_fnn.fnn_native_cpu_player import FNNNativeCPUConfig, FNNNativeCPUPlayer


def _move_from_bytes(game: GameState, move_bytes) -> Move:
    """Map the packed native move back onto the engine's canonical Move."""

    selected = bytes(move_bytes.tolist())
    for move in game.legal_moves():
        if FNNAlphaBetaPlayer._cpu_move_bytes(move) == selected:
            return move
    raise RuntimeError("native search returned a move that is not legal in this state")


def _select_mcts(player: FNNNativeCPUPlayer, game: GameState) -> Move:
    packed = player.choose_move_bytes(player.state_bytes_from_game(game))
    return _move_from_bytes(game, packed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints_fnn/hive_fnn_checkpoint_1080.pt")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--mcts-sims", type=int, default=256)
    parser.add_argument("--ab-nodes", type=int, default=384)
    parser.add_argument("--ab-max-depth", type=int, default=16)
    parser.add_argument("--gumbel-noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    mcts = FNNNativeCPUPlayer.from_checkpoint(
        args.checkpoint,
        config=FNNNativeCPUConfig(
            num_simulations=args.mcts_sims,
            gumbel_considered=16,
            gumbel_noise_scale=args.gumbel_noise,
            torch_threads=1,
        ),
    )
    alphabeta = FNNAlphaBetaPlayer.from_checkpoint(
        args.checkpoint,
        config=AlphaBetaConfig(
            node_budget=args.ab_nodes,
            max_depth=args.ab_max_depth,
            torch_threads=1,
        ),
    )

    scores = {"alphabeta": 0, "mcts": 0, "draw": 0}
    total_time = {"alphabeta": 0.0, "mcts": 0.0}
    total_turns = {"alphabeta": 0, "mcts": 0}
    for game_index in range(args.games):
        game = GameState()
        # Alternate colors so each engine starts an equal number of games.
        white_name = "alphabeta" if game_index % 2 == 0 else "mcts"
        players = {
            "alphabeta": alphabeta,
            "mcts": mcts,
        }
        while game.result == GameResult.IN_PROGRESS and game.turn < args.max_plies:
            name = white_name if game.turn % 2 == 0 else (
                "mcts" if white_name == "alphabeta" else "alphabeta"
            )
            start = time.perf_counter()
            if name == "alphabeta":
                move = players[name].choose_move(game)
            else:
                move = _select_mcts(players[name], game)
            total_time[name] += time.perf_counter() - start
            total_turns[name] += 1
            game.apply_move(move)

        if game.result == GameResult.WHITE_WINS:
            winner = white_name
        elif game.result == GameResult.BLACK_WINS:
            winner = "mcts" if white_name == "alphabeta" else "alphabeta"
        else:
            winner = "draw"
        scores[winner] += 1
        print(
            f"game {game_index + 1}/{args.games}: white={white_name} "
            f"result={game.result.name} winner={winner} plies={game.turn}"
        )

    print("\nArena summary")
    print(f"alpha-beta wins: {scores['alphabeta']}")
    print(f"Gumbel-MCTS wins: {scores['mcts']}")
    print(f"draws: {scores['draw']}")
    for name in ("alphabeta", "mcts"):
        turns = max(1, total_turns[name])
        print(f"{name} mean move time: {total_time[name] / turns:.3f}s ({total_turns[name]} moves)")


if __name__ == "__main__":
    main()
