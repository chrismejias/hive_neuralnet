"""Text-mode single-game CPU FNN player.

Examples:
    python play_fnn_cpu.py --checkpoint checkpoints_fnn/baseline_legacy/hive_fnn_checkpoint_0200.pt
    python play_fnn_cpu.py --checkpoint ... --mode human_vs_ai --color black --simulations 512
"""

from __future__ import annotations

import argparse

from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.play import display_board, display_legal_moves, human_choose_move
from hive_engine.pieces import ALL_EXPANSIONS, Color
from hive_fnn.fnn_cpu_player import FNNCPUPlayer, FNNCPUMCTSConfig


def _describe_move(move: Move) -> str:
    if move.move_type == MoveType.PASS:
        return "PASS"
    if move.move_type == MoveType.PLACE:
        return f"PLACE {move.piece.piece_type.name} at ({move.to.q:+d}, {move.to.r:+d})"
    return (
        f"MOVE {move.piece.piece_type.name} "
        f"({move.from_pos.q:+d}, {move.from_pos.r:+d}) -> "
        f"({move.to.q:+d}, {move.to.r:+d})"
    )


def play_game(mode: str, player: FNNCPUPlayer, human_color: Color) -> None:
    game = GameState(expansions=ALL_EXPANSIONS)
    print(f"\n{'=' * 50}")
    print(f"FNN CPU {mode.replace('_', ' ').upper()}")
    if mode == "human_vs_ai":
        print(f"Human color: {human_color.name}")
    print(f"{'=' * 50}")

    while game.result == GameResult.IN_PROGRESS and game.turn < 300:
        display_board(game)
        if mode == "human_vs_ai" and game.current_player == human_color:
            move = human_choose_move(game)
            print(f"  You chose: {_describe_move(move)}")
        else:
            print(f"  {game.current_player.name} FNN CPU is thinking...")
            move = player.choose_move(game)
            print(f"  AI: {_describe_move(move)}")
        game.apply_move(move)

    display_board(game)
    print(f"\n{'=' * 50}")
    if game.result == GameResult.WHITE_WINS:
        print("WHITE WINS")
    elif game.result == GameResult.BLACK_WINS:
        print("BLACK WINS")
    else:
        print("DRAW")
    print(f"Game lasted {game.turn} plies")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a single CPU FNN Hive game")
    parser.add_argument("--checkpoint", required=True, help="FNN checkpoint path")
    parser.add_argument(
        "--mode",
        choices=["human_vs_ai", "ai_vs_ai"],
        default="ai_vs_ai",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Human color in human_vs_ai mode",
    )
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--gumbel-root",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CPU Gumbel-root halving instead of plain CPU PUCT.",
    )
    parser.add_argument("--gumbel-considered", type=int, default=16)
    parser.add_argument("--gumbel-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--root-workers",
        type=int,
        default=2,
        help="Optional process count for parallel root-candidate Gumbel search.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional torch CPU thread count for FNN forward passes",
    )
    args = parser.parse_args()

    player = FNNCPUPlayer.from_checkpoint(
        args.checkpoint,
        config=FNNCPUMCTSConfig(
            num_simulations=args.simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
            use_gumbel_root=args.gumbel_root,
            gumbel_considered=args.gumbel_considered,
            gumbel_noise_scale=args.gumbel_noise_scale,
            root_parallel_workers=args.root_workers,
            torch_threads=args.threads,
        ),
    )
    human_color = Color.WHITE if args.color == "white" else Color.BLACK
    play_game(args.mode, player, human_color)


if __name__ == "__main__":
    main()
