"""
Interactive play script for Hive AI.

Supports:
  - human_vs_ai: Human plays against the AI
  - ai_vs_ai: Watch two AIs play each other

Usage:
    python -m hive_engine.play --mode human_vs_ai
    python -m hive_engine.play --mode ai_vs_ai --checkpoint checkpoints/best.pt
    python -m hive_engine.play --mode ai_vs_ai --simulations 200
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from hive_engine.device import get_device, device_summary
from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.neural_net import HiveNet, NetConfig
from hive_engine.pieces import Color, PieceType
from hive_engine.hex_coord import HexCoord
from hive_engine.trainer import Trainer


# ── Display ────────────────────────────────────────────────────────


def display_board(game: GameState) -> None:
    """Print the current board state in a human-readable format."""
    print(f"\n{'─'*50}")
    print(f"Turn {game.turn} — {game.current_player.name}'s move")
    print(f"Result: {game.result.name}")

    # Display pieces on board
    board = game.board
    if board.grid:
        print("\nBoard:")
        positions = sorted(
            board.grid.keys(), key=lambda p: (p.r, p.q)
        )
        for pos in positions:
            stack = board.grid[pos]
            pieces_str = ", ".join(
                f"{p.color.name[0]}{p.piece_type.name[:2]}"
                for p in stack
            )
            print(f"  ({pos.q:+d}, {pos.r:+d}): [{pieces_str}]")
    else:
        print("\nBoard is empty.")

    # Display hands
    for color in [Color.WHITE, Color.BLACK]:
        hand = game.hand(color)
        if hand:
            types = {}
            for p in hand:
                name = p.piece_type.name
                types[name] = types.get(name, 0) + 1
            hand_str = ", ".join(f"{name}×{count}" for name, count in types.items())
            print(f"  {color.name} hand: {hand_str}")

    print(f"{'─'*50}")


def display_legal_moves(moves: list[Move]) -> None:
    """Print legal moves in a numbered list."""
    print(f"\nLegal moves ({len(moves)}):")
    for i, move in enumerate(moves):
        if move.move_type == MoveType.PASS:
            print(f"  [{i}] PASS")
        elif move.move_type == MoveType.PLACE:
            print(
                f"  [{i}] PLACE {move.piece.piece_type.name} "
                f"at ({move.to.q:+d}, {move.to.r:+d})"
            )
        elif move.move_type == MoveType.MOVE:
            print(
                f"  [{i}] MOVE {move.piece.piece_type.name} "
                f"({move.from_pos.q:+d}, {move.from_pos.r:+d}) → "
                f"({move.to.q:+d}, {move.to.r:+d})"
            )


# ── AI Player ──────────────────────────────────────────────────────


def ai_choose_move(
    game: GameState,
    net: HiveNet,
    encoder: HiveEncoder,
    config: MCTSConfig,
    move_number: int,
) -> Move:
    """Use MCTS to choose a move."""
    mcts = MCTS(net, encoder, config)
    policy = mcts.search(game, move_number=move_number)

    # Pick best action
    action = int(np.argmax(policy))
    mask = encoder.get_legal_action_mask(game)

    if mask[action] > 0:
        return encoder.decode_action(action, game)

    # Fallback to highest-prob legal action
    legal_actions = np.where(mask > 0)[0]
    best = legal_actions[np.argmax(policy[legal_actions])]
    return encoder.decode_action(int(best), game)


# ── Human Player ───────────────────────────────────────────────────


def human_choose_move(game: GameState) -> Move:
    """Let the human choose a move from the legal move list."""
    moves = game.legal_moves()
    display_legal_moves(moves)

    while True:
        try:
            choice = input(f"\nEnter move number [0-{len(moves)-1}]: ").strip()
            idx = int(choice)
            if 0 <= idx < len(moves):
                return moves[idx]
            print(f"  Invalid: must be 0-{len(moves)-1}")
        except ValueError:
            print("  Invalid: enter a number")
        except (EOFError, KeyboardInterrupt):
            print("\nGame aborted.")
            sys.exit(0)


# ── Game Loop ──────────────────────────────────────────────────────


def play_game(
    mode: str,
    net: HiveNet,
    encoder: HiveEncoder,
    mcts_config: MCTSConfig,
    human_color: Color = Color.WHITE,
) -> None:
    """Play a full game."""
    game = GameState()
    move_number = 0

    print(f"\n{'='*50}")
    print(f"  HIVE — {mode.replace('_', ' ').upper()}")
    if mode == "human_vs_ai":
        print(f"  You are playing as {human_color.name}")
    print(f"{'='*50}")

    while game.result == GameResult.IN_PROGRESS:
        display_board(game)

        if mode == "human_vs_ai" and game.current_player == human_color:
            move = human_choose_move(game)
            print(f"  You chose: {move}")
        else:
            print(f"  {game.current_player.name} AI is thinking...")
            move = ai_choose_move(game, net, encoder, mcts_config, move_number)
            if move.move_type == MoveType.PASS:
                print(f"  AI: PASS")
            elif move.move_type == MoveType.PLACE:
                print(
                    f"  AI: PLACE {move.piece.piece_type.name} "
                    f"at ({move.to.q:+d}, {move.to.r:+d})"
                )
            else:
                print(
                    f"  AI: MOVE {move.piece.piece_type.name} "
                    f"({move.from_pos.q:+d}, {move.from_pos.r:+d}) → "
                    f"({move.to.q:+d}, {move.to.r:+d})"
                )

        game.apply_move(move)
        move_number += 1

        if move_number > 300:
            print("\nGame reached 300 moves — declaring draw.")
            break

    # Final display
    display_board(game)
    print(f"\n{'='*50}")
    if game.result == GameResult.WHITE_WINS:
        print("  WHITE WINS!")
    elif game.result == GameResult.BLACK_WINS:
        print("  BLACK WINS!")
    else:
        print("  DRAW!")
    print(f"  Game lasted {move_number} moves.")
    print(f"{'='*50}\n")


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Play Hive against an AI")
    parser.add_argument(
        "--mode",
        choices=["human_vs_ai", "ai_vs_ai"],
        default="ai_vs_ai",
        help="Game mode (default: ai_vs_ai)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: random network)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Your color in human_vs_ai mode (default: white)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEV",
        help="Compute device: auto, cuda, mps, cpu (default: auto)",
    )
    args = parser.parse_args()

    # Determine device
    device = get_device(args.device)
    print(f"Device: {device_summary(device)}")

    # Load or create network
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        net = Trainer.load_checkpoint(args.checkpoint, device=device)
    else:
        print("No checkpoint specified — using random network.")
        net = HiveNet(NetConfig.small()).to(device)

    net.eval()
    encoder = HiveEncoder()
    mcts_config = MCTSConfig(
        num_simulations=args.simulations,
        temperature=0.0,  # Greedy play
    )

    human_color = Color.WHITE if args.color == "white" else Color.BLACK
    play_game(args.mode, net, encoder, mcts_config, human_color)


if __name__ == "__main__":
    main()
