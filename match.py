"""
Quick head-to-head match between two GNN checkpoints.
Usage: python match.py <checkpoint_a> <checkpoint_b> [--games 5] [--sims 50] [--sims-a N] [--sims-b N]
"""
import argparse
import sys
import time
import numpy as np
import torch

from hive_engine.game_state import GameState, GameResult
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.pieces import Color
from hive_gnn.gnn_trainer import GNNTrainer
from hive_gnn.gnn_encoder import GNNEncoder


def load_net(path: str, device: torch.device):
    trainer = GNNTrainer.from_checkpoint(path)
    net = trainer.best_net.to(device)
    net.eval()
    return net


def play_match(net_a, net_b, encoder, num_games: int, sims_a: int, sims_b: int, device, max_moves: int = 300):
    mcts_cfg_a = MCTSConfig(num_simulations=sims_a, temperature=0.0)
    mcts_cfg_b = MCTSConfig(num_simulations=sims_b, temperature=0.0)

    a_wins = 0.0
    results = []

    for g in range(num_games):
        # Alternate colours
        if g % 2 == 0:
            white_net, black_net = net_a, net_b
            white_cfg, black_cfg = mcts_cfg_a, mcts_cfg_b
            a_is = Color.WHITE
        else:
            white_net, black_net = net_b, net_a
            white_cfg, black_cfg = mcts_cfg_b, mcts_cfg_a
            a_is = Color.BLACK

        mcts_w = MCTS(white_net, encoder, white_cfg)
        mcts_b = MCTS(black_net, encoder, black_cfg)

        game = GameState()
        move_num = 0
        t0 = time.perf_counter()

        while game.result == GameResult.IN_PROGRESS and move_num < max_moves:
            if game.current_player == Color.WHITE:
                policy = mcts_w.search(game, move_num)
            else:
                policy = mcts_b.search(game, move_num)

            mask = encoder.get_legal_action_mask(game)
            action = int(np.argmax(policy))
            if mask[action] > 0:
                move = encoder.decode_action(action, game)
            else:
                legal = np.where(mask > 0)[0]
                if len(legal) == 0:
                    break
                move = encoder.decode_action(int(legal[0]), game)

            game.apply_move(move)
            move_num += 1

        elapsed = time.perf_counter() - t0
        result = game.result

        if result == GameResult.WHITE_WINS:
            a_score = 1.0 if a_is == Color.WHITE else 0.0
        elif result == GameResult.BLACK_WINS:
            a_score = 1.0 if a_is == Color.BLACK else 0.0
        else:  # draw or timeout
            a_score = 0.5

        a_wins += a_score

        a_color = "W" if a_is == Color.WHITE else "B"
        outcome = {1.0: "A wins", 0.5: "draw", 0.0: "B wins"}[a_score]
        results.append(f"  Game {g+1}: A={a_color}, {move_num} moves, "
                       f"{result.name}, {outcome}  ({elapsed:.0f}s)")
        print(results[-1], flush=True)

    return a_wins, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--sims", type=int, default=50, help="Sims for both players (overridden by --sims-a/--sims-b)")
    parser.add_argument("--sims-a", type=int, default=None, help="Sims for player A")
    parser.add_argument("--sims-b", type=int, default=None, help="Sims for player B")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sims_a = args.sims_a if args.sims_a is not None else args.sims
    sims_b = args.sims_b if args.sims_b is not None else args.sims

    device = torch.device(args.device)
    print(f"\nLoading checkpoints...")
    net_a = load_net(args.checkpoint_a, device)
    net_b = load_net(args.checkpoint_b, device)
    encoder = GNNEncoder()

    print(f"A: {args.checkpoint_a}  ({sims_a} sims)")
    print(f"B: {args.checkpoint_b}  ({sims_b} sims)")
    print(f"Games: {args.games}  Device: {args.device}")
    print()

    a_wins, results = play_match(net_a, net_b, encoder, args.games, sims_a, sims_b, device)

    print(f"\nA score: {a_wins}/{args.games} ({100*a_wins/args.games:.0f}%)")


if __name__ == "__main__":
    main()
