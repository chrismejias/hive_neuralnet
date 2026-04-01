"""
Head-to-head match between two GPU-trainer checkpoints (GNN or Transformer).
Auto-detects architecture from net_config saved in checkpoint.

Usage:
    python match.py <checkpoint_a> <checkpoint_b> [--games 20] [--sims 100]
"""
import argparse
import time
import numpy as np
import torch

from hive_engine.game_state import GameState, GameResult
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.pieces import Color
from hive_gnn.gnn_encoder import GNNEncoder
from hive_gnn.gnn_net import GNNNetConfig, HiveGNN
from hive_transformer.transformer_net import TransformerConfig, HiveTransformer
from hive_transformer.transformer_encoder import TransformerEncoder


def load_net(path: str, device: torch.device):
    """Load a GPU-trainer checkpoint, auto-detecting GNN vs Transformer."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net_config = ckpt["net_config"]

    if isinstance(net_config, TransformerConfig):
        net = HiveTransformer(net_config)
        encoder = TransformerEncoder()
        arch = f"Transformer(d={net_config.d_model}, layers={net_config.num_layers})"
    elif isinstance(net_config, GNNNetConfig):
        net = HiveGNN(net_config)
        encoder = GNNEncoder()
        arch = f"GNN(dim={net_config.hidden_dim}, mp={net_config.num_mp_layers})"
    else:
        raise ValueError(f"Unknown net_config type: {type(net_config)}")

    net.load_state_dict(ckpt["model_state_dict"])
    net = net.to(device)
    net.eval()
    iteration = ckpt.get("iteration", "?")
    return net, encoder, arch, iteration


def play_match(
    net_a, encoder_a,
    net_b, encoder_b,
    num_games: int,
    sims_a: int,
    sims_b: int,
    max_moves: int = 300,
):
    mcts_cfg_a = MCTSConfig(num_simulations=sims_a, temperature=0.0)
    mcts_cfg_b = MCTSConfig(num_simulations=sims_b, temperature=0.0)

    a_wins = 0.0
    results = []

    for g in range(num_games):
        if g % 2 == 0:
            white_net, white_enc, white_cfg = net_a, encoder_a, mcts_cfg_a
            black_net, black_enc, black_cfg = net_b, encoder_b, mcts_cfg_b
            a_is = Color.WHITE
        else:
            white_net, white_enc, white_cfg = net_b, encoder_b, mcts_cfg_b
            black_net, black_enc, black_cfg = net_a, encoder_a, mcts_cfg_a
            a_is = Color.BLACK

        mcts_w = MCTS(white_net, white_enc, white_cfg)
        mcts_b = MCTS(black_net, black_enc, black_cfg)

        game = GameState()
        move_num = 0
        t0 = time.perf_counter()

        while game.result == GameResult.IN_PROGRESS and move_num < max_moves:
            if game.current_player == Color.WHITE:
                policy = mcts_w.search(game, move_num)
                mask = white_enc.get_legal_action_mask(game)
                action = int(np.argmax(policy))
                if mask[action] > 0:
                    move = white_enc.decode_action(action, game)
                else:
                    legal = np.where(mask > 0)[0]
                    if len(legal) == 0:
                        break
                    move = white_enc.decode_action(int(legal[0]), game)
            else:
                policy = mcts_b.search(game, move_num)
                mask = black_enc.get_legal_action_mask(game)
                action = int(np.argmax(policy))
                if mask[action] > 0:
                    move = black_enc.decode_action(action, game)
                else:
                    legal = np.where(mask > 0)[0]
                    if len(legal) == 0:
                        break
                    move = black_enc.decode_action(int(legal[0]), game)

            game.apply_move(move)
            move_num += 1

        elapsed = time.perf_counter() - t0

        if game.result == GameResult.WHITE_WINS:
            a_score = 1.0 if a_is == Color.WHITE else 0.0
        elif game.result == GameResult.BLACK_WINS:
            a_score = 1.0 if a_is == Color.BLACK else 0.0
        else:
            a_score = 0.5

        a_wins += a_score
        a_color = "W" if a_is == Color.WHITE else "B"
        outcome = {1.0: "A wins", 0.5: "draw", 0.0: "B wins"}[a_score]
        line = (f"  Game {g+1:2d}: A={a_color}  {move_num:3d} moves  "
                f"{game.result.name:<15}  {outcome}  ({elapsed:.0f}s)")
        results.append(line)
        print(line, flush=True)

    return a_wins, results


def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head match between two GPU-trainer checkpoints."
    )
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=100,
                        help="Sims for both players")
    parser.add_argument("--sims-a", type=int, default=None)
    parser.add_argument("--sims-b", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sims_a = args.sims_a if args.sims_a is not None else args.sims
    sims_b = args.sims_b if args.sims_b is not None else args.sims
    device = torch.device(args.device)

    print("\nLoading checkpoints...")
    net_a, enc_a, arch_a, iter_a = load_net(args.checkpoint_a, device)
    net_b, enc_b, arch_b, iter_b = load_net(args.checkpoint_b, device)

    params_a = sum(p.numel() for p in net_a.parameters())
    params_b = sum(p.numel() for p in net_b.parameters())

    print(f"\nA: {args.checkpoint_a}")
    print(f"   {arch_a}  iter={iter_a}  params={params_a:,}  sims={sims_a}")
    print(f"B: {args.checkpoint_b}")
    print(f"   {arch_b}  iter={iter_b}  params={params_b:,}  sims={sims_b}")
    print(f"\nPlaying {args.games} games on {args.device}...\n")

    a_wins, _ = play_match(
        net_a, enc_a, net_b, enc_b,
        args.games, sims_a, sims_b,
    )

    win_pct = 100 * a_wins / args.games
    print(f"\n{'='*50}")
    print(f"A score: {a_wins}/{args.games}  ({win_pct:.0f}%)")
    print(f"B score: {args.games - a_wins}/{args.games}  ({100 - win_pct:.0f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
