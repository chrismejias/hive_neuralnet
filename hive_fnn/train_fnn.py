"""CLI entry point for FNN training."""

from __future__ import annotations

import argparse

from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_fnn.fnn_trainer import FNNTrainConfig, FNNTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the HiveGo-style feedforward network",
    )
    # Network architecture
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--action-hidden", type=int, default=32)
    p.add_argument(
        "--preset",
        choices=["small", "medium", "large"],
        default=None,
        help="Use a preset config (overrides --hidden-dim etc.)",
    )

    # Self-play
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--gumbel-considered", type=int, default=16)
    p.add_argument("--max-game-length", type=int, default=300)

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_fnn")
    p.add_argument("--checkpoint-keep-every", type=int, default=0)
    p.add_argument("--expansion-mask", type=int, default=0)
    p.add_argument("--draw-keep-rate", type=float, default=1.0)
    p.add_argument("--flat-gumbel", action="store_true",
                   help="Use the legacy flat 1-ply fused-kernel orchestrator "
                        "(no tree search). Default: MCTS tree search.")
    p.add_argument("--puct", action="store_true",
                   help="Use plain PUCT MCTS at the root instead of Gumbel root "
                        "halving. Implemented in a separate orchestrator file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.preset:
        net_config = getattr(FNNConfig, args.preset)()
    else:
        net_config = FNNConfig(
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            action_hidden=args.action_hidden,
        )

    train_config = FNNTrainConfig(
        num_iterations=args.iterations,
        games_per_batch=args.games,
        mcts_simulations=args.simulations,
        max_num_considered=args.gumbel_considered,
        max_game_length=args.max_game_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        buffer_max_size=args.buffer_size,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_keep_every=args.checkpoint_keep_every,
        expansion_mask=args.expansion_mask,
        draw_keep_rate=args.draw_keep_rate,
        flat_gumbel=args.flat_gumbel,
        use_puct=args.puct,
    )

    net = HiveFNN(net_config)
    n_params = net.count_parameters()
    print(f"HiveGo-style FNN: {n_params:,} parameters")
    print(f"  Board encoder: {net_config.feat_dim} -> {net_config.hidden_dim} -> {net_config.embed_dim}")
    print(f"  Action tower: {net_config.embed_dim * 2} -> {net_config.action_hidden} -> 1")
    print(f"  Value head: {net_config.embed_dim} -> 1 -> tanh")
    if args.flat_gumbel:
        print("  Search: flat Gumbel (legacy fused kernel)")
    elif args.puct:
        print("  Search: plain PUCT MCTS")
    else:
        print("  Search: Gumbel-root MCTS")
    del net

    trainer = FNNTrainer(train_config, net_config)
    trainer.run()


if __name__ == "__main__":
    main()
