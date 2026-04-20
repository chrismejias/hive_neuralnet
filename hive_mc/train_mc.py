from __future__ import annotations

import argparse

from hive_mc.mc_trainer import MCTrainConfig, MCTrainer
from hive_mc.mc_transformer import HiveMoveTransformer, MCTransformerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the move-conditioned Hive transformer")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dim-ff", type=int, default=512)
    p.add_argument("--max-candidates", type=int, default=16)

    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--gumbel-considered", type=int, default=16)
    p.add_argument("--max-game-length", type=int, default=300)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_mc")
    p.add_argument("--checkpoint-keep-every", type=int, default=0)
    p.add_argument("--expansion-mask", type=int, default=0)
    p.add_argument("--draw-keep-rate", type=float, default=1.0)
    p.add_argument("--flat-gumbel", action="store_true",
                   help="Use the legacy flat 1-ply Gumbel orchestrator "
                        "(no tree search). Default: MCTS tree search.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    net_config = MCTransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        max_candidates=args.max_candidates,
    )
    train_config = MCTrainConfig(
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
    )

    net = HiveMoveTransformer(net_config)
    n_params = net.count_parameters()
    print(f"Move-conditioned transformer: {n_params:,} parameters")
    print(f"  Trunk: {net_config.num_layers} layers, d={net_config.d_model}")
    print(f"  Screening head: lightweight (move features + root CLS)")
    print(f"  Action head: successor comparison (root vs child CLS)")
    print(f"  Max candidates: {net_config.max_candidates}")
    del net

    trainer = MCTrainer(train_config, net_config)
    trainer.run()


if __name__ == "__main__":
    main()
