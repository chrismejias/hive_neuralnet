"""
CLI entry point for PRS v2 training (default PRS path).

Usage:
    python -m hive_prs.train_prs [options]
"""

from __future__ import annotations

import argparse

from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.prs_trainer_v2 import PRSTrainConfigV2, PRSTrainerV2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the PRS v2 Transformer")

    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dim-ff", type=int, default=512)

    # Self-play
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--simulations", type=int, default=512)
    p.add_argument("--max-considered", type=int, default=16)
    p.add_argument("--max-game-len", type=int, default=300)

    # Training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
    )
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Buffer & checkpoints
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_prs_v2")
    p.add_argument("--checkpoint-keep-every", type=int, default=0)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Misc
    p.add_argument("--expansion-mask", type=int, default=7)
    p.add_argument("--draw-keep-rate", type=float, default=1.0)
    p.add_argument("--nn-max-batch", type=int, default=0)
    p.add_argument(
        "--wave-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the hard-coded Gumbel per-round wave schedule "
            "(PRS v2: 1,2,4,8). Use --no-wave-parallel for pure serial waves."
        ),
    )
    p.add_argument(
        "--augment-prob",
        type=float,
        default=0.5,
        help="Probability of C6 rotation augmentation per training batch",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    net_config = PRSConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
    )

    train_config = PRSTrainConfigV2(
        num_iterations=args.iterations,
        games_per_batch=args.games,
        mcts_simulations=args.simulations,
        max_num_considered=args.max_considered,
        max_game_length=args.max_game_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        buffer_max_size=args.buffer_size,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_keep_every=args.checkpoint_keep_every,
        expansion_mask=args.expansion_mask,
        draw_keep_rate=args.draw_keep_rate,
        nn_max_batch=args.nn_max_batch,
        wave_parallel=args.wave_parallel,
        augment_prob=args.augment_prob,
    )

    net = HivePRSTransformerV2(net_config)
    n_params = net.count_parameters()
    print(f"PRS v2 Transformer: {n_params:,} parameters")
    print(
        f"  d_model={net_config.d_model}, heads={net_config.num_heads}, "
        f"layers={net_config.num_layers}, dim_ff={net_config.dim_feedforward}"
    )
    print("  Policy head: structured 813-slot legal-masked head")
    print(f"  Position table: {net_config.max_positions} (23x23 + 1 off-board)")
    del net

    trainer = PRSTrainerV2(train_config, net_config)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    print(f"\nTraining for {train_config.num_iterations} iterations")
    print(f"  LR: {train_config.learning_rate} ({train_config.lr_schedule})")
    print(f"  Games/iter: {train_config.games_per_batch}")
    print(
        f"  Simulations: {train_config.mcts_simulations} "
        f"(k={train_config.max_num_considered}, "
        f"wave_parallel={train_config.wave_parallel})"
    )
    print(f"  Checkpoint dir: {train_config.checkpoint_dir}")
    print()

    trainer.run()


if __name__ == "__main__":
    main()
