"""
CLI entry point for Transformer-based Hive training.

Usage:
    python -m hive_transformer.train --iterations 20 --games 50 --simulations 100
    python -m hive_transformer.train --d-model 256 --num-layers 12
    python -m hive_transformer.train --resume checkpoints_transformer/hive_transformer_checkpoint_0010.pt
    python -m hive_transformer.train --device cuda
    python -m hive_transformer.train --continuous-updates --no-playout-cap
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from hive_engine.device import get_device, device_summary

from archive.legacy_transformer.hive_transformer.transformer_net import TransformerConfig
from archive.modules.hive_transformer_cpu.transformer_trainer import (
    TransformerTrainConfig,
    TransformerTrainer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer-based Hive AI via AlphaZero self-play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training loop
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--games", type=int, default=50,
        help="Self-play games per iteration.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs per iteration.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Weight decay for Adam optimizer.",
    )

    # LR schedule
    parser.add_argument(
        "--lr-schedule", choices=["constant", "cosine"], default="cosine",
        help="Learning rate schedule.",
    )
    parser.add_argument(
        "--lr-warmup", type=int, default=2,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--lr-min", type=float, default=1e-5,
        help="Minimum learning rate for cosine schedule.",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Maximum gradient norm (0 to disable).",
    )

    # MCTS
    parser.add_argument(
        "--simulations", type=int, default=100,
        help="MCTS simulations per move.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="MCTS temperature for exploration.",
    )
    parser.add_argument(
        "--temp-drop", type=int, default=20,
        help="Move number to switch to greedy.",
    )
    parser.add_argument(
        "--policy-prune-threshold", type=float, default=0.0,
        help="Prune MCTS policy targets below this fraction (0 = disabled).",
    )

    # Arena
    parser.add_argument(
        "--arena-games", type=int, default=10,
        help="Number of arena evaluation games.",
    )
    parser.add_argument(
        "--arena-sims", type=int, default=50,
        help="MCTS simulations for arena games.",
    )
    parser.add_argument(
        "--arena-threshold", type=float, default=0.55,
        help="Win rate threshold to accept new model.",
    )
    parser.add_argument(
        "--continuous-updates", action="store_true",
        help="Skip arena, always promote new model.",
    )

    # Game
    parser.add_argument(
        "--max-game-length", type=int, default=200,
        help="Maximum moves per game.",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=50_000,
        help="Maximum replay buffer size.",
    )
    parser.add_argument(
        "--endgame-ratio", type=float, default=0.0,
        help="Fraction of self-play games starting from endgame positions (default: 0.0).",
    )
    parser.add_argument(
        "--queen-pressure-games", type=int, default=0,
        help="Queen-pressure curriculum examples injected per iteration (0 = disabled).",
    )

    # Playout cap randomization
    parser.add_argument(
        "--no-playout-cap", action="store_true",
        help="Disable playout cap randomization.",
    )
    parser.add_argument(
        "--playout-cap-full-frac", type=float, default=0.25,
        help="Fraction of games using full playouts.",
    )
    parser.add_argument(
        "--playout-cap-min-frac", type=float, default=0.25,
        help="Min fraction of full simulations for capped games.",
    )

    # Auxiliary heads
    parser.add_argument(
        "--no-mobility-head", action="store_true",
        help="Disable mobility auxiliary head.",
    )
    parser.add_argument(
        "--no-surround-head", action="store_true",
        help="Disable queen surround auxiliary head.",
    )
    parser.add_argument(
        "--no-final-mobility-head", action="store_true",
        help="Disable final mobility auxiliary head.",
    )
    parser.add_argument(
        "--mobility-weight", type=float, default=0.15,
        help="Loss weight for mobility auxiliary head.",
    )
    parser.add_argument(
        "--surround-weight", type=float, default=0.15,
        help="Loss weight for queen surround auxiliary head.",
    )
    parser.add_argument(
        "--final-mobility-weight", type=float, default=0.15,
        help="Loss weight for final mobility auxiliary head.",
    )

    # Transformer architecture
    parser.add_argument(
        "--d-model", type=int, default=128,
        help="Transformer model dimension.",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--num-layers", type=int, default=6,
        help="Number of transformer encoder layers.",
    )
    parser.add_argument(
        "--dim-feedforward", type=int, default=512,
        help="Feedforward dimension in transformer layers.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate.",
    )
    parser.add_argument(
        "--policy-conv-channels", type=int, default=32,
        help="Policy head conv channels.",
    )

    # Batched inference (parallel self-play)
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Concurrent self-play game threads.",
    )
    parser.add_argument(
        "--inference-batch", type=int, default=32,
        help="Max states per GPU forward pass during self-play.",
    )
    parser.add_argument(
        "--inference-wait-ms", type=float, default=10.0,
        help="Max ms to wait for inference batch to fill.",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: auto, cpu, cuda, mps.",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed precision even on CUDA.",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints_transformer",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--tensorboard-dir", type=str, default=None,
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint file.",
    )

    return parser.parse_args(argv)


def _build_train_config(args: argparse.Namespace) -> TransformerTrainConfig:
    """Build a TransformerTrainConfig from parsed CLI args."""
    return TransformerTrainConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        lr_schedule=args.lr_schedule,
        lr_warmup_iterations=args.lr_warmup,
        lr_min=args.lr_min,
        mcts_simulations=args.simulations,
        temperature=args.temperature,
        temperature_drop_move=args.temp_drop,
        policy_prune_threshold=args.policy_prune_threshold,
        arena_games=args.arena_games,
        arena_mcts_simulations=args.arena_sims,
        arena_threshold=args.arena_threshold,
        continuous_updates=args.continuous_updates,
        max_game_length=args.max_game_length,
        buffer_max_size=args.buffer_size,
        endgame_ratio=args.endgame_ratio,
        queen_pressure_games=args.queen_pressure_games,
        playout_cap_randomization=not args.no_playout_cap,
        playout_cap_full_fraction=args.playout_cap_full_frac,
        playout_cap_min_fraction=args.playout_cap_min_frac,
        mobility_loss_weight=args.mobility_weight,
        queen_surround_loss_weight=args.surround_weight,
        final_mobility_loss_weight=args.final_mobility_weight,
        num_selfplay_workers=args.workers,
        max_inference_batch_size=args.inference_batch,
        inference_wait_ms=args.inference_wait_ms,
        device=args.device,
        use_amp=False if args.no_amp else None,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.resume:
        # Build the full config from CLI args, then pass all fields as overrides.
        # from_checkpoint() handles num_iterations specially (adds to checkpoint iter).
        # This ensures every CLI arg takes effect on resume — no manual list needed.
        overrides = dataclasses.asdict(_build_train_config(args))
        trainer = TransformerTrainer.from_checkpoint(
            args.resume, config_overrides=overrides
        )
        print(f"Resuming from checkpoint: {args.resume}")
    else:
        train_config = _build_train_config(args)

        net_config = TransformerConfig(
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            policy_conv_channels=args.policy_conv_channels,
            aux_mobility_enabled=not args.no_mobility_head,
            aux_queen_surround_enabled=not args.no_surround_head,
            aux_final_mobility_enabled=not args.no_final_mobility_head,
        )

        trainer = TransformerTrainer(
            config=train_config, net_config=net_config
        )

    # Print config summary
    device = trainer.device
    print(f"\n{'='*60}")
    print("Hive Transformer Training Configuration")
    print(f"{'='*60}")
    print(f"  Device:         {device} ({device_summary(device)})")
    print(f"  AMP:            {trainer.use_amp}")
    print(f"  d_model:        {trainer.net_config.d_model}")
    print(f"  num_heads:      {trainer.net_config.num_heads}")
    print(f"  num_layers:     {trainer.net_config.num_layers}")
    print(f"  dim_feedforward:{trainer.net_config.dim_feedforward}")
    print(f"  Parameters:     {trainer.best_net.count_parameters():,}")
    print(f"  Iterations:     {trainer.config.num_iterations}")
    print(f"  Games/iter:     {trainer.config.games_per_iteration}")
    print(f"  MCTS sims:      {trainer.config.mcts_simulations}")
    print(f"  LR:             {trainer.config.learning_rate}")
    print(f"  LR schedule:    {trainer.config.lr_schedule}")
    print(f"  Batch size:     {trainer.config.batch_size}")
    print(f"  Playout cap:    {trainer.config.playout_cap_randomization}")
    print(f"  Continuous:     {trainer.config.continuous_updates}")
    print(f"  Policy prune:   {trainer.config.policy_prune_threshold}")
    print(f"  Aux mobility:   {trainer.net_config.aux_mobility_enabled}")
    print(f"  Aux surround:   {trainer.net_config.aux_queen_surround_enabled}")
    print(f"  Aux final mob:  {trainer.net_config.aux_final_mobility_enabled}")
    print(f"  Checkpoints:    {trainer.config.checkpoint_dir}")
    print(f"{'='*60}\n")

    trainer.run()


if __name__ == "__main__":
    main()
