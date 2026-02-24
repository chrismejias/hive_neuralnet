"""
Training CLI for Hive AlphaZero.

Provides a command-line interface to launch, configure, and resume
training runs. Supports overriding individual config values via flags
and resuming from a saved checkpoint.

Usage:
    # Start fresh training with defaults:
    python -m hive_engine.train

    # Customize key parameters:
    python -m hive_engine.train --iterations 50 --games 100 --simulations 200

    # Use a small/fast network:
    python -m hive_engine.train --net small --epochs 3

    # Resume from checkpoint:
    python -m hive_engine.train --resume checkpoints/hive_checkpoint_0010.pt

    # Resume with overridden settings:
    python -m hive_engine.train --resume checkpoints/hive_checkpoint_0010.pt \\
        --iterations 100 --lr 5e-4
"""

from __future__ import annotations

import argparse
import sys

from hive_engine.curriculum import CurriculumConfig, run_curriculum
from hive_engine.device import device_summary
from hive_engine.neural_net import NetConfig
from hive_engine.trainer import Trainer, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Hive AI using AlphaZero self-play.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Resume ────────────────────────────────────────────────────
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume training from a checkpoint file.",
    )

    # ── Network ───────────────────────────────────────────────────
    p.add_argument(
        "--net",
        choices=["small", "large"],
        default="small",
        help="Network size preset (default: small).",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=None,
        metavar="N",
        help="Override number of residual blocks.",
    )
    p.add_argument(
        "--filters",
        type=int,
        default=None,
        metavar="N",
        help="Override number of convolutional filters.",
    )

    # ── Self-play ─────────────────────────────────────────────────
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar="N",
        help="Total training iterations (default: 20).",
    )
    p.add_argument(
        "--games",
        type=int,
        default=None,
        metavar="N",
        help="Self-play games per iteration (default: 50).",
    )
    p.add_argument(
        "--simulations",
        type=int,
        default=None,
        metavar="N",
        help="MCTS simulations per move (default: 100).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Exploration temperature (default: 1.0).",
    )
    p.add_argument(
        "--temp-drop",
        type=int,
        default=None,
        metavar="MOVE",
        help="Move number to drop temperature to 0 (default: 20).",
    )

    # ── Endgame bootstrap ─────────────────────────────────────────
    p.add_argument(
        "--endgame-ratio",
        type=float,
        default=None,
        help="Fraction of games starting from endgame positions (default: 0.0).",
    )

    # ── Training ──────────────────────────────────────────────────
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Training batch size (default: 64).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Training epochs per iteration (default: 5).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-3).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (default: 1e-4).",
    )

    # ── LR Scheduling ─────────────────────────────────────────────
    p.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default=None,
        help="LR schedule type (default: constant).",
    )
    p.add_argument(
        "--lr-warmup",
        type=int,
        default=None,
        metavar="N",
        help="Linear warmup iterations (default: 0).",
    )
    p.add_argument(
        "--lr-min",
        type=float,
        default=None,
        help="Minimum LR for cosine schedule (default: 1e-5).",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Max gradient norm (0=disabled, default: 0).",
    )

    # ── Data augmentation ─────────────────────────────────────────
    p.add_argument(
        "--no-augment",
        action="store_true",
        default=False,
        help="Disable 6-fold hex rotational augmentation.",
    )

    # ── TensorBoard ───────────────────────────────────────────────
    p.add_argument(
        "--tensorboard-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="TensorBoard log directory (default: disabled).",
    )

    # ── Device & AMP ──────────────────────────────────────────────
    p.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEV",
        help="Compute device: auto, cuda, cuda:0, mps, cpu (default: auto).",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        default=False,
        help="Disable mixed precision (auto-enabled on CUDA).",
    )

    # ── Curriculum ────────────────────────────────────────────────
    p.add_argument(
        "--curriculum",
        type=str,
        default=None,
        metavar="PATH",
        help="Run curriculum training from a JSON config file.",
    )
    p.add_argument(
        "--curriculum-default",
        action="store_true",
        default=False,
        help="Run the default 3-phase endgame curriculum.",
    )

    # ── Buffer ────────────────────────────────────────────────────
    p.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        metavar="N",
        help="Max replay buffer size (default: 50000).",
    )

    # ── Arena ─────────────────────────────────────────────────────
    p.add_argument(
        "--arena-games",
        type=int,
        default=None,
        metavar="N",
        help="Arena evaluation games (default: 20).",
    )
    p.add_argument(
        "--arena-threshold",
        type=float,
        default=None,
        help="Win rate to accept new model (default: 0.55).",
    )
    p.add_argument(
        "--arena-sims",
        type=int,
        default=None,
        metavar="N",
        help="MCTS simulations in arena games (default: 50).",
    )

    # ── Parallel ──────────────────────────────────────────────────
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel self-play workers (default: 1).",
    )

    # ── Output ────────────────────────────────────────────────────
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for checkpoints (default: checkpoints).",
    )
    p.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path for JSON-lines metrics log.",
    )
    p.add_argument(
        "--max-game-length",
        type=int,
        default=None,
        metavar="N",
        help="Max moves per game (default: 300).",
    )

    return p


def apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Apply CLI argument overrides to a TrainConfig."""
    if args.iterations is not None:
        config.num_iterations = args.iterations
    if args.games is not None:
        config.games_per_iteration = args.games
    if args.simulations is not None:
        config.mcts_simulations = args.simulations
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.temp_drop is not None:
        config.temperature_drop_move = args.temp_drop
    if args.endgame_ratio is not None:
        config.endgame_ratio = args.endgame_ratio
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.buffer_size is not None:
        config.buffer_max_size = args.buffer_size
    if args.arena_games is not None:
        config.arena_games = args.arena_games
    if args.arena_threshold is not None:
        config.arena_threshold = args.arena_threshold
    if args.arena_sims is not None:
        config.arena_mcts_simulations = args.arena_sims
    if args.workers is not None:
        config.num_workers = args.workers
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.metrics_file is not None:
        config.metrics_file = args.metrics_file
    if args.max_game_length is not None:
        config.max_game_length = args.max_game_length
    if args.lr_schedule is not None:
        config.lr_schedule = args.lr_schedule
    if args.lr_warmup is not None:
        config.lr_warmup_iterations = args.lr_warmup
    if args.lr_min is not None:
        config.lr_min = args.lr_min
    if args.grad_clip is not None:
        config.max_grad_norm = args.grad_clip
    if args.no_augment:
        config.augment_symmetry = False
    if args.tensorboard_dir is not None:
        config.tensorboard_dir = args.tensorboard_dir
    if args.device is not None:
        config.device = args.device
    if args.no_amp:
        config.use_amp = False
    return config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.resume:
        # Resume from checkpoint, optionally overriding config
        print(f"Resuming training from: {args.resume}")
        config_override = TrainConfig()
        config_override = apply_overrides(config_override, args)

        # Only pass overrides if the user actually specified flags
        has_overrides = any(
            getattr(args, attr) is not None
            for attr in [
                "iterations", "games", "simulations", "temperature",
                "temp_drop", "endgame_ratio", "batch_size", "epochs",
                "lr", "weight_decay", "buffer_size", "arena_games",
                "arena_threshold", "arena_sims", "workers",
                "checkpoint_dir", "metrics_file", "max_game_length",
                "lr_schedule", "lr_warmup", "lr_min", "grad_clip",
                "tensorboard_dir", "device",
            ]
        ) or args.no_augment or args.no_amp
        trainer = Trainer.from_checkpoint(
            args.resume,
            config_overrides=config_override if has_overrides else None,
        )
    else:
        # Fresh training
        # Build net config
        if args.net == "large":
            net_config = NetConfig.large()
        else:
            net_config = NetConfig.small()

        if args.blocks is not None:
            net_config.num_blocks = args.blocks
        if args.filters is not None:
            net_config.num_filters = args.filters

        # Build train config with overrides
        train_config = TrainConfig()
        train_config = apply_overrides(train_config, args)

        trainer = Trainer(config=train_config, net_config=net_config)

    # Handle curriculum mode
    if args.curriculum_default or args.curriculum:
        if args.curriculum:
            curriculum = CurriculumConfig.from_json_file(args.curriculum)
        else:
            curriculum = CurriculumConfig.default_endgame_curriculum()

        # Build base config and net config
        train_config = TrainConfig()
        train_config = apply_overrides(train_config, args)

        if args.net == "large":
            net_config = NetConfig.large()
        else:
            net_config = NetConfig.small()
        if args.blocks is not None:
            net_config.num_blocks = args.blocks
        if args.filters is not None:
            net_config.num_filters = args.filters

        print(f"\nRunning curriculum training ({len(curriculum.phases)} phases)")
        run_curriculum(
            curriculum, train_config, net_config,
            resume_path=args.resume,
        )
        return

    # Print configuration summary
    cfg = trainer.config
    print(f"\n{'─'*50}")
    print(f"  Device: {device_summary(trainer.device)}")
    if trainer.use_amp:
        print(f"  Mixed precision: on (float16)")
    print(f"  Network: {trainer.net_config.num_blocks} blocks, "
          f"{trainer.net_config.num_filters} filters "
          f"({trainer.best_net.count_parameters():,} params)")
    print(f"  Iterations: {cfg.num_iterations}")
    print(f"  Self-play: {cfg.games_per_iteration} games × "
          f"{cfg.mcts_simulations} sims")
    print(f"  Training: lr={cfg.learning_rate} ({cfg.lr_schedule}), "
          f"batch={cfg.batch_size}, epochs={cfg.num_epochs}")
    if cfg.lr_warmup_iterations > 0:
        print(f"  LR warmup: {cfg.lr_warmup_iterations} iterations")
    if cfg.max_grad_norm > 0:
        print(f"  Gradient clipping: max_norm={cfg.max_grad_norm}")
    print(f"  Augmentation: {'on (6-fold hex)' if cfg.augment_symmetry else 'off'}")
    print(f"  Arena: {cfg.arena_games} games, "
          f"threshold={cfg.arena_threshold:.0%}")
    if cfg.endgame_ratio > 0:
        print(f"  Endgame bootstrap: {cfg.endgame_ratio:.0%} of games")
    if cfg.num_workers > 1:
        print(f"  Parallel self-play: {cfg.num_workers} workers")
    if cfg.tensorboard_dir:
        print(f"  TensorBoard: {cfg.tensorboard_dir}")
    print(f"  Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"{'─'*50}\n")

    trainer.run()


if __name__ == "__main__":
    main()
