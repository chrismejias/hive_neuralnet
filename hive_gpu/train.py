"""
CLI entry point for GPU-accelerated Hive training.

Uses batched GPU MCTS (CPU trees + GPU inference) for fast self-play.

Usage:
    python -m hive_gpu --iterations 20 --games 64 --simulations 100
    python -m hive_gpu --model-size large --games 32
    python -m hive_gpu --resume checkpoints_gpu/hive_gpu_checkpoint_0010.pt
    python -m hive_gpu --device cuda --no-amp
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import io
import os


class _Tee(io.TextIOBase):
    """Write to both a stream and a file simultaneously."""

    def __init__(self, stream: io.TextIOBase, path: str) -> None:
        self._stream = stream
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._file.write(s)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return self._stream.encoding or "utf-8"  # type: ignore[union-attr]


from hive_engine.device import get_device, device_summary

try:
    from hive_gnn.gnn_net import GNNNetConfig, HiveGNN
except ImportError:
    GNNNetConfig = None  # type: ignore[assignment,misc]
    HiveGNN = None       # type: ignore[assignment,misc]
from hive_transformer.transformer_net import TransformerConfig, HiveTransformer
from hive_gpu.gpu_trainer import GPUTrainConfig, GPUTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Hive AI using GPU-accelerated batched MCTS self-play.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training loop
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--games", type=int, default=64,
        help="Number of concurrent self-play games per batch.",
    )
    parser.add_argument(
        "--batches-per-iter", type=int, default=1,
        help="Number of self-play batches per iteration.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Training epochs per iteration.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
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
        "--lr-warmup", type=int, default=3,
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
        "--c-puct", type=float, default=1.5,
        help="PUCT exploration constant.",
    )
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=0.3,
        help="Dirichlet noise alpha for root exploration.",
    )
    parser.add_argument(
        "--dirichlet-epsilon", type=float, default=0.25,
        help="Dirichlet noise weight at root.",
    )
    parser.add_argument(
        "--max-game-length", type=int, default=300,
        help="Maximum moves per game.",
    )
    parser.add_argument(
        "--wave-size", type=int, default=8,
        help="Parallel MCTS sims per wave (1 = sequential).",
    )
    parser.add_argument(
        "--nn-max-batch", type=int, default=0,
        help="Max NN batch size per forward pass (0 = no limit).",
    )

    # Arena (disabled by default — always accept new model)
    parser.add_argument(
        "--arena-games", type=int, default=20,
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
        "--enable-arena", action="store_false", dest="skip_arena",
        help="Enable arena evaluation (default: skipped, new model always accepted).",
    )
    parser.set_defaults(skip_arena=True)

    # Playout cap randomization (KataGo-style) — disabled by default
    parser.add_argument(
        "--playout-cap-randomize", action="store_true", dest="playout_cap_randomize",
        help="Enable playout cap randomization (default: disabled).",
    )
    parser.set_defaults(playout_cap_randomize=False)
    parser.add_argument(
        "--playout-cap-prob", type=float, default=0.25,
        help="Probability of using full simulations per move (rest use fast sims).",
    )
    parser.add_argument(
        "--playout-cap-fast-sims", type=int, default=0,
        help="Number of fast sims (0 = auto: simulations // 8).",
    )

    # Policy softening (KataGo-style) — 0.03 by default
    parser.add_argument(
        "--policy-softening", type=float, default=0.03,
        help="Mix policy targets toward uniform over legal moves (0=off, 0.03 default).",
    )

    # Policy target pruning — zero out low-visit children
    parser.add_argument(
        "--policy-target-pruning", type=float, default=0.02,
        help="Zero out policy targets below this fraction of max visits (0=off).",
    )

    # Root policy temperature — soften NN prior before Dirichlet noise
    parser.add_argument(
        "--root-policy-temp", type=float, default=1.1,
        help="Temperature for root NN prior (>1 = softer, 1 = off).",
    )

    # Shaped Dirichlet noise — scale alpha inversely with legal move count
    parser.add_argument(
        "--no-shaped-dirichlet", action="store_false", dest="shaped_dirichlet",
        help="Disable shaped Dirichlet noise (default: enabled, alpha=10/N_legal).",
    )
    parser.set_defaults(shaped_dirichlet=True)

    # Policy surprise weighting — KL-based sample prioritization
    parser.add_argument(
        "--policy-surprise-weight", type=float, default=1.0,
        help="Scale for KL-based surprise weighting in replay buffer (0=off).",
    )

    # Value head uncertainty — enabled by default
    parser.add_argument(
        "--no-predict-uncertainty", action="store_false", dest="predict_uncertainty",
        help="Disable value head uncertainty prediction (default: enabled).",
    )
    parser.set_defaults(predict_uncertainty=True)

    # MCGS (Monte Carlo Graph Search) — opt-in, off by default
    parser.add_argument(
        "--mcgs", action="store_true", dest="use_mcgs",
        help="Enable MCGS (DAG graph search with transposition detection). "
             "Only beneficial at high sim counts (800+). Default: MCTS tree.",
    )
    parser.set_defaults(use_mcgs=False)

    # Draw downsampling — keep only a fraction of drawn games
    parser.add_argument(
        "--draw-keep-rate", type=float, default=1.0,
        help="Fraction of drawn games to keep for training (0.25 = discard 75%% of draws). "
             "Decisive games are always kept. Default: 1.0 (keep all).",
    )

    # Queen pressure value shaping — soft value targets for drawn games
    parser.add_argument(
        "--queen-pressure-scale", type=float, default=0.4,
        help="For drawn games, use queen surround differential as a soft value target. "
             "value = scale * (opp_queen_surrounded - own_queen_surrounded) / 6. "
             "0.0 = disabled. Default: 0.4.",
    )

    # Expansion pieces
    parser.add_argument(
        "--expansion", type=int, default=0,
        help="Expansion mask: 3-bit (1=Mosquito, 2=Ladybug, 4=Pillbug, 7=all). "
             "-1 = random each iteration. Default: 0 (base game only).",
    )

    # Endgame curriculum learning
    parser.add_argument(
        "--endgame-frac", type=float, default=0.0,
        help="Fraction of games per iteration to start from near-endgame positions "
             "(both queens ~surrounded).  0.0 = disabled, 1.0 = all games. "
             "Only active with --gpu-native. Default: 0.0.",
    )
    parser.add_argument(
        "--endgame-surround", type=int, default=5,
        help="Target neighbor count for both queens in endgame positions "
             "(4 or 5; 5 = one move from losing). Default: 5.",
    )

    # GPU-native MCTS — opt-in, off by default
    parser.add_argument(
        "--gpu-native", action="store_true", dest="use_gpu_native",
        help="Use GPU-native MCTS (tree on GPU, CUDA kernels for select/expand/backprop). "
             "~2x faster than CPU-tree MCTS.",
    )
    parser.set_defaults(use_gpu_native=False)

    # Gumbel AlphaZero search — on by default, opt-out with --no-gumbel
    parser.add_argument(
        "--no-gumbel", action="store_false", dest="use_gumbel",
        help="Disable Gumbel AlphaZero search and fall back to GPU-native MCTS.",
    )
    parser.set_defaults(use_gumbel=True)
    parser.add_argument(
        "--gumbel-considered", type=int, default=32,
        help="Max actions to consider in Gumbel sequential halving (top-k before halving).",
    )
    parser.add_argument(
        "--gumbel-c-visit", type=float, default=50.0,
        help="Gumbel Q-transform c_visit parameter.",
    )
    parser.add_argument(
        "--gumbel-c-scale", type=float, default=1.0,
        help="Gumbel Q-transform c_scale parameter.",
    )

    # Network architecture
    parser.add_argument(
        "--model-size", choices=["small", "large"], default="small",
        help="GNN model size preset (small: ~8M params, large: ~12M params).",
    )
    parser.add_argument(
        "--encoder-type", choices=["gnn", "transformer"], default="transformer",
        help="Encoder type for GPU MCTS (transformer recommended).",
    )

    # Replay buffer
    parser.add_argument(
        "--buffer-size", type=int, default=50_000,
        help="Maximum replay buffer size.",
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
        "--checkpoint-dir", type=str, default="checkpoints_gpu",
        help="Directory for checkpoints.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint file.",
    )

    # Logging
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Append all stdout/stderr output to this file (in addition to the terminal).",
    )

    return parser.parse_args(argv)


def _build_train_config(args: argparse.Namespace) -> GPUTrainConfig:
    """Build a GPUTrainConfig from parsed CLI args."""
    return GPUTrainConfig(
        num_iterations=args.iterations,
        games_per_batch=args.games,
        batches_per_iteration=args.batches_per_iter,
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
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        max_game_length=args.max_game_length,
        arena_games=args.arena_games,
        arena_mcts_simulations=args.arena_sims,
        arena_threshold=args.arena_threshold,
        encoder_type=args.encoder_type,
        wave_size=args.wave_size,
        nn_max_batch=args.nn_max_batch,
        buffer_max_size=args.buffer_size,
        device=args.device,
        use_amp=False if args.no_amp else None,
        checkpoint_dir=args.checkpoint_dir,
        skip_arena=args.skip_arena,
        playout_cap_randomize=args.playout_cap_randomize,
        playout_cap_randomize_prob=args.playout_cap_prob,
        playout_cap_fast_sims=args.playout_cap_fast_sims,
        policy_softening=args.policy_softening,
        policy_surprise_weight=args.policy_surprise_weight,
        policy_target_pruning=args.policy_target_pruning,
        root_policy_temp=args.root_policy_temp,
        shaped_dirichlet=args.shaped_dirichlet,
        use_mcgs=args.use_mcgs,
        use_gpu_native=args.use_gpu_native,
        use_gumbel=args.use_gumbel,
        gumbel_max_considered=args.gumbel_considered,
        gumbel_c_visit=args.gumbel_c_visit,
        gumbel_c_scale=args.gumbel_c_scale,
        draw_keep_rate=args.draw_keep_rate,
        queen_pressure_scale=args.queen_pressure_scale,
        expansion_mask=args.expansion,
        endgame_frac=args.endgame_frac,
        endgame_surround=args.endgame_surround,
    )


def _get_net_config_and_class(args: argparse.Namespace):
    """Return (net_config, net_class) preset based on --encoder-type and --model-size."""
    predict_uncertainty = args.predict_uncertainty
    if args.encoder_type == "transformer":
        cfg = TransformerConfig.large() if args.model_size == "large" else TransformerConfig.small()
        cfg.predict_uncertainty = predict_uncertainty
        return cfg, HiveTransformer
    else:
        cfg = GNNNetConfig.large() if args.model_size == "large" else GNNNetConfig.small()
        cfg.predict_uncertainty = predict_uncertainty
        return cfg, HiveGNN


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Tee stdout/stderr to log file in append mode before any other output
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        sys.stdout = _Tee(sys.stdout, args.log_file)
        sys.stderr = _Tee(sys.stderr, args.log_file)

    if args.resume:
        train_config = _build_train_config(args)
        overrides = dataclasses.asdict(train_config)
        net_config, net_class = _get_net_config_and_class(args)
        # Pass overridable net_config fields so CLI flags (e.g. disabling aux
        # heads) take effect when resuming from an older checkpoint.
        net_cfg_overrides = {
            "aux_mobility_enabled": net_config.aux_mobility_enabled,
            "aux_final_mobility_enabled": net_config.aux_final_mobility_enabled,
            "predict_uncertainty": net_config.predict_uncertainty,
        }
        trainer = GPUTrainer.from_checkpoint(
            args.resume,
            net_class=net_class,
            config_overrides=GPUTrainConfig(**overrides),
            net_config_overrides=net_cfg_overrides,
        )
        print(f"Resuming from checkpoint: {args.resume}")
    else:
        train_config = _build_train_config(args)
        net_config, net_class = _get_net_config_and_class(args)
        trainer = GPUTrainer(config=train_config, net_config=net_config, net_class=net_class)

    # Print config summary
    device = trainer.device
    cfg = trainer.config
    net_cfg = trainer.net_config
    net = trainer.best_net
    num_params = sum(p.numel() for p in net.parameters())
    print(f"\n{'='*60}")
    print("Hive GPU Training Configuration")
    print(f"{'='*60}")
    print(f"  Device:       {device} ({device_summary(device)})")
    print(f"  AMP:          {trainer.use_amp}")
    print(f"  Model size:   {args.model_size} ({num_params:,} params)")
    print(f"  Encoder:      {cfg.encoder_type}")
    print(f"  Iterations:   {cfg.num_iterations}")
    print(f"  Games/batch:  {cfg.games_per_batch}  (×{cfg.batches_per_iteration} batches/iter)")
    fast_sims = cfg.playout_cap_fast_sims or max(1, cfg.mcts_simulations // 8)
    pcr_str = (
        f"  Playout cap:  prob={cfg.playout_cap_randomize_prob} fast={fast_sims}"
        if cfg.playout_cap_randomize else "  Playout cap:  disabled"
    )
    print(f"  MCTS sims:    {cfg.mcts_simulations}  (wave_size={cfg.wave_size})")
    print(pcr_str)
    print(f"  LR:           {cfg.learning_rate}  ({cfg.lr_schedule})")
    print(f"  Batch size:   {cfg.batch_size}")
    print(f"  Buffer:       {cfg.buffer_max_size:,}")
    arena_str = "skipped (auto-accept)" if cfg.skip_arena else f"{cfg.arena_games} games @ {cfg.arena_mcts_simulations} sims"
    print(f"  Arena:        {arena_str}")
    print(f"  Policy soft:  {cfg.policy_softening}")
    print(f"  Pol pruning:  {cfg.policy_target_pruning}")
    print(f"  Root P temp:  {cfg.root_policy_temp}")
    print(f"  Shaped Dir:   {cfg.shaped_dirichlet}")
    print(f"  Surprise wt:  {cfg.policy_surprise_weight}")
    print(f"  Draw keep:    {cfg.draw_keep_rate:.0%} of draws kept")
    qp_str = f"{cfg.queen_pressure_scale}" if cfg.queen_pressure_scale > 0 else "disabled"
    print(f"  Q-pressure:   {qp_str}")
    print(f"  Uncertainty:  {net_cfg.predict_uncertainty}")
    if cfg.use_gumbel:
        search_str = f'Gumbel AlphaZero (k={cfg.gumbel_max_considered})'
    elif cfg.use_mcgs:
        search_str = 'MCGS (DAG)'
    elif cfg.use_gpu_native:
        search_str = 'MCTS GPU-native'
    else:
        search_str = 'MCTS (tree)'
    print(f"  Search:       {search_str}")
    print(f"  Checkpoints:  {cfg.checkpoint_dir}")
    if cfg.expansion_mask < 0:
        exp_str = "random per game (all 8 subsets, split across sub-batches)"
    elif cfg.expansion_mask == 0:
        exp_str = "base game only"
    else:
        pieces = []
        if cfg.expansion_mask & 1: pieces.append("Mosquito")
        if cfg.expansion_mask & 2: pieces.append("Ladybug")
        if cfg.expansion_mask & 4: pieces.append("Pillbug")
        exp_str = "+".join(pieces)
    print(f"  Expansions:   {exp_str}")
    if cfg.endgame_frac > 0.0:
        print(f"  Endgame:      {cfg.endgame_frac:.0%} of games from endgame "
              f"(surround={cfg.endgame_surround})")
    else:
        print(f"  Endgame:      disabled (all games from start)")
    print(f"{'='*60}\n")

    trainer.run()


if __name__ == "__main__":
    main()
