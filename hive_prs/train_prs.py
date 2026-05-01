"""
CLI entry point for PRS v2 training (default PRS path).

Usage:
    python -m hive_prs.train_prs [options]

For long runs, prefer the nohup pattern shown in --help: use python -u,
redirect stdin from /dev/null, append stdout/stderr to the training log, and
capture the PID.
"""

from __future__ import annotations

import argparse
import textwrap

from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.prs_trainer_v2 import PRSTrainConfigV2, PRSTrainerV2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the PRS v2 Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Long-running background launch:
              cd /workspace/hive_neuralnet
              mkdir -p checkpoints_prs_v2
              nohup python3.11 -u -m hive_prs.train_prs \\
                --iterations 1000 \\
                --games 256 \\
                --simulations 256 \\
                --max-considered 16 \\
                --checkpoint-dir checkpoints_prs_v2 \\
                --checkpoint-keep-every 50 \\
                --resume checkpoints_prs_v2/prs_v2_iter_0500.pt \\
                --wave-parallel \\
                >> checkpoints_prs_v2/training.log 2>&1 < /dev/null &
              echo $!

            Checks:
              pgrep -af 'hive_prs.train_prs|train_prs'
              tail -f checkpoints_prs_v2/training.log

            Notes:
              - The current Gumbel defaults standardize on k=16, so the
                training and profiling scripts now pass max-considered 16.
              - python -u keeps startup and per-iteration output unbuffered.
              - < /dev/null prevents the process from inheriting a terminal stdin.
              - If launched through a tool that kills detached children, use a
                persistent shell/tmux/session and run the same command in it.
        """),
    )

    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dim-ff", type=int, default=512)

    # Self-play
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--simulations", type=int, default=256)
    p.add_argument(
        "--simulation-schedule",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-iteration simulation schedule, "
            "for example '1024,2048,4096'. Overrides fixed --simulations."
        ),
    )
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
    p.add_argument("--buffer-size", type=int, default=150_000)
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
        "--deterministic-non-root",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the paper-style deterministic non-root action selection "
            "based on completed Q-values. With wave-parallel enabled, "
            "parallel sims are diversified by a temporary virtual-Q penalty."
        ),
    )
    p.add_argument(
        "--virtual-q-penalty",
        type=float,
        default=0.25,
        help=(
            "Temporary per-node Q penalty used to diversify deterministic "
            "non-root wave-parallel simulations."
        ),
    )
    p.add_argument(
        "--non-root-sigma",
        type=float,
        default=1.0,
        help=(
            "Constant sigma for non-root Gumbel selection: "
            "score(a) = log(prior[a]) + non_root_sigma * Q(a). "
            "Controls how much Q weighs against the log-prior."
        ),
    )
    p.add_argument(
        "--compile-forward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to torch.compile for tensor-only trunk/head paths. "
            "Disabled by default because some environments have unstable "
            "Inductor subprocess compilation."
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
    sim_schedule = ()
    if args.simulation_schedule:
        sim_schedule = tuple(
            int(part.strip())
            for part in args.simulation_schedule.split(",")
            if part.strip()
        )

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
        simulation_schedule=sim_schedule,
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
        deterministic_non_root=args.deterministic_non_root,
        virtual_q_penalty=args.virtual_q_penalty,
        non_root_sigma=args.non_root_sigma,
        compile_forward=args.compile_forward,
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
    if train_config.simulation_schedule:
        print(
            f"  Simulation schedule: {list(train_config.simulation_schedule)} "
            f"(k={train_config.max_num_considered}, "
            f"wave_parallel={train_config.wave_parallel}, "
            f"deterministic_non_root={train_config.deterministic_non_root}, "
            f"virtual_q_penalty={train_config.virtual_q_penalty}, "
            f"compile_forward={train_config.compile_forward})"
        )
    else:
        print(
            f"  Simulations: {train_config.mcts_simulations} "
            f"(k={train_config.max_num_considered}, "
            f"wave_parallel={train_config.wave_parallel}, "
            f"deterministic_non_root={train_config.deterministic_non_root}, "
            f"virtual_q_penalty={train_config.virtual_q_penalty}, "
            f"compile_forward={train_config.compile_forward})"
        )
    print(f"  Checkpoint dir: {train_config.checkpoint_dir}")
    print()

    trainer.run()


if __name__ == "__main__":
    main()
