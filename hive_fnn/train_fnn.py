"""CLI entry point for FNN training.

For long runs, prefer the nohup pattern shown in --help: use python -u,
redirect stdin from /dev/null, append stdout/stderr to the training log, and
capture the PID.
"""

from __future__ import annotations

import argparse
import textwrap

from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_fnn.fnn_trainer import FNNTrainConfig, FNNTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the HiveGo-style feedforward network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Long-running background launch:
              cd /workspace/hive_neuralnet
              mkdir -p checkpoints_fnn
              nohup python3.11 -u -m hive_fnn.train_fnn \\
                --preset large \\
                --iterations 1500 \\
                --games 256 \\
                --simulations 1024 \\
                --gumbel-considered 16 \\
                --checkpoint-dir checkpoints_fnn \\
                --checkpoint-keep-every 50 \\
                --gumbel-wave-parallel \\
                >> checkpoints_fnn/training.log 2>&1 < /dev/null &
              echo $!

            Checks:
              pgrep -af 'hive_fnn.train_fnn|train_fnn'
              tail -f checkpoints_fnn/training.log

            Notes:
              - The bare train_fnn defaults map to the large FNN config
                (64/64/64) unless --preset is supplied.
              - python -u keeps startup and per-iteration output unbuffered.
              - < /dev/null prevents the process from inheriting a terminal stdin.
              - In this environment, a plain background job may be reaped when
                the shell exits. Use a persistent shell/tmux/session or an
                equivalent long-lived launcher, then run the same nohup command
                inside that session.
              - A good manual launch pattern is:
                  nohup python3.11 -u -m hive_fnn.train_fnn ... > checkpoints_fnn/training.log 2>&1 < /dev/null &
                  disown
        """),
    )
    # Network architecture
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--action-hidden", type=int, default=64)
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
    p.add_argument(
        "--simulation-schedule",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-iteration simulation schedule, "
            "for example '1024,2048,4096'. Overrides fixed --simulations."
        ),
    )
    p.add_argument("--gumbel-considered", type=int, default=16)
    p.add_argument("--queen-surround-reserve-slots", type=int, default=10)
    p.add_argument(
        "--queen-surround-reserve-immobile-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict reserved surround slots to moves that leave the opponent "
            "queen with no legal move of any kind, including pillbug or mosquito throws."
        ),
    )
    p.add_argument("--max-game-length", type=int, default=300)
    p.add_argument(
        "--gumbel-wave-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the hard-coded Gumbel per-round wave schedule "
            "(FNN: 2,4,8,16). Use --no-gumbel-wave-parallel for pure serial waves."
        ),
    )
    p.add_argument("--gumbel-wave-size", type=int, default=4)
    p.add_argument("--puct-wave-size", type=int, default=16)

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_fnn")
    p.add_argument("--checkpoint-keep-every", type=int, default=0)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    p.add_argument("--expansion-mask", type=int, default=0)
    p.add_argument("--draw-keep-rate", type=float, default=1.0)
    p.add_argument("--puct", action="store_true",
                   help="Use plain PUCT MCTS at the root instead of Gumbel root "
                        "halving. Implemented in a separate orchestrator file.")
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
        simulation_schedule=sim_schedule,
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
        use_puct=args.puct,
        gumbel_wave_parallel=args.gumbel_wave_parallel,
        gumbel_wave_size=args.gumbel_wave_size,
        puct_wave_size=args.puct_wave_size,
        queen_surround_reserve_slots=args.queen_surround_reserve_slots,
        queen_surround_reserve_immobile_only=args.queen_surround_reserve_immobile_only,
    )

    net = HiveFNN(net_config)
    n_params = net.count_parameters()
    print(f"HiveGo-style FNN: {n_params:,} parameters")
    print(f"  Board encoder: {net_config.feat_dim} -> {net_config.hidden_dim} -> {net_config.embed_dim}")
    print(f"  Action tower: {net_config.embed_dim * 2} -> {net_config.action_hidden} -> 1")
    print(f"  Value head: {net_config.embed_dim} -> 1 -> tanh")
    if args.puct:
        print(f"  Search: plain PUCT MCTS (wave_size={train_config.puct_wave_size})")
    else:
        print(
            "  Search: Gumbel-root MCTS "
            f"(wave_parallel={train_config.gumbel_wave_parallel})"
        )
        if train_config.queen_surround_reserve_slots > 0:
            print(
                "  Root reserve: "
                f"{train_config.queen_surround_reserve_slots} surround slots "
                f"(immobile_only={train_config.queen_surround_reserve_immobile_only})"
            )
    if train_config.simulation_schedule:
        print(f"  Simulation schedule: {list(train_config.simulation_schedule)}")
    del net

    trainer = FNNTrainer(train_config, net_config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
