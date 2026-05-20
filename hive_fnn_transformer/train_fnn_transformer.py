"""CLI entry point for hybrid FNN transformer training."""

from __future__ import annotations

import argparse
import textwrap

from hive_fnn.fnn_network import FNNConfig
from hive_fnn_transformer.fnn_transformer_net import HybridGNNConfig, HiveHybridGNN
from hive_fnn_transformer.fnn_transformer_trainer import HybridTrainConfig, HybridTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the hybrid FNN-policy + transformer-value model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              cd /workspace/hive_neuralnet
              python3.11 -u -m hive_fnn_transformer.train_fnn_transformer \\
                --preset large \\
                --iterations 1 \\
                --games 32 \\
                --simulations 64 \\
                --gumbel-considered 16 \\
                --checkpoint-dir checkpoints_fnn_transformer
        """),
    )

    p.add_argument("--preset", choices=["small", "large"], default="large")
    p.add_argument("--graph-hidden-dim", type=int, default=None)
    p.add_argument("--graph-layers", type=int, default=None)
    p.add_argument(
        "--graph-radius",
        type=int,
        default=2,
        help="Deprecated for the transformer trunk; retained for CLI compatibility.",
    )
    p.add_argument("--graph-mlp-hidden", type=int, default=None)
    p.add_argument("--value-hidden", type=int, default=None)
    p.add_argument("--fnn-preset", choices=["small", "medium", "large"], default=None)

    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument(
        "--simulation-schedule",
        type=str,
        default=None,
        help="Optional comma-separated per-iteration simulation schedule.",
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
    p.add_argument(
        "--short-forced-win-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--probe-win-in-one",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--probe-check-opponent-wins",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--probe-win-in-two",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--max-game-length", type=int, default=300)
    p.add_argument(
        "--gumbel-wave-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--policy-target-temperature", type=float, default=2.0)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_fnn_transformer")
    p.add_argument("--checkpoint-keep-every", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--init-fnn",
        type=str,
        default=None,
        help="Initialize the FNN policy submodule from an FNN checkpoint.",
    )
    p.add_argument("--expansion-mask", type=int, default=0)
    p.add_argument("--draw-keep-rate", type=float, default=1.0)
    return p.parse_args()


def _build_net_config(args: argparse.Namespace) -> HybridGNNConfig:
    cfg = HybridGNNConfig.large() if args.preset == "large" else HybridGNNConfig.small()
    if args.fnn_preset:
        cfg.fnn_config = getattr(FNNConfig, args.fnn_preset)()
    cfg.graph_radius = args.graph_radius
    if args.graph_hidden_dim is not None:
        cfg.graph_hidden_dim = args.graph_hidden_dim
    if args.graph_layers is not None:
        cfg.graph_layers = args.graph_layers
    if args.graph_mlp_hidden is not None:
        cfg.graph_mlp_hidden = args.graph_mlp_hidden
    if args.value_hidden is not None:
        cfg.value_hidden = args.value_hidden
    return cfg


def main() -> None:
    args = parse_args()
    sim_schedule = ()
    if args.simulation_schedule:
        sim_schedule = tuple(
            int(part.strip())
            for part in args.simulation_schedule.split(",")
            if part.strip()
        )

    net_config = _build_net_config(args)
    train_config = HybridTrainConfig(
        num_iterations=args.iterations,
        games_per_batch=args.games,
        mcts_simulations=args.simulations,
        simulation_schedule=sim_schedule,
        max_num_considered=args.gumbel_considered,
        queen_surround_reserve_slots=args.queen_surround_reserve_slots,
        queen_surround_reserve_immobile_only=args.queen_surround_reserve_immobile_only,
        max_game_length=args.max_game_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        buffer_max_size=args.buffer_size,
        policy_target_temperature=args.policy_target_temperature,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_keep_every=args.checkpoint_keep_every,
        expansion_mask=args.expansion_mask,
        draw_keep_rate=args.draw_keep_rate,
        graph_radius=args.graph_radius,
        gumbel_wave_parallel=args.gumbel_wave_parallel,
        short_forced_win_probe=args.short_forced_win_probe,
        probe_win_in_one=args.probe_win_in_one,
        probe_check_opponent_wins=args.probe_check_opponent_wins,
        probe_win_in_two=args.probe_win_in_two,
    )

    net = HiveHybridGNN(net_config)
    print(f"FNN Transformer: {net.count_parameters():,} parameters")
    print(
        f"  FNN feature encoder: {net_config.fnn_config.feat_dim} -> "
        f"{net_config.fnn_config.hidden_dim} -> {net_config.fnn_config.embed_dim}"
    )
    policy_in = net_config.fnn_config.embed_dim * 2 + net.graph_trunk.out_dim
    print(
        f"  Transformer-aware policy: {policy_in} -> "
        f"{net_config.fnn_config.action_hidden} -> 1"
    )
    print(
        f"  Relative transformer/value: hidden={net_config.graph_hidden_dim}, "
        f"layers={net_config.graph_layers}, heads={net_config.num_heads}, "
        f"max_tokens={net_config.max_piece_tokens}"
    )
    print(
        "  Search: Gumbel-root MCTS with non-root PUCT MCTS "
        f"(wave_parallel={train_config.gumbel_wave_parallel})"
    )
    if train_config.queen_surround_reserve_slots > 0:
        print(
            "  Root reserve: "
            f"{train_config.queen_surround_reserve_slots} surround slots "
            f"(immobile_only={train_config.queen_surround_reserve_immobile_only})"
        )
    if train_config.short_forced_win_probe:
        print(
            "  Tactical probe:"
            f" win1={train_config.probe_win_in_one}"
            f" oppwin={train_config.probe_check_opponent_wins}"
            f" win2={train_config.probe_win_in_two}"
        )
    del net

    trainer = HybridTrainer(train_config, net_config)
    if args.init_fnn:
        trainer.load_fnn_checkpoint(args.init_fnn)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
