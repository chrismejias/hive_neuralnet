"""
CLI entry point for PRS Transformer training.

Usage:
    python -m hive_prs.train_prs [options]
    # or
    PYTHONPATH=/workspace/hive_neuralnet python hive_prs/train_prs.py [options]
"""

from __future__ import annotations

import argparse
import sys

from hive_prs.prs_trainer import PRSTrainer, PRSTrainConfig
from hive_prs.prs_transformer import HivePRSTransformer, PRSConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the PRS Transformer")

    # Model
    p.add_argument("--d-model",   type=int,   default=128)
    p.add_argument("--num-heads", type=int,   default=8)
    p.add_argument("--num-layers",type=int,   default=6)
    p.add_argument("--dim-ff",    type=int,   default=512)
    p.add_argument("--d-key",     type=int,   default=64,
                   help="Bilinear key dimension in PRS policy head")

    # Self-play
    p.add_argument("--iterations",     type=int,   default=1500)
    p.add_argument("--games",          type=int,   default=128)
    p.add_argument("--simulations",    type=int,   default=512)
    p.add_argument("--max-considered", type=int,   default=16)
    p.add_argument("--max-game-len",   type=int,   default=300)

    # Training
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--lr-schedule",  type=str,   default="cosine",
                   choices=["cosine", "constant"])
    p.add_argument("--lr-min",       type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Buffer & checkpoints
    p.add_argument("--buffer-size",         type=int, default=100_000)
    p.add_argument("--checkpoint-dir",      type=str, default="checkpoints_prs")
    p.add_argument("--checkpoint-keep-every", type=int, default=0)
    p.add_argument("--resume",              type=str, default=None,
                   help="Path to checkpoint to resume from")

    # Misc
    p.add_argument("--expansion-mask", type=int, default=0)
    p.add_argument("--draw-keep-rate", type=float, default=1.0)
    p.add_argument("--nn-max-batch",   type=int, default=0)
    p.add_argument("--flat-gumbel",    action="store_true",
                   help="Use the legacy flat 1-ply Gumbel orchestrator "
                        "(no tree search). Default: true MCTS tree search.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    net_config = PRSConfig(
        d_model         = args.d_model,
        num_heads       = args.num_heads,
        num_layers      = args.num_layers,
        dim_feedforward = args.dim_ff,
        d_key           = args.d_key,
    )

    train_config = PRSTrainConfig(
        num_iterations         = args.iterations,
        games_per_batch        = args.games,
        mcts_simulations       = args.simulations,
        max_num_considered     = args.max_considered,
        max_game_length        = args.max_game_len,
        batch_size             = args.batch_size,
        num_epochs             = args.epochs,
        learning_rate          = args.lr,
        lr_schedule            = args.lr_schedule,
        lr_min                 = args.lr_min,
        weight_decay           = args.weight_decay,
        buffer_max_size        = args.buffer_size,
        checkpoint_dir         = args.checkpoint_dir,
        checkpoint_keep_every  = args.checkpoint_keep_every,
        expansion_mask         = args.expansion_mask,
        draw_keep_rate         = args.draw_keep_rate,
        nn_max_batch           = args.nn_max_batch,
        flat_gumbel            = args.flat_gumbel,
    )

    # Print model info
    net = HivePRSTransformer(net_config)
    n_params = net.count_parameters()
    print(f"PRS Transformer: {n_params:,} parameters")
    print(f"  d_model={net_config.d_model}, heads={net_config.num_heads}, "
          f"layers={net_config.num_layers}, dim_ff={net_config.dim_feedforward}, "
          f"d_key={net_config.d_key}")
    print(f"  Action space: {net_config.action_space_size}")
    print(f"  Position table: {net_config.max_positions} (23×23 + 1 off-board)")
    del net

    trainer = PRSTrainer(train_config, net_config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print(f"\nTraining for {train_config.num_iterations} iterations")
    print(f"  LR: {train_config.learning_rate} ({train_config.lr_schedule})")
    print(f"  Games/iter: {train_config.games_per_batch}")
    print(f"  Simulations: {train_config.mcts_simulations} "
          f"(k={train_config.max_num_considered})")
    print(f"  Checkpoint dir: {train_config.checkpoint_dir}")
    print()

    trainer.run()


if __name__ == "__main__":
    main()
