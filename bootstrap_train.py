from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from hive_fnn.fnn_network import FNNConfig
from hive_fnn.fnn_replay_buffer import FNNTrainingExample
from hive_fnn.fnn_trainer import FNNTrainConfig, FNNTrainer
from hive_fnn_transformer.fnn_transformer_net import HybridGNNConfig
from hive_fnn_transformer.fnn_transformer_trainer import HybridTrainConfig, HybridTrainer
from hive_prs.action_space import MAX_BOARD
from hive_prs.prs_aux_targets import compute_articulation_target
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.prs_replay_buffer_v2 import PRSReplayBufferV2, PRSTrainingExampleV2
from hive_prs.prs_trainer_v2 import PRSConfig, PRSTrainConfigV2, PRSTrainerV2
from hive_prs.prs_transformer_v3 import HivePRSTransformerV3
from hive_prs.slot_map import N_SLOTS

import hive_gpu


def _build_fnn_trainer(args: argparse.Namespace) -> FNNTrainer:
    train_cfg = FNNTrainConfig(
        num_iterations=max(args.iterations, 1),
        games_per_batch=args.games,
        mcts_simulations=args.simulations,
        max_num_considered=args.gumbel_considered,
        max_game_length=args.max_game_length,
        batch_size=args.fnn_batch_size,
        num_epochs=args.fnn_epochs,
        learning_rate=args.fnn_lr,
        weight_decay=args.fnn_weight_decay,
        buffer_max_size=args.fnn_buffer_size,
        checkpoint_dir=args.fnn_checkpoint_dir,
        checkpoint_keep_every=args.checkpoint_keep_every,
        expansion_mask=args.expansion_mask,
        draw_keep_rate=args.draw_keep_rate,
        endgame_frac=args.endgame_frac,
        endgame_surround=args.endgame_surround,
        gumbel_wave_parallel=True,
        queen_surround_reserve_slots=args.queen_surround_reserve_slots,
        queen_surround_reserve_immobile_only=not args.no_immobile_reserve,
        short_forced_win_probe=args.short_forced_win_probe,
        probe_win_in_one=args.probe_win_in_one,
        probe_check_opponent_wins=args.probe_check_opponent_wins,
        probe_win_in_two=args.probe_win_in_two,
        policy_target_temperature=args.policy_target_temperature,
        adaptive_policy_target_temperature=args.adaptive_policy_target_temperature,
        policy_target_top1_cap=args.policy_target_top1_cap,
        policy_target_min_temperature=args.policy_target_min_temperature,
        policy_target_max_temperature=args.policy_target_max_temperature,
    )
    trainer = FNNTrainer(train_cfg, FNNConfig.large())
    trainer.load_checkpoint(args.fnn_checkpoint)
    return trainer


def _build_hybrid_trainer(args: argparse.Namespace) -> HybridTrainer | None:
    if not args.train_hybrid:
        return None
    train_cfg = HybridTrainConfig(
        num_iterations=max(args.iterations, 1),
        games_per_batch=args.games,
        mcts_simulations=args.simulations,
        max_num_considered=args.gumbel_considered,
        queen_surround_reserve_slots=args.queen_surround_reserve_slots,
        queen_surround_reserve_immobile_only=not args.no_immobile_reserve,
        max_game_length=args.max_game_length,
        batch_size=args.hybrid_batch_size,
        num_epochs=args.hybrid_epochs,
        learning_rate=args.hybrid_lr,
        weight_decay=args.hybrid_weight_decay,
        buffer_max_size=args.hybrid_buffer_size,
        checkpoint_dir=args.hybrid_checkpoint_dir,
        checkpoint_keep_every=args.checkpoint_keep_every,
        expansion_mask=args.expansion_mask,
        draw_keep_rate=args.draw_keep_rate,
        graph_radius=2,
        gumbel_wave_parallel=True,
        policy_target_temperature=args.policy_target_temperature,
        adaptive_policy_target_temperature=args.adaptive_policy_target_temperature,
        policy_target_top1_cap=args.policy_target_top1_cap,
        policy_target_min_temperature=args.policy_target_min_temperature,
        policy_target_max_temperature=args.policy_target_max_temperature,
    )
    trainer = HybridTrainer(train_cfg, HybridGNNConfig.small())
    if args.hybrid_checkpoint:
        trainer.load_checkpoint(args.hybrid_checkpoint)
    elif args.init_hybrid_from_fnn:
        trainer.load_fnn_checkpoint(args.fnn_checkpoint)
    return trainer


def _build_prs_trainer(args: argparse.Namespace) -> PRSTrainerV2 | None:
    if not args.train_prs:
        return None
    train_cfg = PRSTrainConfigV2(
        num_iterations=max(args.iterations, 1),
        games_per_batch=args.games,
        mcts_simulations=args.simulations,
        max_num_considered=args.gumbel_considered,
        max_game_length=args.max_game_length,
        batch_size=args.prs_batch_size,
        num_epochs=args.prs_epochs,
        learning_rate=args.prs_lr,
        weight_decay=args.prs_weight_decay,
        buffer_max_size=args.prs_buffer_size,
        checkpoint_dir=args.prs_checkpoint_dir,
        checkpoint_keep_every=args.checkpoint_keep_every,
        draw_keep_rate=args.draw_keep_rate,
        expansion_mask=args.expansion_mask,
        wave_parallel=True,
        model_version="v3",
        augment_prob=args.prs_augment_prob,
    )
    trainer = PRSTrainerV2(train_cfg, PRSConfig())
    if args.prs_checkpoint:
        trainer.load_checkpoint(args.prs_checkpoint)
    return trainer


def convert_fnn_examples_to_prs(
    examples: list[FNNTrainingExample],
    *,
    chunk_size: int = 256,
) -> list[PRSTrainingExampleV2]:
    if not examples:
        return []
    ext = hive_gpu.load_extension()
    encoder = PRSEncoder()
    out: list[PRSTrainingExampleV2] = []

    for start in range(0, len(examples), chunk_size):
        chunk = examples[start:start + chunk_size]
        states_np = np.stack([ex.state_bytes for ex in chunk], axis=0).astype(np.uint8, copy=False)
        states_gpu = torch.from_numpy(states_np).cuda()
        B = len(chunk)

        prs_batch = encoder.encode_batch(states_gpu, B)
        legal_t, nlegal_t = ext.generate_legal_moves_batch(states_gpu, B)
        kernel_out = ext.prs_v2_classify_batch(
            states_gpu, legal_t, nlegal_t, B, int(legal_t.shape[1]),
        )
        slot_of_legal_np = kernel_out[8].cpu().numpy()

        legal_np = legal_t.cpu().numpy()
        nlegal_np = nlegal_t.cpu().numpy().astype(np.int32, copy=False)
        tf_cpu = prs_batch.token_features.cpu().numpy()
        tp_cpu = prs_batch.token_positions.cpu().numpy()
        tt_cpu = prs_batch.token_types.cpu().numpy()
        nb_cpu = prs_batch.num_board_tokens.cpu().numpy()
        gf_cpu = prs_batch.global_features.cpu().numpy()
        sl_cpu = prs_batch.seq_lengths.cpu().numpy()
        occ_cpu = prs_batch.occupied_cells.cpu().numpy()
        nocc_cpu = prs_batch.num_occupied.cpu().numpy()

        for i, ex in enumerate(chunk):
            n_i = int(nlegal_np[i])
            if n_i != int(ex.policy_target.shape[0]):
                raise ValueError(
                    f"legal move count mismatch during PRS bootstrap conversion: "
                    f"generated={n_i} target={int(ex.policy_target.shape[0])}"
                )

            slot_target = np.zeros(N_SLOTS, dtype=np.float32)
            legal_mask = np.zeros(N_SLOTS, dtype=bool)
            for k in range(n_i):
                slot = int(slot_of_legal_np[i, k])
                if slot < 0:
                    continue
                slot_target[slot] += float(ex.policy_target[k])
                legal_mask[slot] = True
            ssum = float(slot_target.sum())
            if ssum > 0:
                slot_target /= ssum
            elif legal_mask.any():
                slot_target[legal_mask] = 1.0 / float(legal_mask.sum())

            art_target, art_mask = compute_articulation_target(ex.state_bytes, MAX_BOARD)
            S = int(sl_cpu[i])
            out.append(PRSTrainingExampleV2(
                token_features=tf_cpu[i, :S].astype(np.float32, copy=True),
                token_positions=tp_cpu[i, :S].astype(np.int64, copy=True),
                token_types=tt_cpu[i, :S].astype(np.int64, copy=True),
                num_board_tokens=int(nb_cpu[i]),
                global_features=gf_cpu[i].astype(np.float32, copy=True),
                seq_length=S,
                state_bytes=ex.state_bytes.copy(),
                legal_moves=legal_np[i, :n_i].astype(np.uint8, copy=True),
                visit_counts=ex.policy_target.astype(np.float32, copy=True),
                nlegal=n_i,
                occupied_cells=occ_cpu[i].astype(np.int32, copy=True),
                num_occupied=int(nocc_cpu[i]),
                slot_target=slot_target,
                legal_mask=legal_mask,
                articulation_target=art_target,
                articulation_mask=art_mask,
                value_target=float(ex.value_target),
                use_for_value=bool(ex.use_for_value),
            ))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap PRS/Hybrid/FNN training from FNN self-play.")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--simulations", type=int, default=1024)
    p.add_argument("--gumbel-considered", type=int, default=16)
    p.add_argument("--max-game-length", type=int, default=300)
    p.add_argument("--expansion-mask", type=int, default=7)
    p.add_argument("--draw-keep-rate", type=float, default=1.0)
    p.add_argument("--queen-surround-reserve-slots", type=int, default=10)
    p.add_argument("--no-immobile-reserve", action="store_true")
    p.add_argument("--endgame-frac", type=float, default=0.0)
    p.add_argument("--endgame-surround", type=int, default=5)
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
    p.add_argument(
        "--policy-target-temperature",
        type=float,
        default=2.0,
        help="Temperature applied only to the post-search FNN replay policy target.",
    )
    p.add_argument(
        "--adaptive-policy-target-temperature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt replay policy target temperature per position to avoid overly sharp targets.",
    )
    p.add_argument("--policy-target-top1-cap", type=float, default=0.7)
    p.add_argument("--policy-target-min-temperature", type=float, default=1.0)
    p.add_argument("--policy-target-max-temperature", type=float, default=7.0)
    p.add_argument("--checkpoint-keep-every", type=int, default=1)

    p.add_argument("--fnn-checkpoint", required=True)
    p.add_argument("--fnn-checkpoint-dir", type=str, default="checkpoints_fnn_bootstrap")
    p.add_argument("--fnn-batch-size", type=int, default=128)
    p.add_argument("--fnn-epochs", type=int, default=3)
    p.add_argument("--fnn-lr", type=float, default=3e-4)
    p.add_argument("--fnn-weight-decay", type=float, default=1e-4)
    p.add_argument("--fnn-buffer-size", type=int, default=100_000)
    p.add_argument("--train-fnn", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--train-hybrid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hybrid-checkpoint", type=str, default=None)
    p.add_argument("--hybrid-checkpoint-dir", type=str, default="checkpoints_hybrid_bootstrap")
    p.add_argument("--hybrid-batch-size", type=int, default=128)
    p.add_argument("--hybrid-epochs", type=int, default=3)
    p.add_argument("--hybrid-lr", type=float, default=3e-4)
    p.add_argument("--hybrid-weight-decay", type=float, default=1e-4)
    p.add_argument("--hybrid-buffer-size", type=int, default=100_000)
    p.add_argument("--init-hybrid-from-fnn", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--train-prs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prs-checkpoint", type=str, default=None)
    p.add_argument("--prs-checkpoint-dir", type=str, default="checkpoints_prs_bootstrap")
    p.add_argument("--prs-batch-size", type=int, default=256)
    p.add_argument("--prs-epochs", type=int, default=3)
    p.add_argument("--prs-lr", type=float, default=5e-4)
    p.add_argument("--prs-weight-decay", type=float, default=1e-4)
    p.add_argument("--prs-buffer-size", type=int, default=150_000)
    p.add_argument("--prs-augment-prob", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.fnn_checkpoint_dir, exist_ok=True)
    if args.train_hybrid:
        os.makedirs(args.hybrid_checkpoint_dir, exist_ok=True)
    if args.train_prs:
        os.makedirs(args.prs_checkpoint_dir, exist_ok=True)

    fnn_trainer = _build_fnn_trainer(args)
    hybrid_trainer = _build_hybrid_trainer(args)
    prs_trainer = _build_prs_trainer(args)

    start_iter = fnn_trainer._start_iter
    if hybrid_trainer is not None:
        start_iter = max(start_iter, hybrid_trainer._start_iter)
    if prs_trainer is not None:
        start_iter = max(start_iter, prs_trainer._start_iter)

    for iteration in range(start_iter, start_iter + args.iterations):
        print(f"\n{'=' * 72}")
        print(f"Bootstrap Iteration {iteration}")
        print(f"{'=' * 72}")

        t0 = time.time()
        fnn_examples, sp_stats = fnn_trainer._self_play(iteration)
        sp_time = time.time() - t0
        print(
            f"FNN self-play: {len(fnn_examples)} examples, "
            f"{sp_stats['num_games']} games "
            f"(W:{sp_stats['white_wins']} B:{sp_stats['black_wins']} D:{sp_stats['draws']}), "
            f"{sp_time:.1f}s"
        )

        if args.train_fnn:
            fnn_trainer.buffer.add_examples(fnn_examples)
            t0 = time.time()
            loss, loss_dict = fnn_trainer._train(iteration)
            print(
                "FNN train: "
                f"loss={loss:.4f}, {time.time() - t0:.1f}s "
                f"[{' '.join(f'{k}={v:.4f}' for k, v in loss_dict.items())}]"
            )
            fnn_trainer._save_checkpoint(iteration)

        if hybrid_trainer is not None:
            hybrid_trainer.buffer.add_examples(fnn_examples)
            t0 = time.time()
            loss, loss_dict = hybrid_trainer._train(iteration)
            print(
                "Hybrid train: "
                f"loss={loss:.4f}, {time.time() - t0:.1f}s "
                f"[{' '.join(f'{k}={v:.4f}' for k, v in loss_dict.items())}]"
            )
            hybrid_trainer._save_checkpoint(iteration)

        if prs_trainer is not None:
            t0 = time.time()
            prs_examples = convert_fnn_examples_to_prs(fnn_examples)
            conv_time = time.time() - t0
            prs_trainer.buffer.add_examples(prs_examples)
            t0 = time.time()
            loss, loss_dict = prs_trainer._train(iteration)
            print(
                "PRS train: "
                f"{len(prs_examples)} converted examples in {conv_time:.1f}s, "
                f"loss={loss:.4f}, {time.time() - t0:.1f}s "
                f"[{' '.join(f'{k}={v:.4f}' for k, v in loss_dict.items())}]"
            )
            prs_trainer._save_checkpoint(iteration)


if __name__ == "__main__":
    main()
