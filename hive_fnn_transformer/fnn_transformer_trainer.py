"""Training loop for the hybrid FNN-policy + transformer-value model."""

from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

import hive_gpu
from hive_engine.device import get_device
from hive_engine.elo import EloTracker
from hive_fnn.fnn_replay_buffer import FNNReplayBuffer, FNNTrainingBatch
from hive_fnn.fnn_trainer import compute_fnn_loss
from hive_fnn_transformer.gpu_encoder import HybridTransformerGPUEncoder
from hive_fnn_transformer.fnn_transformer_net import HybridGNNConfig, HiveHybridGNN
from hive_fnn_transformer.fnn_transformer_mcts_orchestrator import (
    HybridMCTSConfig,
    HybridMCTSOrchestrator,
)


@dataclass
class HybridTrainConfig:
    num_iterations: int = 1500
    games_per_batch: int = 128
    mcts_simulations: int = 128
    simulation_schedule: tuple[int, ...] = ()
    max_num_considered: int = 16
    queen_surround_reserve_slots: int = 10
    queen_surround_reserve_immobile_only: bool = True
    temperature: float = 1.0
    temperature_drop_move: int = 20
    max_game_length: int = 300

    batch_size: int = 128
    num_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_schedule: str = "cosine"
    lr_warmup_iterations: int = 3
    lr_min: float = 1e-5

    buffer_max_size: int = 100_000
    checkpoint_dir: str = "checkpoints_fnn_transformer"
    checkpoint_keep_every: int = 0

    draw_keep_rate: float = 1.0
    expansion_mask: int = 0
    graph_radius: int = 2
    device: str | None = None
    use_amp: bool | None = None
    gumbel_wave_parallel: bool = True
    short_forced_win_probe: bool = False
    probe_win_in_one: bool = True
    probe_check_opponent_wins: bool = True
    probe_win_in_two: bool = True


def _simulations_for_iteration(
    base_simulations: int,
    schedule: tuple[int, ...],
    iteration: int,
) -> int:
    if not schedule:
        return base_simulations
    return int(schedule[(iteration - 1) % len(schedule)])


class HybridTrainer:
    def __init__(
        self,
        config: HybridTrainConfig | None = None,
        net_config: HybridGNNConfig | None = None,
    ) -> None:
        self.config = config or HybridTrainConfig()
        self.net_config = net_config or HybridGNNConfig.large()
        self.device = get_device(self.config.device)
        self.ext = hive_gpu.load_extension()
        self.best_net = HiveHybridGNN(self.net_config).to(self.device)
        self.graph_encoder = HybridTransformerGPUEncoder()
        self.buffer = FNNReplayBuffer(
            self.config.buffer_max_size,
            device=self.device,
            cache_root_features=self.device.type == "cuda",
            gpu_sampling=self.device.type == "cuda",
        )
        self.elo_tracker = EloTracker()
        self._start_iter = 1
        self.use_amp = (
            self.config.use_amp
            if self.config.use_amp is not None
            else self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def _cleanup(self) -> None:
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _lr(self, iteration: int) -> float:
        cfg = self.config
        base = cfg.learning_rate
        if cfg.lr_warmup_iterations > 0 and iteration <= cfg.lr_warmup_iterations:
            return base * iteration / cfg.lr_warmup_iterations
        if cfg.lr_schedule == "constant":
            return base
        warmup = cfg.lr_warmup_iterations
        total = cfg.num_iterations - warmup
        progress = (iteration - warmup) / max(total, 1)
        return cfg.lr_min + 0.5 * (base - cfg.lr_min) * (1 + math.cos(math.pi * progress))

    def run(self) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        for iteration in range(self._start_iter, self.config.num_iterations + 1):
            self._cleanup()
            sims_this_iter = _simulations_for_iteration(
                self.config.mcts_simulations,
                self.config.simulation_schedule,
                iteration,
            )
            print(f"\n{'=' * 60}")
            print(f"FNN Transformer Iteration {iteration}/{self.config.num_iterations}")
            print(f"  Simulations: {sims_this_iter}")
            print(f"{'=' * 60}")

            t0 = time.time()
            new_examples, stats = self._self_play(iteration)
            self.buffer.add_examples(new_examples)
            print(
                f"  Self-play: {len(new_examples)} examples, "
                f"{stats['num_games']} games "
                f"(W:{stats['white_wins']} B:{stats['black_wins']} D:{stats['draws']}), "
                f"{time.time() - t0:.1f}s"
            )

            t0 = time.time()
            loss, loss_dict = self._train(iteration)
            elo = self.elo_tracker.update(1.0, 0)
            parts = " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            print(f"  Training: loss={loss:.4f}, {time.time() - t0:.1f}s [{parts}]")
            self._save_checkpoint(iteration)
            print(f"  ELO: {elo:.0f}  Buffer: {len(self.buffer)}")

    def _self_play(self, iteration: int) -> tuple[list, dict]:
        cfg = self.config
        sims_this_iter = _simulations_for_iteration(
            cfg.mcts_simulations, cfg.simulation_schedule, iteration,
        )
        self.best_net.eval()
        orch = HybridMCTSOrchestrator(
            self.best_net,
            HybridMCTSConfig(
                num_simulations=sims_this_iter,
                max_num_considered_actions=cfg.max_num_considered,
                queen_surround_reserve_slots=cfg.queen_surround_reserve_slots,
                queen_surround_reserve_immobile_only=cfg.queen_surround_reserve_immobile_only,
                temperature=cfg.temperature,
                temperature_drop_move=cfg.temperature_drop_move,
                batch_size=cfg.games_per_batch,
                max_game_length=cfg.max_game_length,
                expansion_mask=cfg.expansion_mask,
                wave_parallel=cfg.gumbel_wave_parallel,
                graph_radius=cfg.graph_radius,
                short_forced_win_probe=cfg.short_forced_win_probe,
                probe_win_in_one=cfg.probe_win_in_one,
                probe_check_opponent_wins=cfg.probe_check_opponent_wins,
                probe_win_in_two=cfg.probe_win_in_two,
            ),
        )
        raw = orch.self_play_batch()
        flat = []
        stats = {"num_games": 0, "white_wins": 0, "black_wins": 0, "draws": 0}
        for game in raw:
            if not game:
                stats["num_games"] += 1
                stats["draws"] += 1
                continue
            value = game[0].value_target
            stats["num_games"] += 1
            if value > 0:
                stats["white_wins"] += 1
            elif value < 0:
                stats["black_wins"] += 1
            else:
                stats["draws"] += 1
                is_capped_draw = not bool(game[0].use_for_value)
                if is_capped_draw and np.random.random() > cfg.draw_keep_rate:
                    continue
            flat.extend(game)
        return flat, stats

    def _build_forward_batch(
        self,
        state_bytes: torch.Tensor,
        batch_size: int,
        num_actions: torch.Tensor | None = None,
        cached_root_features: torch.Tensor | None = None,
        cached_legal_moves: torch.Tensor | None = None,
    ):
        if cached_root_features is not None and cached_legal_moves is not None:
            if num_actions is None:
                raise ValueError("Cached legal moves require cached action counts")
            legal_moves = cached_legal_moves
            root_features = cached_root_features.float()
            num_legal = num_actions.to(torch.int64)
        else:
            legal_moves, num_legal, root_features = (
                self.ext.generate_legal_moves_and_fnn_features_batch(
                    state_bytes, batch_size,
                )
            )
        graph_batch = self.graph_encoder.encode_batch(
            state_bytes,
            batch_size,
            legal_moves=legal_moves,
            num_legal=num_legal,
        )
        move_features_per_legal = self.ext.hybrid_transformer_move_features_batch(
            state_bytes,
            legal_moves,
            num_legal,
            batch_size,
        )
        num_actions = num_legal.to(torch.int64)

        max_legal = legal_moves.shape[1]
        device = state_bytes.device
        slot_idx = torch.arange(max_legal, device=device, dtype=torch.int64).unsqueeze(0)
        valid = slot_idx < num_actions.unsqueeze(1)
        action_to_root = torch.arange(
            batch_size, device=device, dtype=torch.int64,
        ).unsqueeze(1).expand_as(valid)[valid]
        move_indices = slot_idx.expand_as(valid)[valid]
        total_actions = int(action_to_root.shape[0])

        if total_actions == 0:
            return (
                root_features,
                root_features[:0],
                action_to_root,
                num_actions,
                graph_batch,
                move_features_per_legal.new_zeros((0, move_features_per_legal.shape[-1])),
            )

        succ_features = self.ext.fnn_successor_features_batch(
            state_bytes,
            legal_moves,
            action_to_root,
            move_indices,
            total_actions,
        )
        move_features = move_features_per_legal[valid]
        return root_features, succ_features, action_to_root, num_actions, graph_batch, move_features

    def _train(self, iteration: int) -> tuple[float, dict[str, float]]:
        cfg = self.config
        if len(self.buffer) < cfg.batch_size:
            return 0.0, {}

        opt = optim.Adam(
            self.best_net.parameters(),
            lr=self._lr(iteration),
            weight_decay=cfg.weight_decay,
        )
        total_loss = 0.0
        comp_sums: dict[str, float] = {}
        n_batches = 0
        self.best_net.train()
        prefetch_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

        def fetch_next_batch() -> FNNTrainingBatch:
            if prefetch_stream is None:
                return self.buffer.sample_batch(cfg.batch_size, device=self.device)
            with torch.cuda.stream(prefetch_stream):
                return self.buffer.sample_batch(
                    cfg.batch_size, device=self.device, non_blocking=True,
                )

        for _epoch in range(cfg.num_epochs):
            batches_per_epoch = max(1, len(self.buffer) // cfg.batch_size)
            next_batch = fetch_next_batch()
            for _ in range(batches_per_epoch):
                if prefetch_stream is not None:
                    torch.cuda.current_stream(self.device).wait_stream(prefetch_stream)
                batch = next_batch
                if n_batches + 1 < batches_per_epoch * cfg.num_epochs:
                    next_batch = fetch_next_batch()

                opt.zero_grad(set_to_none=True)
                root_feat, succ_feat, a2r, n_act, graph_batch, move_features = self._build_forward_batch(
                    batch.state_bytes,
                    batch.state_bytes.shape[0],
                    batch.num_actions,
                    batch.root_features,
                    batch.legal_moves,
                )

                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        action_logits, root_values = self.best_net(
                            root_feat, succ_feat, a2r, n_act, graph_batch, move_features,
                        )
                        loss, ld = compute_fnn_loss(
                            action_logits,
                            root_values,
                            batch.policy_targets,
                            batch.value_targets,
                            batch.num_actions,
                            batch.value_mask,
                        )
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.best_net.parameters(), cfg.max_grad_norm,
                    )
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    action_logits, root_values = self.best_net(
                        root_feat, succ_feat, a2r, n_act, graph_batch, move_features,
                    )
                    loss, ld = compute_fnn_loss(
                        action_logits,
                        root_values,
                        batch.policy_targets,
                        batch.value_targets,
                        batch.num_actions,
                        batch.value_mask,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.best_net.parameters(), cfg.max_grad_norm,
                    )
                    opt.step()

                total_loss += float(loss.item())
                n_batches += 1
                for k, v in ld.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())

        if n_batches == 0:
            return 0.0, {}
        return total_loss / n_batches, {k: v / n_batches for k, v in comp_sums.items()}

    def _save_checkpoint(self, iteration: int) -> None:
        keep = self.config.checkpoint_keep_every
        if keep > 0 and iteration % keep != 0:
            return
        path = os.path.join(
            self.config.checkpoint_dir,
            f"hybrid_gnn_checkpoint_{iteration:04d}.pt",
        )
        torch.save(
            {
                "model_state_dict": self.best_net.state_dict(),
                "net_config": self.net_config,
                "train_config": self.config,
                "iteration": iteration,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.best_net.load_state_dict(ckpt["model_state_dict"])
        self._start_iter = int(ckpt["iteration"]) + 1
        print(f"Resumed from {path} (iter {ckpt['iteration']})")

    def load_fnn_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt["model_state_dict"]
        self.best_net.fnn.load_state_dict(state)
        print(f"Loaded FNN policy/value submodule weights from {path}")
