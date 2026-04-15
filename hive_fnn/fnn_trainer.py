"""Training loop for the HiveGo-style FNN."""

from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from hive_engine.device import get_device
from hive_engine.elo import EloTracker
from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_fnn.fnn_orchestrator import (
    FNNCudaOrchestrator, FNNGumbelConfig, _flat_to_padded,
)
from hive_fnn.fnn_replay_buffer import FNNReplayBuffer, FNNTrainingBatch

import hive_gpu


@dataclass
class FNNTrainConfig:
    num_iterations: int = 1500
    games_per_batch: int = 128
    mcts_simulations: int = 128
    max_num_considered: int = 16
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
    checkpoint_dir: str = "checkpoints_fnn"
    checkpoint_keep_every: int = 0

    draw_keep_rate: float = 1.0
    expansion_mask: int = 0
    device: str | None = None
    use_amp: bool | None = None


def _policy_cross_entropy(
    flat_logits: torch.Tensor,
    policy_targets: torch.Tensor,
    num_actions: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy policy loss over ragged legal moves."""
    padded_logits = _flat_to_padded(flat_logits, num_actions, pad_value=float("-inf"))
    max_actions = padded_logits.shape[1]
    action_idx = torch.arange(max_actions, device=num_actions.device).unsqueeze(0)
    legal_mask = action_idx < num_actions.unsqueeze(1)

    masked_logits = padded_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1)
    log_probs = torch.nan_to_num(log_probs, nan=-1000.0, neginf=-1000.0)
    return -(policy_targets[:, :max_actions] * log_probs).sum(dim=1).mean()


def compute_fnn_loss(
    action_logits: torch.Tensor,
    root_values: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    num_actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined loss: policy cross-entropy + value MSE."""
    policy_loss = _policy_cross_entropy(action_logits, policy_targets, num_actions)
    value_loss = F.mse_loss(root_values.squeeze(-1), value_targets.squeeze(-1))

    total = policy_loss + value_loss
    return total, {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }


class FNNTrainer:
    def __init__(
        self,
        config: FNNTrainConfig | None = None,
        net_config: FNNConfig | None = None,
    ) -> None:
        self.config = config or FNNTrainConfig()
        self.net_config = net_config or FNNConfig.medium()
        self.device = get_device(self.config.device)
        self.ext = hive_gpu.load_extension()
        self.best_net = HiveFNN(self.net_config).to(self.device)
        self.buffer = FNNReplayBuffer(self.config.buffer_max_size)
        self.elo_tracker = EloTracker()
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
        for iteration in range(1, self.config.num_iterations + 1):
            self._cleanup()
            print(f"\n{'=' * 60}")
            print(f"FNN Iteration {iteration}/{self.config.num_iterations}")
            print(f"{'=' * 60}")

            t0 = time.time()
            new_examples, stats = self._self_play()
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
            print(
                f"  Training: loss={loss:.4f}, {time.time() - t0:.1f}s [{parts}]"
            )
            self._save_checkpoint(iteration)
            print(f"  ELO: {elo:.0f}  Buffer: {len(self.buffer)}")

    def _self_play(self) -> tuple[list, dict]:
        cfg = self.config
        self.best_net.eval()
        orch = FNNCudaOrchestrator(
            self.best_net,
            FNNGumbelConfig(
                num_simulations=cfg.mcts_simulations,
                max_num_considered_actions=cfg.max_num_considered,
                temperature=cfg.temperature,
                temperature_drop_move=cfg.temperature_drop_move,
                batch_size=cfg.games_per_batch,
                max_game_length=cfg.max_game_length,
                expansion_mask=cfg.expansion_mask,
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
            v = game[0].value_target
            stats["num_games"] += 1
            if v > 0:
                stats["white_wins"] += 1
            elif v < 0:
                stats["black_wins"] += 1
            else:
                stats["draws"] += 1
                if np.random.random() > cfg.draw_keep_rate:
                    continue
            flat.extend(game)
        return flat, stats

    def _build_forward_batch(
        self, state_bytes: torch.Tensor, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode root + all successors for training.

        Returns:
            root_features: (B, feat_dim)
            successor_features: (N_total, feat_dim)
            action_to_root: (N_total,) int64
            num_actions: (B,) int64
        """
        legal_moves, num_legal = self.ext.generate_legal_moves_batch(
            state_bytes, batch_size,
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

        total_actions = action_to_root.shape[0]

        # Root features via CUDA kernel
        root_features = self.ext.extract_fnn_features_batch(
            state_bytes, legal_moves, num_legal, batch_size,
        )

        if total_actions == 0:
            return root_features, root_features[:0], action_to_root, num_actions

        # Build child states and extract features
        child_states = state_bytes[action_to_root].clone()
        moves = legal_moves[action_to_root, move_indices]
        self.ext.apply_moves_batch(child_states, moves, total_actions)

        child_legal, child_nlegal = self.ext.generate_legal_moves_batch(
            child_states, total_actions,
        )
        succ_features = self.ext.extract_fnn_features_batch(
            child_states, child_legal, child_nlegal, total_actions,
        )

        return root_features, succ_features, action_to_root, num_actions

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

        for _epoch in range(cfg.num_epochs):
            batches_per_epoch = max(1, len(self.buffer) // cfg.batch_size)
            for _ in range(batches_per_epoch):
                batch: FNNTrainingBatch = self.buffer.sample_batch(
                    cfg.batch_size,
                ).to(self.device, non_blocking=True)

                opt.zero_grad()
                root_feat, succ_feat, a2r, n_act = self._build_forward_batch(
                    batch.state_bytes, batch.state_bytes.shape[0],
                )

                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        action_logits, root_values = self.best_net(
                            root_feat, succ_feat, a2r, n_act,
                        )
                        loss, ld = compute_fnn_loss(
                            action_logits,
                            root_values,
                            batch.policy_targets,
                            batch.value_targets,
                            batch.num_actions,
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
                        root_feat, succ_feat, a2r, n_act,
                    )
                    loss, ld = compute_fnn_loss(
                        action_logits,
                        root_values,
                        batch.policy_targets,
                        batch.value_targets,
                        batch.num_actions,
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
            f"hive_fnn_checkpoint_{iteration:04d}.pt",
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
