"""Compatibility wrapper for the archived move-conditioned trainer."""

<<<<<<<< HEAD:hive_mc/mc_trainer.py
from archive.legacy_mc.hive_mc.mc_trainer import *
========
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
from hive_gpu.gpu_encoder import GPUTransformerEncoder
from archive.legacy_mc.hive_mc.mc_mcts_orchestrator import MCMCTSConfig, MCMCTSOrchestrator
from archive.legacy_mc.hive_mc.mc_replay_buffer import MCReplayBuffer, MCTrainingBatch
from archive.legacy_mc.hive_mc.mc_transformer import HiveMoveTransformer, MCTransformerConfig
from archive.legacy_mc.hive_mc.mc_utils import build_move_conditioned_batch, flat_to_padded


@dataclass
class MCTrainConfig:
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
    checkpoint_dir: str = "checkpoints_mc"
    checkpoint_keep_every: int = 0

    draw_keep_rate: float = 1.0
    expansion_mask: int = 0
    nn_max_batch: int = 0
    device: str | None = None
    use_amp: bool | None = None


def _policy_cross_entropy(
    flat_logits: torch.Tensor,
    policy_targets: torch.Tensor,
    num_actions: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy policy loss over ragged legal moves."""
    padded_logits = flat_to_padded(flat_logits, num_actions, pad_value=float("-inf"))
    max_actions = padded_logits.shape[1]
    action_idx = torch.arange(max_actions, device=num_actions.device).unsqueeze(0)
    legal_mask = action_idx < num_actions.unsqueeze(1)

    masked_logits = padded_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1)
    log_probs = torch.nan_to_num(log_probs, nan=-1000.0, neginf=-1000.0)
    return -(policy_targets[:, :max_actions] * log_probs).sum(dim=1).mean()


def compute_mc_loss(
    screening_logits: torch.Tensor,
    action_logits: torch.Tensor,
    root_values: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    num_actions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined loss: screening policy + action policy + value MSE."""
    screening_loss = _policy_cross_entropy(screening_logits, policy_targets, num_actions)
    action_loss = _policy_cross_entropy(action_logits, policy_targets, num_actions)
    value_loss = F.mse_loss(root_values.squeeze(-1), value_targets.squeeze(-1))

    total = screening_loss + action_loss + value_loss
    return total, {
        "screening_loss": screening_loss,
        "action_loss": action_loss,
        "value_loss": value_loss,
    }


class MCTrainer:
    def __init__(
        self,
        config: MCTrainConfig | None = None,
        net_config: MCTransformerConfig | None = None,
    ) -> None:
        self.config = config or MCTrainConfig()
        self.net_config = net_config or MCTransformerConfig.small()
        self.device = get_device(self.config.device)
        self.encoder = GPUTransformerEncoder()
        self.best_net = HiveMoveTransformer(self.net_config).to(self.device)
        self.buffer = MCReplayBuffer(self.config.buffer_max_size)
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
            print(f"\n{'='*60}")
            print(f"Move-Conditioned Iteration {iteration}/{self.config.num_iterations}")
            print(f"{'='*60}")

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
            parts = " ".join(
                f"{k}={v:.4f}" for k, v in loss_dict.items()
            )
            print(
                f"  Training: loss={loss:.4f}, {time.time() - t0:.1f}s [{parts}]"
            )
            self._save_checkpoint(iteration)
            print(f"  ELO: {elo:.0f}  Buffer: {len(self.buffer)}")

    def _self_play(self) -> tuple[list, dict]:
        cfg = self.config
        self.best_net.eval()
        orch = MCMCTSOrchestrator(
            self.best_net,
            MCMCTSConfig(
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
                batch: MCTrainingBatch = self.buffer.sample_batch(cfg.batch_size).to(self.device, non_blocking=True)
                opt.zero_grad()
                move_batch = build_move_conditioned_batch(
                    batch.state_bytes,
                    batch.state_bytes.shape[0],
                    encoder=self.encoder,
                    ext=None,
                )
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        screening_logits, action_logits, root_values, _av = self.best_net(move_batch)
                        loss, ld = compute_mc_loss(
                            screening_logits,
                            action_logits,
                            root_values,
                            batch.policy_targets,
                            batch.value_targets,
                            batch.num_actions,
                        )
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.best_net.parameters(), cfg.max_grad_norm)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    screening_logits, action_logits, root_values, _av = self.best_net(move_batch)
                    loss, ld = compute_mc_loss(
                        screening_logits,
                        action_logits,
                        root_values,
                        batch.policy_targets,
                        batch.value_targets,
                        batch.num_actions,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.best_net.parameters(), cfg.max_grad_norm)
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
        path = os.path.join(self.config.checkpoint_dir, f"hive_mc_checkpoint_{iteration:04d}.pt")
        torch.save(
            {
                "model_state_dict": self.best_net.state_dict(),
                "net_config": self.net_config,
                "train_config": self.config,
                "iteration": iteration,
            },
            path,
        )
>>>>>>>> 7c7d146 (Refactor legacy transformer and MC packages):archive/legacy_mc/hive_mc/mc_trainer.py
