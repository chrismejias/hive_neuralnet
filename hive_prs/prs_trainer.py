"""
PRS Transformer training loop.

Usage:
    from hive_prs.prs_trainer import PRSTrainer, PRSTrainConfig
    from hive_prs.prs_transformer import HivePRSTransformer, PRSConfig

    trainer = PRSTrainer(PRSTrainConfig(), PRSConfig.small())
    trainer.run()
"""

from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from hive_engine.device import get_device
from hive_engine.elo import EloTracker

from hive_prs.prs_transformer import HivePRSTransformer, PRSConfig
from hive_prs.prs_replay_buffer import PRSReplayBuffer, PRSTrainingBatch
from hive_prs.prs_orchestrator import PRSGumbelOrchestrator, PRSGumbelConfig
from hive_prs.action_space import ACTION_SPACE_SIZE


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class PRSTrainConfig:
    # Self-play
    num_iterations:            int   = 1500
    games_per_batch:           int   = 128
    mcts_simulations:          int   = 512
    max_num_considered:        int   = 16
    temperature:               float = 1.0
    temperature_drop_move:     int   = 20
    max_game_length:           int   = 300

    # Training
    batch_size:                int   = 256
    num_epochs:                int   = 3
    learning_rate:             float = 5e-4
    weight_decay:              float = 1e-4
    max_grad_norm:             float = 1.0
    policy_softening:          float = 0.0  # disabled for PRS (action space is already small)

    # LR schedule
    lr_schedule:               str   = "cosine"   # "cosine" or "constant"
    lr_warmup_iterations:      int   = 3
    lr_min:                    float = 1e-5

    # Replay buffer
    buffer_max_size:           int   = 100_000

    # Checkpointing
    checkpoint_dir:            str   = "checkpoints_prs"
    checkpoint_keep_every:     int   = 0

    # Arena (skip by default — PRS vs old model comparison done separately)
    skip_arena:                bool  = True
    arena_games:               int   = 20

    # Misc
    draw_keep_rate:            float = 1.0
    expansion_mask:            int   = 0
    nn_max_batch:              int   = 0

    device:                    str | None = None
    use_amp:                   bool | None = None
    compile_net:               bool = True


# ── Loss ───────────────────────────────────────────────────────────────────────


def compute_prs_loss(
    policy_logits: torch.Tensor,    # (B, ACTION_SPACE_SIZE)
    value_pred:    torch.Tensor,    # (B, 1)
    policy_targets: torch.Tensor,  # (B, ACTION_SPACE_SIZE) float32
    value_targets:  torch.Tensor,  # (B, 1) float32
    policy_softening: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Policy CE + value MSE loss for PRS model."""
    B = policy_logits.size(0)

    # Legal mask: any position with non-zero target is legal
    legal_mask = policy_targets > 0   # (B, A)

    # Optional: soften policy targets slightly toward uniform over legal moves
    if policy_softening > 0:
        n_legal = legal_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        uniform = legal_mask.float() / n_legal
        policy_targets = (1 - policy_softening) * policy_targets + policy_softening * uniform

    # Mask illegal action logits to -inf
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

    # Cross-entropy loss (log_softmax of masked logits vs target distribution)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    # Illegal positions have log_probs = -inf; multiply by policy_targets = 0 → nan.
    # Rows where ALL logits are -inf (no legal actions) produce nan log_softmax too.
    # Use nan_to_num to convert both nan and -inf to a large negative finite value.
    log_probs_safe = torch.nan_to_num(log_probs, nan=-1000.0, neginf=-1000.0)
    per_sample = -(policy_targets * log_probs_safe).sum(dim=-1)  # (B,)
    # Skip rows with no policy target (no legal actions stored)
    has_policy = legal_mask.any(dim=-1)  # (B,)
    if has_policy.any():
        policy_loss = per_sample[has_policy].mean()
    else:
        policy_loss = per_sample.mean()

    # Value MSE
    value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets.squeeze(-1))

    total_loss = policy_loss + value_loss

    return total_loss, {
        "policy_loss": policy_loss,
        "value_loss":  value_loss,
    }


# ── Trainer ────────────────────────────────────────────────────────────────────


class PRSTrainer:
    def __init__(
        self,
        config: PRSTrainConfig | None = None,
        net_config: PRSConfig | None  = None,
    ) -> None:
        self.config     = config or PRSTrainConfig()
        self.net_config = net_config or PRSConfig.small()
        self.device     = get_device(self.config.device)

        cfg = self.config
        self.use_amp = cfg.use_amp if cfg.use_amp is not None else (self.device.type == "cuda")
        self._scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Self-play model: always uncompiled — Gumbel search calls the NN with
        # varying batch sizes (B, B×K, B×K/2, …) which cause shape recompilation
        # storms if torch.compile is active here.
        self.best_net = HivePRSTransformer(self.net_config).to(self.device)

        # Training model: compiled once, then reused across iterations by loading
        # weights from best_net at the start of each _train call.
        # Fixed batch size during training → stable shapes → JIT cache always hits.
        self._train_net: nn.Module = HivePRSTransformer(self.net_config).to(self.device)
        if cfg.compile_net and self.device.type == "cuda":
            self._train_net = torch.compile(self._train_net)

        self.buffer      = PRSReplayBuffer(cfg.buffer_max_size)
        self.elo_tracker = EloTracker()
        self._start_iter = 1

    def _lr(self, iteration: int) -> float:
        cfg = self.config
        base = cfg.learning_rate
        if cfg.lr_warmup_iterations > 0 and iteration <= cfg.lr_warmup_iterations:
            return base * iteration / cfg.lr_warmup_iterations
        if cfg.lr_schedule == "constant":
            return base
        warmup   = cfg.lr_warmup_iterations
        total    = cfg.num_iterations - warmup
        progress = (iteration - warmup) / max(total, 1)
        return cfg.lr_min + 0.5 * (base - cfg.lr_min) * (1 + math.cos(math.pi * progress))

    def _cleanup(self) -> None:
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def run(self) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for iteration in range(self._start_iter, self.config.num_iterations + 1):
            self._cleanup()
            print(f"\n{'='*60}")
            print(f"PRS Iteration {iteration}/{self.config.num_iterations}")
            print(f"{'='*60}")

            # ── Self-play ──
            t0 = time.time()
            new_examples, sp_stats = self._self_play(iteration)
            self.buffer.add_examples(new_examples)
            sp_time = time.time() - t0
            print(
                f"  Self-play: {len(new_examples)} examples, "
                f"{sp_stats['num_games']} games "
                f"(W:{sp_stats['white_wins']} B:{sp_stats['black_wins']} "
                f"D:{sp_stats['draws']}), {sp_time:.1f}s"
            )
            self._cleanup()

            # ── Train ──
            t0 = time.time()
            loss, loss_dict = self._train(iteration)
            train_time = time.time() - t0
            comp = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            print(f"  Training: loss={loss:.4f}, {train_time:.1f}s  [{comp}]")
            self._cleanup()

            # ── Promote ──
            # best_net is updated in-place by _train; no reassignment needed.
            elo = self.elo_tracker.update(1.0, self.config.arena_games)
            print(f"  ELO: {elo:.0f}")

            # ── Checkpoint ──
            self._save_checkpoint(iteration)

            total = sp_time + train_time
            print(
                f"  Summary: iter={iteration} loss={loss:.4f} "
                f"elo={elo:.0f} buf={len(self.buffer)} time={total:.0f}s"
            )

    # ── Self-play ───────────────────────────────────────────────────────

    def _self_play(self, iteration: int) -> tuple[list, dict]:
        cfg = self.config
        self.best_net.eval()

        gumbel_cfg = PRSGumbelConfig(
            num_simulations            = cfg.mcts_simulations,
            max_num_considered_actions = cfg.max_num_considered,
            temperature                = cfg.temperature,
            temperature_drop_move      = cfg.temperature_drop_move,
            batch_size                 = cfg.games_per_batch,
            max_game_length            = cfg.max_game_length,
            expansion_mask             = cfg.expansion_mask,
            nn_max_batch               = cfg.nn_max_batch,
        )
        orchestrator = PRSGumbelOrchestrator(self.best_net, gumbel_cfg)
        raw_examples = orchestrator.self_play_batch()

        stats = {"num_games": 0, "white_wins": 0, "black_wins": 0, "draws": 0}
        flat: list = []

        for game_exs in raw_examples:
            if not game_exs:
                stats["draws"] += 1
                stats["num_games"] += 1
                continue

            v = game_exs[0].value_target
            if v > 0:
                stats["white_wins"] += 1
            elif v < 0:
                stats["black_wins"] += 1
            else:
                stats["draws"] += 1
            stats["num_games"] += 1

            # Draw downsampling
            if v == 0.0 and cfg.draw_keep_rate < 1.0:
                if np.random.random() > cfg.draw_keep_rate:
                    continue

            flat.extend(game_exs)

        return flat, stats

    # ── Training step ────────────────────────────────────────────────────

    def _train(self, iteration: int) -> tuple[float, dict]:
        """Train _train_net (compiled) from best_net weights, then sync back.

        Keeps best_net uncompiled (for self-play) while using a persistent compiled
        _train_net for gradient steps (fixed batch size → stable shapes → JIT cache
        always hits after the first compilation on iteration 1).
        """
        cfg = self.config

        if len(self.buffer) < cfg.batch_size:
            return 0.0, {}

        # ── Sync weights: best_net → _train_net ──
        # getattr handles both compiled (OptimizedModule._orig_mod) and plain modules.
        underlying = getattr(self._train_net, "_orig_mod", self._train_net)
        underlying.load_state_dict(self.best_net.state_dict())
        self._train_net.train()

        lr  = self._lr(iteration)
        opt = optim.Adam(self._train_net.parameters(), lr=lr, weight_decay=cfg.weight_decay)

        total_loss_sum = 0.0
        comp_sums: dict[str, float] = {}
        n_batches = 0

        for epoch in range(cfg.num_epochs):
            batches_per_epoch = max(1, len(self.buffer) // cfg.batch_size)
            for _ in range(batches_per_epoch):
                batch: PRSTrainingBatch = self.buffer.sample_batch(cfg.batch_size)
                batch = batch.to(self.device, non_blocking=True)

                opt.zero_grad()

                if self.use_amp and self._scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits, value = self._train_net(batch.prs_batch)
                        loss, ld = compute_prs_loss(
                            logits, value, batch.policy_targets, batch.value_targets,
                            cfg.policy_softening,
                        )
                    self._scaler.scale(loss).backward()
                    if cfg.max_grad_norm > 0:
                        self._scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            self._train_net.parameters(), cfg.max_grad_norm
                        )
                    self._scaler.step(opt)
                    self._scaler.update()
                else:
                    logits, value = self._train_net(batch.prs_batch)
                    loss, ld = compute_prs_loss(
                        logits, value, batch.policy_targets, batch.value_targets,
                        cfg.policy_softening,
                    )
                    loss.backward()
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._train_net.parameters(), cfg.max_grad_norm
                        )
                    opt.step()

                total_loss_sum += loss.item()
                for k, v in ld.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + v.item()
                n_batches += 1

        # ── Sync weights back: _train_net → best_net ──
        self.best_net.load_state_dict(underlying.state_dict())
        self.best_net.eval()

        n = max(n_batches, 1)
        return total_loss_sum / n, {k: v / n for k, v in comp_sums.items()}

    # ── Checkpointing ────────────────────────────────────────────────────

    def _save_checkpoint(self, iteration: int) -> None:
        cfg = self.config
        keep = cfg.checkpoint_keep_every
        if keep > 0 and iteration % keep != 0:
            return
        path = os.path.join(cfg.checkpoint_dir, f"prs_iter_{iteration:04d}.pt")
        torch.save({
            "iteration":   iteration,
            "model_state": self.best_net.state_dict(),  # always save uncompiled weights
            "net_config":  self.net_config,
        }, path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.best_net.load_state_dict(ckpt["model_state"])
        self._start_iter = ckpt["iteration"] + 1
        print(f"Resumed from {path} (iter {ckpt['iteration']})")
