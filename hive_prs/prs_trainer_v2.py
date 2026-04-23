"""PRS v2 training loop (structured 813-slot head).

Differences vs v1 trainer:
  * Uses HivePRSTransformerV2 + PRSReplayBufferV2 + PRSMCTSOrchestratorV2.
  * Loss is masked cross-entropy over 813 slots (legal mask stored with each
    example) + value MSE. No surprise-weighted sampling, no policy softening
    (the slot space is small and already legal-masked).
  * The trunk runs on GPU; `build_head_inputs_from_states` runs CPU-side to
    produce piece/cell indices, then the head runs on GPU. `torch.compile`
    is therefore disabled on the training net (CPU-side bridge would force
    recompilation per batch).
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

from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.prs_replay_buffer_v2 import PRSReplayBufferV2, PRSTrainingBatchV2
from hive_prs.prs_mcts_orchestrator_v2 import (
    PRSMCTSConfigV2, PRSMCTSOrchestratorV2,
)
from hive_prs.prs_v2_bridge import (
    build_head_inputs_from_states,
    build_head_inputs_from_kernel,
)
import hive_gpu
from hive_prs.slot_map import N_SLOTS


# ── Config ─────────────────────────────────────────────────────────────

@dataclass
class PRSTrainConfigV2:
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

    # LR schedule
    lr_schedule:               str   = "cosine"
    lr_warmup_iterations:      int   = 3
    lr_min:                    float = 1e-5

    # Replay buffer
    buffer_max_size:           int   = 100_000

    # Checkpointing
    checkpoint_dir:            str   = "checkpoints_prs_v2"
    checkpoint_keep_every:     int   = 0

    # Arena
    skip_arena:                bool  = True
    arena_games:               int   = 20

    # Misc
    draw_keep_rate:            float = 1.0
    expansion_mask:             int   = 7
    nn_max_batch:              int   = 0
    wave_parallel:             bool  = True

    # C6 (6-fold) rotational augmentation for training batches.
    # With this probability, a random rotation k∈{1..5} is applied to every
    # sample in the batch (state + legal moves), tokens are re-encoded,
    # and slot_target/legal_mask are rebuilt via SlotMapper.  0.0 = off.
    augment_prob:              float = 0.5

    device:                    str | None = None
    use_amp:                   bool | None = None


# ── Loss ───────────────────────────────────────────────────────────────

def compute_prs_v2_loss(
    policy_logits: torch.Tensor,     # (B, N_SLOTS)
    value_pred:    torch.Tensor,     # (B, 1)
    slot_targets:  torch.Tensor,     # (B, N_SLOTS) float32 — sums to 1 per row
    legal_mask:    torch.Tensor,     # (B, N_SLOTS) bool
    value_targets: torch.Tensor,     # (B, 1) float32
    value_mask:    torch.Tensor,     # (B, 1) float32, 1 = include in value loss
) -> tuple[torch.Tensor, dict]:
    """Masked CE over legal slots + value MSE.

    Any logit at an illegal slot is set to -inf before softmax.  Cross-entropy
    is summed only over legal slots (0 × -inf = NaN otherwise).
    """
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

    # Rows with no legal actions at all: softmax would be NaN everywhere.
    any_legal = legal_mask.any(dim=-1, keepdim=True)            # (B, 1)
    safe_logits = torch.where(any_legal, masked_logits, torch.zeros_like(masked_logits))
    log_probs = F.log_softmax(safe_logits, dim=-1)

    # Only sum over legal slots (prevents 0 × -inf = NaN at illegal positions)
    contrib = torch.where(legal_mask, slot_targets * log_probs, torch.zeros_like(log_probs))
    per_sample = -contrib.sum(dim=-1)                            # (B,)

    has_policy = any_legal.squeeze(-1)
    if has_policy.any():
        policy_loss = per_sample[has_policy].mean()
    else:
        policy_loss = per_sample.mean()

    value_diff = (value_pred.squeeze(-1) - value_targets.squeeze(-1)) ** 2
    mask_1d = value_mask.squeeze(-1)
    if mask_1d.sum() > 0:
        value_loss = (value_diff * mask_1d).sum() / mask_1d.sum()
    else:
        value_loss = torch.tensor(0.0, device=policy_logits.device)
    total_loss = policy_loss + value_loss
    return total_loss, {"policy_loss": policy_loss, "value_loss": value_loss}


# ── Trainer ────────────────────────────────────────────────────────────

class PRSTrainerV2:
    def __init__(
        self,
        config: PRSTrainConfigV2 | None = None,
        net_config: PRSConfig | None    = None,
    ) -> None:
        self.config     = config or PRSTrainConfigV2()
        self.net_config = net_config or PRSConfig.small()
        self.device     = get_device(self.config.device)

        cfg = self.config
        self.use_amp = cfg.use_amp if cfg.use_amp is not None else (self.device.type == "cuda")
        self._scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Single net: v2 head uses CPU-side bridge, so torch.compile would
        # recompile every batch and hurt throughput. Keep uncompiled.
        self.best_net: HivePRSTransformerV2 = HivePRSTransformerV2(self.net_config).to(self.device)
        self._train_net = self.best_net   # same instance; no compile gap

        self.buffer      = PRSReplayBufferV2(cfg.buffer_max_size)
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

    # ── Self-play ──────────────────────────────────────────────────

    def _self_play(self, iteration: int) -> tuple[list, dict]:
        cfg = self.config
        self.best_net.eval()

        mcts_cfg = PRSMCTSConfigV2(
            num_simulations            = cfg.mcts_simulations,
            max_num_considered_actions = cfg.max_num_considered,
            temperature                = cfg.temperature,
            temperature_drop_move      = cfg.temperature_drop_move,
            batch_size                 = cfg.games_per_batch,
            max_game_length            = cfg.max_game_length,
            expansion_mask             = cfg.expansion_mask,
            nn_max_batch               = cfg.nn_max_batch,
            wave_parallel              = cfg.wave_parallel,
        )
        orchestrator = PRSMCTSOrchestratorV2(self.best_net, mcts_cfg)
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
            if v == 0.0 and cfg.draw_keep_rate < 1.0:
                if np.random.random() > cfg.draw_keep_rate:
                    continue
            flat.extend(game_exs)
        return flat, stats

    # ── Training step ───────────────────────────────────────────────

    def _forward_train(
        self, batch: PRSTrainingBatchV2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full v2 forward from a training batch (trunk + CUDA bridge + head)."""
        ext = hive_gpu.load_extension()
        B = batch.state_bytes.shape[0]
        # Upload state_bytes once and regenerate legal moves on GPU so the
        # CUDA prs_v2_classify kernel can produce all bridge inputs.
        states_gpu = torch.from_numpy(batch.state_bytes).cuda()
        legal_t, nlegal_t = ext.generate_legal_moves_batch(states_gpu, B)
        kernel_out = ext.prs_v2_classify_batch(
            states_gpu, legal_t, nlegal_t, B, int(legal_t.shape[1]),
        )

        board_h, cls_h, full_h, value = self._train_net.forward_trunk(batch.prs_batch)
        inp, _ = build_head_inputs_from_kernel(board_h, cls_h, full_h, kernel_out)
        logits = self._train_net.head(inp)
        return logits, value

    def _train(self, iteration: int) -> tuple[float, dict]:
        cfg = self.config
        if len(self.buffer) < cfg.batch_size:
            return 0.0, {}

        self._train_net.train()
        lr  = self._lr(iteration)
        opt = optim.Adam(self._train_net.parameters(), lr=lr, weight_decay=cfg.weight_decay)

        total_loss_sum = 0.0
        comp_sums: dict[str, float] = {}
        n_batches = 0

        for _ in range(cfg.num_epochs):
            batches_per_epoch = max(1, len(self.buffer) // cfg.batch_size)
            for _ in range(batches_per_epoch):
                batch = self.buffer.sample_batch(
                    cfg.batch_size, augment_prob=cfg.augment_prob,
                )
                batch = batch.to(self.device, non_blocking=True)

                opt.zero_grad()
                if self.use_amp and self._scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits, value = self._forward_train(batch)
                        loss, ld = compute_prs_v2_loss(
                            logits, value,
                            batch.slot_targets, batch.legal_masks,
                            batch.value_targets, batch.value_masks,
                        )
                    self._scaler.scale(loss).backward()
                    if cfg.max_grad_norm > 0:
                        self._scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            self._train_net.parameters(), cfg.max_grad_norm,
                        )
                    self._scaler.step(opt)
                    self._scaler.update()
                else:
                    logits, value = self._forward_train(batch)
                    loss, ld = compute_prs_v2_loss(
                        logits, value,
                        batch.slot_targets, batch.legal_masks,
                        batch.value_targets, batch.value_masks,
                    )
                    loss.backward()
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._train_net.parameters(), cfg.max_grad_norm,
                        )
                    opt.step()

                total_loss_sum += loss.item()
                for k, v in ld.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + v.item()
                n_batches += 1

        self._train_net.eval()
        n = max(n_batches, 1)
        return total_loss_sum / n, {k: v / n for k, v in comp_sums.items()}

    # ── Main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        for iteration in range(self._start_iter, self.config.num_iterations + 1):
            self._cleanup()
            print(f"\n{'='*60}")
            print(f"PRS v2 Iteration {iteration}/{self.config.num_iterations}")
            print(f"{'='*60}")

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

            t0 = time.time()
            loss, loss_dict = self._train(iteration)
            train_time = time.time() - t0
            comp = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            print(f"  Training: loss={loss:.4f}, {train_time:.1f}s  [{comp}]")
            self._cleanup()

            elo = self.elo_tracker.update(1.0, self.config.arena_games)
            print(f"  ELO: {elo:.0f}")
            self._save_checkpoint(iteration)

            total = sp_time + train_time
            print(
                f"  Summary: iter={iteration} loss={loss:.4f} "
                f"elo={elo:.0f} buf={len(self.buffer)} time={total:.0f}s"
            )

    # ── Checkpointing ────────────────────────────────────────────────

    def _save_checkpoint(self, iteration: int) -> None:
        cfg = self.config
        keep = cfg.checkpoint_keep_every
        if keep > 0 and iteration % keep != 0:
            return
        path = os.path.join(cfg.checkpoint_dir, f"prs_v2_iter_{iteration:04d}.pt")
        torch.save({
            "iteration":   iteration,
            "model_state": self.best_net.state_dict(),
            "net_config":  self.net_config,
        }, path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.best_net.load_state_dict(ckpt["model_state"])
        self._start_iter = ckpt["iteration"] + 1
        print(f"Resumed from {path} (iter {ckpt['iteration']})")
