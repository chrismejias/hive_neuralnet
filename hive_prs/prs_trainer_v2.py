"""PRS v2 training loop (structured 813-slot head).

Differences vs v1 trainer:
  * Uses HivePRSTransformerV2 + PRSReplayBufferV2 + PRSMCTSOrchestratorV2.
  * Loss is masked cross-entropy over 813 slots (legal mask stored with each
    example) + value MSE. No surprise-weighted sampling, no policy softening
    (the slot space is small and already legal-masked).
  * The trunk and head run on GPU; the CUDA bridge produces dynamic head
    inputs from state/legal tensors. The tensor-only trunk/head can be
    torch-compiled while the extension bridge remains outside Dynamo.
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
from hive_prs.prs_transformer_v3 import HivePRSTransformerV3
from hive_prs.prs_replay_buffer_v2 import PRSReplayBufferV2, PRSTrainingBatchV2
from hive_prs.prs_mcts_orchestrator_v2 import (
    PRSMCTSConfigV2, PRSMCTSOrchestratorV2,
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
    simulation_schedule:       tuple[int, ...] = ()
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
    slot_legality_loss_weight:  float = 0.0
    articulation_loss_weight:   float = 0.0

    # LR schedule
    lr_schedule:               str   = "cosine"
    lr_warmup_iterations:      int   = 3
    lr_min:                    float = 1e-5

    # Replay buffer
    buffer_max_size:           int   = 150_000

    # Checkpointing
    checkpoint_dir:            str   = "checkpoints_prs_v3"
    checkpoint_keep_every:     int   = 0

    # Arena
    skip_arena:                bool  = True
    arena_games:               int   = 20

    # Misc
    draw_keep_rate:            float = 1.0
    expansion_mask:             int   = 7
    nn_max_batch:              int   = 0
    wave_parallel:             bool  = True
    deterministic_non_root:    bool  = False
    virtual_q_penalty:         float = 0.25
    non_root_sigma:            float = 1.0
    compile_forward:           bool  = False
    model_version:             str   = "v3"

    # C6 (6-fold) rotational augmentation for training batches.
    # With this probability, a random rotation k∈{1..5} is applied to every
    # sample in the batch (state + legal moves), tokens are re-encoded,
    # and slot_target/legal_mask are rebuilt via SlotMapper.  0.0 = off.
    augment_prob:              float = 0.5

    device:                    str | None = None
    use_amp:                   bool | None = None


def _simulations_for_iteration(
    base_simulations: int,
    schedule: tuple[int, ...],
    iteration: int,
) -> int:
    if not schedule:
        return base_simulations
    return int(schedule[(iteration - 1) % len(schedule)])


# ── Loss ───────────────────────────────────────────────────────────────

def compute_prs_v2_loss(
    policy_logits: torch.Tensor,     # (B, N_SLOTS)
    value_pred:    torch.Tensor,     # (B, 1)
    slot_targets:  torch.Tensor,     # (B, N_SLOTS) float32 — sums to 1 per row
    legal_mask:    torch.Tensor,     # (B, N_SLOTS) bool
    value_targets: torch.Tensor,     # (B, 1) float32
    value_mask:    torch.Tensor,     # (B, 1) float32, 1 = include in value loss
    aux_outputs:   dict[str, torch.Tensor],
    articulation_targets: torch.Tensor,    # (B, MAX_BOARD)
    articulation_mask:    torch.Tensor,    # (B, MAX_BOARD)
    slot_legality_weight: float = 0.15,
    articulation_weight:  float = 0.15,
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
    loss_dict: dict[str, torch.Tensor] = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }

    if (
        slot_legality_weight > 0.0
        and "slot_legality_logits" in aux_outputs
    ):
        sl_logits = aux_outputs["slot_legality_logits"]
        sl_target = legal_mask.float()
        sl_loss = F.binary_cross_entropy_with_logits(sl_logits, sl_target)
        total_loss = total_loss + slot_legality_weight * sl_loss
        loss_dict["slot_legality_loss"] = sl_loss

    if (
        articulation_weight > 0.0
        and "articulation_logits" in aux_outputs
        and articulation_targets.numel() > 0
    ):
        art_logits = aux_outputs["articulation_logits"]
        art_bce = F.binary_cross_entropy_with_logits(
            art_logits, articulation_targets, reduction="none",
        )
        if articulation_mask.sum() > 0:
            art_loss = (art_bce * articulation_mask).sum() / articulation_mask.sum()
        else:
            art_loss = torch.tensor(0.0, device=policy_logits.device)
        total_loss = total_loss + articulation_weight * art_loss
        loss_dict["articulation_loss"] = art_loss

    return total_loss, loss_dict


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

        # Compile only tensor-heavy trunk/head paths; the CUDA bridge stays
        # outside Dynamo because its output structure depends on game state.
        net_cls = HivePRSTransformerV3 if self.config.model_version == "v3" else HivePRSTransformerV2
        self.best_net: HivePRSTransformerV2 = net_cls(self.net_config).to(self.device)
        self.best_net.enable_compiled_forward(cfg.compile_forward and self.device.type == "cuda")
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
        sims_this_iter = _simulations_for_iteration(
            cfg.mcts_simulations, cfg.simulation_schedule, iteration,
        )

        mcts_cfg = PRSMCTSConfigV2(
            num_simulations            = sims_this_iter,
            max_num_considered_actions = cfg.max_num_considered,
            temperature                = cfg.temperature,
            temperature_drop_move      = cfg.temperature_drop_move,
            batch_size                 = cfg.games_per_batch,
            max_game_length            = cfg.max_game_length,
            expansion_mask             = cfg.expansion_mask,
            nn_max_batch               = cfg.nn_max_batch,
            wave_parallel              = cfg.wave_parallel,
            deterministic_non_root     = cfg.deterministic_non_root,
            virtual_q_penalty          = cfg.virtual_q_penalty,
            non_root_sigma             = cfg.non_root_sigma,
            compile_forward            = cfg.compile_forward,
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
            # Only subsample move-cap draws. Genuine drawn results
            # (result == 3) should still train both policy and value.
            is_capped_draw = not bool(game_exs[0].use_for_value)
            if is_capped_draw and cfg.draw_keep_rate < 1.0:
                if np.random.random() > cfg.draw_keep_rate:
                    continue
            flat.extend(game_exs)
        return flat, stats

    # ── Training step ───────────────────────────────────────────────

    def _forward_train(
        self, batch: PRSTrainingBatchV2,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Full v2 forward from a training batch (trunk + CUDA bridge + heads)."""
        ext = hive_gpu.load_extension()
        B = int(batch.state_bytes.shape[0])
        states_gpu = batch.state_bytes
        legal_t = batch.legal_moves_raw
        nlegal_t = batch.nlegal_raw
        kernel_out = ext.prs_v2_classify_batch(
            states_gpu, legal_t, nlegal_t, B, int(legal_t.shape[1]),
        )

        cfg = self.config
        if cfg.slot_legality_loss_weight <= 0.0 and cfg.articulation_loss_weight <= 0.0:
            logits, value = self._train_net.forward_from_kernel(batch.prs_batch, kernel_out)
            return logits, value, {}
        return self._train_net.forward_train_from_kernel(batch.prs_batch, kernel_out)

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
        total_batches = cfg.num_epochs * max(1, len(self.buffer) // cfg.batch_size)
        prefetch_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

        def fetch_next_batch() -> PRSTrainingBatchV2:
            batch = self.buffer.sample_batch(
                cfg.batch_size, augment_prob=cfg.augment_prob,
            )
            if prefetch_stream is None:
                return batch.to(self.device, non_blocking=True)
            with torch.cuda.stream(prefetch_stream):
                return batch.to(self.device, non_blocking=True)

        for _ in range(cfg.num_epochs):
            batches_per_epoch = max(1, len(self.buffer) // cfg.batch_size)
            next_batch = fetch_next_batch()
            for _ in range(batches_per_epoch):
                if prefetch_stream is not None:
                    torch.cuda.current_stream(self.device).wait_stream(prefetch_stream)
                batch = next_batch
                if n_batches + 1 < total_batches:
                    next_batch = fetch_next_batch()

                opt.zero_grad(set_to_none=True)
                if self.use_amp and self._scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits, value, aux = self._forward_train(batch)
                        loss, ld = compute_prs_v2_loss(
                            logits, value,
                            batch.slot_targets, batch.legal_masks,
                            batch.value_targets, batch.value_masks,
                            aux,
                            batch.articulation_targets,
                            batch.articulation_mask,
                            slot_legality_weight=cfg.slot_legality_loss_weight,
                            articulation_weight=cfg.articulation_loss_weight,
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
                    logits, value, aux = self._forward_train(batch)
                    loss, ld = compute_prs_v2_loss(
                        logits, value,
                        batch.slot_targets, batch.legal_masks,
                        batch.value_targets, batch.value_masks,
                        aux,
                        batch.articulation_targets,
                        batch.articulation_mask,
                        slot_legality_weight=cfg.slot_legality_loss_weight,
                        articulation_weight=cfg.articulation_loss_weight,
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
            sims_this_iter = _simulations_for_iteration(
                self.config.mcts_simulations,
                self.config.simulation_schedule,
                iteration,
            )
            print(f"\n{'='*60}")
            print(f"PRS {self.config.model_version} Iteration {iteration}/{self.config.num_iterations}")
            print(f"  Simulations: {sims_this_iter}")
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
        prefix = "prs_v3" if cfg.model_version == "v3" else "prs_v2"
        path = os.path.join(cfg.checkpoint_dir, f"{prefix}_iter_{iteration:04d}.pt")
        torch.save({
            "iteration":   iteration,
            "model_state": self.best_net.state_dict(),
            "net_config":  self.net_config,
            "model_version": cfg.model_version,
        }, path)
        print(f"  Saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model_state = ckpt["model_state"]
        current = self.best_net.state_dict()
        filtered_state = {}
        skipped: list[str] = []
        for key, tensor in model_state.items():
            if key not in current:
                continue
            if current[key].shape != tensor.shape:
                skipped.append(key)
                continue
            filtered_state[key] = tensor
        missing, unexpected = self.best_net.load_state_dict(
            filtered_state, strict=False,
        )
        self._start_iter = ckpt["iteration"] + 1
        if skipped:
            print(f"Checkpoint shape-mismatch keys skipped: {skipped}")
        if missing:
            print(f"Checkpoint missing keys: {missing}")
        if unexpected:
            print(f"Checkpoint unexpected keys: {unexpected}")
        print(f"Resumed from {path} (iter {ckpt['iteration']})")
