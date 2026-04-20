"""End-to-end smoke test for C6-augmented PRS v2 training.

Runs one small self-play + train iteration with augment_prob=1.0 to force
the rotated collate path, then compares losses with augment_prob=0.0.
"""
from __future__ import annotations

import numpy as np
import torch

from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_trainer_v2 import PRSTrainerV2, PRSTrainConfigV2


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = PRSTrainConfigV2(
        num_iterations       = 1,
        games_per_batch      = 16,
        mcts_simulations     = 32,
        max_num_considered   = 8,
        max_game_length      = 10,
        batch_size           = 32,
        num_epochs           = 1,
        learning_rate        = 1e-3,
        buffer_max_size      = 2_000,
        checkpoint_keep_every= 0,
        skip_arena           = True,
        augment_prob         = 1.0,     # force rotation on every batch
    )
    net_cfg = PRSConfig(d_model=64, num_heads=4, num_layers=2, dim_feedforward=128)

    trainer = PRSTrainerV2(cfg, net_cfg)

    # Self-play
    examples, stats = trainer._self_play(iteration=1)
    print(f"Self-play: {len(examples)} examples, stats={stats}")
    trainer.buffer.add_examples(examples)

    # ── Train one batch with augment_prob=1.0 ──
    print("\n── augment_prob=1.0 (rotated path) ──")
    cfg.augment_prob = 1.0
    loss_rot, ld_rot = trainer._train(iteration=1)
    print(f"  loss (rotated): {loss_rot:.4f}  ({ld_rot})")

    # ── Train one batch with augment_prob=0.0 (cached path) ──
    print("\n── augment_prob=0.0 (cached path) ──")
    cfg.augment_prob = 0.0
    loss_cache, ld_cache = trainer._train(iteration=2)
    print(f"  loss (cached):  {loss_cache:.4f}  ({ld_cache})")

    # Both paths should produce finite losses
    assert np.isfinite(loss_rot), "rotated-path loss is not finite"
    assert np.isfinite(loss_cache), "cached-path loss is not finite"

    # ── Sanity: pick a single sample, compare k=0 vs k=1 batches ──
    print("\n── Batch structure sanity ──")
    b_cache = trainer.buffer.sample_batch(8, augment_prob=0.0)
    b_rot   = trainer.buffer.sample_batch(8, augment_prob=1.0)
    print(f"  k=0 batch: slot_targets shape={b_cache.slot_targets.shape}, "
          f"augmentation_k={b_cache.augmentation_k}")
    print(f"  k>0 batch: slot_targets shape={b_rot.slot_targets.shape}, "
          f"augmentation_k={b_rot.augmentation_k}")
    # Both must have per-row prob sums ≈ 1.0 (or 0 for empty positions)
    for lbl, b in [("cache", b_cache), ("rot", b_rot)]:
        sums = b.slot_targets.sum(dim=1).cpu().numpy()
        ok = ((sums > 0.999) & (sums < 1.001)) | (sums < 1e-6)
        assert ok.all(), f"{lbl}: slot_target sums out of [0 or 1]: {sums}"
    print("  slot_targets normalize correctly on both paths: OK")

    print("\n== C6 training smoke: PASS ==")


if __name__ == "__main__":
    main()
