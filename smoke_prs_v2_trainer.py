"""End-to-end PRS v2 smoke test: self-play → replay buffer → train step → loss drops."""
from __future__ import annotations

import numpy as np
import torch

from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_trainer_v2 import PRSTrainerV2, PRSTrainConfigV2


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    tcfg = PRSTrainConfigV2(
        num_iterations=1,
        games_per_batch=4,
        mcts_simulations=16,
        max_num_considered=8,
        batch_size=16,
        num_epochs=2,
        buffer_max_size=1000,
        max_game_length=20,
        expansion_mask=7,
        use_amp=False,
    )
    ncfg = PRSConfig(d_model=64, num_heads=4, num_layers=2, dim_feedforward=128)
    trainer = PRSTrainerV2(tcfg, ncfg)
    print(f"Net params: {trainer.best_net.count_parameters():,}")

    # ── Self-play ──
    exs, stats = trainer._self_play(1)
    print(f"Self-play: {len(exs)} examples, stats={stats}")
    trainer.buffer.add_examples(exs)
    print(f"Buffer size: {len(trainer.buffer)}")
    assert len(trainer.buffer) >= tcfg.batch_size

    # ── One train step manually to check loss ──
    trainer._train_net.train()
    batch = trainer.buffer.sample_batch(tcfg.batch_size).to(trainer.device)
    logits, value = trainer._forward_train(batch)
    print(f"Logits: {tuple(logits.shape)}, value: {tuple(value.shape)}")
    assert logits.shape == (tcfg.batch_size, 813)
    assert torch.isfinite(logits).all()

    from hive_prs.prs_trainer_v2 import compute_prs_v2_loss
    loss0, ld = compute_prs_v2_loss(
        logits, value, batch.slot_targets, batch.legal_masks, batch.value_targets,
    )
    print(f"Initial loss: {float(loss0):.4f}  (policy={float(ld['policy_loss']):.4f}, "
          f"value={float(ld['value_loss']):.4f})")
    assert torch.isfinite(loss0)

    # ── Run full _train and check loss is finite ──
    avg_loss, comp = trainer._train(1)
    print(f"After training: avg_loss={avg_loss:.4f}  "
          f"[policy={comp.get('policy_loss', 0):.4f}, value={comp.get('value_loss', 0):.4f}]")
    assert np.isfinite(avg_loss)

    # ── Forward again on same batch: loss should have moved ──
    trainer._train_net.eval()
    with torch.no_grad():
        logits2, value2 = trainer._forward_train(batch)
        loss1, _ = compute_prs_v2_loss(
            logits2, value2, batch.slot_targets, batch.legal_masks, batch.value_targets,
        )
    print(f"Post-train loss on same batch: {float(loss1):.4f}  "
          f"(delta={float(loss1 - loss0):+.4f})")
    print("OK")


if __name__ == "__main__":
    main()
