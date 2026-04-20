"""Smoke-test PRSMCTSOrchestratorV2 end-to-end.

Runs one small self-play batch and verifies:
  * orchestrator returns a non-empty list of per-game example lists
  * every example has slot_target summing to ~1.0 (or 0 on dead rows)
  * legal_mask is a superset of where slot_target > 0
  * state_bytes round-trips through the replay buffer collator
"""
from __future__ import annotations

import numpy as np
import torch

from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.prs_mcts_orchestrator_v2 import (
    PRSMCTSConfigV2, PRSMCTSOrchestratorV2,
)
from hive_prs.prs_replay_buffer_v2 import PRSReplayBufferV2


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    cfg_net = PRSConfig(d_model=64, num_heads=4, num_layers=2, dim_feedforward=128)
    net = HivePRSTransformerV2(cfg_net).cuda().eval()
    print(f"Net params: {net.count_parameters():,}")

    cfg = PRSMCTSConfigV2(
        num_simulations=16, max_num_considered_actions=8,
        batch_size=4, max_game_length=20, expansion_mask=7,
        wave_size=4, max_tree_nodes=512,
    )
    orch = PRSMCTSOrchestratorV2(net, cfg)

    examples_per_game = orch.self_play_batch()
    print(f"Games: {len(examples_per_game)}")
    for gi, exs in enumerate(examples_per_game):
        print(f"  game {gi}: {len(exs)} examples")

    all_exs = [e for exs in examples_per_game for e in exs]
    assert all_exs, "no examples produced"
    print(f"Total examples: {len(all_exs)}")

    # Validate slot_target + legal_mask invariants
    bad = 0
    for ex in all_exs:
        s = ex.slot_target.sum()
        if not (0.999 <= s <= 1.001):
            bad += 1
        # slot_target > 0 ⇒ legal_mask True
        nonzero = ex.slot_target > 0
        if not np.all(ex.legal_mask[nonzero]):
            bad += 1
    print(f"Invariant violations: {bad}")
    assert bad == 0

    # Replay buffer collation
    buf = PRSReplayBufferV2(max_size=1000)
    buf.add_examples(all_exs)
    batch = buf.sample_batch(min(8, len(buf)))
    print(f"Collated batch: slot_targets={tuple(batch.slot_targets.shape)}, "
          f"state_bytes={batch.state_bytes.shape}, "
          f"value={tuple(batch.value_targets.shape)}")
    # Slot targets should still sum to ~1 per row
    row_sums = batch.slot_targets.sum(dim=1).numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-3), row_sums
    print("OK")


if __name__ == "__main__":
    main()
