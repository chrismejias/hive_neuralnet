import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import cProfile
import pstats
import time
import io
import numpy as np
import torch

from hive_prs.prs_transformer import HivePRSTransformer, PRSConfig
from hive_prs.prs_orchestrator import PRSGumbelOrchestrator, PRSGumbelConfig
from hive_prs.prs_replay_buffer import PRSReplayBuffer
from hive_prs.prs_trainer import compute_prs_loss


def main():
    device = torch.device("cuda")
    net = HivePRSTransformer(PRSConfig()).to(device)
    net.eval()
    print(f"PRS params: {sum(p.numel() for p in net.parameters()):,}")

    # Warm up
    print("Warming up...")
    orch = PRSGumbelOrchestrator(net, PRSGumbelConfig(
        batch_size=4, num_simulations=4, max_num_considered_actions=2, max_game_length=50,
    ))
    orch.self_play_batch()
    torch.cuda.synchronize()
    print("Warm-up done.\n")

    # --- Profile self-play ---
    orch = PRSGumbelOrchestrator(net, PRSGumbelConfig(
        batch_size=64, num_simulations=64, max_num_considered_actions=4, max_game_length=300,
    ))

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    examples = orch.self_play_batch()
    pr.disable()
    torch.cuda.synchronize()
    sp_time = time.perf_counter() - t0

    flat = [ex for game in examples for ex in game]
    n_games = len(examples)
    wins = sum(1 for g in examples if g and g[0].value_target > 0)
    losses = sum(1 for g in examples if g and g[0].value_target < 0)
    draws = n_games - wins - losses
    print(f"Self-play: {sp_time:.1f}s, {len(flat)} examples, {n_games} games (W:{wins} B:{losses} D:{draws})")

    # --- Profile training ---
    if len(flat) >= 8:
        buf = PRSReplayBuffer(100_000)
        buf.add_examples(flat)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)

        pr2 = cProfile.Profile()
        t1 = time.perf_counter()
        pr2.enable()
        for _ in range(max(1, len(flat) // 8)):
            batch = buf.sample_batch(8).to(device)
            opt.zero_grad()
            logits, value = net(batch.prs_batch)
            loss, _ = compute_prs_loss(logits, value, batch.policy_targets, batch.value_targets)
            loss.backward()
            opt.step()
        pr2.disable()
        torch.cuda.synchronize()
        tr_time = time.perf_counter() - t1
        print(f"Training:  {tr_time:.1f}s, {max(1, len(flat)//8)} batches")
    else:
        pr2 = None
        tr_time = 0

    # --- Print profiles ---
    print(f"\n{'='*70}")
    print(f"SELF-PLAY PROFILE (top 40 by cumulative time)")
    print(f"{'='*70}")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    print(s.getvalue())

    print(f"\n{'='*70}")
    print(f"SELF-PLAY PROFILE (top 40 by total/internal time)")
    print(f"{'='*70}")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(40)
    print(s.getvalue())

    if pr2:
        print(f"\n{'='*70}")
        print(f"TRAINING PROFILE (top 40 by cumulative time)")
        print(f"{'='*70}")
        s = io.StringIO()
        ps = pstats.Stats(pr2, stream=s).sort_stats("cumulative")
        ps.print_stats(40)
        print(s.getvalue())

    print(f"\n{'='*70}")
    print(f"SUMMARY: Self-play={sp_time:.1f}s  Training={tr_time:.1f}s  Total={sp_time+tr_time:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
