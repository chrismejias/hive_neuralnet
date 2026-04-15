import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import cProfile
import pstats
import time
import io
import sys
import numpy as np
import torch

from hive_fnn.fnn_network import HiveFNN, FNNConfig
from hive_fnn.fnn_orchestrator import FNNGumbelOrchestrator, FNNGumbelConfig
from hive_fnn.fnn_replay_buffer import FNNReplayBuffer
from hive_fnn.fnn_trainer import compute_fnn_loss

import hive_gpu


def profile_size(preset_name: str, config: FNNConfig, games: int, sims: int, gumbel: int):
    device = torch.device("cuda")
    ext = hive_gpu.load_extension()

    net = HiveFNN(config).to(device)
    net.eval()
    n_params = net.count_parameters()
    print(f"\n{'#'*70}")
    print(f"  FNN {preset_name}: {n_params:,} params")
    print(f"  hidden={config.hidden_dim} embed={config.embed_dim} action_hidden={config.action_hidden}")
    print(f"  Self-play: {games} games, {sims} sims, {gumbel} Gumbel")
    print(f"{'#'*70}")

    # Warm up
    print("Warming up...")
    orch = FNNGumbelOrchestrator(net, FNNGumbelConfig(
        batch_size=2, num_simulations=4, max_num_considered_actions=2, max_game_length=20,
    ))
    orch.self_play_batch()
    torch.cuda.synchronize()
    print("Warm-up done.\n")

    # --- Profile self-play ---
    orch = FNNGumbelOrchestrator(net, FNNGumbelConfig(
        batch_size=games, num_simulations=sims, max_num_considered_actions=gumbel,
        max_game_length=300,
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
    batch_size = min(16, len(flat))
    if batch_size >= 4:
        buf = FNNReplayBuffer(100_000)
        buf.add_examples(flat)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        n_batches = max(1, len(flat) // batch_size)

        pr2 = cProfile.Profile()
        t1 = time.perf_counter()
        pr2.enable()
        for _ in range(n_batches):
            batch = buf.sample_batch(batch_size).to(device)
            opt.zero_grad()

            # Encode root + all successors via CUDA kernel
            B_batch = batch.state_bytes.shape[0]
            legal_moves, num_legal = ext.generate_legal_moves_batch(batch.state_bytes, B_batch)
            num_actions = num_legal.to(torch.int64)
            root_feat = ext.extract_fnn_features_batch(batch.state_bytes, legal_moves, num_legal, B_batch)

            max_legal = legal_moves.shape[1]
            slot_idx = torch.arange(max_legal, device=device, dtype=torch.int64).unsqueeze(0)
            valid = slot_idx < num_actions.unsqueeze(1)
            action_to_root = torch.arange(B_batch, device=device, dtype=torch.int64).unsqueeze(1).expand_as(valid)[valid]
            move_indices = slot_idx.expand_as(valid)[valid]
            total_actions = action_to_root.shape[0]

            if total_actions > 0:
                child_states = batch.state_bytes[action_to_root].clone()
                moves = legal_moves[action_to_root, move_indices]
                ext.apply_moves_batch(child_states, moves, total_actions)
                child_lm, child_nl = ext.generate_legal_moves_batch(child_states, total_actions)
                succ_feat = ext.extract_fnn_features_batch(child_states, child_lm, child_nl, total_actions)
            else:
                succ_feat = root_feat[:0]

            action_logits, root_values = net(root_feat, succ_feat, action_to_root, num_actions)
            loss, _ = compute_fnn_loss(
                action_logits, root_values,
                batch.policy_targets, batch.value_targets, batch.num_actions,
            )
            loss.backward()
            opt.step()
        pr2.disable()
        torch.cuda.synchronize()
        tr_time = time.perf_counter() - t1
        print(f"Training:  {tr_time:.1f}s, {n_batches} batches of {batch_size}")
    else:
        pr2 = None
        tr_time = 0

    # --- Print profiles ---
    print(f"\n{'='*70}")
    print(f"SELF-PLAY PROFILE [{preset_name}] (top 30 by cumulative time)")
    print(f"{'='*70}")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    print(f"\n{'='*70}")
    print(f"SELF-PLAY PROFILE [{preset_name}] (top 30 by total/internal time)")
    print(f"{'='*70}")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    print(s.getvalue())

    if pr2:
        print(f"\n{'='*70}")
        print(f"TRAINING PROFILE [{preset_name}] (top 30 by cumulative time)")
        print(f"{'='*70}")
        s = io.StringIO()
        ps = pstats.Stats(pr2, stream=s).sort_stats("cumulative")
        ps.print_stats(30)
        print(s.getvalue())

    print(f"\n{'='*70}")
    print(f"SUMMARY [{preset_name}]: Self-play={sp_time:.1f}s  Training={tr_time:.1f}s  Total={sp_time+tr_time:.1f}s")
    print(f"{'='*70}")

    # Cleanup
    del net, orch
    torch.cuda.empty_cache()

    return sp_time, tr_time


def main():
    games, sims, gumbel = 32, 16, 4

    presets = [
        ("small", FNNConfig.small()),
        ("medium", FNNConfig.medium()),
        ("large", FNNConfig.large()),
    ]

    results = {}
    for name, cfg in presets:
        sp, tr = profile_size(name, cfg, games, sims, gumbel)
        results[name] = (sp, tr)

    print(f"\n\n{'='*70}")
    print(f"COMPARISON SUMMARY ({games} games, {sims} sims, {gumbel} Gumbel)")
    print(f"{'='*70}")
    print(f"{'Preset':<10} {'Params':>8} {'Self-play':>12} {'Training':>12} {'Total':>12}")
    print(f"{'-'*54}")
    for name, cfg in presets:
        n = HiveFNN(cfg).count_parameters()
        sp, tr = results[name]
        print(f"{name:<10} {n:>8,} {sp:>10.1f}s {tr:>10.1f}s {sp+tr:>10.1f}s")


if __name__ == "__main__":
    main()
