"""Benchmark the CPU-side slot-mapping bridge against trunk/head GPU compute.

Goal: decide whether the CPU bridge is actually the bottleneck that warrants
a CUDA port. Reports per-batch timings for (a) legal-move slot mapping,
(b) head-input bridge, (c) trunk forward, (d) head forward.
"""
from __future__ import annotations

import time
import numpy as np
import torch

import hive_gpu
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.prs_v2_bridge import build_head_inputs_from_states
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSOrchestratorV2, PRSMCTSConfigV2
from hive_prs.slot_map import map_legal_moves


def play_random(ext, states, B, plies):
    for _ in range(plies):
        legal, nlegal = ext.generate_legal_moves_batch(states, B)
        nc = nlegal.cpu().numpy()
        chosen = np.zeros(B, dtype=np.int64)
        for i in range(B):
            chosen[i] = np.random.randint(max(1, int(nc[i])))
        chosen_t = torch.from_numpy(chosen).cuda()
        moves = legal[torch.arange(B, device="cuda"), chosen_t]
        ext.apply_moves_batch(states, moves, B)


def main() -> None:
    ext = hive_gpu.load_extension()
    ext.initialize_tables()
    B = 128
    states = ext.create_initial_states(B, 7)
    play_random(ext, states, B, 20)

    encoder = PRSEncoder()
    cfg = PRSConfig(d_model=128, num_heads=8, num_layers=6, dim_feedforward=512)
    net = HivePRSTransformerV2(cfg).cuda().eval()

    def bench(name, fn, n=5):
        torch.cuda.synchronize()
        # Warmup
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n * 1000
        print(f"  {name:40s} {dt:7.2f} ms / batch")
        return dt

    print(f"B={B}, d_model={cfg.d_model}, layers={cfg.num_layers}\n")

    # ── Primitive timings ─────────────────────────────────────────
    legal_t, nlegal_t = ext.generate_legal_moves_batch(states, B)
    states_cpu = states.cpu().numpy()
    legal_np  = legal_t.cpu().numpy()
    nlegal_np = nlegal_t.cpu().numpy()

    bench("ext.generate_legal_moves_batch", lambda: ext.generate_legal_moves_batch(states, B))
    bench("states.cpu() + legal.cpu()",
          lambda: (states.cpu().numpy(), legal_t.cpu().numpy(), nlegal_t.cpu().numpy()))

    def slot_map_batch():
        for b in range(B):
            n = int(nlegal_np[b])
            if n == 0: continue
            map_legal_moves(states_cpu[b], legal_np[b, :n], n)
    bench("map_legal_moves (CPU loop x B)", slot_map_batch)

    prs_batch = encoder.encode_batch(states, B)
    bench("encoder.encode_batch", lambda: encoder.encode_batch(states, B))

    def trunk_only():
        with torch.no_grad():
            net.forward_trunk(prs_batch)
    bench("trunk.forward", trunk_only)

    with torch.no_grad():
        board_h, cls_h, full_h, value = net.forward_trunk(prs_batch)

    def bridge_only():
        build_head_inputs_from_states(states_cpu, board_h, cls_h, full_h)
    bench("build_head_inputs_from_states", bridge_only)

    inp, _ = build_head_inputs_from_states(states_cpu, board_h, cls_h, full_h)
    def head_only():
        with torch.no_grad():
            net.head(inp)
    bench("head.forward (813 slots)", head_only)

    # ── CUDA bridge (replaces map_legal_moves + build_head_inputs) ──
    def cuda_classify():
        ext.prs_v2_classify_batch(states, legal_t, nlegal_t, B, legal_t.shape[1])
    bench("ext.prs_v2_classify_batch (CUDA)", cuda_classify)

    # ── Full orchestrator self-play (end-to-end) ─────────────────
    print()
    mcfg = PRSMCTSConfigV2(
        num_simulations=64, max_num_considered_actions=8,
        batch_size=B, max_game_length=10, expansion_mask=7,
        wave_size=8, max_tree_nodes=2048,
    )
    orch = PRSMCTSOrchestratorV2(net, mcfg)
    t0 = time.perf_counter()
    orch.self_play_batch()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  Full self-play {B} games × 10 plies × 64 sims: {dt:.1f}s")


if __name__ == "__main__":
    main()
