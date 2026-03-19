"""
Find the CPU/GPU crossover: the batch size where GPU time starts rising
and the number of workers needed to sustain a given batch size.

Run from hive_neuralnet/:
    .venv/Scripts/python profile_crossover.py
"""

import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from hive_engine.game_state import GameState, GameResult
from hive_engine.mcts import MCTS, MCTSConfig
from hive_transformer.transformer_encoder import TransformerEncoder
from hive_transformer.transformer_net import HiveTransformer
from hive_transformer.token_types import HiveTokenBatch
from hive_engine.batched_inference import BatchedInferenceServer, BatchedPredictor
from hive_engine.device import get_device

device = get_device()
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0)
    print(f"     {mem.total_memory / 1e9:.1f} GB VRAM, "
          f"{mem.multi_processor_count} SMs")

ckpt = torch.load(
    "checkpoints_transformer/hive_transformer_checkpoint_0037.pt",
    map_location="cpu", weights_only=False
)
net_config = ckpt["net_config"]
net = HiveTransformer(net_config).to(device)
net.load_state_dict(ckpt["net_state_dict"], strict=False)
net.eval()
print(f"Model: d_model={net_config.d_model}, "
      f"layers={net_config.num_layers}, "
      f"heads={net_config.num_heads}\n")

encoder = TransformerEncoder()

# Build representative game states at different game lengths
import random
random.seed(42)

def make_game_state(num_moves):
    g = GameState()
    for _ in range(num_moves):
        m = g.legal_moves()
        if not m or g.result != GameResult.IN_PROGRESS:
            break
        g.apply_move(random.choice(m))
    return g

early_game  = make_game_state(10)   # ~8 pieces on board
mid_game    = make_game_state(30)   # ~18 pieces
late_game   = make_game_state(60)   # ~22 pieces (max)

seq_mid = encoder.encode_state(mid_game)
mask_mid = encoder.get_legal_action_mask(mid_game)
print(f"State sizes — early: {encoder.encode_state(early_game).num_board_tokens} tokens, "
      f"mid: {seq_mid.num_board_tokens} tokens, "
      f"late: {encoder.encode_state(late_game).num_board_tokens} tokens")

# ── 1. GPU forward pass from B=1 to B=512 ────────────────────────────────────
print("\n" + "="*65)
print("1. GPU forward pass time vs batch size  (mid-game states)")
print("="*65)
print(f"  {'Batch':>6}  {'ms/call':>8}  {'ms/item':>8}  {'items/s':>10}  "
      f"{'vs B=1':>8}")

WARMUP = 5
N_RUNS = 30

ref_throughput = None
for B in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
    try:
        batch = HiveTokenBatch.collate([seq_mid] * B).to(device)
        masks = torch.stack([torch.from_numpy(mask_mid)] * B).to(device)

        with torch.no_grad():
            for _ in range(WARMUP):
                _ = net.forward(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(N_RUNS):
                _ = net.forward(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / N_RUNS * 1000  # ms

        tput = B / elapsed * 1000  # items/sec
        if ref_throughput is None:
            ref_throughput = tput
        ratio = tput / ref_throughput

        # Flag where time starts rising notably (>20% over minimum)
        flag = ""
        if B > 1 and elapsed > 9.3 * 1.20:
            flag = " << time rising"

        print(f"  {B:>6}  {elapsed:>8.2f}  {elapsed/B:>8.3f}  "
              f"{tput:>10.0f}  {ratio:>8.1f}x{flag}")
    except RuntimeError as e:
        print(f"  {B:>6}  OOM: {e}")
        break

# ── 2. Measure encode_state() at each game phase ─────────────────────────────
print("\n" + "="*65)
print("2. encode_state() cost by game phase")
print("="*65)

for label, gs in [("early (10 moves)", early_game),
                  ("mid   (30 moves)", mid_game),
                  ("late  (60 moves)", late_game)]:
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        encoder.encode_state(gs)
    ms = (time.perf_counter() - t0) / N * 1000
    tokens = encoder.encode_state(gs).num_board_tokens
    print(f"  {label}: {ms:.3f} ms/call  ({tokens} board tokens)")

# ── 3. Worker scaling: find max sustainable batch size ───────────────────────
print("\n" + "="*65)
print("3. Worker scaling — sustained batch size & throughput")
print("   (200 sims per worker, mid-game state, pre-encoded)")
print("="*65)
print(f"  {'Workers':>8}  {'MaxBatch':>9}  {'AvgBatch':>9}  "
      f"{'time(s)':>8}  {'sims/s':>8}  {'speedup':>8}")
print(f"  {'-'*60}")

seq_list = [seq_mid] * 300  # pre-encoded states
mask_list = [mask_mid] * 300

SIMS_PER_WORKER = 200
baseline_rate = None

for num_workers in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
    max_batch = num_workers  # set max_batch = num_workers to test scaling
    server = BatchedInferenceServer(
        net=net,
        collate_fn=HiveTokenBatch.collate,
        device=device,
        max_batch_size=max_batch,
        max_wait_ms=5.0,
    )
    server.start()
    batched_net = BatchedPredictor(server)

    barrier = threading.Barrier(num_workers)
    total_sims = num_workers * SIMS_PER_WORKER

    def worker_fn(worker_id):
        barrier.wait()
        for i in range(SIMS_PER_WORKER):
            idx = (worker_id * SIMS_PER_WORKER + i) % len(seq_list)
            batched_net.predict(seq_list[idx], mask_list[idx])

    threads = [
        threading.Thread(target=worker_fn, args=(i,))
        for i in range(num_workers)
    ]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.perf_counter() - t0

    avg_batch = server._total_items / server._total_batches if server._total_batches else 0
    sims_per_sec = total_sims / elapsed
    server.stop()

    if baseline_rate is None:
        baseline_rate = sims_per_sec
    speedup = sims_per_sec / baseline_rate

    print(f"  {num_workers:>8}  {max_batch:>9}  {avg_batch:>9.1f}  "
          f"{elapsed:>8.2f}  {sims_per_sec:>8.0f}  {speedup:>8.1f}x")

# ── 4. Real MCTS crossover: find optimal workers with full encode() ───────────
print("\n" + "="*65)
print("4. Real MCTS crossover — full encode + MCTS with 20 sims/move")
print("   Each thread plays 10 moves with 20 MCTS sims each")
print("="*65)
print(f"  {'Workers':>8}  {'MaxBatch':>9}  {'AvgBatch':>9}  "
      f"{'time(s)':>8}  {'sims/s':>8}  {'speedup':>8}")
print(f"  {'-'*60}")

baseline_mcts = None

def play_n_moves(net_predictor, n_moves=10, n_sims=20):
    """Play n_moves with n_sims MCTS sims each, return total sim count."""
    mcts_cfg = MCTSConfig(num_simulations=n_sims, temperature=1.0)
    mcts = MCTS(net_predictor, encoder, mcts_cfg)
    g = GameState()
    total = 0
    for _ in range(n_moves):
        if g.result != GameResult.IN_PROGRESS:
            break
        policy = mcts.search(g)
        total += n_sims
        action = int(np.argmax(policy))
        moves = g.legal_moves()
        mask = encoder.get_legal_action_mask(g, moves)
        if mask[action] > 0:
            g.apply_move(encoder.decode_action(action, g))
        else:
            legal = np.where(mask > 0)[0]
            if len(legal) == 0:
                break
            g.apply_move(encoder.decode_action(int(legal[0]), g))
    return total

for num_workers in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
    max_batch = num_workers
    server = BatchedInferenceServer(
        net=net,
        collate_fn=HiveTokenBatch.collate,
        device=device,
        max_batch_size=max_batch,
        max_wait_ms=5.0,
    )
    server.start()
    batched_net = BatchedPredictor(server)

    barrier = threading.Barrier(num_workers)
    sim_counts = [0] * num_workers

    def mcts_worker(worker_id):
        barrier.wait()
        sim_counts[worker_id] = play_n_moves(batched_net)

    threads = [
        threading.Thread(target=mcts_worker, args=(i,))
        for i in range(num_workers)
    ]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.perf_counter() - t0

    total_sims = sum(sim_counts)
    avg_batch = server._total_items / server._total_batches if server._total_batches else 0
    sims_per_sec = total_sims / elapsed
    server.stop()

    if baseline_mcts is None:
        baseline_mcts = sims_per_sec
    speedup = sims_per_sec / baseline_mcts

    print(f"  {num_workers:>8}  {max_batch:>9}  {avg_batch:>9.1f}  "
          f"{elapsed:>8.2f}  {sims_per_sec:>8.0f}  {speedup:>8.1f}x")

print("\nDone.")
