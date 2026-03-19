"""
Profile the batched inference pipeline to measure:
  1. encode_state() cost (CPU, GIL-bound)
  2. GPU forward pass cost vs batch size
  3. BatchedInferenceServer round-trip time and actual batch sizes
     at different inference_wait_ms settings

Run from hive_neuralnet/:
    .venv/Scripts/python profile_batching.py
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
from hive_transformer.transformer_net import HiveTransformer, TransformerConfig
from hive_transformer.token_types import HiveTokenBatch
from hive_engine.batched_inference import BatchedInferenceServer, BatchedPredictor
from hive_engine.device import get_device

device = get_device()
print(f"Device: {device}")

# Load checkpoint 0035 for a realistic model
ckpt_path = "checkpoints_transformer/hive_transformer_checkpoint_0035.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
net_config = ckpt["net_config"]
net = HiveTransformer(net_config).to(device)
net.load_state_dict(ckpt["net_state_dict"], strict=False)
net.eval()
print(f"Loaded checkpoint 0035: d_model={net_config.d_model}, layers={net_config.num_layers}")

encoder = TransformerEncoder()

# ── 1. How long does encode_state() take? ────────────────────────────────────
print("\n" + "="*60)
print("1. encode_state() cost (transformer tokenisation)")
print("="*60)

# Build a mid-game state (~20 pieces on board)
gs = GameState()
import random
random.seed(42)
for _ in range(20):
    moves = gs.legal_moves()
    if not moves or gs.result != GameResult.IN_PROGRESS:
        break
    gs.apply_move(random.choice(moves))

N = 200
t0 = time.perf_counter()
for _ in range(N):
    seq = encoder.encode_state(gs)
elapsed = time.perf_counter() - t0
print(f"  encode_state(): {elapsed/N*1000:.2f} ms/call  (avg over {N} calls)")
print(f"  Tokens in sequence: {seq.num_board_tokens} board + metadata")

# ── 2. GPU forward pass vs batch size ─────────────────────────────────────────
print("\n" + "="*60)
print("2. GPU forward pass time vs batch size")
print("="*60)

seq = encoder.encode_state(gs)
mask = encoder.get_legal_action_mask(gs)

for B in [1, 2, 4, 8, 16, 32, 64]:
    batch = HiveTokenBatch.collate([seq] * B).to(device)
    masks = torch.stack([torch.from_numpy(mask)] * B).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = net.forward(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    N = 20
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = net.forward(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_call = elapsed / N * 1000
    ms_per_item = ms_per_call / B
    print(f"  Batch size {B:3d}: {ms_per_call:6.2f} ms/call  = {ms_per_item:.3f} ms/item"
          f"  ({B/ms_per_call*1000:.0f} items/sec)")

# ── 3. BatchedInferenceServer: actual batch sizes at different wait_ms ────────
print("\n" + "="*60)
print("3. BatchedInferenceServer: batch sizes vs wait_ms and num_workers")
print("="*60)

def run_batch_test(num_workers: int, wait_ms: float, num_requests: int = 200):
    """Run N requests from W threads through BatchedInferenceServer, measure batch stats."""
    server = BatchedInferenceServer(
        net=net,
        collate_fn=HiveTokenBatch.collate,
        device=device,
        max_batch_size=32,
        max_wait_ms=wait_ms,
    )
    server.start()
    batched_net = BatchedPredictor(server)

    # Prepare states
    game_states = []
    gs_tmp = GameState()
    random.seed(0)
    for i in range(num_requests):
        moves = gs_tmp.legal_moves()
        if moves and gs_tmp.result == GameResult.IN_PROGRESS:
            gs_tmp = GameState()
            for _ in range(min(i % 30, len(gs_tmp.legal_moves()))):
                m = gs_tmp.legal_moves()
                if m and gs_tmp.result == GameResult.IN_PROGRESS:
                    gs_tmp.apply_move(random.choice(m))
        game_states.append(gs_tmp)

    # Pre-encode all states (to isolate encode time from server measurement)
    encoded = [encoder.encode_state(g) for g in game_states]
    masks = [encoder.get_legal_action_mask(g) for g in game_states]

    # Distribute requests across workers
    requests_per_worker = [[] for _ in range(num_workers)]
    for i, (enc, msk) in enumerate(zip(encoded, masks)):
        requests_per_worker[i % num_workers].append((enc, msk))

    results = []
    barrier = threading.Barrier(num_workers)

    def worker_fn(my_requests):
        barrier.wait()  # start all workers at same time
        for enc, msk in my_requests:
            probs, val = batched_net.predict(enc, msk)

    threads = [
        threading.Thread(target=worker_fn, args=(reqs,))
        for reqs in requests_per_worker
    ]

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    total_batches = server._total_batches
    total_items = server._total_items
    avg_batch = total_items / total_batches if total_batches > 0 else 0
    server.stop()

    throughput = num_requests / elapsed
    return avg_batch, elapsed, throughput


print(f"\n  {'Workers':>8} {'wait_ms':>8} {'avg_batch':>10} {'time(s)':>8} {'items/s':>8}")
print(f"  {'-'*50}")

for num_workers in [1, 4, 8, 16, 32]:
    for wait_ms in [5.0, 10.0, 30.0, 64.0]:
        avg_batch, elapsed, tput = run_batch_test(
            num_workers=num_workers,
            wait_ms=wait_ms,
            num_requests=160,  # divisible by 32
        )
        print(f"  {num_workers:>8} {wait_ms:>8.0f} {avg_batch:>10.1f} {elapsed:>8.2f} {tput:>8.1f}")
    print()

# ── 4. End-to-end: simulate MCTS search across workers ────────────────────────
print("\n" + "="*60)
print("4. End-to-end: MCTS search timing (5 sims) with varying workers")
print("   Measures total round-trip including encode + GIL + GPU wait")
print("="*60)

def run_mcts_test(num_workers: int, wait_ms: float, num_games: int = 32, num_sims: int = 5):
    """Run num_games in parallel, each doing num_sims MCTS steps per move for 5 moves."""
    server = BatchedInferenceServer(
        net=net,
        collate_fn=HiveTokenBatch.collate,
        device=device,
        max_batch_size=32,
        max_wait_ms=wait_ms,
    )
    server.start()
    batched_net = BatchedPredictor(server)

    mcts_config = MCTSConfig(num_simulations=num_sims, temperature=0.0)
    barrier = threading.Barrier(num_workers)

    def play_5_moves():
        barrier.wait()
        g = GameState()
        mcts = MCTS(batched_net, encoder, mcts_config)
        for _ in range(5):
            if g.result != GameResult.IN_PROGRESS:
                break
            policy = mcts.search(g)
            action = int(np.argmax(policy))
            moves = g.legal_moves()
            mask = encoder.get_legal_action_mask(g, moves)
            if mask[action] > 0:
                move = encoder.decode_action(action, g)
                g.apply_move(move)

    threads = [threading.Thread(target=play_5_moves) for _ in range(num_games)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    total_batches = server._total_batches
    avg_batch = server._total_items / total_batches if total_batches > 0 else 0
    server.stop()

    # sims done = num_games * 5 moves * num_sims (plus root expansion)
    return elapsed, avg_batch

print(f"\n  {'Workers':>8} {'wait_ms':>8} {'time(s)':>8} {'avg_batch':>10}")
print(f"  {'-'*40}")
for num_workers in [1, 4, 8, 16, 32]:
    for wait_ms in [10.0, 64.0]:
        elapsed, avg_batch = run_mcts_test(num_workers, wait_ms, num_games=num_workers, num_sims=10)
        print(f"  {num_workers:>8} {wait_ms:>8.0f} {elapsed:>8.2f} {avg_batch:>10.1f}")
    print()

print("\nDone.")
