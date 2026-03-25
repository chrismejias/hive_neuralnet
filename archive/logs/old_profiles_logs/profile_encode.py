"""
Fine-grained profiler for TokenEncoder.encode().

Instruments every logical section within encode() using perf_counter
to find exactly where the 0.5ms per call is spent.

Run from hive_neuralnet/:
    .venv/Scripts/python profile_encode.py
"""

import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")

from hive_engine.board import Board
from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, GameResult
from hive_engine.hex_coord import ALL_DIRECTIONS, HexCoord, _OFFSET_LIST
from hive_engine.pieces import Color, PieceType
from hive_transformer.token_types import (
    TOKEN_FEAT_DIM, GLOBAL_FEAT_DIM,
    TOKEN_TYPE_CLS, TOKEN_TYPE_BOARD, TOKEN_TYPE_HAND,
    OFF_BOARD_POSITION, HiveTokenSequence,
)
from hive_transformer.token_encoder import TokenEncoder

import random
random.seed(42)

# ── Build game states at different lengths ────────────────────────────────────
def make_state(n_moves):
    g = GameState()
    for _ in range(n_moves):
        m = g.legal_moves()
        if not m or g.result != GameResult.IN_PROGRESS:
            break
        g.apply_move(random.choice(m))
    return g

early = make_state(10)
mid   = make_state(30)
late  = make_state(80)

print(f"States: early={len(early.board.grid)} positions, "
      f"mid={len(mid.board.grid)} positions, "
      f"late={len(late.board.grid)} positions")
print()

# ── Instrumented encode() ─────────────────────────────────────────────────────
def encode_instrumented(game_state: GameState, enc: TokenEncoder):
    """Run encode() with per-section timing. Returns dict of times in us."""
    t = {}
    board = game_state.board

    t0 = time.perf_counter()
    center_q, center_r = enc._cached_center(board)
    t["center"] = (time.perf_counter() - t0) * 1e6

    features_list = []
    positions_list = []
    types_list = []
    num_board_tokens = 0

    # CLS token
    t0 = time.perf_counter()
    features_list.append(np.zeros(TOKEN_FEAT_DIM, dtype=np.float32))
    positions_list.append(OFF_BOARD_POSITION)
    types_list.append(TOKEN_TYPE_CLS)
    t["cls"] = (time.perf_counter() - t0) * 1e6

    # Per-piece costs
    t_hex2grid = 0.0
    t_npzeros = 0.0
    t_feat_basic = 0.0
    t_queen_count = 0.0
    t_occ_neighbors = 0.0
    t_empty_mask = 0.0
    t_append = 0.0

    for pos, stack in board.grid.items():
        t0 = time.perf_counter()
        grid = HiveEncoder._hex_to_grid(pos.q, pos.r, center_q, center_r)
        t_hex2grid += time.perf_counter() - t0
        if grid is None:
            continue

        row, col = grid
        stack_height = len(stack)

        for height, piece in enumerate(stack):
            is_top = (height == stack_height - 1)

            t0 = time.perf_counter()
            feat = np.zeros(TOKEN_FEAT_DIM, dtype=np.float32)
            t_npzeros += time.perf_counter() - t0

            t0 = time.perf_counter()
            feat[piece.piece_type.value] = 1.0
            feat[5 + piece.color.value] = 1.0
            feat[7] = 1.0 if height == 0 else 0.0
            feat[8] = 1.0 if is_top else 0.0
            feat[9] = stack_height / 4.0
            feat[10] = 1.0 if piece.piece_type == PieceType.QUEEN else 0.0
            t_feat_basic += time.perf_counter() - t0

            t0 = time.perf_counter()
            q_count = game_state.queen_surrounded_count(piece.color)
            t_queen_count += time.perf_counter() - t0

            t0 = time.perf_counter()
            occ = board.num_occupied_neighbors(pos)
            t_occ_neighbors += time.perf_counter() - t0

            feat[11] = q_count / 6.0
            feat[12] = occ / 6.0

            t0 = time.perf_counter()
            if is_top:
                for d in ALL_DIRECTIONS:
                    neighbor = pos.neighbor(d)
                    if neighbor not in board.grid:
                        feat[13 + d.value] = 1.0
            t_empty_mask += time.perf_counter() - t0

            feat[19] = 0.0
            feat[20] = 0.0
            feat[21] = height / 4.0

            t0 = time.perf_counter()
            features_list.append(feat)
            positions_list.append(row * 13 + col)
            types_list.append(TOKEN_TYPE_BOARD)
            t_append += time.perf_counter() - t0
            num_board_tokens += 1

    t["hex2grid"] = t_hex2grid * 1e6
    t["npzeros"]  = t_npzeros * 1e6
    t["feat_basic"] = t_feat_basic * 1e6
    t["queen_count"] = t_queen_count * 1e6
    t["occ_neighbors"] = t_occ_neighbors * 1e6
    t["empty_mask"] = t_empty_mask * 1e6
    t["append"] = t_append * 1e6

    # Hand tokens
    t0 = time.perf_counter()
    for color in (Color.WHITE, Color.BLACK):
        hand = game_state.hand(color)
        type_counts: dict[PieceType, int] = defaultdict(int)
        for p in hand:
            type_counts[p.piece_type] += 1
        for pt, count in type_counts.items():
            feat = np.zeros(TOKEN_FEAT_DIM, dtype=np.float32)
            feat[pt.value] = 1.0
            feat[5 + color.value] = 1.0
            feat[10] = 1.0 if pt == PieceType.QUEEN else 0.0
            feat[19] = 1.0
            feat[20] = count / pt.count_per_player
            features_list.append(feat)
            positions_list.append(OFF_BOARD_POSITION)
            types_list.append(TOKEN_TYPE_HAND)
    t["hand_tokens"] = (time.perf_counter() - t0) * 1e6

    # Assembly: np.stack + np.array
    t0 = time.perf_counter()
    token_features = np.stack(features_list, axis=0)
    t["np_stack"] = (time.perf_counter() - t0) * 1e6

    t0 = time.perf_counter()
    token_positions = np.array(positions_list, dtype=np.int32)
    token_types = np.array(types_list, dtype=np.int32)
    t["np_arrays"] = (time.perf_counter() - t0) * 1e6

    return t, num_board_tokens


# ── Run warm-up then measure ──────────────────────────────────────────────────
enc = TokenEncoder()
N = 2000

for label, gs in [("early (10 moves)", early),
                  ("mid   (30 moves)", mid),
                  ("late  (80 moves)", late)]:
    # Warm up
    for _ in range(100):
        encode_instrumented(gs, enc)

    totals = defaultdict(float)
    n_pieces = None
    for _ in range(N):
        times, nb = encode_instrumented(gs, enc)
        n_pieces = nb
        for k, v in times.items():
            totals[k] += v

    total_us = sum(totals.values())
    print(f"{'='*60}")
    print(f"{label}  |  {n_pieces} board tokens  |  {total_us/N:.0f} us/call total")
    print(f"{'='*60}")
    print(f"  {'Section':<20} {'us/call':>8}  {'%':>6}  {'us/token':>9}")
    print(f"  {'-'*50}")

    order = [
        "center", "cls", "hex2grid", "npzeros", "feat_basic",
        "queen_count", "occ_neighbors", "empty_mask", "append",
        "hand_tokens", "np_stack", "np_arrays",
    ]
    for k in order:
        us = totals[k] / N
        pct = 100 * us / total_us if total_us > 0 else 0
        per_tok = us / n_pieces if n_pieces else 0
        print(f"  {k:<20} {us:>8.1f}  {pct:>5.1f}%  {per_tok:>9.3f}")

    print(f"  {'TOTAL':<20} {total_us/N:>8.1f}  {'100.0%':>6}")
    print()

# ── Micro-benchmark individual expensive ops ──────────────────────────────────
print("="*60)
print("Micro-benchmarks of hot sub-operations")
print("="*60)

gs = mid
board = gs.board
pos = next(iter(board.grid))

# HexCoord construction
N2 = 100_000
t0 = time.perf_counter()
for _ in range(N2):
    _ = HexCoord(3, -2)
us_hexcoord = (time.perf_counter() - t0) / N2 * 1e6
print(f"  HexCoord(q, r):              {us_hexcoord:.4f} us/call")

# pos.neighbor(d)
t0 = time.perf_counter()
d = ALL_DIRECTIONS[0]
for _ in range(N2):
    _ = pos.neighbor(d)
us_neighbor = (time.perf_counter() - t0) / N2 * 1e6
print(f"  pos.neighbor(d):             {us_neighbor:.4f} us/call")

# HexCoord in dict
t0 = time.perf_counter()
hc = HexCoord(0, 0)
for _ in range(N2):
    _ = hc in board.grid
us_lookup = (time.perf_counter() - t0) / N2 * 1e6
print(f"  HexCoord in board.grid:      {us_lookup:.4f} us/call")

# np.zeros(22)
t0 = time.perf_counter()
for _ in range(N2):
    _ = np.zeros(22, dtype=np.float32)
us_npz = (time.perf_counter() - t0) / N2 * 1e6
print(f"  np.zeros(22, float32):       {us_npz:.4f} us/call")

# queen_surrounded_count
t0 = time.perf_counter()
N3 = 20_000
for _ in range(N3):
    _ = gs.queen_surrounded_count(Color.WHITE)
us_qsc = (time.perf_counter() - t0) / N3 * 1e6
print(f"  queen_surrounded_count():    {us_qsc:.4f} us/call")

# num_occupied_neighbors
t0 = time.perf_counter()
for _ in range(N3):
    _ = board.num_occupied_neighbors(pos)
us_occ = (time.perf_counter() - t0) / N3 * 1e6
print(f"  num_occupied_neighbors():    {us_occ:.4f} us/call")

# np.stack of 30 arrays of 22 floats
feats = [np.zeros(22, dtype=np.float32) for _ in range(30)]
t0 = time.perf_counter()
N4 = 10_000
for _ in range(N4):
    _ = np.stack(feats, axis=0)
us_stack = (time.perf_counter() - t0) / N4 * 1e6
print(f"  np.stack(30 x float32[22]): {us_stack:.4f} us/call")

print()
print("Done.")
