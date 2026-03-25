"""
Profile legal move generation to identify bottlenecks.

Samples real game states from self-play games, then times each
component of legal_moves() independently.

Components measured:
  - find_articulation_points   (One Hive rule for movement)
  - valid_placement_positions  (placement filter)
  - ant BFS                    (generate_slides for each ant)
  - spider walk                (generate_slides for each spider)
  - queen slides               (generate_slides max_distance=1)
  - grasshopper moves
  - beetle moves
  - full legal_moves()         (end-to-end)
"""

import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from hive_engine.game_state import GameState, GameResult
from hive_engine.pieces import Color, PieceType
from hive_engine.mcts import MCTS, MCTSConfig


# ── Collect real game states via short self-play ──────────────────


def collect_game_states(num_games: int = 30, max_moves: int = 80) -> list[GameState]:
    """Play random games (no net, uniform policy) and collect states."""
    states = []
    for g in range(num_games):
        gs = GameState()
        for _ in range(max_moves):
            if gs.result != GameResult.IN_PROGRESS:
                break
            moves = gs.legal_moves()
            if not moves:
                break
            move = moves[np.random.randint(len(moves))]
            gs = gs.copy()
            gs.apply_move(move)
            # Only collect mid-game states (skip first 4 moves, last few)
            if gs.turn >= 8:
                states.append(gs.copy())
        if (g + 1) % 10 == 0:
            print(f"  Collected {len(states)} states from {g+1} games...", flush=True)
    return states


# ── Timing helpers ────────────────────────────────────────────────


def time_fn(fn, n_reps: int = 50) -> float:
    """Return mean time in microseconds over n_reps calls."""
    start = time.perf_counter()
    for _ in range(n_reps):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / n_reps * 1e6  # microseconds


# ── Per-component profiling ───────────────────────────────────────


def profile_state(gs: GameState) -> dict:
    """Time each legal-move component for a single game state."""
    board = gs.board
    color = gs.current_player
    results = {}

    # 1. Articulation points
    results["articulation_points"] = time_fn(board.find_articulation_points)

    # 2. Placement positions
    if gs.turn >= 2:
        results["placement_positions"] = time_fn(
            lambda: board.valid_placement_positions(color)
        )
    else:
        results["placement_positions"] = 0.0

    # 3. Per-piece movement generation
    aps = board.find_articulation_points()
    ant_times, spider_times, queen_times, grass_times, beetle_times = [], [], [], [], []

    for piece in board.pieces_of_color(color):
        pos = board.position_of(piece)
        if pos is None or not board.is_on_top(piece):
            continue
        if pos in aps and board.stack_height(pos) == 1:
            continue  # pinned

        pt = piece.piece_type
        if pt == PieceType.ANT:
            ant_times.append(time_fn(lambda p=piece: board.generate_slides(p, max_distance=-1)))
        elif pt == PieceType.SPIDER:
            spider_times.append(time_fn(lambda p=piece: board.generate_slides(p, max_distance=3)))
        elif pt == PieceType.QUEEN:
            queen_times.append(time_fn(lambda p=piece: board.generate_slides(p, max_distance=1)))
        elif pt == PieceType.GRASSHOPPER:
            grass_times.append(time_fn(lambda p=piece: board.generate_grasshopper_moves(p)))
        elif pt == PieceType.BEETLE:
            beetle_times.append(time_fn(lambda p=piece: board.generate_beetle_moves(p)))

    results["ant_per_piece"]         = np.mean(ant_times)    if ant_times    else 0.0
    results["ant_count"]             = len(ant_times)
    results["spider_per_piece"]      = np.mean(spider_times) if spider_times else 0.0
    results["spider_count"]          = len(spider_times)
    results["queen_per_piece"]       = np.mean(queen_times)  if queen_times  else 0.0
    results["queen_count"]           = len(queen_times)
    results["grasshopper_per_piece"] = np.mean(grass_times)  if grass_times  else 0.0
    results["grasshopper_count"]     = len(grass_times)
    results["beetle_per_piece"]      = np.mean(beetle_times) if beetle_times else 0.0
    results["beetle_count"]          = len(beetle_times)

    # 4. Full legal_moves() (cache cleared each call)
    def full_legal():
        gs._legal_moves_cache = None
        gs.legal_moves()

    results["full_legal_moves"] = time_fn(full_legal, n_reps=20)

    # Board size for context
    results["board_size"] = len(board.grid)

    return results


# ── Main ──────────────────────────────────────────────────────────


def main():
    print("Collecting game states...", flush=True)
    states = collect_game_states(num_games=40, max_moves=100)
    print(f"Collected {len(states)} states total.\n")

    print("Profiling legal move generation...", flush=True)
    all_results = []
    for i, gs in enumerate(states):
        r = profile_state(gs)
        all_results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  Profiled {i+1}/{len(states)}...", flush=True)

    print(f"\nProfiled {len(all_results)} states.\n")

    # ── Aggregate stats ───────────────────────────────────────────

    def stats(key):
        vals = [r[key] for r in all_results if r[key] > 0]
        if not vals:
            return 0, 0, 0
        return np.mean(vals), np.median(vals), np.percentile(vals, 95)

    print(f"{'Component':<30}  {'Mean (us)':>10}  {'Median (us)':>11}  {'P95 (us)':>9}  {'Note'}")
    print("-" * 80)

    # Fixed-cost components
    for key, label, note in [
        ("articulation_points", "Articulation points",     "once per legal_moves()"),
        ("placement_positions", "Placement positions",     "once per legal_moves()"),
        ("full_legal_moves",    "Full legal_moves()",      "end-to-end"),
    ]:
        m, med, p95 = stats(key)
        print(f"{label:<30}  {m:>10.1f}  {med:>11.1f}  {p95:>9.1f}  {note}")

    print()

    # Per-piece movement costs
    for key, count_key, label in [
        ("ant_per_piece",         "ant_count",         "Ant BFS (per ant)"),
        ("spider_per_piece",      "spider_count",      "Spider walk (per spider)"),
        ("queen_per_piece",       "queen_count",       "Queen slide (per queen)"),
        ("grasshopper_per_piece", "grasshopper_count", "Grasshopper jump (per)"),
        ("beetle_per_piece",      "beetle_count",      "Beetle move (per beetle)"),
    ]:
        m, med, p95 = stats(key)
        avg_count = np.mean([r[count_key] for r in all_results])
        total_contribution = m * avg_count
        print(f"{label:<30}  {m:>10.1f}  {med:>11.1f}  {p95:>9.1f}  "
              f"avg {avg_count:.1f} movable => ~{total_contribution:.0f}us total")

    # Summary: estimated total from piece moves
    print()
    print("Board size distribution:")
    sizes = [r["board_size"] for r in all_results]
    for p in [25, 50, 75, 95]:
        print(f"  P{p:2d}: {np.percentile(sizes, p):.0f} pieces on board")

    # Which piece type contributes most total time?
    print()
    print("Estimated total movement time per legal_moves() call:")
    for key, count_key, label in [
        ("ant_per_piece",         "ant_count",         "Ant"),
        ("spider_per_piece",      "spider_count",      "Spider"),
        ("queen_per_piece",       "queen_count",       "Queen"),
        ("grasshopper_per_piece", "grasshopper_count", "Grasshopper"),
        ("beetle_per_piece",      "beetle_count",      "Beetle"),
    ]:
        m, _, _ = stats(key)
        avg_count = np.mean([r[count_key] for r in all_results])
        print(f"  {label:<14}: {m:.1f}us x {avg_count:.1f} = {m * avg_count:.1f}us")


if __name__ == "__main__":
    main()
