"""
Compare legal move generation before and after MoveGenCache optimization.

Measures:
1. Placement positions: board.valid_placement_positions() vs cache
2. Ant moves: board._ant_bfs() (HexCoord) vs cache.get_ant_moves() (tuple BFS)
3. Full legal_moves() end-to-end (includes cache overhead)
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from hive_engine.game_state import GameState, GameResult, MoveType
from hive_engine.pieces import Color, PieceType
from hive_engine.move_gen import MoveGenCache


def collect_game_states(num_games: int = 40, max_moves: int = 100) -> list[GameState]:
    """Play random games and collect mid-game states."""
    states = []
    rng = random.Random(12345)
    for g in range(num_games):
        gs = GameState()
        for _ in range(max_moves):
            if gs.result != GameResult.IN_PROGRESS:
                break
            moves = gs.legal_moves()
            if not moves:
                break
            move = rng.choice(moves)
            gs.apply_move(move)
            if gs.turn >= 8:
                states.append(gs.copy())
        if (g + 1) % 10 == 0:
            print(f"  Collected {len(states)} states from {g+1} games...", flush=True)
    return states


def time_fn(fn, n_reps: int = 50) -> float:
    """Return mean time in microseconds."""
    start = time.perf_counter()
    for _ in range(n_reps):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / n_reps * 1e6


def main():
    print("Collecting game states...", flush=True)
    states = collect_game_states(num_games=40, max_moves=100)
    print(f"Collected {len(states)} states total.\n")

    # ---- Placement comparison ----
    print("=" * 70)
    print("PLACEMENT POSITIONS: old vs cached")
    print("=" * 70)

    old_placement_times = []
    cached_placement_times = []

    for gs in states:
        if gs.turn < 2:
            continue
        board = gs.board
        color = gs.current_player

        # Old: full recompute every time
        old_t = time_fn(lambda: board.valid_placement_positions(color))
        old_placement_times.append(old_t)

        # Cached: lazy init on first call, then cached
        cache = MoveGenCache()
        # First call initializes
        cache.get_placement_positions(board, color)
        # Subsequent calls return cached
        cached_t = time_fn(lambda: cache.get_placement_positions(board, color))
        cached_placement_times.append(cached_t)

    print(f"  Old (full recompute):   mean={np.mean(old_placement_times):.1f}us, "
          f"median={np.median(old_placement_times):.1f}us")
    print(f"  Cached (after init):    mean={np.mean(cached_placement_times):.1f}us, "
          f"median={np.median(cached_placement_times):.1f}us")
    if np.mean(cached_placement_times) > 0:
        print(f"  Speedup: {np.mean(old_placement_times) / max(np.mean(cached_placement_times), 0.01):.0f}x")

    # Measure incremental update cost (notify + query)
    print("\n  Incremental update cost (notify_place + query):")
    incr_times = []
    for gs in states[:200]:
        if gs.turn < 4:
            continue
        board = gs.board
        color = gs.current_player
        cache = MoveGenCache()
        cache.get_placement_positions(board, color)

        # Simulate a placement notification
        from hive_engine.hex_coord import HexCoord, ORIGIN
        from hive_engine.pieces import Piece
        test_pos = ORIGIN
        test_piece = Piece(PieceType.ANT, Color.WHITE, 0)

        def incr_update():
            cache._update_placement_around(board, {test_pos})
        t = time_fn(incr_update, n_reps=100)
        incr_times.append(t)
    print(f"    mean={np.mean(incr_times):.1f}us, median={np.median(incr_times):.1f}us")

    # ---- Ant movement comparison ----
    print("\n" + "=" * 70)
    print("ANT MOVES: HexCoord BFS vs tuple BFS")
    print("=" * 70)

    old_ant_times = []
    cached_ant_times = []
    total_ant_count = 0

    for gs in states:
        board = gs.board
        color = gs.current_player
        if not gs._queen_placed[color]:
            continue

        aps = board.find_articulation_points()

        for piece in board.pieces_of_color(color):
            if piece.piece_type != PieceType.ANT:
                continue
            pos = board.position_of(piece)
            if pos is None or not board.is_on_top(piece):
                continue
            if pos in aps and board.stack_height(pos) == 1:
                continue

            total_ant_count += 1

            # Old: direct HexCoord BFS
            old_t = time_fn(lambda p=pos: board._ant_bfs(p))
            old_ant_times.append(old_t)

            # New: tuple BFS via cache
            cache = gs._move_gen_cache
            cached_t = time_fn(lambda p=pos: cache.get_ant_moves(board, p))
            cached_ant_times.append(cached_t)

    if old_ant_times:
        print(f"  Old (HexCoord BFS):     mean={np.mean(old_ant_times):.1f}us, "
              f"median={np.median(old_ant_times):.1f}us")
        print(f"  New (tuple BFS):        mean={np.mean(cached_ant_times):.1f}us, "
              f"median={np.median(cached_ant_times):.1f}us")
        print(f"  Speedup: {np.mean(old_ant_times) / max(np.mean(cached_ant_times), 0.01):.1f}x")
        print(f"  Total ants profiled: {total_ant_count}")

    # ---- Full legal_moves() comparison ----
    print("\n" + "=" * 70)
    print("FULL legal_moves(): end-to-end comparison")
    print("=" * 70)

    full_times = []
    for gs in states[:500]:
        def full_legal():
            gs._legal_moves_cache = None
            gs.legal_moves()
        t = time_fn(full_legal, n_reps=20)
        full_times.append(t)

    print(f"  legal_moves() with cache: mean={np.mean(full_times):.1f}us, "
          f"median={np.median(full_times):.1f}us, "
          f"p95={np.percentile(full_times, 95):.1f}us")

    # Compare: disable both caches and measure
    # Use a dummy cache that always uses the old methods
    class OldStyleCache:
        """Shim cache that uses old-style methods (no optimization)."""
        def get_placement_positions(self, board, color):
            return board.valid_placement_positions(color)
        def get_ant_moves(self, board, pos):
            return board._ant_bfs(pos)
        def notify_place(self, *a): pass
        def notify_move(self, *a): pass
        def invalidate(self): pass
        def copy(self): return OldStyleCache()

    old_full_times = []
    for gs in states[:500]:
        real_cache = gs._move_gen_cache

        def old_legal():
            gs._legal_moves_cache = None
            gs._move_gen_cache = OldStyleCache()
            gs.legal_moves()
            gs._move_gen_cache = real_cache

        t = time_fn(old_legal, n_reps=20)
        old_full_times.append(t)

    print(f"  legal_moves() without cache: mean={np.mean(old_full_times):.1f}us, "
          f"median={np.median(old_full_times):.1f}us, "
          f"p95={np.percentile(old_full_times, 95):.1f}us")
    print(f"  End-to-end speedup: {np.mean(old_full_times) / max(np.mean(full_times), 0.01):.2f}x")

    # Board size distribution
    print(f"\n  Board sizes: P25={np.percentile([len(gs.board.grid) for gs in states], 25):.0f}, "
          f"P50={np.percentile([len(gs.board.grid) for gs in states], 50):.0f}, "
          f"P75={np.percentile([len(gs.board.grid) for gs in states], 75):.0f}")


if __name__ == "__main__":
    main()
