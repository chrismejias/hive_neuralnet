"""
Benchmark: Generate 10,000 random Hive games to verify correctness and measure performance.

Tracks:
  - Games completed vs draws vs timeouts
  - Average game length
  - Average branching factor
  - Move generation speed
  - Any errors/crashes
"""

import random
import time
import sys
from collections import Counter

from hive_engine.game_state import GameState, GameResult, MoveType


def run_random_game(seed: int, max_turns: int = 200) -> dict:
    """Play a single random game and return statistics."""
    random.seed(seed)
    gs = GameState()

    total_moves_available = 0
    move_gen_time = 0.0
    turn_count = 0

    for turn in range(max_turns):
        if gs.result != GameResult.IN_PROGRESS:
            break

        t0 = time.perf_counter()
        moves = gs.legal_moves()
        t1 = time.perf_counter()

        move_gen_time += (t1 - t0)
        total_moves_available += len(moves)
        turn_count += 1

        move = random.choice(moves)
        gs.apply_move(move)

    return {
        "seed": seed,
        "result": gs.result,
        "turns": gs.turn,
        "avg_branching": total_moves_available / max(turn_count, 1),
        "move_gen_time": move_gen_time,
        "timed_out": gs.result == GameResult.IN_PROGRESS,
    }


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    print(f"Running {num_games} random Hive games...")
    print("=" * 60)

    results = Counter()
    total_turns = 0
    total_branching = 0.0
    total_time = 0.0
    errors = 0
    max_branching = 0.0
    min_branching = float("inf")

    overall_start = time.perf_counter()

    for i in range(num_games):
        try:
            stats = run_random_game(seed=i)
            results[stats["result"]] += 1
            total_turns += stats["turns"]
            total_branching += stats["avg_branching"]
            total_time += stats["move_gen_time"]
            max_branching = max(max_branching, stats["avg_branching"])
            min_branching = min(min_branching, stats["avg_branching"])

            if (i + 1) % 1000 == 0:
                elapsed = time.perf_counter() - overall_start
                rate = (i + 1) / elapsed
                print(f"  [{i+1:>5}/{num_games}] {rate:.1f} games/sec | "
                      f"elapsed: {elapsed:.1f}s")

        except Exception as e:
            errors += 1
            print(f"  ERROR in game {i} (seed={i}): {e}")
            if errors > 10:
                print("Too many errors, aborting.")
                break

    overall_elapsed = time.perf_counter() - overall_start

    print("=" * 60)
    print(f"RESULTS ({num_games} games)")
    print("=" * 60)
    print(f"  White wins:  {results[GameResult.WHITE_WINS]:>6}")
    print(f"  Black wins:  {results[GameResult.BLACK_WINS]:>6}")
    print(f"  Draws:       {results[GameResult.DRAW]:>6}")
    print(f"  Timed out:   {results[GameResult.IN_PROGRESS]:>6} (hit {200} turn limit)")
    print(f"  Errors:      {errors:>6}")
    print()
    completed = num_games - errors
    if completed > 0:
        print(f"  Avg turns/game:      {total_turns / completed:.1f}")
        print(f"  Avg branching factor: {total_branching / completed:.1f}")
        print(f"  Min branching factor: {min_branching:.1f}")
        print(f"  Max branching factor: {max_branching:.1f}")
        print(f"  Avg move gen time:   {total_time / completed * 1000:.2f} ms/game")
        print(f"  Total wall time:     {overall_elapsed:.1f}s")
        print(f"  Throughput:          {completed / overall_elapsed:.1f} games/sec")

    print()
    if errors == 0:
        print("ALL GAMES COMPLETED WITHOUT ERRORS")
    else:
        print(f"WARNING: {errors} games had errors!")

    return errors == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
