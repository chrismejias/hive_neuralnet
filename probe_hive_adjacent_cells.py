"""Measure max hive-adjacent empty cells across random positions.

Plays random legal moves from many initial states with all expansion pieces
enabled (mask=7) and records the count of empty cells adjacent to at least
one occupied cell at every ply.
"""
from __future__ import annotations

import numpy as np
import torch

import hive_gpu

BOARD_SIZE = 23
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
HEIGHT_OFFSET = 5 * NUM_CELLS  # pieces[5][529] comes first
DIR_DCOL = np.array([+1, +1, 0, -1, -1, 0], dtype=np.int32)
DIR_DROW = np.array([0, -1, -1, 0, +1, +1], dtype=np.int32)


def build_neighbor_table() -> np.ndarray:
    tbl = np.full((NUM_CELLS, 6), -1, dtype=np.int32)
    for cell in range(NUM_CELLS):
        r, c = divmod(cell, BOARD_SIZE)
        for d in range(6):
            nc = c + int(DIR_DCOL[d])
            nr = r + int(DIR_DROW[d])
            if 0 <= nc < BOARD_SIZE and 0 <= nr < BOARD_SIZE:
                tbl[cell, d] = nr * BOARD_SIZE + nc
    return tbl


def count_hive_adjacent_empty(heights: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """heights: (B, NUM_CELLS) uint8. Returns (B,) int counts."""
    occ = heights > 0  # (B, N)
    # gather neighbor occupancy; -1 neighbors map to a sentinel (0)
    pad = np.concatenate([occ, np.zeros((occ.shape[0], 1), dtype=bool)], axis=1)
    nbr_idx = np.where(nbr < 0, NUM_CELLS, nbr)  # (N, 6)
    has_occ_nbr = pad[:, nbr_idx].any(axis=2)  # (B, N)
    empty = ~occ
    return (empty & has_occ_nbr).sum(axis=1)


def main() -> None:
    ext = hive_gpu.load_extension()
    ext.initialize_tables()

    BATCH = 1000
    MAX_PLIES = 160
    EXP_MASK = 7

    nbr = build_neighbor_table()
    states = ext.create_initial_states(BATCH, EXP_MASK)
    device = states.device

    max_count = 0
    sum_count = 0
    n_samples = 0
    hist: dict[int, int] = {}
    alive = torch.ones(BATCH, dtype=torch.bool, device=device)

    for ply in range(MAX_PLIES):
        legal, nlegal = ext.generate_legal_moves_batch(states, BATCH)

        heights = states[:, HEIGHT_OFFSET:HEIGHT_OFFSET + NUM_CELLS].cpu().numpy()
        counts = count_hive_adjacent_empty(heights, nbr)
        alive_cpu = alive.cpu().numpy()
        for c, a in zip(counts, alive_cpu):
            if not a:
                continue
            max_count = max(max_count, int(c))
            sum_count += int(c)
            n_samples += 1
            hist[int(c)] = hist.get(int(c), 0) + 1

        nlegal_cpu = nlegal.cpu().numpy()
        chosen = np.zeros(BATCH, dtype=np.int64)
        for i in range(BATCH):
            n = int(nlegal_cpu[i])
            if n == 0:
                alive[i] = False
                chosen[i] = 0
            else:
                chosen[i] = np.random.randint(n)
        chosen_t = torch.from_numpy(chosen).to(device)
        moves = legal[torch.arange(BATCH, device=device), chosen_t]
        ext.apply_moves_batch(states, moves, BATCH)

        finished = ext.check_results_batch(states, BATCH)
        alive &= (finished == 0)
        if not alive.any():
            break

    print(f"Samples: {n_samples}")
    print(f"Max hive-adjacent empty cells: {max_count}")
    print(f"Mean: {sum_count / max(n_samples, 1):.2f}")
    print("\nHistogram (count -> frequency):")
    for k in sorted(hist):
        bar = "#" * min(hist[k] // max(1, n_samples // 50), 60)
        print(f"  {k:3d}: {hist[k]:6d} {bar}")

    cum = 0
    print("\nPercentiles:")
    for pct in [50, 75, 90, 95, 99, 99.5, 99.9, 100]:
        target = n_samples * pct / 100
        acc = 0
        for k in sorted(hist):
            acc += hist[k]
            if acc >= target:
                print(f"  p{pct}: {k}")
                break


if __name__ == "__main__":
    main()
