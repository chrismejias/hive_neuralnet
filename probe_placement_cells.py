"""Measure max distinct placement destination cells per state.

For each state reached during random play, counts the number of unique
to_cell values across all MOVE_PLACE legal moves (current player's
placement destinations only).
"""
from __future__ import annotations

import numpy as np
import torch

import hive_gpu

SIZEOF_MOVE = 6  # bytes per GpuMove (see SIZEOF_GPU_MOVE)
# GpuMove layout (see move_gen.cuh): type(1) + piece_type(1) + from_cell(2) + to_cell(2)
MOVE_PLACE = 0


def main() -> None:
    ext = hive_gpu.load_extension()
    ext.initialize_tables()

    BATCH = 1000
    MAX_PLIES = 160
    EXP_MASK = 7

    states = ext.create_initial_states(BATCH, EXP_MASK)
    device = states.device

    max_count = 0
    sum_count = 0
    n_samples = 0
    hist: dict[int, int] = {}
    # Split tracking by "one hand piece left" scenario
    hist_late: dict[int, int] = {}
    alive = torch.ones(BATCH, dtype=torch.bool, device=device)

    for ply in range(MAX_PLIES):
        legal, nlegal = ext.generate_legal_moves_batch(states, BATCH)
        # legal: (B, MAX_LEGAL_MOVES, 6) uint8
        legal_cpu = legal.cpu().numpy()
        nlegal_cpu = nlegal.cpu().numpy()
        alive_cpu = alive.cpu().numpy()

        # hand counts are in state bytes — layout: pieces[5][529] + height[529] + ... + hands
        # Simpler: count moves with type==MOVE_PLACE and count distinct to_cells
        for i in range(BATCH):
            if not alive_cpu[i]:
                continue
            n = int(nlegal_cpu[i])
            moves = legal_cpu[i, :n]
            place_mask = moves[:, 0] == MOVE_PLACE
            place_moves = moves[place_mask]
            if place_moves.shape[0] == 0:
                continue
            # to_cell = bytes[4] + bytes[5]<<8
            to_cells = place_moves[:, 4].astype(np.int32) + (place_moves[:, 5].astype(np.int32) << 8)
            distinct = int(np.unique(to_cells).size)
            max_count = max(max_count, distinct)
            sum_count += distinct
            n_samples += 1
            hist[distinct] = hist.get(distinct, 0) + 1

            # "Late" = only placements, no moves (would indicate few placements remain),
            # or equivalently: number of distinct placement piece types <=1
            place_types = place_moves[:, 1]
            if int(np.unique(place_types).size) == 1:
                hist_late[distinct] = hist_late.get(distinct, 0) + 1

        # Random action
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

    print(f"Samples (states with >=1 placement): {n_samples}")
    print(f"Max distinct placement cells: {max_count}")
    print(f"Mean: {sum_count / max(n_samples, 1):.2f}")
    print("\nHistogram:")
    for k in sorted(hist):
        print(f"  {k:3d}: {hist[k]:6d}")

    cum = 0
    total = sum(hist.values())
    print("\nPercentiles:")
    acc = 0
    pcts = [50, 90, 95, 99, 99.9, 100]
    idx = 0
    for k in sorted(hist):
        acc += hist[k]
        while idx < len(pcts) and acc >= total * pcts[idx] / 100:
            print(f"  p{pcts[idx]}: {k}")
            idx += 1

    print(f"\n'Only one placement-type' scenarios: {sum(hist_late.values())}")
    if hist_late:
        print(f"  Max: {max(hist_late)}")
        print(f"  Mean: {sum(k*v for k,v in hist_late.items()) / sum(hist_late.values()):.2f}")


if __name__ == "__main__":
    main()
