"""Validate PRS v2 slot mapping across random play with all expansions.

Checks:
  1. Every legal move maps to a valid slot in [0, 813).
  2. Within a state, distinct legal moves map to distinct slots.
  3. Slot occupancy stats: how often each block is hit.
  4. Never exceeds C_MOVE=64 move cells or C_HAND=32 place cells.
"""
from __future__ import annotations

import numpy as np
import torch

import hive_gpu
from hive_prs.slot_map import (
    DIR_OFFSET, THROW_OFFSET, LONG_OFFSET, HAND_OFFSET, PASS_SLOT, N_SLOTS,
    C_MOVE, C_HAND, map_legal_moves,
)

BATCH = 200
MAX_PLIES = 160
EXP_MASK = 7


def block_name(slot: int) -> str:
    if slot == PASS_SLOT:
        return "PASS"
    if slot >= HAND_OFFSET:
        return "HAND"
    if slot >= LONG_OFFSET:
        return "LONG"
    if slot >= THROW_OFFSET:
        return "THROW"
    return "DIR"


def main() -> None:
    ext = hive_gpu.load_extension()
    ext.initialize_tables()

    states = ext.create_initial_states(BATCH, EXP_MASK)
    device = states.device

    total_moves = 0
    unmapped = 0
    collisions = 0
    max_move_cells = 0
    max_place_cells = 0
    block_hits = {"DIR": 0, "THROW": 0, "LONG": 0, "HAND": 0, "PASS": 0}
    alive = torch.ones(BATCH, dtype=torch.bool, device=device)
    unmapped_examples: list = []

    for ply in range(MAX_PLIES):
        legal, nlegal = ext.generate_legal_moves_batch(states, BATCH)
        states_cpu = states.cpu().numpy()
        legal_cpu = legal.cpu().numpy()
        nlegal_cpu = nlegal.cpu().numpy()
        alive_cpu = alive.cpu().numpy()

        for b in range(BATCH):
            if not alive_cpu[b]:
                continue
            n = int(nlegal_cpu[b])
            if n == 0:
                continue
            slots, mcells, pcells = map_legal_moves(states_cpu[b], legal_cpu[b, :n], n)
            max_move_cells = max(max_move_cells, len(mcells))
            max_place_cells = max(max_place_cells, len(pcells))
            total_moves += n

            for i, s in enumerate(slots):
                if s < 0:
                    unmapped += 1
                    if len(unmapped_examples) < 20:
                        unmapped_examples.append(
                            (int(legal_cpu[b, i, 0]), int(legal_cpu[b, i, 1]),
                             int(legal_cpu[b, i, 2]) | (int(legal_cpu[b, i, 3]) << 8),
                             int(legal_cpu[b, i, 4]) | (int(legal_cpu[b, i, 5]) << 8))
                        )
                else:
                    block_hits[block_name(int(s))] += 1

            valid_slots = slots[slots >= 0]
            if len(np.unique(valid_slots)) != len(valid_slots):
                collisions += 1

        # Random step
        chosen = np.zeros(BATCH, dtype=np.int64)
        for i in range(BATCH):
            n = int(nlegal_cpu[i])
            if n == 0:
                alive[i] = False
            else:
                chosen[i] = np.random.randint(n)
        chosen_t = torch.from_numpy(chosen).to(device)
        moves = legal[torch.arange(BATCH, device=device), chosen_t]
        ext.apply_moves_batch(states, moves, BATCH)
        finished = ext.check_results_batch(states, BATCH)
        alive &= (finished == 0)
        if not alive.any():
            break

    print(f"Total legal moves examined: {total_moves}")
    print(f"Unmapped moves: {unmapped}")
    print(f"States with slot-collisions: {collisions}")
    print(f"Max move_cells observed: {max_move_cells} (cap {C_MOVE})")
    print(f"Max place_cells observed: {max_place_cells} (cap {C_HAND})")
    print(f"\nBlock hits:")
    tot = sum(block_hits.values())
    for k, v in block_hits.items():
        pct = 100 * v / max(tot, 1)
        print(f"  {k:5s}: {v:8d}  ({pct:5.1f}%)")

    if unmapped_examples:
        print(f"\nFirst unmapped moves (type, piece, from, to):")
        for ex in unmapped_examples:
            print(f"  type={ex[0]} pt={ex[1]} from={ex[2]} to={ex[3]}")


if __name__ == "__main__":
    main()
