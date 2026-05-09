from __future__ import annotations

import numpy as np

from hive_prs.action_space import MAX_BOARD
from hive_prs.slot_map import HEIGHT_OFFSET, MAX_STACK, NUM_CELLS, NEIGHBORS


def articulation_cells_from_state(state_bytes: np.ndarray) -> set[int]:
    heights = state_bytes[HEIGHT_OFFSET:HEIGHT_OFFSET + NUM_CELLS]
    occ = [cell for cell in range(NUM_CELLS) if int(heights[cell]) > 0]
    if len(occ) <= 2:
        return set()
    occ_set = set(occ)
    adj = {c: [int(nb) for nb in NEIGHBORS[c] if nb >= 0 and int(nb) in occ_set] for c in occ}

    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int] = {}
    aps: set[int] = set()
    t = 0

    def dfs(u: int) -> None:
        nonlocal t
        children = 0
        t += 1
        disc[u] = low[u] = t
        for v in adj[u]:
            if v not in disc:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if u not in parent and children > 1:
                    aps.add(u)
                if u in parent and low[v] >= disc[u]:
                    aps.add(u)
            elif parent.get(u) != v:
                low[u] = min(low[u], disc[v])

    dfs(occ[0])
    return aps


def compute_articulation_target(
    state_bytes: np.ndarray,
    max_board_tokens: int = MAX_BOARD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-board-token articulation labels and mask.

    Token order matches the CUDA state encoder's board-token order:
    cells ascending, then stack levels bottom->top within each cell.

    A token is labeled 1 only if it is the top piece on a ground-level
    single-piece stack whose removal would break one-hive.
    """
    heights = state_bytes[HEIGHT_OFFSET:HEIGHT_OFFSET + NUM_CELLS]
    aps = articulation_cells_from_state(state_bytes)
    target = np.zeros(max_board_tokens, dtype=np.float32)
    mask = np.zeros(max_board_tokens, dtype=np.float32)

    idx = 0
    for cell in range(NUM_CELLS):
        h = int(heights[cell])
        if h == 0:
            continue
        for level in range(h):
            if idx >= max_board_tokens:
                return target, mask
            mask[idx] = 1.0
            if h == 1 and level == 0 and cell in aps:
                target[idx] = 1.0
            idx += 1
    return target, mask
