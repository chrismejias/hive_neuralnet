"""
Data augmentation via hexagonal rotational symmetry.

Exploits the 6-fold rotational symmetry of the hex board to multiply
each training example by 6x. Works at the tensor/action-index level
for efficiency (no GameState manipulation needed).

The 13x13 grid is centered at (6,6). Hex rotation CW 60° maps
cube coords (dq, dr, ds) -> (-ds, -dq, -dr). Applied to the grid:
  col = dq + 6, row = dr + 6  ->  col' = -ds + 6 = dq + dr + 6,
                                  row' = -dq + 6

Pre-computed lookup tables map grid indices and action indices through
each of the 6 rotations (0°, 60°, 120°, 180°, 240°, 300°).

Usage:
    augmented = augment_example(state_tensor, policy, value)
    # Returns list of 6 TrainingExamples (including original)
"""

from __future__ import annotations

import numpy as np

# -- Constants (must match encoder.py) ------------------------------------------

BOARD_SIZE = 13
HALF_BOARD = 6
NUM_GRID_CELLS = BOARD_SIZE * BOARD_SIZE  # 169
NUM_PIECE_TYPES = 5
NUM_PIECE_CHANNELS = 20  # Channels 0-19 are spatial piece data
NUM_META_CHANNELS = 6    # Channels 20-25 are scalar meta data
NUM_CHANNELS = 26

NUM_PLACEMENT_ACTIONS = NUM_PIECE_TYPES * NUM_GRID_CELLS  # 845
MOVEMENT_OFFSET = NUM_PLACEMENT_ACTIONS  # 845
ACTION_SPACE_SIZE = NUM_PLACEMENT_ACTIONS + NUM_GRID_CELLS * NUM_GRID_CELLS + 1  # 29407
PASS_ACTION_INDEX = ACTION_SPACE_SIZE - 1  # 29406


# -- Grid rotation --------------------------------------------------------------

def _rotate_grid_cw(row: int, col: int) -> tuple[int, int]:
    """
    Rotate a grid position 60° clockwise around center (6, 6).

    Grid <-> hex: col = dq + 6, row = dr + 6, ds = -dq - dr.
    CW 60°: (dq, dr, ds) -> (-ds, -dq, -dr).
    So: new_col = -ds + 6 = dq + dr + 6, new_row = -dq + 6.
    """
    dq = col - HALF_BOARD
    dr = row - HALF_BOARD
    # CW rotation
    new_col = dq + dr + HALF_BOARD
    new_row = -dq + HALF_BOARD
    return (new_row, new_col)


def _build_grid_rotation_tables() -> list[np.ndarray]:
    """
    Build lookup tables for grid index rotation.

    Returns a list of 6 arrays (one per rotation k=0..5).
    Each array has shape (169,) mapping flat grid index -> rotated flat grid index.
    Out-of-bounds entries are set to -1.
    """
    tables = []
    for k in range(6):
        table = np.full(NUM_GRID_CELLS, -1, dtype=np.int32)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                r, c = row, col
                for _ in range(k):
                    r, c = _rotate_grid_cw(r, c)
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    src_idx = row * BOARD_SIZE + col
                    dst_idx = r * BOARD_SIZE + c
                    table[src_idx] = dst_idx
        tables.append(table)
    return tables


def _build_policy_rotation_tables(
    grid_tables: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Build lookup tables for action index rotation.

    Returns a list of 6 arrays (one per rotation k=0..5).
    Each array has shape (ACTION_SPACE_SIZE,) mapping action -> rotated action.
    Actions that map out-of-bounds are set to -1.
    """
    tables = []
    for k in range(6):
        gt = grid_tables[k]
        table = np.full(ACTION_SPACE_SIZE, -1, dtype=np.int32)

        # Placement actions: piece_type * 169 + grid_pos
        for pt in range(NUM_PIECE_TYPES):
            for pos in range(NUM_GRID_CELLS):
                src_action = pt * NUM_GRID_CELLS + pos
                new_pos = gt[pos]
                if new_pos >= 0:
                    table[src_action] = pt * NUM_GRID_CELLS + new_pos

        # Movement actions: 845 + src_pos * 169 + dst_pos
        for src in range(NUM_GRID_CELLS):
            new_src = gt[src]
            if new_src < 0:
                continue
            for dst in range(NUM_GRID_CELLS):
                new_dst = gt[dst]
                if new_dst < 0:
                    continue
                src_action = MOVEMENT_OFFSET + src * NUM_GRID_CELLS + dst
                dst_action = MOVEMENT_OFFSET + new_src * NUM_GRID_CELLS + new_dst
                table[src_action] = dst_action

        # Pass action maps to itself
        table[PASS_ACTION_INDEX] = PASS_ACTION_INDEX

        tables.append(table)
    return tables


# Pre-compute tables at module load time
_GRID_TABLES: list[np.ndarray] = _build_grid_rotation_tables()
_POLICY_TABLES: list[np.ndarray] = _build_policy_rotation_tables(_GRID_TABLES)


# -- Augmentation functions -----------------------------------------------------

def rotate_state(state_tensor: np.ndarray, k: int) -> np.ndarray:
    """
    Rotate a state tensor by k x 60° clockwise.

    Args:
        state_tensor: Shape (26, 13, 13), float32.
        k: Number of 60° CW rotations (0-5).

    Returns:
        Rotated tensor of same shape.
    """
    if k == 0:
        return state_tensor.copy()

    gt = _GRID_TABLES[k]
    result = np.zeros_like(state_tensor)

    # Rotate spatial piece channels (0-19)
    for ch in range(NUM_PIECE_CHANNELS):
        for src_idx in range(NUM_GRID_CELLS):
            dst_idx = gt[src_idx]
            if dst_idx >= 0:
                src_row, src_col = divmod(src_idx, BOARD_SIZE)
                dst_row, dst_col = divmod(dst_idx, BOARD_SIZE)
                result[ch, dst_row, dst_col] = state_tensor[ch, src_row, src_col]

    # Copy meta channels unchanged (20-25)
    result[NUM_PIECE_CHANNELS:] = state_tensor[NUM_PIECE_CHANNELS:]

    return result


def rotate_policy(policy: np.ndarray, k: int) -> np.ndarray:
    """
    Rotate a policy vector by k x 60° clockwise.

    Args:
        policy: Shape (29407,), probability distribution.
        k: Number of 60° CW rotations (0-5).

    Returns:
        Rotated and renormalized policy of same shape.
    """
    if k == 0:
        return policy.copy()

    pt = _POLICY_TABLES[k]
    result = np.zeros_like(policy)

    for src_action in range(ACTION_SPACE_SIZE):
        if policy[src_action] > 0:
            dst_action = pt[src_action]
            if dst_action >= 0:
                result[dst_action] += policy[src_action]
            # else: out-of-bounds -> probability lost, will be renormalized

    # Renormalize
    total = result.sum()
    if total > 0:
        result /= total

    return result


def augment_example(
    state_tensor: np.ndarray,
    policy_target: np.ndarray,
    value_target: float,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    Produce 6 rotational variants of a training example.

    Args:
        state_tensor: Shape (26, 13, 13).
        policy_target: Shape (29407,).
        value_target: Scalar value in [-1, 1].

    Returns:
        List of 6 (state, policy, value) tuples, including the original at k=0.
    """
    results = []
    for k in range(6):
        rot_state = rotate_state(state_tensor, k)
        rot_policy = rotate_policy(policy_target, k)
        results.append((rot_state, rot_policy, value_target))
    return results
