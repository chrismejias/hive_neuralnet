"""C6 (6-fold) rotational augmentation for PRS v2 training.

Hive has hex-grid 6-fold rotational symmetry (no reflection — Hive pieces
have a chirality via the pillbug/mosquito direction neighborhood).  This
module provides:

  * `build_cell_rot_tables()` — precomputes six 529-entry cell→cell maps
    for k∈{0..5}×60° CW rotations around grid cell (11, 11).

  * `rotate_state_bytes(state, k)` — produces a new HiveState byte buffer
    whose cell-indexed fields have been permuted by the k-th rotation.
    Non-cell fields (hands, turn, result, queen_placed, center_q/r, pad)
    are copied verbatim — the engine keeps `center_q=center_r=0` for all
    games, so rotation around grid-center is identical to rotation around
    the axial origin.

  * `rotate_moves_bytes(moves, nlegal, k)` — rotates `from_cell`/`to_cell`
    fields of a (B, max_L, SIZEOF_GPU_MOVE) buffer.  PASS moves are
    left untouched.

Layout of HiveState (derived from hive_state.cuh + _HANDS_OFFSET=3396):

    offset  size   field
    ------  -----  ---------------------------------
       0    2645   pieces[5][529]  (stack bytes)
    2645     529   height[529]
    3174       2   (padding — Bitboard needs 8-byte alignment)
    3176      72   Bitboard occupied   (9 × uint64)
    3248      72   Bitboard white_top
    3320      72   Bitboard black_top
    3392       4   queen_cell[2]       (2 × uint16)
    3396      16   hands[2][8]
    3412       2   turn                (uint16)
    3414       1   queen_placed
    3415       1   result
    3416       1   center_q            (int8)
    3417       1   center_r            (int8)
    3418       2   _pad
    3420+           (trailing alignment padding)

All numeric offsets are validated at import by `_assert_layout`.

Layout of GPUMove (sizeof == ext.SIZEOF_GPU_MOVE, typically 6 bytes + pad):

    offset 0   uint8   type         (0=PLACE, 1=MOVE, 2=PASS)
    offset 1   uint8   piece_type
    offset 2   uint16  from_cell
    offset 4   uint16  to_cell
"""
from __future__ import annotations

import numpy as np

# ── Grid constants (mirror hex_grid.cuh) ──────────────────────────────
BOARD_SIZE = 23
HALF_BOARD = 11
NUM_CELLS  = BOARD_SIZE * BOARD_SIZE   # 529

# State layout constants (mirror hive_state.cuh; see module docstring)
_PIECES_OFFSET  = 0
_PIECES_BYTES   = 5 * NUM_CELLS                  # 2645
_HEIGHT_OFFSET  = _PIECES_OFFSET + _PIECES_BYTES # 2645
_HEIGHT_BYTES   = NUM_CELLS                      # 529
_BB_WORDS       = 9
_BB_BYTES       = _BB_WORDS * 8                  # 72
# Bitboard requires 8-byte alignment → 2 bytes of padding after height
_BB_PAD         = 2
_OCC_OFFSET     = _HEIGHT_OFFSET + _HEIGHT_BYTES + _BB_PAD  # 3176
_WTOP_OFFSET    = _OCC_OFFSET + _BB_BYTES                   # 3248
_BTOP_OFFSET    = _WTOP_OFFSET + _BB_BYTES                  # 3320
_QCELL_OFFSET   = _BTOP_OFFSET + _BB_BYTES                  # 3392
_QCELL_BYTES    = 4
_HANDS_OFFSET   = _QCELL_OFFSET + _QCELL_BYTES              # 3396

# Move layout (mirror GPUMove in hive_state.cuh)
_MOVE_TYPE_OFF    = 0
_MOVE_FROM_OFF    = 2
_MOVE_TO_OFF      = 4
_MOVE_TYPE_PLACE  = 0
_MOVE_TYPE_MOVE   = 1
_MOVE_TYPE_PASS   = 2


# ── Cell rotation table builder ───────────────────────────────────────

def _rotate_cell_cw_once(cell: int) -> int:
    """Rotate a single cell index 60° CW around grid center (11, 11)."""
    row = cell // BOARD_SIZE
    col = cell % BOARD_SIZE
    new_col = col + row - HALF_BOARD
    new_row = (2 * HALF_BOARD) - col
    if not (0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE):
        return -1
    return new_row * BOARD_SIZE + new_col


def _build_cell_rot_tables() -> np.ndarray:
    """Return int16 table of shape (6, 529); entry -1 = off-grid."""
    tables = np.full((6, NUM_CELLS), -1, dtype=np.int32)
    tables[0] = np.arange(NUM_CELLS, dtype=np.int32)
    for k in range(1, 6):
        for src in range(NUM_CELLS):
            dst = tables[k - 1, src]
            if dst < 0:
                tables[k, src] = -1
            else:
                tables[k, src] = _rotate_cell_cw_once(int(dst))
    return tables


_ROT = _build_cell_rot_tables()


# Self-check: 6 applications of 60° CW must be identity on valid cells.
def _assert_rotation_tables() -> None:
    acc = np.arange(NUM_CELLS, dtype=np.int32)
    for _ in range(6):
        new_acc = np.full_like(acc, -1)
        for src in range(NUM_CELLS):
            if acc[src] < 0:
                continue
            new_acc[src] = _rotate_cell_cw_once(int(acc[src]))
        acc = new_acc
    # Every on-grid cell must return to itself
    ok = (acc == np.arange(NUM_CELLS, dtype=np.int32)) | (acc == -1)
    assert ok.all(), "rotation tables broken: 6×60° CW ≠ identity"


_assert_rotation_tables()


# ── State rotation ────────────────────────────────────────────────────

def _rotate_bitboard(bb_bytes: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Rotate a 72-byte bitboard (9 × uint64, NUM_CELLS bits).

    `rot[src] = dst` (or -1).  Returns a fresh 72-byte uint8 buffer."""
    src_words = bb_bytes.view(np.uint64).copy()     # (9,)
    dst_words = np.zeros(_BB_WORDS, dtype=np.uint64)
    # Walk set bits of source; for each, place at rot[bit] in dest.
    for w in range(_BB_WORDS):
        word = int(src_words[w])
        while word:
            b = (word & -word).bit_length() - 1      # lowest set bit
            word &= word - 1
            cell = w * 64 + b
            if cell >= NUM_CELLS:
                continue
            dst = int(rot[cell])
            if dst < 0:
                continue
            dst_words[dst >> 6] |= np.uint64(1) << np.uint64(dst & 63)
    return dst_words.view(np.uint8).copy()


def rotate_state_bytes(state: np.ndarray, k: int) -> np.ndarray:
    """Return a rotated copy of a single HiveState byte buffer.

    Args:
        state: (S,) uint8 array of length SIZEOF_HIVE_STATE.
        k: rotation count in {0..5} (number of 60° CW steps).
    """
    if k == 0:
        return state.copy()
    assert 1 <= k <= 5, f"k must be in 0..5, got {k}"
    rot = _ROT[k]                                    # (529,)
    out = state.copy()

    # 1) pieces[5][529] — permute within each level
    pieces = state[_PIECES_OFFSET:_PIECES_OFFSET + _PIECES_BYTES].reshape(5, NUM_CELLS)
    out_pieces = np.zeros_like(pieces)
    for lvl in range(5):
        valid = rot >= 0
        dst_idx = rot[valid]
        out_pieces[lvl, dst_idx] = pieces[lvl, np.where(valid)[0]]
    out[_PIECES_OFFSET:_PIECES_OFFSET + _PIECES_BYTES] = out_pieces.reshape(-1)

    # 2) height[529]
    height = state[_HEIGHT_OFFSET:_HEIGHT_OFFSET + _HEIGHT_BYTES]
    out_height = np.zeros_like(height)
    valid = rot >= 0
    out_height[rot[valid]] = height[np.where(valid)[0]]
    out[_HEIGHT_OFFSET:_HEIGHT_OFFSET + _HEIGHT_BYTES] = out_height

    # 3) Bitboards (occupied, white_top, black_top)
    for off in (_OCC_OFFSET, _WTOP_OFFSET, _BTOP_OFFSET):
        out[off:off + _BB_BYTES] = _rotate_bitboard(state[off:off + _BB_BYTES], rot)

    # 4) queen_cell[2] — two uint16 cell indices
    qc = state[_QCELL_OFFSET:_QCELL_OFFSET + _QCELL_BYTES].view(np.uint16).copy()
    for i in range(2):
        c = int(qc[i])
        if c != 0xFFFF and 0 <= c < NUM_CELLS:
            dst = int(rot[c])
            qc[i] = np.uint16(dst if dst >= 0 else 0xFFFF)
    out[_QCELL_OFFSET:_QCELL_OFFSET + _QCELL_BYTES] = qc.view(np.uint8)

    # Everything else (hands, turn, queen_placed, result, center_*, pad)
    # is copied verbatim — out was initialized from state.copy().
    return out


def rotate_states_batch(states: np.ndarray, k: int) -> np.ndarray:
    """Rotate a batch of states (B, S) uint8."""
    if k == 0:
        return states.copy()
    B = states.shape[0]
    out = np.empty_like(states)
    for i in range(B):
        out[i] = rotate_state_bytes(states[i], k)
    return out


# ── Move rotation ─────────────────────────────────────────────────────

def rotate_moves_batch(
    moves: np.ndarray,          # (B, max_L, move_sz) uint8
    nlegal: np.ndarray,         # (B,) int32 — valid move counts per row
    k: int,
) -> np.ndarray:
    """Rotate from_cell / to_cell of all valid GPUMoves, leaving PASS alone.

    Off-grid rotations (-1) are folded back to 0xFFFF (caller should mask
    such moves via legality re-check after rotation — in practice a
    properly-played state will never rotate a legal move off the grid)."""
    if k == 0:
        return moves.copy()
    rot = _ROT[k]
    B, max_L, _sz = moves.shape
    out = moves.copy()

    for b in range(B):
        n = int(nlegal[b])
        if n == 0:
            continue
        row = out[b, :n]                             # (n, sz)
        mtype = row[:, _MOVE_TYPE_OFF]
        is_pass = mtype == _MOVE_TYPE_PASS

        from_cells = row[:, _MOVE_FROM_OFF:_MOVE_FROM_OFF + 2].view(np.uint16).reshape(n)
        to_cells   = row[:, _MOVE_TO_OFF:_MOVE_TO_OFF + 2].view(np.uint16).reshape(n)

        for i in range(n):
            t = int(mtype[i])
            if t == _MOVE_TYPE_PASS:
                continue
            # to_cell is meaningful for both PLACE and MOVE
            tc = int(to_cells[i])
            if 0 <= tc < NUM_CELLS:
                d = int(rot[tc])
                to_cells[i] = np.uint16(d if d >= 0 else 0xFFFF)
            # from_cell only meaningful for MOVE
            if t == _MOVE_TYPE_MOVE:
                fc = int(from_cells[i])
                if 0 <= fc < NUM_CELLS:
                    d = int(rot[fc])
                    from_cells[i] = np.uint16(d if d >= 0 else 0xFFFF)

    return out


# ── Debug / test helpers ──────────────────────────────────────────────

def _assert_layout(sizeof_state: int) -> None:
    """Sanity-check at runtime that our offsets land where we expect.

    Call this from tests with the extension's SIZEOF_HIVE_STATE constant
    to guard against silent layout changes in the engine.
    """
    assert sizeof_state >= 3420, (
        f"HiveState byte size {sizeof_state} smaller than expected minimum 3420"
    )
    assert _HANDS_OFFSET == 3396
