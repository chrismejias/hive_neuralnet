"""
PRS v2 policy-head slot mapping (Python reference).

Maps each legal move to one of 813 structured slots:

  [   0 :  48 )  dir    : piece (Q1, B1-2, G1-3, P1, M1) × 6 dirs
  [  48 : 108 )  throw  : thrower (P1, M1) × 30 throw-slots
  [ 108 : 556 )  p→cell : piece (A1-3, S1-2, L1, M1) × 64 move-cell tokens
  [ 556 : 812 )  hand   : piece_type (8) × 32 place-cell tokens
  [ 812 : 813 )  PASS

Mosquito classification:
  * tc is neighbor of fc (any occupancy) → dir slot (covers M-as-B climb,
    M-as-Q, M-as-P own-move, M-as-B descend). Direction is fc→tc.
  * tc is far (not neighbor) → long head (covers M-as-A/S/L/G).
  * pillbug throw (detected first by common pillbug-capable neighbor) →
    throw head.

This splits the ambiguity: "same direction via different inherited
abilities" can never co-occur because their destinations differ (neighbor
vs far).

Also emits per-state cell-token enumerations:
  * move_cells (≤64): empty cells adjacent to any occupied cell
  * place_cells (≤32): empty cells adjacent to ≥1 own-color cell AND
                       not adjacent to any opponent-color cell

This module is the ground-truth reference; a CUDA port will replace the
hot path later. For now it runs on CPU from the legal_moves tensor.
"""
from __future__ import annotations

import numpy as np

BOARD_SIZE = 23
NUM_CELLS = BOARD_SIZE * BOARD_SIZE  # 529
NUM_DIRS = 6

# Piece type enum (matches hive_state.cuh)
PT_QUEEN, PT_ANT, PT_GRASSHOPPER, PT_SPIDER = 1, 2, 3, 4
PT_BEETLE, PT_MOSQUITO, PT_LADYBUG, PT_PILLBUG = 5, 6, 7, 8

# Move types
MOVE_PLACE, MOVE_MOVE, MOVE_PASS = 0, 1, 2

# Slot block offsets
DIR_OFFSET   = 0
THROW_OFFSET = 48
LONG_OFFSET  = 108
HAND_OFFSET  = 556
PASS_SLOT    = 812
N_SLOTS      = 813

C_MOVE = 64   # cap on move cell tokens
C_HAND = 32   # cap on placement cell tokens

# ── Direction tables ──────────────────────────────────────────────────
DIR_DCOL = np.array([+1, +1, 0, -1, -1, 0], dtype=np.int32)
DIR_DROW = np.array([0, -1, -1, 0, +1, +1], dtype=np.int32)
DIR_DELTA = DIR_DROW * BOARD_SIZE + DIR_DCOL


def _build_neighbor_table() -> np.ndarray:
    tbl = np.full((NUM_CELLS, NUM_DIRS), -1, dtype=np.int32)
    for cell in range(NUM_CELLS):
        r, c = divmod(cell, BOARD_SIZE)
        for d in range(NUM_DIRS):
            nc = c + int(DIR_DCOL[d])
            nr = r + int(DIR_DROW[d])
            if 0 <= nc < BOARD_SIZE and 0 <= nr < BOARD_SIZE:
                tbl[cell, d] = nr * BOARD_SIZE + nc
    return tbl


NEIGHBORS = _build_neighbor_table()


def direction_of(from_cell: int, to_cell: int) -> int:
    """Return hex direction 0..5 from from_cell to to_cell if adjacent, else -1."""
    for d in range(NUM_DIRS):
        if NEIGHBORS[from_cell, d] == to_cell:
            return d
    return -1


def grasshopper_dir_of(occ: np.ndarray, fc: int, tc: int) -> int:
    """Return direction d if a grasshopper jump from fc in direction d lands on tc.
    occ: (NUM_CELLS,) bool — current occupancy (fc is treated as occupied)."""
    for d in range(NUM_DIRS):
        # Walk from fc in direction d, skip occupied, land on first empty
        cur = NEIGHBORS[fc, d]
        if cur < 0 or not occ[cur]:
            continue  # must start jumping over ≥1 piece
        while cur >= 0 and occ[cur]:
            cur = NEIGHBORS[cur, d]
        if cur == tc:
            return d
    return -1


# ── Canonical instance-index mappings ──────────────────────────────────

# Within-color instance index assignment (for current player's 14 pieces):
#   0: Q1
#   1: B1, 2: B2
#   3: G1, 4: G2, 5: G3
#   6: P1
#   7: M1
#   8: A1, 9: A2, 10: A3
#   11: S1, 12: S2
#   13: L1
# Pieces are ranked within (type) by ascending cell index among in-play pieces
# of current player.

# Mapping from (piece_type, type_instance) → dir_piece_idx [0..7]
_DIR_BASE = {PT_QUEEN: 0, PT_BEETLE: 1, PT_GRASSHOPPER: 3,
             PT_PILLBUG: 6, PT_MOSQUITO: 7}
# Mapping → long_piece_idx [0..6]
_LONG_BASE = {PT_ANT: 0, PT_SPIDER: 3, PT_LADYBUG: 5, PT_MOSQUITO: 6}
# Mapping → throw_piece_idx [0..1]
_THROW_BASE = {PT_PILLBUG: 0, PT_MOSQUITO: 1}


def _instance_idx(top_cells: list[int], fc: int) -> int:
    """Rank of fc in ascending sort of top_cells. Returns -1 if fc not in list."""
    sorted_cells = sorted(top_cells)
    for i, c in enumerate(sorted_cells):
        if c == fc:
            return i
    return -1


# ── State decoding helpers ─────────────────────────────────────────────

HEIGHT_OFFSET = 5 * NUM_CELLS  # in HiveState bytes
MAX_STACK = 5


def decode_heights(state_bytes: np.ndarray) -> np.ndarray:
    """Return (NUM_CELLS,) uint8 height array for a single state."""
    return state_bytes[HEIGHT_OFFSET:HEIGHT_OFFSET + NUM_CELLS]


def decode_top_colors_and_types(state_bytes: np.ndarray):
    """Return (top_color, top_type) arrays of shape (NUM_CELLS,).
    top_color: 0=white 1=black, -1 if empty.
    top_type: PieceType 1..8, 0 if empty."""
    heights = decode_heights(state_bytes)
    pieces = state_bytes[:5 * NUM_CELLS].reshape(MAX_STACK, NUM_CELLS)
    top_color = np.full(NUM_CELLS, -1, dtype=np.int32)
    top_type = np.zeros(NUM_CELLS, dtype=np.int32)
    for cell in range(NUM_CELLS):
        h = int(heights[cell])
        if h == 0:
            continue
        packed = int(pieces[h - 1, cell])
        top_type[cell] = packed & 0x0F
        top_color[cell] = (packed >> 4) & 0x01
    return top_color, top_type, heights


TURN_OFFSET = 3412  # empirically verified; accounts for uint16 alignment padding


def current_color_from_turn(state_bytes: np.ndarray) -> int:
    """Return 0 (white) or 1 (black) from state's turn counter (uint16 LE)."""
    turn = int(state_bytes[TURN_OFFSET]) | (int(state_bytes[TURN_OFFSET + 1]) << 8)
    return turn & 1


# ── Per-color in-play piece enumeration (for instance-idx) ─────────────

def pieces_in_play_by_type(top_color: np.ndarray, top_type: np.ndarray,
                            color: int) -> dict[int, list[int]]:
    """Return {piece_type: [cells]} for top-pieces of given color, sorted ascending."""
    out = {pt: [] for pt in range(1, 9)}
    cells = np.where((top_color == color) & (top_type > 0))[0]
    for cell in cells:
        out[int(top_type[cell])].append(int(cell))
    for pt in out:
        out[pt].sort()
    return out


# ── Cell-token enumerations ────────────────────────────────────────────

def enumerate_move_cells(heights: np.ndarray) -> np.ndarray:
    """Return sorted ascending list of empty cells adjacent to any occupied cell.
    Capped at C_MOVE (extras dropped — should never happen per probe)."""
    occ = heights > 0
    out = []
    for cell in range(NUM_CELLS):
        if occ[cell]:
            continue
        for d in range(NUM_DIRS):
            nb = NEIGHBORS[cell, d]
            if nb >= 0 and occ[nb]:
                out.append(cell)
                break
    if len(out) > C_MOVE:
        out = out[:C_MOVE]
    return np.asarray(out, dtype=np.int32)


def enumerate_place_cells(top_color: np.ndarray, heights: np.ndarray,
                           color: int) -> np.ndarray:
    """Return ascending list of empty cells adjacent to ≥1 own-color piece AND
    not adjacent to any opponent-color piece. Capped at C_HAND."""
    occ = heights > 0
    opp = 1 - color
    out = []
    for cell in range(NUM_CELLS):
        if occ[cell]:
            continue
        has_own = False
        has_opp = False
        for d in range(NUM_DIRS):
            nb = NEIGHBORS[cell, d]
            if nb < 0 or not occ[nb]:
                continue
            if top_color[nb] == color:
                has_own = True
            elif top_color[nb] == opp:
                has_opp = True
        if has_own and not has_opp:
            out.append(cell)
    if len(out) > C_HAND:
        out = out[:C_HAND]
    return np.asarray(out, dtype=np.int32)


# ── Throw detection ───────────────────────────────────────────────────

def find_thrower_cell(
    top_color: np.ndarray, top_type: np.ndarray, heights: np.ndarray,
    fc: int, tc: int, color: int,
) -> int:
    """Return cell of current-player's pillbug-capable piece adjacent to both
    fc and tc, or -1 if no such cell exists (→ move is not a throw)."""
    # Pillbug-capable = top is P, OR top is M with any P neighbor (any color—
    # mosquito copies abilities regardless of adjacency's color per standard rules,
    # but here we just need adjacency to some pillbug).
    # Iterate fc's neighbors in direction order (matches CUDA kernel).
    tc_nbr_set = {int(n) for n in NEIGHBORS[tc] if n >= 0}
    common: list[int] = []
    for d in range(NUM_DIRS):
        n = int(NEIGHBORS[fc, d])
        if n >= 0 and n in tc_nbr_set:
            common.append(n)
    for cand in common:
        if heights[cand] == 0:
            continue
        if top_color[cand] != color:
            continue
        ttype = int(top_type[cand])
        if ttype == PT_PILLBUG:
            return cand
        if ttype == PT_MOSQUITO:
            # Ground-level mosquito adjacent to any pillbug gains throw ability
            if heights[cand] != 1:
                continue  # elevated mosquito acts as beetle only
            for d in range(NUM_DIRS):
                nb = NEIGHBORS[cand, d]
                if nb < 0 or heights[nb] == 0:
                    continue
                if top_type[nb] == PT_PILLBUG:
                    return cand
    return -1


def throw_slot_idx(pb_cell: int, fc: int, tc: int) -> int:
    """Encode throw in [0, 30): 6 target-dirs × 5 dest-dirs.
    target-dir = direction from pillbug to target (fc).
    dest-dir ∈ other 5 directions, compacted by removing target-dir."""
    td = direction_of(pb_cell, fc)  # 0..5
    dd = direction_of(pb_cell, tc)  # 0..5
    assert 0 <= td < 6 and 0 <= dd < 6 and td != dd, (td, dd)
    # compact dd into [0..5) by dropping td
    dd_compact = dd if dd < td else dd - 1
    return td * 5 + dd_compact


def decode_throw_slot(slot: int) -> tuple[int, int]:
    """Inverse of throw_slot_idx — returns (target_dir, dest_dir)."""
    td = slot // 5
    dd_compact = slot % 5
    dd = dd_compact if dd_compact < td else dd_compact + 1
    return td, dd


# ── Main classification ────────────────────────────────────────────────

class SlotMapper:
    """Produces (slot_of_legal, move_cells, place_cells) for a single state."""

    def __init__(self, state_bytes: np.ndarray):
        self.state = state_bytes
        self.top_color, self.top_type, self.heights = decode_top_colors_and_types(state_bytes)
        self.color = current_color_from_turn(state_bytes)
        self.by_type = pieces_in_play_by_type(self.top_color, self.top_type, self.color)
        self.move_cells = enumerate_move_cells(self.heights)
        self.place_cells = enumerate_place_cells(self.top_color, self.heights, self.color)
        # Cell → cell-token index lookups
        self._move_cell_idx = np.full(NUM_CELLS, -1, dtype=np.int32)
        self._move_cell_idx[self.move_cells] = np.arange(len(self.move_cells))
        self._place_cell_idx = np.full(NUM_CELLS, -1, dtype=np.int32)
        self._place_cell_idx[self.place_cells] = np.arange(len(self.place_cells))
        # Ranked instance indices per (type): cell → rank
        self._inst_idx: dict[tuple[int, int], int] = {}
        for pt, cells in self.by_type.items():
            for rank, cell in enumerate(cells):
                self._inst_idx[(pt, cell)] = rank

    def _dir_piece_idx(self, pt: int, fc: int) -> int:
        base = _DIR_BASE.get(pt)
        if base is None:
            return -1
        rank = self._inst_idx.get((pt, fc), -1)
        if rank < 0:
            return -1
        if pt == PT_BEETLE and rank >= 2: return -1
        if pt == PT_GRASSHOPPER and rank >= 3: return -1
        if pt in (PT_QUEEN, PT_PILLBUG, PT_MOSQUITO) and rank >= 1: return -1
        return base + rank

    def _long_piece_idx(self, pt: int, fc: int) -> int:
        base = _LONG_BASE.get(pt)
        if base is None:
            return -1
        rank = self._inst_idx.get((pt, fc), -1)
        if rank < 0:
            return -1
        if pt == PT_ANT and rank >= 3: return -1
        if pt == PT_SPIDER and rank >= 2: return -1
        if pt in (PT_LADYBUG, PT_MOSQUITO) and rank >= 1: return -1
        return base + rank

    def classify(self, move: np.ndarray) -> int:
        """Return slot in [0, N_SLOTS) for a single legal move, or -1 if
        un-mappable (indicates a bug or uncapped-slot overflow)."""
        mtype = int(move[0])
        pt = int(move[1] & 0x0F)
        fc = int(move[2]) | (int(move[3]) << 8)
        tc = int(move[4]) | (int(move[5]) << 8)

        if mtype == MOVE_PASS:
            return PASS_SLOT

        if mtype == MOVE_PLACE:
            cell_tok = int(self._place_cell_idx[tc])
            if cell_tok < 0:
                # First placement case: place_cells enumeration is empty on empty board.
                # Fall back to move_cells index (which for first move is also empty).
                # For first placement, there's only cell (11,11) as legal destination.
                # We still need SOME slot — reuse move_cells mapping as fallback.
                cell_tok = int(self._move_cell_idx[tc])
                if cell_tok < 0:
                    # Truly empty board, tc = origin cell. Encode at position 0.
                    cell_tok = 0
                if cell_tok >= C_HAND:
                    return -1
            return HAND_OFFSET + (pt - 1) * C_HAND + cell_tok

        # MOVE_MOVE: throw detection first
        pb = find_thrower_cell(self.top_color, self.top_type, self.heights,
                               fc, tc, self.color)
        if pb >= 0:
            thrower_type = int(self.top_type[pb])
            thrower_idx = _THROW_BASE[thrower_type]  # 0 or 1
            slot = throw_slot_idx(pb, fc, tc)
            return THROW_OFFSET + thrower_idx * 30 + slot

        # Non-throw MOVE classification by piece type
        if pt in (PT_QUEEN, PT_BEETLE, PT_PILLBUG):
            d = direction_of(fc, tc)
            if d < 0:
                return -1
            pidx = self._dir_piece_idx(pt, fc)
            if pidx < 0:
                return -1
            return DIR_OFFSET + pidx * 6 + d

        if pt == PT_GRASSHOPPER:
            occ = self.heights > 0
            d = grasshopper_dir_of(occ, fc, tc)
            if d < 0:
                return -1
            pidx = self._dir_piece_idx(pt, fc)
            if pidx < 0:
                return -1
            return DIR_OFFSET + pidx * 6 + d

        if pt == PT_MOSQUITO:
            # Neighbor destination → dir slot (covers beetle-climb, Q-style,
            # pillbug-own). Far destination → long slot (A/S/L/G-style).
            d = direction_of(fc, tc)
            if d >= 0:
                pidx = self._dir_piece_idx(PT_MOSQUITO, fc)
                if pidx < 0:
                    return -1
                return DIR_OFFSET + pidx * 6 + d
            cell_tok = int(self._move_cell_idx[tc])
            if cell_tok < 0:
                return -1
            pidx = self._long_piece_idx(PT_MOSQUITO, fc)
            if pidx < 0:
                return -1
            return LONG_OFFSET + pidx * C_MOVE + cell_tok

        if pt in (PT_ANT, PT_SPIDER, PT_LADYBUG):
            cell_tok = int(self._move_cell_idx[tc])
            if cell_tok < 0:
                return -1
            pidx = self._long_piece_idx(pt, fc)
            if pidx < 0:
                return -1
            return LONG_OFFSET + pidx * C_MOVE + cell_tok

        return -1


def map_legal_moves(
    state_bytes: np.ndarray,      # (SIZEOF_HIVE_STATE,) uint8
    legal_moves: np.ndarray,      # (N_legal, 6) uint8
    n_legal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a single state's legal moves to slot indices.

    Returns:
      slot_of_legal: (n_legal,) int32
      move_cells:    (n_move_cells,) int32  — ascending cells, ≤C_MOVE
      place_cells:   (n_place_cells,) int32 — ascending cells, ≤C_HAND
    """
    mapper = SlotMapper(state_bytes)
    slots = np.full(n_legal, -1, dtype=np.int32)
    for i in range(n_legal):
        slots[i] = mapper.classify(legal_moves[i])
    return slots, mapper.move_cells, mapper.place_cells
