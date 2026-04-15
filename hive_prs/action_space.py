"""
Piece-Relative Action Space for Hive.

Every legal move in Hive ends with a piece either placed/moved:
  - adjacent to an existing board piece in one of 6 hex directions, OR
  - stacked on top of an existing board piece (beetle only).

This lets us encode every action as:
    (actor, reference_piece, direction)

where:
  - actor    = board-token index of the moving piece (for MOVE),
                or piece type index (for PLACE)
  - reference = board-token index of the piece being referenced
  - direction = 0-5 for the 6 hex neighbours; 6 for "stack on top"

Board tokens are ordered by ascending cell index within each game state,
matching the order the CUDA encoder emits them.  The first placement
(empty board) needs no reference; it uses a dedicated slot.

Action index layout (flat integer):
  [0, MOVE_ACTIONS)            Movement actions  : MAX_BOARD × MAX_BOARD × 7
  [MOVE_ACTIONS, MOVE_ACTIONS + PLACE_ACTIONS)
                               Placement actions : NUM_TYPES × MAX_BOARD × 6
  [MOVE_ACTIONS + PLACE_ACTIONS]
                               First placement   : NUM_TYPES (board is empty)
  [MOVE_ACTIONS + PLACE_ACTIONS + NUM_TYPES]
                               Pass

Constants
---------
MAX_BOARD  : maximum number of board pieces per game state (28 with all
             expansions, 22 for base-game only).  We use 28 to be safe.
NUM_TYPES  : 8 (Q, A, G, S, B, M, L, P)
DIRECTIONS : 7 (6 hex neighbours + 1 for "on top")
"""

from __future__ import annotations

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# Internal 23×23 CUDA game board
BOARD_CELLS = 23 * 23          # 529

# Fixed maximum board tokens (14 per player with all expansions × 2 players)
MAX_BOARD   = 28
NUM_TYPES   = 8                 # Q=0 A=1 G=2 S=3 B=4 M=5 L=6 P=7
DIRECTIONS  = 7                 # 0-5 hex neighbours, 6 = stack on top

# Hex neighbour offsets on a 23-wide grid  (col_delta, row_delta) → linear delta
# Matches the CUDA DIR_DCOL / DIR_DROW convention: E NE NW W SW SE
_BOARD_SIZE = 23
_DIR_DCOL = [+1, +1,  0, -1, -1,  0]
_DIR_DROW = [ 0, -1, -1,  0, +1, +1]
# Linear cell offset for each direction on a 23-wide grid
DIR_DELTA = [dr * _BOARD_SIZE + dc for dr, dc in zip(_DIR_DROW, _DIR_DCOL)]

# ── Action space sizes ────────────────────────────────────────────────────────

MOVE_ACTIONS  = MAX_BOARD * MAX_BOARD * DIRECTIONS          # 5,488
PLACE_ACTIONS = NUM_TYPES  * MAX_BOARD * (DIRECTIONS - 1)  # 1,344  (no "stack" for placements)
FIRST_PLACE_ACTIONS = NUM_TYPES                            # 8
PASS_ACTION_OFFSET  = MOVE_ACTIONS + PLACE_ACTIONS + FIRST_PLACE_ACTIONS

ACTION_SPACE_SIZE = PASS_ACTION_OFFSET + 1                  # 6,841

# Offsets
_PLACE_OFFSET       = MOVE_ACTIONS                          # 5,488
_FIRST_PLACE_OFFSET = MOVE_ACTIONS + PLACE_ACTIONS          # 6,832

# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_move(actor_tok: int, ref_tok: int, direction: int) -> int:
    """Flat index for a movement action."""
    return actor_tok * (MAX_BOARD * DIRECTIONS) + ref_tok * DIRECTIONS + direction


def encode_place(piece_type: int, ref_tok: int, direction: int) -> int:
    """Flat index for a placement (board non-empty)."""
    return (_PLACE_OFFSET
            + piece_type * (MAX_BOARD * (DIRECTIONS - 1))
            + ref_tok    * (DIRECTIONS - 1)
            + direction)


def encode_first_place(piece_type: int) -> int:
    """Flat index for the very first placement (empty board)."""
    return _FIRST_PLACE_OFFSET + piece_type


def encode_pass() -> int:
    return PASS_ACTION_OFFSET


# ── Decoding helpers ──────────────────────────────────────────────────────────

def decode_action(idx: int) -> dict:
    """Decode a flat action index back to its components (for debugging)."""
    if idx == PASS_ACTION_OFFSET:
        return {"type": "pass"}
    if idx >= _FIRST_PLACE_OFFSET:
        pt = idx - _FIRST_PLACE_OFFSET
        return {"type": "first_place", "piece_type": pt}
    if idx >= _PLACE_OFFSET:
        rem = idx - _PLACE_OFFSET
        pt  = rem // (MAX_BOARD * (DIRECTIONS - 1))
        rem %= MAX_BOARD * (DIRECTIONS - 1)
        ref = rem // (DIRECTIONS - 1)
        d   = rem %  (DIRECTIONS - 1)
        return {"type": "place", "piece_type": pt, "ref_tok": ref, "direction": d}
    # Movement
    actor = idx // (MAX_BOARD * DIRECTIONS)
    rem   = idx %  (MAX_BOARD * DIRECTIONS)
    ref   = rem // DIRECTIONS
    d     = rem %  DIRECTIONS
    return {"type": "move", "actor_tok": actor, "ref_tok": ref, "direction": d}


# Numpy arrays for vectorised ops
_DIR_DELTA_NP = np.array(DIR_DELTA, dtype=np.int32)   # (6,)
_DIR_FROM_NB  = (np.arange(6, dtype=np.int32) + 3) % 6  # opposite direction


# ── All-representations map (for training target deduplication fix) ────────────

def moves_to_all_reps(
    move_bytes:        np.ndarray,   # (N_legal, 6) uint8
    n_legal:           int,
    occupied_cells:    np.ndarray,   # (N_board,) int32 ascending
    canonical_indices: np.ndarray,   # (N_legal,) int32 — from batch_moves_to_action_indices
) -> np.ndarray:
    """
    For each legal move, compute ALL valid PRS indices (not just canonical).

    Returns
    -------
    all_reps : (N_legal, 6) int32, padded with -1.
        Column 0 is the canonical index, columns 1-5 are alternative reps.
    """
    if n_legal == 0:
        return np.empty((0, 6), dtype=np.int32)

    n_board = len(occupied_cells)
    all_reps = np.full((n_legal, 6), -1, dtype=np.int32)
    all_reps[:, 0] = canonical_indices[:n_legal]

    # Build cell→token lookup
    cell_to_tok = np.full(BOARD_CELLS, -1, dtype=np.int32)
    if n_board > 0:
        cell_to_tok[occupied_cells] = np.arange(n_board, dtype=np.int32)

    # Parse all moves at once
    moves  = move_bytes[:n_legal].astype(np.int32)
    mtype  = moves[:, 0]
    pt     = (moves[:, 1] & 0x0F) - 1
    fc     = moves[:, 2] | (moves[:, 3] << 8)
    tc     = moves[:, 4] | (moves[:, 5] << 8)

    fc_safe  = fc.clip(0, BOARD_CELLS - 1)
    tc_safe  = tc.clip(0, BOARD_CELLS - 1)
    actor_tok = np.where(mtype == 1, cell_to_tok[fc_safe], -1)

    to_tok_at_dest = cell_to_tok[tc_safe]
    is_beetle_stack = (mtype == 1) & (to_tok_at_dest >= 0) & (to_tok_at_dest < MAX_BOARD)

    nb_all = tc[:, None] + _DIR_DELTA_NP[None, :]
    in_bounds = (nb_all >= 0) & (nb_all < BOARD_CELLS)
    not_fc = nb_all != fc[:, None]
    nb_safe = nb_all.clip(0, BOARD_CELLS - 1)
    tok_nb  = cell_to_tok[nb_safe]
    occupied_nb = (tok_nb >= 0) & (tok_nb < MAX_BOARD)
    dir_nb = np.broadcast_to(_DIR_FROM_NB[None, :], (n_legal, 6))

    # Compute (N, 6) PRS indices for each neighbour direction
    place_idx = (
        _PLACE_OFFSET
        + pt[:, None] * (MAX_BOARD * (DIRECTIONS - 1))
        + tok_nb * (DIRECTIONS - 1)
        + dir_nb
    )
    move_idx = (
        actor_tok[:, None] * (MAX_BOARD * DIRECTIONS)
        + tok_nb * DIRECTIONS
        + dir_nb
    )

    # Build validity mask per neighbour: (N, 6)
    valid_place = in_bounds & occupied_nb & (mtype == 0)[:, None]
    is_non_stack = (mtype == 1) & ~is_beetle_stack
    valid_move  = in_bounds & not_fc & occupied_nb & is_non_stack[:, None]
    valid_any = valid_place | valid_move

    # Select place or move indices
    combined = np.where(valid_place, place_idx, np.where(valid_move, move_idx, -1))
    # Filter out-of-range
    combined[(combined < 0) | (combined >= ACTION_SPACE_SIZE)] = -1
    # Beetle stacks and invalid canonical → only canonical rep (already in col 0)
    combined[is_beetle_stack] = -1

    # Pack valid alternative reps into columns 1-5 (canonical is col 0)
    # For each row, gather valid values that differ from canonical
    can = canonical_indices[:n_legal]
    for d in range(6):
        col = combined[:, d]
        # Mark entries that equal canonical as -1 (already in col 0)
        col = np.where(col == can, -1, col)
        combined[:, d] = col

    # Compact valid entries into columns 1..5 per row
    # Use vectorized argsort-based compaction
    valid_mask = combined >= 0  # (N, 6)
    n_valid = valid_mask.sum(axis=1)  # (N,)
    has_alts = n_valid > 0
    if has_alts.any():
        # Sort so that valid entries come first (descending by validity)
        # We negate the mask so valid=True sorts first
        sort_key = (~valid_mask).astype(np.int32)
        order = np.argsort(sort_key, axis=1, kind='stable')
        rows = np.arange(n_legal)[:, None]
        sorted_combined = combined[rows, order]
        # Copy compacted alternatives into all_reps columns 1..5
        max_alts = min(5, sorted_combined.shape[1])
        all_reps[has_alts, 1:1+max_alts] = sorted_combined[np.ix_(has_alts, np.arange(max_alts))]

    return all_reps


def batch_moves_to_all_reps(
    legal_np:  np.ndarray,     # (B, MAX_L, 6) uint8
    nlegal_np: np.ndarray,     # (B,) int32
    occ_cpu:   np.ndarray,     # (B, MAX_BOARD) int32
    nocc_cpu:  np.ndarray,     # (B,) int32
    prs_from_move: list[np.ndarray],  # list of B arrays, each (n_legal_i,) int32
) -> list[np.ndarray]:
    """Compute all_reps for each game in the batch. Returns list of (n_legal_i, 6) arrays."""
    B = len(nlegal_np)
    result = []
    for i in range(B):
        nl_i = int(nlegal_np[i])
        no_i = int(nocc_cpu[i])
        result.append(moves_to_all_reps(
            legal_np[i], nl_i, occ_cpu[i, :no_i], prs_from_move[i],
        ))
    return result


# ── Batch vectorised conversion (fast path) ────────────────────────────────────

def batch_moves_to_action_indices(
    legal_np:  np.ndarray,   # (B, MAX_L, 6) uint8
    nlegal_np: np.ndarray,   # (B,) int32
    occ_cpu:   np.ndarray,   # (B, MAX_BOARD) int32  — sorted, -1 = pad
    nocc_cpu:  np.ndarray,   # (B,) int32
) -> list[np.ndarray]:
    """
    Vectorised batch replacement for calling moves_to_action_indices B times.

    Returns a list of B arrays, each of shape (n_legal_i,) int32 with PRS
    action indices (-1 for un-encodable moves).

    Algorithm
    ---------
    For each game g, we need cell→token lookups from occ_cpu[g].
    Since occ_cpu is sorted, a (total_moves, MAX_BOARD) broadcast comparison
    gives us token indices in O(total_moves × MAX_BOARD) numpy operations,
    which is much faster than the equivalent Python loop.
    """
    B = len(nlegal_np)
    total = int(nlegal_np.sum())
    if total == 0:
        return [np.empty(0, dtype=np.int32) for _ in range(B)]

    # ── Flatten all legal moves with their game id ──
    game_ids   = np.repeat(np.arange(B, dtype=np.int32), nlegal_np)  # (total,)
    # Boolean-index across the padded move array — avoids a Python loop + vstack
    max_l      = legal_np.shape[1]
    j_range    = np.arange(max_l, dtype=np.int32)
    valid_j    = j_range[None, :] < nlegal_np[:, None]   # (B, MAX_L) bool
    flat_moves = legal_np[valid_j]                        # (total, 6) uint8

    move_type  = flat_moves[:, 0].astype(np.int32)
    piece_type = (flat_moves[:, 1] & 0x0F).astype(np.int32) - 1      # 0-indexed
    from_cell  = (flat_moves[:, 2].astype(np.int32) |
                  (flat_moves[:, 3].astype(np.int32) << 8))
    to_cell    = (flat_moves[:, 4].astype(np.int32) |
                  (flat_moves[:, 5].astype(np.int32) << 8))
    n_board_g  = nocc_cpu[game_ids]                                    # (total,)

    # ── Vectorised cell→token lookup ──
    # occ_game[m, :] = occupied cells for game of move m  (total, MAX_BOARD)
    occ_game = occ_cpu[game_ids]   # (total, MAX_BOARD)

    def cell_to_tok_vec(cells: np.ndarray) -> np.ndarray:
        """Return token index for each cell (N,); -1 if not occupied.
        Uses the current value of game_ids (which may be narrowed to a subset)."""
        occ_sub = occ_cpu[game_ids]              # (N, MAX_BOARD) — current game_ids
        matches = (occ_sub == cells[:, None])    # (N, MAX_BOARD)
        found   = matches.any(axis=1)
        toks    = np.where(found, matches.argmax(axis=1), -1).astype(np.int32)
        return toks

    # ── Vectorised find_ref_tok: best (smallest-token) neighbour of to_cell ──
    # Exclude one cell (exclude_cell=-1 means no exclusion, i.e. for PLACE).
    def find_ref_tok_vec(
        to_cells: np.ndarray,      # (N,)
        exclude:  np.ndarray,      # (N,) cell to skip (-1 = don't skip)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ref_toks, dir_from_ref) where dir_from_ref is the hex direction
        FROM ref_cell TO to_cell. Both are (N,) int32; -1 means not found."""
        N      = len(to_cells)
        nb     = to_cells[:, None] + _DIR_DELTA_NP[None, :]   # (N, 6)
        valid  = ((nb >= 0) & (nb < BOARD_CELLS))              # (N, 6)
        if exclude is not None:
            valid &= (nb != exclude[:, None])

        # Look up token for each of the N×6 neighbours
        nb_flat      = nb.ravel()                              # (N*6,)
        gids_flat    = np.repeat(game_ids[:N], 6)             # (N*6,) — game id for each nb
        occ_flat     = occ_cpu[gids_flat]                     # (N*6, MAX_BOARD)
        matches_flat = (occ_flat == nb_flat[:, None])         # (N*6, MAX_BOARD)
        found_flat   = matches_flat.any(axis=1) & valid.ravel()
        toks_flat    = np.where(found_flat,
                                matches_flat.argmax(axis=1),
                                MAX_BOARD + 1).astype(np.int32)  # invalid = MAX_BOARD+1
        toks_nb      = toks_flat.reshape(N, 6)                # (N, 6)
        toks_nb[~valid] = MAX_BOARD + 1

        # Ref = neighbour with smallest token index
        ref_dir_local  = toks_nb.argmin(axis=1)               # (N,) — which direction
        ref_toks       = toks_nb[np.arange(N), ref_dir_local] # (N,)
        has_ref        = ref_toks <= MAX_BOARD
        ref_toks       = np.where(has_ref, ref_toks, -1).astype(np.int32)

        # Direction FROM ref_cell TO to_cell is opposite of the direction from
        # to_cell TO ref_cell (opposites differ by 3 mod 6 in our hex scheme).
        dir_from_ref   = np.where(has_ref, (ref_dir_local + 3) % 6, -1).astype(np.int32)
        return ref_toks, dir_from_ref

    # ── PLACE moves ──
    place_m   = move_type == 0
    idx_out   = np.full(total, -1, dtype=np.int32)

    if place_m.any():
        # First placement (board empty for this game)
        first_m = place_m & (n_board_g == 0)
        if first_m.any():
            idx_out[first_m] = _FIRST_PLACE_OFFSET + piece_type[first_m]

        # Normal placement
        norm_m = place_m & (n_board_g > 0)
        if norm_m.any():
            tc_p    = to_cell[norm_m]
            # Override game_ids in find_ref_tok_vec to use the subset
            # Save global game_ids for the subset
            orig_gids  = game_ids
            game_ids   = game_ids[norm_m]   # temporarily narrow for the helper
            ref_toks_p, dir_p = find_ref_tok_vec(tc_p, exclude=np.full(len(tc_p), -1))
            game_ids   = orig_gids          # restore

            valid_p = (ref_toks_p >= 0) & (ref_toks_p < MAX_BOARD) & (dir_p >= 0)
            idx_out[norm_m] = np.where(
                valid_p,
                _PLACE_OFFSET + piece_type[norm_m] * (MAX_BOARD * (DIRECTIONS - 1))
                + ref_toks_p * (DIRECTIONS - 1) + dir_p,
                -1,
            )

    # ── MOVE_PIECE moves ──
    move_m = move_type == 1
    if move_m.any():
        fc_mv = from_cell[move_m]
        tc_mv = to_cell[move_m]

        # Actor token
        orig_gids  = game_ids
        game_ids   = game_ids[move_m]
        actor_toks = cell_to_tok_vec(fc_mv)   # (N_move,)

        # Beetle stack: to_cell is already occupied
        to_toks = cell_to_tok_vec(tc_mv)      # (N_move,)
        is_stack = (to_toks >= 0)

        # Ref tok for non-stack moves (exclude from_cell since it's vacated)
        ref_toks_mv, dir_mv = find_ref_tok_vec(tc_mv, exclude=fc_mv)
        game_ids = orig_gids

        valid_actor = (actor_toks >= 0) & (actor_toks < MAX_BOARD)

        # Regular moves
        valid_reg = valid_actor & ~is_stack & (ref_toks_mv >= 0) & (ref_toks_mv < MAX_BOARD) & (dir_mv >= 0)
        idx_out[move_m] = np.where(
            valid_reg,
            actor_toks * (MAX_BOARD * DIRECTIONS) + ref_toks_mv * DIRECTIONS + dir_mv,
            -1,
        )
        # Beetle stack overwrites regular if applicable
        valid_stack = valid_actor & is_stack & (to_toks < MAX_BOARD)
        idx_out[move_m] = np.where(
            valid_stack,
            actor_toks * (MAX_BOARD * DIRECTIONS) + to_toks * DIRECTIONS + 6,
            idx_out[move_m],
        )

    # ── Split back into per-game arrays ──
    result  = []
    cursor  = 0
    for i in range(B):
        n = int(nlegal_np[i])
        result.append(idx_out[cursor:cursor + n].copy())
        cursor += n
    return result


# ── Per-state legal move → action index conversion ────────────────────────────

def moves_to_action_indices(
    move_bytes: np.ndarray,        # (N_legal, 6) uint8 — raw GPUMove bytes
    n_legal: int,
    occupied_cells: np.ndarray,    # (N_board,) int32 — board cell indices, ascending
) -> np.ndarray:
    """
    Convert raw GPUMove bytes to piece-relative action indices.

    Parameters
    ----------
    move_bytes   : (N_legal, 6) uint8 from generate_legal_moves_batch.
    n_legal      : number of valid moves in move_bytes.
    occupied_cells : sorted list of occupied cell indices on the board
                     (the CUDA encoder emits tokens in this order).

    Returns
    -------
    indices : (n_legal,) int32 — piece-relative action index, or -1 if
              the move falls outside MAX_BOARD (shouldn't happen in practice).
    """
    # Parse GPUMove: byte layout is type(1) + piece_type(1) + from_cell(2 LE) + to_cell(2 LE)
    moves = move_bytes[:n_legal]
    move_type  = moves[:, 0].astype(np.int32)            # 0=PLACE, 1=MOVE_PIECE
    piece_type = (moves[:, 1] & 0x0F).astype(np.int32)  # lower nibble = PieceType (1-indexed)
    from_cell  = (moves[:, 2].astype(np.int32) | (moves[:, 3].astype(np.int32) << 8))
    to_cell    = (moves[:, 4].astype(np.int32) | (moves[:, 5].astype(np.int32) << 8))
    piece_type -= 1  # convert from 1-indexed to 0-indexed

    n_board = len(occupied_cells)
    # Build cell → token-index lookup (sparse, using a dict or array)
    cell_to_tok = np.full(BOARD_CELLS, -1, dtype=np.int32)
    if n_board > 0:
        cell_to_tok[occupied_cells] = np.arange(n_board, dtype=np.int32)

    indices = np.full(n_legal, -1, dtype=np.int32)

    for i in range(n_legal):
        mtype = int(move_type[i])
        pt    = int(piece_type[i])
        fc    = int(from_cell[i])
        tc    = int(to_cell[i])

        if mtype == 0:  # PLACE
            if n_board == 0:
                # First placement — no reference piece
                indices[i] = encode_first_place(pt)
            else:
                # Find canonical reference: occupied neighbour of to_cell with smallest index
                ref_tok = _find_ref_tok(tc, cell_to_tok)
                if ref_tok < 0 or ref_tok >= MAX_BOARD:
                    continue
                d = _direction(ref_cell_for_tok(ref_tok, occupied_cells), tc)
                if d < 0:
                    continue
                indices[i] = encode_place(pt, ref_tok, d)
        else:  # MOVE_PIECE
            actor_tok = cell_to_tok[fc] if 0 <= fc < BOARD_CELLS else -1
            if actor_tok < 0 or actor_tok >= MAX_BOARD:
                continue
            # After the move, to_cell is occupied by the actor; find ref among remaining
            # occupied cells (exclude fc, which is vacated)
            ref_tok = _find_ref_tok_excluding(tc, cell_to_tok, fc)
            if ref_tok < 0 or ref_tok >= MAX_BOARD:
                # Beetle stacking: to_cell itself was already occupied → direction 6
                if cell_to_tok[tc] >= 0:
                    ref_tok = cell_to_tok[tc]
                    if ref_tok >= MAX_BOARD:
                        continue
                    indices[i] = encode_move(actor_tok, ref_tok, 6)
                continue
            ref_cell = occupied_cells[ref_tok]
            d = _direction(ref_cell, tc)
            if d < 0:
                continue
            indices[i] = encode_move(actor_tok, ref_tok, d)

    return indices


def _find_ref_tok(to_cell: int, cell_to_tok: np.ndarray) -> int:
    """Smallest-index occupied neighbour of to_cell."""
    best = MAX_BOARD + 1
    for delta in DIR_DELTA:
        nb = to_cell + delta
        if 0 <= nb < BOARD_CELLS:
            tok = int(cell_to_tok[nb])
            if 0 <= tok < best:
                best = tok
    return best if best <= MAX_BOARD else -1


def _find_ref_tok_excluding(to_cell: int, cell_to_tok: np.ndarray, exclude_cell: int) -> int:
    """Smallest-index occupied neighbour of to_cell, ignoring exclude_cell."""
    best = MAX_BOARD + 1
    for delta in DIR_DELTA:
        nb = to_cell + delta
        if 0 <= nb < BOARD_CELLS and nb != exclude_cell:
            tok = int(cell_to_tok[nb])
            if 0 <= tok < best:
                best = tok
    return best if best <= MAX_BOARD else -1


def ref_cell_for_tok(ref_tok: int, occupied_cells: np.ndarray) -> int:
    return int(occupied_cells[ref_tok])


def _direction(from_cell: int, to_cell: int) -> int:
    """Hex direction index (0-5) from from_cell to to_cell, or -1 if not adjacent."""
    delta = to_cell - from_cell
    for d, dd in enumerate(DIR_DELTA):
        if dd == delta:
            return d
    return -1
