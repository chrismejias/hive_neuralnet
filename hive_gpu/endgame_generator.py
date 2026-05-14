"""
Endgame position generator for curriculum learning.

Generates near-endgame Hive positions where both queens are nearly surrounded
(min_surround <= neighbors <= max_surround out of 6).  Positions are serialized
to the GPU HiveState byte format so they can be used directly as starting states
for MCTS self-play, ensuring every game has meaningful tactical decisions.

GPU HiveState layout (SIZEOF = 3424 bytes):
  [0..2644]     pieces[5][529]      uint8  (level-major, then cell)
  [2645..3173]  height[529]         uint8
  [3174..3175]  padding             (alignment for Bitboard / uint64)
  [3176..3247]  occupied bitboard   9 x uint64 little-endian
  [3248..3319]  white_top bitboard
  [3320..3391]  black_top bitboard
  [3392..3395]  queen_cell[2]       uint16 x 2  (0xFFFF = not placed)
  [3396..3411]  hands[2][8]         uint8 (WHITE then BLACK; idx = PT-1)
  [3412..3413]  turn                uint16
  [3414]        queen_placed        uint8 (bit0=white, bit1=black)
  [3415]        result              uint8 (0=in_progress,1=ww,2=bw,3=draw)
  [3416]        center_q            int8  (recomputed by encoder; set 0)
  [3417]        center_r            int8
  [3418..3419]  _pad[2]
  [3420..3423]  trailing compiler padding
"""

from __future__ import annotations

import random
import struct
from typing import Optional

import numpy as np
import torch

from hive_engine.game_state import GameResult, GameState
from hive_engine.pieces import Color, ExpansionConfig, PieceType

# ── Layout constants (must mirror hive_state.cuh / hex_grid.cuh) ──────────

BOARD_SIZE   = 23
HALF_BOARD   = 11
NUM_CELLS    = 529          # 23 * 23
MAX_STACK    = 5
NUM_PIECE_TYPES = 8         # Q A G S B M L P
BB_WORDS     = 9            # ceil(529 / 64)
SIZEOF_HIVE_STATE = 3424    # verified via ext.SIZEOF_HIVE_STATE

_OFF_PIECES      = 0
_OFF_HEIGHT      = 2645
_OFF_OCCUPIED    = 3176     # after 2-byte alignment padding
_OFF_WHITE_TOP   = 3248
_OFF_BLACK_TOP   = 3320
_OFF_QUEEN_CELL  = 3392
_OFF_HANDS       = 3396
_OFF_TURN        = 3412
_OFF_QUEEN_PLACED = 3414
_OFF_RESULT      = 3415
_OFF_CENTER_Q    = 3416
_OFF_CENTER_R    = 3417

# Python PieceType (0-based) -> GPU PieceType (1-based)
_PT_GPU = {
    PieceType.QUEEN:       1,
    PieceType.ANT:         2,
    PieceType.GRASSHOPPER: 3,
    PieceType.SPIDER:      4,
    PieceType.BEETLE:      5,
    PieceType.MOSQUITO:    6,
    PieceType.LADYBUG:     7,
    PieceType.PILLBUG:     8,
}

# GameResult mapping (Python IntEnum values == GPU enum values)
_RESULT_GPU = {
    GameResult.IN_PROGRESS: 0,
    GameResult.WHITE_WINS:  1,
    GameResult.BLACK_WINS:  2,
    GameResult.DRAW:        3,
}


# ── Coordinate helpers ─────────────────────────────────────────────────────

def _qr_to_cell(q: int, r: int) -> int:
    """Axial (q, r) -> GPU cell index.  Center is (0, 0) = cell 264."""
    col = q + HALF_BOARD
    row = r + HALF_BOARD
    if not (0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE):
        raise ValueError(f"Coordinate ({q}, {r}) outside 23x23 GPU grid")
    return row * BOARD_SIZE + col


def _bb_set(buf: bytearray, bb_offset: int, cell: int) -> None:
    """Set bit `cell` in the bitboard stored at byte offset `bb_offset`."""
    word = cell >> 6
    bit  = cell & 63
    base = bb_offset + word * 8
    lo = int.from_bytes(buf[base:base + 8], 'little')
    lo |= (1 << bit)
    buf[base:base + 8] = lo.to_bytes(8, 'little')


# ── Serialiser ────────────────────────────────────────────────────────────

def gamestate_to_gpu_bytes(state: GameState) -> bytes:
    """
    Serialise a Python GameState to raw GPU HiveState bytes.

    The serialised buffer can be loaded into a [1, SIZEOF_HIVE_STATE] uint8
    tensor and used directly as a starting state for GPU MCTS self-play.
    center_q / center_r are left as 0 because the encoder recomputes them.
    """
    buf = bytearray(SIZEOF_HIVE_STATE)

    # ── Board pieces ──────────────────────────────────────────────
    for pos, stack in state.board.grid.items():
        cell = _qr_to_cell(pos.q, pos.r)
        h = len(stack)
        buf[_OFF_HEIGHT + cell] = h
        _bb_set(buf, _OFF_OCCUPIED, cell)

        top = stack[-1]
        if top.color == Color.WHITE:
            _bb_set(buf, _OFF_WHITE_TOP, cell)
        else:
            _bb_set(buf, _OFF_BLACK_TOP, cell)

        for level, piece in enumerate(stack):
            gpu_pt    = _PT_GPU[piece.piece_type]
            gpu_color = int(piece.color)          # WHITE=0, BLACK=1
            packed    = gpu_pt | (gpu_color << 4)
            buf[_OFF_PIECES + level * NUM_CELLS + cell] = packed

    # ── Queen cell positions ───────────────────────────────────────
    for color in (Color.WHITE, Color.BLACK):
        c = int(color)
        found = False
        for pos, stack in state.board.grid.items():
            if any(p.piece_type == PieceType.QUEEN and p.color == color
                   for p in stack):
                cell = _qr_to_cell(pos.q, pos.r)
                struct.pack_into('<H', buf, _OFF_QUEEN_CELL + c * 2, cell)
                found = True
                break
        if not found:
            struct.pack_into('<H', buf, _OFF_QUEEN_CELL + c * 2, 0xFFFF)

    # ── Hands ─────────────────────────────────────────────────────
    for color in (Color.WHITE, Color.BLACK):
        c = int(color)
        counts: dict[PieceType, int] = {}
        for piece in state._hands[color]:
            counts[piece.piece_type] = counts.get(piece.piece_type, 0) + 1
        for pt, count in counts.items():
            idx = _PT_GPU[pt] - 1   # 0-based index into hands[c]
            buf[_OFF_HANDS + c * NUM_PIECE_TYPES + idx] = count

    # ── Game metadata ──────────────────────────────────────────────
    struct.pack_into('<H', buf, _OFF_TURN, state.turn)

    qp = 0
    if state._queen_placed[Color.WHITE]:
        qp |= 1
    if state._queen_placed[Color.BLACK]:
        qp |= 2
    buf[_OFF_QUEEN_PLACED] = qp

    buf[_OFF_RESULT] = _RESULT_GPU.get(state.result, 0)
    # center_q / center_r stay 0 (encoder recomputes)

    return bytes(buf)


def positions_to_tensor(
    positions: list[bytes],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Pack a list of serialised positions into a [N, SIZEOF] uint8 GPU tensor.

    The tensor can be used directly as `root_states` in GPU MCTS.
    """
    n = len(positions)
    flat = np.frombuffer(b"".join(positions), dtype=np.uint8)
    arr  = flat.reshape(n, SIZEOF_HIVE_STATE).copy()
    return torch.from_numpy(arr).to(device)


# ── Hex neighbour table (precomputed once at import time) ─────────────────

def _build_neighbor_table() -> np.ndarray:
    """Return [529, 6] int16 array of neighbour cell indices (-1 = off-grid)."""
    DIR_DCOL = [+1, +1,  0, -1, -1,  0]
    DIR_DROW = [ 0, -1, -1,  0, +1, +1]
    table = np.full((NUM_CELLS, 6), -1, dtype=np.int16)
    for cell in range(NUM_CELLS):
        row, col = divmod(cell, BOARD_SIZE)
        for d in range(6):
            nr, nc = row + DIR_DROW[d], col + DIR_DCOL[d]
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                table[cell, d] = nr * BOARD_SIZE + nc
    return table

_NEIGHBOR_TABLE: np.ndarray = _build_neighbor_table()


# ── GPU-accelerated endgame generator ─────────────────────────────────────

def generate_endgame_positions(
    n_positions: int,
    expansion_mask: int = -1,       # -1 = random per batch
    surround: int = 5,              # max neighbor count for both queens
    gpu_batch: int = 512,           # parallel games on GPU per wave
    max_steps: int = 400,           # max random moves per game before reset
    verbose: bool = False,
    min_surround: Optional[int] = None,  # inclusive lower bound (default = surround)
    max_surround: Optional[int] = None,  # inclusive upper bound (overrides surround)
    gpu_filter: bool = True,
    mixed_pair: bool = False,
    **_kwargs,
) -> list[bytes]:
    """
    Generate *n_positions* endgame positions using GPU-parallel random rollouts.

    Runs *gpu_batch* games simultaneously on GPU with random moves. After
    each step the queen-surround counts are checked; any game where both
    queens satisfy the requested surround criterion is saved and that slot is
    immediately reset with a fresh game.

    When mixed_pair=True, the hit criterion is asymmetric exact matching:
    one queen must have exactly min_surround neighbours and the other exactly
    max_surround neighbours (in either order).

    ~100-200x faster than pure-Python random play for surround=5.
    """
    if max_surround is not None:
        surround = max_surround
    lo = min_surround if min_surround is not None else surround

    import hive_gpu
    ext = hive_gpu.load_extension()

    nb_table = _NEIGHBOR_TABLE  # [529, 6]

    def _surround_counts(states_np: np.ndarray, qcells: np.ndarray) -> np.ndarray:
        """Vectorised surround count for a batch of queen cell indices."""
        heights = states_np[:, _OFF_HEIGHT: _OFF_HEIGHT + NUM_CELLS]  # [B, 529]
        valid_queen = qcells < NUM_CELLS
        qcells_safe = np.where(valid_queen, qcells, 0).astype(np.int32)
        nb_cells = nb_table[qcells_safe]
        on_grid = nb_cells >= 0
        nb_safe = np.where(on_grid, nb_cells, 0)
        bi = np.arange(len(heights))[:, None]
        nb_occ = (heights[bi, nb_safe] > 0) & on_grid
        counts = nb_occ.sum(axis=1).astype(np.int32)
        counts[~valid_queen] = 0
        return counts

    # Choose a fixed expansion mask for this pool (or random per batch).
    def _pick_mask() -> int:
        return random.randint(0, 7) if expansion_mask < 0 else expansion_mask

    positions: list[bytes] = []
    steps_total = 0
    resets_total = 0
    mask = _pick_mask()

    states = ext.create_initial_states(gpu_batch, mask)
    row_idx = torch.arange(gpu_batch, device="cuda", dtype=torch.int64)
    # track how many steps each game has taken (reset when > max_steps)
    game_steps = np.zeros(gpu_batch, dtype=np.int32)
    step_num   = 0

    while len(positions) < n_positions:
        # ── Random move selection (GPU-side) ─────────────────────
        moves_t, nlegal_t = ext.generate_legal_moves_batch(states, gpu_batch)
        nlegal_safe = nlegal_t.clamp(min=1).to(torch.float32)
        rand_idx = (
            torch.rand(gpu_batch, device="cuda") * nlegal_safe
        ).to(torch.int64).clamp(max=moves_t.shape[1] - 1)
        chosen = moves_t[row_idx, rand_idx]   # [B, MOVE_SIZE]
        ext.apply_moves_batch(states, chosen, gpu_batch)
        game_steps += 1
        steps_total += gpu_batch
        step_num    += 1

        # ── Harvest / reset check (every 3 steps to reduce PCIe traffic) ──
        if step_num % 3 != 0:
            continue

        results_t = ext.check_results_batch(states, gpu_batch)
        results = results_t.cpu().numpy()
        hit = np.zeros(gpu_batch, dtype=bool)
        if gpu_filter:
            hit_t = ext.endgame_hit_mask_batch(
                states, gpu_batch, lo, surround, mixed_pair,
            )
            hit_idx = torch.nonzero(hit_t, as_tuple=False).squeeze(1)
            if hit_idx.numel() > 0:
                hit_rows = hit_idx.cpu().numpy()
                hit[hit_rows] = True
                hit_states = states.index_select(0, hit_idx).cpu().numpy()
            else:
                hit_rows = np.empty((0,), dtype=np.int64)
                hit_states = np.empty((0, SIZEOF_HIVE_STATE), dtype=np.uint8)
        else:
            states_np = states.cpu().numpy()
            both_placed = (states_np[:, _OFF_QUEEN_PLACED] & 3) == 3
            qc_raw = states_np[:, _OFF_QUEEN_CELL: _OFF_QUEEN_CELL + 4].copy()
            queen_cells = qc_raw.view(np.uint16).reshape(gpu_batch, 2)
            candidates = both_placed & (results == 0)
            w_surr = np.zeros(gpu_batch, dtype=np.int32)
            b_surr = np.zeros(gpu_batch, dtype=np.int32)
            if candidates.any():
                ci = np.where(candidates)[0]
                w_surr[ci] = _surround_counts(states_np[ci], queen_cells[ci, 0])
                b_surr[ci] = _surround_counts(states_np[ci], queen_cells[ci, 1])
            if mixed_pair:
                hit = candidates & (
                    ((w_surr == lo) & (b_surr == surround)) |
                    ((w_surr == surround) & (b_surr == lo))
                )
            else:
                hit = (
                    candidates &
                    (w_surr >= lo) & (w_surr <= surround) &
                    (b_surr >= lo) & (b_surr <= surround)
                )
            hit_rows = np.where(hit)[0]
            hit_states = states_np[hit_rows]

        # Save hits
        for row in hit_states:
            positions.append(bytes(row))
            if len(positions) >= n_positions:
                break

        # Reset finished or over-long games (and just-harvested positions)
        needs_reset = hit | (results != 0) | (game_steps > max_steps)
        n_reset = int(needs_reset.sum())
        if n_reset > 0:
            mask = _pick_mask()
            fresh = ext.create_initial_states(n_reset, mask)
            idxs = np.where(needs_reset)[0]
            states[idxs] = fresh[:n_reset]
            game_steps[idxs] = 0
            resets_total += n_reset

    if verbose:
        print(f"[endgame] Generated {len(positions)} positions  "
              f"steps={steps_total}  resets={resets_total}  "
              f"surround={lo}-{surround} mixed_pair={mixed_pair}")

    return positions[:n_positions]
