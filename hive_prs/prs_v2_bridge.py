"""Bridge from SlotMapper outputs + trunk token positions to PRSv2HeadInputs.

Two entry points:

* `build_head_inputs_from_states(state_bytes, board_h, cls_h, full_h)` —
  CPU reference path used by tests and the stand-alone `HivePRSTransformerV2.forward`.
  Matches what the CUDA kernel does, but in slow Python.

* `build_head_inputs_from_kernel(board_h, cls_h, full_h, kernel_out)` —
  unpacks the 15-tuple returned by `ext.prs_v2_classify_batch` into a
  `PRSv2HeadInputs`. All tensors stay on GPU.
"""
from __future__ import annotations

import numpy as np
import torch

from hive_prs.prs_v2_head import (
    PRSv2HeadInputs, N_DIR_PIECES, N_THROW_PIECES, N_LONG_PIECES,
    MB, PAD_CELL_ID,
)
from hive_prs.slot_map import (
    SlotMapper, NEIGHBORS, C_MOVE, C_HAND,
    _DIR_BASE, _LONG_BASE, _THROW_BASE,
    PT_QUEEN, PT_BEETLE, PT_GRASSHOPPER, PT_PILLBUG, PT_MOSQUITO,
    PT_ANT, PT_SPIDER, PT_LADYBUG,
)


# ── CPU reference helpers ──────────────────────────────────────────────────

def _occupied_cells_ascending(heights: np.ndarray) -> np.ndarray:
    """Return cells with height>0, sorted ascending (matches trunk token order)."""
    return np.where(heights > 0)[0]


def _piece_instance_to_board_idx(
    mapper: SlotMapper, occ_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce (dir_idx[8], throw_idx[2], long_idx[7]) → board_token index or -1."""
    cell_to_board_idx = {int(c): i for i, c in enumerate(occ_cells)}
    dir_idx = np.full(N_DIR_PIECES, -1, dtype=np.int64)
    throw_idx = np.full(N_THROW_PIECES, -1, dtype=np.int64)
    long_idx = np.full(N_LONG_PIECES, -1, dtype=np.int64)

    for pt, base in _DIR_BASE.items():
        cells = mapper.by_type.get(pt, [])
        slots = {PT_QUEEN: 1, PT_BEETLE: 2, PT_GRASSHOPPER: 3,
                 PT_PILLBUG: 1, PT_MOSQUITO: 1}[pt]
        for rank in range(slots):
            if rank < len(cells):
                bi = cell_to_board_idx.get(int(cells[rank]), -1)
                dir_idx[base + rank] = bi

    for pt, base in _THROW_BASE.items():
        cells = mapper.by_type.get(pt, [])
        if cells:
            bi = cell_to_board_idx.get(int(cells[0]), -1)
            throw_idx[base] = bi

    for pt, base in _LONG_BASE.items():
        cells = mapper.by_type.get(pt, [])
        slots = {PT_ANT: 3, PT_SPIDER: 2, PT_LADYBUG: 1, PT_MOSQUITO: 1}[pt]
        for rank in range(slots):
            if rank < len(cells):
                bi = cell_to_board_idx.get(int(cells[rank]), -1)
                long_idx[base + rank] = bi

    return dir_idx, throw_idx, long_idx


def _cell_neighbor_board_indices(
    cells: np.ndarray, occ_cells: np.ndarray, cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nbrs[cap, 6], mask[cap], cell_ids[cap])."""
    cell_to_board_idx = np.full(529, -1, dtype=np.int64)
    for i, c in enumerate(occ_cells):
        cell_to_board_idx[c] = i

    C = cap
    nbrs = np.full((C, 6), -1, dtype=np.int64)
    mask = np.zeros(C, dtype=bool)
    cell_ids = np.full(C, -1, dtype=np.int32)
    for i, cell in enumerate(cells):
        if i >= C:
            break
        mask[i] = True
        cell_ids[i] = int(cell)
        for d in range(6):
            nb = NEIGHBORS[cell, d]
            if nb >= 0:
                nbrs[i, d] = cell_to_board_idx[nb]
    return nbrs, mask, cell_ids


def _dir_dest_for_piece(
    fc: int, pt: int, heights: np.ndarray, occ_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dest_cell[6], dest_board_idx[6]) for a single piece at `fc`.

    -1 for invalid / off-board destinations.
    """
    cell_to_board_idx = np.full(529, -1, dtype=np.int64)
    for i, c in enumerate(occ_cells):
        cell_to_board_idx[c] = i

    dest_cell = np.full(6, -1, dtype=np.int32)
    dest_bi   = np.full(6, -1, dtype=np.int64)
    for d in range(6):
        if pt == PT_GRASSHOPPER:
            cur = NEIGHBORS[fc, d]
            if cur >= 0 and heights[cur] != 0:
                while cur >= 0 and heights[cur] != 0:
                    cur = NEIGHBORS[cur, d]
                if cur >= 0:
                    dest_cell[d] = cur
                    dest_bi[d] = -1  # jumped-to empty cell → no board token
        else:
            nb = NEIGHBORS[fc, d]
            if nb >= 0:
                dest_cell[d] = nb
                if heights[nb] != 0:
                    dest_bi[d] = cell_to_board_idx[nb]
    return dest_cell, dest_bi


def _throw_dest_for_thrower(pb: int) -> np.ndarray:
    """Return (30,) int32 dest cells for each throw slot of thrower at cell `pb`."""
    dest = np.full(30, -1, dtype=np.int32)
    for slot in range(30):
        td = slot // 5
        dd_c = slot % 5
        dd = dd_c if dd_c < td else dd_c + 1
        d = NEIGHBORS[pb, dd]
        if d >= 0:
            dest[slot] = d
    return dest


# HiveState.hands is uint8[2][8] stored at byte 3396 (just before `turn`
# at TURN_OFFSET=3412). Derived from struct layout in hive_state.cuh.
_HANDS_OFFSET = 3396


def _decode_hands(state_bytes: np.ndarray) -> np.ndarray:
    """Return (2, 8) uint8 hand-count array from raw HiveState bytes."""
    return state_bytes[_HANDS_OFFSET:_HANDS_OFFSET + 16].reshape(2, 8)


def _hand_token_positions(
    state_bytes: np.ndarray, board_n: int,
) -> np.ndarray:
    """Match state_encoder.cuh Step 4: for c in 0..1, p in 0..7, if hands[c][p]>0
    emit a token at pos = 1 + board_n + running_k. Returns (16,) int64."""
    hands = _decode_hands(state_bytes)
    out = np.full(16, -1, dtype=np.int64)
    running = 0
    for c in range(2):
        for p in range(8):
            if int(hands[c, p]) > 0:
                out[c * 8 + p] = 1 + board_n + running
                running += 1
    return out


# ── CPU path ──────────────────────────────────────────────────────────────

def build_head_inputs_from_states(
    state_bytes_cpu: np.ndarray,       # (B, SIZEOF_HIVE_STATE) uint8
    board_h: torch.Tensor,             # (B, MB, d)  from trunk
    cls_h:   torch.Tensor,             # (B, d)
    full_h:  torch.Tensor,             # (B, S, d)  from trunk
) -> tuple[PRSv2HeadInputs, list[SlotMapper]]:
    """CPU reference: build head inputs for a batch.

    Returns (inputs, per-batch SlotMappers). The mappers are kept so the
    trainer/orchestrator can still call `map_legal_moves` for slot-of-legal.
    """
    B = state_bytes_cpu.shape[0]
    device = board_h.device

    dir_idx_np      = np.full((B, N_DIR_PIECES),   -1, dtype=np.int64)
    throw_idx_np    = np.full((B, N_THROW_PIECES), -1, dtype=np.int64)
    long_idx_np     = np.full((B, N_LONG_PIECES),  -1, dtype=np.int64)
    move_nbrs_np    = np.full((B, C_MOVE, 6),      -1, dtype=np.int64)
    place_nbrs_np   = np.full((B, C_HAND, 6),      -1, dtype=np.int64)
    move_mask_np    = np.zeros((B, C_MOVE), dtype=bool)
    place_mask_np   = np.zeros((B, C_HAND), dtype=bool)
    move_cellid_np  = np.full((B, C_MOVE),         -1, dtype=np.int32)
    place_cellid_np = np.full((B, C_HAND),         -1, dtype=np.int32)
    dir_dest_np     = np.full((B, 8, 6),           -1, dtype=np.int32)
    dir_destb_np    = np.full((B, 8, 6),           -1, dtype=np.int64)
    throw_dest_np   = np.full((B, 2, 30),          -1, dtype=np.int32)
    hand_tokidx_np  = np.full((B, 16),             -1, dtype=np.int64)
    current_color_np = np.zeros(B, dtype=np.int64)

    # Mapping: (piece-type, rank-in-type) → slot index in the 8-wide dir layout
    dir_layout = [
        (PT_QUEEN, 0), (PT_BEETLE, 0), (PT_BEETLE, 1),
        (PT_GRASSHOPPER, 0), (PT_GRASSHOPPER, 1), (PT_GRASSHOPPER, 2),
        (PT_PILLBUG, 0), (PT_MOSQUITO, 0),
    ]

    mappers: list[SlotMapper] = []
    for b in range(B):
        sb = state_bytes_cpu[b]
        mapper = SlotMapper(sb)
        mappers.append(mapper)
        heights = mapper.heights
        occ_cells = _occupied_cells_ascending(heights)
        current_color_np[b] = mapper.color

        di, ti, li = _piece_instance_to_board_idx(mapper, occ_cells)
        dir_idx_np[b] = di
        throw_idx_np[b] = ti
        long_idx_np[b] = li

        mn, mm, mci = _cell_neighbor_board_indices(mapper.move_cells, occ_cells, C_MOVE)
        pn, pm, pci = _cell_neighbor_board_indices(mapper.place_cells, occ_cells, C_HAND)
        move_nbrs_np[b]   = mn
        move_mask_np[b]   = mm
        move_cellid_np[b] = mci
        place_nbrs_np[b]   = pn
        place_mask_np[b]   = pm
        place_cellid_np[b] = pci

        # dir-destination per piece slot
        for slot_i, (pt, rank) in enumerate(dir_layout):
            cells = mapper.by_type.get(pt, [])
            if rank < len(cells):
                fc = int(cells[rank])
                dc, db = _dir_dest_for_piece(fc, pt, heights, occ_cells)
                dir_dest_np[b, slot_i]  = dc
                dir_destb_np[b, slot_i] = db

        # throw-destination per thrower (P at slot 0, M at slot 1)
        for slot_i, pt in enumerate((PT_PILLBUG, PT_MOSQUITO)):
            cells = mapper.by_type.get(pt, [])
            if cells:
                pb = int(cells[0])
                throw_dest_np[b, slot_i] = _throw_dest_for_thrower(pb)

        # hand-token positions
        hand_tokidx_np[b] = _hand_token_positions(sb, len(occ_cells))

    to_dev = lambda a, dt=torch.long: torch.from_numpy(a).to(device=device, dtype=dt)
    inp = PRSv2HeadInputs(
        board_h            = board_h,
        cls_h              = cls_h,
        full_h             = full_h,
        dir_piece_idx      = to_dev(dir_idx_np),
        throw_piece_idx    = to_dev(throw_idx_np),
        long_piece_idx     = to_dev(long_idx_np),
        move_nbrs          = to_dev(move_nbrs_np),
        place_nbrs         = to_dev(place_nbrs_np),
        move_mask          = torch.from_numpy(move_mask_np).to(device=device),
        place_mask         = torch.from_numpy(place_mask_np).to(device=device),
        move_cell_ids      = to_dev(move_cellid_np, dt=torch.int32),
        place_cell_ids     = to_dev(place_cellid_np, dt=torch.int32),
        dir_dest_cell      = to_dev(dir_dest_np, dt=torch.int32),
        dir_dest_board_idx = to_dev(dir_destb_np),
        throw_dest_cell    = to_dev(throw_dest_np, dt=torch.int32),
        hand_token_idx     = to_dev(hand_tokidx_np),
        current_color      = to_dev(current_color_np),
    )
    return inp, mappers


# ── GPU path ──────────────────────────────────────────────────────────────

def build_head_inputs_from_kernel(
    board_h: torch.Tensor,           # (B, MB, d)
    cls_h:   torch.Tensor,           # (B, d)
    full_h:  torch.Tensor,           # (B, S, d)
    kernel_out: tuple,               # 15-tuple from ext.prs_v2_classify_batch
) -> tuple[PRSv2HeadInputs, torch.Tensor]:
    """Build head inputs from CUDA `prs_v2_classify_batch` output.

    Returns (PRSv2HeadInputs, slot_of_legal int32 (B, MAX_L)).
    """
    (dir_piece_idx, throw_piece_idx, long_piece_idx,
     move_nbrs, place_nbrs, move_mask, place_mask,
     current_color, slot_of_legal,
     move_cell_ids, place_cell_ids,
     dir_dest_cell, dir_dest_board_idx, throw_dest_cell,
     hand_token_idx) = kernel_out
    inp = PRSv2HeadInputs(
        board_h            = board_h,
        cls_h              = cls_h,
        full_h             = full_h,
        dir_piece_idx      = dir_piece_idx,
        throw_piece_idx    = throw_piece_idx,
        long_piece_idx     = long_piece_idx,
        move_nbrs          = move_nbrs,
        place_nbrs         = place_nbrs,
        move_mask          = move_mask,
        place_mask         = place_mask,
        move_cell_ids      = move_cell_ids,
        place_cell_ids     = place_cell_ids,
        dir_dest_cell      = dir_dest_cell,
        dir_dest_board_idx = dir_dest_board_idx,
        throw_dest_cell    = throw_dest_cell,
        hand_token_idx     = hand_token_idx,
        current_color      = current_color,
    )
    return inp, slot_of_legal
