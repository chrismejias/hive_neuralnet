/**
 * prs_v2_slot.cuh — CUDA port of hive_prs.slot_map + prs_v2_bridge.
 *
 * Produces, for a batch of HiveStates and their legal-move arrays:
 *   * slot_of_legal[B, max_legal]  int32 — slot index in [0, 813) or -1
 *   * dir_piece_idx[B, 8]          int64 — board-token idx per dir piece
 *   * throw_piece_idx[B, 2]        int64
 *   * long_piece_idx[B, 7]         int64
 *   * move_nbrs[B, 64, 6]          int64 — adjacent board-token idx per move cell
 *   * place_nbrs[B, 32, 6]         int64
 *   * move_mask[B, 64]             bool
 *   * place_mask[B, 32]            bool
 *   * current_color[B]             int64
 *   * move_cell_ids[B, 64]         int32 — absolute cell index (0..529) per move-cell
 *   * place_cell_ids[B, 32]        int32 — absolute cell index per place-cell
 *   * dir_dest_cell[B, 8, 6]       int32 — dest cell for (dir-piece-slot, dir); -1 invalid
 *   * dir_dest_board_idx[B, 8, 6]  int64 — board-token idx of dest if occupied, -1 otherwise
 *   * throw_dest_cell[B, 2, 30]    int32 — dest cell for (thrower-slot, throw-slot); -1 invalid
 *   * hand_token_idx[B, 16]        int64 — trunk seq position of hand token per
 *                                  (color*8 + type) slot; -1 if hand count==0
 *
 * Layout matches hive_prs.slot_map exactly; see that module for semantics.
 *
 * Kernel topology: one block per state, 32 threads per block.
 *   - Thread 0 builds per-state aux arrays in shared memory.
 *   - All 32 threads cooperate (strided) on legal-move classification.
 */
#pragma once

#include <cstdint>
#include "hex_grid.cuh"
#include "hive_state.cuh"

namespace hive_gpu {

// ── Slot-space constants (must match Python slot_map.py) ───────────

constexpr int PRS_V2_C_MOVE       = 64;
constexpr int PRS_V2_C_HAND       = 32;
constexpr int PRS_V2_DIR_OFFSET   = 0;
constexpr int PRS_V2_THROW_OFFSET = 48;
constexpr int PRS_V2_LONG_OFFSET  = 108;
constexpr int PRS_V2_HAND_OFFSET  = 556;
constexpr int PRS_V2_PASS_SLOT    = 812;
constexpr int PRS_V2_N_SLOTS      = 813;

constexpr int PRS_V2_N_DIR_PIECES   = 8;
constexpr int PRS_V2_N_THROW_PIECES = 2;
constexpr int PRS_V2_N_LONG_PIECES  = 7;

#ifdef __CUDACC__

// Per-piece-type bases (mirror _DIR_BASE / _LONG_BASE / _THROW_BASE in slot_map.py).
// Index by PieceType enum (1..8); 0 entry is unused.
__device__ __forceinline__ int prs_v2_dir_base(int pt) {
    switch (pt) {
        case PT_QUEEN:       return 0;
        case PT_BEETLE:      return 1;
        case PT_GRASSHOPPER: return 3;
        case PT_PILLBUG:     return 6;
        case PT_MOSQUITO:    return 7;
        default:             return -1;
    }
}
__device__ __forceinline__ int prs_v2_dir_slots(int pt) {
    switch (pt) {
        case PT_QUEEN:       return 1;
        case PT_BEETLE:      return 2;
        case PT_GRASSHOPPER: return 3;
        case PT_PILLBUG:     return 1;
        case PT_MOSQUITO:    return 1;
        default:             return 0;
    }
}
__device__ __forceinline__ int prs_v2_long_base(int pt) {
    switch (pt) {
        case PT_ANT:      return 0;
        case PT_SPIDER:   return 3;
        case PT_LADYBUG:  return 5;
        case PT_MOSQUITO: return 6;
        default:          return -1;
    }
}
__device__ __forceinline__ int prs_v2_long_slots(int pt) {
    switch (pt) {
        case PT_ANT:      return 3;
        case PT_SPIDER:   return 2;
        case PT_LADYBUG:  return 1;
        case PT_MOSQUITO: return 1;
        default:          return 0;
    }
}
__device__ __forceinline__ int prs_v2_throw_base(int pt) {
    switch (pt) {
        case PT_PILLBUG:  return 0;
        case PT_MOSQUITO: return 1;
        default:          return -1;
    }
}

// Hex direction from `from` to `to` (0..5) if adjacent, else -1.
__device__ __forceinline__ int prs_v2_direction_of(int from, int to) {
    #pragma unroll
    for (int d = 0; d < NUM_DIRS; d++) {
        if (NEIGHBORS[from][d] == to) return d;
    }
    return -1;
}

// Encode pillbug throw: (target_dir, dest_dir_compacted) -> [0, 30)
// dest_dir is in 0..5 minus the target_dir (5 remaining slots).
__device__ __forceinline__ int prs_v2_throw_slot_idx(int pb, int fc, int tc) {
    int td = prs_v2_direction_of(pb, fc);
    int dd = prs_v2_direction_of(pb, tc);
    if (td < 0 || dd < 0 || td == dd) return -1;
    int dd_compact = (dd < td) ? dd : (dd - 1);
    return td * 5 + dd_compact;
}

// Grasshopper jump: starting from `fc` in direction `d`, walk over occupied
// cells; if first empty cell == `tc`, this is the direction (0..5) of the jump.
// Returns the matching direction or -1.
__device__ __forceinline__ int prs_v2_grasshopper_dir_of(
    const uint8_t* heights, int fc, int tc
) {
    #pragma unroll
    for (int d = 0; d < NUM_DIRS; d++) {
        int cur = NEIGHBORS[fc][d];
        if (cur < 0 || heights[cur] == 0) continue;  // must jump ≥1 piece
        while (cur >= 0 && heights[cur] != 0) {
            cur = NEIGHBORS[cur][d];
        }
        if (cur == tc) return d;
    }
    return -1;
}

// ── Per-state auxiliary tables (lives in shared memory) ────────────

struct PrsV2Aux {
    // Top-of-stack cell metadata
    int8_t   top_color[NUM_CELLS];   // 0=white, 1=black, -1=empty
    uint8_t  top_type[NUM_CELLS];    // PieceType (0=empty, 1..8)
    // Mappings cell → token-index
    int16_t  cell_to_board_idx[NUM_CELLS];   // pos in ascending occupied-cell list, -1 if empty
    int16_t  cell_to_move_idx[NUM_CELLS];    // pos in move_cells, -1 if not present
    int16_t  cell_to_place_idx[NUM_CELLS];   // pos in place_cells, -1 if not present
    // Cell enumerations
    int16_t  move_cells[PRS_V2_C_MOVE];      // ascending; padding = -1
    int16_t  place_cells[PRS_V2_C_HAND];
    int16_t  num_move_cells;
    int16_t  num_place_cells;
    // Current-player piece-instance lists (for slot ranking).
    // pieces_by_type[pt][rank] = cell, sorted ascending; -1 = absent.
    int16_t  pieces_by_type[NUM_PIECE_TYPES + 1][3];
    uint8_t  num_pieces_by_type[NUM_PIECE_TYPES + 1];
    int      current_color;  // 0 or 1
    int16_t  board_n;        // total count of occupied cells (= number of board tokens)
};

// Find the rank of `cell` within pieces_by_type[pt]; -1 if not present.
__device__ __forceinline__ int prs_v2_inst_idx(
    const PrsV2Aux& aux, int pt, int cell
) {
    int n = aux.num_pieces_by_type[pt];
    #pragma unroll
    for (int r = 0; r < 3; r++) {
        if (r >= n) break;
        if (aux.pieces_by_type[pt][r] == (int16_t)cell) return r;
    }
    return -1;
}

// Build the aux struct (single-thread; called only from thread 0).
__device__ inline void prs_v2_build_aux(
    const HiveState& s, PrsV2Aux& aux
) {
    // Decode top_color / top_type from heights + pieces
    for (int c = 0; c < NUM_CELLS; c++) {
        int h = s.height[c];
        if (h == 0) {
            aux.top_color[c] = -1;
            aux.top_type[c]  = 0;
        } else {
            uint8_t packed = s.pieces[h - 1][c];
            aux.top_type[c]  = (uint8_t)(packed & 0x0F);
            aux.top_color[c] = (int8_t)((packed >> 4) & 0x01);
        }
    }

    // Build cell_to_board_idx + ascending occupied-cell enumeration
    int board_n = 0;
    for (int c = 0; c < NUM_CELLS; c++) {
        if (s.height[c] > 0) {
            aux.cell_to_board_idx[c] = (int16_t)board_n;
            board_n++;
        } else {
            aux.cell_to_board_idx[c] = -1;
        }
    }
    aux.board_n = (int16_t)board_n;

    // Enumerate move_cells (empty, adjacent to ≥1 occupied)
    aux.num_move_cells = 0;
    for (int c = 0; c < NUM_CELLS; c++) {
        aux.cell_to_move_idx[c] = -1;
    }
    for (int c = 0; c < NUM_CELLS; c++) {
        if (s.height[c] != 0) continue;
        bool any_occ = false;
        #pragma unroll
        for (int d = 0; d < NUM_DIRS; d++) {
            int nb = NEIGHBORS[c][d];
            if (nb >= 0 && s.height[nb] != 0) { any_occ = true; break; }
        }
        if (any_occ && aux.num_move_cells < PRS_V2_C_MOVE) {
            int idx = aux.num_move_cells;
            aux.move_cells[idx] = (int16_t)c;
            aux.cell_to_move_idx[c] = (int16_t)idx;
            aux.num_move_cells++;
        }
    }
    for (int i = aux.num_move_cells; i < PRS_V2_C_MOVE; i++) {
        aux.move_cells[i] = -1;
    }

    // current color
    aux.current_color = (int)(s.turn & 1);
    int my = aux.current_color;
    int opp = 1 - my;

    // Enumerate place_cells (empty, adj to ≥1 own-color, none opp-color)
    aux.num_place_cells = 0;
    for (int c = 0; c < NUM_CELLS; c++) {
        aux.cell_to_place_idx[c] = -1;
    }
    for (int c = 0; c < NUM_CELLS; c++) {
        if (s.height[c] != 0) continue;
        bool has_own = false;
        bool has_opp = false;
        #pragma unroll
        for (int d = 0; d < NUM_DIRS; d++) {
            int nb = NEIGHBORS[c][d];
            if (nb < 0 || s.height[nb] == 0) continue;
            int8_t tc = aux.top_color[nb];
            if (tc == my) has_own = true;
            else if (tc == opp) has_opp = true;
        }
        if (has_own && !has_opp && aux.num_place_cells < PRS_V2_C_HAND) {
            int idx = aux.num_place_cells;
            aux.place_cells[idx] = (int16_t)c;
            aux.cell_to_place_idx[c] = (int16_t)idx;
            aux.num_place_cells++;
        }
    }
    for (int i = aux.num_place_cells; i < PRS_V2_C_HAND; i++) {
        aux.place_cells[i] = -1;
    }

    // pieces_by_type for current color (ascending cell order, capped at 3)
    for (int pt = 0; pt <= NUM_PIECE_TYPES; pt++) {
        aux.num_pieces_by_type[pt] = 0;
        aux.pieces_by_type[pt][0] = -1;
        aux.pieces_by_type[pt][1] = -1;
        aux.pieces_by_type[pt][2] = -1;
    }
    for (int c = 0; c < NUM_CELLS; c++) {
        if (s.height[c] == 0) continue;
        if (aux.top_color[c] != my) continue;
        int pt = aux.top_type[c];
        if (pt < 1 || pt > NUM_PIECE_TYPES) continue;
        int n = aux.num_pieces_by_type[pt];
        if (n < 3) {
            aux.pieces_by_type[pt][n] = (int16_t)c;
            aux.num_pieces_by_type[pt] = (uint8_t)(n + 1);
        }
    }
}

// Detect a pillbug-style throw: returns the thrower cell or -1.
__device__ inline int prs_v2_find_thrower_cell(
    const PrsV2Aux& aux, const uint8_t* heights, int fc, int tc, int my_color
) {
    // Common neighbors of fc and tc
    int16_t fc_n[NUM_DIRS], tc_n[NUM_DIRS];
    #pragma unroll
    for (int d = 0; d < NUM_DIRS; d++) {
        fc_n[d] = NEIGHBORS[fc][d];
        tc_n[d] = NEIGHBORS[tc][d];
    }
    #pragma unroll
    for (int i = 0; i < NUM_DIRS; i++) {
        int cand = fc_n[i];
        if (cand < 0) continue;
        bool common = false;
        #pragma unroll
        for (int j = 0; j < NUM_DIRS; j++) {
            if (tc_n[j] == cand) { common = true; break; }
        }
        if (!common) continue;
        if (heights[cand] == 0) continue;
        if (aux.top_color[cand] != (int8_t)my_color) continue;
        int ttype = aux.top_type[cand];
        if (ttype == PT_PILLBUG) return cand;
        if (ttype == PT_MOSQUITO) {
            // ground-level mosquito adjacent to a pillbug also throws
            if (heights[cand] != 1) continue;
            #pragma unroll
            for (int d = 0; d < NUM_DIRS; d++) {
                int nb = NEIGHBORS[cand][d];
                if (nb < 0 || heights[nb] == 0) continue;
                if (aux.top_type[nb] == PT_PILLBUG) return cand;
            }
        }
    }
    return -1;
}

// Per-piece dir/long index (return -1 if piece-instance is out of slots / absent).
__device__ __forceinline__ int prs_v2_dir_piece_idx(
    const PrsV2Aux& aux, int pt, int fc
) {
    int base = prs_v2_dir_base(pt);
    if (base < 0) return -1;
    int rank = prs_v2_inst_idx(aux, pt, fc);
    if (rank < 0 || rank >= prs_v2_dir_slots(pt)) return -1;
    return base + rank;
}
__device__ __forceinline__ int prs_v2_long_piece_idx(
    const PrsV2Aux& aux, int pt, int fc
) {
    int base = prs_v2_long_base(pt);
    if (base < 0) return -1;
    int rank = prs_v2_inst_idx(aux, pt, fc);
    if (rank < 0 || rank >= prs_v2_long_slots(pt)) return -1;
    return base + rank;
}

// Classify a single legal move into its slot. Mirrors SlotMapper.classify.
__device__ inline int prs_v2_classify(
    const PrsV2Aux& aux, const uint8_t* heights, const GPUMove& m
) {
    if (m.type == MOVE_PASS) return PRS_V2_PASS_SLOT;

    int pt = (int)m.piece_type;
    int fc = (int)m.from_cell;
    int tc = (int)m.to_cell;

    if (m.type == MOVE_PLACE) {
        int cell_tok = (tc >= 0 && tc < NUM_CELLS) ? aux.cell_to_place_idx[tc] : -1;
        if (cell_tok < 0) {
            cell_tok = (tc >= 0 && tc < NUM_CELLS) ? aux.cell_to_move_idx[tc] : -1;
            if (cell_tok < 0) cell_tok = 0;  // first-placement fallback
            if (cell_tok >= PRS_V2_C_HAND) return -1;
        }
        return PRS_V2_HAND_OFFSET + (pt - 1) * PRS_V2_C_HAND + cell_tok;
    }

    // MOVE_MOVE: throw detection first
    int pb = prs_v2_find_thrower_cell(aux, heights, fc, tc, aux.current_color);
    if (pb >= 0) {
        int thrower_type = aux.top_type[pb];
        int tbase = prs_v2_throw_base(thrower_type);
        if (tbase < 0) return -1;
        int slot = prs_v2_throw_slot_idx(pb, fc, tc);
        if (slot < 0) return -1;
        return PRS_V2_THROW_OFFSET + tbase * 30 + slot;
    }

    if (pt == PT_QUEEN || pt == PT_BEETLE || pt == PT_PILLBUG) {
        int d = prs_v2_direction_of(fc, tc);
        if (d < 0) return -1;
        int pidx = prs_v2_dir_piece_idx(aux, pt, fc);
        if (pidx < 0) return -1;
        return PRS_V2_DIR_OFFSET + pidx * 6 + d;
    }

    if (pt == PT_GRASSHOPPER) {
        int d = prs_v2_grasshopper_dir_of(heights, fc, tc);
        if (d < 0) return -1;
        int pidx = prs_v2_dir_piece_idx(aux, pt, fc);
        if (pidx < 0) return -1;
        return PRS_V2_DIR_OFFSET + pidx * 6 + d;
    }

    if (pt == PT_MOSQUITO) {
        int d = prs_v2_direction_of(fc, tc);
        if (d >= 0) {
            int pidx = prs_v2_dir_piece_idx(aux, PT_MOSQUITO, fc);
            if (pidx < 0) return -1;
            return PRS_V2_DIR_OFFSET + pidx * 6 + d;
        }
        int cell_tok = (tc >= 0 && tc < NUM_CELLS) ? aux.cell_to_move_idx[tc] : -1;
        if (cell_tok < 0) return -1;
        int pidx = prs_v2_long_piece_idx(aux, PT_MOSQUITO, fc);
        if (pidx < 0) return -1;
        return PRS_V2_LONG_OFFSET + pidx * PRS_V2_C_MOVE + cell_tok;
    }

    if (pt == PT_ANT || pt == PT_SPIDER || pt == PT_LADYBUG) {
        int cell_tok = (tc >= 0 && tc < NUM_CELLS) ? aux.cell_to_move_idx[tc] : -1;
        if (cell_tok < 0) return -1;
        int pidx = prs_v2_long_piece_idx(aux, pt, fc);
        if (pidx < 0) return -1;
        return PRS_V2_LONG_OFFSET + pidx * PRS_V2_C_MOVE + cell_tok;
    }

    return -1;
}

// ── Main kernel: one block per state, 32 threads per block ─────────

__global__ void prs_v2_classify_kernel(
    const HiveState* __restrict__ states,
    const GPUMove*   __restrict__ legal_moves,    // [B, max_legal]
    const int*       __restrict__ num_legal,      // [B]
    int B, int max_legal,
    // Outputs
    int64_t* dir_piece_idx,       // [B, 8]
    int64_t* throw_piece_idx,     // [B, 2]
    int64_t* long_piece_idx,      // [B, 7]
    int64_t* move_nbrs,           // [B, 64, 6]
    int64_t* place_nbrs,          // [B, 32, 6]
    bool*    move_mask,           // [B, 64]
    bool*    place_mask,          // [B, 32]
    int64_t* current_color,       // [B]
    int32_t* slot_of_legal,       // [B, max_legal]
    int32_t* move_cell_ids,       // [B, 64]       absolute cell index; -1 pad
    int32_t* place_cell_ids,      // [B, 32]       absolute cell index; -1 pad
    int32_t* dir_dest_cell,       // [B, 8, 6]     dest cell; -1 invalid
    int64_t* dir_dest_board_idx,  // [B, 8, 6]     board-token idx if occupied; -1 otherwise
    int32_t* throw_dest_cell,     // [B, 2, 30]    dest cell; -1 invalid
    int64_t* hand_token_idx       // [B, 16]       trunk seq pos per (color*8+type); -1 if count=0
) {
    int b = blockIdx.x;
    if (b >= B) return;

    __shared__ PrsV2Aux aux;

    const HiveState& s = states[b];
    const uint8_t* heights = s.height;

    // ── Phase 1: build aux (thread 0 only — single-threaded for simplicity) ──
    if (threadIdx.x == 0) {
        prs_v2_build_aux(s, aux);
    }
    __syncthreads();

    // ── Phase 2: head-input arrays (thread 0 — small, branchy) ──
    if (threadIdx.x == 0) {
        // dir_piece_idx (8 entries: Q1, B1, B2, G1, G2, G3, P1, M1)
        // Each entry: rank-i piece's cell -> board_idx, or -1
        int64_t* dpi = dir_piece_idx + (int64_t)b * 8;
        const int dir_pts[5] = {PT_QUEEN, PT_BEETLE, PT_GRASSHOPPER, PT_PILLBUG, PT_MOSQUITO};
        for (int i = 0; i < 8; i++) dpi[i] = -1;
        for (int k = 0; k < 5; k++) {
            int pt = dir_pts[k];
            int base = prs_v2_dir_base(pt);
            int slots = prs_v2_dir_slots(pt);
            int n = aux.num_pieces_by_type[pt];
            for (int r = 0; r < slots; r++) {
                if (r < n) {
                    int cell = aux.pieces_by_type[pt][r];
                    dpi[base + r] = (int64_t)aux.cell_to_board_idx[cell];
                }
            }
        }

        // throw_piece_idx (2: P1, M1) — first instance only
        int64_t* tpi = throw_piece_idx + (int64_t)b * 2;
        tpi[0] = tpi[1] = -1;
        const int throw_pts[2] = {PT_PILLBUG, PT_MOSQUITO};
        for (int k = 0; k < 2; k++) {
            int pt = throw_pts[k];
            int base = prs_v2_throw_base(pt);
            if (aux.num_pieces_by_type[pt] > 0) {
                int cell = aux.pieces_by_type[pt][0];
                tpi[base] = (int64_t)aux.cell_to_board_idx[cell];
            }
        }

        // long_piece_idx (7: A1-3, S1-2, L1, M1)
        int64_t* lpi = long_piece_idx + (int64_t)b * 7;
        for (int i = 0; i < 7; i++) lpi[i] = -1;
        const int long_pts[4] = {PT_ANT, PT_SPIDER, PT_LADYBUG, PT_MOSQUITO};
        for (int k = 0; k < 4; k++) {
            int pt = long_pts[k];
            int base = prs_v2_long_base(pt);
            int slots = prs_v2_long_slots(pt);
            int n = aux.num_pieces_by_type[pt];
            for (int r = 0; r < slots; r++) {
                if (r < n) {
                    int cell = aux.pieces_by_type[pt][r];
                    lpi[base + r] = (int64_t)aux.cell_to_board_idx[cell];
                }
            }
        }

        current_color[b] = (int64_t)aux.current_color;

        // ── dir_dest_cell[B, 8, 6] + dir_dest_board_idx[B, 8, 6] ──
        // Ordered: Q1, B1, B2, G1, G2, G3, P1, M1 (mirror dir_piece_idx).
        int32_t* ddc = dir_dest_cell      + (int64_t)b * 8 * NUM_DIRS;
        int64_t* ddb = dir_dest_board_idx + (int64_t)b * 8 * NUM_DIRS;
        for (int i = 0; i < 8 * NUM_DIRS; i++) {
            ddc[i] = -1;
            ddb[i] = -1;
        }
        const int dir_pts_all[5] = {PT_QUEEN, PT_BEETLE, PT_GRASSHOPPER, PT_PILLBUG, PT_MOSQUITO};
        for (int k = 0; k < 5; k++) {
            int pt = dir_pts_all[k];
            int base = prs_v2_dir_base(pt);
            int slots = prs_v2_dir_slots(pt);
            int n = aux.num_pieces_by_type[pt];
            for (int r = 0; r < slots; r++) {
                if (r >= n) continue;
                int fc = aux.pieces_by_type[pt][r];
                for (int d = 0; d < NUM_DIRS; d++) {
                    int dest = -1;
                    if (pt == PT_GRASSHOPPER) {
                        // Walk in direction d until we leave the occupied stretch.
                        int cur = NEIGHBORS[fc][d];
                        if (cur >= 0 && heights[cur] != 0) {
                            while (cur >= 0 && heights[cur] != 0) {
                                cur = NEIGHBORS[cur][d];
                            }
                            dest = cur;  // first empty cell (or -1 if off-board)
                        }
                    } else {
                        // Q / B / P / M-close: adjacent neighbor.
                        dest = NEIGHBORS[fc][d];
                    }
                    int slot = (base + r) * NUM_DIRS + d;
                    if (dest >= 0) {
                        ddc[slot] = (int32_t)dest;
                        ddb[slot] = (heights[dest] != 0)
                            ? (int64_t)aux.cell_to_board_idx[dest]
                            : (int64_t)-1;
                    }
                }
            }
        }

        // ── hand_token_idx[B, 16] ──
        // Mirror state_encoder: for c in 0..1, for p in 0..7, if hands[c][p] > 0
        // emit a hand token. Trunk seq position = 1 + board_n + running_k.
        int64_t* hti = hand_token_idx + (int64_t)b * 16;
        for (int i = 0; i < 16; i++) hti[i] = -1;
        int running_k = 0;
        for (int c = 0; c < 2; c++) {
            for (int p = 0; p < NUM_PIECE_TYPES; p++) {
                if (s.hands[c][p] > 0) {
                    hti[c * 8 + p] = (int64_t)(1 + aux.board_n + running_k);
                    running_k++;
                }
            }
        }

        // ── throw_dest_cell[B, 2, 30] ──
        // For each (thrower P1, M1) × (30 throw slots), destination = NEIGHBORS[pb, dd].
        // dd is decoded from slot = td * 5 + dd_compact; dd = dd_compact < td ? dd_compact : dd_compact+1.
        int32_t* tdc = throw_dest_cell + (int64_t)b * 2 * 30;
        for (int i = 0; i < 2 * 30; i++) tdc[i] = -1;
        const int throw_pts_all[2] = {PT_PILLBUG, PT_MOSQUITO};
        for (int k = 0; k < 2; k++) {
            int pt = throw_pts_all[k];
            int tbase = prs_v2_throw_base(pt);
            if (aux.num_pieces_by_type[pt] == 0) continue;
            int pb = aux.pieces_by_type[pt][0];
            for (int slot = 0; slot < 30; slot++) {
                int td = slot / 5;
                int dd_c = slot % 5;
                int dd = (dd_c < td) ? dd_c : (dd_c + 1);
                int dest = NEIGHBORS[pb][dd];
                tdc[tbase * 30 + slot] = (int32_t)dest;
            }
        }
    }
    __syncthreads();

    // ── Phase 3: cell-neighbor tables (parallel over cells) ──
    // move_nbrs[b, c, d] for c in [0, 64)
    int64_t* mnp = move_nbrs      + (int64_t)b * PRS_V2_C_MOVE * NUM_DIRS;
    bool*    mmp = move_mask      + (int64_t)b * PRS_V2_C_MOVE;
    int32_t* mci = move_cell_ids  + (int64_t)b * PRS_V2_C_MOVE;
    for (int c = threadIdx.x; c < PRS_V2_C_MOVE; c += blockDim.x) {
        int cell = aux.move_cells[c];
        bool active = (c < aux.num_move_cells);
        mmp[c] = active;
        mci[c] = active ? (int32_t)cell : (int32_t)-1;
        for (int d = 0; d < NUM_DIRS; d++) {
            int v = -1;
            if (active) {
                int nb = NEIGHBORS[cell][d];
                if (nb >= 0) v = aux.cell_to_board_idx[nb];
            }
            mnp[(int64_t)c * NUM_DIRS + d] = (int64_t)v;
        }
    }
    int64_t* pnp = place_nbrs     + (int64_t)b * PRS_V2_C_HAND * NUM_DIRS;
    bool*    pmp = place_mask     + (int64_t)b * PRS_V2_C_HAND;
    int32_t* pci = place_cell_ids + (int64_t)b * PRS_V2_C_HAND;
    for (int c = threadIdx.x; c < PRS_V2_C_HAND; c += blockDim.x) {
        int cell = aux.place_cells[c];
        bool active = (c < aux.num_place_cells);
        pmp[c] = active;
        pci[c] = active ? (int32_t)cell : (int32_t)-1;
        for (int d = 0; d < NUM_DIRS; d++) {
            int v = -1;
            if (active) {
                int nb = NEIGHBORS[cell][d];
                if (nb >= 0) v = aux.cell_to_board_idx[nb];
            }
            pnp[(int64_t)c * NUM_DIRS + d] = (int64_t)v;
        }
    }

    // ── Phase 4: classify legal moves (strided across threads) ──
    int n_l = num_legal[b];
    const GPUMove* my_moves = legal_moves + (int64_t)b * max_legal;
    int32_t* my_slots = slot_of_legal + (int64_t)b * max_legal;
    for (int i = threadIdx.x; i < max_legal; i += blockDim.x) {
        int slot = -1;
        if (i < n_l) {
            slot = prs_v2_classify(aux, heights, my_moves[i]);
        }
        my_slots[i] = (int32_t)slot;
    }
}

#endif  // __CUDACC__

}  // namespace hive_gpu
