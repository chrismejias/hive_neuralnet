/**
 * hex_grid.cuh — Pre-computed hex grid constants for GPU move generation.
 *
 * Uses a 23×23 centroid-centered grid (larger than the 13×13 encoder grid
 * to avoid re-centering). Cell index = row * 23 + col (0–528).
 *
 * Flat-top hex orientation with 6 directions: E, NE, NW, W, SW, SE.
 * Axial coordinates (q, r) map to grid as:
 *   col = q - center_q + 11
 *   row = r - center_r + 11
 *
 * Direction offsets in (dq, dr):
 *   E=(+1,0), NE=(+1,-1), NW=(0,-1), W=(-1,0), SW=(-1,+1), SE=(0,+1)
 * Which translates to grid offsets (dcol, drow):
 *   E=(+1,0), NE=(+1,-1), NW=(0,-1), W=(-1,0), SW=(-1,+1), SE=(0,+1)
 */

#pragma once

#include <cstdint>

namespace hive_gpu {

constexpr int BOARD_SIZE = 23;
constexpr int HALF_BOARD = 11;
constexpr int NUM_CELLS  = BOARD_SIZE * BOARD_SIZE;  // 529
constexpr int NUM_DIRS   = 6;

// Number of uint64_t words needed for a NUM_CELLS bitboard
constexpr int BB_WORDS = (NUM_CELLS + 63) / 64;  // 9

// Direction indices
constexpr int DIR_E  = 0;
constexpr int DIR_NE = 1;
constexpr int DIR_NW = 2;
constexpr int DIR_W  = 3;
constexpr int DIR_SW = 4;
constexpr int DIR_SE = 5;

// Direction offsets in (dcol, drow) form — same as (dq, dr)
constexpr int DIR_DCOL[6] = { +1, +1,  0, -1, -1,  0 };
constexpr int DIR_DROW[6] = {  0, -1, -1,  0, +1, +1 };

// Clockwise direction: cw[d] = (d + 5) % 6
constexpr int DIR_CW[6]  = { 5, 0, 1, 2, 3, 4 };
// Counter-clockwise direction: ccw[d] = (d + 1) % 6
constexpr int DIR_CCW[6] = { 1, 2, 3, 4, 5, 0 };
// Opposite direction: opp[d] = (d + 3) % 6
constexpr int DIR_OPP[6] = { 3, 4, 5, 0, 1, 2 };

// ── Host-side table generation ──────────────────────────────────────

/**
 * NEIGHBOR_TABLE[cell][dir] = neighbor cell index, or -1 if off-grid.
 * Generated at init time by init_hex_tables().
 */
inline int16_t HOST_NEIGHBOR_TABLE[NUM_CELLS][NUM_DIRS];

/**
 * SLIDE_TABLE[cell][dir] = {dest_cell, cw_flank_cell, ccw_flank_cell}
 * All as int16_t, -1 if off-grid.
 * Generated at init time by init_hex_tables().
 */
inline int16_t HOST_SLIDE_TABLE[NUM_CELLS][NUM_DIRS][3];

/**
 * Initialize the host-side lookup tables. Must be called once at startup.
 */
inline void init_hex_tables() {
    for (int cell = 0; cell < NUM_CELLS; cell++) {
        int row = cell / BOARD_SIZE;
        int col = cell % BOARD_SIZE;

        for (int d = 0; d < NUM_DIRS; d++) {
            int nr = row + DIR_DROW[d];
            int nc = col + DIR_DCOL[d];
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                HOST_NEIGHBOR_TABLE[cell][d] = (int16_t)(nr * BOARD_SIZE + nc);
            } else {
                HOST_NEIGHBOR_TABLE[cell][d] = -1;
            }

            // Slide flanking: dest, cw_flank, ccw_flank
            // dest = neighbor in direction d
            HOST_SLIDE_TABLE[cell][d][0] = HOST_NEIGHBOR_TABLE[cell][d];

            // CW flank = neighbor in clockwise direction
            int cw_d = DIR_CW[d];
            int cw_r = row + DIR_DROW[cw_d];
            int cw_c = col + DIR_DCOL[cw_d];
            if (cw_r >= 0 && cw_r < BOARD_SIZE && cw_c >= 0 && cw_c < BOARD_SIZE) {
                HOST_SLIDE_TABLE[cell][d][1] = (int16_t)(cw_r * BOARD_SIZE + cw_c);
            } else {
                HOST_SLIDE_TABLE[cell][d][1] = -1;
            }

            // CCW flank = neighbor in counter-clockwise direction
            int ccw_d = DIR_CCW[d];
            int ccw_r = row + DIR_DROW[ccw_d];
            int ccw_c = col + DIR_DCOL[ccw_d];
            if (ccw_r >= 0 && ccw_r < BOARD_SIZE && ccw_c >= 0 && ccw_c < BOARD_SIZE) {
                HOST_SLIDE_TABLE[cell][d][2] = (int16_t)(ccw_r * BOARD_SIZE + ccw_c);
            } else {
                HOST_SLIDE_TABLE[cell][d][2] = -1;
            }
        }
    }
}

// ── Device-side constant memory tables ──────────────────────────────

#ifdef __CUDACC__

// Defined once per .cu translation unit (only game_logic.cu includes this via CUDACC).
__constant__ int16_t NEIGHBORS[NUM_CELLS][NUM_DIRS];
__constant__ int16_t SLIDE_FLANKS[NUM_CELLS][NUM_DIRS][3];

/**
 * Copy host tables to CUDA constant memory. Call once after init_hex_tables().
 */
inline void copy_tables_to_device() {
    cudaMemcpyToSymbol(NEIGHBORS, HOST_NEIGHBOR_TABLE, sizeof(HOST_NEIGHBOR_TABLE));
    cudaMemcpyToSymbol(SLIDE_FLANKS, HOST_SLIDE_TABLE, sizeof(HOST_SLIDE_TABLE));
}

// ── Device helper functions ─────────────────────────────────────────

__device__ __forceinline__ int16_t neighbor_of(int cell, int dir) {
    return NEIGHBORS[cell][dir];
}

__device__ __forceinline__ int cell_from_grid(int row, int col) {
    return row * BOARD_SIZE + col;
}

__device__ __forceinline__ int cell_row(int cell) {
    return cell / BOARD_SIZE;
}

__device__ __forceinline__ int cell_col(int cell) {
    return cell % BOARD_SIZE;
}

// Convert hex (q, r) to cell index given center offset, returns -1 if off-grid
__device__ __forceinline__ int hex_to_cell(int q, int r, int center_q, int center_r) {
    int col = q - center_q + HALF_BOARD;
    int row = r - center_r + HALF_BOARD;
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return -1;
    return row * BOARD_SIZE + col;
}

// Convert cell index to hex (q, r)
__device__ __forceinline__ void cell_to_hex(int cell, int center_q, int center_r,
                                             int& q, int& r) {
    int col = cell % BOARD_SIZE;
    int row = cell / BOARD_SIZE;
    q = col - HALF_BOARD + center_q;
    r = row - HALF_BOARD + center_r;
}

#endif  // __CUDACC__

}  // namespace hive_gpu
