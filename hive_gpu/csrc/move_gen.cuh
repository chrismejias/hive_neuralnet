/**
 * move_gen.cuh — Legal move generation kernels for Hive on GPU.
 *
 * Implements per-piece-type move generation and full legal_moves().
 * All functions are __device__ and operate on a single HiveState.
 *
 * Piece types:
 *   Queen      — 1-step slide (max 6 destinations)
 *   Ant        — unlimited slides via BFS (max ~80 destinations)
 *   Spider     — exactly 3-step slide (max ~12 destinations)
 *   Grasshopper— straight-line jump over pieces (max 6 destinations)
 *   Beetle     — slide or climb, special elevated gate rules (max 6 destinations)
 *   Mosquito   — copies adjacent top pieces' abilities; beetle when elevated
 *   Ladybug    — 3-step: ascend → traverse → descend
 *   Pillbug    — 1-step slide + special throw (handled in generate_legal_moves)
 */

#pragma once

#include "hive_state.cuh"
#include "hex_grid.cuh"
#include "articulation.cuh"

namespace hive_gpu {

#ifdef __CUDACC__

// Max inline buffer size for ant destinations
constexpr int MAX_ANT_DESTS = 128;

enum MovegenProfileIdx {
    MGP_CALLS = 0,
    MGP_PLACEMENT_CALLS = 1,
    MGP_PLACEMENT_MOVES = 2,
    MGP_QUEEN_CALLS = 3,
    MGP_QUEEN_MOVES = 4,
    MGP_ANT_CALLS = 5,
    MGP_ANT_MOVES = 6,
    MGP_SPIDER_CALLS = 7,
    MGP_SPIDER_MOVES = 8,
    MGP_GRASSHOPPER_CALLS = 9,
    MGP_GRASSHOPPER_MOVES = 10,
    MGP_BEETLE_CALLS = 11,
    MGP_BEETLE_MOVES = 12,
    MGP_MOSQUITO_CALLS = 13,
    MGP_MOSQUITO_MOVES = 14,
    MGP_LADYBUG_CALLS = 15,
    MGP_LADYBUG_MOVES = 16,
    MGP_PILLBUG_CALLS = 17,
    MGP_PILLBUG_MOVES = 18,
    MGP_THROW_CALLS = 19,
    MGP_THROW_MOVES = 20,
    MGP_PASS_MOVES = 21,
    MGP_MAX = 22,
};

__device__ __managed__ unsigned long long MOVEGEN_PROFILE[MGP_MAX];
__device__ __managed__ bool MOVEGEN_PROFILE_ENABLED = false;

__device__ __forceinline__ void mgp_add(int idx, unsigned long long value) {
    if (MOVEGEN_PROFILE_ENABLED) {
        atomicAdd(&MOVEGEN_PROFILE[idx], value);
    }
}

// ── Slide check ─────────────────────────────────────────────────────

/**
 * Check if a ground-level slide from `from_cell` in direction `dir` is valid.
 * `exclude_cell` is treated as empty (the moving piece's original position).
 *
 * Valid slide requires:
 * 1. Destination is empty (or == exclude_cell)
 * 2. Not gate-blocked (both flanking neighbors occupied)
 * 3. Maintains contact (at least one flanking neighbor occupied)
 */
__device__ __forceinline__ bool can_slide(const HiveState& s,
                                           int from_cell, int dir,
                                           int exclude_cell) {
    int16_t dest = SLIDE_FLANKS[from_cell][dir][0];
    int16_t cw   = SLIDE_FLANKS[from_cell][dir][1];
    int16_t ccw  = SLIDE_FLANKS[from_cell][dir][2];

    if (dest < 0) return false;  // off grid

    // Destination must be empty (treat exclude_cell as empty)
    if (s.occupied.get(dest) && dest != exclude_cell) return false;

    // Check flanking positions
    bool cw_occ  = (cw  >= 0) && s.occupied.get(cw)  && (cw  != exclude_cell);
    bool ccw_occ = (ccw >= 0) && s.occupied.get(ccw) && (ccw != exclude_cell);

    // Gate: both occupied → blocked
    if (cw_occ && ccw_occ) return false;
    // No contact: neither occupied → can't slide (would detach from hive)
    if (!cw_occ && !ccw_occ) return false;

    return true;
}

__device__ __forceinline__ bool can_slide_occ(const Bitboard& occ,
                                               int from_cell, int dir) {
    int16_t dest = SLIDE_FLANKS[from_cell][dir][0];
    int16_t cw   = SLIDE_FLANKS[from_cell][dir][1];
    int16_t ccw  = SLIDE_FLANKS[from_cell][dir][2];

    if (dest < 0) return false;
    if (occ.get(dest)) return false;

    bool cw_occ  = (cw  >= 0) && occ.get(cw);
    bool ccw_occ = (ccw >= 0) && occ.get(ccw);
    return (cw_occ || ccw_occ) && !(cw_occ && ccw_occ);
}

// ── Queen moves ─────────────────────────────────────────────────────

/**
 * Generate queen moves (1-step slide in any of 6 directions).
 * @param out  Output array for destination cells (max 6)
 * @return     Number of valid destinations
 */
__device__ inline int gen_queen_moves(const HiveState& s, int cell,
                                       uint16_t* out) {
    int count = 0;
    for (int d = 0; d < NUM_DIRS; d++) {
        if (can_slide(s, cell, d, cell)) {
            int16_t dest = SLIDE_FLANKS[cell][d][0];
            out[count++] = (uint16_t)dest;
        }
    }
    return count;
}

// ── Grasshopper moves ───────────────────────────────────────────────

/**
 * Generate grasshopper moves (jump in straight line over 1+ pieces).
 * @param out  Output array for destination cells (max 6)
 * @return     Number of valid destinations
 */
__device__ inline int gen_grasshopper_moves(const HiveState& s, int cell,
                                              uint16_t* out) {
    int count = 0;
    for (int d = 0; d < NUM_DIRS; d++) {
        // Must start by jumping over at least one piece
        int16_t pos = NEIGHBORS[cell][d];
        if (pos < 0 || !s.occupied.get(pos)) continue;

        // Follow direction until first empty cell
        while (pos >= 0 && s.occupied.get(pos)) {
            pos = NEIGHBORS[pos][d];
        }
        if (pos >= 0) {
            out[count++] = (uint16_t)pos;
        }
    }
    return count;
}

// ── Ant moves (BFS) ─────────────────────────────────────────────────

/**
 * Generate ant moves via BFS on slide-adjacent empty cells.
 * Uses a 529-bit visited set in local memory.
 * @param out  Output array for destination cells (max MAX_ANT_DESTS)
 * @return     Number of valid destinations
 */
__device__ inline int gen_ant_moves(const HiveState& s, int cell,
                                     uint16_t* out) {
    Bitboard occ = s.occupied;
    occ.clr(cell);  // the moving ant's source is empty during the search

    // BFS visited set (289 bits)
    Bitboard visited;
    visited.clear();
    visited.set(cell);

    // BFS frontier in local array
    uint16_t frontier[MAX_ANT_DESTS];
    int front_read = 0, front_write = 0;

    // Seed frontier with initial slide destinations
    for (int d = 0; d < NUM_DIRS; d++) {
        if (can_slide_occ(occ, cell, d)) {
            int16_t dest = SLIDE_FLANKS[cell][d][0];
            if (!visited.get(dest)) {
                visited.set(dest);
                if (front_write < MAX_ANT_DESTS)
                    frontier[front_write++] = (uint16_t)dest;
            }
        }
    }

    int count = front_write;  // initial destinations are already results
    // Copy initial frontier to output
    for (int i = 0; i < front_write; i++) {
        out[i] = frontier[i];
    }

    // BFS expansion
    while (front_read < front_write && count < MAX_ANT_DESTS) {
        int cur = frontier[front_read++];
        for (int d = 0; d < NUM_DIRS; d++) {
            if (can_slide_occ(occ, cur, d)) {
                int16_t dest = SLIDE_FLANKS[cur][d][0];
                if (dest >= 0 && !visited.get(dest)) {
                    visited.set(dest);
                    out[count++] = (uint16_t)dest;
                    if (front_write < MAX_ANT_DESTS) {
                        frontier[front_write++] = (uint16_t)dest;
                    }
                    if (count >= MAX_ANT_DESTS) break;
                }
            }
        }
    }

    return count;
}

// ── Spider moves (3-step DFS) ───────────────────────────────────────

/**
 * Generate spider moves: exactly 3 slides, no revisiting.
 * @param out  Output array for destination cells (max 32)
 * @return     Number of valid destinations
 */
__device__ inline int gen_spider_moves(const HiveState& s, int cell,
                                        uint16_t* out) {
    constexpr int MAX_SPIDER_DESTS = 32;
    int count = 0;

    // Result dedup set
    Bitboard result_set;
    result_set.clear();

    // 3-nested loops (unrolled DFS, matching CPU spider implementation)
    for (int d1 = 0; d1 < NUM_DIRS; d1++) {
        if (!can_slide(s, cell, d1, cell)) continue;
        int16_t p1 = SLIDE_FLANKS[cell][d1][0];
        if (p1 < 0) continue;

        for (int d2 = 0; d2 < NUM_DIRS; d2++) {
            if (!can_slide(s, p1, d2, cell)) continue;
            int16_t p2 = SLIDE_FLANKS[p1][d2][0];
            if (p2 < 0 || p2 == cell || p2 == p1) continue;  // no revisit

            for (int d3 = 0; d3 < NUM_DIRS; d3++) {
                if (!can_slide(s, p2, d3, cell)) continue;
                int16_t p3 = SLIDE_FLANKS[p2][d3][0];
                if (p3 < 0 || p3 == cell || p3 == p1 || p3 == p2) continue;

                if (!result_set.get(p3) && count < MAX_SPIDER_DESTS) {
                    result_set.set(p3);
                    out[count++] = (uint16_t)p3;
                }
            }
        }
    }

    return count;
}

// ── Beetle moves ────────────────────────────────────────────────────

/**
 * Generate beetle moves: ground-level slide OR elevated movement.
 *
 * Elevated gate rule: A gate blocks a beetle only when BOTH flanking
 * neighbors have stack height STRICTLY GREATER than the movement height.
 * movement_height = max(dest_height, src_height_after_removal)
 *
 * @param out  Output array for destination cells (max 6)
 * @return     Number of valid destinations
 */
__device__ inline int gen_beetle_moves(const HiveState& s, int cell,
                                        uint16_t* out) {
    int count = 0;
    int src_height_after = s.height[cell] - 1;  // height after removing beetle

    for (int d = 0; d < NUM_DIRS; d++) {
        int16_t dest = NEIGHBORS[cell][d];
        if (dest < 0) continue;

        int dest_height = s.height[dest];
        int move_height = max(dest_height, src_height_after);

        if (move_height > 0) {
            // Elevated move (climbing up, across top, or down)
            // Gate check: both flanking heights must be STRICTLY GREATER
            int16_t cw  = SLIDE_FLANKS[cell][d][1];
            int16_t ccw = SLIDE_FLANKS[cell][d][2];
            int cw_h  = (cw  >= 0) ? s.height[cw]  : 0;
            int ccw_h = (ccw >= 0) ? s.height[ccw] : 0;

            if (cw_h > move_height && ccw_h > move_height) {
                continue;  // Gate blocked at this elevation
            }

            out[count++] = (uint16_t)dest;
        } else {
            // Ground-level slide (both src and dest are at ground level)
            if (can_slide(s, cell, d, cell)) {
                out[count++] = (uint16_t)dest;
            }
        }
    }

    return count;
}

// ── Ladybug moves ───────────────────────────────────────────────────

/**
 * Generate ladybug moves: 3-step elevated traversal.
 * Step 1: ascend onto an occupied neighbor
 * Step 2: traverse across another occupied neighbor (at elevation)
 * Step 3: descend to an empty neighbor of step 2's position
 *
 * Gate checks use elevated beetle-style logic at each step.
 * @param out  Output array for destination cells (max 32)
 * @return     Number of valid destinations
 */
__device__ inline int gen_ladybug_moves(const HiveState& s, int cell,
                                          uint16_t* out) {
    constexpr int MAX_LADYBUG_DESTS = 32;
    int count = 0;

    Bitboard result_set;
    result_set.clear();

    // Step 1: ascend — move onto an occupied neighbor
    for (int d1 = 0; d1 < NUM_DIRS; d1++) {
        int16_t p1 = NEIGHBORS[cell][d1];
        if (p1 < 0 || !s.occupied.get(p1)) continue;

        // Elevated gate check for ascent
        int src_h_after = s.height[cell] - 1;  // height after removing ladybug
        int dst_h = s.height[p1];
        int move_h = max(src_h_after, dst_h);
        int16_t cw1  = SLIDE_FLANKS[cell][d1][1];
        int16_t ccw1 = SLIDE_FLANKS[cell][d1][2];
        int cw1_h  = (cw1  >= 0) ? s.height[cw1]  : 0;
        int ccw1_h = (ccw1 >= 0) ? s.height[ccw1] : 0;
        if (cw1_h > move_h && ccw1_h > move_h) continue;  // gate blocked

        // Step 2: traverse — move across another occupied neighbor (elevated)
        for (int d2 = 0; d2 < NUM_DIRS; d2++) {
            int16_t p2 = NEIGHBORS[p1][d2];
            if (p2 < 0 || p2 == cell || !s.occupied.get(p2)) continue;

            // Elevated gate check for traverse (on top of p1 → on top of p2)
            int trav_h = max(s.height[p1], s.height[p2]);
            int16_t cw2  = SLIDE_FLANKS[p1][d2][1];
            int16_t ccw2 = SLIDE_FLANKS[p1][d2][2];
            int cw2_h  = (cw2  >= 0) ? s.height[cw2]  : 0;
            int ccw2_h = (ccw2 >= 0) ? s.height[ccw2] : 0;
            if (cw2_h > trav_h && ccw2_h > trav_h) continue;

            // Step 3: descend — move to an empty neighbor of p2
            for (int d3 = 0; d3 < NUM_DIRS; d3++) {
                int16_t p3 = NEIGHBORS[p2][d3];
                if (p3 < 0 || p3 == cell || p3 == p1 || s.occupied.get(p3)) continue;

                // Elevated gate check for descent (on top of p2 → ground at p3)
                int desc_h = max(s.height[p2], 0);  // dest is empty, height 0
                int16_t cw3  = SLIDE_FLANKS[p2][d3][1];
                int16_t ccw3 = SLIDE_FLANKS[p2][d3][2];
                int cw3_h  = (cw3  >= 0) ? s.height[cw3]  : 0;
                int ccw3_h = (ccw3 >= 0) ? s.height[ccw3] : 0;
                if (cw3_h > desc_h && ccw3_h > desc_h) continue;

                if (!result_set.get(p3) && count < MAX_LADYBUG_DESTS) {
                    result_set.set(p3);
                    out[count++] = (uint16_t)p3;
                }
            }
        }
    }

    return count;
}

// ── Pillbug moves (standard) ────────────────────────────────────────

/**
 * Generate standard pillbug moves: 1-step slide (same as queen).
 */
__device__ inline int gen_pillbug_moves(const HiveState& s, int cell,
                                          uint16_t* out) {
    return gen_queen_moves(s, cell, out);
}

// ── Pillbug special moves (throw) ───────────────────────────────────

/**
 * Check elevated gate between two adjacent cells at a given height.
 * Returns true if gate is BLOCKED.
 */
__device__ __forceinline__ bool elevated_gate_blocked(const HiveState& s,
                                                       int from_cell, int dir,
                                                       int move_height) {
    int16_t cw  = SLIDE_FLANKS[from_cell][dir][1];
    int16_t ccw = SLIDE_FLANKS[from_cell][dir][2];
    int cw_h  = (cw  >= 0) ? s.height[cw]  : 0;
    int ccw_h = (ccw >= 0) ? s.height[ccw] : 0;
    return (cw_h > move_height && ccw_h > move_height);
}

/**
 * Find the direction index from cell A to adjacent cell B.
 * Returns -1 if not adjacent.
 */
__device__ __forceinline__ int find_direction(int from_cell, int to_cell) {
    for (int d = 0; d < NUM_DIRS; d++) {
        if (NEIGHBORS[from_cell][d] == to_cell) return d;
    }
    return -1;
}

/**
 * Generate pillbug special throw moves from a given position.
 * The piece at pb_cell throws adjacent top pieces to empty neighbors.
 *
 * @param s         Game state
 * @param pb_cell   Position of the pillbug (or mosquito acting as pillbug)
 * @param ap_mask   Articulation point mask
 * @param out       Output moves array
 * @param count     Current count (appends to existing moves)
 * @return          Updated count
 */
__device__ inline int gen_pillbug_throws(const HiveState& s, int pb_cell,
                                           const Bitboard& ap_mask,
                                           GPUMove* out, int count) {
    int pb_height = s.height[pb_cell];

    // For each adjacent occupied cell (target to throw)
    for (int dt = 0; dt < NUM_DIRS; dt++) {
        int16_t target_cell = NEIGHBORS[pb_cell][dt];
        if (target_cell < 0 || !s.occupied.get(target_cell)) continue;

        // Target must be on top and not pinned
        if (is_pinned(s, ap_mask, target_cell)) continue;

        // Target must have stack height 1 (only ground piece can be thrown)
        // Actually: target top piece is thrown, but only if it's the only piece
        // (if stack > 1, the top piece isn't pinned by AP check, so it can be thrown)
        // The AP check already handles whether removing the piece breaks the hive.

        // Gate check: lift from target over pillbug
        // The piece lifts to pillbug height
        int lift_h = max(s.height[target_cell] - 1, pb_height);
        // Find opposite direction (target→pillbug)
        int opp_dt = find_direction(target_cell, pb_cell);
        if (opp_dt < 0) continue;
        if (elevated_gate_blocked(s, target_cell, opp_dt, lift_h)) continue;

        // For each empty neighbor of pillbug (not target position)
        for (int dd = 0; dd < NUM_DIRS; dd++) {
            int16_t dest_cell = NEIGHBORS[pb_cell][dd];
            if (dest_cell < 0 || dest_cell == target_cell) continue;
            if (s.occupied.get(dest_cell)) continue;

            // Gate check: drop from pillbug to dest
            int drop_h = max(pb_height, 0);  // dest is empty
            if (elevated_gate_blocked(s, pb_cell, dd, drop_h)) continue;

            if (count < MAX_LEGAL_MOVES) {
                out[count].type = MOVE_MOVE;
                out[count].piece_type = top_piece_type_at(s, target_cell);
                out[count].from_cell = (uint16_t)target_cell;
                out[count].to_cell = (uint16_t)dest_cell;
                count++;
            }
        }
    }

    return count;
}

// ── Mosquito moves ──────────────────────────────────────────────────

/**
 * Generate mosquito moves by copying abilities of adjacent top pieces.
 * If elevated (stack height > 1), acts as beetle.
 * If adjacent only to mosquitos, cannot move.
 */
__device__ inline int gen_mosquito_moves(const HiveState& s, int cell,
                                           uint16_t* out) {
    // If elevated, act as beetle
    if (s.height[cell] > 1) {
        return gen_beetle_moves(s, cell, out);
    }

    // Collect unique piece types of adjacent top pieces (skip MOSQUITO)
    bool has_type[NUM_PIECE_TYPES + 1] = {};  // indexed by PieceType enum value
    for (int d = 0; d < NUM_DIRS; d++) {
        int16_t nb = NEIGHBORS[cell][d];
        if (nb < 0 || !s.occupied.get(nb)) continue;
        PieceType npt = top_piece_type_at(s, nb);
        if (npt != PT_MOSQUITO) {
            has_type[npt] = true;
        }
    }

    // Union all results via bitboard dedup
    Bitboard result_set;
    result_set.clear();
    int count = 0;

    // Temporary buffer for sub-generators
    uint16_t tmp[MAX_ANT_DESTS];

    if (has_type[PT_QUEEN] || has_type[PT_PILLBUG]) {
        // Both queen and pillbug have same standard movement (1-step slide)
        int n = gen_queen_moves(s, cell, tmp);
        for (int i = 0; i < n; i++) {
            if (!result_set.get(tmp[i])) {
                result_set.set(tmp[i]);
                if (count < MAX_ANT_DESTS) out[count++] = tmp[i];
            }
        }
    }

    if (has_type[PT_ANT]) {
        int n = gen_ant_moves(s, cell, tmp);
        for (int i = 0; i < n; i++) {
            if (!result_set.get(tmp[i])) {
                result_set.set(tmp[i]);
                if (count < MAX_ANT_DESTS) out[count++] = tmp[i];
            }
        }
    }

    if (has_type[PT_GRASSHOPPER]) {
        int n = gen_grasshopper_moves(s, cell, tmp);
        for (int i = 0; i < n; i++) {
            if (!result_set.get(tmp[i])) {
                result_set.set(tmp[i]);
                if (count < MAX_ANT_DESTS) out[count++] = tmp[i];
            }
        }
    }

    if (has_type[PT_SPIDER]) {
        int n = gen_spider_moves(s, cell, tmp);
        for (int i = 0; i < n; i++) {
            if (!result_set.get(tmp[i])) {
                result_set.set(tmp[i]);
                if (count < MAX_ANT_DESTS) out[count++] = tmp[i];
            }
        }
    }

    if (has_type[PT_BEETLE]) {
        int n = gen_beetle_moves(s, cell, tmp);
        for (int i = 0; i < n; i++) {
            if (!result_set.get(tmp[i])) {
                result_set.set(tmp[i]);
                if (count < MAX_ANT_DESTS) out[count++] = tmp[i];
            }
        }
    }

    if (has_type[PT_LADYBUG]) {
        int n = gen_ladybug_moves(s, cell, tmp);
        for (int i = 0; i < n; i++) {
            if (!result_set.get(tmp[i])) {
                result_set.set(tmp[i]);
                if (count < MAX_ANT_DESTS) out[count++] = tmp[i];
            }
        }
    }

    // Note: pillbug special throw is handled separately in generate_legal_moves

    return count;
}

// ── Placement position generation ───────────────────────────────────

/**
 * Find valid placement positions for a color.
 *
 * Rules (after first 2 turns):
 * - Must be adjacent to at least one friendly piece
 * - Must NOT be adjacent to any enemy piece
 * - Top-piece color determines friendliness
 *
 * @param result  Output Bitboard with valid positions marked
 * @return        Number of valid positions
 */
__device__ inline int find_placement_positions(const HiveState& s, Color color,
                                                Bitboard& result) {
    result.clear();
    int count = 0;

    // Special case: turn 0, place at center
    if (s.turn == 0) {
        int center = cell_from_grid(HALF_BOARD, HALF_BOARD);
        result.set(center);
        return 1;
    }

    // Special case: turn 1, place adjacent to first piece (enemy adjacency OK)
    if (s.turn == 1) {
        // Find the one occupied cell
        for (int wi = 0; wi < BB_WORDS; wi++) {
            uint64_t bits = s.occupied.w[wi];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int cell = wi * 64 + bit;
                if (cell < NUM_CELLS) {
                    // Add all empty neighbors
                    for (int d = 0; d < NUM_DIRS; d++) {
                        int16_t nb = NEIGHBORS[cell][d];
                        if (nb >= 0 && !s.occupied.get(nb)) {
                            result.set(nb);
                            count++;
                        }
                    }
                }
                bits &= bits - 1;
            }
        }
        return count;
    }

    // Normal placement: adjacent to friendly only
    const Bitboard& friendly = (color == WHITE) ? s.white_top : s.black_top;
    const Bitboard& enemy    = (color == WHITE) ? s.black_top : s.white_top;

    // Bitboard of checked positions (to avoid re-checking)
    Bitboard checked;
    checked.clear();

    // For each friendly cell, check its empty neighbors
    for (int wi = 0; wi < BB_WORDS; wi++) {
        uint64_t bits = friendly.w[wi];
        while (bits) {
            int bit = __ffsll(bits) - 1;
            int cell = wi * 64 + bit;
            bits &= bits - 1;
            if (cell >= NUM_CELLS) continue;

            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[cell][d];
                if (nb < 0 || s.occupied.get(nb) || checked.get(nb)) continue;
                checked.set(nb);

                // Check: no enemy neighbor
                bool has_enemy = false;
                for (int d2 = 0; d2 < NUM_DIRS; d2++) {
                    int16_t nb2 = NEIGHBORS[nb][d2];
                    if (nb2 >= 0 && enemy.get(nb2)) {
                        has_enemy = true;
                        break;
                    }
                }
                if (!has_enemy) {
                    result.set(nb);
                    count++;
                }
            }
        }
    }

    return count;
}

// ── Full legal move generation ──────────────────────────────────────

/**
 * Generate all legal moves for the current player.
 *
 * @param s    The current game state
 * @param out  Output array for moves (max MAX_LEGAL_MOVES)
 * @return     Number of legal moves generated
 */
__device__ inline int generate_legal_moves(const HiveState& s, GPUMove* out) {
    if (s.result != IN_PROGRESS) return 0;
    mgp_add(MGP_CALLS, 1);

    Color color = current_player(s);
    int ptn = player_turn_number(s);
    int count = 0;

    // ── Placement moves ─────────────────────────────────────────

    // Check if queen must be placed (turn 3, queen not yet placed)
    bool must_place_queen = (ptn == 3 && !is_queen_placed(s, color));

    // Determine placeable piece types
    bool can_place_type[NUM_PIECE_TYPES];
    bool has_any_hand = false;
    for (int p = 0; p < NUM_PIECE_TYPES; p++) {
        if (must_place_queen) {
            can_place_type[p] = (p == 0) && (s.hands[color][0] > 0);  // Queen only
        } else {
            can_place_type[p] = (s.hands[color][p] > 0);
        }
        if (can_place_type[p]) has_any_hand = true;
    }

    if (has_any_hand) {
        Bitboard placements;
        int npos = find_placement_positions(s, color, placements);
        mgp_add(MGP_PLACEMENT_CALLS, 1);

        if (npos > 0) {
            // Queen can't be placed on player's first turn
            bool first_turn = (ptn == 0);

            // Scan placement cells once, then emit every placeable type for
            // that destination. Previously this rescanned the same bitboard
            // once per piece type.
            for (int wi = 0; wi < BB_WORDS; wi++) {
                uint64_t bits = placements.w[wi];
                while (bits) {
                    int bit = __ffsll(bits) - 1;
                    int cell = wi * 64 + bit;
                    bits &= bits - 1;
                    if (cell >= NUM_CELLS) continue;

                    for (int p = 0; p < NUM_PIECE_TYPES; p++) {
                        if (!can_place_type[p]) continue;
                        if (first_turn && p == 0) continue;  // No queen on first turn
                        if (count < MAX_LEGAL_MOVES) {
                            out[count].type = MOVE_PLACE;
                            out[count].piece_type = (PieceType)(p + 1);
                            out[count].from_cell = 0;  // unused
                            out[count].to_cell = (uint16_t)cell;
                            count++;
                            mgp_add(MGP_PLACEMENT_MOVES, 1);
                        }
                    }
                }
            }
        }
    }

    // ── Movement moves (only if queen is placed) ────────────────

    if (is_queen_placed(s, color)) {
        Bitboard ap_mask = find_articulation_points(s);

        const Bitboard& my_pieces = (color == WHITE) ? s.white_top : s.black_top;

        for (int wi = 0; wi < BB_WORDS; wi++) {
            uint64_t bits = my_pieces.w[wi];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int cell = wi * 64 + bit;
                bits &= bits - 1;
                if (cell >= NUM_CELLS) continue;

                // Only top pieces can move
                // (white_top/black_top already tracks top-piece ownership)

                // Check if pinned
                if (is_pinned(s, ap_mask, cell)) continue;

                // Generate type-specific destinations
                uint16_t dests[MAX_ANT_DESTS];
                int ndests = 0;
                PieceType pt = top_piece_type_at(s, cell);

                switch (pt) {
                    case PT_QUEEN:
                        mgp_add(MGP_QUEEN_CALLS, 1);
                        ndests = gen_queen_moves(s, cell, dests);
                        mgp_add(MGP_QUEEN_MOVES, ndests);
                        break;
                    case PT_ANT:
                        mgp_add(MGP_ANT_CALLS, 1);
                        ndests = gen_ant_moves(s, cell, dests);
                        mgp_add(MGP_ANT_MOVES, ndests);
                        break;
                    case PT_GRASSHOPPER:
                        mgp_add(MGP_GRASSHOPPER_CALLS, 1);
                        ndests = gen_grasshopper_moves(s, cell, dests);
                        mgp_add(MGP_GRASSHOPPER_MOVES, ndests);
                        break;
                    case PT_SPIDER:
                        mgp_add(MGP_SPIDER_CALLS, 1);
                        ndests = gen_spider_moves(s, cell, dests);
                        mgp_add(MGP_SPIDER_MOVES, ndests);
                        break;
                    case PT_BEETLE:
                        mgp_add(MGP_BEETLE_CALLS, 1);
                        ndests = gen_beetle_moves(s, cell, dests);
                        mgp_add(MGP_BEETLE_MOVES, ndests);
                        break;
                    case PT_MOSQUITO:
                        mgp_add(MGP_MOSQUITO_CALLS, 1);
                        ndests = gen_mosquito_moves(s, cell, dests);
                        mgp_add(MGP_MOSQUITO_MOVES, ndests);
                        break;
                    case PT_LADYBUG:
                        mgp_add(MGP_LADYBUG_CALLS, 1);
                        ndests = gen_ladybug_moves(s, cell, dests);
                        mgp_add(MGP_LADYBUG_MOVES, ndests);
                        break;
                    case PT_PILLBUG:
                        mgp_add(MGP_PILLBUG_CALLS, 1);
                        ndests = gen_pillbug_moves(s, cell, dests);
                        mgp_add(MGP_PILLBUG_MOVES, ndests);
                        break;
                    default:
                        break;
                }

                for (int i = 0; i < ndests && count < MAX_LEGAL_MOVES; i++) {
                    out[count].type = MOVE_MOVE;
                    out[count].piece_type = pt;
                    out[count].from_cell = (uint16_t)cell;
                    out[count].to_cell = dests[i];
                    count++;
                }
            }
        }

        // ── Pillbug special ability: throw adjacent pieces ──────
        // Second pass: for each friendly pillbug on top, generate throws.
        // Pillbug can throw even if it's an AP (but not if under a beetle,
        // which is already handled since only top pieces are in my_pieces).
        for (int wi = 0; wi < BB_WORDS; wi++) {
            uint64_t bits = my_pieces.w[wi];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int cell = wi * 64 + bit;
                bits &= bits - 1;
                if (cell >= NUM_CELLS) continue;

                PieceType pt = top_piece_type_at(s, cell);

                // Pillbug on top → generate throws
                if (pt == PT_PILLBUG) {
                    int before = count;
                    count = gen_pillbug_throws(s, cell, ap_mask, out, count);
                    mgp_add(MGP_THROW_CALLS, 1);
                    mgp_add(MGP_THROW_MOVES, count - before);
                }

                // Mosquito on ground adjacent to any pillbug → generate throws
                if (pt == PT_MOSQUITO && s.height[cell] == 1) {
                    bool adj_pillbug = false;
                    for (int d = 0; d < NUM_DIRS; d++) {
                        int16_t nb = NEIGHBORS[cell][d];
                        if (nb >= 0 && s.occupied.get(nb) &&
                            top_piece_type_at(s, nb) == PT_PILLBUG) {
                            adj_pillbug = true;
                            break;
                        }
                    }
                    if (adj_pillbug) {
                        int before = count;
                        count = gen_pillbug_throws(s, cell, ap_mask, out, count);
                        mgp_add(MGP_THROW_CALLS, 1);
                        mgp_add(MGP_THROW_MOVES, count - before);
                    }
                }
            }
        }
    }

    // ── Pass if no moves available ──────────────────────────────

    if (count == 0) {
        out[0].type = MOVE_PASS;
        out[0].piece_type = PT_EMPTY;
        out[0].from_cell = 0;
        out[0].to_cell = 0;
        count = 1;
        mgp_add(MGP_PASS_MOVES, 1);
    }

    return count;
}

#endif  // __CUDACC__

}  // namespace hive_gpu
