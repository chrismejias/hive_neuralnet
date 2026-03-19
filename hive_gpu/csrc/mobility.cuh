/**
 * mobility.cuh — Per-piece mobility computation on GPU.
 *
 * Computes a binary mobility target for each board piece node, iterating
 * cells in the same order as encode_states_kernel (cell 0..NUM_CELLS-1,
 * level 0..h-1) to ensure node index alignment.
 *
 * Mobility definition: a piece is mobile (1.0) if:
 *   1. It can make at least one legal move on its own, OR
 *   2. An enemy pillbug (or ground mosquito adjacent to a pillbug) could
 *      throw it (measuring opponent's ability to move this piece).
 *
 * When both_players=true, checks mobility for ALL pieces regardless of
 * color (used for final_mobility at game end).
 */

#pragma once

#include "hive_state.cuh"
#include "hex_grid.cuh"
#include "articulation.cuh"
#include "move_gen.cuh"
#include "state_encoder.cuh"  // for MAX_ENC_NODES

namespace hive_gpu {

#ifdef __CUDACC__

/**
 * Check if a piece at `cell` can be thrown by an enemy pillbug.
 *
 * The piece must be:
 *   - A top piece (only top pieces can be thrown)
 *   - Not pinned (AP with stack height 1)
 *   - Adjacent to an enemy pillbug (or ground mosquito adjacent to a pillbug)
 *   - Gate checks pass for at least one (lift, drop) pair
 *
 * @param s         Game state
 * @param cell      Cell of the piece to check
 * @param ap_mask   Articulation point mask
 * @param enemy     Color of the enemy (the thrower)
 * @return          true if an enemy pillbug can throw this piece
 */
__device__ inline bool can_be_thrown_by_enemy(
    const HiveState& s, int cell,
    const Bitboard& ap_mask, Color enemy
) {
    // Piece must not be pinned
    if (is_pinned(s, ap_mask, cell)) return false;

    // Check each neighbor for an enemy pillbug (or ground mosquito adj to pillbug)
    for (int d = 0; d < NUM_DIRS; d++) {
        int16_t pb_cell = NEIGHBORS[cell][d];
        if (pb_cell < 0 || !s.occupied.get(pb_cell)) continue;

        // Must be enemy's top piece
        Color top_c = top_piece_color_at(s, pb_cell);
        if (top_c != enemy) continue;

        PieceType pb_pt = top_piece_type_at(s, pb_cell);
        bool is_pillbug_actor = false;

        if (pb_pt == PT_PILLBUG) {
            is_pillbug_actor = true;
        } else if (pb_pt == PT_MOSQUITO && s.height[pb_cell] == 1) {
            // Ground mosquito: check if adjacent to any pillbug
            for (int d2 = 0; d2 < NUM_DIRS; d2++) {
                int16_t nb2 = NEIGHBORS[pb_cell][d2];
                if (nb2 >= 0 && s.occupied.get(nb2) &&
                    top_piece_type_at(s, nb2) == PT_PILLBUG) {
                    is_pillbug_actor = true;
                    break;
                }
            }
        }

        if (!is_pillbug_actor) continue;

        // This neighbor is an enemy pillbug actor. Check gate for lifting.
        int pb_height = s.height[pb_cell];

        // Gate check: lift piece from cell over pillbug
        int lift_h = max(s.height[cell] - 1, pb_height);
        int opp_d = find_direction(cell, pb_cell);
        if (opp_d < 0) continue;
        if (elevated_gate_blocked(s, cell, opp_d, lift_h)) continue;

        // Check if at least one empty neighbor of pillbug (not cell) passes drop gate
        for (int dd = 0; dd < NUM_DIRS; dd++) {
            int16_t dest = NEIGHBORS[pb_cell][dd];
            if (dest < 0 || dest == cell || s.occupied.get(dest)) continue;

            int drop_h = max(pb_height, 0);
            if (!elevated_gate_blocked(s, pb_cell, dd, drop_h)) {
                return true;  // Found a valid throw destination
            }
        }
    }

    return false;
}

/**
 * Compute per-piece mobility for a single HiveState.
 *
 * Iterates cells in the same order as encode_states_kernel to ensure
 * mobility[node_idx] corresponds to the same piece as node_features[node_idx].
 *
 * @param s             Game state
 * @param mobility_out  Output array [MAX_ENC_NODES], filled with 0.0/1.0
 * @param both_players  If true, check mobility for both colors' pieces
 * @return              Number of board nodes enumerated
 */
__device__ inline int compute_piece_mobility(
    const HiveState& s,
    float* mobility_out,
    bool both_players
) {
    // Zero output
    for (int i = 0; i < MAX_ENC_NODES; i++) mobility_out[i] = 0.0f;

    Color cur = current_player(s);
    Color opp = (Color)(1 - cur);

    // Compute articulation points once
    Bitboard ap_mask = find_articulation_points(s);

    // Check queen placement for each color
    bool queen_placed_cur = is_queen_placed(s, cur);
    bool queen_placed_opp = is_queen_placed(s, opp);

    int node_count = 0;

    // Iterate cells in same order as encode_states_kernel
    for (int cell = 0; cell < NUM_CELLS; cell++) {
        int h = s.height[cell];
        if (h == 0) continue;

        for (int level = 0; level < h; level++) {
            if (node_count >= MAX_ENC_NODES) break;

            uint8_t packed = s.pieces[level][cell];
            Color pc = cell_color(packed);
            PieceType pt = cell_piece_type(packed);
            bool is_top = (level == h - 1);

            float mobile = 0.0f;

            if (is_top) {
                // Determine if we should check this piece's own-move mobility
                bool check_own_moves = false;
                if (both_players) {
                    // Check both colors
                    bool q_placed = (pc == cur) ? queen_placed_cur : queen_placed_opp;
                    check_own_moves = q_placed;
                } else {
                    // Only check current player's pieces for own moves
                    check_own_moves = (pc == cur) && queen_placed_cur;
                }

                if (check_own_moves && !is_pinned(s, ap_mask, cell)) {
                    // Generate moves for this piece type
                    uint16_t dests[MAX_ANT_DESTS];
                    int ndests = 0;

                    switch (pt) {
                        case PT_QUEEN:
                            ndests = gen_queen_moves(s, cell, dests);
                            break;
                        case PT_ANT:
                            ndests = gen_ant_moves(s, cell, dests);
                            break;
                        case PT_GRASSHOPPER:
                            ndests = gen_grasshopper_moves(s, cell, dests);
                            break;
                        case PT_SPIDER:
                            ndests = gen_spider_moves(s, cell, dests);
                            break;
                        case PT_BEETLE:
                            ndests = gen_beetle_moves(s, cell, dests);
                            break;
                        case PT_MOSQUITO:
                            ndests = gen_mosquito_moves(s, cell, dests);
                            break;
                        case PT_LADYBUG:
                            ndests = gen_ladybug_moves(s, cell, dests);
                            break;
                        case PT_PILLBUG:
                            ndests = gen_pillbug_moves(s, cell, dests);
                            break;
                        default:
                            break;
                    }

                    if (ndests > 0) mobile = 1.0f;
                }

                // Check enemy pillbug throw (piece can be moved by opponent)
                if (mobile < 1.0f) {
                    // Determine which color is the "enemy" that could throw this piece
                    // For both_players mode, check both directions
                    if (both_players) {
                        Color enemy_of_piece = (Color)(1 - pc);
                        bool enemy_q_placed = (enemy_of_piece == cur) ? queen_placed_cur : queen_placed_opp;
                        if (enemy_q_placed) {
                            if (can_be_thrown_by_enemy(s, cell, ap_mask, enemy_of_piece)) {
                                mobile = 1.0f;
                            }
                        }
                    } else {
                        // Normal mode: check if enemy (opponent of current player) can throw
                        // current player's pieces via pillbug
                        if (pc == cur && queen_placed_opp) {
                            if (can_be_thrown_by_enemy(s, cell, ap_mask, opp)) {
                                mobile = 1.0f;
                            }
                        }
                    }
                }
            }
            // Non-top pieces (buried under beetles) are never mobile

            mobility_out[node_count] = mobile;
            node_count++;
        }
    }

    return node_count;
}

#endif  // __CUDACC__

}  // namespace hive_gpu
