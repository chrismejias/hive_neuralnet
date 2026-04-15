/**
 * fnn_features.cuh — CUDA kernel for HiveGo-style FNN feature extraction.
 *
 * Produces a fixed-size feature vector directly from HiveState,
 * bypassing the full encode_states_batch pipeline (no per-node features,
 * no edges, no graph construction).
 *
 * Feature layout (FNN_FEAT_DIM = 88):
 *   [0:16]   count_on_board    — visible top pieces per type(8) × color(2)
 *   [16:32]  count_in_hand     — hand piece counts per type(8) × color(2)
 *   [32:48]  queen_neighbors   — top pieces adjacent to opponent queen, per type(8) × color(2)
 *   [48:64]  avg_dist_to_opp_q — avg hex distance to opponent queen, per type(8) × color(2)
 *   [64:80]  can_move          — piece type has ≥1 legal MOVE, per type(8) × color(2)
 *   [80:82]  num_single        — board pieces with 0 occupied neighbors, per color(2)
 *   [82:84]  queen_covered     — queen not on top (beetle covering), per color(2)
 *   [84:86]  num_placement_pos — unique placement destinations from legal moves, per color(2)
 *   [86]     moves_to_draw     — normalized turn count
 *   [87]     move_number       — turn / 100
 *
 * Must be included from game_logic.cu (needs NEIGHBORS constant memory).
 */

#pragma once

#include "hex_grid.cuh"
#include "hive_state.cuh"

namespace hive_gpu {

constexpr int FNN_FEAT_DIM = 88;
// Draw is at move 200 in standard Hive
constexpr int DRAW_MOVE_LIMIT = 200;

#ifdef __CUDACC__

__device__ __forceinline__ int hex_distance(int cell_a, int cell_b) {
    // Axial coordinates
    int q_a = cell_a % BOARD_SIZE - HALF_BOARD;
    int r_a = cell_a / BOARD_SIZE - HALF_BOARD;
    int q_b = cell_b % BOARD_SIZE - HALF_BOARD;
    int r_b = cell_b / BOARD_SIZE - HALF_BOARD;
    int dq = q_b - q_a;
    int dr = r_b - r_a;
    // Cube distance: max(|dq|, |dr|, |dq+dr|)
    int ds = -(dq + dr);
    int adq = dq < 0 ? -dq : dq;
    int adr = dr < 0 ? -dr : dr;
    int ads = ds < 0 ? -ds : ds;
    int mx = adq;
    if (adr > mx) mx = adr;
    if (ads > mx) mx = ads;
    return mx;
}

/**
 * Device function: extract FNN features for a single game state.
 *
 * Can be called from any kernel (selfplay, batch feature extraction, etc.).
 */
__device__ inline void extract_fnn_features_device(
    const HiveState& s,
    const GPUMove* my_moves,   // legal moves for this state
    int n_legal,               // number of legal moves
    float* f                   // [FNN_FEAT_DIM] output
) {
    // Zero output
    for (int i = 0; i < FNN_FEAT_DIM; i++) f[i] = 0.0f;

    // ── count_on_board [0:16] + queen_neighbors [32:48] +
    //    avg_dist_to_opp_q [48:64] + num_single [80:82] +
    //    queen_covered [82:84] ─────────────────────────────

    // Accumulators for avg distance
    float dist_sum[2][NUM_PIECE_TYPES];  // [color][type]
    int dist_count[2][NUM_PIECE_TYPES];
    for (int c = 0; c < 2; c++) {
        for (int t = 0; t < NUM_PIECE_TYPES; t++) {
            dist_sum[c][t] = 0.0f;
            dist_count[c][t] = 0;
        }
    }

    uint16_t opp_queen[2];  // opponent queen cell for each color
    opp_queen[0] = s.queen_cell[1];  // white's opponent is black's queen
    opp_queen[1] = s.queen_cell[0];  // black's opponent is white's queen

    for (int cell = 0; cell < NUM_CELLS; cell++) {
        int h = s.height[cell];
        if (h == 0) continue;

        // Top piece
        uint8_t top = s.pieces[h - 1][cell];
        PieceType pt = cell_piece_type(top);
        Color pc = cell_color(top);
        int type_idx = (int)pt - 1;  // 0-indexed
        int tc_idx = type_idx * 2 + (int)pc;  // type×color index

        // count_on_board
        f[tc_idx] += 1.0f;

        // Count occupied neighbors for this cell
        int occ_nb = 0;
        for (int d = 0; d < NUM_DIRS; d++) {
            int16_t nb = NEIGHBORS[cell][d];
            if (nb >= 0 && s.height[nb] > 0) occ_nb++;
        }

        // num_single: top pieces with 0 occupied neighbors
        if (occ_nb == 0) {
            f[80 + (int)pc] += 1.0f;
        }

        // queen_neighbors: is this cell adjacent to opponent's queen?
        uint16_t opp_q = opp_queen[(int)pc];
        if (opp_q != 0xFFFF) {
            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[cell][d];
                if (nb >= 0 && (uint16_t)nb == opp_q) {
                    f[32 + tc_idx] += 1.0f;
                    break;
                }
            }
        }

        // avg_dist_to_opp_q: distance to opponent's queen
        if (opp_q != 0xFFFF) {
            int d = hex_distance(cell, (int)opp_q);
            dist_sum[(int)pc][type_idx] += (float)d;
            dist_count[(int)pc][type_idx] += 1;
        }

        // queen_covered: queen exists but not on top
        if (pt != PT_QUEEN) {
            // Check all levels below top for a queen at this cell
            for (int lv = 0; lv < h - 1; lv++) {
                uint8_t below = s.pieces[lv][cell];
                if (cell_piece_type(below) == PT_QUEEN) {
                    Color qc = cell_color(below);
                    f[82 + (int)qc] = 1.0f;
                }
            }
        }
    }

    // Normalize avg_dist_to_opp_q [48:64]
    for (int c = 0; c < 2; c++) {
        for (int t = 0; t < NUM_PIECE_TYPES; t++) {
            int tc_idx = t * 2 + c;
            if (dist_count[c][t] > 0) {
                f[48 + tc_idx] = dist_sum[c][t] / (float)dist_count[c][t] / 10.0f;
            }
        }
    }

    // ── count_in_hand [16:32] ───────────────────────────────
    for (int c = 0; c < 2; c++) {
        for (int t = 0; t < NUM_PIECE_TYPES; t++) {
            int tc_idx = t * 2 + c;
            float max_count = (float)pieces_per_type(t);
            f[16 + tc_idx] = (float)s.hands[c][t] / (max_count > 0.0f ? max_count : 1.0f);
        }
    }

    // ── Features from legal moves: can_move [64:80], num_placement_pos [84:86] ──

    // For can_move: track which type×color combos have a MOVE
    uint32_t can_move_flags = 0;

    // Track seen placement destinations with a small linear scan
    uint16_t seen_place_dst[2][MAX_LEGAL_MOVES];
    int seen_place_count[2] = {0, 0};

    Color cur = current_player(s);

    for (int m = 0; m < n_legal; m++) {
        const GPUMove& mv = my_moves[m];

        if (mv.type == MOVE_MOVE) {
            // Mark this piece type × color as "can move"
            int pt_idx = (int)mv.piece_type - 1;
            int flag_bit = pt_idx * 2 + (int)cur;
            can_move_flags |= (1u << flag_bit);
        } else if (mv.type == MOVE_PLACE) {
            // Count unique placement destinations
            bool seen = false;
            for (int j = 0; j < seen_place_count[(int)cur]; j++) {
                if (seen_place_dst[(int)cur][j] == mv.to_cell) {
                    seen = true;
                    break;
                }
            }
            if (!seen && seen_place_count[(int)cur] < MAX_LEGAL_MOVES) {
                seen_place_dst[(int)cur][seen_place_count[(int)cur]++] = mv.to_cell;
            }
        }
    }

    // Write can_move flags
    for (int i = 0; i < 16; i++) {
        if (can_move_flags & (1u << i)) {
            f[64 + i] = 1.0f;
        }
    }

    // Write num_placement_pos (normalized by ~10 typical positions)
    for (int c = 0; c < 2; c++) {
        f[84 + c] = (float)seen_place_count[c] / 10.0f;
    }

    // ── moves_to_draw [86] ───────────────────────────────────
    int moves_left = DRAW_MOVE_LIMIT - (int)s.turn;
    if (moves_left < 0) moves_left = 0;
    f[86] = (float)moves_left / (float)DRAW_MOVE_LIMIT;

    // ── move_number [87] ────────────────────────────────────
    float t = (float)s.turn / 100.0f;
    f[87] = t < 1.0f ? t : 1.0f;
}

__global__ void extract_fnn_features_kernel(
    const HiveState* states,
    const GPUMove* legal_moves,   // [B, MAX_LEGAL_MOVES]
    const int* num_legal,         // [B]
    float* features_out,          // [B, FNN_FEAT_DIM]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    extract_fnn_features_device(
        states[idx],
        legal_moves + idx * MAX_LEGAL_MOVES,
        num_legal[idx],
        features_out + idx * FNN_FEAT_DIM
    );
}

#endif  // __CUDACC__

}  // namespace hive_gpu
