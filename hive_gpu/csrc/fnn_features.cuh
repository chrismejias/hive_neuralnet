/**
 * fnn_features.cuh — CUDA kernel for HiveGo-style FNN feature extraction.
 *
 * Produces a fixed-size feature vector directly from HiveState,
 * bypassing the full encode_states_batch pipeline (no per-node features,
 * no edges, no graph construction).
 *
 * Feature layout (FNN_FEAT_DIM = 122):
 *   [0:16]   count_on_board    — visible top pieces per type(8) × color(2)
 *   [16:32]  count_in_hand     — hand piece counts per type(8) × color(2)
 *   [32:48]  queen_neighbors   — top pieces adjacent to opponent queen, per type(8) × color(2)
 *   [48:64]  avg_dist_to_opp_q — avg hex distance to opponent queen, per type(8) × color(2)
 *   [64:80]  can_move_count    — number of distinct pieces with ≥1 legal MOVE attributable to owner, per type(8) × color(2)
 *   [80:96]  articulation_cnt  — number of ground-level articulation-point top pieces, per type(8) × color(2)
 *   [96:98]  num_single        — board pieces with 0 occupied neighbors, per color(2)
 *   [98:100] queen_covered     — queen not on top, per color(2)
 *   [100:102] num_placement_pos — unique placement destinations from legal moves, per color(2)
 *   [102]    moves_to_draw     — normalized turn count
 *   [103]    move_number       — turn / 100
 *   [104:106] pillbug_capable  — owner has an uncovered pillbug OR ground mosquito
 *                                adjacent to a usable pillbug this turn, per color(2)
 *   [106:108] throwable_own    — own-color pieces adjacent to own pillbug-capable cell, per color(2)
 *   [108:110] throwable_opp    — own-color pieces adjacent to opposing pillbug-capable cell (threatened), per color(2)
 *   [110:116] white_q_surround — one-hot surround count buckets 1..6 for white queen
 *   [116:122] black_q_surround — one-hot surround count buckets 1..6 for black queen
 *
 * Must be included from game_logic.cu (needs NEIGHBORS constant memory).
 */

#pragma once

#include "hex_grid.cuh"
#include "hive_state.cuh"
#include "articulation.cuh"

namespace hive_gpu {

constexpr int FNN_FEAT_DIM = 122;
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
    //    avg_dist_to_opp_q [48:64] + articulation_cnt [80:96] +
    //    num_single [96:98] + queen_covered [98:100] ─────────

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
    int queen_surround_counts[2] = {0, 0};

    Bitboard ap_mask = find_articulation_points(s);

    constexpr int PB_CELLS_PER_COLOR = 8;   // generous; real games see <= 2
    uint16_t pb_cells[2][PB_CELLS_PER_COLOR];
    int      pb_count[2] = {0, 0};

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
            f[96 + (int)pc] += 1.0f;
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

        // Queen surround counts for both queens. This is fused into the main
        // occupied-top-piece pass so we do not need a second neighborhood scan.
        for (int qc = 0; qc < 2; qc++) {
            uint16_t qcell = s.queen_cell[qc];
            if (qcell == 0xFFFF) continue;
            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[(int)qcell][d];
                if (nb >= 0 && (uint16_t)nb == (uint16_t)cell) {
                    queen_surround_counts[qc] += 1;
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

        // articulation_cnt: only count top pieces on ground-level AP cells.
        // Stacked pieces (e.g. elevated beetles) are intentionally excluded.
        if (h == 1 && ap_mask.get(cell)) {
            f[80 + tc_idx] += 1.0f;
        }

        // queen_covered: queen exists but not on top
        if (pt != PT_QUEEN) {
            // Check all levels below top for a queen at this cell
            for (int lv = 0; lv < h - 1; lv++) {
                uint8_t below = s.pieces[lv][cell];
                if (cell_piece_type(below) == PT_QUEEN) {
                    Color qc = cell_color(below);
                    f[98 + (int)qc] = 1.0f;
                }
            }
        }

        // pillbug_capable [104:106]
        bool is_capable = (s.stunned_cell != (uint16_t)cell) && (pt == PT_PILLBUG);
        if (!is_capable && s.stunned_cell != (uint16_t)cell && pt == PT_MOSQUITO && h == 1) {
            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[cell][d];
                if (nb < 0 || s.height[nb] == 0) continue;
                if (s.stunned_cell == (uint16_t)nb) continue;
                if (top_piece_type_at(s, nb) == PT_PILLBUG) {
                    is_capable = true;
                    break;
                }
            }
        }
        if (is_capable) {
            f[104 + (int)pc] = 1.0f;
            if (pb_count[(int)pc] < PB_CELLS_PER_COLOR) {
                pb_cells[(int)pc][pb_count[(int)pc]++] = (uint16_t)cell;
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

    // ── Features from legal moves: can_move_count [64:80], num_placement_pos [100:102] ──

    // For can_move_count: track distinct source cells per type×color bucket.
    Bitboard can_move_seen[16];
    for (int i = 0; i < 16; i++) can_move_seen[i].clear();

    // Track seen placement destinations with a small linear scan
    uint16_t seen_place_dst[2][MAX_LEGAL_MOVES];
    int seen_place_count[2] = {0, 0};

    Color cur = current_player(s);

    for (int m = 0; m < n_legal; m++) {
        const GPUMove& mv = my_moves[m];

        if (mv.type == MOVE_MOVE) {
            // Count distinct pieces with at least one legal MOVE. Pillbug throws
            // (and mosquito-as-pillbug throws) move opponent pieces, so the
            // piece's actual owner must be read from the board — we cannot
            // assume the mover equals the current player.
            int pt_idx = (int)mv.piece_type - 1;
            int src_h = s.height[mv.from_cell];
            Color mover_color = cur;
            if (src_h > 0) {
                uint8_t packed = s.pieces[src_h - 1][mv.from_cell];
                mover_color = cell_color(packed);
            }
            int flag_bit = pt_idx * 2 + (int)mover_color;
            can_move_seen[flag_bit].set(mv.from_cell);
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

    // Write can_move counts
    for (int i = 0; i < 16; i++) {
        f[64 + i] = (float)can_move_seen[i].popcount();
    }

    // Write num_placement_pos (normalized by ~10 typical positions)
    for (int c = 0; c < 2; c++) {
        f[100 + c] = (float)seen_place_count[c] / 10.0f;
    }

    // ── moves_to_draw [102] ───────────────────────────────────
    int moves_left = DRAW_MOVE_LIMIT - (int)s.turn;
    if (moves_left < 0) moves_left = 0;
    f[102] = (float)moves_left / (float)DRAW_MOVE_LIMIT;

    // ── move_number [103] ────────────────────────────────────
    float t = (float)s.turn / 100.0f;
    f[103] = t < 1.0f ? t : 1.0f;

    // ── pillbug_capable [104:106], throwable_own [106:108], throwable_opp [108:110] ──
    //
    // A cell is "pillbug-capable" for color c if:
    //   (a) c owns a pillbug on top of its stack, OR
    //   (b) c owns a ground-level mosquito adjacent to ANY pillbug (friendly
    //       or enemy) — per Hive rules, mosquitoes copy any adjacent piece's
    //       ability, so such a mosquito inherits the pillbug's throw ability
    //       for its owner's turn.
    //
    // Gate/pin legality is intentionally NOT enforced here — this is a
    // structural board feature. The `can_move` block already captures the
    // turn-specific legality for the current player. Stunned pillbug cells
    // are excluded because they are not usable this turn.
    //
    // throwable_own[c] = count of c-color top pieces adjacent to c's own
    //                    pillbug-capable cells (repositioning material).
    // throwable_opp[c] = count of c-color top pieces adjacent to (1−c)'s
    //                    pillbug-capable cells (threatened pieces).
    //
    // Pieces adjacent to both sides' pillbugs get counted in both buckets,
    // which is the intended signal (double pressure).
    for (int c = 0; c < 2; c++) {
        for (int pi = 0; pi < pb_count[c]; pi++) {
            int pbc = (int)pb_cells[c][pi];
            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[pbc][d];
                if (nb < 0 || s.height[nb] == 0) continue;
                if (s.stunned_cell == (uint16_t)nb) continue;
                uint8_t ntop = s.pieces[s.height[nb] - 1][nb];
                Color nc = cell_color(ntop);
                if ((int)nc == c) {
                    // c's own piece adjacent to c's pillbug — c could reposition it
                    f[106 + c] += 1.0f;
                } else {
                    // Piece owned by nc adjacent to c's pillbug — nc's piece is
                    // under threat from c's pillbug.
                    f[108 + (int)nc] += 1.0f;
                }
            }
        }
    }

    // ── queen surround one-hot buckets [110:122] ────────────────
    for (int qc = 0; qc < 2; qc++) {
        int surround = queen_surround_counts[qc];
        if (surround >= 1 && surround <= 6) {
            f[110 + qc * 6 + (surround - 1)] = 1.0f;
        }
    }
}

#ifndef HIVE_CPU_NATIVE
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
#endif  // HIVE_CPU_NATIVE

#endif  // __CUDACC__

}  // namespace hive_gpu
