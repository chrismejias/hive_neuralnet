/**
 * state_encoder.cuh — CUDA kernel for encoding HiveState into NN input features.
 *
 * Produces per-piece 25-dim feature vectors compatible with both GNN and Transformer
 * encoders, plus edge data for GNN graph construction.
 *
 * Must be included from a .cu file that has access to NEIGHBORS constant memory
 * (i.e., game_logic.cu).
 */

#pragma once

#include "hex_grid.cuh"
#include "hive_state.cuh"
#include "articulation.cuh"

namespace hive_gpu {

// ── Encoder constants ────────────────────────────────────────────────

constexpr int MAX_ENC_NODES = 48;    // 28 board pieces + 16 hand tokens + margin
constexpr int MAX_ENC_EDGES = 200;   // ~132 spatial + ~16 vertical max
constexpr int NODE_FEAT_DIM = 25;
constexpr int EDGE_FEAT_DIM = 9;
constexpr int GLOBAL_FEAT_DIM = 6;
constexpr int ENC_GRID = 17;         // 17×17 grid for NN spatial scatter
constexpr int ENC_HALF = 8;

// Hybrid GNN value-trunk encoder. Kept separate from the legacy encoder ABI:
// radius-2 spatial neighborhoods need a wider edge feature vector and more
// padded edge slots than the PRS/FNN graph encoder above.
constexpr int HYBRID_MAX_NODES = 48;
constexpr int HYBRID_MAX_EDGES = 640;
constexpr int HYBRID_MAX_PIECE_TOKENS = 28;
constexpr int HYBRID_NODE_FEAT_DIM = 26;
constexpr int HYBRID_GLOBAL_FEAT_DIM = 6;
constexpr int HYBRID_MOVE_FEAT_DIM = 25;
constexpr int HYBRID_MAX_RADIUS = 2;
constexpr int HYBRID_EDGE_FEAT_DIM = 3;

// ── Action space constants (matches hive_engine/encoder.py) ──────────
constexpr int NUM_ENC_GRID_CELLS = ENC_GRID * ENC_GRID;           // 169
constexpr int NUM_PLACEMENT_ACTIONS = NUM_PIECE_TYPES * NUM_ENC_GRID_CELLS;  // 1352
constexpr int MOVEMENT_OFFSET = NUM_PLACEMENT_ACTIONS;             // 1352
constexpr int ACTION_SPACE_SIZE = NUM_PLACEMENT_ACTIONS + NUM_ENC_GRID_CELLS * NUM_ENC_GRID_CELLS + 1; // 29914
constexpr int PASS_ACTION_INDEX = ACTION_SPACE_SIZE - 1;           // 29913

#ifdef __CUDACC__

// Pieces per player by type index (Q=1, A=3, G=3, S=2, B=2, M=1, L=1, P=1)
__device__ constexpr float COUNTS_PER_PLAYER[NUM_PIECE_TYPES] = {1.0f, 3.0f, 3.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f};

// Banker's rounding (round half to even) to match Python's round().
// Uses integer arithmetic: for x = sum/n, compute via 2*sum/n to detect .5 cases.
__device__ __forceinline__ int bankers_round_div(int sum, int n) {
    // Compute floor division and remainder
    // Using truncation toward zero (C++ behavior for ints)
    int q = sum / n;        // truncated quotient
    int rem = sum - q * n;  // remainder (same sign as sum)

    // Need to decide: round q or q+1 (or q-1 for negative)?
    // Compare |2*rem| against n to determine rounding
    int abs_2rem = (2 * rem < 0) ? -(2 * rem) : (2 * rem);

    if (abs_2rem > n) {
        // |frac| > 0.5, round away from zero
        return (sum > 0) ? q + 1 : q - 1;
    } else if (abs_2rem < n) {
        // |frac| < 0.5, truncate (round toward zero)
        return q;
    } else {
        // Exactly 0.5 — banker's rule: round to even
        // q is the truncated value; the "true" candidates are q and q+sign(sum)
        int candidate = (sum > 0) ? q + 1 : q - 1;
        // Pick whichever of q, candidate is even
        if ((q & 1) == 0) return q;  // q is even, keep it
        return candidate;  // candidate is the other, must be even
    }
}

__device__ __forceinline__ int hex_distance_delta(int dq, int dr) {
    int ds = dq + dr;
    int adq = dq < 0 ? -dq : dq;
    int adr = dr < 0 ? -dr : dr;
    int ads = ds < 0 ? -ds : ds;
    int m = adq > adr ? adq : adr;
    return m > ads ? m : ads;
}

__global__ void hybrid_gnn_encode_states_kernel(
    const HiveState* states,
    const GPUMove* legal_moves,     // [B, MAX_LEGAL_MOVES] or nullptr
    const int* num_legal,           // [B] or nullptr
    float* node_features,       // [B, HYBRID_MAX_NODES, HYBRID_NODE_FEAT_DIM]
    int64_t* edge_src,          // [B, HYBRID_MAX_EDGES]
    int64_t* edge_dst,          // [B, HYBRID_MAX_EDGES]
    float* edge_features,       // [B, HYBRID_MAX_EDGES, HYBRID_EDGE_FEAT_DIM]
    bool* node_mask,            // [B, HYBRID_MAX_NODES]
    bool* edge_mask,            // [B, HYBRID_MAX_EDGES]
    float* global_features,     // [B, HYBRID_GLOBAL_FEAT_DIM]
    int* num_nodes_out,         // [B]
    int* num_edges_out,         // [B]
    int batch_size,
    int radius,
    bool use_move_flags
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    if (radius < 1) radius = 1;
    if (radius > HYBRID_MAX_RADIUS) radius = HYBRID_MAX_RADIUS;

    const HiveState& s = states[idx];
    float* nf = node_features + idx * HYBRID_MAX_NODES * HYBRID_NODE_FEAT_DIM;
    int64_t* es = edge_src + idx * HYBRID_MAX_EDGES;
    int64_t* ed = edge_dst + idx * HYBRID_MAX_EDGES;
    float* ef = edge_features + idx * HYBRID_MAX_EDGES * HYBRID_EDGE_FEAT_DIM;
    bool* nm = node_mask + idx * HYBRID_MAX_NODES;
    bool* em = edge_mask + idx * HYBRID_MAX_EDGES;
    float* gf = global_features + idx * HYBRID_GLOBAL_FEAT_DIM;

    for (int i = 0; i < HYBRID_MAX_NODES * HYBRID_NODE_FEAT_DIM; ++i) nf[i] = 0.0f;
    for (int i = 0; i < HYBRID_MAX_EDGES * HYBRID_EDGE_FEAT_DIM; ++i) ef[i] = 0.0f;
    for (int i = 0; i < HYBRID_MAX_NODES; ++i) nm[i] = false;
    for (int i = 0; i < HYBRID_MAX_EDGES; ++i) {
        es[i] = 0;
        ed[i] = 0;
        em[i] = false;
    }
    for (int i = 0; i < HYBRID_GLOBAL_FEAT_DIM; ++i) gf[i] = 0.0f;

    float queen_surround[2] = {0.0f, 0.0f};
    for (int c = 0; c < 2; ++c) {
        uint16_t qc = s.queen_cell[c];
        if (qc != 0xFFFF) {
            int cnt = 0;
            for (int d = 0; d < NUM_DIRS; ++d) {
                int16_t nb = NEIGHBORS[qc][d];
                if (nb >= 0 && s.height[nb] > 0) cnt++;
            }
            queen_surround[c] = cnt / 6.0f;
        }
    }

    int node_count = 0;
    int16_t top_node_at[NUM_CELLS];
    int16_t first_node_at[NUM_CELLS];
    for (int cell = 0; cell < NUM_CELLS; ++cell) {
        top_node_at[cell] = -1;
        first_node_at[cell] = -1;
    }

    for (int cell = 0; cell < NUM_CELLS; ++cell) {
        int h = s.height[cell];
        if (h == 0) continue;

        int occ_flags[NUM_DIRS];
        int occ_count = 0;
        for (int d = 0; d < NUM_DIRS; ++d) {
            int16_t nb = NEIGHBORS[cell][d];
            occ_flags[d] = (nb >= 0 && s.height[nb] > 0) ? 1 : 0;
            occ_count += occ_flags[d];
        }

        first_node_at[cell] = (int16_t)node_count;
        for (int level = 0; level < h; ++level) {
            if (node_count >= HYBRID_MAX_NODES) break;
            uint8_t packed = s.pieces[level][cell];
            PieceType pt = cell_piece_type(packed);
            Color pc = cell_color(packed);
            bool is_top = (level == h - 1);

            float* f = nf + node_count * HYBRID_NODE_FEAT_DIM;
            f[pt - 1] = 1.0f;
            f[8 + pc] = 1.0f;
            f[10] = (level == 0) ? 1.0f : 0.0f;
            f[11] = is_top ? 1.0f : 0.0f;
            f[12] = h / 4.0f;
            f[13] = (pt == PT_QUEEN) ? 1.0f : 0.0f;
            f[14] = queen_surround[pc];
            f[15] = occ_count / 6.0f;
            if (is_top) {
                for (int d = 0; d < NUM_DIRS; ++d) {
                    if (!occ_flags[d]) f[16 + d] = 1.0f;
                }
                top_node_at[cell] = (int16_t)node_count;
            }
            f[24] = level * 0.25f;
            nm[node_count] = true;
            node_count++;
        }
    }

    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < NUM_PIECE_TYPES; ++p) {
            int count = s.hands[c][p];
            if (count == 0) continue;
            if (node_count >= HYBRID_MAX_NODES) break;
            float* f = nf + node_count * HYBRID_NODE_FEAT_DIM;
            f[p] = 1.0f;
            f[8 + c] = 1.0f;
            if (p == 0) f[13] = 1.0f;
            f[22] = 1.0f;
            f[23] = count / COUNTS_PER_PLAYER[p];
            nm[node_count] = true;
            node_count++;
        }
    }

    int edge_count = 0;
    for (int src_cell = 0; src_cell < NUM_CELLS; ++src_cell) {
        int16_t src_node = top_node_at[src_cell];
        if (src_node < 0) continue;
        int src_col = src_cell % BOARD_SIZE;
        int src_row = src_cell / BOARD_SIZE;
        for (int dist = 1; dist <= radius; ++dist) {
            for (int dq = -radius; dq <= radius; ++dq) {
                for (int dr = -radius; dr <= radius; ++dr) {
                    if (hex_distance_delta(dq, dr) != dist) continue;
                    int dst_col = src_col + dq;
                    int dst_row = src_row + dr;
                    if (dst_col < 0 || dst_col >= BOARD_SIZE ||
                        dst_row < 0 || dst_row >= BOARD_SIZE) {
                        continue;
                    }
                    int dst_cell = dst_row * BOARD_SIZE + dst_col;
                    int16_t dst_node = top_node_at[dst_cell];
                    if (dst_node < 0) continue;
                    if (edge_count >= HYBRID_MAX_EDGES) break;

                    es[edge_count] = src_node;
                    ed[edge_count] = dst_node;
                    float* e = ef + edge_count * HYBRID_EDGE_FEAT_DIM;
                    e[0] = (float)dq / (float)radius;
                    e[1] = (float)dr / (float)radius;
                    em[edge_count] = true;
                    edge_count++;
                }
                if (edge_count >= HYBRID_MAX_EDGES) break;
            }
            if (edge_count >= HYBRID_MAX_EDGES) break;
        }
        if (edge_count >= HYBRID_MAX_EDGES) break;
    }

    for (int cell = 0; cell < NUM_CELLS; ++cell) {
        int h = s.height[cell];
        if (h < 2) continue;
        int16_t base = first_node_at[cell];
        if (base < 0) continue;
        for (int level = 0; level < h - 1; ++level) {
            int lower = base + level;
            int upper = base + level + 1;
            if (lower >= HYBRID_MAX_NODES || upper >= HYBRID_MAX_NODES) break;
            if (edge_count < HYBRID_MAX_EDGES) {
                es[edge_count] = lower;
                ed[edge_count] = upper;
                ef[edge_count * HYBRID_EDGE_FEAT_DIM + 2] = 1.0f;
                em[edge_count] = true;
                edge_count++;
            }
            if (edge_count < HYBRID_MAX_EDGES) {
                es[edge_count] = upper;
                ed[edge_count] = lower;
                ef[edge_count * HYBRID_EDGE_FEAT_DIM + 2] = 1.0f;
                em[edge_count] = true;
                edge_count++;
            }
        }
    }

    Color cur = current_player(s);
    gf[0] = (cur == WHITE) ? 1.0f : 0.0f;
    gf[1] = (s.turn / 100.0f < 1.0f) ? s.turn / 100.0f : 1.0f;
    gf[2] = is_queen_placed(s, WHITE) ? 1.0f : 0.0f;
    gf[3] = is_queen_placed(s, BLACK) ? 1.0f : 0.0f;
    int white_hand = 0;
    int black_hand = 0;
    for (int p = 0; p < NUM_PIECE_TYPES; ++p) {
        white_hand += s.hands[0][p];
        black_hand += s.hands[1][p];
    }
    gf[4] = white_hand / 14.0f;
    gf[5] = black_hand / 14.0f;

    num_nodes_out[idx] = node_count;
    num_edges_out[idx] = edge_count;
}

__global__ void hybrid_transformer_encode_states_kernel(
    const HiveState* states,
    const GPUMove* legal_moves,
    const int* num_legal,
    float* token_features,     // [B, HYBRID_MAX_PIECE_TOKENS, HYBRID_NODE_FEAT_DIM]
    int* token_q,              // [B, HYBRID_MAX_PIECE_TOKENS]
    int* token_r,              // [B, HYBRID_MAX_PIECE_TOKENS]
    int* token_z,              // [B, HYBRID_MAX_PIECE_TOKENS]
    bool* token_mask,          // [B, HYBRID_MAX_PIECE_TOKENS]
    float* global_features,    // [B, HYBRID_GLOBAL_FEAT_DIM]
    int* num_tokens_out,       // [B]
    int batch_size,
    bool use_move_flags
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const HiveState& s = states[idx];
    float* tf = token_features + idx * HYBRID_MAX_PIECE_TOKENS * HYBRID_NODE_FEAT_DIM;
    int* tq = token_q + idx * HYBRID_MAX_PIECE_TOKENS;
    int* tr = token_r + idx * HYBRID_MAX_PIECE_TOKENS;
    int* tz = token_z + idx * HYBRID_MAX_PIECE_TOKENS;
    bool* tm = token_mask + idx * HYBRID_MAX_PIECE_TOKENS;
    float* gf = global_features + idx * HYBRID_GLOBAL_FEAT_DIM;

    for (int i = 0; i < HYBRID_MAX_PIECE_TOKENS * HYBRID_NODE_FEAT_DIM; ++i) tf[i] = 0.0f;
    for (int i = 0; i < HYBRID_MAX_PIECE_TOKENS; ++i) {
        tq[i] = 0;
        tr[i] = 0;
        tz[i] = 0;
        tm[i] = false;
    }
    for (int i = 0; i < HYBRID_GLOBAL_FEAT_DIM; ++i) gf[i] = 0.0f;

    float queen_surround[2] = {0.0f, 0.0f};
    for (int c = 0; c < 2; ++c) {
        uint16_t qc = s.queen_cell[c];
        if (qc != 0xFFFF) {
            int cnt = 0;
            for (int d = 0; d < NUM_DIRS; ++d) {
                int16_t nb = NEIGHBORS[qc][d];
                if (nb >= 0 && s.height[nb] > 0) cnt++;
            }
            queen_surround[c] = cnt / 6.0f;
        }
    }

    int token_count = 0;
    for (int cell = 0; cell < NUM_CELLS; ++cell) {
        int h = s.height[cell];
        if (h == 0) continue;

        int occ_flags[NUM_DIRS];
        int occ_count = 0;
        for (int d = 0; d < NUM_DIRS; ++d) {
            int16_t nb = NEIGHBORS[cell][d];
            occ_flags[d] = (nb >= 0 && s.height[nb] > 0) ? 1 : 0;
            occ_count += occ_flags[d];
        }

        int col = cell % BOARD_SIZE;
        int row = cell / BOARD_SIZE;
        for (int level = 0; level < h; ++level) {
            if (token_count >= HYBRID_MAX_PIECE_TOKENS) break;
            uint8_t packed = s.pieces[level][cell];
            PieceType pt = cell_piece_type(packed);
            Color pc = cell_color(packed);
            bool is_top = (level == h - 1);

            float* f = tf + token_count * HYBRID_NODE_FEAT_DIM;
            f[pt - 1] = 1.0f;
            f[8 + pc] = 1.0f;
            f[10] = (level == 0) ? 1.0f : 0.0f;
            f[11] = is_top ? 1.0f : 0.0f;
            f[12] = h / 4.0f;
            f[13] = (pt == PT_QUEEN) ? 1.0f : 0.0f;
            f[14] = queen_surround[pc];
            f[15] = occ_count / 6.0f;
            if (is_top) {
                for (int d = 0; d < NUM_DIRS; ++d) {
                    if (!occ_flags[d]) f[16 + d] = 1.0f;
                }
            }
            f[24] = level * 0.25f;
            f[25] = is_stunned_cell(s, cell) ? 1.0f : 0.0f;

            tq[token_count] = col;
            tr[token_count] = row;
            tz[token_count] = level;
            tm[token_count] = true;
            token_count++;
        }
    }

    Color cur = current_player(s);
    gf[0] = (cur == WHITE) ? 1.0f : 0.0f;
    gf[1] = (s.turn / 100.0f < 1.0f) ? s.turn / 100.0f : 1.0f;
    gf[2] = is_queen_placed(s, WHITE) ? 1.0f : 0.0f;
    gf[3] = is_queen_placed(s, BLACK) ? 1.0f : 0.0f;
    int white_hand = 0;
    int black_hand = 0;
    for (int p = 0; p < NUM_PIECE_TYPES; ++p) {
        white_hand += s.hands[0][p];
        black_hand += s.hands[1][p];
    }
    gf[4] = white_hand / 14.0f;
    gf[5] = black_hand / 14.0f;

    num_tokens_out[idx] = token_count;
}

__global__ void hybrid_transformer_move_features_kernel(
    const HiveState* states,
    const GPUMove* legal_moves,
    const int* num_legal,
    float* move_features,   // [B, MAX_LEGAL_MOVES, HYBRID_MOVE_FEAT_DIM]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const HiveState& s = states[idx];
    const GPUMove* my_moves = legal_moves + (int64_t)idx * MAX_LEGAL_MOVES;
    float* out = move_features + (int64_t)idx * MAX_LEGAL_MOVES * HYBRID_MOVE_FEAT_DIM;
    int nlegal = num_legal[idx];
    for (int i = 0; i < MAX_LEGAL_MOVES * HYBRID_MOVE_FEAT_DIM; ++i) out[i] = 0.0f;

    Color mover = current_player(s);
    Color opp = (mover == WHITE) ? BLACK : WHITE;
    int own_q = (int)s.queen_cell[mover];
    int opp_q = (int)s.queen_cell[opp];
    int own_q_col = (own_q >= 0 && own_q != 0xFFFF) ? (own_q % BOARD_SIZE - HALF_BOARD) : 0;
    int own_q_row = (own_q >= 0 && own_q != 0xFFFF) ? (own_q / BOARD_SIZE - HALF_BOARD) : 0;
    int opp_q_col = (opp_q >= 0 && opp_q != 0xFFFF) ? (opp_q % BOARD_SIZE - HALF_BOARD) : 0;
    int opp_q_row = (opp_q >= 0 && opp_q != 0xFFFF) ? (opp_q / BOARD_SIZE - HALF_BOARD) : 0;

    for (int m = 0; m < nlegal; ++m) {
        const GPUMove& mv = my_moves[m];
        float* f = out + (int64_t)m * HYBRID_MOVE_FEAT_DIM;

        f[(int)mv.type] = 1.0f;  // 0: place, 1: move, 2: pass
        if (mv.piece_type >= PT_QUEEN && mv.piece_type <= PT_PILLBUG) {
            f[3 + ((int)mv.piece_type - 1)] = 1.0f;
        }

        bool has_source = (mv.type == MOVE_MOVE);
        f[11] = has_source ? 1.0f : 0.0f;

        int src_q = 0;
        int src_r = 0;
        int dst_q = 0;
        int dst_r = 0;
        int src_stack = 0;
        int dst_stack = 0;
        int dst_occ = 0;

        if (mv.to_cell < NUM_CELLS) {
            dst_q = (int)mv.to_cell % BOARD_SIZE - HALF_BOARD;
            dst_r = (int)mv.to_cell / BOARD_SIZE - HALF_BOARD;
            dst_stack = s.height[mv.to_cell];
            dst_occ = dst_stack > 0 ? 1 : 0;
        }
        if (has_source && mv.from_cell < NUM_CELLS) {
            src_q = (int)mv.from_cell % BOARD_SIZE - HALF_BOARD;
            src_r = (int)mv.from_cell / BOARD_SIZE - HALF_BOARD;
            src_stack = s.height[mv.from_cell];
        }

        f[12] = has_source ? (float)(dst_q - src_q) / (float)HALF_BOARD : 0.0f;
        f[13] = has_source ? (float)(dst_r - src_r) / (float)HALF_BOARD : 0.0f;

        f[14] = (float)(dst_q - own_q_col) / (float)HALF_BOARD;
        f[15] = (float)(dst_r - own_q_row) / (float)HALF_BOARD;
        f[16] = (float)(dst_q - opp_q_col) / (float)HALF_BOARD;
        f[17] = (float)(dst_r - opp_q_row) / (float)HALF_BOARD;

        f[18] = has_source ? (float)(src_q - own_q_col) / (float)HALF_BOARD : 0.0f;
        f[19] = has_source ? (float)(src_r - own_q_row) / (float)HALF_BOARD : 0.0f;
        f[20] = has_source ? (float)(src_q - opp_q_col) / (float)HALF_BOARD : 0.0f;
        f[21] = has_source ? (float)(src_r - opp_q_row) / (float)HALF_BOARD : 0.0f;

        f[22] = (float)src_stack / (float)MAX_STACK;
        f[23] = (float)dst_stack / (float)MAX_STACK;
        f[24] = (float)dst_occ;
    }
}

// ── Encode kernel ────────────────────────────────────────────────────

__global__ void encode_states_kernel(
    const HiveState* states,
    float* node_features,        // [B, MAX_ENC_NODES, NODE_FEAT_DIM]
    int* node_grid_pos,          // [B, MAX_ENC_NODES, 2]
    int* node_piece_types,       // [B, MAX_ENC_NODES]
    float* global_features,      // [B, GLOBAL_FEAT_DIM]
    int* num_nodes_out,          // [B]
    int* num_board_nodes_out,    // [B]
    int* edge_index,             // [B, MAX_ENC_EDGES, 2]
    float* edge_features,        // [B, MAX_ENC_EDGES, EDGE_FEAT_DIM]
    int* num_edges_out,          // [B]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const HiveState& s = states[idx];

    // Output pointers for this game
    float* nf  = node_features    + idx * MAX_ENC_NODES * NODE_FEAT_DIM;
    int*   ngp = node_grid_pos    + idx * MAX_ENC_NODES * 2;
    int*   npt = node_piece_types + idx * MAX_ENC_NODES;
    float* gf  = global_features  + idx * GLOBAL_FEAT_DIM;
    int*   ei  = edge_index       + idx * MAX_ENC_EDGES * 2;
    float* ef  = edge_features    + idx * MAX_ENC_EDGES * EDGE_FEAT_DIM;

    // Zero all outputs for this game
    for (int i = 0; i < MAX_ENC_NODES * NODE_FEAT_DIM; i++) nf[i] = 0.0f;
    for (int i = 0; i < MAX_ENC_NODES * 2; i++) ngp[i] = -1;
    for (int i = 0; i < MAX_ENC_NODES; i++) npt[i] = -1;
    for (int i = 0; i < GLOBAL_FEAT_DIM; i++) gf[i] = 0.0f;
    for (int i = 0; i < MAX_ENC_EDGES * 2; i++) ei[i] = 0;
    for (int i = 0; i < MAX_ENC_EDGES * EDGE_FEAT_DIM; i++) ef[i] = 0.0f;

    // ═══ Step 1: Compute centroid ═══
    int sum_q = 0, sum_r = 0, n_occ = 0;
    for (int cell = 0; cell < NUM_CELLS; cell++) {
        if (s.height[cell] > 0) {
            sum_q += cell % BOARD_SIZE - HALF_BOARD;
            sum_r += cell / BOARD_SIZE - HALF_BOARD;
            n_occ++;
        }
    }
    int center_q = 0, center_r = 0;
    if (n_occ > 0) {
        center_q = bankers_round_div(sum_q, n_occ);
        center_r = bankers_round_div(sum_r, n_occ);
    }

    // ═══ Step 2: Queen surround counts ═══
    float queen_surround[2] = {0.0f, 0.0f};
    for (int c = 0; c < 2; c++) {
        uint16_t qc = s.queen_cell[c];
        if (qc != 0xFFFF) {
            int cnt = 0;
            for (int d = 0; d < NUM_DIRS; d++) {
                int16_t nb = NEIGHBORS[qc][d];
                if (nb >= 0 && s.height[nb] > 0) cnt++;
            }
            queen_surround[c] = cnt / 6.0f;
        }
    }

    // ═══ Step 3: Enumerate board pieces ═══
    int node_count = 0;

    // Track node indices per cell for edge building
    int16_t top_node_at[NUM_CELLS];
    int16_t first_node_at[NUM_CELLS];
    for (int i = 0; i < NUM_CELLS; i++) {
        top_node_at[i] = -1;
        first_node_at[i] = -1;
    }

    for (int cell = 0; cell < NUM_CELLS; cell++) {
        int h = s.height[cell];
        if (h == 0) continue;

        int q = cell % BOARD_SIZE - HALF_BOARD;
        int r = cell / BOARD_SIZE - HALF_BOARD;

        // Grid position in 13×13 centered on centroid
        int grid_col = q - center_q + ENC_HALF;
        int grid_row = r - center_r + ENC_HALF;
        if (grid_col < 0) grid_col = 0;
        if (grid_col >= ENC_GRID) grid_col = ENC_GRID - 1;
        if (grid_row < 0) grid_row = 0;
        if (grid_row >= ENC_GRID) grid_row = ENC_GRID - 1;

        // Compute occupied neighbor flags
        int occ_flags[6];
        int occ_count = 0;
        for (int d = 0; d < NUM_DIRS; d++) {
            int16_t nb = NEIGHBORS[cell][d];
            occ_flags[d] = (nb >= 0 && s.height[nb] > 0) ? 1 : 0;
            occ_count += occ_flags[d];
        }

        first_node_at[cell] = (int16_t)node_count;

        for (int level = 0; level < h; level++) {
            if (node_count >= MAX_ENC_NODES) break;

            uint8_t packed = s.pieces[level][cell];
            PieceType pt = cell_piece_type(packed);
            Color pc = cell_color(packed);
            bool is_top = (level == h - 1);

            float* f = nf + node_count * NODE_FEAT_DIM;

            // [0:8] piece type one-hot
            f[pt - 1] = 1.0f;

            // [8:10] color one-hot
            f[8 + pc] = 1.0f;

            // [10] is_on_ground
            f[10] = (level == 0) ? 1.0f : 0.0f;

            // [11] is_on_top
            f[11] = is_top ? 1.0f : 0.0f;

            // [12] stack_height / 4.0
            f[12] = h / 4.0f;

            // [13] is_queen
            f[13] = (pt == PT_QUEEN) ? 1.0f : 0.0f;

            // [14] queen_surround / 6.0 (same for all pieces of this color)
            f[14] = queen_surround[pc];

            // [15] num_occupied_neighbors / 6.0
            f[15] = occ_count / 6.0f;

            // [16:22] empty_dir_mask (top pieces only)
            if (is_top) {
                for (int d = 0; d < NUM_DIRS; d++) {
                    if (!occ_flags[d]) f[16 + d] = 1.0f;
                }
            }

            // [22] is_hand_node = 0 (already zeroed)
            // [23] count_remaining = 0 (already zeroed)

            // [24] stack_position
            f[24] = level * 0.25f;

            // Grid position
            ngp[node_count * 2 + 0] = grid_row;
            ngp[node_count * 2 + 1] = grid_col;

            // Piece type (0-indexed: Q=0, A=1, G=2, S=3, B=4)
            npt[node_count] = pt - 1;

            if (is_top) {
                top_node_at[cell] = (int16_t)node_count;
            }

            node_count++;
        }
    }

    int board_node_count = node_count;

    // ═══ Step 4: Hand tokens ═══
    for (int c = 0; c < 2; c++) {
        for (int p = 0; p < NUM_PIECE_TYPES; p++) {
            int count = s.hands[c][p];
            if (count == 0) continue;
            if (node_count >= MAX_ENC_NODES) break;

            float* f = nf + node_count * NODE_FEAT_DIM;

            // [p] piece type one-hot (p is 0-indexed: Q=0, A=1, ...)
            f[p] = 1.0f;

            // [8+c] color one-hot
            f[8 + c] = 1.0f;

            // [13] is_queen
            if (p == 0) f[13] = 1.0f;

            // [22] is_hand_node
            f[22] = 1.0f;

            // [23] count_remaining / count_per_player
            f[23] = count / COUNTS_PER_PLAYER[p];

            // Grid position stays (-1, -1) from zeroing
            // node_piece_types stays -1 for hand nodes

            node_count++;
        }
    }

    num_nodes_out[idx] = node_count;
    num_board_nodes_out[idx] = board_node_count;

    // ═══ Step 5: Build edges ═══
    int edge_count = 0;

    // Spatial edges: between top pieces at adjacent cells
    for (int cell = 0; cell < NUM_CELLS; cell++) {
        int16_t src_node = top_node_at[cell];
        if (src_node < 0) continue;

        for (int d = 0; d < NUM_DIRS; d++) {
            int16_t nb_cell = NEIGHBORS[cell][d];
            if (nb_cell < 0) continue;
            int16_t dst_node = top_node_at[nb_cell];
            if (dst_node < 0) continue;
            if (edge_count >= MAX_ENC_EDGES) break;

            ei[edge_count * 2 + 0] = src_node;
            ei[edge_count * 2 + 1] = dst_node;

            // Compute dq, dr from cell positions
            int src_q = cell % BOARD_SIZE - HALF_BOARD;
            int src_r = cell / BOARD_SIZE - HALF_BOARD;
            int dst_q = nb_cell % BOARD_SIZE - HALF_BOARD;
            int dst_r = nb_cell / BOARD_SIZE - HALF_BOARD;

            float* e = ef + edge_count * EDGE_FEAT_DIM;
            e[0] = (float)(dst_q - src_q);  // dq
            e[1] = (float)(dst_r - src_r);  // dr
            e[2 + d] = 1.0f;                // direction one-hot
            // e[8] = 0 (not stacked, already zeroed)

            edge_count++;
        }
    }

    // Vertical edges: between consecutive stack levels (bidirectional)
    for (int cell = 0; cell < NUM_CELLS; cell++) {
        int h = s.height[cell];
        if (h < 2) continue;
        int16_t base = first_node_at[cell];
        if (base < 0) continue;

        for (int level = 0; level < h - 1; level++) {
            int lower = base + level;
            int upper = base + level + 1;
            if (lower >= MAX_ENC_NODES || upper >= MAX_ENC_NODES) break;

            // lower → upper
            if (edge_count < MAX_ENC_EDGES) {
                ei[edge_count * 2 + 0] = lower;
                ei[edge_count * 2 + 1] = upper;
                ef[edge_count * EDGE_FEAT_DIM + 8] = 1.0f;  // is_stacked
                edge_count++;
            }
            // upper → lower
            if (edge_count < MAX_ENC_EDGES) {
                ei[edge_count * 2 + 0] = upper;
                ei[edge_count * 2 + 1] = lower;
                ef[edge_count * EDGE_FEAT_DIM + 8] = 1.0f;  // is_stacked
                edge_count++;
            }
        }
    }

    num_edges_out[idx] = edge_count;

    // ═══ Step 6: Global features ═══
    Color cur = current_player(s);
    gf[0] = (cur == WHITE) ? 1.0f : 0.0f;
    gf[1] = (s.turn / 100.0f < 1.0f) ? s.turn / 100.0f : 1.0f;
    gf[2] = is_queen_placed(s, WHITE) ? 1.0f : 0.0f;
    gf[3] = is_queen_placed(s, BLACK) ? 1.0f : 0.0f;

    // Hand sizes
    int white_hand = 0, black_hand = 0;
    for (int p = 0; p < NUM_PIECE_TYPES; p++) {
        white_hand += s.hands[0][p];
        black_hand += s.hands[1][p];
    }
    gf[4] = white_hand / 14.0f;
    gf[5] = black_hand / 14.0f;
}

// ── Legal mask kernel ─────────────────────────────────────────────────
//
// Maps GPU legal moves to the 29407-dim action space used by the NN.
// Action encoding matches hive_engine/encoder.py exactly:
//   PLACE:  piece_type_0idx * 169 + enc_row * 13 + enc_col
//   MOVE:   845 + src_enc_idx * 169 + dst_enc_idx
//   PASS:   29406

__device__ __forceinline__ void compute_centroid(
    const HiveState& s, int& center_q, int& center_r
) {
    int sum_q = 0, sum_r = 0, n_occ = 0;
    for (int cell = 0; cell < NUM_CELLS; cell++) {
        if (s.height[cell] > 0) {
            sum_q += cell % BOARD_SIZE - HALF_BOARD;
            sum_r += cell / BOARD_SIZE - HALF_BOARD;
            n_occ++;
        }
    }
    center_q = 0;
    center_r = 0;
    if (n_occ > 0) {
        center_q = bankers_round_div(sum_q, n_occ);
        center_r = bankers_round_div(sum_r, n_occ);
    }
}

__global__ void legal_mask_kernel(
    const HiveState* states,
    const GPUMove* moves,     // [B, MAX_LEGAL_MOVES]
    const int* num_legal,     // [B]
    float* masks,             // [B, ACTION_SPACE_SIZE]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const HiveState& s = states[idx];
    const GPUMove* my_moves = moves + idx * MAX_LEGAL_MOVES;
    int n_legal = num_legal[idx];
    float* mask = masks + idx * ACTION_SPACE_SIZE;

    // Compute centroid (same as encode_states_kernel)
    int center_q, center_r;
    compute_centroid(s, center_q, center_r);

    for (int m = 0; m < n_legal; m++) {
        const GPUMove& mv = my_moves[m];

        if (mv.type == MOVE_PASS) {
            mask[PASS_ACTION_INDEX] = 1.0f;
            continue;
        }

        if (mv.type == MOVE_PLACE) {
            // Map to_cell to encoder grid
            int to_q = mv.to_cell % BOARD_SIZE - HALF_BOARD;
            int to_r = mv.to_cell / BOARD_SIZE - HALF_BOARD;
            int enc_col = to_q - center_q + ENC_HALF;
            int enc_row = to_r - center_r + ENC_HALF;

            // Skip if outside encoder grid (matches CPU try/except ValueError)
            if (enc_col < 0 || enc_col >= ENC_GRID || enc_row < 0 || enc_row >= ENC_GRID)
                continue;

            // piece_type is 1-based (PT_QUEEN=1..PT_BEETLE=5), action uses 0-based
            int pt_0idx = (int)mv.piece_type - 1;
            int pos_idx = enc_row * ENC_GRID + enc_col;
            int action = pt_0idx * NUM_ENC_GRID_CELLS + pos_idx;
            mask[action] = 1.0f;
        } else {
            // MOVE_MOVE
            int from_q = mv.from_cell % BOARD_SIZE - HALF_BOARD;
            int from_r = mv.from_cell / BOARD_SIZE - HALF_BOARD;
            int to_q = mv.to_cell % BOARD_SIZE - HALF_BOARD;
            int to_r = mv.to_cell / BOARD_SIZE - HALF_BOARD;

            int src_enc_col = from_q - center_q + ENC_HALF;
            int src_enc_row = from_r - center_r + ENC_HALF;
            int dst_enc_col = to_q - center_q + ENC_HALF;
            int dst_enc_row = to_r - center_r + ENC_HALF;

            // Skip if either position is outside encoder grid
            if (src_enc_col < 0 || src_enc_col >= ENC_GRID ||
                src_enc_row < 0 || src_enc_row >= ENC_GRID ||
                dst_enc_col < 0 || dst_enc_col >= ENC_GRID ||
                dst_enc_row < 0 || dst_enc_row >= ENC_GRID)
                continue;

            int src_idx = src_enc_row * ENC_GRID + src_enc_col;
            int dst_idx = dst_enc_row * ENC_GRID + dst_enc_col;
            int action = MOVEMENT_OFFSET + src_idx * NUM_ENC_GRID_CELLS + dst_idx;
            mask[action] = 1.0f;
        }
    }
}

// ── Centroid kernel ─────────────────────────────────────────────────
//
// Computes per-state centroids on GPU, avoiding CPU round-trips.

__global__ void compute_centroids_kernel(
    const HiveState* states,
    int* centroids_out,       // [batch_size, 2]  (center_q, center_r)
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const HiveState& s = states[idx];
    int center_q, center_r;
    compute_centroid(s, center_q, center_r);

    centroids_out[idx * 2 + 0] = center_q;
    centroids_out[idx * 2 + 1] = center_r;
}

#endif  // __CUDACC__

}  // namespace hive_gpu
