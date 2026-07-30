// GPU-resident iterative-deepening alpha-beta search for the FNN.
#pragma once

#include "fnn_selfplay.cuh"

namespace hive_gpu {

#ifdef __CUDACC__

constexpr float AB_INF = 1000.0f;
constexpr float AB_MATE = 10.0f;
constexpr float AB_PVS_EPSILON = 1e-4f;
constexpr int AB_MAX_PV = 64;
constexpr int AB_HISTORY_SIZE = 1024;
constexpr int AB_COUNTERMOVE_SIZE = 1024;
constexpr int AB_CONTINUATION_SIZE = 2048;
constexpr int AB_ORDERING_SIZE =
    AB_HISTORY_SIZE + AB_COUNTERMOVE_SIZE + AB_CONTINUATION_SIZE;

enum ABTTBound : uint8_t {
    AB_TT_EMPTY = 0,
    AB_TT_EXACT = 1,
    AB_TT_LOWER = 2,
    AB_TT_UPPER = 3,
};

struct ABTT {
    uint64_t* keys;
    float* values;
    int16_t* depths;
    uint8_t* bounds;
    uint32_t* moves;
    int* generations;
    int generation;
    int mask;
};

struct ABStats {
    int nodes;
    int cutoffs;
    int tt_hits;
    int pvs_researches;
    int lmr_reductions;
    int qnodes;
    int forced_win_probes;
    int tactical_moves;
    bool aborted;
    bool q_exhausted;
};

struct ABSearchConfig {
    float aspiration_window;
    int lmr_min_depth;
    int lmr_min_move;
    int lmr_reduction;
    int quiescence_plies;
    float quiescence_budget_fraction;
    bool force_win_probes;
    int tactical_mask;
    float branching_allocation;
    float early_stop_score;
    int early_stop_min_depth;
    bool recursive_threat_qsearch;
    bool forced_extensions;
    int forced_extension_max_chain;
    bool singular_extensions;
    int singular_min_depth;
    float singular_margin;
    bool proof_search;
    int proof_max_plies;
    float proof_budget_fraction;
    int proof_trigger_surround;
    bool persistent_tt;
    bool countermove_ordering;
    bool continuation_history;
    bool internal_heuristic_ordering;
};

constexpr int AB_TACTICAL_IMMOBILIZE = 1;
constexpr int AB_TACTICAL_OPP_SURROUND = 2;
constexpr int AB_TACTICAL_OWN_RELIEF = 4;
constexpr int AB_TACTICAL_QUEEN_THREAT = 8;

__device__ inline ABSearchConfig ab_make_search_config(const float* values) {
    ABSearchConfig config;
    config.aspiration_window = max(0.0f, values[0]);
    config.lmr_min_depth = max(1, (int)values[1]);
    config.lmr_min_move = max(1, (int)values[2]);
    config.lmr_reduction = max(0, (int)values[3]);
    config.recursive_threat_qsearch = values[11] >= 0.5f;
    config.quiescence_plies = min(
        config.recursive_threat_qsearch ? 4 : 1,
        max(0, (int)values[4]));
    config.quiescence_budget_fraction = min(0.95f, max(0.0f, values[5]));
    config.force_win_probes = values[6] >= 0.5f;
    config.tactical_mask = (int)values[7];
    config.branching_allocation = min(0.75f, max(-0.75f, values[8]));
    config.early_stop_score = min(9.99f, max(1.0f, values[9]));
    config.early_stop_min_depth = max(1, (int)values[10]);
    config.forced_extensions = values[12] >= 0.5f;
    config.forced_extension_max_chain = max(0, (int)values[13]);
    config.singular_extensions = values[14] >= 0.5f;
    config.singular_min_depth = max(2, (int)values[15]);
    config.singular_margin = max(0.0f, values[16]);
    config.proof_search = values[17] >= 0.5f;
    config.proof_max_plies = max(1, (int)values[18]);
    config.proof_budget_fraction = min(0.75f, max(0.0f, values[19]));
    config.proof_trigger_surround = min(5, max(0, (int)values[20]));
    config.persistent_tt = values[21] >= 0.5f;
    config.countermove_ordering = values[22] >= 0.5f;
    config.continuation_history = values[23] >= 0.5f;
    config.internal_heuristic_ordering = values[24] >= 0.5f;
    return config;
}

__device__ inline bool ab_move_equal(const GPUMove& a, const GPUMove& b) {
    return a.type == b.type && a.piece_type == b.piece_type &&
           a.from_cell == b.from_cell && a.to_cell == b.to_cell;
}

using ABPackedMove = uint32_t;
constexpr ABPackedMove AB_NO_MOVE = 0xFFFFFFFFU;

__device__ __host__ __forceinline__ ABPackedMove ab_pack_move(
    const GPUMove& move
) {
    uint32_t from = move.from_cell < NUM_CELLS ? move.from_cell : 0x3FFU;
    uint32_t to = move.to_cell < NUM_CELLS ? move.to_cell : 0x3FFU;
    return ((uint32_t)move.type & 0x3U) |
        (((uint32_t)move.piece_type & 0xFU) << 2) |
        (from << 6) | (to << 16);
}

__device__ __host__ __forceinline__ GPUMove ab_unpack_move(
    ABPackedMove packed
) {
    GPUMove move = {};
    move.type = (MoveType)(packed & 0x3U);
    move.piece_type = (PieceType)((packed >> 2) & 0xFU);
    uint16_t from = (uint16_t)((packed >> 6) & 0x3FFU);
    uint16_t to = (uint16_t)((packed >> 16) & 0x3FFU);
    move.from_cell = from == 0x3FFU ? 0xFFFF : from;
    move.to_cell = to == 0x3FFU ? 0xFFFF : to;
    return move;
}

__device__ __forceinline__ uint64_t ab_mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ inline uint64_t ab_cell_hash(const HiveState& state, int cell) {
    if (cell < 0 || cell >= NUM_CELLS) return 0;
    uint64_t hash = ab_mix64(0x100000ULL + (uint64_t)cell * 17ULL + state.height[cell]);
    for (int level = 0; level < state.height[cell]; ++level) {
        uint64_t key = (uint64_t)cell | ((uint64_t)level << 10) |
            ((uint64_t)state.pieces[level][cell] << 16);
        hash ^= ab_mix64(key);
    }
    return hash;
}

__device__ inline uint64_t ab_metadata_hash(const HiveState& state) {
    uint64_t hash = ab_mix64(0x200000ULL + state.turn);
    hash ^= ab_mix64(0x210000ULL + state.stunned_cell);
    hash ^= ab_mix64(0x220000ULL + (uint64_t)state.result);
    for (int color = 0; color < 2; ++color) {
        for (int type = 0; type < NUM_PIECE_TYPES; ++type) {
            uint64_t key = 0x300000ULL + (uint64_t)(color * NUM_PIECE_TYPES + type) * 16ULL +
                state.hands[color][type];
            hash ^= ab_mix64(key);
        }
    }
    return hash;
}

__device__ inline uint64_t ab_hash_state(const HiveState& state) {
    uint64_t hash = ab_metadata_hash(state);
    for (int cell = 0; cell < NUM_CELLS; ++cell) {
        if (state.height[cell] > 0) hash ^= ab_cell_hash(state, cell);
    }
    return hash == 0 ? 1ULL : hash;
}

struct ABUndo {
    uint16_t from_cell;
    uint16_t to_cell;
    uint16_t turn;
    uint16_t stunned_cell;
    uint16_t queen_cell[2];
    uint8_t queen_placed;
    uint8_t result;
    uint8_t hand_color;
    uint8_t hand_type;
    uint8_t hand_count;
    uint8_t from_height;
    uint8_t to_height;
    uint8_t from_pieces[MAX_STACK];
    uint8_t to_pieces[MAX_STACK];
    uint8_t cell_flags[2];
};

__device__ __forceinline__ uint8_t ab_cell_flags(const HiveState& state, int cell) {
    if (cell < 0 || cell >= NUM_CELLS) return 0;
    return (state.occupied.get(cell) ? 1 : 0) |
        (state.white_top.get(cell) ? 2 : 0) |
        (state.black_top.get(cell) ? 4 : 0);
}

__device__ __forceinline__ void ab_restore_cell_flags(
    HiveState& state, int cell, uint8_t flags
) {
    if (cell < 0 || cell >= NUM_CELLS) return;
    if (flags & 1) state.occupied.set(cell); else state.occupied.clr(cell);
    if (flags & 2) state.white_top.set(cell); else state.white_top.clr(cell);
    if (flags & 4) state.black_top.set(cell); else state.black_top.clr(cell);
}

__device__ inline void ab_capture_undo(
    const HiveState& state, const GPUMove& move, ABUndo& undo
) {
    undo.from_cell = move.type == MOVE_MOVE ? move.from_cell : 0xFFFF;
    undo.to_cell = move.type == MOVE_PASS ? 0xFFFF : move.to_cell;
    undo.turn = state.turn;
    undo.stunned_cell = state.stunned_cell;
    undo.queen_cell[0] = state.queen_cell[0];
    undo.queen_cell[1] = state.queen_cell[1];
    undo.queen_placed = state.queen_placed;
    undo.result = (uint8_t)state.result;
    undo.hand_color = 0xFF;
    undo.hand_type = 0xFF;
    undo.hand_count = 0;
    if (move.type == MOVE_PLACE) {
        undo.hand_color = (uint8_t)current_player(state);
        undo.hand_type = (uint8_t)((int)move.piece_type - 1);
        undo.hand_count = state.hands[undo.hand_color][undo.hand_type];
    }
    undo.from_height = undo.from_cell != 0xFFFF ? state.height[undo.from_cell] : 0;
    undo.to_height = undo.to_cell != 0xFFFF ? state.height[undo.to_cell] : 0;
    undo.cell_flags[0] = ab_cell_flags(state, undo.from_cell);
    undo.cell_flags[1] = ab_cell_flags(state, undo.to_cell);
    for (int level = 0; level < MAX_STACK; ++level) {
        undo.from_pieces[level] = undo.from_cell != 0xFFFF ?
            state.pieces[level][undo.from_cell] : 0;
        undo.to_pieces[level] = undo.to_cell != 0xFFFF ?
            state.pieces[level][undo.to_cell] : 0;
    }

}

__device__ inline uint64_t ab_make_move(
    HiveState& state, const GPUMove& move, uint64_t hash, ABUndo& undo
) {
    ab_capture_undo(state, move, undo);
    uint64_t next_hash = hash ^ ab_metadata_hash(state);
    if (undo.from_cell != 0xFFFF) next_hash ^= ab_cell_hash(state, undo.from_cell);
    if (undo.to_cell != 0xFFFF && undo.to_cell != undo.from_cell) {
        next_hash ^= ab_cell_hash(state, undo.to_cell);
    }
    apply_move(state, move);
    next_hash ^= ab_metadata_hash(state);
    if (undo.from_cell != 0xFFFF) next_hash ^= ab_cell_hash(state, undo.from_cell);
    if (undo.to_cell != 0xFFFF && undo.to_cell != undo.from_cell) {
        next_hash ^= ab_cell_hash(state, undo.to_cell);
    }
    return next_hash == 0 ? 1ULL : next_hash;
}

__device__ inline void ab_make_move_unhashed(
    HiveState& state, const GPUMove& move, ABUndo& undo
) {
    ab_capture_undo(state, move, undo);
    apply_move(state, move);
}

__device__ inline void ab_unmake_move(HiveState& state, const ABUndo& undo) {
    state.turn = undo.turn;
    state.stunned_cell = undo.stunned_cell;
    state.queen_cell[0] = undo.queen_cell[0];
    state.queen_cell[1] = undo.queen_cell[1];
    state.queen_placed = undo.queen_placed;
    state.result = (GameResult)undo.result;
    if (undo.hand_color < 2 && undo.hand_type < NUM_PIECE_TYPES) {
        state.hands[undo.hand_color][undo.hand_type] = undo.hand_count;
    }
    if (undo.from_cell != 0xFFFF) {
        state.height[undo.from_cell] = undo.from_height;
        for (int level = 0; level < MAX_STACK; ++level) {
            state.pieces[level][undo.from_cell] = undo.from_pieces[level];
        }
        ab_restore_cell_flags(state, undo.from_cell, undo.cell_flags[0]);
    }
    if (undo.to_cell != 0xFFFF && undo.to_cell != undo.from_cell) {
        state.height[undo.to_cell] = undo.to_height;
        for (int level = 0; level < MAX_STACK; ++level) {
            state.pieces[level][undo.to_cell] = undo.to_pieces[level];
        }
        ab_restore_cell_flags(state, undo.to_cell, undo.cell_flags[1]);
    }
}

__device__ inline float ab_terminal_value(const HiveState& state, int ply) {
    if (state.result == DRAW || state.result == IN_PROGRESS) return 0.0f;
    Color side = current_player(state);
    bool side_won = (state.result == WHITE_WINS && side == WHITE) ||
                    (state.result == BLACK_WINS && side == BLACK);
    return (side_won ? 1.0f : -1.0f) * (AB_MATE - min(ply, 100) * 0.01f);
}

__device__ inline float ab_evaluate_with_moves_and_ap(
    const HiveState& state, const GPUMove* moves, int n,
    const Bitboard& ap_mask, const float* params, const FNNWeights& weights
) {
    float features[FNN_FEAT_DIM];
    float embed[FNN_MAX_EMBED];
    extract_fnn_features_with_ap_device(
        state, moves, n, ap_mask, features);
    fnn_encode(features, embed, params, weights);
    return fnn_value(embed, params, weights);
}

__device__ inline float ab_evaluate(
    const HiveState& state, const float* params, const FNNWeights& weights
) {
    GPUMove moves[MAX_LEGAL_MOVES];
    MovegenStateCache cache;
    int n = generate_fnn_feature_moves_with_cache(state, moves, cache);
    return ab_evaluate_with_moves_and_ap(
        state, moves, n, cache.ap_mask, params, weights);
}

__device__ __forceinline__ bool ab_adjacent_to(int cell, uint16_t target) {
    if (cell < 0 || cell >= NUM_CELLS || target == 0xFFFF) return false;
    for (int direction = 0; direction < NUM_DIRS; ++direction) {
        if (NEIGHBORS[(int)target][direction] == cell) return true;
    }
    return false;
}

__device__ __forceinline__ int ab_move_history_index(const GPUMove& move) {
    uint32_t key = (uint32_t)move.to_cell * 37U + (uint32_t)move.from_cell * 11U +
        (uint32_t)move.piece_type * 131U + (uint32_t)move.type * 521U;
    return (int)(key & (AB_HISTORY_SIZE - 1));
}

__device__ inline float ab_search_order_score(
    const HiveState& state, const GPUMove& move, int ply,
    const ABPackedMove* killers, const int* history,
    const ABSearchConfig& config, ABPackedMove previous_move
) {
    Color mover = current_player(state);
    Color opponent = mover == WHITE ? BLACK : WHITE;
    int move_index = ab_move_history_index(move);
    ABPackedMove packed = ab_pack_move(move);
    float score = (float)history[move_index];
    if (previous_move != AB_NO_MOVE) {
        int previous_index = (int)(previous_move & (AB_HISTORY_SIZE - 1));
        if (config.countermove_ordering &&
            (ABPackedMove)history[AB_HISTORY_SIZE + previous_index] == packed) {
            score += 850000.0f;
        }
        if (config.continuation_history) {
            int continuation_index = (int)(
                (previous_move * 33U + (uint32_t)move_index) &
                (AB_CONTINUATION_SIZE - 1));
            score += (float)history[
                AB_HISTORY_SIZE + AB_COUNTERMOVE_SIZE + continuation_index];
        }
    }
    if (ply < AB_MAX_PV) {
        if (packed == killers[ply * 2]) score += 1000000.0f;
        else if (packed == killers[ply * 2 + 1]) score += 900000.0f;
    }
    if (ab_adjacent_to((int)move.to_cell, state.queen_cell[opponent])) {
        score += 10000.0f;
    }
    if (move.type == MOVE_MOVE &&
        ab_adjacent_to((int)move.from_cell, state.queen_cell[mover]) &&
        !ab_adjacent_to((int)move.to_cell, state.queen_cell[mover])) {
        score += 8000.0f;
    }
    if (move.piece_type == PT_BEETLE || move.piece_type == PT_QUEEN ||
        move.piece_type == PT_PILLBUG) {
        score += 100.0f;
    }
    if (config.internal_heuristic_ordering) {
        int pressure = queen_surround_count_for_color_device(state, opponent);
        if (ab_adjacent_to((int)move.to_cell, state.queen_cell[opponent])) {
            score += 2500.0f * pressure;
        }
        if (move.type == MOVE_MOVE &&
            ab_adjacent_to((int)move.from_cell, state.queen_cell[opponent]) &&
            !ab_adjacent_to((int)move.to_cell, state.queen_cell[opponent])) {
            score -= 4000.0f;
        }
        if (move.piece_type == PT_BEETLE || move.piece_type == PT_PILLBUG) {
            score += 250.0f * pressure;
        }
    }
    return score;
}

__device__ inline void ab_order_move_at_rank(
    const HiveState& state, ABPackedMove* moves, int n, int rank,
    int ply, const ABPackedMove* killers, const int* history,
    ABPackedMove tt_move, const ABSearchConfig& config,
    ABPackedMove previous_move
) {
    int best_index = rank;
    float best_score = -1e30f;
    for (int i = rank; i < n; ++i) {
        GPUMove move = ab_unpack_move(moves[i]);
        float score = ab_search_order_score(
            state, move, ply, killers, history, config, previous_move);
        if (moves[i] == tt_move) {
            score += 1e20f;
        }
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }
    if (best_index != rank) {
        // Stable extraction preserves the old selected-mask tie order while
        // eliminating that extra per-frame array.
        ABPackedMove best = moves[best_index];
        for (int i = best_index; i > rank; --i) moves[i] = moves[i - 1];
        moves[rank] = best;
    }
}

__device__ inline void ab_order_native_move_at_rank(
    const HiveState& state, GPUMove* moves, int n, int rank,
    int ply, const ABPackedMove* killers, const int* history,
    ABPackedMove tt_move, const ABSearchConfig& config,
    ABPackedMove previous_move
) {
    int best_index = rank;
    float best_score = -1e30f;
    for (int i = rank; i < n; ++i) {
        float score = ab_search_order_score(
            state, moves[i], ply, killers, history, config, previous_move);
        if (ab_pack_move(moves[i]) == tt_move) score += 1e20f;
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }
    if (best_index != rank) {
        GPUMove best = moves[best_index];
        for (int i = best_index; i > rank; --i) moves[i] = moves[i - 1];
        moves[rank] = best;
    }
}

__device__ __noinline__ int ab_generate_packed_moves(
    const HiveState& state, ABPackedMove* packed
) {
    GPUMove generated[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(state, generated);
    for (int i = 0; i < n; ++i) packed[i] = ab_pack_move(generated[i]);
    return n;
}

__device__ __forceinline__ float ab_tt_store_value(
    float value, int ply, bool normalize
) {
    if (!normalize) return value;
    if (value > AB_MATE - 1.1f) return value + min(ply, 100) * 0.01f;
    if (value < -AB_MATE + 1.1f) return value - min(ply, 100) * 0.01f;
    return value;
}

__device__ __forceinline__ float ab_tt_load_value(
    float value, int ply, bool normalize
) {
    if (!normalize) return value;
    if (value > AB_MATE - 1.1f) return value - min(ply, 100) * 0.01f;
    if (value < -AB_MATE + 1.1f) return value + min(ply, 100) * 0.01f;
    return value;
}

__device__ inline void ab_tt_store(
    const ABTT& tt, uint64_t key, int depth, float value, uint8_t bound,
    const GPUMove& best_move, int ply, bool normalize_mate
) {
    int slot = (int)(key & (uint64_t)tt.mask);
    bool occupied = tt.generations[slot] == tt.generation;
    if (!occupied || tt.keys[slot] == key || depth >= (int)tt.depths[slot]) {
        // Publish the key last so a partially-written entry is never accepted.
        tt.values[slot] = ab_tt_store_value(value, ply, normalize_mate);
        tt.depths[slot] = (int16_t)depth;
        tt.bounds[slot] = bound;
        tt.moves[slot] = ab_pack_move(best_move);
        __threadfence();
        tt.keys[slot] = key;
        tt.generations[slot] = tt.generation;
    }
}

__device__ inline bool ab_take_node(
    ABStats* stats, int node_budget, const ABSearchConfig& config,
    bool quiescence = false
) {
    int qnode_budget = max(
        1, (int)(node_budget * config.quiescence_budget_fraction));
    if (quiescence && stats->qnodes >= qnode_budget) {
        stats->q_exhausted = true;
        return false;
    }
    if (stats->nodes >= node_budget) {
        stats->aborted = true;
        return false;
    }
    ++stats->nodes;
    if (quiescence) ++stats->qnodes;
    return true;
}

__device__ inline bool ab_color_won(const HiveState& state, Color color) {
    return (color == WHITE && state.result == WHITE_WINS) ||
           (color == BLACK && state.result == BLACK_WINS);
}

__device__ inline bool ab_side_under_immediate_threat(const HiveState& state) {
    if (state.result != IN_PROGRESS) return false;
    Color side = current_player(state);
    if (queen_surround_count_for_color_device(state, side) < 5) return false;
    HiveState probe = state;
    probe.turn ^= 1U;
    probe.stunned_cell = 0xFFFF;
    return has_immediate_surround_win_for_current_player(probe);
}

__device__ inline bool ab_power_piece_mobile(
    const HiveState& state, int cell, MovegenStateCache& cache
) {
    if (cell < 0 || cell >= NUM_CELLS || state.height[cell] == 0 ||
        is_stunned_cell(state, cell) || is_pinned(cache, cell)) {
        return false;
    }
    PieceType type = top_piece_type_at(state, cell);
    if (type == PT_QUEEN) return has_queen_move(state, cell);
    if (type == PT_BEETLE) return has_beetle_move(state, cell);
    if (type == PT_ANT) {
        const Bitboard& perimeter = ensure_base_perimeter(state, cache);
        return has_ant_move_with_perimeter(state, cell, perimeter);
    }
    return false;
}

__device__ inline Bitboard ab_mobile_power_pieces(
    const HiveState& state, Color color, MovegenStateCache& cache
) {
    Bitboard mobile;
    mobile.clear();
    const Bitboard& tops = color == WHITE ? state.white_top : state.black_top;
    for (int wi = 0; wi < BB_WORDS; ++wi) {
        uint64_t bits = tops.w[wi];
        while (bits) {
            int bit = __ffsll(bits) - 1;
            int cell = wi * 64 + bit;
            bits &= bits - 1;
            if (cell < NUM_CELLS && ab_power_piece_mobile(state, cell, cache)) {
                mobile.set(cell);
            }
        }
    }
    return mobile;
}

__device__ inline bool ab_immobilizes_from_mask(
    const Bitboard& mobile_before, Color target, const GPUMove& move,
    const HiveState& child, MovegenStateCache& child_cache
) {
    for (int wi = 0; wi < BB_WORDS; ++wi) {
        uint64_t bits = mobile_before.w[wi];
        while (bits) {
            int bit = __ffsll(bits) - 1;
            int cell = wi * 64 + bit;
            bits &= bits - 1;
            if (move.type == MOVE_MOVE && (int)move.from_cell == cell) continue;
            bool still_mobile = false;
            if (cell < NUM_CELLS && child.height[cell] > 0 &&
                top_piece_color_at(child, cell) == target) {
                still_mobile = ab_power_piece_mobile(
                    child, cell, child_cache);
            }
            if (!still_mobile) return true;
        }
    }
    return false;
}

__device__ inline bool ab_creates_queen_threat(
    const HiveState& child, Color mover
) {
    if (child.result != IN_PROGRESS) return false;
    Color opponent = mover == WHITE ? BLACK : WHITE;
    if (queen_surround_count_for_color_device(child, opponent) != 5) {
        return false;
    }
    HiveState probe = child;
    probe.turn = (uint16_t)((probe.turn & ~1U) | (uint16_t)mover);
    probe.stunned_cell = 0xFFFF;
    return has_immediate_surround_win_for_current_player(probe);
}

__device__ inline bool ab_priority_q_candidate(
    const HiveState& state, const GPUMove& move, Color mover, Color opponent,
    const Bitboard& mobile_power
) {
    if (ab_adjacent_to((int)move.to_cell, state.queen_cell[opponent])) {
        return true;
    }
    if (move.type == MOVE_MOVE) {
        if (move.from_cell == state.queen_cell[mover] ||
            ab_adjacent_to((int)move.from_cell, state.queen_cell[mover])) {
            return true;
        }
        for (int wi = 0; wi < BB_WORDS; ++wi) {
            uint64_t bits = mobile_power.w[wi];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int cell = wi * 64 + bit;
                bits &= bits - 1;
                if ((int)move.to_cell == cell ||
                    ab_adjacent_to((int)move.to_cell, (uint16_t)cell) ||
                    ab_adjacent_to((int)move.from_cell, (uint16_t)cell)) {
                    return true;
                }
            }
        }
    }
    return false;
}

__device__ bool ab_prove_forced_win(
    HiveState& state, Color attacker, int remaining, int* nodes, int limit
) {
    if (state.result != IN_PROGRESS) return ab_color_won(state, attacker);
    if (remaining <= 0 || *nodes >= limit) return false;
    ++(*nodes);
    GPUMove moves[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(state, moves);
    if (n <= 0) return false;
    bool attacker_turn = current_player(state) == attacker;
    if (attacker_turn) {
        for (int i = 0; i < n; ++i) {
            ABUndo undo;
            ab_make_move_unhashed(state, moves[i], undo);
            bool proven = ab_prove_forced_win(
                state, attacker, remaining - 1, nodes, limit);
            ab_unmake_move(state, undo);
            if (proven) return true;
            if (*nodes >= limit) return false;
        }
        return false;
    }
    for (int i = 0; i < n; ++i) {
        ABUndo undo;
        ab_make_move_unhashed(state, moves[i], undo);
        bool proven = ab_prove_forced_win(
            state, attacker, remaining - 1, nodes, limit);
        ab_unmake_move(state, undo);
        if (!proven) return false;
    }
    return true;
}

__device__ float ab_quiescence(
    HiveState& state, float alpha, float beta, int ply,
    const float* params, const FNNWeights& weights, int node_budget,
    ABStats* stats, const ABSearchConfig& config, int remaining = -1
) {
    if (state.result != IN_PROGRESS) return ab_terminal_value(state, ply);
    int qplies = remaining < 0 ? config.quiescence_plies : remaining;
    if (qplies <= 0) {
        return ab_evaluate(state, params, weights);
    }
    GPUMove moves[MAX_LEGAL_MOVES];
    MovegenStateCache state_cache;
    int n = generate_legal_moves_with_cache(state, moves, state_cache);
    float best = ab_evaluate_with_moves_and_ap(
        state, moves, n, state_cache.ap_mask, params, weights);
    if (best >= beta || n <= 0) return best;
    alpha = max(alpha, best);

    Color mover = current_player(state);
    Color tactical_target = mover == WHITE ? BLACK : WHITE;
    Bitboard mobile_power;
    mobile_power.clear();
    if (config.tactical_mask & AB_TACTICAL_IMMOBILIZE) {
        mobile_power = ab_mobile_power_pieces(
            state, tactical_target, state_cache);
    }
    int own_before = queen_surround_count_for_color_device(state, mover);
    for (int phase = 0; phase < 2; ++phase) {
        for (int index = 0; index < n; ++index) {
            bool priority = ab_priority_q_candidate(
                state, moves[index], mover, tactical_target, mobile_power);
            if (priority != (phase == 0)) continue;
            if (!ab_take_node(stats, node_budget, config, true)) {
                return stats->aborted ? 0.0f : best;
            }
            Color opponent = mover == WHITE ? BLACK : WHITE;
            ABUndo undo;
            ab_make_move_unhashed(state, moves[index], undo);
            bool mover_won = ab_color_won(state, mover);
            int own_after = queen_surround_count_for_color_device(state, mover);
            bool tactical = mover_won ||
                ((config.tactical_mask & AB_TACTICAL_OWN_RELIEF) &&
                 own_after < own_before) ||
                ((config.tactical_mask & AB_TACTICAL_QUEEN_THREAT) &&
                 queen_surround_count_for_color_device(state, opponent) == 5 &&
                 ab_creates_queen_threat(state, mover));

            MovegenStateCache child_cache;
            bool child_cache_ready = false;
            if (!tactical &&
                (config.tactical_mask & AB_TACTICAL_IMMOBILIZE) &&
                !mobile_power.is_zero()) {
                init_movegen_state_cache(state, child_cache);
                child_cache_ready = true;
                tactical = ab_immobilizes_from_mask(
                    mobile_power, tactical_target, moves[index], state,
                    child_cache);
            }
            if (!tactical) {
                ab_unmake_move(state, undo);
                continue;
            }
            ++stats->tactical_moves;
            float value;
            if (state.result != IN_PROGRESS) {
                value = -ab_terminal_value(state, ply + 1);
            } else if (config.recursive_threat_qsearch && qplies > 1) {
                value = -ab_quiescence(
                    state, -beta, -alpha, ply + 1, params, weights,
                    node_budget, stats, config, qplies - 1);
            } else {
                GPUMove feature_moves[MAX_LEGAL_MOVES];
                if (!child_cache_ready) {
                    init_movegen_state_cache(state, child_cache);
                }
                int feature_n = generate_fnn_feature_moves_from_cache(
                    state, feature_moves, child_cache);
                value = -ab_evaluate_with_moves_and_ap(
                    state, feature_moves, feature_n, child_cache.ap_mask,
                    params, weights);
            }
            ab_unmake_move(state, undo);
            best = max(best, value);
            alpha = max(alpha, best);
            if (alpha >= beta) {
                ++stats->cutoffs;
                return best;
            }
        }
    }
    return best;
}

__device__ float ab_negamax(
    HiveState& state, uint64_t hash, int depth, float alpha, float beta, int ply,
    const float* params, const FNNWeights& weights, int node_budget,
    ABStats* stats, const ABTT& tt, const ABSearchConfig& config,
    ABPackedMove* killers, int* history, int extension_chain = 0,
    ABPackedMove previous_move = AB_NO_MOVE
) {
    if (!ab_take_node(stats, node_budget, config)) return 0.0f;
    if (state.result != IN_PROGRESS) return ab_terminal_value(state, ply);
    if (depth <= 0) {
        return ab_quiescence(
            state, alpha, beta, ply, params, weights, node_budget,
            stats, config);
    }

    const float alpha_start = alpha;
    const float beta_start = beta;
    const uint64_t key = hash;
    const int tt_slot = (int)(key & (uint64_t)tt.mask);
    ABPackedMove tt_move = AB_NO_MOVE;
    bool has_tt_move = false;
    if (tt.generations[tt_slot] == tt.generation &&
        tt.keys[tt_slot] == key) {
        has_tt_move = true;
        tt_move = tt.moves[tt_slot];
        if ((int)tt.depths[tt_slot] >= depth) {
            ++stats->tt_hits;
            float tt_value = ab_tt_load_value(
                tt.values[tt_slot], ply, config.persistent_tt);
            uint8_t tt_bound = tt.bounds[tt_slot];
            if (tt_bound == AB_TT_EXACT) return tt_value;
            if (tt_bound == AB_TT_LOWER) alpha = max(alpha, tt_value);
            if (tt_bound == AB_TT_UPPER) beta = min(beta, tt_value);
            if (alpha >= beta) return tt_value;
        }
    }

    GPUMove moves[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(state, moves);
    if (n <= 0) return ab_evaluate(state, params, weights);
    bool extend_node = config.forced_extensions &&
        extension_chain < config.forced_extension_max_chain &&
        (n == 1 || ab_side_under_immediate_threat(state));
    int next_depth = depth - 1 + (extend_node ? 1 : 0);
    int next_extension_chain = extend_node ? extension_chain + 1 : 0;
    ABPackedMove singular_move = AB_NO_MOVE;
    if (config.singular_extensions && has_tt_move &&
        depth >= config.singular_min_depth &&
        (int)tt.depths[tt_slot] >= depth - 1 &&
        (tt.bounds[tt_slot] == AB_TT_EXACT ||
         tt.bounds[tt_slot] == AB_TT_LOWER)) {
        float threshold = ab_tt_load_value(
            tt.values[tt_slot], ply, config.persistent_tt) -
            config.singular_margin;
        bool singular = true;
        ABSearchConfig verify_config = config;
        verify_config.singular_extensions = false;
        int verify_depth = max(0, depth / 2 - 1);
        for (int i = 0; i < n; ++i) {
            if (ab_pack_move(moves[i]) == tt_move) continue;
            ABUndo verify_undo;
            uint64_t verify_hash = ab_make_move(
                state, moves[i], hash, verify_undo);
            float alternative = -ab_negamax(
                state, verify_hash, verify_depth, -AB_INF, -threshold,
                ply + 1, params, weights, node_budget, stats, tt,
                verify_config, killers, history, 0, ab_pack_move(moves[i]));
            ab_unmake_move(state, verify_undo);
            if (stats->aborted || alternative >= threshold) {
                singular = false;
                break;
            }
        }
        if (singular) singular_move = tt_move;
    }

    float best = -AB_INF;
    GPUMove best_move = moves[0];
    for (int rank = 0; rank < n; ++rank) {
        ab_order_native_move_at_rank(
            state, moves, n, rank, ply, killers, history,
            has_tt_move ? tt_move : AB_NO_MOVE, config, previous_move);
        GPUMove move = moves[rank];
        int move_depth = next_depth +
            (ab_pack_move(move) == singular_move ? 1 : 0);
        ABUndo undo;
        uint64_t child_hash = ab_make_move(state, move, hash, undo);

        float value;
        if (rank == 0) {
            value = -ab_negamax(
                state, child_hash, move_depth, -beta, -alpha, ply + 1,
                params, weights, node_budget, stats, tt, config,
                killers, history, next_extension_chain, ab_pack_move(move));
        } else {
            int reduction = (
                depth >= config.lmr_min_depth &&
                rank >= config.lmr_min_move
            ) ? min(config.lmr_reduction, max(0, depth - 1)) : 0;
            if (reduction) ++stats->lmr_reductions;
            value = -ab_negamax(
                state, child_hash, move_depth - reduction,
                -alpha - AB_PVS_EPSILON, -alpha, ply + 1, params, weights,
                node_budget, stats, tt, config, killers, history,
                next_extension_chain);
            if (!stats->aborted && reduction && value > alpha) {
                ++stats->pvs_researches;
                value = -ab_negamax(
                    state, child_hash, move_depth, -alpha - AB_PVS_EPSILON,
                    -alpha, ply + 1, params, weights, node_budget, stats, tt,
                    config, killers, history, next_extension_chain, ab_pack_move(move));
            }
            if (!stats->aborted && value > alpha && value < beta) {
                ++stats->pvs_researches;
                value = -ab_negamax(
                    state, child_hash, move_depth, -beta, -alpha, ply + 1,
                    params, weights, node_budget, stats, tt, config,
                    killers, history, next_extension_chain, ab_pack_move(move));
            }
        }
        ab_unmake_move(state, undo);
        if (stats->aborted) return 0.0f;
        if (value > best) {
            best = value;
            best_move = move;
        }
        alpha = max(alpha, best);
        if (alpha >= beta) {
            ++stats->cutoffs;
            int history_index = ab_move_history_index(move);
            history[history_index] = min(
                1000000, history[history_index] + depth * depth);
            if (previous_move != AB_NO_MOVE && config.countermove_ordering) {
                int previous_index = (int)(
                    previous_move & (AB_HISTORY_SIZE - 1));
                history[AB_HISTORY_SIZE + previous_index] = (int)ab_pack_move(move);
            }
            if (previous_move != AB_NO_MOVE && config.continuation_history) {
                int continuation_index = (int)(
                    (previous_move * 33U + (uint32_t)history_index) &
                    (AB_CONTINUATION_SIZE - 1));
                int slot = AB_HISTORY_SIZE + AB_COUNTERMOVE_SIZE +
                    continuation_index;
                history[slot] = min(1000000, history[slot] + depth * depth);
            }
            ABPackedMove packed = ab_pack_move(move);
            if (ply < AB_MAX_PV && packed != killers[ply * 2]) {
                killers[ply * 2 + 1] = killers[ply * 2];
                killers[ply * 2] = packed;
            }
            break;
        }
    }

    uint8_t bound = AB_TT_EXACT;
    if (best <= alpha_start) bound = AB_TT_UPPER;
    else if (best >= beta_start) bound = AB_TT_LOWER;
    ab_tt_store(
        tt, key, depth, best, bound, best_move, ply, config.persistent_tt);
    return best;
}

struct ABExplicitFrame {
    uint64_t hash;
    float alpha;
    float beta;
    float alpha_start;
    float beta_start;
    float best;
    int depth;
    int ply;
    int move_count;
    int rank;
    int reduction;
    uint8_t phase;
    uint8_t entered;
    uint8_t has_tt_move;
    uint8_t padding;
    ABPackedMove best_move;
    ABPackedMove tt_move;
    ABPackedMove current_move;
    ABUndo undo;
};

__device__ __forceinline__ void ab_init_explicit_frame(
    ABExplicitFrame& frame, uint64_t hash, int depth,
    float alpha, float beta, int ply
) {
    frame.hash = hash;
    frame.alpha = alpha;
    frame.beta = beta;
    frame.depth = depth;
    frame.ply = ply;
    frame.entered = 0;
    frame.phase = 0;
    frame.has_tt_move = 0;
    frame.tt_move = AB_NO_MOVE;
    frame.current_move = AB_NO_MOVE;
}

__device__ inline void ab_unwind_explicit_state(
    HiveState& state, ABExplicitFrame* frames, int top
) {
    for (int parent = top - 1; parent >= 0; --parent) {
        ab_unmake_move(state, frames[parent].undo);
    }
}

/**
 * Iterative equivalent of ab_negamax.
 *
 * Frames and per-ply move lists live in global per-game workspaces. This
 * keeps traversal serial for each game while allowing a warp to advance
 * independent games through the same resumable state machine.
 */
__device__ float ab_negamax_explicit(
    HiveState& state, uint64_t hash, int depth, float alpha, float beta,
    int ply, const float* params, const FNNWeights& weights,
    int node_budget, ABStats* stats, const ABTT& tt,
    const ABSearchConfig& config, ABPackedMove* killers, int* history,
    ABExplicitFrame* frames, ABPackedMove* move_stack
) {
    int top = 0;
    ab_init_explicit_frame(frames[0], hash, depth, alpha, beta, ply);
    bool returning = false;
    float returned = 0.0f;

    while (top >= 0) {
        ABExplicitFrame& frame = frames[top];
        ABPackedMove* moves = move_stack + top * MAX_LEGAL_MOVES;

        if (returning) {
            ab_unmake_move(state, frame.undo);
            float value = -returned;
            returning = false;

            if (stats->aborted) {
                ab_unwind_explicit_state(state, frames, top);
                return 0.0f;
            }

            if (frame.rank > 0 && frame.phase == 0 &&
                frame.reduction > 0 && value > frame.alpha) {
                ++stats->pvs_researches;
                frame.phase = 1;
                GPUMove move = ab_unpack_move(frame.current_move);
                uint64_t child_hash = ab_make_move(
                    state, move, frame.hash, frame.undo);
                if (top + 1 >= AB_MAX_PV) {
                    value = -ab_quiescence(
                        state, -frame.alpha - AB_PVS_EPSILON, -frame.alpha,
                        frame.ply + 1, params, weights, node_budget, stats,
                        config);
                    ab_unmake_move(state, frame.undo);
                } else {
                    ++top;
                    ab_init_explicit_frame(
                        frames[top], child_hash, frame.depth - 1,
                        -frame.alpha - AB_PVS_EPSILON, -frame.alpha,
                        frame.ply + 1);
                    continue;
                }
            }

            if (frame.rank > 0 && frame.phase < 2 &&
                value > frame.alpha && value < frame.beta) {
                ++stats->pvs_researches;
                frame.phase = 2;
                GPUMove move = ab_unpack_move(frame.current_move);
                uint64_t child_hash = ab_make_move(
                    state, move, frame.hash, frame.undo);
                if (top + 1 >= AB_MAX_PV) {
                    value = -ab_quiescence(
                        state, -frame.beta, -frame.alpha, frame.ply + 1,
                        params, weights, node_budget, stats, config);
                    ab_unmake_move(state, frame.undo);
                } else {
                    ++top;
                    ab_init_explicit_frame(
                        frames[top], child_hash, frame.depth - 1,
                        -frame.beta, -frame.alpha, frame.ply + 1);
                    continue;
                }
            }

            if (value > frame.best) {
                frame.best = value;
                frame.best_move = frame.current_move;
            }
            frame.alpha = max(frame.alpha, frame.best);
            if (frame.alpha >= frame.beta) {
                ++stats->cutoffs;
                GPUMove move = ab_unpack_move(frame.current_move);
                int history_index = ab_move_history_index(move);
                history[history_index] = min(
                    1000000, history[history_index] + frame.depth * frame.depth);
                ABPackedMove previous = top > 0 ?
                    frames[top - 1].current_move : AB_NO_MOVE;
                if (previous != AB_NO_MOVE && config.countermove_ordering) {
                    int previous_index = (int)(
                        previous & (AB_HISTORY_SIZE - 1));
                    history[AB_HISTORY_SIZE + previous_index] =
                        (int)frame.current_move;
                }
                if (previous != AB_NO_MOVE && config.continuation_history) {
                    int continuation_index = (int)(
                        (previous * 33U + (uint32_t)history_index) &
                        (AB_CONTINUATION_SIZE - 1));
                    int slot = AB_HISTORY_SIZE + AB_COUNTERMOVE_SIZE +
                        continuation_index;
                    history[slot] = min(
                        1000000, history[slot] + frame.depth * frame.depth);
                }
                if (frame.ply < AB_MAX_PV &&
                    frame.current_move != killers[frame.ply * 2]) {
                    killers[frame.ply * 2 + 1] = killers[frame.ply * 2];
                    killers[frame.ply * 2] = frame.current_move;
                }
                frame.rank = frame.move_count;
            } else {
                ++frame.rank;
            }
            continue;
        }

        if (!frame.entered) {
            frame.entered = 1;
            float immediate = 0.0f;
            bool complete = false;
            if (!ab_take_node(stats, node_budget, config)) {
                immediate = 0.0f;
                complete = true;
            } else if (state.result != IN_PROGRESS) {
                immediate = ab_terminal_value(state, frame.ply);
                complete = true;
            } else if (frame.depth <= 0) {
                __syncwarp(__activemask());
                immediate = ab_quiescence(
                    state, frame.alpha, frame.beta, frame.ply, params,
                    weights, node_budget, stats, config);
                complete = true;
            } else {
                frame.alpha_start = frame.alpha;
                frame.beta_start = frame.beta;
                int tt_slot = (int)(frame.hash & (uint64_t)tt.mask);
                if (tt.generations[tt_slot] == tt.generation &&
                    tt.keys[tt_slot] == frame.hash) {
                    frame.has_tt_move = 1;
                    frame.tt_move = tt.moves[tt_slot];
                    if ((int)tt.depths[tt_slot] >= frame.depth) {
                        ++stats->tt_hits;
                        float tt_value = ab_tt_load_value(
                            tt.values[tt_slot], frame.ply,
                            config.persistent_tt);
                        uint8_t tt_bound = tt.bounds[tt_slot];
                        if (tt_bound == AB_TT_EXACT) {
                            immediate = tt_value;
                            complete = true;
                        } else {
                            if (tt_bound == AB_TT_LOWER) {
                                frame.alpha = max(frame.alpha, tt_value);
                            }
                            if (tt_bound == AB_TT_UPPER) {
                                frame.beta = min(frame.beta, tt_value);
                            }
                            if (frame.alpha >= frame.beta) {
                                immediate = tt_value;
                                complete = true;
                            }
                        }
                    }
                }
                if (!complete) {
                    frame.move_count = ab_generate_packed_moves(state, moves);
                    if (frame.move_count <= 0) {
                        __syncwarp(__activemask());
                        immediate = ab_evaluate(state, params, weights);
                        complete = true;
                    } else {
                        frame.rank = 0;
                        frame.best = -AB_INF;
                        frame.best_move = moves[0];
                    }
                }
            }

            if (complete) {
                if (top == 0) return immediate;
                --top;
                returned = immediate;
                returning = true;
            }
            continue;
        }

        if (frame.rank >= frame.move_count) {
            uint8_t bound = AB_TT_EXACT;
            if (frame.best <= frame.alpha_start) bound = AB_TT_UPPER;
            else if (frame.best >= frame.beta_start) bound = AB_TT_LOWER;
            ab_tt_store(
                tt, frame.hash, frame.depth, frame.best, bound,
                ab_unpack_move(frame.best_move), frame.ply,
                config.persistent_tt);
            float result = frame.best;
            if (top == 0) return result;
            --top;
            returned = result;
            returning = true;
            continue;
        }

        ab_order_move_at_rank(
            state, moves, frame.move_count, frame.rank, frame.ply, killers,
            history, frame.has_tt_move ? frame.tt_move : AB_NO_MOVE, config,
            top > 0 ? frames[top - 1].current_move : AB_NO_MOVE);
        frame.current_move = moves[frame.rank];
        frame.phase = 0;
        frame.reduction = (
            frame.rank > 0 && frame.depth >= config.lmr_min_depth &&
            frame.rank >= config.lmr_min_move
        ) ? min(config.lmr_reduction, max(0, frame.depth - 1)) : 0;
        if (frame.reduction) ++stats->lmr_reductions;

        GPUMove move = ab_unpack_move(frame.current_move);
        uint64_t child_hash = ab_make_move(
            state, move, frame.hash, frame.undo);
        int child_depth = frame.depth - 1 - frame.reduction;
        float child_alpha = frame.rank == 0 ? -frame.beta :
            -frame.alpha - AB_PVS_EPSILON;
        float child_beta = frame.rank == 0 ? -frame.alpha : -frame.alpha;
        if (top + 1 >= AB_MAX_PV) {
            returned = ab_quiescence(
                state, child_alpha, child_beta, frame.ply + 1, params,
                weights, node_budget, stats, config);
            returning = true;
        } else {
            ++top;
            ab_init_explicit_frame(
                frames[top], child_hash, child_depth, child_alpha,
                child_beta, frame.ply + 1);
        }
    }
    return 0.0f;
}

template <bool EXPLICIT_STACK>
__device__ inline float ab_run_negamax(
    HiveState& state, uint64_t hash, int depth, float alpha, float beta,
    int ply, const float* params, const FNNWeights& weights,
    int node_budget, ABStats* stats, const ABTT& tt,
    const ABSearchConfig& config, ABPackedMove* killers, int* history,
    ABExplicitFrame* frames, ABPackedMove* move_stack
) {
    if constexpr (EXPLICIT_STACK) {
        return ab_negamax_explicit(
            state, hash, depth, alpha, beta, ply, params, weights,
            node_budget, stats, tt, config, killers, history, frames,
            move_stack);
    }
    return ab_negamax(
        state, hash, depth, alpha, beta, ply, params, weights, node_budget,
        stats, tt, config, killers, history);
}

__device__ inline void ab_write_pv(
    const HiveState& root, uint64_t root_hash,
    const GPUMove& root_move, int depth,
    const ABTT& tt, GPUMove* out, int* out_length
) {
    int length = 0;
    HiveState state = root;
    uint64_t hash = root_hash;
    GPUMove move = root_move;
    for (int ply = 0; ply < depth && ply < AB_MAX_PV; ++ply) {
        GPUMove legal[MAX_LEGAL_MOVES];
        int n = generate_legal_moves(state, legal);
        bool found = false;
        for (int i = 0; i < n; ++i) {
            if (ab_move_equal(legal[i], move)) {
                move = legal[i];
                found = true;
                break;
            }
        }
        if (!found) break;
        out[length++] = move;
        ABUndo undo;
        hash = ab_make_move(state, move, hash, undo);
        if (state.result != IN_PROGRESS) break;

        int slot = (int)(hash & (uint64_t)tt.mask);
        if (tt.generations[slot] != tt.generation ||
            tt.keys[slot] != hash) break;
        move = ab_unpack_move(tt.moves[slot]);
    }
    *out_length = length;
}

template <bool EXPLICIT_STACK>
__global__ void fnn_alphabeta_kernel(
    const HiveState* states, const float* params, int hidden_dim, int embed_dim,
    int action_hidden, const float* search_config_values, int node_budget,
    int max_depth, int root_exact_count, GPUMove* out_moves,
    float* out_values, int* out_stats, float* out_raw_values,
    GPUMove* out_root_moves, int* out_num_legal, float* out_root_scores,
    uint8_t* out_root_bounds, int* out_selected_indices,
    uint64_t* tt_keys, float* tt_values, int16_t* tt_depths,
    uint8_t* tt_bounds, ABPackedMove* tt_moves,
    int* tt_generations, int tt_generation,
    float* order_workspace, float* iteration_value_workspace,
    uint8_t* iteration_bound_workspace, ABPackedMove* killer_workspace,
    int* history_workspace, ABExplicitFrame* frame_workspace,
    ABPackedMove* move_stack_workspace, int tt_entries, int batch_size
) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= batch_size) return;
    FNNWeights weights = make_fnn_weights(hidden_dim, embed_dim, action_hidden);
    ABSearchConfig config = ab_make_search_config(search_config_values);
    ABTT tt = {
        tt_keys + (int64_t)game * tt_entries,
        tt_values + (int64_t)game * tt_entries,
        tt_depths + (int64_t)game * tt_entries,
        tt_bounds + (int64_t)game * tt_entries,
        tt_moves + (int64_t)game * tt_entries,
        tt_generations + (int64_t)game * tt_entries,
        tt_generation,
        tt_entries - 1,
    };
    HiveState root = states[game];
    uint64_t root_hash = ab_hash_state(root);
    int root_offset = game * MAX_LEGAL_MOVES;
    GPUMove* root_moves = out_root_moves + root_offset;
    MovegenStateCache root_cache;
    int n = generate_legal_moves_with_cache(root, root_moves, root_cache);
    out_num_legal[game] = n;
    out_raw_values[game] = ab_evaluate_with_moves_and_ap(
        root, root_moves, n, root_cache.ap_mask, params, weights);
    if (n <= 0) return;
    Color root_player = current_player(root);
    int proof_nodes = 0;
    if (config.proof_search &&
        queen_surround_count_for_color_device(root,
            root_player == WHITE ? BLACK : WHITE) >=
            config.proof_trigger_surround) {
        int proof_limit = max(1, (int)(node_budget * config.proof_budget_fraction));
        for (int i = 0; i < n && proof_nodes < proof_limit; ++i) {
            ABUndo undo;
            ab_make_move_unhashed(root, root_moves[i], undo);
            bool proven = ab_prove_forced_win(
                root, root_player, config.proof_max_plies - 1,
                &proof_nodes, proof_limit);
            ab_unmake_move(root, undo);
            if (proven) {
                out_moves[game] = root_moves[i];
                out_values[game] = AB_MATE - 0.01f;
                int* game_stats = out_stats + game * 9;
                game_stats[0] = config.proof_max_plies;
                game_stats[1] = proof_nodes;
                game_stats[7] = proof_nodes;
                game_stats[8] = 1;
                out_root_scores[root_offset + i] = AB_MATE - 0.01f;
                out_root_bounds[root_offset + i] = AB_TT_EXACT;
                out_selected_indices[game] = i;
                return;
            }
        }
    }
    float branch_scale = 1.0f + config.branching_allocation *
        ((float)n / 24.0f - 1.0f);
    branch_scale = min(1.5f, max(0.5f, branch_scale));
    int search_node_budget = max(
        1, (int)((node_budget - proof_nodes) * branch_scale));
    for (int i = 0; i < n; ++i) {
        ABUndo undo;
        ab_make_move(root, root_moves[i], root_hash, undo);
        bool immediate_win = ab_color_won(root, root_player);
        ab_unmake_move(root, undo);
        if (immediate_win) {
            out_moves[game] = root_moves[i];
            out_values[game] = AB_MATE - 0.01f;
            int* game_stats = out_stats + game * 9;
            game_stats[0] = 1;
            game_stats[1] = i + 1;
            game_stats[6] = 0;
            game_stats[7] = 1;
            game_stats[8] = 1;
            out_root_scores[root_offset + i] = AB_MATE - 0.01f;
            out_root_bounds[root_offset + i] = AB_TT_EXACT;
            out_selected_indices[game] = i;
            return;
        }
    }

    float* order_scores = order_workspace + root_offset;
    float* iteration_values = iteration_value_workspace + root_offset;
    uint8_t* iteration_bounds = iteration_bound_workspace + root_offset;
    ABPackedMove* killers = killer_workspace + game * AB_MAX_PV * 2;
    int* history = history_workspace + game * AB_HISTORY_SIZE;
    ABExplicitFrame* frames = frame_workspace + game * AB_MAX_PV;
    ABPackedMove* move_stack = move_stack_workspace +
        (int64_t)game * AB_MAX_PV * MAX_LEGAL_MOVES;
    // Seed root ordering cheaply; completed depth-one scores replace it.
    for (int i = 0; i < n; ++i) {
        order_scores[i] = ab_search_order_score(
            root, root_moves[i], 0, killers, history, config, AB_NO_MOVE);
    }
    GPUMove best_move = root_moves[0];
    float best_value = out_raw_values[game];
    int best_index = 0;
    int completed = 0;
    ABStats stats = {};
    stats.nodes = proof_nodes;
    stats.forced_win_probes = proof_nodes;
    root_exact_count = max(1, min(root_exact_count, n));

    for (int depth = 1; depth <= max_depth; ++depth) {
        for (int i = 0; i < n; ++i) {
            // Once a depth completes, its root values are substantially better
            // MultiPV ordering signals than the initial heuristics.
            if (completed > 0) {
                order_scores[i] = out_root_scores[root_offset + i];
                if (ab_move_equal(root_moves[i], best_move)) {
                    order_scores[i] = 1e30f;
                }
            }
            iteration_values[i] = 0.0f;
            iteration_bounds[i] = AB_TT_EMPTY;
        }
        float alpha = -AB_INF;
        float value = -AB_INF;
        GPUMove move = best_move;
        int move_index = best_index;
        for (int rank = 0; rank < n; ++rank) {
            int index = -1;
            float ordering_score = -1e30f;
            for (int i = 0; i < n; ++i) {
                if (order_scores[i] > ordering_score) {
                    ordering_score = order_scores[i];
                    index = i;
                }
            }
            order_scores[index] = -1e30f;
            ABUndo undo;
            uint64_t child_hash = ab_make_move(
                root, root_moves[index], root_hash, undo);

            float child_value;
            uint8_t root_bound = AB_TT_EXACT;
            if (rank < root_exact_count) {
                // Full-window searches produce exact scores safe for opening
                // diversity. Remaining root moves retain cheaper PVS scouts.
                bool use_aspiration =
                    rank == 0 && root_exact_count == 1 && completed > 0 &&
                    config.aspiration_window > 0.0f;
                if (use_aspiration) {
                    float low = best_value - config.aspiration_window;
                    float high = best_value + config.aspiration_window;
                    child_value = -ab_run_negamax<EXPLICIT_STACK>(
                        root, child_hash, depth - 1, -high, -low, 1,
                        params, weights, search_node_budget, &stats, tt,
                        config, killers, history, frames, move_stack);
                    if (!stats.aborted &&
                        (child_value <= low || child_value >= high)) {
                        ++stats.pvs_researches;
                        child_value = -ab_run_negamax<EXPLICIT_STACK>(
                            root, child_hash, depth - 1, -AB_INF, AB_INF, 1,
                            params, weights, search_node_budget, &stats, tt,
                            config, killers, history, frames, move_stack);
                    }
                } else {
                    child_value = -ab_run_negamax<EXPLICIT_STACK>(
                        root, child_hash, depth - 1, -AB_INF, AB_INF, 1,
                        params, weights, search_node_budget, &stats, tt,
                        config, killers, history, frames, move_stack);
                }
            } else {
                child_value = -ab_run_negamax<EXPLICIT_STACK>(
                    root, child_hash, depth - 1,
                    -alpha - AB_PVS_EPSILON, -alpha, 1, params, weights,
                    search_node_budget, &stats, tt, config, killers, history,
                    frames, move_stack);
                if (!stats.aborted && child_value > alpha) {
                    ++stats.pvs_researches;
                    child_value = -ab_run_negamax<EXPLICIT_STACK>(
                        root, child_hash, depth - 1, -AB_INF, -alpha, 1,
                        params, weights, search_node_budget, &stats, tt,
                        config, killers, history, frames, move_stack);
                    root_bound = AB_TT_EXACT;
                } else {
                    // A failed root scout proves only that this move cannot
                    // exceed the current exact best score.
                    root_bound = AB_TT_UPPER;
                }
            }
            ab_unmake_move(root, undo);
            if (stats.aborted) break;
            iteration_values[index] = child_value;
            iteration_bounds[index] = root_bound;
            if (child_value > value) {
                value = child_value;
                move = root_moves[index];
                move_index = index;
            }
            alpha = max(alpha, value);
        }
        if (stats.aborted) break;
        best_move = move;
        best_value = value;
        best_index = move_index;
        completed = depth;
        for (int i = 0; i < n; ++i) {
            out_root_scores[root_offset + i] = iteration_values[i];
            out_root_bounds[root_offset + i] = iteration_bounds[i];
        }
        out_selected_indices[game] = best_index;
        if (completed >= config.early_stop_min_depth &&
            fabsf(value) >= config.early_stop_score) {
            break;
        }
    }

    out_moves[game] = best_move;
    out_values[game] = best_value;
    int* game_stats = out_stats + game * 9;
    game_stats[0] = completed;
    game_stats[1] = stats.nodes;
    game_stats[2] = stats.cutoffs;
    game_stats[3] = stats.tt_hits;
    game_stats[4] = stats.pvs_researches;
    game_stats[5] = stats.lmr_reductions;
    game_stats[6] = stats.qnodes;
    game_stats[7] = stats.forced_win_probes;
    game_stats[8] = stats.tactical_moves;
}

__global__ void fnn_alphabeta_pv_kernel(
    const HiveState* states, const GPUMove* best_moves, const int* stats,
    const uint64_t* tt_keys, const float* tt_values,
    const int16_t* tt_depths, const uint8_t* tt_bounds,
    const ABPackedMove* tt_moves, int tt_entries, GPUMove* out_pv_moves,
    const int* tt_generations, int tt_generation,
    int* out_pv_lengths, int batch_size
) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= batch_size) return;
    ABTT tt = {
        const_cast<uint64_t*>(tt_keys + (int64_t)game * tt_entries),
        const_cast<float*>(tt_values + (int64_t)game * tt_entries),
        const_cast<int16_t*>(tt_depths + (int64_t)game * tt_entries),
        const_cast<uint8_t*>(tt_bounds + (int64_t)game * tt_entries),
        const_cast<ABPackedMove*>(tt_moves + (int64_t)game * tt_entries),
        const_cast<int*>(tt_generations + (int64_t)game * tt_entries),
        tt_generation,
        tt_entries - 1,
    };
    int depth = max(0, min(stats[game * 9], AB_MAX_PV));
    if (depth == 0) return;
    HiveState root = states[game];
    ab_write_pv(
        root, ab_hash_state(root), best_moves[game], depth, tt,
        out_pv_moves + game * AB_MAX_PV, out_pv_lengths + game);
}

#endif
} // namespace hive_gpu
