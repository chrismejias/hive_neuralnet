// GPU-resident iterative-deepening alpha-beta search for the FNN.
#pragma once

#include "fnn_selfplay.cuh"

namespace hive_gpu {

#ifdef __CUDACC__

constexpr float AB_INF = 1000.0f;
constexpr float AB_MATE = 10.0f;
constexpr float AB_PVS_EPSILON = 1e-4f;
constexpr int AB_MAX_PV = 64;

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
    GPUMove* moves;
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
    float policy_ordering_weight;
    float tactical_ordering_weight;
    float branching_allocation;
    float early_stop_score;
    int early_stop_min_depth;
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
    config.quiescence_plies = max(0, (int)values[4]);
    config.quiescence_budget_fraction =
        min(0.95f, max(0.0f, values[5]));
    config.force_win_probes = values[6] >= 0.5f;
    config.tactical_mask = (int)values[7];
    config.policy_ordering_weight = max(0.0f, values[8]);
    config.tactical_ordering_weight = max(0.0f, values[9]);
    config.branching_allocation =
        min(0.75f, max(-0.75f, values[10]));
    config.early_stop_score = min(9.99f, max(1.0f, values[11]));
    config.early_stop_min_depth = max(1, (int)values[12]);
    return config;
}

__device__ inline bool ab_move_equal(const GPUMove& a, const GPUMove& b) {
    return a.type == b.type && a.piece_type == b.piece_type &&
           a.from_cell == b.from_cell && a.to_cell == b.to_cell;
}

__device__ inline uint64_t ab_hash_state(const HiveState& state) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&state);
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < (int)sizeof(HiveState); ++i) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ULL;
    }
    // Zero denotes an unused TT slot.
    return hash == 0 ? 1ULL : hash;
}

__device__ inline float ab_terminal_value(const HiveState& state, int ply) {
    if (state.result == DRAW || state.result == IN_PROGRESS) return 0.0f;
    Color side = current_player(state);
    bool side_won = (state.result == WHITE_WINS && side == WHITE) ||
                    (state.result == BLACK_WINS && side == BLACK);
    return (side_won ? 1.0f : -1.0f) * (AB_MATE - min(ply, 100) * 0.01f);
}

__device__ inline float ab_evaluate(
    const HiveState& state, const float* params, const FNNWeights& weights
) {
    GPUMove moves[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(state, moves);
    float features[FNN_FEAT_DIM];
    float embed[FNN_MAX_EMBED];
    extract_fnn_features_device(state, moves, n, features);
    fnn_encode(features, embed, params, weights);
    return fnn_value(embed, params, weights);
}

__device__ inline float ab_ordering_heuristic(
    const HiveState& state, const HiveState& child
) {
    Color mover = current_player(state);
    Color opponent = mover == WHITE ? BLACK : WHITE;
    bool mover_won = (child.result == WHITE_WINS && mover == WHITE) ||
                     (child.result == BLACK_WINS && mover == BLACK);
    if (mover_won) return 100.0f;
    int opp_gain =
        queen_surround_count_for_color_device(child, opponent) -
        queen_surround_count_for_color_device(state, opponent);
    int own_relief =
        queen_surround_count_for_color_device(state, mover) -
        queen_surround_count_for_color_device(child, mover);
    return (float)(opp_gain + own_relief);
}

__device__ inline void ab_policy_scores(
    const HiveState& state, const GPUMove* moves, int n,
    const float* params, const FNNWeights& weights,
    const ABSearchConfig& config, float* scores
) {
    float root_features[FNN_FEAT_DIM], root_embed[FNN_MAX_EMBED];
    extract_fnn_features_device(state, moves, n, root_features);
    fnn_encode(root_features, root_embed, params, weights);
    for (int i = 0; i < n; ++i) {
        HiveState child = state;
        apply_move(child, moves[i]);
        GPUMove child_moves[MAX_LEGAL_MOVES];
        int child_n = generate_legal_moves(child, child_moves);
        float child_features[FNN_FEAT_DIM], child_embed[FNN_MAX_EMBED];
        extract_fnn_features_device(child, child_moves, child_n, child_features);
        fnn_encode(child_features, child_embed, params, weights);
        scores[i] =
            config.policy_ordering_weight * fnn_score_action(
                root_embed, child_embed, root_features, child_features,
                params, weights)
            + config.tactical_ordering_weight *
                ab_ordering_heuristic(state, child);
    }
}

__device__ inline void ab_tt_store(
    const ABTT& tt, uint64_t key, int depth, float value, uint8_t bound,
    const GPUMove& best_move
) {
    int slot = (int)(key & (uint64_t)tt.mask);
    if (tt.keys[slot] == key || depth >= (int)tt.depths[slot]) {
        // Publish the key last so a partially-written entry is never accepted.
        tt.values[slot] = value;
        tt.depths[slot] = (int16_t)depth;
        tt.bounds[slot] = bound;
        tt.moves[slot] = best_move;
        __threadfence();
        tt.keys[slot] = key;
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

/**
 * Prove a win for the side to move in at most three plies.
 *
 * The surround gates make the exhaustive all-replies probe affordable:
 * surround 5 can be won in one move, while surround 4/5 can potentially be
 * won by move-reply-move. Unlike the older diagnostic probe, every legal
 * defensive reply is checked.
 */
__device__ inline int ab_forced_win_distance(
    const HiveState& state, int node_budget, ABStats* stats,
    const ABSearchConfig& config
) {
    if (!config.force_win_probes) return 0;
    Color player = current_player(state);
    Color opponent = player == WHITE ? BLACK : WHITE;
    int surround = queen_surround_count_for_color_device(state, opponent);
    if (surround < 4) return 0;
    ++stats->forced_win_probes;

    GPUMove moves[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(state, moves);
    for (int i = 0; i < n; ++i) {
        HiveState child = state;
        apply_move(child, moves[i]);
        if (!ab_take_node(stats, node_budget, config, true)) return 0;
        if (ab_color_won(child, player)) return 1;
        if (child.result != IN_PROGRESS) continue;

        // A three-ply win needs the target queen to be one cell from defeat
        // after the attacker's first move.
        if (queen_surround_count_for_color_device(child, opponent) != 5) continue;

        GPUMove replies[MAX_LEGAL_MOVES];
        int nr = generate_legal_moves(child, replies);
        bool wins_after_every_reply = nr > 0;
        for (int r = 0; r < nr; ++r) {
            HiveState grandchild = child;
            apply_move(grandchild, replies[r]);
            if (!ab_take_node(stats, node_budget, config, true)) return 0;
            if (ab_color_won(grandchild, opponent) ||
                grandchild.result == DRAW) {
                wins_after_every_reply = false;
                break;
            }
            if (ab_color_won(grandchild, player)) continue;
            if (grandchild.result != IN_PROGRESS ||
                !has_immediate_surround_win_for_current_player(grandchild)) {
                wins_after_every_reply = false;
                break;
            }
        }
        if (wins_after_every_reply) return 3;
    }
    return 0;
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

__device__ inline bool ab_immobilizes_power_piece(
    const HiveState& state, const GPUMove& move, const HiveState& child,
    const GPUMove* child_moves, int child_n
) {
    Color opponent = current_player(child);
    MovegenStateCache cache;
    init_movegen_state_cache(state, cache);
    const Bitboard& tops = opponent == WHITE ? state.white_top : state.black_top;
    for (int wi = 0; wi < BB_WORDS; ++wi) {
        uint64_t bits = tops.w[wi];
        while (bits) {
            int bit = __ffsll(bits) - 1;
            int cell = wi * 64 + bit;
            bits &= bits - 1;
            if (!ab_power_piece_mobile(state, cell, cache)) continue;

            // Relocating an enemy piece is not immobilization; its identity
            // has simply moved to a new cell.
            if (move.type == MOVE_MOVE && (int)move.from_cell == cell) continue;

            bool still_mobile = false;
            if (cell < NUM_CELLS && child.height[cell] > 0 &&
                top_piece_color_at(child, cell) == opponent) {
                for (int i = 0; i < child_n; ++i) {
                    if (child_moves[i].type == MOVE_MOVE &&
                        (int)child_moves[i].from_cell == cell) {
                        still_mobile = true;
                        break;
                    }
                }
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
    HiveState probe = child;
    probe.turn = (uint16_t)((probe.turn & ~1U) | (uint16_t)mover);
    probe.stunned_cell = 0xFFFF;
    return has_immediate_surround_win_for_current_player(probe);
}

__device__ inline bool ab_is_tactical_move(
    const HiveState& state, const GPUMove& move, const HiveState& child,
    const GPUMove* child_moves, int child_n, const ABSearchConfig& config
) {
    Color mover = current_player(state);
    Color opponent = mover == WHITE ? BLACK : WHITE;
    int own_before = queen_surround_count_for_color_device(state, mover);
    int opp_before = queen_surround_count_for_color_device(state, opponent);
    if ((config.tactical_mask & AB_TACTICAL_OPP_SURROUND) &&
        queen_surround_count_for_color_device(child, opponent) > opp_before) {
        return true;
    }
    if ((config.tactical_mask & AB_TACTICAL_OWN_RELIEF) &&
        queen_surround_count_for_color_device(child, mover) < own_before) {
        return true;
    }
    if ((config.tactical_mask & AB_TACTICAL_QUEEN_THREAT) &&
        ab_creates_queen_threat(child, mover)) {
        return true;
    }
    return (config.tactical_mask & AB_TACTICAL_IMMOBILIZE) &&
           ab_immobilizes_power_piece(
               state, move, child, child_moves, child_n);
}

__device__ float ab_quiescence(
    const HiveState& state, float alpha, float beta, int ply, int qplies,
    const float* params, const FNNWeights& weights, int node_budget,
    ABStats* stats, bool count_node, const ABSearchConfig& config
) {
    if (state.result != IN_PROGRESS) return ab_terminal_value(state, ply);
    if (count_node && !ab_take_node(stats, node_budget, config, true)) {
        return stats->aborted ? 0.0f : ab_evaluate(state, params, weights);
    }
    if (!count_node) ++stats->qnodes;

    int win_distance = ab_forced_win_distance(
        state, node_budget, stats, config);
    if (stats->aborted) return 0.0f;
    if (stats->q_exhausted) return ab_evaluate(state, params, weights);
    if (win_distance > 0) {
        return AB_MATE - min(ply + win_distance, 100) * 0.01f;
    }

    float best = ab_evaluate(state, params, weights);
    if (qplies <= 0 || best >= beta) return best;
    alpha = max(alpha, best);

    GPUMove moves[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(state, moves);
    if (n <= 0) return best;
    float scores[MAX_LEGAL_MOVES];
    ab_policy_scores(state, moves, n, params, weights, config, scores);

    for (int rank = 0; rank < n; ++rank) {
        int index = -1;
        float ordering_score = -1e30f;
        for (int i = 0; i < n; ++i) {
            if (scores[i] > ordering_score) {
                ordering_score = scores[i];
                index = i;
            }
        }
        scores[index] = -1e30f;
        HiveState child = state;
        apply_move(child, moves[index]);
        GPUMove child_moves[MAX_LEGAL_MOVES];
        int child_n = generate_legal_moves(child, child_moves);
        if (!ab_is_tactical_move(
                state, moves[index], child, child_moves, child_n, config)) {
            continue;
        }
        ++stats->tactical_moves;
        float value = -ab_quiescence(
            child, -beta, -alpha, ply + 1, qplies - 1,
            params, weights, node_budget, stats, true, config);
        if (stats->aborted) return 0.0f;
        best = max(best, value);
        alpha = max(alpha, best);
        if (alpha >= beta) {
            ++stats->cutoffs;
            break;
        }
    }
    return best;
}

__device__ float ab_negamax(
    const HiveState& state, int depth, float alpha, float beta, int ply,
    const float* params, const FNNWeights& weights, int node_budget,
    ABStats* stats, const ABTT& tt, const ABSearchConfig& config
) {
    if (!ab_take_node(stats, node_budget, config)) return 0.0f;
    if (state.result != IN_PROGRESS) return ab_terminal_value(state, ply);
    if (depth <= 0) {
        return ab_quiescence(
            state, alpha, beta, ply, config.quiescence_plies,
            params, weights, node_budget, stats, false, config);
    }

    const float alpha_start = alpha;
    const float beta_start = beta;
    const uint64_t key = ab_hash_state(state);
    const int tt_slot = (int)(key & (uint64_t)tt.mask);
    GPUMove tt_move = {};
    bool has_tt_move = false;
    if (tt.keys[tt_slot] == key) {
        has_tt_move = true;
        tt_move = tt.moves[tt_slot];
        if ((int)tt.depths[tt_slot] >= depth) {
            ++stats->tt_hits;
            float tt_value = tt.values[tt_slot];
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

    float scores[MAX_LEGAL_MOVES];
    ab_policy_scores(state, moves, n, params, weights, config, scores);
    if (has_tt_move) {
        for (int i = 0; i < n; ++i) {
            if (ab_move_equal(moves[i], tt_move)) {
                scores[i] = 1e30f;
                break;
            }
        }
    }

    float best = -AB_INF;
    GPUMove best_move = moves[0];
    for (int rank = 0; rank < n; ++rank) {
        int index = -1;
        float ordering_score = -1e30f;
        for (int i = 0; i < n; ++i) {
            if (scores[i] > ordering_score) {
                ordering_score = scores[i];
                index = i;
            }
        }
        scores[index] = -1e30f;
        HiveState child = state;
        apply_move(child, moves[index]);

        float value;
        if (rank == 0) {
            value = -ab_negamax(
                child, depth - 1, -beta, -alpha, ply + 1,
                params, weights, node_budget, stats, tt, config);
        } else {
            int reduction = (
                depth >= config.lmr_min_depth &&
                rank >= config.lmr_min_move
            ) ? min(config.lmr_reduction, max(0, depth - 1)) : 0;
            if (reduction) ++stats->lmr_reductions;
            value = -ab_negamax(
                child, depth - 1 - reduction, -alpha - AB_PVS_EPSILON, -alpha,
                ply + 1, params, weights, node_budget, stats, tt, config);
            if (!stats->aborted && reduction && value > alpha) {
                ++stats->pvs_researches;
                value = -ab_negamax(
                    child, depth - 1, -alpha - AB_PVS_EPSILON, -alpha,
                    ply + 1, params, weights, node_budget, stats, tt, config);
            }
            if (!stats->aborted && value > alpha && value < beta) {
                ++stats->pvs_researches;
                value = -ab_negamax(
                    child, depth - 1, -beta, -alpha, ply + 1,
                    params, weights, node_budget, stats, tt, config);
            }
        }
        if (stats->aborted) return 0.0f;
        if (value > best) {
            best = value;
            best_move = moves[index];
        }
        alpha = max(alpha, best);
        if (alpha >= beta) {
            ++stats->cutoffs;
            break;
        }
    }

    uint8_t bound = AB_TT_EXACT;
    if (best <= alpha_start) bound = AB_TT_UPPER;
    else if (best >= beta_start) bound = AB_TT_LOWER;
    ab_tt_store(tt, key, depth, best, bound, best_move);
    return best;
}

__device__ inline void ab_write_pv(
    const HiveState& root, const GPUMove& root_move, int depth,
    const ABTT& tt, GPUMove* out, int* out_length
) {
    int length = 0;
    HiveState state = root;
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
        apply_move(state, move);
        if (state.result != IN_PROGRESS) break;

        uint64_t key = ab_hash_state(state);
        int slot = (int)(key & (uint64_t)tt.mask);
        if (tt.keys[slot] != key) break;
        move = tt.moves[slot];
    }
    *out_length = length;
}

__global__ void fnn_alphabeta_kernel(
    const HiveState* states, const float* params, int hidden_dim, int embed_dim,
    int action_hidden, const float* search_config_values, int node_budget,
    int max_depth, int root_exact_count, GPUMove* out_moves,
    float* out_values, int* out_stats, float* out_raw_values,
    GPUMove* out_root_moves, int* out_num_legal, float* out_root_scores,
    uint8_t* out_root_bounds, int* out_selected_indices,
    GPUMove* out_pv_moves, int* out_pv_lengths,
    uint64_t* tt_keys, float* tt_values, int16_t* tt_depths,
    uint8_t* tt_bounds, GPUMove* tt_moves, int tt_entries, int batch_size
) {
    int game = blockIdx.x;
    if (game >= batch_size || threadIdx.x != 0) return;
    FNNWeights weights = make_fnn_weights(hidden_dim, embed_dim, action_hidden);
    ABSearchConfig config = ab_make_search_config(search_config_values);
    ABTT tt = {
        tt_keys + (int64_t)game * tt_entries,
        tt_values + (int64_t)game * tt_entries,
        tt_depths + (int64_t)game * tt_entries,
        tt_bounds + (int64_t)game * tt_entries,
        tt_moves + (int64_t)game * tt_entries,
        tt_entries - 1,
    };
    HiveState root = states[game];
    GPUMove root_moves[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(root, root_moves);
    int root_offset = game * MAX_LEGAL_MOVES;
    int pv_offset = game * AB_MAX_PV;
    out_num_legal[game] = n;
    out_raw_values[game] = ab_evaluate(root, params, weights);
    for (int i = 0; i < n; ++i) {
        out_root_moves[root_offset + i] = root_moves[i];
    }
    if (n <= 0) return;
    float branch_scale = 1.0f + config.branching_allocation *
        ((float)n / 24.0f - 1.0f);
    branch_scale = min(1.5f, max(0.5f, branch_scale));
    int search_node_budget = max(1, (int)(node_budget * branch_scale));
    Color root_player = current_player(root);
    for (int i = 0; i < n; ++i) {
        HiveState child = root;
        apply_move(child, root_moves[i]);
        if (ab_color_won(child, root_player)) {
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
            out_pv_moves[pv_offset] = root_moves[i];
            out_pv_lengths[game] = 1;
            return;
        }
    }

    float policy_scores[MAX_LEGAL_MOVES];
    float previous_values[MAX_LEGAL_MOVES];
    ab_policy_scores(
        root, root_moves, n, params, weights, config, policy_scores);
    GPUMove best_move = root_moves[0];
    float best_value = out_raw_values[game];
    int best_index = 0;
    int completed = 0;
    ABStats stats = {};
    root_exact_count = max(1, min(root_exact_count, n));
    for (int i = 0; i < n; ++i) {
        previous_values[i] = policy_scores[i];
    }

    for (int depth = 1; depth <= max_depth; ++depth) {
        float scores[MAX_LEGAL_MOVES];
        float iteration_values[MAX_LEGAL_MOVES];
        uint8_t iteration_bounds[MAX_LEGAL_MOVES];
        for (int i = 0; i < n; ++i) {
            // Once a depth completes, its root values are substantially better
            // MultiPV ordering signals than the policy prior.
            scores[i] = completed > 0 ? previous_values[i] : policy_scores[i];
            if (ab_move_equal(root_moves[i], best_move)) scores[i] = 1e30f;
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
                if (scores[i] > ordering_score) {
                    ordering_score = scores[i];
                    index = i;
                }
            }
            scores[index] = -1e30f;
            HiveState child = root;
            apply_move(child, root_moves[index]);

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
                    child_value = -ab_negamax(
                        child, depth - 1, -high, -low, 1, params, weights,
                        search_node_budget, &stats, tt, config);
                    if (!stats.aborted &&
                        (child_value <= low || child_value >= high)) {
                        ++stats.pvs_researches;
                        child_value = -ab_negamax(
                            child, depth - 1, -AB_INF, AB_INF, 1,
                            params, weights, search_node_budget, &stats, tt,
                            config);
                    }
                } else {
                    child_value = -ab_negamax(
                        child, depth - 1, -AB_INF, AB_INF, 1,
                        params, weights, search_node_budget, &stats, tt,
                        config);
                }
            } else {
                child_value = -ab_negamax(
                    child, depth - 1, -alpha - AB_PVS_EPSILON, -alpha, 1,
                    params, weights, search_node_budget, &stats, tt, config);
                if (!stats.aborted && child_value > alpha) {
                    ++stats.pvs_researches;
                    child_value = -ab_negamax(
                        child, depth - 1, -AB_INF, -alpha, 1, params, weights,
                        search_node_budget, &stats, tt, config);
                    root_bound = AB_TT_EXACT;
                } else {
                    // A failed root scout proves only that this move cannot
                    // exceed the current exact best score.
                    root_bound = AB_TT_UPPER;
                }
            }
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
            previous_values[i] = iteration_values[i];
            out_root_scores[root_offset + i] = iteration_values[i];
            out_root_bounds[root_offset + i] = iteration_bounds[i];
        }
        out_selected_indices[game] = best_index;
        ab_write_pv(
            root, best_move, completed, tt,
            out_pv_moves + pv_offset, out_pv_lengths + game);
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

#endif
} // namespace hive_gpu
