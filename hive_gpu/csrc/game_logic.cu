/**
 * game_logic.cu — CUDA kernels for batch game operations.
 *
 * Provides batched versions of:
 *   - State initialization
 *   - Legal move generation
 *   - Move application
 *   - Game-over detection
 *
 * Also defines the __constant__ memory tables (NEIGHBORS, SLIDE_FLANKS).
 */

// Use ATen directly to avoid VS 2025 conflicts in torch/extension.h's dynamo headers
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/core/DeviceType.h>
#include <tuple>
#include "hex_grid.cuh"
#include "hive_state.cuh"
#include "move_gen.cuh"
#include "state_encoder.cuh"
#include "mobility.cuh"
#include "mcts_tree.cuh"
#include "fnn_features.cuh"
#include "fnn_selfplay.cuh"
#include "prs_v2_slot.cuh"

namespace hive_gpu {

void reset_movegen_profile() {
#ifdef __CUDACC__
    cudaMemset(MOVEGEN_PROFILE, 0, sizeof(unsigned long long) * MGP_MAX);
    cudaDeviceSynchronize();
    MOVEGEN_PROFILE_ENABLED = true;
#endif
}

at::Tensor get_movegen_profile() {
    auto opts = at::TensorOptions().dtype(c10::kLong).device(c10::kCPU);
    auto out = at::empty({MGP_MAX}, opts);
#ifdef __CUDACC__
    cudaDeviceSynchronize();
    auto* dst = static_cast<int64_t*>(out.data_ptr());
    for (int i = 0; i < MGP_MAX; ++i) {
        dst[i] = static_cast<int64_t>(MOVEGEN_PROFILE[i]);
    }
    MOVEGEN_PROFILE_ENABLED = false;
#endif
    return out;
}

// ── Kernel: initialize batch of states ─────────────────────────────

__global__ void init_states_kernel(HiveState* states, int batch_size, uint8_t expansion_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        init_state(states[idx], expansion_mask);
    }
}

// ── Kernel: generate legal moves for a batch of states ─────────────

__global__ void generate_legal_moves_kernel(
    const HiveState* states,
    GPUMove* moves_out,       // [MAX_LEGAL_MOVES * batch_size]
    int* num_legal_out,       // [batch_size]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        GPUMove* my_moves = moves_out + idx * MAX_LEGAL_MOVES;
        num_legal_out[idx] = generate_legal_moves(states[idx], my_moves);
    }
}

__global__ void generate_legal_moves_and_fnn_features_kernel(
    const HiveState* states,
    GPUMove* moves_out,       // [MAX_LEGAL_MOVES * batch_size]
    int* num_legal_out,       // [batch_size]
    float* features_out,      // [FNN_FEAT_DIM * batch_size]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        GPUMove* my_moves = moves_out + idx * MAX_LEGAL_MOVES;
        int n = generate_legal_moves(states[idx], my_moves);
        num_legal_out[idx] = n;
        extract_fnn_features_device(
            states[idx], my_moves, n, features_out + idx * FNN_FEAT_DIM);
    }
}

__global__ void queen_escape_flags_kernel(
    const HiveState* states,
    uint8_t* flags_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        flags_out[idx] = has_queen_escape_move(states[idx]) ? 1 : 0;
    }
}

__global__ void endgame_hit_mask_kernel(
    const HiveState* states,
    uint8_t* hit_out,
    int min_surround,
    int max_surround,
    uint8_t require_mixed_pair,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const HiveState& s = states[idx];
        if (s.result != IN_PROGRESS ||
            !is_queen_placed(s, WHITE) ||
            !is_queen_placed(s, BLACK)) {
            hit_out[idx] = 0;
            return;
        }
        int w = num_occupied_neighbors(s, s.queen_cell[WHITE]);
        int b = num_occupied_neighbors(s, s.queen_cell[BLACK]);
        bool in_range = (
            w >= min_surround && w <= max_surround &&
            b >= min_surround && b <= max_surround
        );
        if (!in_range) {
            hit_out[idx] = 0;
            return;
        }
        if (require_mixed_pair) {
            hit_out[idx] = (
                (w == min_surround && b == max_surround) ||
                (w == max_surround && b == min_surround)
            ) ? 1 : 0;
        } else {
            hit_out[idx] = 1;
        }
    }
}

__global__ void fnn_successor_features_kernel(
    const HiveState* states,
    const GPUMove* legal_moves,       // [B, MAX_LEGAL_MOVES]
    const int64_t* action_to_root,    // [N]
    const int64_t* move_indices,      // [N]
    float* features_out,              // [N, FNN_FEAT_DIM]
    int num_actions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_actions) {
        int root = (int)action_to_root[idx];
        int move_i = (int)move_indices[idx];

        unsigned long long t0 = 0, t1 = 0;
        if (MOVEGEN_PROFILE_ENABLED) t0 = clock64();
        HiveState child = states[root];
        if (MOVEGEN_PROFILE_ENABLED) {
            t1 = clock64();
            mgp_add(MGP_FNN_SUCC_CALLS, 1);
            mgp_add(MGP_FNN_SUCC_COPY_CYCLES, t1 - t0);
            t0 = t1;
        }

        const GPUMove& mv = legal_moves[(int64_t)root * MAX_LEGAL_MOVES + move_i];
        apply_move(child, mv);
        if (MOVEGEN_PROFILE_ENABLED) {
            t1 = clock64();
            mgp_add(MGP_FNN_SUCC_APPLY_CYCLES, t1 - t0);
            t0 = t1;
        }

        GPUMove child_moves[MAX_LEGAL_MOVES];
        int child_nlegal = generate_fnn_feature_moves(child, child_moves);
        if (MOVEGEN_PROFILE_ENABLED) {
            t1 = clock64();
            mgp_add(MGP_FNN_SUCC_LEGAL_CYCLES, t1 - t0);
            t0 = t1;
        }

        extract_fnn_features_device(
            child, child_moves, child_nlegal,
            features_out + (int64_t)idx * FNN_FEAT_DIM);
        if (MOVEGEN_PROFILE_ENABLED) {
            t1 = clock64();
            mgp_add(MGP_FNN_SUCC_FEATURE_CYCLES, t1 - t0);
        }
    }
}

// ── Kernel: apply moves to a batch of states ───────────────────────

__global__ void apply_moves_kernel(
    HiveState* states,
    const GPUMove* moves,     // [batch_size] — one move per game
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        apply_move(states[idx], moves[idx]);
    }
}

// ── Kernel: check game over for a batch of states ──────────────────

__global__ void check_results_kernel(
    const HiveState* states,
    int* results_out,         // [batch_size]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        results_out[idx] = (int)states[idx].result;
    }
}

__device__ __forceinline__ int queen_surround_count_for_color(
    const HiveState& s,
    Color c
) {
    if (!is_queen_placed(s, c)) return 0;
    return num_occupied_neighbors(s, s.queen_cell[c]);
}

__device__ inline bool state_has_immediate_win_for_current_player(
    const HiveState& s
) {
    return has_immediate_surround_win_for_current_player(s);
}

__global__ void root_tactical_probe_kernel(
    const HiveState* states,
    const GPUMove* legal_moves,       // [B, MAX_LEGAL_MOVES]
    const int* num_legal,             // [B]
    const float* priors,              // [B, MAX_LEGAL_MOVES]
    int64_t* winning_move_out,        // [B]
    uint8_t* allowed_slot_out,        // [B, MAX_LEGAL_MOVES]
    uint8_t* forced_random_out,       // [B]
    uint8_t enable_win_in_one,
    uint8_t enable_check_opponent_wins,
    uint8_t enable_win_in_two,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const HiveState& root = states[idx];
    const GPUMove* root_moves = legal_moves + (int64_t)idx * MAX_LEGAL_MOVES;
    const float* root_priors = priors + (int64_t)idx * MAX_LEGAL_MOVES;
    uint8_t* allowed = allowed_slot_out + (int64_t)idx * MAX_LEGAL_MOVES;
    int nlegal = num_legal[idx];

    winning_move_out[idx] = -1;
    forced_random_out[idx] = 0;
    for (int i = 0; i < MAX_LEGAL_MOVES; ++i) {
        allowed[i] = (i < nlegal) ? 1 : 0;
    }
    if (nlegal <= 0 || root.result != IN_PROGRESS) return;

    Color root_player = current_player(root);
    Color opp_color = (root_player == WHITE) ? BLACK : WHITE;
    int opp_surround = queen_surround_count_for_color(root, opp_color);
    int own_surround = queen_surround_count_for_color(root, root_player);

    // 1. Immediate win probe, gated to opponent surround == 5.
    if (enable_win_in_one && opp_surround == 5) {
        int best_move = -1;
        float best_prior = -1e30f;
        for (int i = 0; i < nlegal; ++i) {
            HiveState child = root;
            apply_move(child, root_moves[i]);
            if ((root_player == WHITE && child.result == WHITE_WINS) ||
                (root_player == BLACK && child.result == BLACK_WINS)) {
                if (best_move < 0 || root_priors[i] > best_prior) {
                    best_move = i;
                    best_prior = root_priors[i];
                }
            }
        }
        if (best_move >= 0) {
            winning_move_out[idx] = best_move;
            return;
        }
    }

    // 2. Opponent immediate-win screen, only when our own queen surround is 5.
    if (enable_check_opponent_wins && own_surround == 5) {
        bool any_safe = false;
        for (int i = 0; i < nlegal; ++i) {
            HiveState child = root;
            apply_move(child, root_moves[i]);
            bool opp_can_win = state_has_immediate_win_for_current_player(child);
            allowed[i] = opp_can_win ? 0 : 1;
            any_safe |= !opp_can_win;
        }
        if (!any_safe) {
            forced_random_out[idx] = 1;
            return;
        }
    }

    // 3. Forced win-in-2/3 tactical probe on opponent surround 4 or 5.
    if (!enable_win_in_two || !(opp_surround == 4 || opp_surround == 5)) return;

    for (int i = 0; i < nlegal; ++i) {
        if (!allowed[i]) continue;
        HiveState child = root;
        apply_move(child, root_moves[i]);
        int child_surround = queen_surround_count_for_color(child, opp_color);
        if (opp_surround == 4 && child_surround <= opp_surround) continue;
        if (opp_surround == 5 && child_surround < opp_surround) continue;

        GPUMove replies[MAX_LEGAL_MOVES];
        bool all_non_decreasing = true;
        int nreply = generate_non_decreasing_surround_replies(
            child, opp_color, child_surround, replies, all_non_decreasing
        );
        if (!all_non_decreasing) {
            continue;
        }
        bool candidate_ok = true;
        for (int r = 0; r < nreply; ++r) {
            HiveState grand = child;
            apply_move(grand, replies[r]);
            if (!state_has_immediate_win_for_current_player(grand)) {
                candidate_ok = false;
                break;
            }
        }
        if (candidate_ok) {
            winning_move_out[idx] = i;
            return;
        }
    }
}

// ── Host wrapper functions (called from Python via pybind11) ────────

/**
 * Initialize constant memory tables. Must be called once before any kernel.
 */
void initialize_tables() {
    init_hex_tables();
    copy_tables_to_device();
}

/**
 * Allocate a batch of HiveStates on GPU and initialize them.
 * Returns a tensor wrapping the raw GPU memory (treated as opaque bytes).
 */
at::Tensor create_initial_states(int batch_size, int expansion_mask) {
    // Allocate as raw bytes tensor
    auto options = at::TensorOptions()
        .dtype(c10::kByte)
        .device(c10::kCUDA);
    auto states_tensor = at::zeros({batch_size, (int64_t)sizeof(HiveState)}, options);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    init_states_kernel<<<blocks, threads>>>(states_ptr, batch_size, (uint8_t)expansion_mask);
    cudaDeviceSynchronize();

    return states_tensor;
}

/**
 * Generate legal moves for all states in the batch.
 * Returns: (moves_tensor, num_legal_tensor)
 *   moves_tensor: [batch_size, MAX_LEGAL_MOVES, sizeof(GPUMove)] as uint8
 *   num_legal_tensor: [batch_size] as int32
 */
std::tuple<at::Tensor, at::Tensor> generate_legal_moves_batch(
    at::Tensor states_tensor,
    int batch_size
) {
    auto options_u8 = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);
    auto options_i32 = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);

    auto moves_tensor = at::zeros(
        {batch_size, MAX_LEGAL_MOVES, (int64_t)sizeof(GPUMove)}, options_u8);
    auto num_legal = at::zeros({batch_size}, options_i32);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    GPUMove* moves_ptr = reinterpret_cast<GPUMove*>(moves_tensor.data_ptr());
    int* num_legal_ptr = static_cast<int*>(num_legal.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    generate_legal_moves_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, num_legal_ptr, batch_size);
    // No cudaDeviceSynchronize: null-stream ordering guarantees correctness
    // for GPU-to-GPU passes; PyTorch syncs automatically on .cpu() readback.

    return std::make_tuple(moves_tensor, num_legal);
}

/**
 * Apply one move per game in the batch.
 * moves_tensor: [batch_size, sizeof(GPUMove)] as uint8
 */
void apply_moves_batch(
    at::Tensor states_tensor,
    at::Tensor moves_tensor,
    int batch_size
) {
    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    const GPUMove* moves_ptr = reinterpret_cast<const GPUMove*>(moves_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    apply_moves_kernel<<<blocks, threads>>>(states_ptr, moves_ptr, batch_size);
    // No cudaDeviceSynchronize: null-stream ordering is sufficient.
}

/**
 * Get game results for all states in the batch.
 * Returns: [batch_size] int32 tensor (0=in_progress, 1=white_wins, 2=black_wins, 3=draw)
 */
at::Tensor check_results_batch(at::Tensor states_tensor, int batch_size) {
    auto options = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto results = at::zeros({batch_size}, options);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    int* results_ptr = static_cast<int*>(results.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    check_results_kernel<<<blocks, threads>>>(states_ptr, results_ptr, batch_size);
    // No cudaDeviceSynchronize: null-stream ordering handles GPU-to-GPU;
    // PyTorch syncs on .cpu() when callers read back to host.

    return results;
}

/**
 * Encode batch of HiveStates into NN input features.
 * Returns tuple of 9 tensors (see state_encoder.cuh for layout).
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
encode_states_batch(at::Tensor states_tensor, int batch_size) {
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto opts_i = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);

    auto node_features    = at::zeros({batch_size, MAX_ENC_NODES, NODE_FEAT_DIM}, opts_f);
    auto node_grid_pos    = at::full({batch_size, MAX_ENC_NODES, 2}, -1, opts_i);
    auto node_piece_types = at::full({batch_size, MAX_ENC_NODES}, -1, opts_i);
    auto global_features  = at::zeros({batch_size, GLOBAL_FEAT_DIM}, opts_f);
    auto num_nodes        = at::zeros({batch_size}, opts_i);
    auto num_board_nodes  = at::zeros({batch_size}, opts_i);
    auto edge_index       = at::zeros({batch_size, MAX_ENC_EDGES, 2}, opts_i);
    auto edge_features    = at::zeros({batch_size, MAX_ENC_EDGES, EDGE_FEAT_DIM}, opts_f);
    auto num_edges        = at::zeros({batch_size}, opts_i);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    encode_states_kernel<<<blocks, threads>>>(
        states_ptr,
        static_cast<float*>(node_features.data_ptr()),
        static_cast<int*>(node_grid_pos.data_ptr()),
        static_cast<int*>(node_piece_types.data_ptr()),
        static_cast<float*>(global_features.data_ptr()),
        static_cast<int*>(num_nodes.data_ptr()),
        static_cast<int*>(num_board_nodes.data_ptr()),
        static_cast<int*>(edge_index.data_ptr()),
        static_cast<float*>(edge_features.data_ptr()),
        static_cast<int*>(num_edges.data_ptr()),
        batch_size
    );
    cudaDeviceSynchronize();

    return std::make_tuple(
        node_features, node_grid_pos, node_piece_types, global_features,
        num_nodes, num_board_nodes, edge_index, edge_features, num_edges
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hybrid_gnn_encode_batch(at::Tensor states_tensor, int batch_size, int radius) {
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto opts_i = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto opts_l = at::TensorOptions().dtype(c10::kLong).device(c10::kCUDA);
    auto opts_b = at::TensorOptions().dtype(c10::kBool).device(c10::kCUDA);

    auto node_features = at::zeros(
        {batch_size, HYBRID_MAX_NODES, HYBRID_NODE_FEAT_DIM}, opts_f);
    auto edge_src = at::zeros({batch_size, HYBRID_MAX_EDGES}, opts_l);
    auto edge_dst = at::zeros({batch_size, HYBRID_MAX_EDGES}, opts_l);
    auto edge_features = at::zeros(
        {batch_size, HYBRID_MAX_EDGES, HYBRID_EDGE_FEAT_DIM}, opts_f);
    auto node_mask = at::zeros({batch_size, HYBRID_MAX_NODES}, opts_b);
    auto edge_mask = at::zeros({batch_size, HYBRID_MAX_EDGES}, opts_b);
    auto global_features = at::zeros(
        {batch_size, HYBRID_GLOBAL_FEAT_DIM}, opts_f);
    auto num_nodes = at::zeros({batch_size}, opts_i);
    auto num_edges = at::zeros({batch_size}, opts_i);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    hybrid_gnn_encode_states_kernel<<<blocks, threads>>>(
        states_ptr,
        nullptr,
        nullptr,
        static_cast<float*>(node_features.data_ptr()),
        static_cast<int64_t*>(edge_src.data_ptr()),
        static_cast<int64_t*>(edge_dst.data_ptr()),
        static_cast<float*>(edge_features.data_ptr()),
        static_cast<bool*>(node_mask.data_ptr()),
        static_cast<bool*>(edge_mask.data_ptr()),
        static_cast<float*>(global_features.data_ptr()),
        static_cast<int*>(num_nodes.data_ptr()),
        static_cast<int*>(num_edges.data_ptr()),
        batch_size,
        radius,
        false
    );

    return std::make_tuple(
        node_features, edge_src, edge_dst, edge_features,
        node_mask, edge_mask, global_features, num_nodes, num_edges
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
hybrid_gnn_encode_with_moves_batch(
    at::Tensor states_tensor,
    at::Tensor legal_moves_tensor,
    at::Tensor num_legal_tensor,
    int batch_size,
    int radius
) {
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto opts_i = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto opts_l = at::TensorOptions().dtype(c10::kLong).device(c10::kCUDA);
    auto opts_b = at::TensorOptions().dtype(c10::kBool).device(c10::kCUDA);

    auto node_features = at::zeros(
        {batch_size, HYBRID_MAX_NODES, HYBRID_NODE_FEAT_DIM}, opts_f);
    auto edge_src = at::zeros({batch_size, HYBRID_MAX_EDGES}, opts_l);
    auto edge_dst = at::zeros({batch_size, HYBRID_MAX_EDGES}, opts_l);
    auto edge_features = at::zeros(
        {batch_size, HYBRID_MAX_EDGES, HYBRID_EDGE_FEAT_DIM}, opts_f);
    auto node_mask = at::zeros({batch_size, HYBRID_MAX_NODES}, opts_b);
    auto edge_mask = at::zeros({batch_size, HYBRID_MAX_EDGES}, opts_b);
    auto global_features = at::zeros(
        {batch_size, HYBRID_GLOBAL_FEAT_DIM}, opts_f);
    auto num_nodes = at::zeros({batch_size}, opts_i);
    auto num_edges = at::zeros({batch_size}, opts_i);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    GPUMove* legal_moves_ptr = reinterpret_cast<GPUMove*>(legal_moves_tensor.data_ptr());
    int* num_legal_ptr = static_cast<int*>(num_legal_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    hybrid_gnn_encode_states_kernel<<<blocks, threads>>>(
        states_ptr,
        legal_moves_ptr,
        num_legal_ptr,
        static_cast<float*>(node_features.data_ptr()),
        static_cast<int64_t*>(edge_src.data_ptr()),
        static_cast<int64_t*>(edge_dst.data_ptr()),
        static_cast<float*>(edge_features.data_ptr()),
        static_cast<bool*>(node_mask.data_ptr()),
        static_cast<bool*>(edge_mask.data_ptr()),
        static_cast<float*>(global_features.data_ptr()),
        static_cast<int*>(num_nodes.data_ptr()),
        static_cast<int*>(num_edges.data_ptr()),
        batch_size,
        radius,
        true
    );

    return std::make_tuple(
        node_features, edge_src, edge_dst, edge_features,
        node_mask, edge_mask, global_features, num_nodes, num_edges
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
hybrid_transformer_encode_with_moves_batch(
    at::Tensor states_tensor,
    at::Tensor legal_moves_tensor,
    at::Tensor num_legal_tensor,
    int batch_size
) {
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto opts_i = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto opts_b = at::TensorOptions().dtype(c10::kBool).device(c10::kCUDA);

    auto token_features = at::zeros(
        {batch_size, HYBRID_MAX_PIECE_TOKENS, HYBRID_NODE_FEAT_DIM}, opts_f);
    auto token_q = at::zeros({batch_size, HYBRID_MAX_PIECE_TOKENS}, opts_i);
    auto token_r = at::zeros({batch_size, HYBRID_MAX_PIECE_TOKENS}, opts_i);
    auto token_z = at::zeros({batch_size, HYBRID_MAX_PIECE_TOKENS}, opts_i);
    auto token_mask = at::zeros({batch_size, HYBRID_MAX_PIECE_TOKENS}, opts_b);
    auto global_features = at::zeros(
        {batch_size, HYBRID_GLOBAL_FEAT_DIM}, opts_f);
    auto num_tokens = at::zeros({batch_size}, opts_i);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    GPUMove* legal_moves_ptr = reinterpret_cast<GPUMove*>(legal_moves_tensor.data_ptr());
    int* num_legal_ptr = static_cast<int*>(num_legal_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    hybrid_transformer_encode_states_kernel<<<blocks, threads>>>(
        states_ptr,
        legal_moves_ptr,
        num_legal_ptr,
        static_cast<float*>(token_features.data_ptr()),
        static_cast<int*>(token_q.data_ptr()),
        static_cast<int*>(token_r.data_ptr()),
        static_cast<int*>(token_z.data_ptr()),
        static_cast<bool*>(token_mask.data_ptr()),
        static_cast<float*>(global_features.data_ptr()),
        static_cast<int*>(num_tokens.data_ptr()),
        batch_size,
        true
    );

    return std::make_tuple(
        token_features,
        token_q,
        token_r,
        token_z,
        token_mask,
        global_features,
        num_tokens
    );
}

at::Tensor hybrid_transformer_move_features_batch(
    at::Tensor states_tensor,
    at::Tensor legal_moves_tensor,
    at::Tensor num_legal_tensor,
    int batch_size
) {
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto move_features = at::zeros(
        {batch_size, MAX_LEGAL_MOVES, HYBRID_MOVE_FEAT_DIM}, opts_f);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    GPUMove* legal_moves_ptr = reinterpret_cast<GPUMove*>(legal_moves_tensor.data_ptr());
    int* num_legal_ptr = static_cast<int*>(num_legal_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    hybrid_transformer_move_features_kernel<<<blocks, threads>>>(
        states_ptr,
        legal_moves_ptr,
        num_legal_ptr,
        static_cast<float*>(move_features.data_ptr()),
        batch_size
    );
    return move_features;
}

/**
 * Generate legal action masks for the 29407-dim action space.
 * Calls move generation internally, then maps moves to action indices.
 * Returns: (masks, num_legal)
 *   masks: [batch_size, 29407] float32
 *   num_legal: [batch_size] int32
 */
std::tuple<at::Tensor, at::Tensor> generate_legal_mask_batch(
    at::Tensor states_tensor,
    int batch_size
) {
    auto opts_u8  = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);
    auto opts_f   = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto opts_i   = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);

    // Step 1: Generate legal moves
    auto moves_tensor = at::zeros(
        {batch_size, MAX_LEGAL_MOVES, (int64_t)sizeof(GPUMove)}, opts_u8);
    auto num_legal = at::zeros({batch_size}, opts_i);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    GPUMove* moves_ptr = reinterpret_cast<GPUMove*>(moves_tensor.data_ptr());
    int* num_legal_ptr = static_cast<int*>(num_legal.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    generate_legal_moves_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, num_legal_ptr, batch_size);
    // No sync: legal_mask_kernel on the same stream reads the result correctly.

    // Step 2: Map moves to action mask
    auto masks = at::zeros({batch_size, (int64_t)ACTION_SPACE_SIZE}, opts_f);
    float* masks_ptr = static_cast<float*>(masks.data_ptr());

    legal_mask_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, num_legal_ptr, masks_ptr, batch_size);
    // No sync: null-stream ordering; PyTorch syncs on .cpu() when needed.

    return std::make_tuple(masks, num_legal);
}

/**
 * Fused legal-moves + mask generation.
 * Runs generate_legal_moves_kernel once and legal_mask_kernel once,
 * returning all three outputs. Replaces separate calls to
 * generate_legal_moves_batch + generate_legal_mask_batch in the MCTS
 * wave loop, halving move-generation kernel launches per wave.
 *
 * Returns: (moves_tensor [B, MAX_LEGAL_MOVES, sizeof(GPUMove)] uint8,
 *           num_legal    [B] int32,
 *           masks        [B, ACTION_SPACE_SIZE] float32)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> generate_legal_moves_and_mask_batch(
    at::Tensor states_tensor, int batch_size
) {
    auto opts_u8 = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);
    auto opts_f  = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto opts_i  = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);

    auto moves_tensor = at::zeros(
        {batch_size, MAX_LEGAL_MOVES, (int64_t)sizeof(GPUMove)}, opts_u8);
    auto num_legal = at::zeros({batch_size}, opts_i);
    auto masks     = at::zeros({batch_size, (int64_t)ACTION_SPACE_SIZE}, opts_f);

    HiveState* states_ptr  = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    GPUMove*   moves_ptr   = reinterpret_cast<GPUMove*>(moves_tensor.data_ptr());
    int*       nl_ptr      = static_cast<int*>(num_legal.data_ptr());
    float*     masks_ptr   = static_cast<float*>(masks.data_ptr());

    int threads = 256;
    int blocks  = (batch_size + threads - 1) / threads;

    generate_legal_moves_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, nl_ptr, batch_size);
    // Same-stream ordering: legal_mask_kernel starts after generate completes.
    legal_mask_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, nl_ptr, masks_ptr, batch_size);

    return std::make_tuple(moves_tensor, num_legal, masks);
}

__global__ void legal_moves_to_actions_kernel(
    const HiveState* states,
    const GPUMove* moves,     // [B, MAX_LEGAL_MOVES]
    const int* num_legal,     // [B]
    int* action_indices,      // [B, MAX_LEGAL_MOVES]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const HiveState& s = states[idx];
    const GPUMove* my_moves = moves + idx * MAX_LEGAL_MOVES;
    int* my_actions = action_indices + idx * MAX_LEGAL_MOVES;
    int n_legal = num_legal[idx];

    int center_q, center_r;
    compute_centroid(s, center_q, center_r);

    for (int m = 0; m < MAX_LEGAL_MOVES; ++m) {
        my_actions[m] = -1;
    }
    for (int m = 0; m < n_legal; ++m) {
        my_actions[m] = gpu_move_to_action(my_moves[m], center_q, center_r);
    }
}

at::Tensor legal_moves_to_actions_batch(
    at::Tensor states_tensor,
    at::Tensor legal_moves_tensor,
    at::Tensor num_legal_tensor,
    int batch_size
) {
    auto opts_i = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto action_indices = at::full({batch_size, MAX_LEGAL_MOVES}, -1, opts_i);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    GPUMove* moves_ptr = reinterpret_cast<GPUMove*>(legal_moves_tensor.data_ptr());
    int* num_legal_ptr = static_cast<int*>(num_legal_tensor.data_ptr());
    int* action_ptr = static_cast<int*>(action_indices.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    legal_moves_to_actions_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, num_legal_ptr, action_ptr, batch_size);

    return action_indices;
}

// ── Kernel: compute per-piece mobility for a batch of states ────────

__global__ void compute_mobility_kernel(
    const HiveState* states,
    float* mobility_out,       // [batch_size, MAX_ENC_NODES]
    int* num_board_nodes_out,  // [batch_size]
    int batch_size,
    bool both_players
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float* mob = mobility_out + idx * MAX_ENC_NODES;
    int num_board = compute_piece_mobility(states[idx], mob, both_players);
    num_board_nodes_out[idx] = num_board;
}

/**
 * Compute per-piece mobility targets for a batch of states.
 * Returns: (mobility_tensor [B, MAX_ENC_NODES] float32,
 *           num_board_nodes [B] int32)
 */
std::tuple<at::Tensor, at::Tensor> compute_mobility_batch(
    at::Tensor states_tensor, int batch_size, bool both_players
) {
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto opts_i = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);

    auto mobility = at::zeros({batch_size, (int64_t)MAX_ENC_NODES}, opts_f);
    auto num_board_nodes = at::zeros({batch_size}, opts_i);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    compute_mobility_kernel<<<blocks, threads>>>(
        states_ptr,
        static_cast<float*>(mobility.data_ptr()),
        static_cast<int*>(num_board_nodes.data_ptr()),
        batch_size,
        both_players
    );
    cudaDeviceSynchronize();

    return std::make_tuple(mobility, num_board_nodes);
}

/**
 * Compute per-state centroids on GPU.
 * Returns: centroids [B, 2] int32 (center_q, center_r)
 */
at::Tensor compute_centroids_batch(at::Tensor states_tensor, int batch_size) {
    auto opts_i = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto centroids = at::zeros({batch_size, 2}, opts_i);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    compute_centroids_kernel<<<blocks, threads>>>(
        states_ptr,
        static_cast<int*>(centroids.data_ptr()),
        batch_size
    );
    cudaDeviceSynchronize();

    return centroids;
}

// ═══════════════════════════════════════════════════════════════════
// GPU-native MCTS host wrappers
// ═══════════════════════════════════════════════════════════════════

namespace {
MCTSTree build_tree(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt, int max_nodes
) {
    MCTSTree t;
    t.visit_count    = static_cast<int*>(vc.data_ptr());
    t.total_value    = static_cast<float*>(tv.data_ptr());
    t.prior          = static_cast<float*>(pr.data_ptr());
    t.parent_idx     = static_cast<int*>(pa.data_ptr());
    t.move_bytes     = static_cast<uint8_t*>(mb.data_ptr());
    t.action_idx     = static_cast<int*>(ai.data_ptr());
    t.first_child    = static_cast<int*>(fc.data_ptr());
    t.num_children   = static_cast<int*>(nc.data_ptr());
    t.is_terminal    = static_cast<int8_t*>(it.data_ptr());
    t.terminal_value = static_cast<float*>(tv2.data_ptr());
    t.node_count     = static_cast<int*>(cnt.data_ptr());
    t.max_nodes      = max_nodes;
    return t;
}
}  // anon namespace

/**
 * MCTS select: PUCT walk from root to leaf for W*B simultaneous sims.
 * Returns (leaf_indices, move_paths, path_lengths, vl_paths, vl_lengths).
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mcts_select_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor game_active, at::Tensor root_nodes,
    float c_puct, int B, int W, int max_nodes
) {
    int total = W * B;
    auto oi = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto ou = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);

    auto leaf_idx   = at::zeros({total}, oi);
    auto move_paths = at::zeros({total, MAX_TREE_DEPTH, (int64_t)sizeof(GPUMove)}, ou);
    auto path_lens  = at::zeros({total}, oi);
    auto vl_paths   = at::zeros({total, MAX_TREE_DEPTH}, oi);
    auto vl_lens    = at::zeros({total}, oi);

    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_select_kernel<<<blocks, threads>>>(
        tree,
        static_cast<int*>(leaf_idx.data_ptr()),
        static_cast<uint8_t*>(move_paths.data_ptr()),
        static_cast<int*>(path_lens.data_ptr()),
        static_cast<int*>(vl_paths.data_ptr()),
        static_cast<int*>(vl_lens.data_ptr()),
        static_cast<const int8_t*>(game_active.data_ptr()),
        static_cast<const int*>(root_nodes.data_ptr()),
        c_puct, B, total);
    // No sync: next kernel on same stream reads outputs after this completes.

    return std::make_tuple(leaf_idx, move_paths, path_lens, vl_paths, vl_lens);
}

/**
 * MCTS replay: replay move paths to compute leaf states.
 */
void mcts_replay_batch(
    at::Tensor root_states, at::Tensor leaf_states,
    at::Tensor move_paths, at::Tensor path_lengths, at::Tensor leaf_indices,
    int B, int total
) {
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_replay_kernel<<<blocks, threads>>>(
        reinterpret_cast<const HiveState*>(root_states.data_ptr()),
        reinterpret_cast<HiveState*>(leaf_states.data_ptr()),
        static_cast<const uint8_t*>(move_paths.data_ptr()),
        static_cast<const int*>(path_lengths.data_ptr()),
        static_cast<const int*>(leaf_indices.data_ptr()),
        B, total);
    // No sync: downstream kernels on same stream see leaf_states correctly.
}

/**
 * MCTS expand: create children for unexpanded, non-terminal leaves.
 * Returns was_expanded [total_sims] int8.
 */
at::Tensor mcts_expand_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor leaf_indices, at::Tensor leaf_states,
    at::Tensor legal_moves, at::Tensor num_legal,
    at::Tensor action_probs, at::Tensor results,
    int B, int total, int max_nodes
) {
    auto was_expanded = at::zeros({total}, at::TensorOptions().dtype(c10::kChar).device(c10::kCUDA));
    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_expand_kernel<<<blocks, threads>>>(
        tree,
        static_cast<const int*>(leaf_indices.data_ptr()),
        reinterpret_cast<const HiveState*>(leaf_states.data_ptr()),
        reinterpret_cast<const GPUMove*>(legal_moves.data_ptr()),
        static_cast<const int*>(num_legal.data_ptr()),
        static_cast<const float*>(action_probs.data_ptr()),
        static_cast<const int*>(results.data_ptr()),
        static_cast<int8_t*>(was_expanded.data_ptr()),
        B, total);
    // No sync: backprop kernel on same stream reads was_expanded after this completes.

    return was_expanded;
}

/**
 * Fused MCTS expand + backprop.
 * Combines mcts_expand_kernel and mcts_backprop_kernel into a single kernel
 * launch: was_expanded lives in a register rather than global memory,
 * eliminating one kernel launch and the associated global memory traffic.
 */
void mcts_expand_and_backprop_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor leaf_indices, at::Tensor leaf_states,
    at::Tensor legal_moves, at::Tensor num_legal,
    at::Tensor action_probs, at::Tensor results,
    at::Tensor nn_values, at::Tensor vl_paths, at::Tensor vl_lengths,
    int B, int total, int max_nodes
) {
    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_expand_and_backprop_kernel<<<blocks, threads>>>(
        tree,
        static_cast<const int*>(leaf_indices.data_ptr()),
        reinterpret_cast<const HiveState*>(leaf_states.data_ptr()),
        reinterpret_cast<const GPUMove*>(legal_moves.data_ptr()),
        static_cast<const int*>(num_legal.data_ptr()),
        static_cast<const float*>(action_probs.data_ptr()),
        static_cast<const int*>(results.data_ptr()),
        static_cast<const float*>(nn_values.data_ptr()),
        static_cast<const int*>(vl_paths.data_ptr()),
        static_cast<const int*>(vl_lengths.data_ptr()),
        B, total);
    // No sync: next wave's select runs on same stream after this completes.
}

/**
 * MCTS select with root alive-mask (for Gumbel Sequential Halving).
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mcts_select_with_root_mask_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor game_active, at::Tensor root_nodes,
    at::Tensor alive_mask, int max_root_children,
    float c_puct, int B, int W, int max_nodes
) {
    int total = W * B;
    auto oi = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto ou = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);

    auto leaf_idx   = at::zeros({total}, oi);
    auto move_paths = at::zeros({total, MAX_TREE_DEPTH, (int64_t)sizeof(GPUMove)}, ou);
    auto path_lens  = at::zeros({total}, oi);
    auto vl_paths   = at::zeros({total, MAX_TREE_DEPTH}, oi);
    auto vl_lens    = at::zeros({total}, oi);

    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_select_with_root_mask_kernel<<<blocks, threads>>>(
        tree,
        static_cast<int*>(leaf_idx.data_ptr()),
        static_cast<uint8_t*>(move_paths.data_ptr()),
        static_cast<int*>(path_lens.data_ptr()),
        static_cast<int*>(vl_paths.data_ptr()),
        static_cast<int*>(vl_lens.data_ptr()),
        static_cast<const int8_t*>(game_active.data_ptr()),
        static_cast<const int*>(root_nodes.data_ptr()),
        static_cast<const int8_t*>(alive_mask.data_ptr()),
        max_root_children,
        c_puct, B, total);

    return std::make_tuple(leaf_idx, move_paths, path_lens, vl_paths, vl_lens);
}

/**
 * MCTS select with explicit root slots (for equal-budget Gumbel halving).
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mcts_select_with_root_slots_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor game_active, at::Tensor root_nodes,
    at::Tensor root_slots, int num_candidates, int max_root_children,
    float c_puct, int B, int W, int max_nodes
) {
    int total = W * num_candidates * B;
    auto oi = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto ou = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);

    auto leaf_idx   = at::zeros({total}, oi);
    auto move_paths = at::zeros({total, MAX_TREE_DEPTH, (int64_t)sizeof(GPUMove)}, ou);
    auto path_lens  = at::zeros({total}, oi);
    auto vl_paths   = at::zeros({total, MAX_TREE_DEPTH}, oi);
    auto vl_lens    = at::zeros({total}, oi);

    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_select_with_root_slots_kernel<<<blocks, threads>>>(
        tree,
        static_cast<int*>(leaf_idx.data_ptr()),
        static_cast<uint8_t*>(move_paths.data_ptr()),
        static_cast<int*>(path_lens.data_ptr()),
        static_cast<int*>(vl_paths.data_ptr()),
        static_cast<int*>(vl_lens.data_ptr()),
        static_cast<const int8_t*>(game_active.data_ptr()),
        static_cast<const int*>(root_nodes.data_ptr()),
        static_cast<const int*>(root_slots.data_ptr()),
        num_candidates, max_root_children,
        c_puct, B, total);

    return std::make_tuple(leaf_idx, move_paths, path_lens, vl_paths, vl_lens);
}

/**
 * Fused root-slot select + replay + terminal check + legal/FNN feature build.
 *
 * This is the largest prefix of one Gumbel wave that can stay inside the CUDA
 * extension before Python must run the FNN forward pass.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mcts_select_replay_legal_fnn_root_slots_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor root_states,
    at::Tensor game_active, at::Tensor root_nodes,
    at::Tensor root_slots, int num_candidates, int max_root_children,
    float c_puct, int B, int W, int max_nodes
) {
    int total = W * num_candidates * B;
    auto oi = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto ou = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);
    auto of = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);

    auto leaf_idx   = at::zeros({total}, oi);
    auto move_paths = at::zeros({total, MAX_TREE_DEPTH, (int64_t)sizeof(GPUMove)}, ou);
    auto path_lens  = at::zeros({total}, oi);
    auto vl_paths   = at::zeros({total, MAX_TREE_DEPTH}, oi);
    auto vl_lens    = at::zeros({total}, oi);
    auto leaf_states = at::zeros({total, root_states.size(1)}, ou);
    auto results = at::zeros({total}, oi);
    auto legal_moves = at::zeros(
        {total, (int64_t)MAX_LEGAL_MOVES, (int64_t)sizeof(GPUMove)}, ou);
    auto num_legal = at::zeros({total}, oi);
    auto features = at::zeros({total, (int64_t)FNN_FEAT_DIM}, of);

    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mcts_select_with_root_slots_kernel<<<blocks, threads>>>(
        tree,
        static_cast<int*>(leaf_idx.data_ptr()),
        static_cast<uint8_t*>(move_paths.data_ptr()),
        static_cast<int*>(path_lens.data_ptr()),
        static_cast<int*>(vl_paths.data_ptr()),
        static_cast<int*>(vl_lens.data_ptr()),
        static_cast<const int8_t*>(game_active.data_ptr()),
        static_cast<const int*>(root_nodes.data_ptr()),
        static_cast<const int*>(root_slots.data_ptr()),
        num_candidates, max_root_children,
        c_puct, B, total);

    mcts_replay_kernel<<<blocks, threads>>>(
        reinterpret_cast<const HiveState*>(root_states.data_ptr()),
        reinterpret_cast<HiveState*>(leaf_states.data_ptr()),
        static_cast<const uint8_t*>(move_paths.data_ptr()),
        static_cast<const int*>(path_lens.data_ptr()),
        static_cast<const int*>(leaf_idx.data_ptr()),
        B, total);

    check_results_kernel<<<blocks, threads>>>(
        reinterpret_cast<HiveState*>(leaf_states.data_ptr()),
        static_cast<int*>(results.data_ptr()),
        total);

    generate_legal_moves_and_fnn_features_kernel<<<blocks, threads>>>(
        reinterpret_cast<const HiveState*>(leaf_states.data_ptr()),
        reinterpret_cast<GPUMove*>(legal_moves.data_ptr()),
        static_cast<int*>(num_legal.data_ptr()),
        static_cast<float*>(features.data_ptr()),
        total);

    return std::make_tuple(
        leaf_idx, leaf_states, legal_moves, num_legal, features,
        results, vl_paths, vl_lens);
}

/**
 * MCTS expand with dense per-legal-move priors (no ACTION_SPACE indirection).
 */
at::Tensor mcts_expand_dense_priors_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor leaf_indices, at::Tensor leaf_states,
    at::Tensor legal_moves, at::Tensor num_legal,
    at::Tensor priors_per_legal, at::Tensor results,
    int B, int total, int max_nodes
) {
    auto was_expanded = at::zeros({total}, at::TensorOptions().dtype(c10::kChar).device(c10::kCUDA));
    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_expand_dense_priors_kernel<<<blocks, threads>>>(
        tree,
        static_cast<const int*>(leaf_indices.data_ptr()),
        reinterpret_cast<const HiveState*>(leaf_states.data_ptr()),
        reinterpret_cast<const GPUMove*>(legal_moves.data_ptr()),
        static_cast<const int*>(num_legal.data_ptr()),
        static_cast<const float*>(priors_per_legal.data_ptr()),
        static_cast<const int*>(results.data_ptr()),
        static_cast<int8_t*>(was_expanded.data_ptr()),
        B, total);

    return was_expanded;
}

/**
 * MCTS expand + backprop fused (dense-priors variant).
 */
void mcts_expand_and_backprop_dense_priors_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor leaf_indices, at::Tensor leaf_states,
    at::Tensor legal_moves, at::Tensor num_legal,
    at::Tensor priors_per_legal, at::Tensor results,
    at::Tensor nn_values, at::Tensor vl_paths, at::Tensor vl_lengths,
    int B, int total, int max_nodes
) {
    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_expand_and_backprop_dense_priors_kernel<<<blocks, threads>>>(
        tree,
        static_cast<const int*>(leaf_indices.data_ptr()),
        reinterpret_cast<const HiveState*>(leaf_states.data_ptr()),
        reinterpret_cast<const GPUMove*>(legal_moves.data_ptr()),
        static_cast<const int*>(num_legal.data_ptr()),
        static_cast<const float*>(priors_per_legal.data_ptr()),
        static_cast<const int*>(results.data_ptr()),
        static_cast<const float*>(nn_values.data_ptr()),
        static_cast<const int*>(vl_paths.data_ptr()),
        static_cast<const int*>(vl_lengths.data_ptr()),
        B, total);
}

/**
 * MCTS backprop: undo virtual loss, propagate values from leaf to root.
 */
void mcts_backprop_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor leaf_indices, at::Tensor nn_values,
    at::Tensor vl_paths, at::Tensor vl_lengths, at::Tensor was_expanded,
    int B, int total, int max_nodes
) {
    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    mcts_backprop_kernel<<<blocks, threads>>>(
        tree,
        static_cast<const int*>(leaf_indices.data_ptr()),
        static_cast<const float*>(nn_values.data_ptr()),
        static_cast<const int*>(vl_paths.data_ptr()),
        static_cast<const int*>(vl_lengths.data_ptr()),
        static_cast<const int8_t*>(was_expanded.data_ptr()),
        B, total);
    // No sync: next operation on same stream executes after backprop completes.
}

/**
 * MCTS extract policy: build policy vectors from root visit counts.
 * Returns [B, ACTION_SPACE_SIZE] float32.
 */
at::Tensor mcts_extract_policy_batch(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor move_numbers, at::Tensor root_nodes,
    float temperature, int temp_drop_move, float pruning_threshold,
    int B, int max_nodes
) {
    auto policies = at::zeros(
        {B, (int64_t)ACTION_SPACE_SIZE},
        at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA));
    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (B + threads - 1) / threads;
    mcts_extract_policy_kernel<<<blocks, threads>>>(
        tree,
        static_cast<float*>(policies.data_ptr()),
        static_cast<const int*>(move_numbers.data_ptr()),
        static_cast<const int*>(root_nodes.data_ptr()),
        temperature, temp_drop_move, pruning_threshold, B);
    // No sync: PyTorch syncs on .cpu() when Python reads the result.

    return policies;
}

/**
 * MCTS root noise: apply root policy temperature + Dirichlet noise.
 */
void mcts_apply_root_noise(
    at::Tensor vc, at::Tensor tv, at::Tensor pr, at::Tensor pa,
    at::Tensor mb, at::Tensor ai, at::Tensor fc, at::Tensor nc,
    at::Tensor it, at::Tensor tv2, at::Tensor cnt,
    at::Tensor noise, at::Tensor root_nodes, int max_children_pad,
    float dir_eps, float root_policy_temp,
    int B, int max_nodes
) {
    MCTSTree tree = build_tree(vc, tv, pr, pa, mb, ai, fc, nc, it, tv2, cnt, max_nodes);

    int threads = 256;
    int blocks  = (B + threads - 1) / threads;
    mcts_root_noise_kernel<<<blocks, threads>>>(
        tree,
        static_cast<const float*>(noise.data_ptr()),
        static_cast<const int*>(root_nodes.data_ptr()),
        max_children_pad, dir_eps, root_policy_temp, B);
    // No sync: same-stream ordering; next op sees updated priors correctly.
}

/**
 * Extract FNN board features directly from HiveState + legal moves.
 * Bypasses full encode_states_batch pipeline for ~5x speedup.
 * Returns: features [B, FNN_FEAT_DIM] float32
 */
at::Tensor extract_fnn_features_batch(
    at::Tensor states_tensor,
    at::Tensor legal_moves_tensor,
    at::Tensor num_legal_tensor,
    int batch_size
) {
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto features = at::zeros({batch_size, (int64_t)FNN_FEAT_DIM}, opts_f);

    HiveState* states_ptr = reinterpret_cast<HiveState*>(states_tensor.data_ptr());
    GPUMove* moves_ptr = reinterpret_cast<GPUMove*>(legal_moves_tensor.data_ptr());
    int* num_legal_ptr = static_cast<int*>(num_legal_tensor.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    extract_fnn_features_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, num_legal_ptr,
        static_cast<float*>(features.data_ptr()),
        batch_size);
    // No sync: null-stream ordering sufficient for GPU-to-GPU.

    return features;
}

/**
 * Generate legal moves and FNN features in one per-state kernel.
 *
 * This avoids launching a separate feature-extraction kernel that rereads the
 * just-generated legal move list from global memory.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor>
generate_legal_moves_and_fnn_features_batch(
    at::Tensor states_tensor,
    int batch_size
) {
    auto opts_u8 = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);
    auto opts_i = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);

    auto moves_tensor = at::zeros(
        {batch_size, (int64_t)MAX_LEGAL_MOVES, (int64_t)sizeof(GPUMove)}, opts_u8);
    auto num_legal = at::zeros({batch_size}, opts_i);
    auto features = at::zeros({batch_size, (int64_t)FNN_FEAT_DIM}, opts_f);

    const HiveState* states_ptr = reinterpret_cast<const HiveState*>(states_tensor.data_ptr());
    GPUMove* moves_ptr = reinterpret_cast<GPUMove*>(moves_tensor.data_ptr());
    int* num_legal_ptr = static_cast<int*>(num_legal.data_ptr());
    float* features_ptr = static_cast<float*>(features.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    generate_legal_moves_and_fnn_features_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, num_legal_ptr, features_ptr, batch_size);

    return std::make_tuple(moves_tensor, num_legal, features);
}

at::Tensor queen_escape_flags_batch(
    at::Tensor states_tensor,
    int batch_size
) {
    auto opts_u8 = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);
    auto flags = at::zeros({batch_size}, opts_u8);

    const HiveState* states_ptr = reinterpret_cast<const HiveState*>(states_tensor.data_ptr());
    uint8_t* flags_ptr = static_cast<uint8_t*>(flags.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    queen_escape_flags_kernel<<<blocks, threads>>>(
        states_ptr, flags_ptr, batch_size);

    return flags;
}

at::Tensor endgame_hit_mask_batch(
    at::Tensor states_tensor,
    int batch_size,
    int min_surround,
    int max_surround,
    bool require_mixed_pair
) {
    auto opts_u8 = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);
    auto hit = at::zeros({batch_size}, opts_u8);

    const HiveState* states_ptr = reinterpret_cast<const HiveState*>(states_tensor.data_ptr());
    uint8_t* hit_ptr = static_cast<uint8_t*>(hit.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    endgame_hit_mask_kernel<<<blocks, threads>>>(
        states_ptr, hit_ptr, min_surround, max_surround,
        require_mixed_pair ? 1 : 0, batch_size);

    return hit;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
root_tactical_probe_batch(
    at::Tensor states_tensor,
    at::Tensor legal_moves_tensor,
    at::Tensor num_legal_tensor,
    at::Tensor priors_tensor,
    int batch_size,
    bool enable_win_in_one,
    bool enable_check_opponent_wins,
    bool enable_win_in_two
) {
    auto opts_i64 = at::TensorOptions().dtype(c10::kLong).device(c10::kCUDA);
    auto opts_u8 = at::TensorOptions().dtype(c10::kByte).device(c10::kCUDA);

    auto winning = at::full({batch_size}, -1, opts_i64);
    auto allowed = at::zeros({batch_size, (int64_t)MAX_LEGAL_MOVES}, opts_u8);
    auto forced = at::zeros({batch_size}, opts_u8);

    const HiveState* states_ptr = reinterpret_cast<const HiveState*>(states_tensor.data_ptr());
    const GPUMove* moves_ptr = reinterpret_cast<const GPUMove*>(legal_moves_tensor.data_ptr());
    const int* num_legal_ptr = static_cast<const int*>(num_legal_tensor.data_ptr());
    const float* priors_ptr = static_cast<const float*>(priors_tensor.data_ptr());
    int64_t* winning_ptr = static_cast<int64_t*>(winning.data_ptr());
    uint8_t* allowed_ptr = static_cast<uint8_t*>(allowed.data_ptr());
    uint8_t* forced_ptr = static_cast<uint8_t*>(forced.data_ptr());

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    root_tactical_probe_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, num_legal_ptr, priors_ptr,
        winning_ptr, allowed_ptr, forced_ptr,
        enable_win_in_one ? 1 : 0,
        enable_check_opponent_wins ? 1 : 0,
        enable_win_in_two ? 1 : 0,
        batch_size);

    return std::make_tuple(winning, allowed, forced);
}

/**
 * Build each legal successor locally and return only its FNN features.
 *
 * Used by FNN action scoring: successor features are needed, but successor
 * states and successor legal move lists are not otherwise consumed.
 */
at::Tensor fnn_successor_features_batch(
    at::Tensor states_tensor,
    at::Tensor legal_moves_tensor,
    at::Tensor action_to_root_tensor,
    at::Tensor move_indices_tensor,
    int num_actions
) {
    auto opts_f = at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA);
    auto features = at::zeros({num_actions, (int64_t)FNN_FEAT_DIM}, opts_f);

    const HiveState* states_ptr = reinterpret_cast<const HiveState*>(states_tensor.data_ptr());
    const GPUMove* moves_ptr = reinterpret_cast<const GPUMove*>(legal_moves_tensor.data_ptr());
    const int64_t* action_to_root = static_cast<const int64_t*>(action_to_root_tensor.data_ptr());
    const int64_t* move_indices = static_cast<const int64_t*>(move_indices_tensor.data_ptr());
    float* features_ptr = static_cast<float*>(features.data_ptr());

    int threads = 256;
    int blocks = (num_actions + threads - 1) / threads;
    fnn_successor_features_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, action_to_root, move_indices,
        features_ptr, num_actions);

    return features;
}

/**
 * GPU-native Gumbel AlphaZero self-play with FNN.
 *
 * Runs the complete self-play loop on GPU: move generation, feature
 * extraction, FNN forward pass, Gumbel sequential halving, and move
 * application — all in a single kernel launch.
 *
 * Returns: (states, policy_probs, policy_indices, num_legal,
 *           num_candidates, lengths, results)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
fnn_selfplay_batch(
    at::Tensor weights,
    int hidden_dim, int embed_dim, int action_hidden,
    int batch_size, int max_game_length,
    int num_simulations, int max_considered,
    float c_visit, float c_scale,
    int temperature_drop_move, int expansion_mask,
    int64_t rng_seed
) {
    auto dev = weights.device();
    auto opts_u8 = at::TensorOptions().dtype(c10::kByte).device(dev);
    auto opts_f32 = at::TensorOptions().dtype(c10::kFloat).device(dev);
    auto opts_i32 = at::TensorOptions().dtype(c10::kInt).device(dev);

    int ss = (int)sizeof(HiveState);

    auto out_states = at::zeros(
        {batch_size, max_game_length, (int64_t)ss}, opts_u8);
    auto out_policy_probs = at::zeros(
        {batch_size, max_game_length, (int64_t)max_considered}, opts_f32);
    auto out_policy_indices = at::full(
        {batch_size, max_game_length, (int64_t)max_considered}, -1, opts_i32);
    auto out_num_legal = at::zeros(
        {batch_size, (int64_t)max_game_length}, opts_i32);
    auto out_num_candidates = at::zeros(
        {batch_size, (int64_t)max_game_length}, opts_i32);
    auto out_lengths = at::zeros({batch_size}, opts_i32);
    auto out_results = at::zeros({batch_size}, opts_i32);

    dim3 grid(batch_size);
    dim3 block(SELFPLAY_BLOCK_SIZE);

    fnn_selfplay_kernel<<<grid, block>>>(
        weights.data_ptr<float>(),
        hidden_dim, embed_dim, action_hidden,
        out_states.data_ptr<uint8_t>(),
        out_policy_probs.data_ptr<float>(),
        out_policy_indices.data_ptr<int>(),
        out_num_legal.data_ptr<int>(),
        out_num_candidates.data_ptr<int>(),
        out_lengths.data_ptr<int>(),
        out_results.data_ptr<int>(),
        batch_size, max_game_length,
        num_simulations, max_considered,
        c_visit, c_scale,
        temperature_drop_move, expansion_mask,
        rng_seed, ss
    );

    // Synchronize — kernel may run for seconds
    cudaDeviceSynchronize();

    return {out_states, out_policy_probs, out_policy_indices,
            out_num_legal, out_num_candidates, out_lengths, out_results};
}

// ── PRS v2 slot mapping + head-input bridge ─────────────────────────

/**
 * One-shot batched build of all PRS v2 head inputs + per-legal slot indices.
 *
 * Inputs:
 *   states_tensor       : [B, sizeof(HiveState)]  uint8
 *   legal_moves_tensor  : [B, max_legal, sizeof(GPUMove)] uint8
 *   num_legal_tensor    : [B]  int32
 *
 * Returns a 16-tuple (all on CUDA):
 *   dir_piece_idx   [B, 8]            int64
 *   throw_piece_idx [B, 2]            int64
 *   long_piece_idx  [B, 7]            int64
 *   move_nbrs       [B, 64, 6]        int64
 *   place_nbrs      [B, 32, 6]        int64
 *   move_mask       [B, 64]           bool
 *   place_mask      [B, 32]           bool
 *   current_color   [B]               int64
 *   slot_of_legal   [B, max_legal]    int32   (-1 = padding / un-mappable)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
prs_v2_classify_batch(
    at::Tensor states_tensor,
    at::Tensor legal_moves_tensor,
    at::Tensor num_legal_tensor,
    int batch_size, int max_legal
) {
    auto opts_i64  = at::TensorOptions().dtype(c10::kLong).device(c10::kCUDA);
    auto opts_i32  = at::TensorOptions().dtype(c10::kInt).device(c10::kCUDA);
    auto opts_bool = at::TensorOptions().dtype(c10::kBool).device(c10::kCUDA);

    auto dir_piece_idx      = at::full({batch_size, PRS_V2_N_DIR_PIECES},   -1, opts_i64);
    auto throw_piece_idx    = at::full({batch_size, PRS_V2_N_THROW_PIECES}, -1, opts_i64);
    auto long_piece_idx     = at::full({batch_size, PRS_V2_N_LONG_PIECES},  -1, opts_i64);
    auto move_nbrs          = at::full({batch_size, PRS_V2_C_MOVE, NUM_DIRS}, -1, opts_i64);
    auto place_nbrs         = at::full({batch_size, PRS_V2_C_HAND, NUM_DIRS}, -1, opts_i64);
    auto move_mask          = at::zeros({batch_size, PRS_V2_C_MOVE}, opts_bool);
    auto place_mask         = at::zeros({batch_size, PRS_V2_C_HAND}, opts_bool);
    auto current_color      = at::zeros({batch_size}, opts_i64);
    auto slot_of_legal      = at::full({batch_size, max_legal}, -1, opts_i32);
    auto move_cell_ids      = at::full({batch_size, PRS_V2_C_MOVE}, -1, opts_i32);
    auto place_cell_ids     = at::full({batch_size, PRS_V2_C_HAND}, -1, opts_i32);
    auto dir_dest_cell      = at::full({batch_size, PRS_V2_N_DIR_PIECES, NUM_DIRS}, -1, opts_i32);
    auto dir_dest_board_idx = at::full({batch_size, PRS_V2_N_DIR_PIECES, NUM_DIRS}, -1, opts_i64);
    auto dir_dest_nbrs      = at::full({batch_size, PRS_V2_N_DIR_PIECES, NUM_DIRS, NUM_DIRS}, -1, opts_i64);
    auto throw_dest_cell    = at::full({batch_size, PRS_V2_N_THROW_PIECES, 30}, -1, opts_i32);
    auto hand_token_idx     = at::full({batch_size, 16}, -1, opts_i64);

    const HiveState* states_ptr = reinterpret_cast<const HiveState*>(
        states_tensor.data_ptr());
    const GPUMove* moves_ptr = reinterpret_cast<const GPUMove*>(
        legal_moves_tensor.data_ptr());
    const int* num_legal_ptr = static_cast<const int*>(num_legal_tensor.data_ptr());

    dim3 grid(batch_size);
    dim3 block(32);
    prs_v2_classify_kernel<<<grid, block>>>(
        states_ptr, moves_ptr, num_legal_ptr,
        batch_size, max_legal,
        static_cast<int64_t*>(dir_piece_idx.data_ptr()),
        static_cast<int64_t*>(throw_piece_idx.data_ptr()),
        static_cast<int64_t*>(long_piece_idx.data_ptr()),
        static_cast<int64_t*>(move_nbrs.data_ptr()),
        static_cast<int64_t*>(place_nbrs.data_ptr()),
        static_cast<bool*>(move_mask.data_ptr()),
        static_cast<bool*>(place_mask.data_ptr()),
        static_cast<int64_t*>(current_color.data_ptr()),
        static_cast<int32_t*>(slot_of_legal.data_ptr()),
        static_cast<int32_t*>(move_cell_ids.data_ptr()),
        static_cast<int32_t*>(place_cell_ids.data_ptr()),
        static_cast<int32_t*>(dir_dest_cell.data_ptr()),
        static_cast<int64_t*>(dir_dest_board_idx.data_ptr()),
        static_cast<int64_t*>(dir_dest_nbrs.data_ptr()),
        static_cast<int32_t*>(throw_dest_cell.data_ptr()),
        static_cast<int64_t*>(hand_token_idx.data_ptr())
    );
    // No explicit sync; PyTorch null-stream ordering serializes the next op.

    return std::make_tuple(
        dir_piece_idx, throw_piece_idx, long_piece_idx,
        move_nbrs, place_nbrs, move_mask, place_mask,
        current_color, slot_of_legal,
        move_cell_ids, place_cell_ids,
        dir_dest_cell, dir_dest_board_idx, dir_dest_nbrs, throw_dest_cell,
        hand_token_idx
    );
}

}  // namespace hive_gpu
