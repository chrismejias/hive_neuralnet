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

namespace hive_gpu {

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
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();

    // Step 2: Map moves to action mask
    auto masks = at::zeros({batch_size, (int64_t)ACTION_SPACE_SIZE}, opts_f);
    float* masks_ptr = static_cast<float*>(masks.data_ptr());

    legal_mask_kernel<<<blocks, threads>>>(
        states_ptr, moves_ptr, num_legal_ptr, masks_ptr, batch_size);
    cudaDeviceSynchronize();

    return std::make_tuple(masks, num_legal);
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

}  // namespace hive_gpu
