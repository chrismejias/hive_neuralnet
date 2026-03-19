/**
 * bindings.cpp — pybind11/torch bindings for hive_gpu CUDA extension.
 *
 * Exposes the following functions to Python:
 *   - initialize_tables(): Set up constant memory lookup tables
 *   - create_initial_states(batch_size): Allocate and init game states
 *   - generate_legal_moves_batch(states, batch_size): Get legal moves
 *   - apply_moves_batch(states, moves, batch_size): Apply moves
 *   - check_results_batch(states, batch_size): Get game results
 *   - encode_states_batch(states, batch_size): Encode states for NN input
 */

#include <torch/extension.h>
#include "hive_state.cuh"
#include "state_encoder.cuh"  // for encoder constants (kernel guarded by __CUDACC__)

namespace hive_gpu {

// Forward declarations (defined in game_logic.cu)
void initialize_tables();
torch::Tensor create_initial_states(int batch_size, int expansion_mask);
std::tuple<torch::Tensor, torch::Tensor> generate_legal_moves_batch(
    torch::Tensor states_tensor, int batch_size);
void apply_moves_batch(
    torch::Tensor states_tensor, torch::Tensor moves_tensor, int batch_size);
torch::Tensor check_results_batch(torch::Tensor states_tensor, int batch_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
encode_states_batch(torch::Tensor states_tensor, int batch_size);
std::tuple<torch::Tensor, torch::Tensor> generate_legal_mask_batch(
    torch::Tensor states_tensor, int batch_size);
std::tuple<torch::Tensor, torch::Tensor> compute_mobility_batch(
    torch::Tensor states_tensor, int batch_size, bool both_players);
torch::Tensor compute_centroids_batch(
    torch::Tensor states_tensor, int batch_size);

}  // namespace hive_gpu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Hive GPU: CUDA-accelerated game logic for Hive";

    m.def("initialize_tables", &hive_gpu::initialize_tables,
          "Initialize hex grid lookup tables in CUDA constant memory");

    m.def("create_initial_states", &hive_gpu::create_initial_states,
          "Allocate and initialize a batch of game states on GPU",
          py::arg("batch_size"), py::arg("expansion_mask") = 0);

    m.def("generate_legal_moves_batch", &hive_gpu::generate_legal_moves_batch,
          "Generate legal moves for a batch of states",
          py::arg("states"), py::arg("batch_size"));

    m.def("apply_moves_batch", &hive_gpu::apply_moves_batch,
          "Apply one move per game in the batch",
          py::arg("states"), py::arg("moves"), py::arg("batch_size"));

    m.def("check_results_batch", &hive_gpu::check_results_batch,
          "Get game results for all states in the batch",
          py::arg("states"), py::arg("batch_size"));

    m.def("encode_states_batch", &hive_gpu::encode_states_batch,
          "Encode batch of HiveStates into NN input features",
          py::arg("states"), py::arg("batch_size"));

    m.def("generate_legal_mask_batch", &hive_gpu::generate_legal_mask_batch,
          "Generate legal action masks (29407-dim) for a batch of states",
          py::arg("states"), py::arg("batch_size"));

    m.def("compute_mobility_batch", &hive_gpu::compute_mobility_batch,
          "Compute per-piece mobility targets for a batch of states",
          py::arg("states"), py::arg("batch_size"), py::arg("both_players") = false);

    m.def("compute_centroids_batch", &hive_gpu::compute_centroids_batch,
          "Compute per-state centroids on GPU",
          py::arg("states"), py::arg("batch_size"));

    // Export constants for Python use
    m.attr("BOARD_SIZE") = hive_gpu::BOARD_SIZE;
    m.attr("NUM_CELLS") = hive_gpu::NUM_CELLS;
    m.attr("MAX_LEGAL_MOVES") = hive_gpu::MAX_LEGAL_MOVES;
    m.attr("MAX_STACK") = hive_gpu::MAX_STACK;
    m.attr("SIZEOF_HIVE_STATE") = (int)sizeof(hive_gpu::HiveState);
    m.attr("SIZEOF_GPU_MOVE") = (int)sizeof(hive_gpu::GPUMove);
    m.attr("MAX_ENC_NODES") = hive_gpu::MAX_ENC_NODES;
    m.attr("MAX_ENC_EDGES") = hive_gpu::MAX_ENC_EDGES;
    m.attr("NODE_FEAT_DIM") = hive_gpu::NODE_FEAT_DIM;
    m.attr("EDGE_FEAT_DIM") = hive_gpu::EDGE_FEAT_DIM;
    m.attr("GLOBAL_FEAT_DIM") = hive_gpu::GLOBAL_FEAT_DIM;
    m.attr("ENC_GRID") = hive_gpu::ENC_GRID;
    m.attr("ACTION_SPACE_SIZE") = hive_gpu::ACTION_SPACE_SIZE;
    m.attr("PASS_ACTION_INDEX") = hive_gpu::PASS_ACTION_INDEX;
    m.attr("MOVEMENT_OFFSET") = hive_gpu::MOVEMENT_OFFSET;
}
