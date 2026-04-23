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
#include "fnn_features.cuh"   // for FNN_FEAT_DIM constant
#include "mcts_tree.cuh"     // for tree constants

namespace hive_gpu {

// Forward declarations (defined in game_logic.cu)
void initialize_tables();
void reset_movegen_profile();
torch::Tensor get_movegen_profile();
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> generate_legal_moves_and_mask_batch(
    torch::Tensor states_tensor, int batch_size);
std::tuple<torch::Tensor, torch::Tensor> compute_mobility_batch(
    torch::Tensor states_tensor, int batch_size, bool both_players);
torch::Tensor compute_centroids_batch(
    torch::Tensor states_tensor, int batch_size);
torch::Tensor legal_moves_to_actions_batch(
    torch::Tensor states_tensor,
    torch::Tensor legal_moves_tensor,
    torch::Tensor num_legal_tensor,
    int batch_size);
torch::Tensor extract_fnn_features_batch(
    torch::Tensor states_tensor,
    torch::Tensor legal_moves_tensor,
    torch::Tensor num_legal_tensor,
    int batch_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
generate_legal_moves_and_fnn_features_batch(
    torch::Tensor states_tensor,
    int batch_size);
torch::Tensor fnn_successor_features_batch(
    torch::Tensor states_tensor,
    torch::Tensor legal_moves_tensor,
    torch::Tensor action_to_root_tensor,
    torch::Tensor move_indices_tensor,
    int num_actions);

// GPU-native FNN self-play
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fnn_selfplay_batch(
    torch::Tensor weights,
    int hidden_dim, int embed_dim, int action_hidden,
    int batch_size, int max_game_length,
    int num_simulations, int max_considered,
    float c_visit, float c_scale,
    int temperature_drop_move, int expansion_mask,
    int64_t rng_seed);

// GPU-native MCTS
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mcts_select_batch(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor game_active, torch::Tensor root_nodes,
    float c_puct, int B, int W, int max_nodes);

void mcts_replay_batch(
    torch::Tensor root_states, torch::Tensor leaf_states,
    torch::Tensor move_paths, torch::Tensor path_lengths, torch::Tensor leaf_indices,
    int B, int total);

torch::Tensor mcts_expand_batch(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor leaf_indices, torch::Tensor leaf_states,
    torch::Tensor legal_moves, torch::Tensor num_legal,
    torch::Tensor action_probs, torch::Tensor results,
    int B, int total, int max_nodes);

void mcts_backprop_batch(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor leaf_indices, torch::Tensor nn_values,
    torch::Tensor vl_paths, torch::Tensor vl_lengths, torch::Tensor was_expanded,
    int B, int total, int max_nodes);

void mcts_expand_and_backprop_batch(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor leaf_indices, torch::Tensor leaf_states,
    torch::Tensor legal_moves, torch::Tensor num_legal,
    torch::Tensor action_probs, torch::Tensor results,
    torch::Tensor nn_values, torch::Tensor vl_paths, torch::Tensor vl_lengths,
    int B, int total, int max_nodes);

torch::Tensor mcts_extract_policy_batch(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor move_numbers, torch::Tensor root_nodes,
    float temperature, int temp_drop_move, float pruning_threshold,
    int B, int max_nodes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mcts_select_with_root_mask_batch(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor game_active, torch::Tensor root_nodes,
    torch::Tensor alive_mask, int max_root_children,
    float c_puct, int B, int W, int max_nodes);

torch::Tensor mcts_expand_dense_priors_batch(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor leaf_indices, torch::Tensor leaf_states,
    torch::Tensor legal_moves, torch::Tensor num_legal,
    torch::Tensor priors_per_legal, torch::Tensor results,
    int B, int total, int max_nodes);

void mcts_expand_and_backprop_dense_priors_batch(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor leaf_indices, torch::Tensor leaf_states,
    torch::Tensor legal_moves, torch::Tensor num_legal,
    torch::Tensor priors_per_legal, torch::Tensor results,
    torch::Tensor nn_values, torch::Tensor vl_paths, torch::Tensor vl_lengths,
    int B, int total, int max_nodes);

void mcts_apply_root_noise(
    torch::Tensor vc, torch::Tensor tv, torch::Tensor pr, torch::Tensor pa,
    torch::Tensor mb, torch::Tensor ai, torch::Tensor fc, torch::Tensor nc,
    torch::Tensor it, torch::Tensor tv2, torch::Tensor cnt,
    torch::Tensor noise, torch::Tensor root_nodes, int max_children_pad,
    float dir_eps, float root_policy_temp,
    int B, int max_nodes);

// PRS v2: slot mapping + head-input bridge (16-tuple of GPU tensors)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
prs_v2_classify_batch(
    torch::Tensor states_tensor,
    torch::Tensor legal_moves_tensor,
    torch::Tensor num_legal_tensor,
    int batch_size, int max_legal);

}  // namespace hive_gpu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Hive GPU: CUDA-accelerated game logic for Hive";

    m.def("initialize_tables", &hive_gpu::initialize_tables,
          "Initialize hex grid lookup tables in CUDA constant memory");
    m.def("reset_movegen_profile", &hive_gpu::reset_movegen_profile,
          "Reset CUDA move generation profile counters");
    m.def("get_movegen_profile", &hive_gpu::get_movegen_profile,
          "Read CUDA move generation profile counters");

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

    m.def("generate_legal_moves_and_mask_batch", &hive_gpu::generate_legal_moves_and_mask_batch,
          "Generate legal moves and action masks in one fused kernel pass",
          py::arg("states"), py::arg("batch_size"));

    m.def("compute_mobility_batch", &hive_gpu::compute_mobility_batch,
          "Compute per-piece mobility targets for a batch of states",
          py::arg("states"), py::arg("batch_size"), py::arg("both_players") = false);

    m.def("compute_centroids_batch", &hive_gpu::compute_centroids_batch,
          "Compute per-state centroids on GPU",
          py::arg("states"), py::arg("batch_size"));

    m.def("legal_moves_to_actions_batch", &hive_gpu::legal_moves_to_actions_batch,
          "Map generated legal moves to action indices on GPU",
          py::arg("states"), py::arg("legal_moves"), py::arg("num_legal"),
          py::arg("batch_size"));

    m.def("extract_fnn_features_batch", &hive_gpu::extract_fnn_features_batch,
          "Extract FNN board features directly from HiveState + legal moves",
          py::arg("states"), py::arg("legal_moves"), py::arg("num_legal"),
          py::arg("batch_size"));

    m.def("generate_legal_moves_and_fnn_features_batch",
          &hive_gpu::generate_legal_moves_and_fnn_features_batch,
          "Generate legal moves and extract FNN features in one kernel",
          py::arg("states"), py::arg("batch_size"));

    m.def("fnn_successor_features_batch",
          &hive_gpu::fnn_successor_features_batch,
          "Apply legal moves locally and extract only FNN successor features",
          py::arg("states"), py::arg("legal_moves"),
          py::arg("action_to_root"), py::arg("move_indices"),
          py::arg("num_actions"));

    // ── GPU-native FNN self-play ────────────────────────────────────
    m.def("fnn_selfplay_batch", &hive_gpu::fnn_selfplay_batch,
          "GPU-native Gumbel AlphaZero self-play with FNN",
          py::arg("weights"), py::arg("hidden_dim"), py::arg("embed_dim"),
          py::arg("action_hidden"), py::arg("batch_size"),
          py::arg("max_game_length"), py::arg("num_simulations"),
          py::arg("max_considered"), py::arg("c_visit"), py::arg("c_scale"),
          py::arg("temperature_drop_move"), py::arg("expansion_mask"),
          py::arg("rng_seed"));

    // ── GPU-native MCTS ────────────────────────────────────────────
    m.def("mcts_select_batch", &hive_gpu::mcts_select_batch,
          "MCTS PUCT selection (GPU-native) — root_nodes enables tree reuse");
    m.def("mcts_replay_batch", &hive_gpu::mcts_replay_batch,
          "MCTS move-path replay to compute leaf states");
    m.def("mcts_expand_batch", &hive_gpu::mcts_expand_batch,
          "MCTS expand non-terminal unexpanded leaves");
    m.def("mcts_backprop_batch", &hive_gpu::mcts_backprop_batch,
          "MCTS backpropagate values from leaf to root");
    m.def("mcts_expand_and_backprop_batch", &hive_gpu::mcts_expand_and_backprop_batch,
          "Fused MCTS expand + backprop in one kernel launch");
    m.def("mcts_extract_policy_batch", &hive_gpu::mcts_extract_policy_batch,
          "Extract MCTS policy from root visit counts");
    m.def("mcts_apply_root_noise", &hive_gpu::mcts_apply_root_noise,
          "Apply root policy temperature + Dirichlet noise");
    m.def("mcts_select_with_root_mask_batch", &hive_gpu::mcts_select_with_root_mask_batch,
          "MCTS PUCT selection with root alive-mask (Gumbel Sequential Halving)");
    m.def("mcts_expand_dense_priors_batch", &hive_gpu::mcts_expand_dense_priors_batch,
          "MCTS expand with per-legal-move priors (no ACTION_SPACE indirection)");
    m.def("mcts_expand_and_backprop_dense_priors_batch",
          &hive_gpu::mcts_expand_and_backprop_dense_priors_batch,
          "Fused MCTS expand + backprop with per-legal-move priors");

    // ── PRS v2: slot mapping + head-input bridge ────────────────────
    m.def("prs_v2_classify_batch", &hive_gpu::prs_v2_classify_batch,
          "PRS v2: classify legal moves into 813 slots and build head inputs",
          py::arg("states"), py::arg("legal_moves"), py::arg("num_legal"),
          py::arg("batch_size"), py::arg("max_legal"));

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
    m.attr("DEFAULT_MAX_TREE_NODES") = hive_gpu::DEFAULT_MAX_TREE_NODES;
    m.attr("MAX_TREE_DEPTH") = hive_gpu::MAX_TREE_DEPTH;
    m.attr("FNN_FEAT_DIM") = hive_gpu::FNN_FEAT_DIM;
}
