/**
 * mcts_tree.cuh — GPU-native MCTS tree data structure and kernels.
 *
 * Provides a flat Structure-of-Arrays tree layout on GPU with kernels for:
 *   - PUCT selection with virtual loss
 *   - Move path replay to compute leaf states
 *   - Node expansion from NN policy
 *   - Value backpropagation with alternating signs
 *   - Policy extraction from visit counts
 *   - Root noise/temperature application
 *
 * Tree layout: each game has MAX_TREE_NODES node slots. Node 0 = root.
 * Children are allocated contiguously: node.first_child to
 * node.first_child + node.num_children - 1.
 */

#pragma once

#include "hive_state.cuh"
#include "state_encoder.cuh"

namespace hive_gpu {

// ── MCTS Tree constants ───────────────────────────────────────────

constexpr int DEFAULT_MAX_TREE_NODES = 32768;
constexpr int MAX_TREE_DEPTH = 128;

// ── GPU MCTS Tree structure (SoA) ─────────────────────────────────

struct MCTSTree {
    int*     visit_count;      // [B, max_nodes]
    float*   total_value;      // [B, max_nodes]
    float*   prior;            // [B, max_nodes]
    int*     parent_idx;       // [B, max_nodes]  (-1 = no parent)
    uint8_t* move_bytes;       // [B, max_nodes, sizeof(GPUMove)]
    int*     action_idx;       // [B, max_nodes]  (-1 = root/none)
    int*     first_child;      // [B, max_nodes]  (-1 = unexpanded, -2 = expanding)
    int*     num_children;     // [B, max_nodes]
    int8_t*  is_terminal;      // [B, max_nodes]
    float*   terminal_value;   // [B, max_nodes]
    int*     node_count;       // [B]  (atomic allocator, starts at 1 for root)
    int      max_nodes;
};

#ifdef __CUDACC__

// ── Inline helpers ────────────────────────────────────────────────

__device__ __forceinline__ int tree_idx(const MCTSTree& t, int game, int node) {
    return game * t.max_nodes + node;
}

__device__ __forceinline__ float puct_score(
    const MCTSTree& t, int game, int parent_node, int child_node, float c_puct
) {
    int ci = tree_idx(t, game, child_node);
    int pi = tree_idx(t, game, parent_node);
    int child_visits = t.visit_count[ci];
    float q = (child_visits > 0) ? (-t.total_value[ci] / (float)child_visits) : 0.0f;
    float exploration = c_puct * t.prior[ci]
                      * sqrtf((float)t.visit_count[pi])
                      / (1.0f + (float)child_visits);
    return q + exploration;
}

// Map a GPUMove to an action index (same logic as legal_mask_kernel).
__device__ __forceinline__ int gpu_move_to_action(
    const GPUMove& mv, int center_q, int center_r
) {
    if (mv.type == MOVE_PASS) return PASS_ACTION_INDEX;

    if (mv.type == MOVE_PLACE) {
        int to_q = mv.to_cell % BOARD_SIZE - HALF_BOARD;
        int to_r = mv.to_cell / BOARD_SIZE - HALF_BOARD;
        int enc_col = to_q - center_q + ENC_HALF;
        int enc_row = to_r - center_r + ENC_HALF;
        if (enc_col < 0 || enc_col >= ENC_GRID ||
            enc_row < 0 || enc_row >= ENC_GRID)
            return -1;
        int pt_0idx = (int)mv.piece_type - 1;
        int pos_idx = enc_row * ENC_GRID + enc_col;
        return pt_0idx * NUM_ENC_GRID_CELLS + pos_idx;
    }

    // MOVE_MOVE
    int from_q = mv.from_cell % BOARD_SIZE - HALF_BOARD;
    int from_r = mv.from_cell / BOARD_SIZE - HALF_BOARD;
    int to_q   = mv.to_cell   % BOARD_SIZE - HALF_BOARD;
    int to_r   = mv.to_cell   / BOARD_SIZE - HALF_BOARD;
    int sc = from_q - center_q + ENC_HALF;
    int sr = from_r - center_r + ENC_HALF;
    int dc = to_q   - center_q + ENC_HALF;
    int dr = to_r   - center_r + ENC_HALF;
    if (sc < 0 || sc >= ENC_GRID || sr < 0 || sr >= ENC_GRID ||
        dc < 0 || dc >= ENC_GRID || dr < 0 || dr >= ENC_GRID)
        return -1;
    int src = sr * ENC_GRID + sc;
    int dst = dr * ENC_GRID + dc;
    return MOVEMENT_OFFSET + src * NUM_ENC_GRID_CELLS + dst;
}

// Terminal value from game result + turn.
__device__ __forceinline__ float result_to_value(int result, uint16_t turn) {
    if (result == DRAW)       return 0.0f;
    int player = turn & 1;   // 0=white, 1=black
    if (result == WHITE_WINS) return (player == 0) ?  1.0f : -1.0f;
    if (result == BLACK_WINS) return (player == 0) ? -1.0f :  1.0f;
    return 0.0f;
}

// ── Kernels ──────────────────────────────────────────────────────

/**
 * SELECT kernel — one thread per simulation (total_sims = W * B).
 *
 * Walks from root to an unexpanded/terminal leaf using PUCT.
 * Applies virtual loss on each visited node below the root.
 *
 * Outputs:
 *   leaf_indices  — node index of the leaf
 *   move_paths    — GPUMove bytes along the path (for state replay)
 *   path_lengths  — depth of the leaf from root
 *   vl_paths      — node indices visited (for virtual loss undo)
 *   vl_lengths    — same as path_lengths
 */
__global__ void mcts_select_kernel(
    MCTSTree tree,
    int*      leaf_indices,    // [total_sims]
    uint8_t*  move_paths,      // [total_sims, MAX_TREE_DEPTH, sizeof(GPUMove)]
    int*      path_lengths,    // [total_sims]
    int*      vl_paths,        // [total_sims, MAX_TREE_DEPTH]
    int*      vl_lengths,      // [total_sims]
    const int8_t* game_active, // [B]
    float c_puct,
    int B, int total_sims
) {
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= total_sims) return;

    int game = sim_idx % B;

    if (!game_active[game]) {
        leaf_indices[sim_idx] = -1;
        path_lengths[sim_idx] = 0;
        vl_lengths[sim_idx]   = 0;
        return;
    }

    constexpr int MOVE_SZ = (int)sizeof(GPUMove);
    GPUMove* my_moves = reinterpret_cast<GPUMove*>(
        move_paths + (int64_t)sim_idx * MAX_TREE_DEPTH * MOVE_SZ);
    int* my_vl = vl_paths + (int64_t)sim_idx * MAX_TREE_DEPTH;

    int node = 0;   // start at root
    int path_len = 0;

    while (true) {
        int ni = tree_idx(tree, game, node);

        // Stop at unexpanded, terminal, or childless nodes
        int fc = tree.first_child[ni];
        if (fc < 0 || tree.is_terminal[ni] || tree.num_children[ni] == 0)
            break;

        // PUCT: pick child with highest score
        int nc = tree.num_children[ni];
        int best_child = fc;
        float best_score = -1e30f;
        for (int c = 0; c < nc; c++) {
            float score = puct_score(tree, game, node, fc + c, c_puct);
            if (score > best_score) {
                best_score = score;
                best_child = fc + c;
            }
        }

        node = best_child;

        // Virtual loss
        int ni2 = tree_idx(tree, game, node);
        atomicAdd(&tree.visit_count[ni2], 1);
        atomicAdd(&tree.total_value[ni2], -1.0f);

        // Record path
        if (path_len < MAX_TREE_DEPTH) {
            my_vl[path_len] = node;
            // Copy move bytes from tree
            int64_t mb_base = ((int64_t)game * tree.max_nodes + node) * MOVE_SZ;
            uint8_t* dst = reinterpret_cast<uint8_t*>(&my_moves[path_len]);
            for (int b = 0; b < MOVE_SZ; b++)
                dst[b] = tree.move_bytes[mb_base + b];
            path_len++;
        }
    }

    leaf_indices[sim_idx] = node;
    path_lengths[sim_idx] = path_len;
    vl_lengths[sim_idx]   = path_len;
}

/**
 * REPLAY kernel — replay move paths to compute leaf states.
 *
 * One thread per simulation. Copies root state then applies moves sequentially.
 */
__global__ void mcts_replay_kernel(
    const HiveState* root_states,  // [B]
    HiveState*       leaf_states,  // [total_sims]
    const uint8_t*   move_paths,   // [total_sims, MAX_TREE_DEPTH, sizeof(GPUMove)]
    const int*       path_lengths, // [total_sims]
    const int*       leaf_indices, // [total_sims]  (-1 = skip)
    int B, int total_sims
) {
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= total_sims) return;
    if (leaf_indices[sim_idx] < 0) return;

    int game = sim_idx % B;
    leaf_states[sim_idx] = root_states[game];

    constexpr int MOVE_SZ = (int)sizeof(GPUMove);
    const GPUMove* my_moves = reinterpret_cast<const GPUMove*>(
        move_paths + (int64_t)sim_idx * MAX_TREE_DEPTH * MOVE_SZ);

    int depth = path_lengths[sim_idx];
    for (int d = 0; d < depth; d++)
        apply_move(leaf_states[sim_idx], my_moves[d]);
}

/**
 * EXPAND kernel — create children for non-terminal, unexpanded leaves.
 *
 * One thread per simulation. Uses atomicCAS on first_child to ensure
 * only one thread expands a given node.
 *
 * Outputs was_expanded[sim_idx] = 1 if this thread performed the expansion.
 */
__global__ void mcts_expand_kernel(
    MCTSTree          tree,
    const int*        leaf_indices,   // [total_sims]
    const HiveState*  leaf_states,    // [total_sims]
    const GPUMove*    legal_moves,    // [total_sims, MAX_LEGAL_MOVES]
    const int*        num_legal,      // [total_sims]
    const float*      action_probs,   // [total_sims, ACTION_SPACE_SIZE]
    const int*        results,        // [total_sims]
    int8_t*           was_expanded,   // [total_sims] output
    int B, int total_sims
) {
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= total_sims) return;

    int leaf = leaf_indices[sim_idx];
    if (leaf < 0) { was_expanded[sim_idx] = 0; return; }

    int game = sim_idx % B;
    int li   = tree_idx(tree, game, leaf);

    // Terminal?
    if (results[sim_idx] != 0) {
        tree.is_terminal[li]    = 1;
        tree.terminal_value[li] = result_to_value(
            results[sim_idx], leaf_states[sim_idx].turn);
        was_expanded[sim_idx] = 0;
        return;
    }

    // Atomic guard: only one thread expands this node
    int old_fc = atomicCAS(&tree.first_child[li], -1, -2);
    if (old_fc != -1) { was_expanded[sim_idx] = 0; return; }

    int n_legal = num_legal[sim_idx];
    if (n_legal <= 0) {
        tree.first_child[li] = -1;   // revert
        was_expanded[sim_idx] = 0;
        return;
    }

    // Compute centroid for action-index mapping
    int cq, cr;
    compute_centroid(leaf_states[sim_idx], cq, cr);

    // Allocate children
    int base = atomicAdd(&tree.node_count[game], n_legal);
    if (base + n_legal > tree.max_nodes) {
        atomicAdd(&tree.node_count[game], -n_legal);
        tree.first_child[li] = -1;
        was_expanded[sim_idx] = 0;
        return;
    }

    const GPUMove* my_moves = legal_moves + (int64_t)sim_idx * MAX_LEGAL_MOVES;
    const float*   probs    = action_probs + (int64_t)sim_idx * ACTION_SPACE_SIZE;
    constexpr int  MOVE_SZ  = (int)sizeof(GPUMove);

    int created = 0;
    for (int m = 0; m < n_legal; m++) {
        const GPUMove& mv = my_moves[m];
        int action = gpu_move_to_action(mv, cq, cr);
        if (action < 0) continue;

        int child_node = base + created;
        int ci = tree_idx(tree, game, child_node);

        tree.visit_count[ci]    = 0;
        tree.total_value[ci]    = 0.0f;
        tree.prior[ci]          = probs[action];
        tree.parent_idx[ci]     = leaf;
        tree.action_idx[ci]     = action;
        tree.first_child[ci]    = -1;
        tree.num_children[ci]   = 0;
        tree.is_terminal[ci]    = 0;
        tree.terminal_value[ci] = 0.0f;

        // Copy move bytes
        int64_t mb_base = ((int64_t)game * tree.max_nodes + child_node) * MOVE_SZ;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(&mv);
        for (int b = 0; b < MOVE_SZ; b++)
            tree.move_bytes[mb_base + b] = src[b];

        created++;
    }

    tree.num_children[li] = created;

    // Release unused node slots
    if (created < n_legal)
        atomicAdd(&tree.node_count[game], -(n_legal - created));

    // Ensure children visible before publishing first_child
    __threadfence();
    tree.first_child[li] = base;

    was_expanded[sim_idx] = 1;
}

/**
 * BACKPROP kernel — undo virtual loss, propagate value from leaf to root.
 *
 * Value alternates sign at each tree level (two-player zero-sum).
 */
__global__ void mcts_backprop_kernel(
    MCTSTree       tree,
    const int*     leaf_indices,   // [total_sims]
    const float*   nn_values,      // [total_sims]
    const int*     vl_paths,       // [total_sims, MAX_TREE_DEPTH]
    const int*     vl_lengths,     // [total_sims]
    const int8_t*  was_expanded,   // [total_sims]
    int B, int total_sims
) {
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= total_sims) return;

    int leaf = leaf_indices[sim_idx];
    if (leaf < 0) return;

    int game = sim_idx % B;
    int li   = tree_idx(tree, game, leaf);

    // Determine value for this leaf
    float value;
    if (tree.is_terminal[li])
        value = tree.terminal_value[li];
    else if (was_expanded[sim_idx])
        value = nn_values[sim_idx];
    else
        value = 0.0f;

    // Undo virtual loss
    int vl_len     = vl_lengths[sim_idx];
    const int* vl  = vl_paths + (int64_t)sim_idx * MAX_TREE_DEPTH;
    for (int k = 0; k < vl_len; k++) {
        int ni = tree_idx(tree, game, vl[k]);
        atomicAdd(&tree.visit_count[ni], -1);
        atomicAdd(&tree.total_value[ni],  1.0f);
    }

    // Normal backpropagation with alternating signs
    int node = leaf;
    float v  = value;
    while (node >= 0) {
        int ni = tree_idx(tree, game, node);
        atomicAdd(&tree.visit_count[ni], 1);
        atomicAdd(&tree.total_value[ni], v);
        v    = -v;
        node = tree.parent_idx[ni];
    }
}

/**
 * EXTRACT POLICY kernel — build [B, ACTION_SPACE_SIZE] from root visit counts.
 */
__global__ void mcts_extract_policy_kernel(
    MCTSTree    tree,
    float*      policies,          // [B, ACTION_SPACE_SIZE]
    const int*  move_numbers,      // [B]
    float temperature,
    int   temp_drop_move,
    float pruning_threshold,
    int B
) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= B) return;

    float* policy   = policies + (int64_t)game * ACTION_SPACE_SIZE;
    int    root_idx = tree_idx(tree, game, 0);
    int    fc       = tree.first_child[root_idx];
    int    nc       = tree.num_children[root_idx];
    if (fc < 0 || nc == 0) return;

    float temp = temperature;
    if (move_numbers[game] >= temp_drop_move) temp = 0.0f;

    // Greedy (temp=0)
    if (temp <= 0.0f) {
        int best_action = -1;
        int best_visits = -1;
        for (int c = 0; c < nc; c++) {
            int ci = tree_idx(tree, game, fc + c);
            int v  = tree.visit_count[ci];
            if (v > best_visits) { best_visits = v; best_action = tree.action_idx[ci]; }
        }
        if (best_action >= 0 && best_action < ACTION_SPACE_SIZE)
            policy[best_action] = 1.0f;
        return;
    }

    // Find max visits for pruning
    float max_v = 0.0f;
    for (int c = 0; c < nc; c++) {
        float v = (float)tree.visit_count[tree_idx(tree, game, fc + c)];
        if (v > max_v) max_v = v;
    }

    float sum = 0.0f;
    for (int c = 0; c < nc; c++) {
        int ci  = tree_idx(tree, game, fc + c);
        float v = (float)tree.visit_count[ci];
        if (pruning_threshold > 0.0f && max_v > 0.0f && v < max_v * pruning_threshold)
            v = 0.0f;
        if (temp != 1.0f) v = powf(v, 1.0f / temp);
        int action = tree.action_idx[ci];
        if (action >= 0 && action < ACTION_SPACE_SIZE) {
            policy[action] = v;
            sum += v;
        }
    }

    // Normalise
    if (sum > 0.0f) {
        for (int c = 0; c < nc; c++) {
            int action = tree.action_idx[tree_idx(tree, game, fc + c)];
            if (action >= 0 && action < ACTION_SPACE_SIZE)
                policy[action] /= sum;
        }
    } else {
        float uniform = 1.0f / (float)nc;
        for (int c = 0; c < nc; c++) {
            int action = tree.action_idx[tree_idx(tree, game, fc + c)];
            if (action >= 0 && action < ACTION_SPACE_SIZE)
                policy[action] = uniform;
        }
    }
}

/**
 * ROOT NOISE kernel — apply root policy temperature + Dirichlet noise.
 */
__global__ void mcts_root_noise_kernel(
    MCTSTree     tree,
    const float* dirichlet_noise,  // [B, max_children_pad]
    int   max_children_pad,
    float dir_eps,
    float root_policy_temp,
    int B
) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= B) return;

    int root_idx = tree_idx(tree, game, 0);
    int fc = tree.first_child[root_idx];
    int nc = tree.num_children[root_idx];
    if (fc < 0 || nc == 0) return;

    const float* noise = dirichlet_noise + (int64_t)game * max_children_pad;

    // Root policy temperature: prior^(1/T), renormalise
    if (root_policy_temp > 1.0f) {
        float inv_t = 1.0f / root_policy_temp;
        float sum   = 0.0f;
        for (int c = 0; c < nc; c++) {
            int ci   = tree_idx(tree, game, fc + c);
            float p  = powf(tree.prior[ci], inv_t);
            tree.prior[ci] = p;
            sum += p;
        }
        if (sum > 0.0f) {
            for (int c = 0; c < nc; c++)
                tree.prior[tree_idx(tree, game, fc + c)] /= sum;
        }
    }

    // Dirichlet noise
    if (dir_eps > 0.0f) {
        for (int c = 0; c < nc; c++) {
            int ci = tree_idx(tree, game, fc + c);
            tree.prior[ci] = (1.0f - dir_eps) * tree.prior[ci] + dir_eps * noise[c];
        }
    }
}

#endif  // __CUDACC__

}  // namespace hive_gpu
