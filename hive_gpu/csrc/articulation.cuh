/**
 * articulation.cuh — Articulation point detection on GPU.
 *
 * Finds pinned pieces using iterative Tarjan's algorithm on the compact
 * subgraph of occupied cells (max 22 vertices, 66 edges).
 *
 * Returns a Bitboard where set bits indicate articulation points
 * (cells whose removal would disconnect the hive).
 *
 * Pieces at articulation points with stack_height == 1 are pinned and
 * cannot move (One Hive rule). Pieces with height >= 2 at AP cells
 * can still move their top piece (beetle), since removing the top
 * doesn't disconnect the ground-level graph.
 */

#pragma once

#include "hive_state.cuh"
#include "hex_grid.cuh"

namespace hive_gpu {

#ifdef __CUDACC__

/**
 * Find all articulation points in the hive graph.
 *
 * Uses iterative Tarjan's DFS entirely in local memory.
 * V ≤ 22, E ≤ 66, so all arrays fit in registers/local memory.
 *
 * @param s  The current game state
 * @return   Bitboard with bits set at articulation point cells
 */
__device__ inline Bitboard find_articulation_points(const HiveState& s) {
    Bitboard ap_mask;
    ap_mask.clear();

    int num_occ = s.occupied.popcount();
    if (num_occ <= 2) return ap_mask;  // No APs possible with ≤ 2 pieces

    // ── Build compact graph ───────────────────────────────────────
    // Map occupied cells to dense indices 0..N-1
    // Note: we use a simple scan of the bitboard words since MAX_PIECES=22
    // is small. We don't need cell_to_idx for all 289 cells.
    int idx_to_cell[MAX_PIECES];
    int N = 0;

    for (int wi = 0; wi < BB_WORDS; wi++) {
        uint64_t bits = s.occupied.w[wi];
        while (bits) {
            int bit = __ffsll(bits) - 1;  // find first set bit (0-indexed)
            int cell = wi * 64 + bit;
            if (cell < NUM_CELLS && N < MAX_PIECES) {
                idx_to_cell[N] = cell;
                N++;
            }
            bits &= bits - 1;  // clear lowest set bit
        }
    }

    if (N <= 2) return ap_mask;

    // Build cell → dense index mapping (only for occupied cells)
    // We use a small hash or linear scan since N ≤ 22
    // For correctness, just do linear search (N is tiny)
    auto cell_to_idx = [&](int cell) -> int {
        for (int i = 0; i < N; i++) {
            if (idx_to_cell[i] == cell) return i;
        }
        return -1;
    };

    // Build adjacency list (compressed)
    uint8_t adj_count[MAX_PIECES];
    uint8_t adj_list[MAX_PIECES * 6];  // max 6 neighbors per vertex
    int adj_start[MAX_PIECES];

    int total_edges = 0;
    for (int i = 0; i < N; i++) {
        adj_start[i] = total_edges;
        adj_count[i] = 0;
        int cell = idx_to_cell[i];
        for (int d = 0; d < NUM_DIRS; d++) {
            int16_t nb = NEIGHBORS[cell][d];
            if (nb >= 0 && s.occupied.get(nb)) {
                int nb_idx = cell_to_idx(nb);
                if (nb_idx >= 0) {
                    adj_list[total_edges] = (uint8_t)nb_idx;
                    total_edges++;
                    adj_count[i]++;
                }
            }
        }
    }

    // ── Iterative Tarjan's DFS ────────────────────────────────────
    int disc[MAX_PIECES];
    int low[MAX_PIECES];
    int parent[MAX_PIECES];
    uint8_t child_count[MAX_PIECES];
    uint8_t stack_ni[MAX_PIECES];  // neighbor iterator index per stack frame

    for (int i = 0; i < N; i++) {
        disc[i] = -1;  // unvisited
        parent[i] = -1;
        child_count[i] = 0;
    }

    int timer = 0;

    // Start DFS from vertex 0
    int start = 0;
    disc[start] = low[start] = timer++;
    stack_ni[0] = 0;

    // DFS stack: stack[k] = vertex being processed
    int dfs_stack[MAX_PIECES];
    dfs_stack[0] = start;
    int stack_top = 1;

    while (stack_top > 0) {
        int u = dfs_stack[stack_top - 1];
        uint8_t ni = stack_ni[stack_top - 1];

        if (ni < adj_count[u]) {
            stack_ni[stack_top - 1] = ni + 1;
            int v = adj_list[adj_start[u] + ni];

            if (disc[v] == -1) {
                // Tree edge: u → v
                parent[v] = u;
                child_count[u]++;
                disc[v] = low[v] = timer++;
                stack_ni[stack_top] = 0;
                dfs_stack[stack_top] = v;
                stack_top++;
            } else if (v != parent[u]) {
                // Back edge
                if (disc[v] < low[u]) {
                    low[u] = disc[v];
                }
            }
        } else {
            // Done with u, pop
            stack_top--;
            if (stack_top > 0) {
                int p = dfs_stack[stack_top - 1];
                if (low[u] < low[p]) {
                    low[p] = low[u];
                }
                // Check AP conditions
                if (parent[p] == -1) {
                    // Root: AP if >1 children in DFS tree
                    if (child_count[p] > 1) {
                        ap_mask.set(idx_to_cell[p]);
                    }
                } else {
                    // Non-root: AP if low[u] >= disc[p]
                    if (low[u] >= disc[p]) {
                        ap_mask.set(idx_to_cell[p]);
                    }
                }
            }
        }
    }

    return ap_mask;
}

/**
 * Check if a piece at `cell` is pinned (articulation point + stack_height == 1).
 * Pieces on top of stacks at AP cells can still move because removing the top
 * piece doesn't disconnect the ground-level hive graph.
 */
__device__ __forceinline__ bool is_pinned(const HiveState& s,
                                           const Bitboard& ap_mask,
                                           int cell) {
    return ap_mask.get(cell) && s.height[cell] == 1;
}

#endif  // __CUDACC__

}  // namespace hive_gpu
