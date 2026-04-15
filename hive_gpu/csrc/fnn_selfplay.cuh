/**
 * fnn_selfplay.cuh — GPU-native Gumbel AlphaZero self-play for FNN.
 *
 * Fuses the entire self-play loop into a single kernel launch:
 *   - Legal move generation
 *   - Feature extraction (88-dim)
 *   - FNN forward pass (encode + value + action scoring)
 *   - Gumbel sequential halving
 *   - Move selection + application
 *   - Training data recording
 *
 * One thread-block per game, SELFPLAY_BLOCK_SIZE threads per block.
 * Thread 0 handles root evaluation + Gumbel + move application.
 * All threads cooperate on parallel successor evaluation.
 *
 * Must be included from game_logic.cu (needs constant memory + device functions).
 */

#pragma once

#include "move_gen.cuh"
#include "fnn_features.cuh"

namespace hive_gpu {

// ── Configuration ────────────────────────────────────────────────────

constexpr int SELFPLAY_BLOCK_SIZE = 32;
constexpr int FNN_MAX_HIDDEN = 64;
constexpr int FNN_MAX_EMBED = 64;
constexpr int FNN_MAX_ACTION_HIDDEN = 64;
constexpr int MAX_GUMBEL_CANDIDATES = 64;

// ── RNG (xorshift64) ────────────────────────────────────────────────

struct SelfPlayRNG {
    uint64_t s;

    __device__ void seed(uint64_t base, int game_idx) {
        s = base ^ ((uint64_t)game_idx * 6364136223846793005ULL + 1442695040888963407ULL);
        next(); next(); next();  // warm up
    }

    __device__ uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }

    __device__ float uniform() {
        uint32_t v = (uint32_t)(next() >> 16);
        return (float)(v & 0xFFFFFF) / 16777216.0f;
    }

    __device__ float gumbel() {
        float u = uniform();
        u = fmaxf(u, 1e-6f);
        u = fminf(u, 1.0f - 1e-6f);
        return -logf(-logf(u));
    }
};

// ── FNN weight layout ────────────────────────────────────────────────

struct FNNWeights {
    int H, E, AH;           // hidden, embed, action_hidden dims
    int fc1_w, fc1_b;       // Linear(88, H)
    int ln1_w, ln1_b;       // LayerNorm(H)
    int fc2_w, fc2_b;       // Linear(H, E)
    int val_w, val_b;       // Linear(E, 1)
    int act1_w, act1_b;     // Linear(2E, AH)
    int act2_w, act2_b;     // Linear(AH, 1)
};

__device__ FNNWeights make_fnn_weights(int H, int E, int AH) {
    FNNWeights w;
    w.H = H; w.E = E; w.AH = AH;
    int o = 0;
    w.fc1_w = o; o += FNN_FEAT_DIM * H;
    w.fc1_b = o; o += H;
    w.ln1_w = o; o += H;
    w.ln1_b = o; o += H;
    w.fc2_w = o; o += H * E;
    w.fc2_b = o; o += E;
    w.val_w = o; o += E;
    w.val_b = o; o += 1;
    w.act1_w = o; o += 2 * E * AH;
    w.act1_b = o; o += AH;
    w.act2_w = o; o += AH;
    w.act2_b = o; o += 1;
    return w;
}

// ── FNN device functions ─────────────────────────────────────────────

__device__ inline void fnn_linear(
    const float* __restrict__ W, const float* __restrict__ b,
    const float* x, float* y,
    int in_dim, int out_dim
) {
    for (int o = 0; o < out_dim; o++) {
        float sum = b[o];
        for (int i = 0; i < in_dim; i++) {
            sum += W[o * in_dim + i] * x[i];
        }
        y[o] = sum;
    }
}

__device__ inline void fnn_encode(
    const float* features,               // [FNN_FEAT_DIM]
    float* embed,                         // [E]
    const float* __restrict__ params,
    const FNNWeights& w
) {
    float hidden[FNN_MAX_HIDDEN];

    // fc1 + sigmoid
    fnn_linear(params + w.fc1_w, params + w.fc1_b,
               features, hidden, FNN_FEAT_DIM, w.H);
    for (int i = 0; i < w.H; i++)
        hidden[i] = 1.0f / (1.0f + expf(-hidden[i]));

    // LayerNorm
    float mean = 0.0f;
    for (int i = 0; i < w.H; i++) mean += hidden[i];
    mean /= (float)w.H;

    float var = 0.0f;
    for (int i = 0; i < w.H; i++) {
        float d = hidden[i] - mean;
        var += d * d;
    }
    var /= (float)w.H;
    float inv_std = rsqrtf(var + 1e-5f);

    for (int i = 0; i < w.H; i++)
        hidden[i] = (hidden[i] - mean) * inv_std
                     * params[w.ln1_w + i] + params[w.ln1_b + i];

    // fc2 (no activation)
    fnn_linear(params + w.fc2_w, params + w.fc2_b,
               hidden, embed, w.H, w.E);
}

__device__ inline float fnn_value(
    const float* embed,
    const float* __restrict__ params,
    const FNNWeights& w
) {
    float sum = params[w.val_b];
    for (int i = 0; i < w.E; i++)
        sum += params[w.val_w + i] * embed[i];
    return tanhf(sum);
}

__device__ inline float fnn_score_action(
    const float* root_embed,
    const float* succ_embed,
    const float* __restrict__ params,
    const FNNWeights& w
) {
    float combined[FNN_MAX_EMBED * 2];
    for (int i = 0; i < w.E; i++) combined[i] = root_embed[i];
    for (int i = 0; i < w.E; i++) combined[w.E + i] = succ_embed[i];

    float ah[FNN_MAX_ACTION_HIDDEN];
    fnn_linear(params + w.act1_w, params + w.act1_b,
               combined, ah, 2 * w.E, w.AH);
    for (int i = 0; i < w.AH; i++)
        ah[i] = 1.0f / (1.0f + expf(-ah[i]));

    float logit = params[w.act2_b];
    for (int i = 0; i < w.AH; i++)
        logit += params[w.act2_w + i] * ah[i];
    return logit;
}

// ── Shared memory layout ─────────────────────────────────────────────

struct SelfPlayShared {
    HiveState state;
    GPUMove legal_moves[MAX_LEGAL_MOVES];
    float root_features[FNN_FEAT_DIM];
    float root_embed[FNN_MAX_EMBED];
    float action_logits[MAX_LEGAL_MOVES];
    float succ_values[MAX_LEGAL_MOVES];
    SelfPlayRNG rng;
    int num_legal;
    int move_number;
    int game_done;
};

// ── Main self-play kernel ────────────────────────────────────────────

#ifdef __CUDACC__

__global__ void fnn_selfplay_kernel(
    const float* __restrict__ params,     // flattened FNN weights
    int hidden_dim, int embed_dim, int action_hidden,
    uint8_t* out_states,                  // [B, max_len, sizeof_state]
    float* out_policy_probs,              // [B, max_len, max_considered]
    int* out_policy_indices,              // [B, max_len, max_considered]
    int* out_num_legal,                   // [B, max_len]
    int* out_num_candidates,              // [B, max_len]
    int* out_lengths,                     // [B]
    int* out_results,                     // [B]
    int batch_size,
    int max_game_length,
    int num_simulations,
    int max_considered,
    float c_visit, float c_scale,
    int temperature_drop_move,
    int expansion_mask,
    int64_t rng_seed,
    int sizeof_state
) {
    int game_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (game_idx >= batch_size) return;

    __shared__ SelfPlayShared sh;

    FNNWeights wt = make_fnn_weights(hidden_dim, embed_dim, action_hidden);

    // ── Initialize game state ──
    if (tid == 0) {
        init_state(sh.state, (uint8_t)expansion_mask);
        sh.move_number = 0;
        sh.game_done = 0;
        sh.rng.seed((uint64_t)rng_seed, game_idx);
    }
    __syncthreads();

    // ── Game loop ──
    while (sh.game_done == 0) {

        // ── Step 1: Thread 0 generates legal moves ──
        if (tid == 0) {
            sh.num_legal = generate_legal_moves(sh.state, sh.legal_moves);
            if (sh.num_legal == 0) {
                sh.game_done = 1;
            }
        }
        __syncthreads();
        if (sh.game_done) break;

        int nl = sh.num_legal;

        // ── Step 2: Thread 0 extracts root features + encodes ──
        if (tid == 0) {
            extract_fnn_features_device(
                sh.state, sh.legal_moves, nl, sh.root_features);
            fnn_encode(sh.root_features, sh.root_embed, params, wt);
        }
        __syncthreads();

        // ── Step 3: All threads evaluate successors in parallel ──
        for (int i = tid; i < nl; i += blockDim.x) {
            // Local copy of state
            HiveState temp = sh.state;
            apply_move(temp, sh.legal_moves[i]);

            // Generate successor's legal moves (needed for feature extraction)
            GPUMove temp_legal[MAX_LEGAL_MOVES];
            int temp_nl = generate_legal_moves(temp, temp_legal);

            // Extract features
            float feat[FNN_FEAT_DIM];
            extract_fnn_features_device(temp, temp_legal, temp_nl, feat);

            // Encode successor
            float emb[FNN_MAX_EMBED];
            fnn_encode(feat, emb, params, wt);

            // Score action + get successor value
            sh.action_logits[i] = fnn_score_action(sh.root_embed, emb, params, wt);
            sh.succ_values[i] = fnn_value(emb, params, wt);
        }
        __syncthreads();

        // ── Step 4: Thread 0 does Gumbel selection + records data ──
        if (tid == 0) {
            int step = sh.move_number;
            int step_off = game_idx * max_game_length + step;

            // Save state for training
            {
                const uint8_t* src = (const uint8_t*)&sh.state;
                uint8_t* dst = out_states + (int64_t)step_off * sizeof_state;
                for (int b = 0; b < sizeof_state; b++) dst[b] = src[b];
            }
            out_num_legal[step_off] = nl;

            // ── Gumbel candidate selection ──
            int nc = nl < max_considered ? nl : max_considered;

            // Add Gumbel noise to all logits
            float perturbed[MAX_LEGAL_MOVES];
            for (int i = 0; i < nl; i++)
                perturbed[i] = sh.action_logits[i] + sh.rng.gumbel();

            // Top-nc by greedy selection (nc is small, nl ≤ 256)
            int cand_idx[MAX_GUMBEL_CANDIDATES];
            float cand_scores[MAX_GUMBEL_CANDIDATES];
            for (int k = 0; k < nc; k++) {
                float best = -1e30f;
                int best_i = 0;
                for (int i = 0; i < nl; i++) {
                    if (perturbed[i] > best) {
                        best = perturbed[i];
                        best_i = i;
                    }
                }
                cand_idx[k] = best_i;
                cand_scores[k] = best;
                perturbed[best_i] = -1e30f;  // prevent re-selection
            }

            // ── Sequential halving ──
            float cand_values[MAX_GUMBEL_CANDIDATES];
            int visits[MAX_GUMBEL_CANDIDATES];
            float q_sums[MAX_GUMBEL_CANDIDATES];
            bool alive[MAX_GUMBEL_CANDIDATES];
            for (int k = 0; k < nc; k++) {
                cand_values[k] = sh.succ_values[cand_idx[k]];
                visits[k] = 0;
                q_sums[k] = 0.0f;
                alive[k] = true;
            }

            int remaining_sims = num_simulations;
            int nrounds = 1;
            { int tmp = nc; while (tmp > 1) { nrounds++; tmp = (tmp + 1) / 2; } }

            for (int r = 0; r < nrounds && remaining_sims > 0; r++) {
                int alive_count = 0;
                for (int k = 0; k < nc; k++)
                    if (alive[k]) alive_count++;
                if (alive_count == 0) break;

                int sims_each = remaining_sims / alive_count;
                if (sims_each < 1) sims_each = 1;

                for (int k = 0; k < nc; k++) {
                    if (alive[k]) {
                        q_sums[k] += (-cand_values[k]) * (float)sims_each;
                        visits[k] += sims_each;
                    }
                }
                remaining_sims -= sims_each * alive_count;
                if (remaining_sims < 0) remaining_sims = 0;

                // Eliminate bottom half (except last round)
                if (r < nrounds - 1 && alive_count > 1) {
                    float max_v = 1.0f;
                    for (int k = 0; k < nc; k++)
                        if ((float)visits[k] > max_v) max_v = (float)visits[k];

                    float sigma[MAX_GUMBEL_CANDIDATES];
                    for (int k = 0; k < nc; k++) {
                        if (alive[k] && visits[k] > 0) {
                            float qm = q_sums[k] / (float)visits[k];
                            sigma[k] = cand_scores[k] +
                                       (c_visit + max_v) * c_scale * qm;
                        } else {
                            sigma[k] = -1e30f;
                        }
                    }

                    int keep = (alive_count + 1) / 2;
                    if (keep < 1) keep = 1;

                    // Rank-based elimination (O(nc^2), nc ≤ 64)
                    for (int k = 0; k < nc; k++) {
                        if (!alive[k]) continue;
                        int better = 0;
                        for (int j = 0; j < nc; j++)
                            if (alive[j] && sigma[j] > sigma[k]) better++;
                        if (better >= keep) alive[k] = false;
                    }
                }
            }

            // ── Build policy from visit counts ──
            int total_v = 0;
            for (int k = 0; k < nc; k++) total_v += visits[k];

            float* pp = out_policy_probs + (int64_t)step_off * max_considered;
            int* pi = out_policy_indices + (int64_t)step_off * max_considered;

            for (int k = 0; k < nc; k++) {
                pp[k] = total_v > 0
                    ? (float)visits[k] / (float)total_v
                    : 1.0f / (float)nc;
                pi[k] = cand_idx[k];
            }
            for (int k = nc; k < max_considered; k++) {
                pp[k] = 0.0f;
                pi[k] = -1;
            }
            out_num_candidates[step_off] = nc;

            // ── Select move ──
            int chosen_legal_idx;
            if (step >= temperature_drop_move) {
                // Greedy: pick candidate with most visits
                int best_k = 0;
                float best_p = pp[0];
                for (int k = 1; k < nc; k++) {
                    if (pp[k] > best_p) { best_p = pp[k]; best_k = k; }
                }
                chosen_legal_idx = cand_idx[best_k];
            } else {
                // Sample from policy
                float u = sh.rng.uniform();
                float cum = 0.0f;
                chosen_legal_idx = cand_idx[nc - 1];  // fallback
                for (int k = 0; k < nc; k++) {
                    cum += pp[k];
                    if (u < cum) { chosen_legal_idx = cand_idx[k]; break; }
                }
            }

            // ── Apply move and advance ──
            apply_move(sh.state, sh.legal_moves[chosen_legal_idx]);
            sh.move_number++;

            if (sh.state.result != IN_PROGRESS ||
                sh.move_number >= max_game_length) {
                sh.game_done = 1;
            }
        }
        __syncthreads();
    }

    // ── Write final results ──
    if (tid == 0) {
        out_lengths[game_idx] = sh.move_number;
        out_results[game_idx] = (int)sh.state.result;
    }
}

#endif  // __CUDACC__

}  // namespace hive_gpu
