#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <torch/extension.h>
#include <pybind11/numpy.h>

#define HIVE_CPU_NATIVE 1
#define __CUDACC__ 1
#define __device__
#define __host__
#define __forceinline__ inline
#define __global__
#define __managed__
#define __constant__
#define max(a, b) (((a) > (b)) ? (a) : (b))

#ifndef __CUDA_ARCH__
static inline int __ffsll(unsigned long long x) {
    if (x == 0) return 0;
#ifdef _MSC_VER
    unsigned long idx = 0;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx) + 1;
#else
    return __builtin_ctzll(x) + 1;
#endif
}
#endif

template <typename T>
static inline void atomicAdd(T* dst, T value) {
    *dst += value;
}

#include "hex_grid.cuh"
#include "hive_state.cuh"
#include "articulation.cuh"
#include "move_gen.cuh"
#include "fnn_features.cuh"

#undef max

namespace py = pybind11;

namespace hive_gpu {
namespace {

struct CPUUndo {
    uint16_t from_cell, to_cell, turn, stunned_cell, queen_cell[2];
    uint8_t queen_placed, result, hand_color, hand_type, hand_count;
    uint8_t from_height, to_height, from_pieces[MAX_STACK], to_pieces[MAX_STACK];
    uint8_t cell_flags[2];
};

inline uint8_t cpu_cell_flags(const HiveState& state, int cell) {
    if (cell < 0 || cell >= NUM_CELLS) return 0;
    return (state.occupied.get(cell) ? 1 : 0) |
        (state.white_top.get(cell) ? 2 : 0) |
        (state.black_top.get(cell) ? 4 : 0);
}

inline void cpu_restore_cell_flags(HiveState& state, int cell, uint8_t flags) {
    if (cell < 0 || cell >= NUM_CELLS) return;
    if (flags & 1) state.occupied.set(cell); else state.occupied.clr(cell);
    if (flags & 2) state.white_top.set(cell); else state.white_top.clr(cell);
    if (flags & 4) state.black_top.set(cell); else state.black_top.clr(cell);
}

inline void cpu_make_move(HiveState& state, const GPUMove& move, CPUUndo& undo) {
    undo.from_cell = move.type == MOVE_MOVE ? move.from_cell : 0xFFFF;
    undo.to_cell = move.type == MOVE_PASS ? 0xFFFF : move.to_cell;
    undo.turn = state.turn;
    undo.stunned_cell = state.stunned_cell;
    undo.queen_cell[0] = state.queen_cell[0];
    undo.queen_cell[1] = state.queen_cell[1];
    undo.queen_placed = state.queen_placed;
    undo.result = static_cast<uint8_t>(state.result);
    undo.hand_color = undo.hand_type = 0xFF;
    undo.hand_count = 0;
    if (move.type == MOVE_PLACE) {
        undo.hand_color = static_cast<uint8_t>(current_player(state));
        undo.hand_type = static_cast<uint8_t>(static_cast<int>(move.piece_type) - 1);
        undo.hand_count = state.hands[undo.hand_color][undo.hand_type];
    }
    undo.from_height = undo.from_cell != 0xFFFF ? state.height[undo.from_cell] : 0;
    undo.to_height = undo.to_cell != 0xFFFF ? state.height[undo.to_cell] : 0;
    undo.cell_flags[0] = cpu_cell_flags(state, undo.from_cell);
    undo.cell_flags[1] = cpu_cell_flags(state, undo.to_cell);
    for (int level = 0; level < MAX_STACK; ++level) {
        undo.from_pieces[level] = undo.from_cell != 0xFFFF ?
            state.pieces[level][undo.from_cell] : 0;
        undo.to_pieces[level] = undo.to_cell != 0xFFFF ?
            state.pieces[level][undo.to_cell] : 0;
    }
    apply_move(state, move);
}

inline void cpu_unmake_move(HiveState& state, const CPUUndo& undo) {
    state.turn = undo.turn;
    state.stunned_cell = undo.stunned_cell;
    state.queen_cell[0] = undo.queen_cell[0];
    state.queen_cell[1] = undo.queen_cell[1];
    state.queen_placed = undo.queen_placed;
    state.result = static_cast<GameResult>(undo.result);
    if (undo.hand_color < 2 && undo.hand_type < NUM_PIECE_TYPES) {
        state.hands[undo.hand_color][undo.hand_type] = undo.hand_count;
    }
    if (undo.from_cell != 0xFFFF) {
        state.height[undo.from_cell] = undo.from_height;
        for (int level = 0; level < MAX_STACK; ++level) {
            state.pieces[level][undo.from_cell] = undo.from_pieces[level];
        }
        cpu_restore_cell_flags(state, undo.from_cell, undo.cell_flags[0]);
    }
    if (undo.to_cell != 0xFFFF && undo.to_cell != undo.from_cell) {
        state.height[undo.to_cell] = undo.to_height;
        for (int level = 0; level < MAX_STACK; ++level) {
            state.pieces[level][undo.to_cell] = undo.to_pieces[level];
        }
        cpu_restore_cell_flags(state, undo.to_cell, undo.cell_flags[1]);
    }
}

inline uint64_t cpu_mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline uint64_t cpu_cell_hash(const HiveState& state, int cell) {
    if (cell < 0 || cell >= NUM_CELLS) return 0;
    uint64_t hash = cpu_mix64(0x100000ULL + static_cast<uint64_t>(cell) * 17ULL + state.height[cell]);
    for (int level = 0; level < state.height[cell]; ++level) {
        uint64_t key = static_cast<uint64_t>(cell) |
            (static_cast<uint64_t>(level) << 10) |
            (static_cast<uint64_t>(state.pieces[level][cell]) << 16);
        hash ^= cpu_mix64(key);
    }
    return hash;
}

inline uint64_t cpu_metadata_hash(const HiveState& state) {
    uint64_t hash = cpu_mix64(0x200000ULL + state.turn);
    hash ^= cpu_mix64(0x210000ULL + state.stunned_cell);
    hash ^= cpu_mix64(0x220000ULL + static_cast<uint64_t>(state.result));
    for (int c = 0; c < 2; ++c) for (int type = 0; type < NUM_PIECE_TYPES; ++type) {
        hash ^= cpu_mix64(0x300000ULL + static_cast<uint64_t>(c * NUM_PIECE_TYPES + type) * 16ULL +
                          state.hands[c][type]);
    }
    return hash;
}

inline uint64_t cpu_hash_state(const HiveState& state) {
    uint64_t hash = cpu_metadata_hash(state);
    for (int wi = 0; wi < BB_WORDS; ++wi) {
        uint64_t bits = state.occupied.w[wi];
        while (bits) {
            int bit = __ffsll(bits) - 1;
            int cell = wi * 64 + bit;
            bits &= bits - 1;
            if (cell < NUM_CELLS) hash ^= cpu_cell_hash(state, cell);
        }
    }
    return hash ? hash : 1ULL;
}

inline uint64_t cpu_make_move_hashed(
    HiveState& state, const GPUMove& move, uint64_t hash, CPUUndo& undo
) {
    uint16_t from = move.type == MOVE_MOVE ? move.from_cell : 0xFFFF;
    uint16_t to = move.type == MOVE_PASS ? 0xFFFF : move.to_cell;
    uint64_t next = hash ^ cpu_metadata_hash(state);
    if (from != 0xFFFF) next ^= cpu_cell_hash(state, from);
    if (to != 0xFFFF && to != from) next ^= cpu_cell_hash(state, to);
    cpu_make_move(state, move, undo);
    next ^= cpu_metadata_hash(state);
    if (from != 0xFFFF) next ^= cpu_cell_hash(state, from);
    if (to != 0xFFFF && to != from) next ^= cpu_cell_hash(state, to);
    return next ? next : 1ULL;
}

void init_cpu_tables_once() {
    static bool initialized = false;
    if (initialized) return;
    init_hex_tables();
    std::memcpy(NEIGHBORS, HOST_NEIGHBOR_TABLE, sizeof(HOST_NEIGHBOR_TABLE));
    std::memcpy(SLIDE_FLANKS, HOST_SLIDE_TABLE, sizeof(HOST_SLIDE_TABLE));
    initialized = true;
}

HiveState state_from_bytes(const py::bytes& raw) {
    std::string s = raw;
    if (s.size() != sizeof(HiveState)) {
        throw std::runtime_error("state byte length does not match HiveState");
    }
    HiveState state;
    std::memcpy(&state, s.data(), sizeof(HiveState));
    return state;
}

py::bytes state_to_bytes(const HiveState& state) {
    return py::bytes(reinterpret_cast<const char*>(&state), sizeof(HiveState));
}

GPUMove move_from_row(const uint8_t* row) {
    GPUMove mv;
    std::memcpy(&mv, row, sizeof(GPUMove));
    return mv;
}

}  // namespace

py::bytes cpu_create_initial_state(int expansion_mask) {
    init_cpu_tables_once();
    HiveState state;
    init_state(state, static_cast<uint8_t>(expansion_mask));
    return state_to_bytes(state);
}

py::bytes cpu_apply_move(py::bytes raw, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> move_arr) {
    init_cpu_tables_once();
    HiveState state = state_from_bytes(raw);
    auto info = move_arr.request();
    if (info.size < static_cast<py::ssize_t>(sizeof(GPUMove))) {
        throw std::runtime_error("move array is too small");
    }
    GPUMove mv = move_from_row(static_cast<const uint8_t*>(info.ptr));
    apply_move(state, mv);
    return state_to_bytes(state);
}

int cpu_check_result(py::bytes raw) {
    init_cpu_tables_once();
    HiveState state = state_from_bytes(raw);
    check_game_over(state);
    return static_cast<int>(state.result);
}

std::tuple<py::array_t<uint8_t>, int, py::array_t<float>>
cpu_legal_moves_and_fnn_features(py::bytes raw) {
    init_cpu_tables_once();
    HiveState state = state_from_bytes(raw);
    GPUMove moves[MAX_LEGAL_MOVES];
    MovegenStateCache cache;
    int n = generate_legal_moves_with_cache(state, moves, cache);

    auto moves_out = py::array_t<uint8_t>({MAX_LEGAL_MOVES, static_cast<int>(sizeof(GPUMove))});
    auto moves_info = moves_out.request();
    std::memset(moves_info.ptr, 0, static_cast<size_t>(moves_info.size));
    std::memcpy(moves_info.ptr, moves, static_cast<size_t>(n) * sizeof(GPUMove));

    auto features = py::array_t<float>({FNN_FEAT_DIM});
    auto feat_info = features.request();
    extract_fnn_features_with_ap_device(
        state, moves, n, cache.ap_mask, static_cast<float*>(feat_info.ptr));
    return {moves_out, n, features};
}

std::tuple<py::array_t<float>, std::vector<py::bytes>, py::array_t<int>>
cpu_successor_features(
    py::bytes raw,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> moves_arr,
    int num_moves
) {
    init_cpu_tables_once();
    HiveState root = state_from_bytes(raw);
    auto moves_info = moves_arr.request();
    if (moves_info.ndim != 2 || moves_info.shape[1] < static_cast<py::ssize_t>(sizeof(GPUMove))) {
        throw std::runtime_error("moves must have shape [N, SIZEOF_GPU_MOVE]");
    }
    num_moves = std::max(0, std::min(num_moves, static_cast<int>(moves_info.shape[0])));

    auto features = py::array_t<float>({num_moves, FNN_FEAT_DIM});
    auto results = py::array_t<int>({num_moves});
    auto feat_info = features.request();
    auto result_info = results.request();
    float* feat_ptr = static_cast<float*>(feat_info.ptr);
    int* result_ptr = static_cast<int*>(result_info.ptr);
    const uint8_t* moves_ptr = static_cast<const uint8_t*>(moves_info.ptr);

    std::vector<py::bytes> child_states;
    child_states.reserve(static_cast<size_t>(num_moves));

    for (int i = 0; i < num_moves; ++i) {
        HiveState child = root;
        GPUMove mv = move_from_row(moves_ptr + static_cast<size_t>(i) * moves_info.strides[0]);
        apply_move(child, mv);
        check_game_over(child);
        result_ptr[i] = static_cast<int>(child.result);

        GPUMove child_moves[MAX_LEGAL_MOVES];
        MovegenStateCache child_cache;
        int child_n = generate_fnn_feature_moves_with_cache(
            child, child_moves, child_cache);
        extract_fnn_features_with_ap_device(
            child, child_moves, child_n, child_cache.ap_mask,
            feat_ptr + static_cast<size_t>(i) * FNN_FEAT_DIM);
        child_states.push_back(state_to_bytes(child));
    }

    return {features, child_states, results};
}

std::tuple<std::vector<py::bytes>, py::array_t<int>> cpu_successors(
    py::bytes raw,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> moves_arr,
    int num_moves
) {
    init_cpu_tables_once();
    HiveState state = state_from_bytes(raw);
    auto moves_info = moves_arr.request();
    if (moves_info.ndim != 2 ||
        moves_info.shape[1] < static_cast<py::ssize_t>(sizeof(GPUMove))) {
        throw std::runtime_error("moves must have shape [N, SIZEOF_GPU_MOVE]");
    }
    num_moves = std::max(0, std::min(num_moves, static_cast<int>(moves_info.shape[0])));
    auto results = py::array_t<int>({num_moves});
    int* result_ptr = static_cast<int*>(results.request().ptr);
    const uint8_t* moves_ptr = static_cast<const uint8_t*>(moves_info.ptr);
    std::vector<py::bytes> child_states;
    child_states.reserve(static_cast<size_t>(num_moves));
    for (int i = 0; i < num_moves; ++i) {
        GPUMove move = move_from_row(
            moves_ptr + static_cast<size_t>(i) * moves_info.strides[0]);
        CPUUndo undo;
        cpu_make_move(state, move, undo);
        result_ptr[i] = static_cast<int>(state.result);
        child_states.push_back(state_to_bytes(state));
        cpu_unmake_move(state, undo);
    }
    return {child_states, results};
}

py::dict cpu_benchmark_primitives(py::bytes raw, int iterations) {
    init_cpu_tables_once();
    HiveState root = state_from_bytes(raw);
    GPUMove moves[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(root, moves);
    if (n <= 0) throw std::runtime_error("benchmark position has no legal moves");
    iterations = std::max(1, iterations);
    volatile uint64_t sink = 0;
    using clock = std::chrono::steady_clock;

    auto start = clock::now();
    for (int i = 0; i < iterations; ++i) {
        GPUMove generated[MAX_LEGAL_MOVES];
        sink += static_cast<uint64_t>(generate_legal_moves(root, generated));
    }
    double movegen = std::chrono::duration<double>(clock::now() - start).count();

    start = clock::now();
    for (int i = 0; i < iterations; ++i) {
        HiveState child = root;
        apply_move(child, moves[i % n]);
        sink += child.turn;
    }
    double copy_apply = std::chrono::duration<double>(clock::now() - start).count();

    HiveState mutable_state = root;
    start = clock::now();
    for (int i = 0; i < iterations; ++i) {
        CPUUndo undo;
        cpu_make_move(mutable_state, moves[i % n], undo);
        sink += mutable_state.turn;
        cpu_unmake_move(mutable_state, undo);
    }
    double make_unmake = std::chrono::duration<double>(clock::now() - start).count();
    if (std::memcmp(&mutable_state, &root, sizeof(HiveState)) != 0) {
        throw std::runtime_error("make/unmake failed to restore state exactly");
    }

    py::dict out;
    out["iterations"] = iterations;
    out["legal_moves"] = n;
    out["state_bytes"] = sizeof(HiveState);
    out["movegen_seconds"] = movegen;
    out["copy_apply_seconds"] = copy_apply;
    out["make_unmake_seconds"] = make_unmake;
    out["sink"] = sink;
    return out;
}

struct CPUFNNWeights {
    int h, e, fc1_w, fc1_b, ln_w, ln_b, fc2_w, fc2_b, val_w, val_b;
};

CPUFNNWeights cpu_fnn_layout(int h, int e) {
    CPUFNNWeights w{h, e};
    int o = 0;
    w.fc1_w = o; o += FNN_FEAT_DIM * h;
    w.fc1_b = o; o += h;
    w.ln_w = o; o += h;
    w.ln_b = o; o += h;
    w.fc2_w = o; o += h * e;
    w.fc2_b = o; o += e;
    w.val_w = o; o += e;
    w.val_b = o;
    return w;
}

float cpu_fnn_value(const HiveState& state, const float* p, const CPUFNNWeights& w) {
    GPUMove feature_moves[MAX_LEGAL_MOVES];
    MovegenStateCache cache;
    int n = generate_fnn_feature_moves_with_cache(state, feature_moves, cache);
    float f[FNN_FEAT_DIM];
    extract_fnn_features_with_ap_device(state, feature_moves, n, cache.ap_mask, f);
    float hidden[64], embed[64];
    for (int o = 0; o < w.h; ++o) {
        float sum = p[w.fc1_b + o];
        const float* row = p + w.fc1_w + o * FNN_FEAT_DIM;
        for (int i = 0; i < FNN_FEAT_DIM; ++i) sum += row[i] * f[i];
        hidden[o] = 1.0f / (1.0f + std::exp(-sum));
    }
    float mean = 0.0f;
    for (int i = 0; i < w.h; ++i) mean += hidden[i];
    mean /= static_cast<float>(w.h);
    float var = 0.0f;
    for (int i = 0; i < w.h; ++i) { float d = hidden[i] - mean; var += d * d; }
    float inv = 1.0f / std::sqrt(var / static_cast<float>(w.h) + 1e-5f);
    for (int i = 0; i < w.h; ++i) {
        hidden[i] = (hidden[i] - mean) * inv * p[w.ln_w + i] + p[w.ln_b + i];
    }
    for (int o = 0; o < w.e; ++o) {
        float sum = p[w.fc2_b + o];
        const float* row = p + w.fc2_w + o * w.h;
        for (int i = 0; i < w.h; ++i) sum += row[i] * hidden[i];
        embed[o] = sum;
    }
    float value = p[w.val_b];
    for (int i = 0; i < w.e; ++i) value += p[w.val_w + i] * embed[i];
    return std::tanh(value);
}

struct CPUSearch {
    struct TTEntry {
        uint64_t key = 0;
        float value = 0;
        int16_t depth = -1;
        uint8_t bound = 0;
        GPUMove best_move{};
    };
    const float* params;
    CPUFNNWeights weights;
    int budget, nodes = 0, cutoffs = 0, tt_hits = 0;
    int qnodes = 0, tactical_moves = 0, extension_nodes = 0;
    int quiescence_plies = 1, qnode_budget = 1, tactical_mask = 15;
    bool recursive_threat_qsearch = false, aborted = false;
    int forced_extension_max_chain = 0;
    std::vector<TTEntry> tt = std::vector<TTEntry>(1 << 18);

    float terminal(const HiveState& state, int ply) const {
        if (state.result == DRAW || state.result == IN_PROGRESS) return 0.0f;
        Color side = current_player(state);
        bool won = (side == WHITE && state.result == WHITE_WINS) ||
            (side == BLACK && state.result == BLACK_WINS);
        return (won ? 1.0f : -1.0f) * (10.0f - std::min(ply, 100) * 0.01f);
    }
    bool color_won(const HiveState& state, Color color) const {
        return (color == WHITE && state.result == WHITE_WINS) ||
               (color == BLACK && state.result == BLACK_WINS);
    }
    bool adjacent_to(int cell, uint16_t target) const {
        if (cell < 0 || cell >= NUM_CELLS || target >= NUM_CELLS) return false;
        for (int d = 0; d < NUM_DIRS; ++d) if (NEIGHBORS[target][d] == cell) return true;
        return false;
    }
    bool side_under_immediate_threat(const HiveState& state) const {
        if (state.result != IN_PROGRESS) return false;
        Color side = current_player(state);
        if (queen_surround_count_for_color_device(state, side) < 5) return false;
        HiveState probe = state;
        probe.turn ^= 1U;
        probe.stunned_cell = 0xFFFF;
        return has_immediate_surround_win_for_current_player(probe);
    }
    bool power_piece_mobile(const HiveState& state, int cell, MovegenStateCache& cache) const {
        if (cell < 0 || cell >= NUM_CELLS || state.height[cell] == 0 ||
            is_stunned_cell(state, cell) || is_pinned(cache, cell)) return false;
        PieceType type = top_piece_type_at(state, cell);
        if (type == PT_QUEEN) return has_queen_move(state, cell);
        if (type == PT_BEETLE) return has_beetle_move(state, cell);
        if (type == PT_ANT) {
            const Bitboard& perimeter = ensure_base_perimeter(state, cache);
            return has_ant_move_with_perimeter(state, cell, perimeter);
        }
        return false;
    }
    Bitboard mobile_power_pieces(
        const HiveState& state, Color color, MovegenStateCache& cache
    ) const {
        Bitboard mobile;
        mobile.clear();
        const Bitboard& tops = color == WHITE ? state.white_top : state.black_top;
        for (int wi = 0; wi < BB_WORDS; ++wi) {
            uint64_t bits = tops.w[wi];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int cell = wi * 64 + bit;
                bits &= bits - 1;
                if (cell < NUM_CELLS && power_piece_mobile(state, cell, cache)) mobile.set(cell);
            }
        }
        return mobile;
    }
    bool immobilizes_from_mask(
        const Bitboard& before, Color target, const GPUMove& move,
        const HiveState& child, MovegenStateCache& child_cache
    ) const {
        for (int wi = 0; wi < BB_WORDS; ++wi) {
            uint64_t bits = before.w[wi];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int cell = wi * 64 + bit;
                bits &= bits - 1;
                if (move.type == MOVE_MOVE && (int)move.from_cell == cell) continue;
                bool still_mobile = cell < NUM_CELLS && child.height[cell] > 0 &&
                    top_piece_color_at(child, cell) == target &&
                    power_piece_mobile(child, cell, child_cache);
                if (!still_mobile) return true;
            }
        }
        return false;
    }
    bool creates_queen_threat(const HiveState& child, Color mover) const {
        if (child.result != IN_PROGRESS) return false;
        Color opponent = mover == WHITE ? BLACK : WHITE;
        if (queen_surround_count_for_color_device(child, opponent) != 5) return false;
        HiveState probe = child;
        probe.turn = (uint16_t)((probe.turn & ~1U) | (uint16_t)mover);
        probe.stunned_cell = 0xFFFF;
        return has_immediate_surround_win_for_current_player(probe);
    }
    bool priority_q_candidate(
        const HiveState& state, const GPUMove& move, Color mover,
        Color opponent, const Bitboard& mobile
    ) const {
        if (adjacent_to((int)move.to_cell, state.queen_cell[opponent])) return true;
        if (move.type != MOVE_MOVE) return false;
        if (move.from_cell == state.queen_cell[mover] ||
            adjacent_to((int)move.from_cell, state.queen_cell[mover])) return true;
        for (int wi = 0; wi < BB_WORDS; ++wi) {
            uint64_t bits = mobile.w[wi];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int cell = wi * 64 + bit;
                bits &= bits - 1;
                if ((int)move.to_cell == cell || adjacent_to((int)move.to_cell, cell) ||
                    adjacent_to((int)move.from_cell, cell)) return true;
            }
        }
        return false;
    }
    float order_score(const HiveState& state, const GPUMove& move) const {
        Color mover = current_player(state);
        Color opponent = mover == WHITE ? BLACK : WHITE;
        float score = 0.0f;
        for (int d = 0; d < NUM_DIRS; ++d) {
            if (state.queen_cell[opponent] < NUM_CELLS &&
                NEIGHBORS[state.queen_cell[opponent]][d] == move.to_cell) score += 10000.0f;
            if (state.queen_cell[mover] < NUM_CELLS && move.type == MOVE_MOVE &&
                NEIGHBORS[state.queen_cell[mover]][d] == move.from_cell) score += 1000.0f;
        }
        if (move.piece_type == PT_BEETLE || move.piece_type == PT_QUEEN) score += 100.0f;
        return score;
    }
    void order(GPUMove* moves, int n, const HiveState& state) const {
        for (int rank = 0; rank < n; ++rank) {
            int best = rank; float value = order_score(state, moves[rank]);
            for (int i = rank + 1; i < n; ++i) {
                float candidate = order_score(state, moves[i]);
                if (candidate > value) { value = candidate; best = i; }
            }
            if (best != rank) std::swap(moves[rank], moves[best]);
        }
    }
    float quiescence(HiveState& state, float alpha, float beta, int ply, int remaining = -1) {
        if (state.result != IN_PROGRESS) return terminal(state, ply);
        int qplies = remaining < 0 ? quiescence_plies : remaining;
        if (qplies <= 0) return cpu_fnn_value(state, params, weights);
        GPUMove moves[MAX_LEGAL_MOVES];
        MovegenStateCache cache;
        int n = generate_legal_moves_with_cache(state, moves, cache);
        float best = cpu_fnn_value(state, params, weights);
        if (best >= beta || n <= 0) return best;
        alpha = std::max(alpha, best);
        Color mover = current_player(state);
        Color opponent = mover == WHITE ? BLACK : WHITE;
        Bitboard mobile;
        mobile.clear();
        if (tactical_mask & 1) mobile = mobile_power_pieces(state, opponent, cache);
        int own_before = queen_surround_count_for_color_device(state, mover);
        int opp_before = queen_surround_count_for_color_device(state, opponent);
        for (int phase = 0; phase < 2; ++phase) for (int i = 0; i < n; ++i) {
            bool priority = priority_q_candidate(state, moves[i], mover, opponent, mobile);
            if (priority != (phase == 0)) continue;
            if (nodes >= budget) { aborted = true; return 0.0f; }
            if (qnodes >= qnode_budget) return best;
            ++nodes; ++qnodes;
            CPUUndo undo;
            cpu_make_move(state, moves[i], undo);
            int own_after = queen_surround_count_for_color_device(state, mover);
            int opp_after = queen_surround_count_for_color_device(state, opponent);
            bool tactical = color_won(state, mover) ||
                ((tactical_mask & 2) && opp_after > opp_before) ||
                ((tactical_mask & 4) && own_after < own_before) ||
                ((tactical_mask & 8) && opp_after == 5 && creates_queen_threat(state, mover));
            MovegenStateCache child_cache;
            if (!tactical && (tactical_mask & 1) && !mobile.is_zero()) {
                init_movegen_state_cache(state, child_cache);
                tactical = immobilizes_from_mask(mobile, opponent, moves[i], state, child_cache);
            }
            if (!tactical) { cpu_unmake_move(state, undo); continue; }
            ++tactical_moves;
            float value = state.result != IN_PROGRESS ? -terminal(state, ply + 1) :
                (recursive_threat_qsearch && qplies > 1 ?
                    -quiescence(state, -beta, -alpha, ply + 1, qplies - 1) :
                    -cpu_fnn_value(state, params, weights));
            cpu_unmake_move(state, undo);
            if (aborted) return 0.0f;
            best = std::max(best, value);
            alpha = std::max(alpha, best);
            if (alpha >= beta) { ++cutoffs; return best; }
        }
        return best;
    }
    float negamax(
        HiveState& state, uint64_t hash, int depth, float alpha, float beta,
        int ply, int extension_chain = 0
    ) {
        if (++nodes > budget) { aborted = true; return 0.0f; }
        if (state.result != IN_PROGRESS) return terminal(state, ply);
        if (depth <= 0) return quiescence(state, alpha, beta, ply);
        float alpha_start = alpha;
        TTEntry& entry = tt[hash & (tt.size() - 1)];
        if (entry.key == hash && entry.depth >= depth) {
            ++tt_hits;
            if (entry.bound == 1) return entry.value;
            if (entry.bound == 2) alpha = std::max(alpha, entry.value);
            if (entry.bound == 3) beta = std::min(beta, entry.value);
            if (alpha >= beta) return entry.value;
        }
        GPUMove moves[MAX_LEGAL_MOVES];
        int n = generate_legal_moves(state, moves);
        order(moves, n, state);
        bool extend = forced_extension_max_chain > 0 &&
            extension_chain < forced_extension_max_chain &&
            (n == 1 || side_under_immediate_threat(state));
        int next_depth = depth - 1 + (extend ? 1 : 0);
        int next_chain = extend ? extension_chain + 1 : 0;
        if (extend) ++extension_nodes;
        if (entry.key == hash) for (int i = 0; i < n; ++i) {
            if (std::memcmp(&moves[i], &entry.best_move, sizeof(GPUMove)) == 0) {
                std::swap(moves[0], moves[i]); break;
            }
        }
        float best = -1000.0f;
        GPUMove best_move = moves[0];
        for (int i = 0; i < n; ++i) {
            CPUUndo undo;
            uint64_t child_hash = cpu_make_move_hashed(state, moves[i], hash, undo);
            float value = -negamax(
                state, child_hash, next_depth, -beta, -alpha, ply + 1, next_chain);
            cpu_unmake_move(state, undo);
            if (aborted) return 0.0f;
            if (value > best) { best = value; best_move = moves[i]; }
            alpha = std::max(alpha, best);
            if (alpha >= beta) { ++cutoffs; break; }
        }
        entry.key = hash; entry.value = best; entry.depth = static_cast<int16_t>(depth);
        entry.bound = best <= alpha_start ? 3 : (best >= beta ? 2 : 1);
        entry.best_move = best_move;
        return best;
    }
};

py::tuple cpu_native_alpha_beta(
    py::bytes raw,
    py::array_t<float, py::array::c_style | py::array::forcecast> params_arr,
    int hidden_dim, int embed_dim, int node_budget, int max_depth,
    int quiescence_plies, float quiescence_budget_fraction, int tactical_mask,
    bool recursive_threat_qsearch, int forced_extension_max_chain
) {
    init_cpu_tables_once();
    if (hidden_dim <= 0 || hidden_dim > 64 || embed_dim <= 0 || embed_dim > 64) {
        throw std::runtime_error("native CPU FNN dimensions must be in [1, 64]");
    }
    HiveState root = state_from_bytes(raw);
    auto params = params_arr.request();
    CPUSearch search{static_cast<const float*>(params.ptr), cpu_fnn_layout(hidden_dim, embed_dim),
                     std::max(1, node_budget)};
    search.recursive_threat_qsearch = recursive_threat_qsearch;
    search.quiescence_plies = std::max(0, std::min(
        recursive_threat_qsearch ? 4 : 1, quiescence_plies));
    search.qnode_budget = std::max(1, static_cast<int>(node_budget *
        std::max(0.0f, std::min(0.95f, quiescence_budget_fraction))));
    search.tactical_mask = tactical_mask & 15;
    search.forced_extension_max_chain = std::max(0, forced_extension_max_chain);
    GPUMove root_moves[MAX_LEGAL_MOVES];
    int n = generate_legal_moves(root, root_moves);
    if (n <= 0) throw std::runtime_error("position has no legal moves");
    search.order(root_moves, n, root);
    GPUMove best = root_moves[0]; float best_value = cpu_fnn_value(root, search.params, search.weights);
    uint64_t root_hash = cpu_hash_state(root);
    int completed = 0;
    for (int depth = 1; depth <= max_depth; ++depth) {
        GPUMove iteration_best = best; float iteration_value = -1000.0f, alpha = -1000.0f;
        for (int i = 0; i < n; ++i) {
            CPUUndo undo; uint64_t child_hash = cpu_make_move_hashed(root, root_moves[i], root_hash, undo);
            float value = -search.negamax(root, child_hash, depth - 1, -1000.0f, -alpha, 1);
            cpu_unmake_move(root, undo);
            if (search.aborted) break;
            if (value > iteration_value) { iteration_value = value; iteration_best = root_moves[i]; }
            alpha = std::max(alpha, iteration_value);
        }
        if (search.aborted) break;
        best = iteration_best; best_value = iteration_value; completed = depth;
        for (int i = 0; i < n; ++i) if (std::memcmp(&root_moves[i], &best, sizeof(GPUMove)) == 0) {
            std::swap(root_moves[0], root_moves[i]); break;
        }
    }
    auto move_out = py::array_t<uint8_t>({static_cast<int>(sizeof(GPUMove))});
    std::memcpy(move_out.request().ptr, &best, sizeof(GPUMove));
    py::dict stats;
    stats["depth"] = completed; stats["nodes"] = search.nodes;
    stats["cutoffs"] = search.cutoffs; stats["value"] = best_value;
    stats["tt_hits"] = search.tt_hits;
    stats["qnodes"] = search.qnodes;
    stats["tactical_moves"] = search.tactical_moves;
    stats["forced_extensions"] = search.extension_nodes;
    return py::make_tuple(move_out, stats);
}

float cpu_native_value(
    py::bytes raw,
    py::array_t<float, py::array::c_style | py::array::forcecast> params_arr,
    int hidden_dim, int embed_dim
) {
    init_cpu_tables_once();
    HiveState state = state_from_bytes(raw);
    auto params = params_arr.request();
    return cpu_fnn_value(
        state, static_cast<const float*>(params.ptr),
        cpu_fnn_layout(hidden_dim, embed_dim));
}

}  // namespace hive_gpu

PYBIND11_MODULE(hive_cpu_native_ext, m) {
    m.def("create_initial_state", &hive_gpu::cpu_create_initial_state);
    m.def("apply_move", &hive_gpu::cpu_apply_move);
    m.def("check_result", &hive_gpu::cpu_check_result);
    m.def("legal_moves_and_fnn_features", &hive_gpu::cpu_legal_moves_and_fnn_features);
    m.def("successor_features", &hive_gpu::cpu_successor_features);
    m.def("successors", &hive_gpu::cpu_successors);
    m.def("benchmark_primitives", &hive_gpu::cpu_benchmark_primitives);
    m.def("native_alpha_beta", &hive_gpu::cpu_native_alpha_beta);
    m.def("native_value", &hive_gpu::cpu_native_value);
    m.attr("SIZEOF_HIVE_STATE") = sizeof(hive_gpu::HiveState);
    m.attr("SIZEOF_GPU_MOVE") = sizeof(hive_gpu::GPUMove);
    m.attr("MAX_LEGAL_MOVES") = hive_gpu::MAX_LEGAL_MOVES;
    m.attr("FNN_FEAT_DIM") = hive_gpu::FNN_FEAT_DIM;
}
