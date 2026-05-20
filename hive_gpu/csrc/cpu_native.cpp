#include <algorithm>
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
    int n = generate_legal_moves(state, moves);

    auto moves_out = py::array_t<uint8_t>({MAX_LEGAL_MOVES, static_cast<int>(sizeof(GPUMove))});
    auto moves_info = moves_out.request();
    std::memset(moves_info.ptr, 0, static_cast<size_t>(moves_info.size));
    std::memcpy(moves_info.ptr, moves, static_cast<size_t>(n) * sizeof(GPUMove));

    auto features = py::array_t<float>({FNN_FEAT_DIM});
    auto feat_info = features.request();
    extract_fnn_features_device(state, moves, n, static_cast<float*>(feat_info.ptr));
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
        int child_n = generate_legal_moves(child, child_moves);
        extract_fnn_features_device(
            child,
            child_moves,
            child_n,
            feat_ptr + static_cast<size_t>(i) * FNN_FEAT_DIM
        );
        child_states.push_back(state_to_bytes(child));
    }

    return {features, child_states, results};
}

}  // namespace hive_gpu

PYBIND11_MODULE(hive_cpu_native_ext, m) {
    m.def("create_initial_state", &hive_gpu::cpu_create_initial_state);
    m.def("apply_move", &hive_gpu::cpu_apply_move);
    m.def("check_result", &hive_gpu::cpu_check_result);
    m.def("legal_moves_and_fnn_features", &hive_gpu::cpu_legal_moves_and_fnn_features);
    m.def("successor_features", &hive_gpu::cpu_successor_features);
    m.attr("SIZEOF_HIVE_STATE") = sizeof(hive_gpu::HiveState);
    m.attr("SIZEOF_GPU_MOVE") = sizeof(hive_gpu::GPUMove);
    m.attr("MAX_LEGAL_MOVES") = hive_gpu::MAX_LEGAL_MOVES;
    m.attr("FNN_FEAT_DIM") = hive_gpu::FNN_FEAT_DIM;
}
