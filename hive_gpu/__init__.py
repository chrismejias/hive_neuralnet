"""
hive_gpu — GPU-accelerated game logic and MCTS for Hive.

This package provides CUDA C++ kernels for:
- HiveState representation (17×17 hex grid, full beetle stacking)
- Legal move generation (all 5 piece types + articulation points)
- Batched game operations (init, move gen, apply move, game over)

Usage:
    import hive_gpu

    # Load the extension and initialize lookup tables
    ext = hive_gpu.load_extension()

    # Create a batch of initial game states
    states = ext.create_initial_states(batch_size=64)

    # Generate legal moves
    moves, num_legal = ext.generate_legal_moves_batch(states, 64)
"""

from __future__ import annotations

import os
import torch
from torch.utils.cpp_extension import load

# On Windows, add torch's lib dir and CUDA bin to the DLL search path so the
# extension's dependent DLLs (c10.dll, torch_cuda.dll, cudart64_*.dll) are found.
if os.name == "nt":
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
    _cuda_bin = os.path.join(
        os.environ.get("CUDA_HOME", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"),
        "bin",
    )
    if os.path.isdir(_cuda_bin):
        os.add_dll_directory(_cuda_bin)

_extension = None


def load_extension():
    """
    Load the hive_gpu CUDA extension. Tries to import the pre-built .pyd first;
    falls back to JIT compilation via torch.utils.cpp_extension.load().
    Returns the compiled extension module.
    """
    global _extension
    if _extension is not None:
        return _extension

    # Try importing pre-built extension (built via setup.py build_ext --inplace)
    pkg_dir = os.path.dirname(__file__)
    if pkg_dir not in __import__("sys").path:
        __import__("sys").path.insert(0, pkg_dir)

    try:
        import hive_gpu_ext as _ext
        _extension = _ext
    except ImportError:
        # Fall back to JIT compilation
        csrc_dir = os.path.join(pkg_dir, "csrc")
        _extension = load(
            name="hive_gpu_ext",
            sources=[
                os.path.join(csrc_dir, "game_logic.cu"),
                os.path.join(csrc_dir, "bindings.cpp"),
            ],
            extra_include_paths=[csrc_dir],
            extra_cuda_cflags=[
                "-O3",
                "--expt-relaxed-constexpr",
                "-std=c++17",
                "--allow-unsupported-compiler",
            ],
            extra_cflags=["-O3"],
            verbose=True,
        )

    # Initialize constant memory tables
    _extension.initialize_tables()

    # Patch any missing methods added after the pre-built .pyd was compiled.
    _patch_missing_methods(_extension)

    return _extension


def _patch_missing_methods(ext) -> None:
    """Add Python fallbacks for methods missing from older pre-built extensions."""

    if hasattr(ext, "legal_moves_to_actions_batch"):
        return  # already present in the compiled extension

    import numpy as np

    BOARD_SIZE = ext.BOARD_SIZE
    HALF = BOARD_SIZE // 2
    ENC_GRID = ext.ENC_GRID
    ENC_HALF = ENC_GRID // 2
    NUM_ENC = ENC_GRID * ENC_GRID
    PASS_IDX = ext.PASS_ACTION_INDEX
    MOVE_OFFSET = ext.MOVEMENT_OFFSET
    MAX_LEGAL = ext.MAX_LEGAL_MOVES

    def _gpu_move_to_action(move_raw, center_q: int, center_r: int) -> int:
        m_type = int(move_raw[0])
        m_pt   = int(move_raw[1])
        m_from = int(move_raw[2]) | (int(move_raw[3]) << 8)
        m_to   = int(move_raw[4]) | (int(move_raw[5]) << 8)
        if m_type == 2:          # PASS
            return PASS_IDX
        if m_type == 0:          # PLACE
            to_q = m_to % BOARD_SIZE - HALF
            to_r = m_to // BOARD_SIZE - HALF
            ec = to_q - center_q + ENC_HALF
            er = to_r - center_r + ENC_HALF
            if ec < 0 or ec >= ENC_GRID or er < 0 or er >= ENC_GRID:
                return -1
            return (m_pt - 1) * NUM_ENC + er * ENC_GRID + ec
        else:                    # MOVE
            fq = m_from % BOARD_SIZE - HALF
            fr = m_from // BOARD_SIZE - HALF
            tq = m_to % BOARD_SIZE - HALF
            tr = m_to // BOARD_SIZE - HALF
            sc = fq - center_q + ENC_HALF
            sr = fr - center_r + ENC_HALF
            dc = tq - center_q + ENC_HALF
            dr = tr - center_r + ENC_HALF
            if (sc < 0 or sc >= ENC_GRID or sr < 0 or sr >= ENC_GRID or
                    dc < 0 or dc >= ENC_GRID or dr < 0 or dr >= ENC_GRID):
                return -1
            return MOVE_OFFSET + (sr * ENC_GRID + sc) * NUM_ENC + (dr * ENC_GRID + dc)

    def legal_moves_to_actions_batch_py(
        states_tensor: "torch.Tensor",
        legal_moves_tensor: "torch.Tensor",
        num_legal_tensor: "torch.Tensor",
        batch_size: int,
    ) -> "torch.Tensor":
        """Pure-Python fallback for legal_moves_to_actions_batch (missing from old .pyd)."""
        centroids = ext.compute_centroids_batch(states_tensor, batch_size).cpu().numpy()
        moves_np = legal_moves_tensor.cpu().numpy()   # [B, MAX_LEGAL, move_size]
        num_np   = num_legal_tensor.cpu().numpy()     # [B]
        result   = np.full((batch_size, MAX_LEGAL), -1, dtype=np.int32)
        for i in range(batch_size):
            cq = int(round(centroids[i, 0]))
            cr = int(round(centroids[i, 1]))
            n  = int(num_np[i])
            for m in range(n):
                result[i, m] = _gpu_move_to_action(moves_np[i, m], cq, cr)
        return torch.tensor(result, dtype=torch.int32, device="cuda")

    # Attach as a module-level attribute so it looks like a native binding.
    import types
    ext.legal_moves_to_actions_batch = legal_moves_to_actions_batch_py
