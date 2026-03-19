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

    return _extension
