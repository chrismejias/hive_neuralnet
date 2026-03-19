"""
Build configuration for the hive_gpu CUDA extension.

Usage:
    pip install -e hive_gpu/
    # or
    cd hive_gpu && python setup.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hive_gpu",
    ext_modules=[
        CUDAExtension(
            name="hive_gpu_ext",
            sources=[
                "csrc/game_logic.cu",
                "csrc/bindings.cpp",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                    "--allow-unsupported-compiler",  # VS 2025 not yet in CUDA's allowlist
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
