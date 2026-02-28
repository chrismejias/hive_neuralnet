"""
Device detection and GPU optimization utilities.

Provides a single function to select the best available compute device
(CUDA > MPS > CPU) and apply backend-specific optimizations.

Usage:
    from hive_engine.device import get_device

    device = get_device()              # auto-detect best device
    device = get_device("cuda")        # force CUDA (error if unavailable)
    device = get_device("cuda:1")      # specific GPU
    device = get_device("cpu")         # force CPU
"""

from __future__ import annotations

import torch


def get_device(device_str: str | None = None) -> torch.device:
    """
    Resolve and return a torch device, applying backend optimizations.

    Auto-detection order: CUDA > MPS > CPU.
    When CUDA is selected, enables cuDNN benchmark mode for faster
    convolutions at the cost of a small warm-up on the first batch.

    Args:
        device_str: Optional device string. Examples:
            - None / "auto"  : auto-detect best device
            - "cuda"         : first CUDA GPU (fails if unavailable)
            - "cuda:0"       : specific CUDA GPU index
            - "mps"          : Apple Metal Performance Shaders
            - "cpu"          : force CPU

    Returns:
        A torch.device object.

    Raises:
        RuntimeError: If the requested device is not available.
    """
    if device_str is None or device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

        # Validate availability
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Ensure NVIDIA drivers and CUDA toolkit are installed."
            )
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS requested but not available on this system."
            )

    # Backend optimizations
    if device.type == "cuda":
        # cuDNN benchmark: auto-tunes convolution algorithms for the
        # input sizes we use (26×13×13), giving ~10-30% speedup after
        # the first batch.
        torch.backends.cudnn.benchmark = True

    return device


def device_summary(device: torch.device) -> str:
    """
    Return a human-readable summary of the device.

    Includes GPU name and memory for CUDA devices.
    """
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return f"{device} ({name}, {mem_gb:.1f} GB)"
    elif device.type == "mps":
        return f"{device} (Apple Metal)"
    else:
        return str(device)
