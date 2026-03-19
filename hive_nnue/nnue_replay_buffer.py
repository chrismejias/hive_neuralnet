"""
Replay buffer for NNUE training.

Stores (feature_vector, policy_target, value_target) tuples and provides
batch sampling as torch tensors ready for GPU training.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


# ── Training Example ──────────────────────────────────────────────


class NNUETrainingExample(NamedTuple):
    """A single training example from self-play (NNUE-based)."""

    features: np.ndarray        # (428,) — NNUE feature vector
    policy_target: np.ndarray   # (29407,) — MCTS visit distribution
    value_target: float         # +1 (won), -1 (lost), 0 (draw)


# ── Replay Buffer ─────────────────────────────────────────────────


class NNUEReplayBuffer:
    """
    Circular replay buffer for NNUE training examples.

    Stores NNUETrainingExample instances from self-play games and
    provides random batch sampling as torch tensors.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self.buffer: deque[NNUETrainingExample] = deque(maxlen=max_size)

    def add_examples(self, examples: list[NNUETrainingExample]) -> None:
        """Add a list of training examples to the buffer."""
        self.buffer.extend(examples)

    def sample_batch(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.

        Returns:
            (features, policy_targets, value_targets) where:
                features: shape (batch_size, 428) float32
                policy_targets: shape (batch_size, 29407) float32
                value_targets: shape (batch_size, 1) float32
        """
        samples = random.sample(
            list(self.buffer), min(batch_size, len(self.buffer))
        )

        features = np.array(
            [s.features for s in samples], dtype=np.float32
        )
        policies = np.array(
            [s.policy_target for s in samples], dtype=np.float32
        )
        values = np.array(
            [s.value_target for s in samples], dtype=np.float32
        ).reshape(-1, 1)

        return (
            torch.from_numpy(features),
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )

    def __len__(self) -> int:
        return len(self.buffer)
