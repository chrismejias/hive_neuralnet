"""
Token-sequence-aware replay buffer for Transformer training.

Stores (HiveTokenSequence, policy_target, value_target) tuples and provides
batch sampling that returns HiveTokenBatch objects ready for GPU training.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from hive_transformer.token_types import HiveTokenSequence, HiveTokenBatch


# ── Training Example ──────────────────────────────────────────────


class TransformerTrainingExample(NamedTuple):
    """A single training example from self-play (token-based)."""

    sequence: HiveTokenSequence       # Token representation of the state
    policy_target: np.ndarray         # (29407,) — MCTS visit distribution
    value_target: float               # +1 (won), -1 (lost), 0 (draw)


# ── Replay Buffer ─────────────────────────────────────────────────


class TokenReplayBuffer:
    """
    Circular replay buffer for Transformer training examples.

    Stores TransformerTrainingExample instances from self-play games
    and provides random batch sampling. When sampling, token sequences
    are collated into a HiveTokenBatch for efficient batched forward
    passes with padding.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self.buffer: deque[TransformerTrainingExample] = deque(maxlen=max_size)

    def add_examples(self, examples: list[TransformerTrainingExample]) -> None:
        """Add a list of training examples to the buffer."""
        self.buffer.extend(examples)

    def sample_batch(
        self, batch_size: int
    ) -> tuple[HiveTokenBatch, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.

        Returns:
            (token_batch, policy_targets, value_targets) where:
                token_batch: HiveTokenBatch (all tensors on CPU)
                policy_targets: shape (batch_size, 29407) float32
                value_targets: shape (batch_size, 1) float32
        """
        samples = random.sample(
            list(self.buffer), min(batch_size, len(self.buffer))
        )

        sequences = [s.sequence for s in samples]
        policies = np.array(
            [s.policy_target for s in samples], dtype=np.float32
        )
        values = np.array(
            [s.value_target for s in samples], dtype=np.float32
        ).reshape(-1, 1)

        token_batch = HiveTokenBatch.collate(sequences)

        return (
            token_batch,
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )

    def __len__(self) -> int:
        return len(self.buffer)
