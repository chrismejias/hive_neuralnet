"""
Graph-aware replay buffer for GNN training.

Stores (HiveGraph, policy_target, value_target) tuples and provides
batch sampling that returns HiveGraphBatch objects ready for GPU training.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from hive_gnn.graph_types import HiveGraph, HiveGraphBatch


# ── Training Example ──────────────────────────────────────────────


class GNNTrainingExample(NamedTuple):
    """A single training example from self-play (graph-based)."""

    graph: HiveGraph                # Graph representation of the state
    policy_target: np.ndarray       # (29407,) — MCTS visit distribution
    value_target: float             # +1 (won), -1 (lost), 0 (draw)


# ── Replay Buffer ─────────────────────────────────────────────────


class GraphReplayBuffer:
    """
    Circular replay buffer for GNN training examples.

    Stores GNNTrainingExample instances from self-play games and
    provides random batch sampling. When sampling, graphs are
    collated into a HiveGraphBatch for efficient batched forward
    passes.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self.buffer: deque[GNNTrainingExample] = deque(maxlen=max_size)

    def add_examples(self, examples: list[GNNTrainingExample]) -> None:
        """Add a list of training examples to the buffer."""
        self.buffer.extend(examples)

    def sample_batch(
        self, batch_size: int
    ) -> tuple[HiveGraphBatch, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch from the buffer.

        Returns:
            (graph_batch, policy_targets, value_targets) where:
                graph_batch: HiveGraphBatch (all tensors on CPU)
                policy_targets: shape (batch_size, 29407) float32
                value_targets: shape (batch_size, 1) float32
        """
        samples = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

        graphs = [s.graph for s in samples]
        policies = np.array([s.policy_target for s in samples], dtype=np.float32)
        values = np.array(
            [s.value_target for s in samples], dtype=np.float32
        ).reshape(-1, 1)

        graph_batch = HiveGraphBatch.collate(graphs)

        return (
            graph_batch,
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )

    def __len__(self) -> int:
        return len(self.buffer)
