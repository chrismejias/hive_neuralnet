"""
Graph-aware replay buffer for GNN training.

Stores (HiveGraph, policy_target, value_target, aux_targets) tuples and
provides batch sampling that returns GNNTrainingBatch objects ready for
GPU training.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from hive_gnn.graph_types import HiveGraph, HiveGraphBatch


# ── Training Example ──────────────────────────────────────────────


class GNNTrainingExample(NamedTuple):
    """A single training example from self-play (graph-based)."""

    graph: HiveGraph                        # Graph representation of the state
    policy_target: np.ndarray               # (29407,) — MCTS visit distribution
    value_target: float                     # +1 (won), -1 (lost), 0 (draw)
    mobility_target: np.ndarray             # (num_piece_nodes,) float32 binary
    queen_surround_target: np.ndarray       # (num_piece_nodes, 2) float32 binary
    queen_surround_mask: np.ndarray         # (2,) float32 — 1.0 if queen placed
    final_mobility_target: np.ndarray       # (num_piece_nodes,) float32 binary — end-of-game mobility
    use_for_value: bool                     # playout cap: only full-playout games train value


# ── Training Batch ────────────────────────────────────────────────


@dataclass
class GNNTrainingBatch:
    """Batched training data for one training step."""

    graph_batch: HiveGraphBatch
    policy_targets: torch.Tensor            # (B, 29407)
    value_targets: torch.Tensor             # (B, 1)
    mobility_targets: torch.Tensor          # (total_piece_nodes,)
    queen_surround_targets: torch.Tensor    # (total_piece_nodes, 2)
    queen_surround_mask: torch.Tensor       # (B, 2)
    final_mobility_targets: torch.Tensor    # (total_piece_nodes,)
    value_mask: torch.Tensor                # (B,) — 1.0 if use_for_value

    def to(self, device: torch.device) -> GNNTrainingBatch:
        """Return a new batch with all tensors on *device*."""
        return GNNTrainingBatch(
            graph_batch=self.graph_batch.to(device),
            policy_targets=self.policy_targets.to(device),
            value_targets=self.value_targets.to(device),
            mobility_targets=self.mobility_targets.to(device),
            queen_surround_targets=self.queen_surround_targets.to(device),
            queen_surround_mask=self.queen_surround_mask.to(device),
            final_mobility_targets=self.final_mobility_targets.to(device),
            value_mask=self.value_mask.to(device),
        )


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

    def sample_batch(self, batch_size: int) -> GNNTrainingBatch:
        """
        Sample a random batch from the buffer.

        Returns:
            A GNNTrainingBatch with all tensors on CPU.
        """
        samples = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

        graphs = [s.graph for s in samples]
        policies = np.array([s.policy_target for s in samples], dtype=np.float32)
        values = np.array(
            [s.value_target for s in samples], dtype=np.float32
        ).reshape(-1, 1)

        # Concatenate per-node auxiliary targets (aligned with piece_node_batch)
        mobility_list = [s.mobility_target for s in samples]
        surround_list = [s.queen_surround_target for s in samples]
        final_mob_list = [s.final_mobility_target for s in samples]

        if any(m.shape[0] > 0 for m in mobility_list):
            mobility_targets = np.concatenate(mobility_list, axis=0)
        else:
            mobility_targets = np.zeros((0,), dtype=np.float32)

        if any(s.shape[0] > 0 for s in surround_list):
            queen_surround_targets = np.concatenate(surround_list, axis=0)
        else:
            queen_surround_targets = np.zeros((0, 2), dtype=np.float32)

        if any(fm.shape[0] > 0 for fm in final_mob_list):
            final_mobility_targets = np.concatenate(final_mob_list, axis=0)
        else:
            final_mobility_targets = np.zeros((0,), dtype=np.float32)

        queen_surround_mask = np.array(
            [s.queen_surround_mask for s in samples], dtype=np.float32
        )

        value_mask = np.array(
            [1.0 if s.use_for_value else 0.0 for s in samples], dtype=np.float32
        )

        graph_batch = HiveGraphBatch.collate(graphs)

        return GNNTrainingBatch(
            graph_batch=graph_batch,
            policy_targets=torch.from_numpy(policies),
            value_targets=torch.from_numpy(values),
            mobility_targets=torch.from_numpy(mobility_targets),
            queen_surround_targets=torch.from_numpy(queen_surround_targets),
            queen_surround_mask=torch.from_numpy(queen_surround_mask),
            final_mobility_targets=torch.from_numpy(final_mobility_targets),
            value_mask=torch.from_numpy(value_mask),
        )

    def __len__(self) -> int:
        return len(self.buffer)
