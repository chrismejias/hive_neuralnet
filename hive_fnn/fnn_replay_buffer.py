"""Replay buffer for FNN training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


class FNNTrainingExample(NamedTuple):
    """Root-state training example with policy over its legal move list."""

    state_bytes: np.ndarray      # (SIZEOF_HIVE_STATE,) uint8
    policy_target: np.ndarray    # (n_legal,) float32
    value_target: float
    use_for_value: bool = True
    surprise_weight: float = 1.0


@dataclass
class FNNTrainingBatch:
    state_bytes: torch.Tensor       # (B, state_size) uint8
    policy_targets: torch.Tensor    # (B, max_actions) float32
    num_actions: torch.Tensor       # (B,) int64
    value_targets: torch.Tensor     # (B, 1) float32
    value_mask: torch.Tensor        # (B, 1) float32

    def to(self, device: torch.device, non_blocking: bool = False) -> FNNTrainingBatch:
        return FNNTrainingBatch(
            state_bytes=self.state_bytes.to(device, non_blocking=non_blocking),
            policy_targets=self.policy_targets.to(device, non_blocking=non_blocking),
            num_actions=self.num_actions.to(device, non_blocking=non_blocking),
            value_targets=self.value_targets.to(device, non_blocking=non_blocking),
            value_mask=self.value_mask.to(device, non_blocking=non_blocking),
        )


class FNNReplayBuffer:
    def __init__(self, max_size: int = 100_000) -> None:
        self.max_size = int(max_size)
        self.buffer: list[FNNTrainingExample] = []
        self._next_idx = 0
        self._weights_dirty = True
        self._cached_weights: np.ndarray | None = None
        self._uniform_weights = True

    def __len__(self) -> int:
        return len(self.buffer)

    def add_examples(self, examples: list[FNNTrainingExample]) -> None:
        if not examples:
            return
        buf = self.buffer
        max_size = self.max_size
        next_idx = self._next_idx
        uniform = self._uniform_weights

        for ex in examples:
            if len(buf) < max_size:
                buf.append(ex)
            else:
                buf[next_idx] = ex
                next_idx = (next_idx + 1) % max_size
            if ex.surprise_weight != 1.0:
                uniform = False

        self._next_idx = next_idx
        self._uniform_weights = uniform
        self._weights_dirty = True

    def _get_weights(self) -> np.ndarray:
        if self._weights_dirty or self._cached_weights is None or len(self._cached_weights) != len(self.buffer):
            self._cached_weights = np.fromiter(
                (e.surprise_weight for e in self.buffer),
                dtype=np.float32,
                count=len(self.buffer),
            )
            self._cached_weights /= self._cached_weights.sum()
            self._weights_dirty = False
        return self._cached_weights

    def sample_batch(self, batch_size: int) -> FNNTrainingBatch:
        buf = self.buffer
        n = len(buf)
        if self._uniform_weights:
            indices = np.random.randint(0, n, size=batch_size)
        else:
            weights = self._get_weights()
            indices = np.random.choice(n, size=batch_size, p=weights)
        samples = [buf[i] for i in indices]
        return _collate(samples)


def _collate(samples: list[FNNTrainingExample]) -> FNNTrainingBatch:
    B = len(samples)
    state_size = max(s.state_bytes.shape[0] for s in samples)
    max_actions = max(s.policy_target.shape[0] for s in samples)

    states = np.zeros((B, state_size), dtype=np.uint8)
    policy = np.zeros((B, max_actions), dtype=np.float32)
    num_actions = np.zeros((B,), dtype=np.int64)
    values = np.zeros((B, 1), dtype=np.float32)
    value_mask = np.zeros((B, 1), dtype=np.float32)

    for i, s in enumerate(samples):
        states[i, : s.state_bytes.shape[0]] = s.state_bytes
        n = s.policy_target.shape[0]
        policy[i, :n] = s.policy_target
        num_actions[i] = n
        values[i, 0] = s.value_target
        value_mask[i, 0] = float(s.use_for_value)

    return FNNTrainingBatch(
        state_bytes=torch.from_numpy(states).pin_memory(),
        policy_targets=torch.from_numpy(policy).pin_memory(),
        num_actions=torch.from_numpy(num_actions).pin_memory(),
        value_targets=torch.from_numpy(values).pin_memory(),
        value_mask=torch.from_numpy(value_mask).pin_memory(),
    )
