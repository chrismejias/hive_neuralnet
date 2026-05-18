from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


class MCTrainingExample(NamedTuple):
    """Root-state training example with targets over its legal move list."""

    state_bytes: np.ndarray      # (SIZEOF_HIVE_STATE,) uint8
    policy_target: np.ndarray    # (n_legal,) float32, aligned with engine legal-move order
    value_target: float
    surprise_weight: float = 1.0


@dataclass
class MCTrainingBatch:
    state_bytes: torch.Tensor       # (B, state_size) uint8
    policy_targets: torch.Tensor    # (B, max_actions) float32
    num_actions: torch.Tensor       # (B,) int64
    value_targets: torch.Tensor     # (B, 1) float32

    def to(self, device, non_blocking: bool = False) -> MCTrainingBatch:
        return MCTrainingBatch(
            state_bytes=self.state_bytes.to(device, non_blocking=non_blocking),
            policy_targets=self.policy_targets.to(device, non_blocking=non_blocking),
            num_actions=self.num_actions.to(device, non_blocking=non_blocking),
            value_targets=self.value_targets.to(device, non_blocking=non_blocking),
        )


class MCReplayBuffer:
    def __init__(self, max_size: int = 100_000) -> None:
        self.buffer: deque[MCTrainingExample] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_examples(self, examples: list[MCTrainingExample]) -> None:
        self.buffer.extend(examples)

    def sample_batch(self, batch_size: int) -> MCTrainingBatch:
        buf = list(self.buffer)
        weights = np.array([e.surprise_weight for e in buf], dtype=np.float32)
        weights /= weights.sum()
        indices = np.random.choice(len(buf), size=batch_size, p=weights)
        samples = [buf[i] for i in indices]
        return _collate(samples)


def _collate(samples: list[MCTrainingExample]) -> MCTrainingBatch:
    B = len(samples)
    state_size = max(s.state_bytes.shape[0] for s in samples)
    max_actions = max(s.policy_target.shape[0] for s in samples)

    states = np.zeros((B, state_size), dtype=np.uint8)
    policy = np.zeros((B, max_actions), dtype=np.float32)
    num_actions = np.zeros((B,), dtype=np.int64)
    values = np.zeros((B, 1), dtype=np.float32)

    for i, s in enumerate(samples):
        states[i, :s.state_bytes.shape[0]] = s.state_bytes
        n = s.policy_target.shape[0]
        policy[i, :n] = s.policy_target
        num_actions[i] = n
        values[i, 0] = s.value_target

    return MCTrainingBatch(
        state_bytes=torch.from_numpy(states).pin_memory(),
        policy_targets=torch.from_numpy(policy).pin_memory(),
        num_actions=torch.from_numpy(num_actions).pin_memory(),
        value_targets=torch.from_numpy(values).pin_memory(),
    )
