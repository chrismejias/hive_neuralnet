"""Replay buffer for PRS (Piece-Relative Space) training examples."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from hive_prs.action_space import ACTION_SPACE_SIZE, MAX_BOARD
from hive_prs.prs_encoder import PRSTokenBatch, PRS_OFF_BOARD


class PRSTrainingExample(NamedTuple):
    """One self-play position stored in the PRS replay buffer."""

    # Token fields (kept as numpy arrays for compact CPU storage)
    token_features:   np.ndarray   # (S, 25) float32
    token_positions:  np.ndarray   # (S,) int32  — 0-528 board, 529 off-board
    token_types:      np.ndarray   # (S,) int8
    num_board_tokens: int
    global_features:  np.ndarray   # (6,) float32
    seq_length:       int

    # PRS-specific
    occupied_cells:   np.ndarray   # (MAX_BOARD,) int32  — -1 = padding
    num_occupied:     int

    # Targets
    policy_target:    np.ndarray   # (ACTION_SPACE_SIZE,) float32
    value_target:     float

    # Sampling weight
    surprise_weight:  float = 1.0


@dataclass
class PRSTrainingBatch:
    """Batched data for one PRS training step."""

    prs_batch:       PRSTokenBatch
    policy_targets:  torch.Tensor   # (B, ACTION_SPACE_SIZE) float32
    value_targets:   torch.Tensor   # (B, 1) float32

    def to(self, device, non_blocking: bool = False) -> PRSTrainingBatch:
        return PRSTrainingBatch(
            prs_batch      = self.prs_batch.to(device, non_blocking=non_blocking),
            policy_targets = self.policy_targets.to(device, non_blocking=non_blocking),
            value_targets  = self.value_targets.to(device, non_blocking=non_blocking),
        )


class PRSReplayBuffer:
    """Circular replay buffer for PRSTrainingExample instances."""

    def __init__(self, max_size: int = 100_000) -> None:
        self.buffer: deque[PRSTrainingExample] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_examples(self, examples: list[PRSTrainingExample]) -> None:
        self.buffer.extend(examples)

    def sample_batch(self, batch_size: int) -> PRSTrainingBatch:
        """Sample a random batch and collate into a PRSTrainingBatch."""
        # Convert deque to list once: enables O(1) random access instead of
        # O(n/4) per deque.__getitem__ call (256 × O(n/4) = slow for large buffers).
        buf = list(self.buffer)
        weights = np.array([e.surprise_weight for e in buf], dtype=np.float32)
        weights /= weights.sum()
        indices = np.random.choice(len(buf), size=batch_size, p=weights)
        samples = [buf[i] for i in indices]
        return _collate(samples)


def _collate(samples: list[PRSTrainingExample]) -> PRSTrainingBatch:
    B = len(samples)
    max_seq = max(s.seq_length for s in samples)

    feat   = np.zeros((B, max_seq, 25),  dtype=np.float32)
    pos    = np.full( (B, max_seq),      PRS_OFF_BOARD, dtype=np.int64)
    types  = np.zeros((B, max_seq),      dtype=np.int64)
    mask   = np.zeros((B, max_seq),      dtype=bool)
    gf     = np.zeros((B, 6),            dtype=np.float32)
    nb     = np.zeros(B,                 dtype=np.int64)
    sl     = np.zeros(B,                 dtype=np.int64)
    occ    = np.full( (B, MAX_BOARD),    -1, dtype=np.int32)
    nocc   = np.zeros(B,                 dtype=np.int32)
    policy = np.zeros((B, ACTION_SPACE_SIZE), dtype=np.float32)
    value  = np.zeros((B, 1),            dtype=np.float32)

    for i, s in enumerate(samples):
        S = s.seq_length
        feat[i,  :S]   = s.token_features[:S]
        pos[ i,  :S]   = s.token_positions[:S]
        types[i, :S]   = s.token_types[:S]
        mask[ i, :S]   = True
        gf[i]          = s.global_features
        nb[i]          = s.num_board_tokens
        sl[i]          = S
        no = s.num_occupied
        occ[i, :no]    = s.occupied_cells[:no]
        nocc[i]        = no
        policy[i]      = s.policy_target
        value[i, 0]    = s.value_target

    prs = PRSTokenBatch(
        token_features   = torch.from_numpy(feat).pin_memory(),
        token_positions  = torch.from_numpy(pos).pin_memory(),
        token_types      = torch.from_numpy(types).pin_memory(),
        attention_mask   = torch.from_numpy(mask).pin_memory(),
        num_board_tokens = torch.from_numpy(nb).pin_memory(),
        global_features  = torch.from_numpy(gf).pin_memory(),
        seq_lengths      = torch.from_numpy(sl).pin_memory(),
        occupied_cells   = torch.from_numpy(occ).pin_memory(),
        num_occupied     = torch.from_numpy(nocc).pin_memory(),
    )

    return PRSTrainingBatch(
        prs_batch      = prs,
        policy_targets = torch.from_numpy(policy).pin_memory(),
        value_targets  = torch.from_numpy(value).pin_memory(),
    )
