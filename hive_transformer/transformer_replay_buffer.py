"""
Token-sequence-aware replay buffer for Transformer training.

Stores (HiveTokenSequence, policy_target, value_target, aux_targets) tuples
and provides batch sampling that returns TransformerTrainingBatch objects
ready for GPU training.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from hive_transformer.token_types import (
    HiveTokenSequence,
    HiveTokenBatch,
    TOKEN_TYPE_BOARD,
)


# ── Training Example ──────────────────────────────────────────────


class TransformerTrainingExample(NamedTuple):
    """A single training example from self-play (token-based)."""

    sequence: HiveTokenSequence           # Token representation of the state
    policy_target: np.ndarray             # (29407,) — MCTS visit distribution
    value_target: float                   # +1 (won), -1 (lost), 0 (draw)
    mobility_target: np.ndarray           # (num_board_tokens,) float32 binary
    queen_surround_target: np.ndarray     # (num_board_tokens, 2) float32 binary
    queen_surround_mask: np.ndarray       # (2,) float32 — 1.0 if queen placed
    final_mobility_target: np.ndarray     # (num_board_tokens,) float32 binary
    use_for_value: bool                   # playout cap: only full-playout games train value


# ── Training Batch ────────────────────────────────────────────────


@dataclass
class TransformerTrainingBatch:
    """Batched training data for one training step."""

    token_batch: HiveTokenBatch
    policy_targets: torch.Tensor            # (B, 29407)
    value_targets: torch.Tensor             # (B, 1)
    mobility_targets: torch.Tensor          # (total_board_tokens,)
    queen_surround_targets: torch.Tensor    # (total_board_tokens, 2)
    queen_surround_mask: torch.Tensor       # (B, 2)
    final_mobility_targets: torch.Tensor    # (total_board_tokens,)
    value_mask: torch.Tensor                # (B,) — 1.0 if use_for_value
    board_token_batch: torch.Tensor         # (total_board_tokens,) — seq index per board token

    def to(self, device: torch.device) -> TransformerTrainingBatch:
        """Return a new batch with all tensors on *device*."""
        return TransformerTrainingBatch(
            token_batch=self.token_batch.to(device),
            policy_targets=self.policy_targets.to(device),
            value_targets=self.value_targets.to(device),
            mobility_targets=self.mobility_targets.to(device),
            queen_surround_targets=self.queen_surround_targets.to(device),
            queen_surround_mask=self.queen_surround_mask.to(device),
            final_mobility_targets=self.final_mobility_targets.to(device),
            value_mask=self.value_mask.to(device),
            board_token_batch=self.board_token_batch.to(device),
        )


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
    ) -> TransformerTrainingBatch:
        """
        Sample a random batch from the buffer.

        Returns:
            A TransformerTrainingBatch with all tensors on CPU.
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

        # Concatenate per-board-token auxiliary targets
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
            [1.0 if s.use_for_value else 0.0 for s in samples],
            dtype=np.float32,
        )

        # Build board_token_batch: sequence index for each board token
        board_token_batch_list: list[np.ndarray] = []
        for i, seq in enumerate(sequences):
            n_board = seq.num_board_tokens
            if n_board > 0:
                board_token_batch_list.append(
                    np.full(n_board, i, dtype=np.int64)
                )
        if board_token_batch_list:
            board_token_batch = np.concatenate(board_token_batch_list, axis=0)
        else:
            board_token_batch = np.zeros((0,), dtype=np.int64)

        token_batch = HiveTokenBatch.collate(sequences)

        return TransformerTrainingBatch(
            token_batch=token_batch,
            policy_targets=torch.from_numpy(policies),
            value_targets=torch.from_numpy(values),
            mobility_targets=torch.from_numpy(mobility_targets),
            queen_surround_targets=torch.from_numpy(queen_surround_targets),
            queen_surround_mask=torch.from_numpy(queen_surround_mask),
            final_mobility_targets=torch.from_numpy(final_mobility_targets),
            value_mask=torch.from_numpy(value_mask),
            board_token_batch=torch.from_numpy(board_token_batch),
        )

    def __len__(self) -> int:
        return len(self.buffer)
