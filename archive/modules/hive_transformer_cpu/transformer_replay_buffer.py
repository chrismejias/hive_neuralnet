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
    TOKEN_FEAT_DIM,
    OFF_BOARD_POSITION,
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
    surprise_weight: float = 1.0          # KL-based sample weight for surprise weighting


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
        Sample a batch from the buffer, weighted by surprise_weight.

        Examples with higher policy surprise (KL divergence between NN
        prior and MCTS result) are sampled more frequently.

        Returns:
            A TransformerTrainingBatch with all tensors on CPU.
        """
        buf_list = list(self.buffer)
        n = min(batch_size, len(buf_list))
        weights = np.array([ex.surprise_weight for ex in buf_list], dtype=np.float64)
        w_sum = weights.sum()
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones(len(buf_list), dtype=np.float64) / len(buf_list)
        # Fix floating-point rounding so probabilities sum to exactly 1
        weights /= weights.sum()
        indices = np.random.choice(len(buf_list), size=n, replace=False, p=weights)
        samples = [buf_list[i] for i in indices]

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


# ── GPU-resident Replay Buffer ────────────────────────────────────────


class GPUTokenReplayBuffer:
    """GPU-resident circular replay buffer for Transformer training.

    All data is stored as pre-padded GPU tensors, eliminating CPU collation
    overhead during sampling.  Variable-length token sequences are padded to
    MAX_SEQ_LEN; variable-length board-token targets are padded to
    MAX_BOARD_TOKENS.  Sampling uses GPU-side multinomial selection and
    masked gather, so the returned TransformerTrainingBatch is already on GPU.

    Memory (50 k examples, float16 where possible):
      - policy_targets (29407-dim):  ~2.9 GB
      - token features + other data: ~0.2 GB
      Total: ~3.1 GB
    """

    MAX_SEQ_LEN = 64
    MAX_BOARD_TOKENS = 64
    GLOBAL_FEAT_DIM = 6

    def __init__(
        self,
        max_size: int = 50_000,
        device: str = "cuda",
    ) -> None:
        self.max_size = max_size
        self.device = torch.device(device)
        self._size = 0
        self._write_idx = 0
        self.action_space_size: int | None = None  # set on first add_examples

        dev = self.device
        M = max_size
        S = self.MAX_SEQ_LEN
        N = self.MAX_BOARD_TOKENS
        F = TOKEN_FEAT_DIM
        G = self.GLOBAL_FEAT_DIM

        # Token sequence (float16 to save memory)
        self._token_features     = torch.zeros(M, S, F, dtype=torch.float16, device=dev)
        self._token_positions    = torch.full((M, S), OFF_BOARD_POSITION, dtype=torch.int32, device=dev)
        self._token_types        = torch.zeros(M, S, dtype=torch.int32, device=dev)
        self._seq_lengths        = torch.zeros(M, dtype=torch.int32, device=dev)
        self._num_board_tokens   = torch.zeros(M, dtype=torch.int32, device=dev)
        self._global_features    = torch.zeros(M, G, dtype=torch.float16, device=dev)

        # Training targets (policy lazily allocated on first add_examples)
        self._policy_targets: torch.Tensor | None = None
        self._value_targets           = torch.zeros(M, dtype=torch.float32, device=dev)
        self._mobility_targets        = torch.zeros(M, N, dtype=torch.float16, device=dev)
        self._queen_surround_targets  = torch.zeros(M, N, 2, dtype=torch.float16, device=dev)
        self._queen_surround_mask     = torch.zeros(M, 2, dtype=torch.float16, device=dev)
        self._final_mobility_targets  = torch.zeros(M, N, dtype=torch.float16, device=dev)
        self._value_mask              = torch.zeros(M, dtype=torch.float32, device=dev)
        self._surprise_weights        = torch.ones(M, dtype=torch.float32, device=dev)

    def _ensure_policy_buffer(self, action_space_size: int) -> None:
        if self._policy_targets is None:
            self.action_space_size = action_space_size
            self._policy_targets = torch.zeros(
                self.max_size, action_space_size, dtype=torch.float16, device=self.device
            )

    def add_examples(self, examples: list[TransformerTrainingExample]) -> None:
        """Batch-upload examples to GPU using a single transfer per tensor."""
        n = len(examples)
        if n == 0:
            return

        A = len(examples[0].policy_target)
        self._ensure_policy_buffer(A)

        S = self.MAX_SEQ_LEN
        N = self.MAX_BOARD_TOKENS
        F = TOKEN_FEAT_DIM
        G = self.GLOBAL_FEAT_DIM

        # Build CPU arrays
        feat_buf    = np.zeros((n, S, F), dtype=np.float32)
        pos_buf     = np.full((n, S), OFF_BOARD_POSITION, dtype=np.int32)
        types_buf   = np.zeros((n, S), dtype=np.int32)
        slen_buf    = np.zeros(n, dtype=np.int32)
        nboard_buf  = np.zeros(n, dtype=np.int32)
        gfeat_buf   = np.zeros((n, G), dtype=np.float32)
        pol_buf     = np.zeros((n, A), dtype=np.float32)
        val_buf     = np.zeros(n, dtype=np.float32)
        mob_buf     = np.zeros((n, N), dtype=np.float32)
        qs_buf      = np.zeros((n, N, 2), dtype=np.float32)
        qsmask_buf  = np.zeros((n, 2), dtype=np.float32)
        fmob_buf    = np.zeros((n, N), dtype=np.float32)
        vmask_buf   = np.zeros(n, dtype=np.float32)
        sw_buf      = np.ones(n, dtype=np.float32)

        for k, ex in enumerate(examples):
            seq = ex.sequence
            slen = min(seq.seq_len, S)
            nb   = min(seq.num_board_tokens, N)

            feat_buf[k, :slen]   = seq.token_features[:slen]
            pos_buf[k, :slen]    = seq.token_positions[:slen]
            types_buf[k, :slen]  = seq.token_types[:slen]
            slen_buf[k]          = slen
            nboard_buf[k]        = nb
            gfeat_buf[k]         = seq.global_features

            pol_buf[k]           = ex.policy_target
            val_buf[k]           = ex.value_target
            if nb > 0:
                mob_buf[k, :nb]  = ex.mobility_target[:nb]
                qs_buf[k, :nb]   = ex.queen_surround_target[:nb]
                fmob_buf[k, :nb] = ex.final_mobility_target[:nb]
            qsmask_buf[k]        = ex.queen_surround_mask
            vmask_buf[k]         = 1.0 if ex.use_for_value else 0.0
            sw_buf[k]            = ex.surprise_weight

        # Compute write indices (circular)
        start = self._write_idx
        if start + n <= self.max_size:
            idx = slice(start, start + n)
            self._write_idx = (start + n) % self.max_size
        else:
            # Wrap-around: upload in two shots
            first = self.max_size - start
            self._upload_slice(slice(start, self.max_size),
                               feat_buf[:first], pos_buf[:first], types_buf[:first],
                               slen_buf[:first], nboard_buf[:first], gfeat_buf[:first],
                               pol_buf[:first], val_buf[:first], mob_buf[:first],
                               qs_buf[:first], qsmask_buf[:first], fmob_buf[:first],
                               vmask_buf[:first], sw_buf[:first])
            self._upload_slice(slice(0, n - first),
                               feat_buf[first:], pos_buf[first:], types_buf[first:],
                               slen_buf[first:], nboard_buf[first:], gfeat_buf[first:],
                               pol_buf[first:], val_buf[first:], mob_buf[first:],
                               qs_buf[first:], qsmask_buf[first:], fmob_buf[first:],
                               vmask_buf[first:], sw_buf[first:])
            self._write_idx = n - first
            self._size = min(self._size + n, self.max_size)
            return

        self._upload_slice(idx,
                           feat_buf, pos_buf, types_buf, slen_buf, nboard_buf, gfeat_buf,
                           pol_buf, val_buf, mob_buf, qs_buf, qsmask_buf, fmob_buf,
                           vmask_buf, sw_buf)
        self._size = min(self._size + n, self.max_size)

    def _upload_slice(self, idx, feat, pos, types, slen, nboard, gfeat,
                      pol, val, mob, qs, qsmask, fmob, vmask, sw) -> None:
        dev = self.device
        self._token_features[idx]            = torch.from_numpy(feat).half().to(dev)
        self._token_positions[idx]           = torch.from_numpy(pos).to(dev)
        self._token_types[idx]               = torch.from_numpy(types).to(dev)
        self._seq_lengths[idx]               = torch.from_numpy(slen).to(dev)
        self._num_board_tokens[idx]          = torch.from_numpy(nboard).to(dev)
        self._global_features[idx]           = torch.from_numpy(gfeat).half().to(dev)
        self._policy_targets[idx]            = torch.from_numpy(pol).half().to(dev)
        self._value_targets[idx]             = torch.from_numpy(val).to(dev)
        self._mobility_targets[idx]          = torch.from_numpy(mob).half().to(dev)
        self._queen_surround_targets[idx]    = torch.from_numpy(qs).half().to(dev)
        self._queen_surround_mask[idx]       = torch.from_numpy(qsmask).half().to(dev)
        self._final_mobility_targets[idx]    = torch.from_numpy(fmob).half().to(dev)
        self._value_mask[idx]                = torch.from_numpy(vmask).to(dev)
        self._surprise_weights[idx]          = torch.from_numpy(sw).to(dev)

    def sample_batch(self, batch_size: int) -> TransformerTrainingBatch:
        """GPU-side weighted sampling — returns a batch already on GPU."""
        n = min(batch_size, self._size)
        dev = self.device

        # Weighted multinomial sampling entirely on GPU
        weights = self._surprise_weights[:self._size]
        indices = torch.multinomial(weights, n, replacement=False)

        # Gather token sequences
        seq_lengths      = self._seq_lengths[indices].long()       # (n,)
        num_board_tokens = self._num_board_tokens[indices].long()  # (n,)
        max_len          = int(seq_lengths.max().item())

        token_features  = self._token_features[indices, :max_len].float()   # (n, max_len, F)
        token_positions = self._token_positions[indices, :max_len].long()   # (n, max_len)
        token_types     = self._token_types[indices, :max_len].long()       # (n, max_len)
        global_features = self._global_features[indices].float()            # (n, G)

        # Attention mask from seq_lengths
        rng = torch.arange(max_len, device=dev).unsqueeze(0)               # (1, max_len)
        attention_mask = rng < seq_lengths.unsqueeze(1)                    # (n, max_len)

        token_batch = HiveTokenBatch(
            token_features=token_features,
            token_positions=token_positions,
            token_types=token_types,
            attention_mask=attention_mask,
            num_board_tokens=num_board_tokens,
            global_features=global_features,
            seq_lengths=seq_lengths,
        )

        assert self._policy_targets is not None, "sample_batch called before add_examples"
        policy_targets       = self._policy_targets[indices].float()       # (n, A)
        value_targets        = self._value_targets[indices].unsqueeze(1)   # (n, 1)
        value_mask           = self._value_mask[indices]                   # (n,)
        queen_surround_mask  = self._queen_surround_mask[indices].float()  # (n, 2)

        # Variable-length board-token targets: vectorised masked gather
        max_board = int(num_board_tokens.max().item()) if self._size > 0 else 0
        if max_board > 0:
            j = torch.arange(max_board, device=dev).unsqueeze(0)           # (1, max_board)
            board_mask = j < num_board_tokens.unsqueeze(1)                 # (n, max_board)

            mob_padded   = self._mobility_targets[indices, :max_board].float()      # (n, max_board)
            qs_padded    = self._queen_surround_targets[indices, :max_board].float()# (n, max_board, 2)
            fmob_padded  = self._final_mobility_targets[indices, :max_board].float()# (n, max_board)

            mobility_targets       = mob_padded[board_mask]    # (total_board_tokens,)
            queen_surround_targets = qs_padded[board_mask]     # (total_board_tokens, 2)
            final_mobility_targets = fmob_padded[board_mask]   # (total_board_tokens,)

            seq_idx = torch.arange(n, device=dev).unsqueeze(1).expand(-1, max_board)
            board_token_batch = seq_idx[board_mask].long()     # (total_board_tokens,)
        else:
            mobility_targets       = torch.zeros(0, device=dev)
            queen_surround_targets = torch.zeros(0, 2, device=dev)
            final_mobility_targets = torch.zeros(0, device=dev)
            board_token_batch      = torch.zeros(0, dtype=torch.int64, device=dev)

        return TransformerTrainingBatch(
            token_batch=token_batch,
            policy_targets=policy_targets,
            value_targets=value_targets,
            mobility_targets=mobility_targets,
            queen_surround_targets=queen_surround_targets,
            queen_surround_mask=queen_surround_mask,
            final_mobility_targets=final_mobility_targets,
            value_mask=value_mask,
            board_token_batch=board_token_batch,
        )

    def __len__(self) -> int:
        return self._size
