"""Token-based data structures for the Hive Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN_FEAT_DIM = 21
GLOBAL_FEAT_DIM = 6
MAX_SEQ_LEN = 35  # 22 board + 10 hand + 1 CLS + 2 margin

TOKEN_TYPE_CLS = 0
TOKEN_TYPE_BOARD = 1
TOKEN_TYPE_HAND = 2

OFF_BOARD_POSITION = 169  # Grid has 13*13=169 cells; this is the off-board index


# ---------------------------------------------------------------------------
# Single-sequence representation (numpy, CPU)
# ---------------------------------------------------------------------------


@dataclass
class HiveTokenSequence:
    """Numpy-based token sequence for a single Hive game state.

    Token ordering: [CLS, board_piece₁, ..., board_pieceₙ, hand₁, ..., handₘ]

    The first token is always CLS (used by the value head).
    Board piece tokens occupy indices 1..num_board_tokens.
    Hand tokens occupy the remaining indices.
    """

    token_features: np.ndarray     # (S, 21) float32
    token_positions: np.ndarray    # (S,) int32 — 0-168 for board, 169 for off-board
    token_types: np.ndarray        # (S,) int32 — 0=CLS, 1=board, 2=hand
    num_board_tokens: int          # count of board piece tokens (not including CLS)
    global_features: np.ndarray    # (6,) float32

    @property
    def seq_len(self) -> int:
        """Total number of tokens in this sequence."""
        return self.token_features.shape[0]


# ---------------------------------------------------------------------------
# Batched representation (torch, GPU-ready)
# ---------------------------------------------------------------------------


@dataclass
class HiveTokenBatch:
    """Torch-based padded batch of token sequences for GPU training.

    All sequences are padded to the maximum length in the batch.
    The attention_mask indicates which positions are real tokens (True)
    vs padding (False).
    """

    token_features: torch.Tensor     # (B, max_S, 21) float32
    token_positions: torch.Tensor    # (B, max_S) int64
    token_types: torch.Tensor        # (B, max_S) int64
    attention_mask: torch.Tensor     # (B, max_S) bool — True=real, False=pad
    num_board_tokens: torch.Tensor   # (B,) int64
    global_features: torch.Tensor    # (B, 6) float32
    seq_lengths: torch.Tensor        # (B,) int64 — original lengths

    # ---- factory -----------------------------------------------------------

    @staticmethod
    def collate(sequences: list[HiveTokenSequence]) -> HiveTokenBatch:
        """Pad and merge multiple HiveTokenSequence instances into a batch.

        Sequences are padded to the maximum length in the batch with
        zeros for features and OFF_BOARD_POSITION for positions.
        """
        batch_size = len(sequences)
        lengths = [s.seq_len for s in sequences]
        max_len = max(lengths)

        # Pre-allocate padded arrays
        feat_padded = np.zeros(
            (batch_size, max_len, TOKEN_FEAT_DIM), dtype=np.float32
        )
        pos_padded = np.full(
            (batch_size, max_len), OFF_BOARD_POSITION, dtype=np.int32
        )
        types_padded = np.zeros((batch_size, max_len), dtype=np.int32)
        mask = np.zeros((batch_size, max_len), dtype=bool)
        num_board = np.zeros(batch_size, dtype=np.int64)
        global_feats = np.zeros(
            (batch_size, GLOBAL_FEAT_DIM), dtype=np.float32
        )

        for i, seq in enumerate(sequences):
            slen = seq.seq_len
            feat_padded[i, :slen] = seq.token_features
            pos_padded[i, :slen] = seq.token_positions
            types_padded[i, :slen] = seq.token_types
            mask[i, :slen] = True
            num_board[i] = seq.num_board_tokens
            global_feats[i] = seq.global_features

        return HiveTokenBatch(
            token_features=torch.from_numpy(feat_padded),
            token_positions=torch.from_numpy(pos_padded).long(),
            token_types=torch.from_numpy(types_padded).long(),
            attention_mask=torch.from_numpy(mask),
            num_board_tokens=torch.from_numpy(num_board),
            global_features=torch.from_numpy(global_feats),
            seq_lengths=torch.tensor(lengths, dtype=torch.int64),
        )

    # ---- device transfer ---------------------------------------------------

    def to(self, device: torch.device) -> HiveTokenBatch:
        """Return a new HiveTokenBatch with all tensors on *device*."""
        return HiveTokenBatch(
            token_features=self.token_features.to(device),
            token_positions=self.token_positions.to(device),
            token_types=self.token_types.to(device),
            attention_mask=self.attention_mask.to(device),
            num_board_tokens=self.num_board_tokens.to(device),
            global_features=self.global_features.to(device),
            seq_lengths=self.seq_lengths.to(device),
        )
