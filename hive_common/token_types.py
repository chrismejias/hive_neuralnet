"""Token-based data structures shared by the live tokenized model paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

TOKEN_FEAT_DIM = 25
GLOBAL_FEAT_DIM = 6
MAX_SEQ_LEN = 55

TOKEN_TYPE_CLS = 0
TOKEN_TYPE_BOARD = 1
TOKEN_TYPE_HAND = 2

OFF_BOARD_POSITION = 289


@dataclass
class HiveTokenSequence:
    token_features: np.ndarray
    token_positions: np.ndarray
    token_types: np.ndarray
    num_board_tokens: int
    global_features: np.ndarray

    @property
    def seq_len(self) -> int:
        return self.token_features.shape[0]


@dataclass
class HiveTokenBatch:
    token_features: torch.Tensor
    token_positions: torch.Tensor
    token_types: torch.Tensor
    attention_mask: torch.Tensor
    num_board_tokens: torch.Tensor
    global_features: torch.Tensor
    seq_lengths: torch.Tensor

    @staticmethod
    def collate(sequences: list[HiveTokenSequence]) -> "HiveTokenBatch":
        batch_size = len(sequences)
        lengths = [s.seq_len for s in sequences]
        max_len = max(lengths)

        feat_padded = np.zeros((batch_size, max_len, TOKEN_FEAT_DIM), dtype=np.float32)
        pos_padded = np.full((batch_size, max_len), OFF_BOARD_POSITION, dtype=np.int32)
        types_padded = np.zeros((batch_size, max_len), dtype=np.int32)
        mask = np.zeros((batch_size, max_len), dtype=bool)
        num_board = np.zeros(batch_size, dtype=np.int64)
        global_feats = np.zeros((batch_size, GLOBAL_FEAT_DIM), dtype=np.float32)

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
