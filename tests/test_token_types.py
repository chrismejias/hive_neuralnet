"""Tests for hive_transformer.token_types data structures."""

import numpy as np
import pytest
import torch

from hive_transformer.token_types import (
    TOKEN_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    TOKEN_TYPE_CLS,
    TOKEN_TYPE_BOARD,
    TOKEN_TYPE_HAND,
    OFF_BOARD_POSITION,
    HiveTokenSequence,
    HiveTokenBatch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sequence(
    num_board: int = 3,
    num_hand: int = 2,
) -> HiveTokenSequence:
    """Create a minimal HiveTokenSequence."""
    total = 1 + num_board + num_hand  # CLS + board + hand

    features = np.random.randn(total, TOKEN_FEAT_DIM).astype(np.float32)

    positions = np.full(total, OFF_BOARD_POSITION, dtype=np.int32)
    # Board tokens get valid grid positions
    for i in range(num_board):
        positions[1 + i] = np.random.randint(0, 169)

    types = np.zeros(total, dtype=np.int32)
    types[0] = TOKEN_TYPE_CLS
    types[1:1 + num_board] = TOKEN_TYPE_BOARD
    types[1 + num_board:] = TOKEN_TYPE_HAND

    global_features = np.random.randn(GLOBAL_FEAT_DIM).astype(np.float32)

    return HiveTokenSequence(
        token_features=features,
        token_positions=positions,
        token_types=types,
        num_board_tokens=num_board,
        global_features=global_features,
    )


# ---------------------------------------------------------------------------
# TestHiveTokenSequence
# ---------------------------------------------------------------------------


class TestHiveTokenSequence:
    def test_create_simple(self):
        seq = _make_sequence(3, 2)
        assert seq.token_features.shape == (6, TOKEN_FEAT_DIM)
        assert seq.token_positions.shape == (6,)
        assert seq.token_types.shape == (6,)
        assert seq.num_board_tokens == 3
        assert seq.global_features.shape == (GLOBAL_FEAT_DIM,)

    def test_seq_len(self):
        seq = _make_sequence(4, 3)
        assert seq.seq_len == 8  # 1 CLS + 4 board + 3 hand

    def test_token_ordering(self):
        seq = _make_sequence(2, 1)
        assert seq.token_types[0] == TOKEN_TYPE_CLS
        assert seq.token_types[1] == TOKEN_TYPE_BOARD
        assert seq.token_types[2] == TOKEN_TYPE_BOARD
        assert seq.token_types[3] == TOKEN_TYPE_HAND

    def test_cls_position_off_board(self):
        seq = _make_sequence(2, 1)
        assert seq.token_positions[0] == OFF_BOARD_POSITION

    def test_hand_positions_off_board(self):
        seq = _make_sequence(2, 3)
        for i in range(3):  # 3 hand tokens
            assert seq.token_positions[3 + i] == OFF_BOARD_POSITION

    def test_board_positions_in_range(self):
        seq = _make_sequence(5, 0)
        for i in range(5):
            pos = seq.token_positions[1 + i]
            assert 0 <= pos <= 168  # 13*13 - 1

    def test_empty_board(self):
        seq = _make_sequence(0, 10)
        assert seq.seq_len == 11  # CLS + 10 hand
        assert seq.num_board_tokens == 0


# ---------------------------------------------------------------------------
# TestHiveTokenBatch
# ---------------------------------------------------------------------------


class TestHiveTokenBatch:
    def test_collate_single(self):
        seq = _make_sequence(3, 2)
        batch = HiveTokenBatch.collate([seq])

        assert batch.token_features.shape == (1, 6, TOKEN_FEAT_DIM)
        assert batch.token_positions.shape == (1, 6)
        assert batch.token_types.shape == (1, 6)
        assert batch.attention_mask.shape == (1, 6)
        assert batch.attention_mask.all()  # No padding for single sequence
        assert batch.num_board_tokens.shape == (1,)
        assert batch.num_board_tokens[0] == 3
        assert batch.global_features.shape == (1, GLOBAL_FEAT_DIM)
        assert batch.seq_lengths.shape == (1,)
        assert batch.seq_lengths[0] == 6

    def test_collate_multiple_same_length(self):
        seqs = [_make_sequence(3, 2) for _ in range(4)]
        batch = HiveTokenBatch.collate(seqs)

        assert batch.token_features.shape == (4, 6, TOKEN_FEAT_DIM)
        assert batch.attention_mask.all()  # All same length, no padding

    def test_collate_different_lengths(self):
        s1 = _make_sequence(2, 1)   # len 4
        s2 = _make_sequence(5, 3)   # len 9
        s3 = _make_sequence(0, 2)   # len 3

        batch = HiveTokenBatch.collate([s1, s2, s3])

        assert batch.token_features.shape == (3, 9, TOKEN_FEAT_DIM)  # padded to max
        assert batch.token_positions.shape == (3, 9)
        assert batch.attention_mask.shape == (3, 9)

        # Check masks
        assert batch.attention_mask[0, :4].all()
        assert not batch.attention_mask[0, 4:].any()
        assert batch.attention_mask[1, :9].all()
        assert batch.attention_mask[2, :3].all()
        assert not batch.attention_mask[2, 3:].any()

    def test_padding_values(self):
        s1 = _make_sequence(1, 0)  # len 2
        s2 = _make_sequence(3, 1)  # len 5

        batch = HiveTokenBatch.collate([s1, s2])

        # Padded positions should be OFF_BOARD_POSITION
        assert batch.token_positions[0, 2] == OFF_BOARD_POSITION
        assert batch.token_positions[0, 3] == OFF_BOARD_POSITION

        # Padded features should be zeros
        assert (batch.token_features[0, 2:] == 0).all()

    def test_to_device(self):
        seq = _make_sequence(2, 1)
        batch = HiveTokenBatch.collate([seq])

        cpu = torch.device("cpu")
        batch_cpu = batch.to(cpu)

        assert batch_cpu.token_features.device == cpu
        assert batch_cpu.token_positions.device == cpu
        assert batch_cpu.token_types.device == cpu
        assert batch_cpu.attention_mask.device == cpu
        assert batch_cpu.num_board_tokens.device == cpu
        assert batch_cpu.global_features.device == cpu
        assert batch_cpu.seq_lengths.device == cpu

    def test_dtypes(self):
        seq = _make_sequence(2, 1)
        batch = HiveTokenBatch.collate([seq])

        assert batch.token_features.dtype == torch.float32
        assert batch.token_positions.dtype == torch.int64
        assert batch.token_types.dtype == torch.int64
        assert batch.attention_mask.dtype == torch.bool
        assert batch.num_board_tokens.dtype == torch.int64
        assert batch.global_features.dtype == torch.float32
        assert batch.seq_lengths.dtype == torch.int64

    def test_seq_lengths_preserved(self):
        s1 = _make_sequence(1, 1)  # len 3
        s2 = _make_sequence(4, 2)  # len 7

        batch = HiveTokenBatch.collate([s1, s2])

        assert batch.seq_lengths[0] == 3
        assert batch.seq_lengths[1] == 7
