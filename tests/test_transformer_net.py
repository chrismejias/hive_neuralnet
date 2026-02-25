"""Tests for hive_transformer.transformer_net — Transformer network."""

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
from hive_transformer.transformer_net import (
    TransformerConfig,
    TransformerPolicyHead,
    TransformerValueHead,
    HiveTransformer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sequence(num_board=3, num_hand=2):
    total = 1 + num_board + num_hand
    features = np.random.randn(total, TOKEN_FEAT_DIM).astype(np.float32)
    positions = np.full(total, OFF_BOARD_POSITION, dtype=np.int32)
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


def _make_batch(sequences=None):
    if sequences is None:
        sequences = [_make_sequence(3, 2), _make_sequence(2, 1)]
    return HiveTokenBatch.collate(sequences)


SMALL_CFG = TransformerConfig(d_model=32, num_heads=4, num_layers=2, dim_feedforward=64)


# ---------------------------------------------------------------------------
# TestTransformerConfig
# ---------------------------------------------------------------------------


class TestTransformerConfig:
    def test_default(self):
        cfg = TransformerConfig()
        assert cfg.d_model == 128
        assert cfg.num_heads == 8
        assert cfg.num_layers == 6
        assert cfg.action_space_size == 29407

    def test_small(self):
        cfg = TransformerConfig.small()
        assert cfg.d_model == 128
        assert cfg.num_layers == 6

    def test_large(self):
        cfg = TransformerConfig.large()
        assert cfg.d_model == 256
        assert cfg.num_layers == 12


# ---------------------------------------------------------------------------
# TestTransformerPolicyHead
# ---------------------------------------------------------------------------


class TestTransformerPolicyHead:
    def test_output_shape(self):
        cfg = SMALL_CFG
        head = TransformerPolicyHead(cfg)
        grid = torch.randn(2, cfg.d_model, 13, 13)
        global_feat = torch.randn(2, GLOBAL_FEAT_DIM)
        out = head(grid, global_feat)
        assert out.shape == (2, cfg.action_space_size)

    def test_single_batch(self):
        cfg = SMALL_CFG
        head = TransformerPolicyHead(cfg)
        grid = torch.randn(1, cfg.d_model, 13, 13)
        global_feat = torch.randn(1, GLOBAL_FEAT_DIM)
        out = head(grid, global_feat)
        assert out.shape == (1, cfg.action_space_size)


# ---------------------------------------------------------------------------
# TestTransformerValueHead
# ---------------------------------------------------------------------------


class TestTransformerValueHead:
    def test_output_shape(self):
        head = TransformerValueHead(32, GLOBAL_FEAT_DIM)
        cls_emb = torch.randn(2, 32)
        global_feat = torch.randn(2, GLOBAL_FEAT_DIM)
        out = head(cls_emb, global_feat)
        assert out.shape == (2, 1)

    def test_output_range(self):
        head = TransformerValueHead(32, GLOBAL_FEAT_DIM)
        cls_emb = torch.randn(1, 32)
        global_feat = torch.randn(1, GLOBAL_FEAT_DIM)
        out = head(cls_emb, global_feat)
        assert -1.0 <= out.item() <= 1.0


# ---------------------------------------------------------------------------
# TestHiveTransformer
# ---------------------------------------------------------------------------


class TestHiveTransformer:
    def test_forward_output_shapes(self):
        net = HiveTransformer(SMALL_CFG)
        batch = _make_batch()
        policy, value = net(batch)
        B = batch.global_features.size(0)
        assert policy.shape == (B, SMALL_CFG.action_space_size)
        assert value.shape == (B, 1)

    def test_forward_single(self):
        net = HiveTransformer(SMALL_CFG)
        seq = _make_sequence(4, 2)
        batch = HiveTokenBatch.collate([seq])
        policy, value = net(batch)
        assert policy.shape == (1, SMALL_CFG.action_space_size)
        assert value.shape == (1, 1)

    def test_forward_different_lengths(self):
        net = HiveTransformer(SMALL_CFG)
        s1 = _make_sequence(1, 1)   # short
        s2 = _make_sequence(5, 3)   # long
        batch = HiveTokenBatch.collate([s1, s2])
        policy, value = net(batch)
        assert policy.shape == (2, SMALL_CFG.action_space_size)
        assert value.shape == (2, 1)

    def test_forward_only_hand(self):
        """Empty board — only CLS + hand tokens."""
        net = HiveTransformer(SMALL_CFG)
        seq = _make_sequence(0, 10)
        batch = HiveTokenBatch.collate([seq])
        policy, value = net(batch)
        assert policy.shape == (1, SMALL_CFG.action_space_size)
        assert value.shape == (1, 1)

    def test_value_in_range(self):
        net = HiveTransformer(SMALL_CFG)
        batch = _make_batch()
        _, value = net(batch)
        for v in value.flatten():
            assert -1.0 <= v.item() <= 1.0

    def test_gradients_flow(self):
        net = HiveTransformer(SMALL_CFG)
        net.train()
        batch = _make_batch()
        policy, value = net(batch)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_count_parameters(self):
        net = HiveTransformer(SMALL_CFG)
        count = net.count_parameters()
        assert count > 0

    def test_predict_interface(self):
        net = HiveTransformer(SMALL_CFG)
        seq = _make_sequence(3, 2)
        legal_mask = np.zeros(SMALL_CFG.action_space_size, dtype=np.float32)
        legal_mask[0] = 1.0
        legal_mask[100] = 1.0
        legal_mask[1000] = 1.0

        probs, val = net.predict(seq, legal_mask)

        assert probs.shape == (SMALL_CFG.action_space_size,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert probs[0] > 0.0
        assert probs[1] == 0.0  # illegal
        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_predict_no_legal(self):
        net = HiveTransformer(SMALL_CFG)
        seq = _make_sequence(2, 1)
        legal_mask = np.zeros(SMALL_CFG.action_space_size, dtype=np.float32)
        probs, val = net.predict(seq, legal_mask)
        assert probs.sum() == 0.0

    def test_default_config(self):
        net = HiveTransformer()
        assert net.config.d_model == 128

    def test_batch_size_three(self):
        net = HiveTransformer(SMALL_CFG)
        seqs = [_make_sequence(2, 1), _make_sequence(4, 3), _make_sequence(1, 1)]
        batch = HiveTokenBatch.collate(seqs)
        policy, value = net(batch)
        assert policy.shape == (3, SMALL_CFG.action_space_size)
        assert value.shape == (3, 1)
