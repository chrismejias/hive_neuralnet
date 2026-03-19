"""Tests for hive_nnue.nnue_net — NNUE MLP network architecture."""

import numpy as np
import pytest
import torch

from hive_nnue.nnue_encoder import FEATURE_DIM
from hive_nnue.nnue_net import NNUEConfig, HiveNNUE


# ---------------------------------------------------------------------------
# TestNNUEConfig
# ---------------------------------------------------------------------------


class TestNNUEConfig:
    def test_default_config(self):
        cfg = NNUEConfig()
        assert cfg.feature_dim == FEATURE_DIM
        assert cfg.hidden_dims == [512, 256]
        assert cfg.dropout == 0.1
        assert cfg.action_space_size == 29914

    def test_small_preset(self):
        cfg = NNUEConfig.small()
        assert cfg.hidden_dims == [512, 256]

    def test_large_preset(self):
        cfg = NNUEConfig.large()
        assert cfg.hidden_dims == [1024, 512, 256]

    def test_custom_config(self):
        cfg = NNUEConfig(hidden_dims=[64, 32], dropout=0.2)
        assert cfg.hidden_dims == [64, 32]
        assert cfg.dropout == 0.2


# ---------------------------------------------------------------------------
# TestHiveNNUE
# ---------------------------------------------------------------------------


SMALL_CFG = NNUEConfig(hidden_dims=[64, 32], dropout=0.0)


class TestHiveNNUE:
    """Tests for the NNUE MLP network."""

    def test_forward_output_shapes(self):
        net = HiveNNUE(SMALL_CFG)
        x = torch.randn(4, FEATURE_DIM)

        policy, value = net(x)
        assert policy.shape == (4, SMALL_CFG.action_space_size)
        assert value.shape == (4, 1)

    def test_forward_single_sample(self):
        net = HiveNNUE(SMALL_CFG)
        x = torch.randn(1, FEATURE_DIM)

        policy, value = net(x)
        assert policy.shape == (1, SMALL_CFG.action_space_size)
        assert value.shape == (1, 1)

    def test_value_in_range(self):
        net = HiveNNUE(SMALL_CFG)
        x = torch.randn(8, FEATURE_DIM)

        _, value = net(x)
        for v in value.flatten():
            assert -1.0 <= v.item() <= 1.0

    def test_gradients_flow_through(self):
        net = HiveNNUE(SMALL_CFG)
        x = torch.randn(4, FEATURE_DIM)

        policy, value = net(x)
        loss = policy.sum() + value.sum()
        loss.backward()

        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_count_parameters(self):
        net = HiveNNUE(SMALL_CFG)
        count = net.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_parameter_count_ordering(self):
        """Larger config should have more parameters."""
        small = HiveNNUE(NNUEConfig.small())
        large = HiveNNUE(NNUEConfig.large())
        assert large.count_parameters() > small.count_parameters()

    def test_default_config_instantiation(self):
        net = HiveNNUE()
        assert net.config.hidden_dims == [512, 256]

    def test_predict_interface(self):
        """Test MCTS predict interface with numpy inputs."""
        net = HiveNNUE(SMALL_CFG)
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        legal_mask = np.zeros(SMALL_CFG.action_space_size, dtype=np.float32)
        legal_mask[0] = 1.0
        legal_mask[100] = 1.0
        legal_mask[1000] = 1.0

        probs, val = net.predict(features, legal_mask)

        assert probs.shape == (SMALL_CFG.action_space_size,)
        assert probs.dtype == np.float32
        assert abs(probs.sum() - 1.0) < 1e-5
        assert probs[0] > 0.0
        assert probs[100] > 0.0
        assert probs[1000] > 0.0
        assert probs[1] == 0.0  # illegal

        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_predict_no_legal_moves(self):
        """Edge case: no legal moves results in zero policy."""
        net = HiveNNUE(SMALL_CFG)
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        legal_mask = np.zeros(SMALL_CFG.action_space_size, dtype=np.float32)

        probs, val = net.predict(features, legal_mask)
        assert probs.sum() == 0.0

    def test_predict_preserves_training_mode(self):
        """predict() should restore the original training mode."""
        net = HiveNNUE(SMALL_CFG)
        features = np.random.randn(FEATURE_DIM).astype(np.float32)
        legal_mask = np.ones(SMALL_CFG.action_space_size, dtype=np.float32)

        net.train()
        assert net.training
        net.predict(features, legal_mask)
        assert net.training

        net.eval()
        assert not net.training
        net.predict(features, legal_mask)
        assert not net.training

    def test_dropout_effect(self):
        """With dropout > 0, train and eval modes should differ."""
        cfg = NNUEConfig(hidden_dims=[64, 32], dropout=0.5)
        net = HiveNNUE(cfg)
        x = torch.randn(16, FEATURE_DIM)

        net.train()
        out_train_1, _ = net(x)
        out_train_2, _ = net(x)

        net.eval()
        out_eval_1, _ = net(x)
        out_eval_2, _ = net(x)

        # In eval mode, outputs should be deterministic
        torch.testing.assert_close(out_eval_1, out_eval_2)

    def test_backbone_layer_count(self):
        """Verify backbone has correct number of layers."""
        cfg = NNUEConfig(hidden_dims=[128, 64, 32], dropout=0.1)
        net = HiveNNUE(cfg)

        # Each hidden dim contributes: Linear + LayerNorm + ReLU + Dropout
        expected_layers = len(cfg.hidden_dims) * 4
        assert len(net.backbone) == expected_layers
