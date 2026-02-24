"""Tests for the HiveNet neural network."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hive_engine.encoder import HiveEncoder
from hive_engine.neural_net import HiveNet, NetConfig, ResidualBlock, compute_loss
from hive_engine.game_state import GameState


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def small_config():
    return NetConfig.small()


@pytest.fixture
def small_net(small_config):
    return HiveNet(small_config)


@pytest.fixture
def encoder():
    return HiveEncoder()


@pytest.fixture
def game_state():
    return GameState()


# ── NetConfig Tests ──────────────────────────────────────────────


class TestNetConfig:
    def test_default_config(self):
        cfg = NetConfig()
        assert cfg.num_blocks == 5
        assert cfg.num_filters == 128
        assert cfg.input_channels == 26
        assert cfg.action_space_size == 29407
        assert cfg.board_size == 13

    def test_small_preset(self):
        cfg = NetConfig.small()
        assert cfg.num_blocks == 5
        assert cfg.num_filters == 128

    def test_large_preset(self):
        cfg = NetConfig.large()
        assert cfg.num_blocks == 10
        assert cfg.num_filters == 256


# ── ResidualBlock Tests ──────────────────────────────────────────


class TestResidualBlock:
    def test_output_shape_matches_input(self):
        block = ResidualBlock(64)
        x = torch.randn(2, 64, 13, 13)
        out = block(x)
        assert out.shape == (2, 64, 13, 13)

    def test_skip_connection_preserves_gradient(self):
        block = ResidualBlock(32)
        x = torch.randn(1, 32, 13, 13, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ── HiveNet Tests ────────────────────────────────────────────────


class TestHiveNet:
    def test_forward_output_shapes(self, small_net):
        x = torch.randn(4, 26, 13, 13)
        policy, value = small_net(x)
        assert policy.shape == (4, 29407)
        assert value.shape == (4, 1)

    def test_forward_single_sample(self, small_net):
        x = torch.randn(1, 26, 13, 13)
        policy, value = small_net(x)
        assert policy.shape == (1, 29407)
        assert value.shape == (1, 1)

    def test_value_in_range(self, small_net):
        """Value head uses tanh, so output must be in [-1, 1]."""
        x = torch.randn(8, 26, 13, 13)
        _, value = small_net(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_gradient_flows(self, small_net):
        """Ensure loss.backward() works without errors."""
        x = torch.randn(2, 26, 13, 13)
        policy, value = small_net(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        # Check at least one parameter has gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_net.parameters()
        )
        assert has_grad

    def test_default_config_used(self):
        net = HiveNet()
        assert net.config.num_blocks == 5
        assert net.config.num_filters == 128

    def test_parameter_count_small(self):
        net = HiveNet(NetConfig.small())
        params = net.count_parameters()
        # Expect roughly 8-12M for small config (policy head dominates
        # due to 256→29407 linear layer = ~7.5M params)
        assert 5_000_000 < params < 15_000_000, f"Unexpected param count: {params}"

    def test_parameter_count_large_gt_small(self):
        small = HiveNet(NetConfig.small())
        large = HiveNet(NetConfig.large())
        assert large.count_parameters() > small.count_parameters()

    def test_eval_mode_deterministic(self, small_net):
        """In eval mode, same input should produce same output (no dropout etc)."""
        small_net.eval()
        x = torch.randn(1, 26, 13, 13)
        p1, v1 = small_net(x)
        p2, v2 = small_net(x)
        assert torch.allclose(p1, p2)
        assert torch.allclose(v1, v2)


# ── Predict Tests ────────────────────────────────────────────────


class TestPredict:
    def test_predict_output_shapes(self, small_net, encoder, game_state):
        state = encoder.encode_state(game_state)
        mask = encoder.get_legal_action_mask(game_state)
        probs, value = small_net.predict(state, mask)
        assert probs.shape == (29407,)
        assert isinstance(value, float)

    def test_predict_probs_sum_to_one(self, small_net, encoder, game_state):
        state = encoder.encode_state(game_state)
        mask = encoder.get_legal_action_mask(game_state)
        probs, _ = small_net.predict(state, mask)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_predict_only_legal_actions(self, small_net, encoder, game_state):
        """Probabilities should be zero for illegal actions."""
        state = encoder.encode_state(game_state)
        mask = encoder.get_legal_action_mask(game_state)
        probs, _ = small_net.predict(state, mask)
        illegal_mask = 1.0 - mask
        illegal_prob = (probs * illegal_mask).sum()
        assert illegal_prob < 1e-6

    def test_predict_value_in_range(self, small_net, encoder, game_state):
        state = encoder.encode_state(game_state)
        mask = encoder.get_legal_action_mask(game_state)
        _, value = small_net.predict(state, mask)
        assert -1.0 <= value <= 1.0

    def test_predict_all_zero_mask(self, small_net, encoder, game_state):
        """Edge case: all-zero mask should return all-zero probabilities."""
        state = encoder.encode_state(game_state)
        mask = np.zeros(29407, dtype=np.float32)
        probs, _ = small_net.predict(state, mask)
        assert probs.sum() == 0.0

    def test_predict_restores_training_mode(self, small_net, encoder, game_state):
        """predict() should restore the original training mode."""
        small_net.train()
        state = encoder.encode_state(game_state)
        mask = encoder.get_legal_action_mask(game_state)
        small_net.predict(state, mask)
        assert small_net.training is True

    def test_predict_after_moves(self, small_net, encoder):
        """Test predict on a state with pieces on the board."""
        gs = GameState()
        import random
        random.seed(42)
        # Play a few random moves
        for _ in range(6):
            moves = gs.legal_moves()
            if not moves:
                break
            gs.apply_move(random.choice(moves))

        state = encoder.encode_state(gs)
        mask = encoder.get_legal_action_mask(gs)
        probs, value = small_net.predict(state, mask)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert -1.0 <= value <= 1.0


# ── Loss Function Tests ─────────────────────────────────────────


class TestComputeLoss:
    def test_loss_is_positive_scalar(self, small_net):
        x = torch.randn(4, 26, 13, 13)
        policy_logits, value_pred = small_net(x)

        # Create dummy targets
        target_policy = torch.zeros(4, 29407)
        target_policy[:, 0] = 1.0  # One-hot
        target_value = torch.zeros(4, 1)

        total, pol, val = compute_loss(
            policy_logits, value_pred, target_policy, target_value
        )
        assert total.dim() == 0  # Scalar
        assert total.item() > 0
        assert pol.item() >= 0
        assert val.item() >= 0

    def test_loss_backward_no_error(self, small_net):
        x = torch.randn(2, 26, 13, 13)
        policy_logits, value_pred = small_net(x)

        target_policy = torch.zeros(2, 29407)
        target_policy[:, 5] = 1.0
        target_value = torch.tensor([[1.0], [-1.0]])

        total, _, _ = compute_loss(
            policy_logits, value_pred, target_policy, target_value
        )
        total.backward()  # Should not raise

    def test_perfect_prediction_low_loss(self):
        """If predictions match targets, loss should be low."""
        # Create deterministic small net
        torch.manual_seed(0)
        net = HiveNet(NetConfig(num_blocks=1, num_filters=16))
        x = torch.randn(1, 26, 13, 13)
        policy_logits, value_pred = net(x)

        # Use the network's own output as target
        target_policy = F.softmax(policy_logits.detach(), dim=1)
        target_value = value_pred.detach()

        total, pol, val = compute_loss(
            policy_logits, value_pred, target_policy, target_value
        )
        # Value loss should be ~0 since target == prediction
        assert val.item() < 1e-6
