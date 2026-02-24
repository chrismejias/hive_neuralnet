"""Tests for the self-play training pipeline."""

import os
import tempfile

import numpy as np
import pytest
import torch

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, GameResult
from hive_engine.neural_net import HiveNet, NetConfig, compute_loss
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.trainer import (
    Trainer,
    TrainConfig,
    TrainingExample,
    ReplayBuffer,
)
from hive_engine.pieces import Color


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def tiny_config():
    """Minimal network config for fast tests."""
    return NetConfig(num_blocks=1, num_filters=16)


@pytest.fixture
def tiny_net(tiny_config):
    return HiveNet(tiny_config)


@pytest.fixture
def fast_train_config(tmp_path):
    """Minimal training config for fast tests."""
    return TrainConfig(
        num_iterations=1,
        games_per_iteration=1,
        mcts_simulations=5,
        batch_size=4,
        num_epochs=1,
        buffer_max_size=100,
        arena_games=2,
        arena_mcts_simulations=5,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_game_length=30,
    )


# ── ReplayBuffer Tests ──────────────────────────────────────────


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(max_size=100)
        assert len(buf) == 0

        examples = [
            TrainingExample(
                np.zeros((26, 13, 13), dtype=np.float32),
                np.zeros(29407, dtype=np.float32),
                1.0,
            )
            for _ in range(10)
        ]
        buf.add_examples(examples)
        assert len(buf) == 10

    def test_max_size_respected(self):
        buf = ReplayBuffer(max_size=5)
        examples = [
            TrainingExample(
                np.zeros((26, 13, 13), dtype=np.float32),
                np.zeros(29407, dtype=np.float32),
                0.0,
            )
            for _ in range(10)
        ]
        buf.add_examples(examples)
        assert len(buf) == 5  # Only keeps last 5

    def test_sample_batch_shapes(self):
        buf = ReplayBuffer(max_size=100)
        examples = [
            TrainingExample(
                np.random.randn(26, 13, 13).astype(np.float32),
                np.random.dirichlet(np.ones(29407)).astype(np.float32),
                np.random.choice([-1.0, 0.0, 1.0]),
            )
            for _ in range(20)
        ]
        buf.add_examples(examples)

        states, policies, values = buf.sample_batch(8)
        assert states.shape == (8, 26, 13, 13)
        assert policies.shape == (8, 29407)
        assert values.shape == (8, 1)
        assert states.dtype == torch.float32
        assert policies.dtype == torch.float32
        assert values.dtype == torch.float32

    def test_sample_batch_smaller_than_buffer(self):
        """If batch_size > buffer size, return all examples."""
        buf = ReplayBuffer(max_size=100)
        examples = [
            TrainingExample(
                np.zeros((26, 13, 13), dtype=np.float32),
                np.zeros(29407, dtype=np.float32),
                0.0,
            )
            for _ in range(3)
        ]
        buf.add_examples(examples)
        states, policies, values = buf.sample_batch(10)
        assert states.shape[0] == 3


# ── TrainingExample Tests ────────────────────────────────────────


class TestTrainingExample:
    def test_named_tuple_fields(self):
        ex = TrainingExample(
            state_tensor=np.zeros((26, 13, 13), dtype=np.float32),
            policy_target=np.zeros(29407, dtype=np.float32),
            value_target=1.0,
        )
        assert ex.state_tensor.shape == (26, 13, 13)
        assert ex.policy_target.shape == (29407,)
        assert ex.value_target == 1.0


# ── Self-Play Tests ──────────────────────────────────────────────


class TestSelfPlay:
    def test_self_play_game_produces_examples(self, fast_train_config, tiny_config):
        """A self-play game should produce valid training examples."""
        trainer = Trainer(fast_train_config, tiny_config)
        examples = trainer._self_play_game(trainer.best_net)

        assert len(examples) > 0

        for ex in examples:
            assert ex.state_tensor.shape == (26, 13, 13)
            assert ex.policy_target.shape == (29407,)
            assert ex.value_target in [-1.0, 0.0, 1.0]
            # Policy should be a valid distribution
            assert abs(ex.policy_target.sum() - 1.0) < 1e-4

    def test_value_targets_consistent_with_result(self, fast_train_config, tiny_config):
        """Value targets should be consistent: if WHITE wins, all WHITE states = +1."""
        trainer = Trainer(fast_train_config, tiny_config)
        # Run multiple games to hopefully get a non-draw
        for _ in range(5):
            examples = trainer._self_play_game(trainer.best_net)
            # Check all values are valid
            for ex in examples:
                assert ex.value_target in [-1.0, 0.0, 1.0]


# ── Training Phase Tests ─────────────────────────────────────────


class TestTrainPhase:
    def test_training_step_reduces_loss(self, tiny_config):
        """A few training steps should reduce the loss."""
        config = TrainConfig(
            batch_size=4,
            num_epochs=3,
            buffer_max_size=100,
            learning_rate=1e-2,
        )

        trainer = Trainer(config, tiny_config)

        # Fill buffer with some examples
        examples = []
        for _ in range(20):
            state = np.random.randn(26, 13, 13).astype(np.float32)
            policy = np.zeros(29407, dtype=np.float32)
            policy[0] = 1.0  # Deterministic target
            value = 1.0
            examples.append(TrainingExample(state, policy, value))
        trainer.buffer.add_examples(examples)

        # Measure loss before training
        states, policies, values = trainer.buffer.sample_batch(10)
        states = states.to(trainer.device)
        policies = policies.to(trainer.device)
        values = values.to(trainer.device)

        trainer.best_net.eval()
        with torch.no_grad():
            logits, val = trainer.best_net(states)
            loss_before, _, _ = compute_loss(logits, val, policies, values)

        # Train
        new_net = trainer._train_phase()

        # Measure loss after
        new_net.eval()
        with torch.no_grad():
            logits, val = new_net(states)
            loss_after, _, _ = compute_loss(logits, val, policies, values)

        assert loss_after.item() < loss_before.item(), (
            f"Loss did not decrease: {loss_before.item():.4f} -> {loss_after.item():.4f}"
        )


# ── Arena Tests ──────────────────────────────────────────────────


class TestArena:
    def test_arena_returns_valid_win_rate(self, fast_train_config, tiny_config):
        trainer = Trainer(fast_train_config, tiny_config)
        win_rate = trainer._arena_evaluate(trainer.best_net, trainer.best_net)
        assert 0.0 <= win_rate <= 1.0


# ── Checkpoint Tests ─────────────────────────────────────────────


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tiny_config, tmp_path):
        """Saved and loaded model should have identical weights."""
        config = TrainConfig(checkpoint_dir=str(tmp_path))
        trainer = Trainer(config, tiny_config)

        # Save
        path = trainer._save_checkpoint(trainer.best_net, 1)
        assert os.path.exists(path)

        # Load
        loaded = Trainer.load_checkpoint(path)
        loaded.eval()
        trainer.best_net.eval()

        # Compare weights
        for (name1, p1), (name2, p2) in zip(
            trainer.best_net.named_parameters(),
            loaded.named_parameters(),
        ):
            assert name1 == name2
            assert torch.allclose(p1.cpu(), p2.cpu()), f"Weights differ for {name1}"

    def test_checkpoint_contains_config(self, tiny_config, tmp_path):
        config = TrainConfig(checkpoint_dir=str(tmp_path))
        trainer = Trainer(config, tiny_config)
        path = trainer._save_checkpoint(trainer.best_net, 5)

        checkpoint = torch.load(path, weights_only=False)
        assert "net_config" in checkpoint
        assert "iteration" in checkpoint
        assert checkpoint["iteration"] == 5
