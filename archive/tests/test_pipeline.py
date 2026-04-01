"""
Pipeline smoke tests for the full Hive AlphaZero training pipeline.

These tests verify that all components integrate correctly:
- Encoder → Neural Net → MCTS → Trainer → Checkpointing
- Metrics logging
- Training resumption from checkpoint
- CLI argument parsing

Tests use tiny networks and minimal iterations to keep execution fast.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, GameResult
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.metrics import IterationMetrics, MetricsLogger
from hive_engine.neural_net import HiveNet, NetConfig, compute_loss
from hive_engine.trainer import (
    Trainer,
    TrainConfig,
    TrainingExample,
    ReplayBuffer,
    SelfPlayStats,
    TrainStats,
)
from hive_engine.train import build_parser, apply_overrides, main as train_main
from hive_engine.pieces import Color


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def tiny_config():
    return NetConfig(num_blocks=1, num_filters=16)


@pytest.fixture
def fast_train_config(tmp_path):
    return TrainConfig(
        num_iterations=2,
        games_per_iteration=1,
        mcts_simulations=3,
        batch_size=4,
        num_epochs=1,
        buffer_max_size=100,
        arena_games=2,
        arena_mcts_simulations=3,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        metrics_file=str(tmp_path / "metrics.jsonl"),
        max_game_length=20,
    )


# ── Encoder ↔ Net Integration ────────────────────────────────────


class TestEncoderNetIntegration:
    def test_encode_predict_roundtrip(self, tiny_config):
        """Encoder output can be fed into the neural net."""
        encoder = HiveEncoder()
        net = HiveNet(tiny_config)
        net.eval()

        game = GameState()
        state_tensor = encoder.encode_state(game)
        legal_moves = game.legal_moves()
        mask = encoder.get_legal_action_mask(game, legal_moves)

        action_probs, value = net.predict(state_tensor, mask)

        assert action_probs.shape == (encoder.ACTION_SPACE_SIZE,)
        assert -1.0 <= value <= 1.0
        # Probability only on legal actions
        illegal_mask = (mask == 0)
        assert np.all(action_probs[illegal_mask] < 1e-6)

    def test_centroid_cache_consistency(self):
        """Cached centroid should match non-cached computation."""
        encoder = HiveEncoder()
        game = GameState()

        # Make a few moves to create a non-trivial board
        moves = game.legal_moves()
        if moves:
            game.apply_move(moves[0])
        moves = game.legal_moves()
        if moves:
            game.apply_move(moves[0])

        cached = encoder._cached_center(game.board)
        direct = encoder._compute_center(game.board)
        assert cached == direct


# ── MCTS ↔ Net Integration ───────────────────────────────────────


class TestMCTSIntegration:
    def test_mcts_search_returns_valid_policy(self, tiny_config):
        """MCTS search should return a valid probability distribution."""
        net = HiveNet(tiny_config)
        encoder = HiveEncoder()
        config = MCTSConfig(num_simulations=5)
        mcts = MCTS(net, encoder, config)

        game = GameState()
        policy = mcts.search(game, move_number=0)

        assert policy.shape == (encoder.ACTION_SPACE_SIZE,)
        assert abs(policy.sum() - 1.0) < 1e-4
        assert np.all(policy >= 0)


# ── MetricsLogger Tests ──────────────────────────────────────────


class TestMetricsLogger:
    def test_log_and_read_back(self, tmp_path):
        """Metrics written to file can be read back as JSON."""
        path = str(tmp_path / "test_metrics.jsonl")
        logger = MetricsLogger(path)
        logger.open()

        m = IterationMetrics(iteration=1)
        m.selfplay_num_games = 10
        m.train_avg_loss = 5.42
        m.arena_new_win_rate = 0.6
        m.arena_model_accepted = True
        logger.log_iteration(m)

        m2 = IterationMetrics(iteration=2)
        m2.selfplay_num_games = 10
        m2.train_avg_loss = 4.31
        logger.log_iteration(m2)

        logger.close()

        # Read back
        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["iteration"] == 1
        assert data["selfplay_num_games"] == 10
        assert data["train_avg_loss"] == 5.42
        assert data["arena_model_accepted"] is True

    def test_history_accumulates(self):
        """In-memory history should accumulate metrics."""
        logger = MetricsLogger(None)  # No file
        logger.open()

        for i in range(3):
            m = IterationMetrics(iteration=i + 1)
            m.selfplay_num_games = 5
            logger.log_iteration(m)

        assert len(logger.history) == 3
        assert logger._cumulative_games == 15

        logger.close()


# ── Self-Play Stats ──────────────────────────────────────────────


class TestSelfPlayStats:
    def test_avg_game_length(self):
        stats = SelfPlayStats(
            num_games=4, total_game_length=120,
        )
        assert stats.avg_game_length == 30.0

    def test_avg_game_length_zero(self):
        stats = SelfPlayStats()
        assert stats.avg_game_length == 0.0


# ── Full Pipeline Smoke Test ─────────────────────────────────────


class TestFullPipeline:
    def test_trainer_run_two_iterations(self, fast_train_config, tiny_config):
        """Full training loop should complete 2 iterations without error."""
        trainer = Trainer(fast_train_config, tiny_config)
        trainer.run()

        # Check that checkpoints were created
        ckpt_dir = fast_train_config.checkpoint_dir
        assert os.path.isdir(ckpt_dir)
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        assert len(checkpoints) >= 2

        # Check that metrics file was created
        assert os.path.exists(fast_train_config.metrics_file)
        with open(fast_train_config.metrics_file) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_save_and_resume_checkpoint(self, fast_train_config, tiny_config, tmp_path):
        """Training should be resumable from a checkpoint."""
        # Run 1 iteration
        config1 = TrainConfig(
            num_iterations=1,
            games_per_iteration=1,
            mcts_simulations=3,
            batch_size=4,
            num_epochs=1,
            buffer_max_size=100,
            arena_games=2,
            arena_mcts_simulations=3,
            checkpoint_dir=str(tmp_path / "ckpt"),
            max_game_length=20,
        )
        trainer1 = Trainer(config1, tiny_config)
        trainer1.run()

        # Find the checkpoint
        ckpt_dir = config1.checkpoint_dir
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        assert len(ckpts) >= 1
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])

        # Resume with more iterations
        config2 = TrainConfig(
            num_iterations=2,
            games_per_iteration=1,
            mcts_simulations=3,
            batch_size=4,
            num_epochs=1,
            buffer_max_size=100,
            arena_games=2,
            arena_mcts_simulations=3,
            checkpoint_dir=str(tmp_path / "ckpt2"),
            max_game_length=20,
        )
        trainer2 = Trainer.from_checkpoint(ckpt_path, config_overrides=config2)
        assert trainer2._start_iteration == 2  # Should resume from iteration 2
        assert len(trainer2.buffer) > 0  # Buffer should be restored
        trainer2.run()

    def test_load_checkpoint_for_inference(self, fast_train_config, tiny_config):
        """load_checkpoint should return a working model for inference."""
        trainer = Trainer(fast_train_config, tiny_config)
        trainer.run()

        ckpt_dir = fast_train_config.checkpoint_dir
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])

        net = Trainer.load_checkpoint(ckpt_path)
        net.eval()

        encoder = HiveEncoder()
        game = GameState()
        state = encoder.encode_state(game)
        mask = encoder.get_legal_action_mask(game)
        probs, value = net.predict(state, mask)

        assert probs.shape == (encoder.ACTION_SPACE_SIZE,)
        assert -1.0 <= value <= 1.0


# ── CLI Tests ─────────────────────────────────────────────────────


class TestCLI:
    def test_parser_defaults(self):
        """CLI parser should accept no arguments."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.resume is None
        assert args.net == "small"
        assert args.iterations is None

    def test_parser_all_flags(self):
        """CLI parser should accept all documented flags."""
        parser = build_parser()
        args = parser.parse_args([
            "--net", "large",
            "--blocks", "3",
            "--filters", "32",
            "--iterations", "10",
            "--games", "20",
            "--simulations", "50",
            "--temperature", "0.8",
            "--temp-drop", "15",
            "--batch-size", "32",
            "--epochs", "3",
            "--lr", "5e-4",
            "--weight-decay", "1e-5",
            "--buffer-size", "10000",
            "--arena-games", "10",
            "--arena-threshold", "0.6",
            "--arena-sims", "25",
            "--workers", "2",
            "--checkpoint-dir", "/tmp/ckpt",
            "--max-game-length", "200",
        ])
        assert args.net == "large"
        assert args.blocks == 3
        assert args.iterations == 10
        assert args.workers == 2

    def test_apply_overrides(self):
        """apply_overrides should update the config correctly."""
        parser = build_parser()
        args = parser.parse_args(["--iterations", "50", "--lr", "1e-4"])
        config = TrainConfig()
        config = apply_overrides(config, args)
        assert config.num_iterations == 50
        assert config.learning_rate == 1e-4
        # Unspecified values should remain at defaults
        assert config.games_per_iteration == 50

    def test_cli_main_runs(self, tmp_path):
        """CLI main function should complete a tiny training run."""
        train_main([
            "--iterations", "1",
            "--games", "1",
            "--simulations", "3",
            "--batch-size", "4",
            "--epochs", "1",
            "--arena-games", "2",
            "--arena-sims", "3",
            "--blocks", "1",
            "--filters", "16",
            "--max-game-length", "15",
            "--checkpoint-dir", str(tmp_path / "cli_ckpt"),
        ])

        ckpts = [
            f for f in os.listdir(tmp_path / "cli_ckpt")
            if f.endswith(".pt")
        ]
        assert len(ckpts) >= 1
