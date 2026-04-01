"""Tests for TensorBoard logging."""

import os
import pytest

from hive_engine.metrics import IterationMetrics
from hive_engine.tb_logger import TBLogger


class TestTBLogger:
    """Tests for the TBLogger class."""

    def test_disabled_when_no_dir(self):
        """Logger is disabled when log_dir is None."""
        tb = TBLogger(log_dir=None)
        assert not tb.enabled
        # Should not raise
        tb.log_iteration(IterationMetrics(iteration=1))
        tb.close()

    def test_enabled_with_dir(self, tmp_path):
        """Logger is enabled when log_dir is provided and TB is installed."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_available = True
        except ImportError:
            tb_available = False

        tb = TBLogger(log_dir=str(tmp_path / "tb_logs"))
        if tb_available:
            assert tb.enabled
        else:
            assert not tb.enabled
        tb.close()

    def test_log_iteration_creates_events(self, tmp_path):
        """Logging creates TensorBoard event files."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            pytest.skip("TensorBoard not installed")

        log_dir = str(tmp_path / "tb_logs")
        tb = TBLogger(log_dir=log_dir)

        metrics = IterationMetrics(
            iteration=1,
            train_avg_loss=0.5,
            train_avg_policy_loss=0.3,
            train_avg_value_loss=0.2,
            train_learning_rate=1e-3,
            selfplay_avg_game_length=25.0,
            selfplay_num_examples=100,
            selfplay_white_wins=5,
            selfplay_black_wins=3,
            selfplay_draws=2,
            arena_new_win_rate=0.6,
            arena_model_accepted=True,
            elo_rating=16.0,
            buffer_size=500,
            cumulative_games=10,
            cumulative_examples=100,
        )
        tb.log_iteration(metrics)
        tb.close()

        # Verify event files were created
        assert os.path.exists(log_dir)
        files = os.listdir(log_dir)
        event_files = [f for f in files if "events.out.tfevents" in f]
        assert len(event_files) > 0

    def test_multiple_iterations(self, tmp_path):
        """Can log multiple iterations without error."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            pytest.skip("TensorBoard not installed")

        tb = TBLogger(log_dir=str(tmp_path / "tb_logs"))

        for i in range(1, 4):
            metrics = IterationMetrics(
                iteration=i,
                train_avg_loss=1.0 / i,
            )
            tb.log_iteration(metrics)

        tb.close()

    def test_close_idempotent(self, tmp_path):
        """Calling close() multiple times is safe."""
        tb = TBLogger(log_dir=str(tmp_path / "tb_logs"))
        tb.close()
        tb.close()  # Should not raise

    def test_log_after_close_is_noop(self, tmp_path):
        """Logging after close is a no-op."""
        tb = TBLogger(log_dir=str(tmp_path / "tb_logs"))
        tb.close()
        # Should not raise
        tb.log_iteration(IterationMetrics(iteration=1))
