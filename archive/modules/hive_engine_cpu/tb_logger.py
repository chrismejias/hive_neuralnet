"""
Optional TensorBoard logging for Hive AlphaZero training.

Wraps torch.utils.tensorboard.SummaryWriter with graceful fallback
when TensorBoard is not installed. Logs all training metrics per
iteration for visual monitoring.

Usage:
    tb = TBLogger("runs/experiment1")
    tb.log_iteration(metrics)
    tb.close()
"""

from __future__ import annotations

from hive_engine.metrics import IterationMetrics


class TBLogger:
    """
    Optional TensorBoard logger that no-ops if tensorboard is not installed.

    Creates a SummaryWriter in the given log_dir and writes scalar metrics
    for each training iteration. If log_dir is None or tensorboard is not
    available, all methods silently do nothing.
    """

    def __init__(self, log_dir: str | None = None) -> None:
        self.writer = None
        self.log_dir = log_dir
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                pass  # TensorBoard not installed — silent no-op

    @property
    def enabled(self) -> bool:
        """Whether TensorBoard logging is active."""
        return self.writer is not None

    def log_iteration(self, metrics: IterationMetrics) -> None:
        """Log all metrics from one training iteration."""
        if not self.writer:
            return

        step = metrics.iteration

        # Training metrics
        self.writer.add_scalar("train/total_loss", metrics.train_avg_loss, step)
        self.writer.add_scalar("train/policy_loss", metrics.train_avg_policy_loss, step)
        self.writer.add_scalar("train/value_loss", metrics.train_avg_value_loss, step)
        self.writer.add_scalar("train/learning_rate", metrics.train_learning_rate, step)

        # Self-play metrics
        self.writer.add_scalar("selfplay/avg_game_length", metrics.selfplay_avg_game_length, step)
        self.writer.add_scalar("selfplay/num_examples", metrics.selfplay_num_examples, step)
        self.writer.add_scalar("selfplay/white_wins", metrics.selfplay_white_wins, step)
        self.writer.add_scalar("selfplay/black_wins", metrics.selfplay_black_wins, step)
        self.writer.add_scalar("selfplay/draws", metrics.selfplay_draws, step)

        # Arena metrics
        self.writer.add_scalar("arena/win_rate", metrics.arena_new_win_rate, step)
        self.writer.add_scalar("arena/model_accepted", int(metrics.arena_model_accepted), step)

        # ELO
        self.writer.add_scalar("elo/rating", metrics.elo_rating, step)

        # Buffer
        self.writer.add_scalar("buffer/size", metrics.buffer_size, step)

        # Cumulative
        self.writer.add_scalar("cumulative/games", metrics.cumulative_games, step)
        self.writer.add_scalar("cumulative/examples", metrics.cumulative_examples, step)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
            self.writer = None
