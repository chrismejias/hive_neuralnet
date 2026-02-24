"""
Training metrics collection and logging for Hive AlphaZero.

Provides structured per-iteration metrics and a JSON-lines logger
for analyzing training progress.

Usage:
    logger = MetricsLogger("metrics.jsonl")
    logger.open()
    m = IterationMetrics(iteration=1)
    m.selfplay_num_games = 50
    ...
    logger.log_iteration(m)
    logger.close()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import TextIO


@dataclass
class IterationMetrics:
    """Metrics for a single training iteration."""

    iteration: int = 0
    timestamp: float = 0.0

    # Self-play metrics
    selfplay_time_sec: float = 0.0
    selfplay_num_games: int = 0
    selfplay_num_examples: int = 0
    selfplay_avg_game_length: float = 0.0
    selfplay_white_wins: int = 0
    selfplay_black_wins: int = 0
    selfplay_draws: int = 0

    # Training metrics
    train_time_sec: float = 0.0
    train_avg_loss: float = 0.0
    train_avg_policy_loss: float = 0.0
    train_avg_value_loss: float = 0.0
    train_num_batches: int = 0

    # Arena metrics
    arena_time_sec: float = 0.0
    arena_new_win_rate: float = 0.0
    arena_model_accepted: bool = False

    # ELO rating
    elo_rating: float = 0.0

    # Learning rate
    train_learning_rate: float = 0.0

    # Buffer state
    buffer_size: int = 0

    # Cumulative
    cumulative_games: int = 0
    cumulative_examples: int = 0
    total_elapsed_sec: float = 0.0


class MetricsLogger:
    """
    Logs training metrics to stdout and optionally to a JSON-lines file.

    Each iteration's metrics are appended as a single JSON line,
    enabling streaming reads and easy parsing. The file survives
    mid-write crashes since each line is flushed immediately.
    """

    def __init__(self, metrics_file: str | None = None) -> None:
        self.metrics_file = metrics_file
        self._file_handle: TextIO | None = None
        self._history: list[IterationMetrics] = []
        self._cumulative_games: int = 0
        self._cumulative_examples: int = 0
        self._start_time: float = time.time()

    def open(self) -> None:
        """Open the metrics file for appending."""
        if self.metrics_file:
            dir_name = os.path.dirname(self.metrics_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._file_handle = open(self.metrics_file, "a")

    def close(self) -> None:
        """Close the metrics file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def log_iteration(self, metrics: IterationMetrics) -> None:
        """Record and persist one iteration's metrics."""
        metrics.timestamp = time.time()
        metrics.total_elapsed_sec = time.time() - self._start_time

        self._cumulative_games += metrics.selfplay_num_games
        self._cumulative_examples += metrics.selfplay_num_examples
        metrics.cumulative_games = self._cumulative_games
        metrics.cumulative_examples = self._cumulative_examples

        self._history.append(metrics)

        # Write to JSON-lines file
        if self._file_handle:
            self._file_handle.write(json.dumps(asdict(metrics)) + "\n")
            self._file_handle.flush()

    @property
    def history(self) -> list[IterationMetrics]:
        """Access the in-memory metrics history."""
        return self._history
