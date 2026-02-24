"""
ELO rating tracking for Hive AlphaZero training.

Tracks model strength over training iterations by computing ELO ratings
from arena results. The opponent is always the previous best model.

Usage:
    tracker = EloTracker()
    rating = tracker.update(win_rate=0.65, games=20)
    print(f"Current ELO: {tracker.current_rating}")
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EloTracker:
    """
    Track ELO ratings across training iterations.

    After each arena evaluation, call update() with the new model's win rate
    against the current best. The rating adjusts based on the standard ELO
    formula.

    Attributes:
        ratings: List of ELO ratings, one per update (index = iteration-1).
        k_factor: Controls rating volatility (higher = larger swings).
        initial_rating: Starting ELO for iteration 0.
    """

    ratings: list[float] = field(default_factory=list)
    k_factor: float = 32.0
    initial_rating: float = 0.0

    def update(self, win_rate: float, games: int) -> float:
        """
        Update ELO rating based on arena results.

        The opponent is always the current best model, whose rating
        is the last entry in self.ratings (or initial_rating if empty).

        Args:
            win_rate: New model's win rate in [0, 1].
            games: Number of arena games played (for future extensions).

        Returns:
            The new ELO rating.
        """
        current = self.ratings[-1] if self.ratings else self.initial_rating
        opponent = current  # Arena opponent is current best

        expected = 1.0 / (1.0 + 10.0 ** ((opponent - current) / 400.0))
        actual = win_rate
        new_rating = current + self.k_factor * (actual - expected)
        self.ratings.append(new_rating)
        return new_rating

    @property
    def current_rating(self) -> float:
        """The most recent ELO rating."""
        return self.ratings[-1] if self.ratings else self.initial_rating
