"""Tests for ELO rating tracking."""

import pytest

from hive_engine.elo import EloTracker


class TestEloTracker:
    """Tests for the EloTracker class."""

    def test_initial_rating(self):
        tracker = EloTracker()
        assert tracker.current_rating == 0.0

    def test_custom_initial_rating(self):
        tracker = EloTracker(initial_rating=1500.0)
        assert tracker.current_rating == 1500.0

    def test_perfect_win_increases_rating(self):
        tracker = EloTracker()
        new_rating = tracker.update(win_rate=1.0, games=20)
        assert new_rating > 0.0
        assert tracker.current_rating == new_rating

    def test_perfect_loss_decreases_rating(self):
        tracker = EloTracker()
        new_rating = tracker.update(win_rate=0.0, games=20)
        assert new_rating < 0.0

    def test_draw_stays_similar(self):
        """50% win rate against equal opponent → minimal change."""
        tracker = EloTracker()
        new_rating = tracker.update(win_rate=0.5, games=20)
        assert abs(new_rating) < 1.0  # Should be ~0

    def test_k_factor_controls_magnitude(self):
        """Higher K-factor produces larger rating changes."""
        tracker_low = EloTracker(k_factor=16.0)
        tracker_high = EloTracker(k_factor=64.0)

        tracker_low.update(win_rate=0.8, games=20)
        tracker_high.update(win_rate=0.8, games=20)

        assert abs(tracker_high.current_rating) > abs(tracker_low.current_rating)

    def test_ratings_history_grows(self):
        tracker = EloTracker()
        assert len(tracker.ratings) == 0

        tracker.update(win_rate=0.6, games=10)
        assert len(tracker.ratings) == 1

        tracker.update(win_rate=0.7, games=10)
        assert len(tracker.ratings) == 2

        tracker.update(win_rate=0.4, games=10)
        assert len(tracker.ratings) == 3

    def test_sequential_wins_increase_monotonically(self):
        """Consistent winning should increase rating over time."""
        tracker = EloTracker()
        for _ in range(5):
            tracker.update(win_rate=0.75, games=20)

        # All ratings should be increasing
        for i in range(1, len(tracker.ratings)):
            assert tracker.ratings[i] > tracker.ratings[i - 1]

    def test_expected_value_formula(self):
        """When current == opponent, expected = 0.5."""
        tracker = EloTracker(initial_rating=1000.0)
        # Win rate of 0.5 against self → expected = 0.5, delta = 0
        tracker.update(win_rate=0.5, games=20)
        assert abs(tracker.current_rating - 1000.0) < 0.01

    def test_serialization_roundtrip(self):
        """Ratings list can be saved and restored."""
        tracker = EloTracker(k_factor=24.0, initial_rating=100.0)
        tracker.update(win_rate=0.6, games=10)
        tracker.update(win_rate=0.7, games=10)

        # Simulate save/restore
        saved = {
            "ratings": list(tracker.ratings),
            "k_factor": tracker.k_factor,
            "initial_rating": tracker.initial_rating,
        }

        restored = EloTracker(
            ratings=saved["ratings"],
            k_factor=saved["k_factor"],
            initial_rating=saved["initial_rating"],
        )
        assert restored.current_rating == tracker.current_rating
        assert len(restored.ratings) == len(tracker.ratings)
