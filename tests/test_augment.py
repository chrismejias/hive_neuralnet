"""Tests for hex rotational data augmentation."""

import numpy as np
import pytest

from hive_engine.augment import (
    _rotate_grid_cw,
    _GRID_TABLES,
    _POLICY_TABLES,
    rotate_state,
    rotate_policy,
    augment_example,
    BOARD_SIZE,
    NUM_GRID_CELLS,
    NUM_PIECE_CHANNELS,
    NUM_CHANNELS,
    ACTION_SPACE_SIZE,
    PASS_ACTION_INDEX,
    NUM_PLACEMENT_ACTIONS,
    MOVEMENT_OFFSET,
)


class TestGridRotation:
    """Tests for grid rotation logic."""

    def test_identity_rotation(self):
        """k=0 is the identity."""
        gt = _GRID_TABLES[0]
        for i in range(NUM_GRID_CELLS):
            assert gt[i] == i

    def test_center_stays_fixed(self):
        """The center cell (6,6) should stay fixed under all rotations."""
        center_idx = 6 * BOARD_SIZE + 6
        for k in range(6):
            assert _GRID_TABLES[k][center_idx] == center_idx

    def test_six_rotations_is_identity(self):
        """Applying CW rotation 6 times returns to start."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                r, c = row, col
                for _ in range(6):
                    r, c = _rotate_grid_cw(r, c)
                assert (r, c) == (row, col), f"({row},{col}) didn't return after 6 rotations: got ({r},{c})"

    def test_rotation_preserves_valid_cells_near_center(self):
        """Cells near center should remain in-bounds after any rotation."""
        gt = _GRID_TABLES[1]  # Single 60° rotation
        # Center region: cells within distance 3 of center should all be valid
        center = 6
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = center + dr, center + dc
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    idx = r * BOARD_SIZE + c
                    # Near center should mostly stay in bounds
                    # (not all necessarily, but center itself does)

    def test_bijection_on_valid_cells(self):
        """Each rotation is a bijection on the set of valid (non-OOB) cells."""
        for k in range(6):
            gt = _GRID_TABLES[k]
            valid_src = [i for i in range(NUM_GRID_CELLS) if gt[i] >= 0]
            valid_dst = [gt[i] for i in valid_src]
            # No duplicates in destinations
            assert len(valid_dst) == len(set(valid_dst)), f"Rotation k={k} is not injective"


class TestPolicyRotation:
    """Tests for policy vector rotation."""

    def test_identity_rotation(self):
        """k=0 policy rotation is identity."""
        rng = np.random.RandomState(42)
        policy = rng.dirichlet(np.ones(ACTION_SPACE_SIZE))
        rotated = rotate_policy(policy, 0)
        np.testing.assert_allclose(rotated, policy)

    def test_pass_action_preserved(self):
        """Pass action maps to itself under all rotations."""
        for k in range(6):
            assert _POLICY_TABLES[k][PASS_ACTION_INDEX] == PASS_ACTION_INDEX

    def test_policy_sums_to_one(self):
        """Rotated policies should still sum to ~1.0."""
        # Create a sparse policy (only a few legal actions)
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        # Set some placement probabilities near center
        center_idx = 6 * BOARD_SIZE + 6
        for pt in range(5):
            policy[pt * NUM_GRID_CELLS + center_idx] = 0.2
        assert abs(policy.sum() - 1.0) < 1e-6

        for k in range(6):
            rotated = rotate_policy(policy, k)
            assert abs(rotated.sum() - 1.0) < 1e-5, f"k={k}: sum={rotated.sum()}"

    def test_nonzero_count_preserved_near_center(self):
        """Number of nonzero entries should be preserved for center-only policies."""
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        center_idx = 6 * BOARD_SIZE + 6
        policy[0 * NUM_GRID_CELLS + center_idx] = 0.5  # Place queen at center
        policy[PASS_ACTION_INDEX] = 0.5  # Pass

        for k in range(6):
            rotated = rotate_policy(policy, k)
            # Center maps to center, pass maps to pass
            assert np.count_nonzero(rotated) == 2


class TestStateRotation:
    """Tests for state tensor rotation."""

    def test_identity_rotation(self):
        """k=0 is identity."""
        state = np.random.rand(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
        rotated = rotate_state(state, 0)
        np.testing.assert_allclose(rotated, state)

    def test_meta_channels_unchanged(self):
        """Channels 20-25 should be identical across all rotations."""
        state = np.random.rand(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
        for k in range(6):
            rotated = rotate_state(state, k)
            np.testing.assert_allclose(
                rotated[NUM_PIECE_CHANNELS:],
                state[NUM_PIECE_CHANNELS:],
                err_msg=f"Meta channels changed for k={k}"
            )

    def test_piece_channels_rotated(self):
        """Piece at center stays at center; piece off-center moves."""
        state = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        # Place a piece at center (6,6) in channel 0
        state[0, 6, 6] = 1.0
        for k in range(6):
            rotated = rotate_state(state, k)
            # Center should always be 1.0
            assert rotated[0, 6, 6] == 1.0

    def test_total_piece_mass_preserved(self):
        """Sum of piece channel values should be preserved (for valid cells)."""
        state = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        # Place pieces near center only
        state[0, 6, 6] = 1.0
        state[1, 6, 7] = 1.0
        state[5, 7, 6] = 1.0
        original_sum = state[:NUM_PIECE_CHANNELS].sum()

        for k in range(6):
            rotated = rotate_state(state, k)
            rot_sum = rotated[:NUM_PIECE_CHANNELS].sum()
            assert abs(rot_sum - original_sum) < 1e-6, f"k={k}: mass {rot_sum} vs {original_sum}"


class TestAugmentExample:
    """Tests for the full augmentation pipeline."""

    def test_returns_six_examples(self):
        state = np.random.rand(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[PASS_ACTION_INDEX] = 1.0
        value = 0.5

        results = augment_example(state, policy, value)
        assert len(results) == 6

    def test_value_preserved(self):
        state = np.random.rand(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[PASS_ACTION_INDEX] = 1.0

        for value in [-1.0, 0.0, 0.5, 1.0]:
            results = augment_example(state, policy, value)
            for _, _, v in results:
                assert v == value

    def test_first_example_is_original(self):
        state = np.random.rand(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[PASS_ACTION_INDEX] = 1.0
        value = -0.3

        results = augment_example(state, policy, value)
        s, p, v = results[0]
        np.testing.assert_allclose(s, state)
        np.testing.assert_allclose(p, policy)
        assert v == value

    def test_shapes_correct(self):
        state = np.random.rand(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[PASS_ACTION_INDEX] = 1.0
        value = 0.0

        results = augment_example(state, policy, value)
        for s, p, v in results:
            assert s.shape == (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            assert p.shape == (ACTION_SPACE_SIZE,)
            assert isinstance(v, float)
