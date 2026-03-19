"""Tests for hive_nnue.nnue_encoder — NNUE feature extraction and encoder wrapper."""

import numpy as np
import pytest

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, Move, MoveType
from hive_engine.hex_coord import ORIGIN, ALL_DIRECTIONS
from hive_engine.pieces import Color, PieceType

from hive_nnue.nnue_encoder import (
    NNUEFeatureEncoder,
    NNUEEncoder,
    FEATURE_DIM,
    FEATURES_PER_PIECE,
    NUM_PIECES_PER_PLAYER,
    NUM_DISTANCE_BUCKETS,
    NUM_GLOBAL_FEATURES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def encoder():
    return NNUEEncoder()


@pytest.fixture
def feature_encoder():
    return NNUEFeatureEncoder()


@pytest.fixture
def hive_encoder():
    return HiveEncoder()


@pytest.fixture
def game_state():
    return GameState()


def _place(state: GameState, piece_type: PieceType, pos):
    """Place a piece from current player's hand."""
    hand = state.hand(state.current_player)
    piece = next(p for p in hand if p.piece_type == piece_type)
    state.apply_move(Move(MoveType.PLACE, piece, pos))
    return state


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify feature dimension constants are consistent."""

    def test_feature_dim(self):
        expected = FEATURES_PER_PIECE * NUM_PIECES_PER_PLAYER * 2 + NUM_GLOBAL_FEATURES
        assert FEATURE_DIM == expected
        assert FEATURE_DIM == 428

    def test_features_per_piece(self):
        assert FEATURES_PER_PIECE == 19

    def test_num_pieces_per_player(self):
        assert NUM_PIECES_PER_PLAYER == 11

    def test_distance_buckets(self):
        assert NUM_DISTANCE_BUCKETS == 7

    def test_global_features(self):
        assert NUM_GLOBAL_FEATURES == 10

    def test_action_space_matches(self):
        assert NNUEEncoder.ACTION_SPACE_SIZE == HiveEncoder.ACTION_SPACE_SIZE

    def test_board_size_matches(self):
        assert NNUEEncoder.BOARD_SIZE == HiveEncoder.BOARD_SIZE

    def test_pass_action_matches(self):
        assert NNUEEncoder.PASS_ACTION_INDEX == HiveEncoder.PASS_ACTION_INDEX


# ---------------------------------------------------------------------------
# TestFeatureEncoder
# ---------------------------------------------------------------------------


class TestFeatureEncoder:
    """Tests for the raw NNUE feature extraction."""

    def test_empty_board_shape(self, feature_encoder, game_state):
        features = feature_encoder.encode(game_state)
        assert features.shape == (FEATURE_DIM,)
        assert features.dtype == np.float32

    def test_empty_board_all_in_hand(self, feature_encoder, game_state):
        """On an empty board, all pieces should have is_in_hand = 1."""
        features = feature_encoder.encode(game_state)
        for i in range(NUM_PIECES_PER_PLAYER * 2):
            offset = i * FEATURES_PER_PIECE
            is_on_board = features[offset + 0]
            is_in_hand = features[offset + 1]
            assert is_on_board == 0.0
            assert is_in_hand == 1.0

    def test_placed_piece_on_board(self, feature_encoder):
        game = GameState()
        _place(game, PieceType.QUEEN, ORIGIN)

        features = feature_encoder.encode(game)
        # Now it's Black's turn, so Black is "current player"
        # The queen that was placed is White's, which is now the opponent
        # Opponent pieces start at offset 209
        # Queen is index 0 in the canonical ordering
        opp_queen_offset = NUM_PIECES_PER_PLAYER * FEATURES_PER_PIECE + 0 * FEATURES_PER_PIECE
        assert features[opp_queen_offset + 0] == 1.0  # is_on_board
        assert features[opp_queen_offset + 1] == 0.0  # is_in_hand

    def test_features_bounded(self, feature_encoder):
        """All feature values should be in [0, 1]."""
        game = GameState()
        _place(game, PieceType.QUEEN, ORIGIN)
        neighbor = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        _place(game, PieceType.QUEEN, neighbor)

        features = feature_encoder.encode(game)
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_global_features_region(self, feature_encoder, game_state):
        features = feature_encoder.encode(game_state)
        g_start = FEATURES_PER_PIECE * NUM_PIECES_PER_PLAYER * 2  # 418

        # Current player (White = 1.0 on turn 0)
        assert features[g_start + 0] == 1.0

        # Turn progress (turn 0 / 100 = 0.0)
        assert features[g_start + 1] == 0.0

        # No queens placed yet
        assert features[g_start + 2] == 0.0
        assert features[g_start + 3] == 0.0

        # Queen surrounded counts should be 0
        assert features[g_start + 4] == 0.0
        assert features[g_start + 5] == 0.0

        # Hand counts: 11/11 = 1.0
        assert features[g_start + 6] == 1.0
        assert features[g_start + 7] == 1.0

        # Board counts: 0/11 = 0.0
        assert features[g_start + 8] == 0.0
        assert features[g_start + 9] == 0.0

    def test_perspective_flip(self, feature_encoder):
        """Features should be from current player's perspective."""
        game = GameState()
        _place(game, PieceType.QUEEN, ORIGIN)

        # Now it's Black's turn
        features_black = feature_encoder.encode(game)

        # Black's own pieces are first 209 features
        # White's (opponent) pieces are next 209
        g_start = FEATURES_PER_PIECE * NUM_PIECES_PER_PLAYER * 2
        # Current player is Black = 0.0
        assert features_black[g_start + 0] == 0.0

    def test_queen_distance_encoding(self, feature_encoder):
        """Distance-to-queen features should be one-hot in buckets."""
        game = GameState()
        _place(game, PieceType.QUEEN, ORIGIN)
        neighbor = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        _place(game, PieceType.QUEEN, neighbor)

        # Place an ant for White adjacent to the white queen
        ant_pos = ORIGIN.neighbor(ALL_DIRECTIONS[3])
        _place(game, PieceType.ANT, ant_pos)

        features = feature_encoder.encode(game)

        # Now it's Black's turn. White pieces are opponent (offset 220+).
        # White ant1 is index 1 in canonical order (Q, A1, A2, A3, ...)
        opp_ant_offset = NUM_PIECES_PER_PLAYER * FEATURES_PER_PIECE + 1 * FEATURES_PER_PIECE

        # Distance to own queen (opponent's queen = white queen at ORIGIN)
        # Ant is at a neighbor of ORIGIN, so distance = 1
        dist_own = features[opp_ant_offset + 4: opp_ant_offset + 11]
        assert dist_own.sum() == 1.0  # one-hot

    def test_multi_piece_game(self, feature_encoder):
        """Encoding works for a game with several pieces placed."""
        game = GameState()
        _place(game, PieceType.QUEEN, ORIGIN)
        _place(game, PieceType.QUEEN, ORIGIN.neighbor(ALL_DIRECTIONS[0]))
        _place(game, PieceType.ANT, ORIGIN.neighbor(ALL_DIRECTIONS[3]))
        _place(game, PieceType.ANT, ORIGIN.neighbor(ALL_DIRECTIONS[0]).neighbor(ALL_DIRECTIONS[1]))

        features = feature_encoder.encode(game)
        assert features.shape == (FEATURE_DIM,)
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)


# ---------------------------------------------------------------------------
# TestNNUEEncoder (wrapper)
# ---------------------------------------------------------------------------


class TestEncodeState:
    """Verify encode_state returns correct feature vectors."""

    def test_returns_ndarray(self, encoder, game_state):
        result = encoder.encode_state(game_state)
        assert isinstance(result, np.ndarray)
        assert result.shape == (FEATURE_DIM,)
        assert result.dtype == np.float32


class TestEncodeAction:
    """Verify encode_action matches HiveEncoder exactly."""

    def test_placement_matches(self, encoder, hive_encoder, game_state):
        moves = game_state.legal_moves()
        for move in moves[:5]:
            nnue_idx = encoder.encode_action(move, game_state)
            hive_idx = hive_encoder.encode_action(move, game_state)
            assert nnue_idx == hive_idx

    def test_action_range(self, encoder, game_state):
        moves = game_state.legal_moves()
        for move in moves:
            idx = encoder.encode_action(move, game_state)
            assert 0 <= idx < NNUEEncoder.ACTION_SPACE_SIZE


class TestDecodeAction:
    """Verify decode_action round-trips correctly."""

    def test_roundtrip(self, encoder, game_state):
        moves = game_state.legal_moves()
        for move in moves[:5]:
            idx = encoder.encode_action(move, game_state)
            decoded = encoder.decode_action(idx, game_state)
            assert decoded.move_type == move.move_type
            assert decoded.piece == move.piece
            assert decoded.to == move.to


class TestLegalActionMask:
    """Verify legal action mask matches HiveEncoder."""

    def test_mask_shape(self, encoder, game_state):
        mask = encoder.get_legal_action_mask(game_state)
        assert mask.shape == (NNUEEncoder.ACTION_SPACE_SIZE,)
        assert mask.dtype == np.float32

    def test_mask_matches_hive_encoder(self, encoder, hive_encoder, game_state):
        nnue_mask = encoder.get_legal_action_mask(game_state)
        hive_mask = hive_encoder.get_legal_action_mask(game_state)
        np.testing.assert_array_equal(nnue_mask, hive_mask)

    def test_mask_has_legal_moves(self, encoder, game_state):
        mask = encoder.get_legal_action_mask(game_state)
        assert mask.sum() > 0

    def test_mask_with_pieces(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        _place(s, PieceType.QUEEN, ORIGIN.neighbor(ALL_DIRECTIONS[0]))

        mask = encoder.get_legal_action_mask(s)
        assert mask.shape == (NNUEEncoder.ACTION_SPACE_SIZE,)
        assert mask.sum() > 0

    def test_mask_with_explicit_moves(self, encoder, game_state):
        moves = game_state.legal_moves()
        mask = encoder.get_legal_action_mask(game_state, moves)
        assert mask.sum() == len(moves)
