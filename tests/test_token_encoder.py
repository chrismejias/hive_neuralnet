"""Tests for hive_transformer.token_encoder — GameState → HiveTokenSequence."""

import numpy as np
import pytest

from hive_engine.game_state import GameState, Move, MoveType
from hive_engine.hex_coord import ORIGIN, ALL_DIRECTIONS
from hive_engine.pieces import Color, PieceType

from hive_transformer.token_encoder import TokenEncoder
from hive_transformer.token_types import (
    TOKEN_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    TOKEN_TYPE_CLS,
    TOKEN_TYPE_BOARD,
    TOKEN_TYPE_HAND,
    OFF_BOARD_POSITION,
)


@pytest.fixture
def encoder():
    return TokenEncoder()


def _place(state, piece_type, pos):
    """Place a piece from current player's hand (in-place)."""
    hand = state.hand(state.current_player)
    piece = next(p for p in hand if p.piece_type == piece_type)
    state.apply_move(Move(MoveType.PLACE, piece, pos))
    return state


class TestEmptyBoard:
    def test_cls_token_present(self, encoder):
        seq = encoder.encode(GameState())
        assert seq.token_types[0] == TOKEN_TYPE_CLS

    def test_no_board_tokens(self, encoder):
        seq = encoder.encode(GameState())
        assert seq.num_board_tokens == 0

    def test_hand_tokens_count(self, encoder):
        seq = encoder.encode(GameState())
        # 5 piece types × 2 colors = 10 hand tokens + 1 CLS
        assert seq.seq_len == 11

    def test_cls_features_zero(self, encoder):
        seq = encoder.encode(GameState())
        assert (seq.token_features[0] == 0).all()

    def test_cls_position(self, encoder):
        seq = encoder.encode(GameState())
        assert seq.token_positions[0] == OFF_BOARD_POSITION

    def test_global_features(self, encoder):
        seq = encoder.encode(GameState())
        assert seq.global_features.shape == (GLOBAL_FEAT_DIM,)
        assert seq.global_features[0] == 1.0  # WHITE to move
        assert seq.global_features[1] == 0.0  # turn 0

    def test_hand_token_types(self, encoder):
        seq = encoder.encode(GameState())
        for i in range(1, seq.seq_len):
            assert seq.token_types[i] == TOKEN_TYPE_HAND

    def test_hand_positions_off_board(self, encoder):
        seq = encoder.encode(GameState())
        for i in range(1, seq.seq_len):
            assert seq.token_positions[i] == OFF_BOARD_POSITION


class TestSinglePiece:
    def test_one_board_token(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        seq = encoder.encode(s)
        assert seq.num_board_tokens == 1

    def test_board_token_type(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        seq = encoder.encode(s)
        assert seq.token_types[1] == TOKEN_TYPE_BOARD

    def test_board_position_valid(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        seq = encoder.encode(s)
        pos = seq.token_positions[1]
        assert 0 <= pos <= 168

    def test_queen_features(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        seq = encoder.encode(s)
        feat = seq.token_features[1]
        assert feat[PieceType.QUEEN.value] == 1.0
        assert feat[5] == 1.0  # WHITE
        assert feat[7] == 1.0  # is_on_ground
        assert feat[8] == 1.0  # is_on_top
        assert feat[10] == 1.0  # is_queen
        assert feat[19] == 0.0  # not hand

    def test_feature_dim(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        seq = encoder.encode(s)
        assert seq.token_features.shape[1] == TOKEN_FEAT_DIM


class TestTwoPieces:
    def _two_piece_state(self):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        _place(s, PieceType.QUEEN, ORIGIN.neighbor(ALL_DIRECTIONS[0]))
        return s

    def test_two_board_tokens(self, encoder):
        s = self._two_piece_state()
        seq = encoder.encode(s)
        assert seq.num_board_tokens == 2

    def test_different_positions(self, encoder):
        s = self._two_piece_state()
        seq = encoder.encode(s)
        pos1 = seq.token_positions[1]
        pos2 = seq.token_positions[2]
        assert pos1 != pos2

    def test_occupied_neighbors(self, encoder):
        s = self._two_piece_state()
        seq = encoder.encode(s)
        # Both pieces have 1 occupied neighbor
        for i in [1, 2]:
            assert seq.token_features[i, 12] == pytest.approx(1.0 / 6.0)


class TestGlobalFeatures:
    def test_queen_placed_flags(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        seq = encoder.encode(s)
        assert seq.global_features[2] == 1.0  # white queen placed
        assert seq.global_features[3] == 0.0  # black not placed

    def test_hand_fraction(self, encoder):
        seq = encoder.encode(GameState())
        assert seq.global_features[4] == pytest.approx(1.0)  # full hand
        assert seq.global_features[5] == pytest.approx(1.0)

    def test_turn_progress(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        _place(s, PieceType.QUEEN, ORIGIN.neighbor(ALL_DIRECTIONS[0]))
        seq = encoder.encode(s)
        assert seq.global_features[1] == pytest.approx(2.0 / 100.0)


class TestHandTokens:
    def test_hand_is_hand_flag(self, encoder):
        seq = encoder.encode(GameState())
        for i in range(1, seq.seq_len):
            assert seq.token_features[i, 19] == 1.0

    def test_hand_diminishes(self, encoder):
        s = GameState()
        _place(s, PieceType.ANT, ORIGIN)
        seq = encoder.encode(s)
        # Find white ant hand token
        for i in range(seq.num_board_tokens + 1, seq.seq_len):
            feat = seq.token_features[i]
            if feat[PieceType.ANT.value] == 1.0 and feat[5] == 1.0:  # ANT + WHITE
                assert feat[20] == pytest.approx(2.0 / 3.0)  # 2 remaining of 3
                break


class TestDtypes:
    def test_all_dtypes(self, encoder):
        seq = encoder.encode(GameState())
        assert seq.token_features.dtype == np.float32
        assert seq.token_positions.dtype == np.int32
        assert seq.token_types.dtype == np.int32
        assert seq.global_features.dtype == np.float32
