"""Tests for archived transformer token encoding."""

import numpy as np
import pytest

from hive_engine.game_state import GameState, Move, MoveType
from hive_engine.hex_coord import ORIGIN, ALL_DIRECTIONS
from hive_engine.pieces import Color, PieceType

from archive.legacy_transformer.hive_transformer.token_encoder import TokenEncoder
from hive_common.token_types import (
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
        assert feat[8] == 1.0  # WHITE
        assert feat[10] == 1.0  # is_on_ground
        assert feat[11] == 1.0  # is_on_top
        assert feat[13] == 1.0  # is_queen
        assert feat[22] == 0.0  # not hand

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
            assert seq.token_features[i, 15] == pytest.approx(1.0 / 6.0)


class TestGlobalFeatures:
    def test_queen_placed_flags(self, encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        seq = encoder.encode(s)
        assert seq.global_features[2] == 1.0  # white queen placed
        assert seq.global_features[3] == 0.0  # black not placed

    def test_hand_fraction(self, encoder):
        seq = encoder.encode(GameState())
        assert seq.global_features[4] == pytest.approx(11.0 / 14.0)  # full hand
        assert seq.global_features[5] == pytest.approx(11.0 / 14.0)

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
            assert seq.token_features[i, 22] == 1.0

    def test_hand_diminishes(self, encoder):
        s = GameState()
        _place(s, PieceType.ANT, ORIGIN)
        seq = encoder.encode(s)
        # Find white ant hand token
        for i in range(seq.num_board_tokens + 1, seq.seq_len):
            feat = seq.token_features[i]
            if feat[PieceType.ANT.value] == 1.0 and feat[8] == 1.0:  # ANT + WHITE
                assert feat[23] == pytest.approx(2.0 / 3.0)  # 2 remaining of 3
                break


class TestDtypes:
    def test_all_dtypes(self, encoder):
        seq = encoder.encode(GameState())
        assert seq.token_features.dtype == np.float32
        assert seq.token_positions.dtype == np.int32
        assert seq.token_types.dtype == np.int32
        assert seq.global_features.dtype == np.float32


# ---------------------------------------------------------------------------
# TestStackedPieces
# ---------------------------------------------------------------------------


class TestStackedPieces:
    """Token encoding with beetles creating stacked positions."""

    def _stacked_state(self):
        """Create a state with a beetle on top of a piece.

        Board after setup:
            ORIGIN: White Queen (bottom) + White Beetle (top) -- STACK of 2
            ORIGIN.E: Black Queen
            ORIGIN.E.E: Black Ant
        """
        from hive_engine.hex_coord import HexCoord
        s = GameState()
        # Turn 0: White Queen at origin
        _place(s, PieceType.QUEEN, ORIGIN)
        # Turn 1: Black Queen east
        e = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        _place(s, PieceType.QUEEN, e)
        # Turn 2: White Beetle west of origin
        w = ORIGIN.neighbor(ALL_DIRECTIONS[3])
        _place(s, PieceType.BEETLE, w)
        # Turn 3: Black Ant east of black queen
        ee = e.neighbor(ALL_DIRECTIONS[0])
        _place(s, PieceType.ANT, ee)
        # Turn 4: Move white beetle from W onto origin (on top of queen)
        beetle = None
        for p in s.board.grid[w]:
            if p.piece_type == PieceType.BEETLE:
                beetle = p
                break
        move = Move(MoveType.MOVE, beetle, ORIGIN)
        s.apply_move(move)
        return s

    def test_stacked_creates_extra_tokens(self, encoder):
        """Stacked position should produce 2 board tokens instead of 1."""
        s = self._stacked_state()
        seq = encoder.encode(s)
        # 3 positions: ORIGIN (stack of 2), E (1), EE (1) → 4 board tokens
        assert seq.num_board_tokens == 4

    def test_stacked_tokens_share_position(self, encoder):
        """Both pieces in a stack should have the same position index."""
        s = self._stacked_state()
        seq = encoder.encode(s)
        # Board tokens are at indices 1..num_board_tokens
        board_positions = seq.token_positions[1:1 + seq.num_board_tokens]
        from collections import Counter
        pos_counts = Counter(board_positions.tolist())
        stacked = [pos for pos, cnt in pos_counts.items() if cnt > 1]
        assert len(stacked) >= 1

    def test_stack_position_feature(self, encoder):
        """Bottom piece stack_position=0.0, top piece stack_position=0.25."""
        s = self._stacked_state()
        seq = encoder.encode(s)
        board_positions = seq.token_positions[1:1 + seq.num_board_tokens]
        from collections import Counter
        pos_counts = Counter(board_positions.tolist())
        for pos, cnt in pos_counts.items():
            if cnt == 2:
                indices = [i + 1 for i in range(seq.num_board_tokens)
                           if seq.token_positions[i + 1] == pos]
                # Bottom piece
                assert seq.token_features[indices[0], 24] == pytest.approx(0.0)
                # Top piece
                assert seq.token_features[indices[1], 24] == pytest.approx(0.25)

    def test_is_on_ground_and_top(self, encoder):
        """Bottom: is_on_ground=1, is_on_top=0. Top: opposite."""
        s = self._stacked_state()
        seq = encoder.encode(s)
        board_positions = seq.token_positions[1:1 + seq.num_board_tokens]
        from collections import Counter
        pos_counts = Counter(board_positions.tolist())
        for pos, cnt in pos_counts.items():
            if cnt == 2:
                indices = [i + 1 for i in range(seq.num_board_tokens)
                           if seq.token_positions[i + 1] == pos]
                # Bottom piece
                assert seq.token_features[indices[0], 10] == 1.0   # is_on_ground
                assert seq.token_features[indices[0], 11] == 0.0   # NOT is_on_top
                # Top piece
                assert seq.token_features[indices[1], 10] == 0.0   # NOT is_on_ground
                assert seq.token_features[indices[1], 11] == 1.0   # is_on_top

    def test_all_tokens_are_board_type(self, encoder):
        """All stacked piece tokens should have BOARD type."""
        s = self._stacked_state()
        seq = encoder.encode(s)
        for i in range(1, 1 + seq.num_board_tokens):
            assert seq.token_types[i] == TOKEN_TYPE_BOARD

    def test_feature_dim_with_stacking(self, encoder):
        """Feature dimension should be TOKEN_FEAT_DIM (25) with stacking."""
        s = self._stacked_state()
        seq = encoder.encode(s)
        assert seq.token_features.shape[1] == TOKEN_FEAT_DIM
