"""
Tests for TransformerEncoder — the wrapper that provides a unified
interface for MCTS and the trainer, using token sequences.
"""

import numpy as np
import pytest

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, Move, MoveType
from hive_engine.hex_coord import ORIGIN, ALL_DIRECTIONS
from hive_engine.pieces import Color, PieceType

from hive_transformer.token_encoder import TokenEncoder
from hive_transformer.token_types import HiveTokenSequence
from hive_transformer.transformer_encoder import TransformerEncoder


# ---------------------------------------------------------------------------
# TestTransformerEncoderInit
# ---------------------------------------------------------------------------


class TestTransformerEncoderInit:
    """Verify construction and class-level constants."""

    def test_init(self):
        enc = TransformerEncoder()
        assert enc._token_encoder is not None
        assert enc._hive_encoder is not None

    def test_class_constants(self):
        enc = TransformerEncoder()
        assert enc.ACTION_SPACE_SIZE == HiveEncoder.ACTION_SPACE_SIZE
        assert enc.BOARD_SIZE == HiveEncoder.BOARD_SIZE
        assert enc.NUM_CHANNELS == HiveEncoder.NUM_CHANNELS
        assert enc.PASS_ACTION_INDEX == HiveEncoder.PASS_ACTION_INDEX


# ---------------------------------------------------------------------------
# TestEncodeState
# ---------------------------------------------------------------------------


class TestEncodeState:
    """encode_state should return a HiveTokenSequence."""

    def test_encode_initial_state(self):
        enc = TransformerEncoder()
        game = GameState()
        seq = enc.encode_state(game)

        assert isinstance(seq, HiveTokenSequence)
        # Initial state: 1 CLS + 0 board + hand tokens
        assert seq.seq_len >= 1
        assert seq.num_board_tokens == 0

    def test_encode_state_with_pieces(self):
        enc = TransformerEncoder()
        game = GameState()

        # Place a piece
        hand = game.hand(Color.WHITE)
        queen = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        game.apply_move(Move(MoveType.PLACE, queen, ORIGIN))

        seq = enc.encode_state(game)
        assert isinstance(seq, HiveTokenSequence)
        assert seq.num_board_tokens >= 1

    def test_encode_state_returns_same_as_token_encoder(self):
        """TransformerEncoder.encode_state should delegate to TokenEncoder."""
        enc = TransformerEncoder()
        tok_enc = TokenEncoder()
        game = GameState()

        seq1 = enc.encode_state(game)
        seq2 = tok_enc.encode(game)

        np.testing.assert_array_equal(seq1.token_features, seq2.token_features)
        np.testing.assert_array_equal(seq1.token_positions, seq2.token_positions)
        np.testing.assert_array_equal(seq1.token_types, seq2.token_types)
        np.testing.assert_array_equal(seq1.global_features, seq2.global_features)
        assert seq1.num_board_tokens == seq2.num_board_tokens


# ---------------------------------------------------------------------------
# TestEncodeAction
# ---------------------------------------------------------------------------


class TestEncodeAction:
    """encode_action should delegate to HiveEncoder."""

    def test_encode_placement(self):
        enc = TransformerEncoder()
        hive_enc = HiveEncoder()
        game = GameState()

        hand = game.hand(Color.WHITE)
        queen = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        move = Move(MoveType.PLACE, queen, ORIGIN)

        idx1 = enc.encode_action(move, game)
        idx2 = hive_enc.encode_action(move, game)
        assert idx1 == idx2

    def test_encode_action_is_valid_index(self):
        enc = TransformerEncoder()
        game = GameState()

        hand = game.hand(Color.WHITE)
        queen = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        move = Move(MoveType.PLACE, queen, ORIGIN)

        idx = enc.encode_action(move, game)
        assert 0 <= idx < enc.ACTION_SPACE_SIZE


# ---------------------------------------------------------------------------
# TestDecodeAction
# ---------------------------------------------------------------------------


class TestDecodeAction:
    """decode_action should delegate to HiveEncoder."""

    def test_roundtrip(self):
        enc = TransformerEncoder()
        game = GameState()

        hand = game.hand(Color.WHITE)
        queen = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        move = Move(MoveType.PLACE, queen, ORIGIN)

        idx = enc.encode_action(move, game)
        decoded = enc.decode_action(idx, game)

        assert decoded.move_type == move.move_type
        assert decoded.piece.piece_type == move.piece.piece_type
        assert decoded.to == move.to


# ---------------------------------------------------------------------------
# TestGetLegalActionMask
# ---------------------------------------------------------------------------


class TestGetLegalActionMask:
    """get_legal_action_mask should delegate to HiveEncoder."""

    def test_initial_state_mask(self):
        enc = TransformerEncoder()
        game = GameState()

        mask = enc.get_legal_action_mask(game)
        assert mask.shape == (enc.ACTION_SPACE_SIZE,)
        assert mask.dtype == np.float32 or mask.dtype == np.bool_

        # Some actions should be legal on the initial board
        assert mask.sum() > 0

    def test_mask_matches_hive_encoder(self):
        enc = TransformerEncoder()
        hive_enc = HiveEncoder()
        game = GameState()

        mask1 = enc.get_legal_action_mask(game)
        mask2 = hive_enc.get_legal_action_mask(game)

        np.testing.assert_array_equal(mask1, mask2)

    def test_mask_with_legal_moves_arg(self):
        enc = TransformerEncoder()
        game = GameState()
        legal_moves = game.legal_moves()

        mask = enc.get_legal_action_mask(game, legal_moves)
        assert mask.shape == (enc.ACTION_SPACE_SIZE,)
        assert mask.sum() > 0

    def test_mask_after_moves(self):
        enc = TransformerEncoder()
        game = GameState()

        # Place a few pieces
        hand = game.hand(Color.WHITE)
        queen_w = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        game.apply_move(Move(MoveType.PLACE, queen_w, ORIGIN))

        hand_b = game.hand(Color.BLACK)
        queen_b = next(p for p in hand_b if p.piece_type == PieceType.QUEEN)
        neighbor = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        game.apply_move(Move(MoveType.PLACE, queen_b, neighbor))

        mask = enc.get_legal_action_mask(game)
        assert mask.shape == (enc.ACTION_SPACE_SIZE,)
        assert mask.sum() > 0
