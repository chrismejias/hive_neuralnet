"""Tests for hive_gnn.gnn_encoder — GNNEncoder interface wrapper."""

import numpy as np
import pytest

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, Move, MoveType
from hive_engine.hex_coord import ORIGIN, ALL_DIRECTIONS
from hive_engine.pieces import Color, PieceType

from hive_gnn.gnn_encoder import GNNEncoder
from hive_gnn.graph_types import HiveGraph, NODE_FEAT_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gnn_encoder():
    return GNNEncoder()


@pytest.fixture
def hive_encoder():
    return HiveEncoder()


@pytest.fixture
def game_state():
    return GameState()


def _place(state: GameState, piece_type: PieceType, pos):
    """Place a piece from current player's hand (in-place)."""
    hand = state.hand(state.current_player)
    piece = next(p for p in hand if p.piece_type == piece_type)
    move = Move(move_type=MoveType.PLACE, piece=piece, to=pos)
    state.apply_move(move)
    return state


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constants match between GNNEncoder and HiveEncoder."""

    def test_action_space_size(self):
        assert GNNEncoder.ACTION_SPACE_SIZE == HiveEncoder.ACTION_SPACE_SIZE

    def test_board_size(self):
        assert GNNEncoder.BOARD_SIZE == HiveEncoder.BOARD_SIZE

    def test_pass_action_index(self):
        assert GNNEncoder.PASS_ACTION_INDEX == HiveEncoder.PASS_ACTION_INDEX


# ---------------------------------------------------------------------------
# TestEncodeState
# ---------------------------------------------------------------------------


class TestEncodeState:
    """Verify encode_state returns a HiveGraph."""

    def test_returns_hive_graph(self, gnn_encoder, game_state):
        result = gnn_encoder.encode_state(game_state)
        assert isinstance(result, HiveGraph)

    def test_node_features_shape(self, gnn_encoder, game_state):
        graph = gnn_encoder.encode_state(game_state)
        assert graph.node_features.shape[1] == NODE_FEAT_DIM

    def test_with_pieces(self, gnn_encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        graph = gnn_encoder.encode_state(s)
        assert graph.num_piece_nodes == 1
        assert isinstance(graph, HiveGraph)


# ---------------------------------------------------------------------------
# TestEncodeAction
# ---------------------------------------------------------------------------


class TestEncodeAction:
    """Verify encode_action matches HiveEncoder exactly."""

    def test_placement_matches(self, gnn_encoder, hive_encoder, game_state):
        moves = game_state.legal_moves()
        for move in moves[:5]:  # Test first few
            gnn_idx = gnn_encoder.encode_action(move, game_state)
            hive_idx = hive_encoder.encode_action(move, game_state)
            assert gnn_idx == hive_idx

    def test_action_range(self, gnn_encoder, game_state):
        moves = game_state.legal_moves()
        for move in moves:
            idx = gnn_encoder.encode_action(move, game_state)
            assert 0 <= idx < GNNEncoder.ACTION_SPACE_SIZE


# ---------------------------------------------------------------------------
# TestDecodeAction
# ---------------------------------------------------------------------------


class TestDecodeAction:
    """Verify decode_action matches HiveEncoder."""

    def test_roundtrip(self, gnn_encoder, game_state):
        moves = game_state.legal_moves()
        for move in moves[:5]:
            idx = gnn_encoder.encode_action(move, game_state)
            decoded = gnn_encoder.decode_action(idx, game_state)
            assert decoded.move_type == move.move_type
            assert decoded.piece == move.piece
            assert decoded.to == move.to


# ---------------------------------------------------------------------------
# TestLegalActionMask
# ---------------------------------------------------------------------------


class TestLegalActionMask:
    """Verify legal action mask matches HiveEncoder."""

    def test_mask_shape(self, gnn_encoder, game_state):
        mask = gnn_encoder.get_legal_action_mask(game_state)
        assert mask.shape == (GNNEncoder.ACTION_SPACE_SIZE,)
        assert mask.dtype == np.float32

    def test_mask_matches_hive_encoder(self, gnn_encoder, hive_encoder, game_state):
        gnn_mask = gnn_encoder.get_legal_action_mask(game_state)
        hive_mask = hive_encoder.get_legal_action_mask(game_state)
        np.testing.assert_array_equal(gnn_mask, hive_mask)

    def test_mask_has_legal_moves(self, gnn_encoder, game_state):
        mask = gnn_encoder.get_legal_action_mask(game_state)
        assert mask.sum() > 0  # Fresh game has legal placements

    def test_mask_with_pieces_on_board(self, gnn_encoder):
        s = GameState()
        _place(s, PieceType.QUEEN, ORIGIN)
        neighbor = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        _place(s, PieceType.QUEEN, neighbor)

        mask = gnn_encoder.get_legal_action_mask(s)
        assert mask.shape == (GNNEncoder.ACTION_SPACE_SIZE,)
        assert mask.sum() > 0

    def test_mask_with_explicit_moves(self, gnn_encoder, game_state):
        moves = game_state.legal_moves()
        mask = gnn_encoder.get_legal_action_mask(game_state, moves)
        assert mask.sum() == len(moves)
