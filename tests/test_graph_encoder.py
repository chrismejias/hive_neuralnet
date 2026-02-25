"""Tests for hive_gnn.graph_encoder – GameState → HiveGraph conversion."""

import numpy as np
import pytest

from hive_engine.game_state import GameState, Move, MoveType
from hive_engine.hex_coord import HexCoord, ORIGIN, ALL_DIRECTIONS
from hive_engine.pieces import Color, PieceType, Piece

from hive_gnn.graph_encoder import GraphEncoder
from hive_gnn.graph_types import NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def encoder():
    return GraphEncoder()


@pytest.fixture
def empty_state():
    return GameState()


def _place(state: GameState, piece_type: PieceType, pos: HexCoord) -> GameState:
    """Helper: place a piece from the current player's hand at pos (in-place)."""
    hand = state.hand(state.current_player)
    piece = next(p for p in hand if p.piece_type == piece_type)
    move = Move(move_type=MoveType.PLACE, piece=piece, to=pos)
    state.apply_move(move)
    return state


# ---------------------------------------------------------------------------
# TestEmptyBoard
# ---------------------------------------------------------------------------


class TestEmptyBoard:
    """Graph encoding of a fresh game (no pieces on board)."""

    def test_no_piece_nodes(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        assert g.num_piece_nodes == 0

    def test_hand_nodes_present(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        # Both players have 5 piece types in hand → up to 10 hand nodes
        num_hand = g.node_features.shape[0] - g.num_piece_nodes
        assert num_hand == 10  # 5 types × 2 colors

    def test_no_edges(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        assert g.edge_index.shape == (2, 0)
        assert g.edge_features.shape == (0, EDGE_FEAT_DIM)

    def test_global_features_shape(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        assert g.global_features.shape == (GLOBAL_FEAT_DIM,)
        assert g.global_features.dtype == np.float32

    def test_global_current_player_white(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        assert g.global_features[0] == 1.0  # WHITE

    def test_global_turn_zero(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        assert g.global_features[1] == 0.0  # turn 0

    def test_hand_node_features(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        # All nodes should be hand nodes
        for i in range(g.node_features.shape[0]):
            assert g.node_features[i, 19] == 1.0  # is_hand_node


# ---------------------------------------------------------------------------
# TestSinglePiece
# ---------------------------------------------------------------------------


class TestSinglePiece:
    """Graph encoding after placing a single piece."""

    def test_one_piece_node(self, encoder, empty_state):
        s = _place(empty_state, PieceType.QUEEN, ORIGIN)
        g = encoder.encode(s)
        assert g.num_piece_nodes == 1

    def test_piece_node_features(self, encoder, empty_state):
        s = _place(empty_state, PieceType.QUEEN, ORIGIN)
        g = encoder.encode(s)

        feat = g.node_features[0]
        # Queen one-hot
        assert feat[PieceType.QUEEN.value] == 1.0
        assert feat[PieceType.ANT.value] == 0.0
        # Color: WHITE
        assert feat[5] == 1.0  # WHITE
        assert feat[6] == 0.0  # BLACK
        # is_on_ground
        assert feat[7] == 1.0
        # is_on_top
        assert feat[8] == 1.0
        # stack_height / 4
        assert feat[9] == pytest.approx(0.25)
        # is_queen
        assert feat[10] == 1.0
        # No occupied neighbors for a single piece
        assert feat[12] == 0.0
        # All 6 directions empty
        for d in range(6):
            assert feat[13 + d] == 1.0

    def test_no_edges_for_single_piece(self, encoder, empty_state):
        s = _place(empty_state, PieceType.QUEEN, ORIGIN)
        g = encoder.encode(s)
        assert g.edge_index.shape == (2, 0)

    def test_positions_array(self, encoder, empty_state):
        s = _place(empty_state, PieceType.QUEEN, ORIGIN)
        g = encoder.encode(s)
        assert g.node_positions.shape == (1, 2)

    def test_piece_types_array(self, encoder, empty_state):
        s = _place(empty_state, PieceType.QUEEN, ORIGIN)
        g = encoder.encode(s)
        assert g.node_piece_types.shape == (1,)
        assert g.node_piece_types[0] == PieceType.QUEEN.value


# ---------------------------------------------------------------------------
# TestTwoPieces
# ---------------------------------------------------------------------------


class TestTwoPieces:
    """Graph encoding with two adjacent pieces → edges appear."""

    def _two_piece_state(self):
        s = GameState()
        s = _place(s, PieceType.QUEEN, ORIGIN)  # White Queen at origin
        neighbor_pos = ORIGIN.neighbor(ALL_DIRECTIONS[0])  # East
        s = _place(s, PieceType.QUEEN, neighbor_pos)  # Black Queen adjacent
        return s

    def test_two_piece_nodes(self, encoder):
        s = self._two_piece_state()
        g = encoder.encode(s)
        assert g.num_piece_nodes == 2

    def test_bidirectional_edges(self, encoder):
        s = self._two_piece_state()
        g = encoder.encode(s)
        # Two adjacent pieces → 2 edges (bidirectional)
        assert g.edge_index.shape[1] == 2

    def test_edge_features_shape(self, encoder):
        s = self._two_piece_state()
        g = encoder.encode(s)
        assert g.edge_features.shape == (2, EDGE_FEAT_DIM)
        assert g.edge_features.dtype == np.float32

    def test_edge_direction_onehot(self, encoder):
        s = self._two_piece_state()
        g = encoder.encode(s)
        # Each edge should have exactly one direction bit set
        for e in range(g.edge_features.shape[0]):
            dir_bits = g.edge_features[e, 2:8]
            assert dir_bits.sum() == pytest.approx(1.0)

    def test_edge_dq_dr_opposite(self, encoder):
        s = self._two_piece_state()
        g = encoder.encode(s)
        # Edges are bidirectional, so dq/dr should be negatives of each other
        e0_dq = g.edge_features[0, 0]
        e1_dq = g.edge_features[1, 0]
        assert e0_dq == pytest.approx(-e1_dq)

    def test_occupied_neighbors_updated(self, encoder):
        s = self._two_piece_state()
        g = encoder.encode(s)
        # Each piece has 1 occupied neighbor
        for i in range(2):
            assert g.node_features[i, 12] == pytest.approx(1.0 / 6.0)

    def test_empty_dir_mask(self, encoder):
        s = self._two_piece_state()
        g = encoder.encode(s)
        # Each piece has 5 empty directions (1 occupied neighbor)
        for i in range(2):
            empty_dirs = g.node_features[i, 13:19].sum()
            assert empty_dirs == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TestGlobalFeatures
# ---------------------------------------------------------------------------


class TestGlobalFeatures:
    """Verify global features update correctly."""

    def test_queen_placed_flags(self, encoder):
        s = GameState()
        s = _place(s, PieceType.QUEEN, ORIGIN)  # White queen placed
        g = encoder.encode(s)
        assert g.global_features[2] == 1.0  # white queen placed
        assert g.global_features[3] == 0.0  # black queen not placed

    def test_hand_fraction(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        # Full hand: 11 pieces / 11.0 = 1.0
        assert g.global_features[4] == pytest.approx(1.0)
        assert g.global_features[5] == pytest.approx(1.0)

    def test_hand_decreases_after_placement(self, encoder):
        s = GameState()
        s = _place(s, PieceType.QUEEN, ORIGIN)
        g = encoder.encode(s)
        assert g.global_features[4] == pytest.approx(10.0 / 11.0)  # White played 1

    def test_turn_progress(self, encoder):
        s = GameState()
        s = _place(s, PieceType.QUEEN, ORIGIN)  # turn 0→1
        neighbor = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        s = _place(s, PieceType.QUEEN, neighbor)  # turn 1→2
        g = encoder.encode(s)
        assert g.global_features[1] == pytest.approx(2.0 / 100.0)


# ---------------------------------------------------------------------------
# TestNodeFeatureDimensions
# ---------------------------------------------------------------------------


class TestNodeFeatureDimensions:
    """Verify feature dimensions are consistent."""

    def test_node_feature_dim(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        assert g.node_features.shape[1] == NODE_FEAT_DIM

    def test_dtypes(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        assert g.node_features.dtype == np.float32
        assert g.edge_index.dtype == np.int64
        assert g.edge_features.dtype == np.float32
        assert g.global_features.dtype == np.float32
        assert g.node_positions.dtype == np.int32
        assert g.node_piece_types.dtype == np.int32


# ---------------------------------------------------------------------------
# TestHandNodes
# ---------------------------------------------------------------------------


class TestHandNodes:
    """Verify hand node encoding."""

    def test_hand_count_feature(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        # All nodes are hand nodes in empty state
        for i in range(g.node_features.shape[0]):
            feat = g.node_features[i]
            assert feat[19] == 1.0  # is_hand_node
            assert feat[20] > 0.0  # count_remaining > 0

    def test_hand_nodes_have_no_spatial_features(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        for i in range(g.node_features.shape[0]):
            feat = g.node_features[i]
            assert feat[7] == 0.0  # is_on_ground
            assert feat[8] == 0.0  # is_on_top
            assert feat[9] == 0.0  # stack_height
            assert feat[12] == 0.0  # num_occupied_neighbors
            # empty_dir_mask all zeros for hand
            assert feat[13:19].sum() == 0.0

    def test_hand_diminishes_after_placement(self, encoder):
        s = GameState()
        g_before = encoder.encode(s)
        n_hand_before = g_before.node_features.shape[0]

        # Place all 3 ants for white and 1 for black
        s = _place(s, PieceType.ANT, ORIGIN)
        neighbor = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        s = _place(s, PieceType.ANT, neighbor)

        g_after = encoder.encode(s)
        # 2 piece nodes on board now, hand nodes adjusted
        assert g_after.num_piece_nodes == 2
        # Total nodes should be different (2 piece + remaining hand types)
        total_after = g_after.node_features.shape[0]
        # Hand nodes have piece count feature updated
        hand_nodes = g_after.node_features[g_after.num_piece_nodes:]
        # Find the white ant hand node
        for i in range(hand_nodes.shape[0]):
            feat = hand_nodes[i]
            if feat[PieceType.ANT.value] == 1.0 and feat[5] == 1.0:  # ANT + WHITE
                # Was 3, now 2 → 2/3
                assert feat[20] == pytest.approx(2.0 / 3.0)
                break


# ---------------------------------------------------------------------------
# TestMultiplePieces
# ---------------------------------------------------------------------------


class TestMultiplePieces:
    """Encoding with several pieces creating a small hive."""

    def _small_hive(self):
        """Place 3 adjacent pieces in a line."""
        s = GameState()
        s = _place(s, PieceType.QUEEN, ORIGIN)
        e = ORIGIN.neighbor(ALL_DIRECTIONS[0])  # East
        s = _place(s, PieceType.QUEEN, e)
        w = ORIGIN.neighbor(ALL_DIRECTIONS[3])  # West
        s = _place(s, PieceType.ANT, w)
        return s

    def test_three_piece_nodes(self, encoder):
        s = self._small_hive()
        g = encoder.encode(s)
        assert g.num_piece_nodes == 3

    def test_edge_count(self, encoder):
        s = self._small_hive()
        g = encoder.encode(s)
        # Three pieces in a line: W--O--E
        # W↔O: 2 edges, O↔E: 2 edges = 4 total
        assert g.edge_index.shape[1] == 4

    def test_is_stacked_zero(self, encoder):
        s = self._small_hive()
        g = encoder.encode(s)
        # No stacked pieces, so is_stacked should be 0 for all edges
        assert (g.edge_features[:, 8] == 0.0).all()
