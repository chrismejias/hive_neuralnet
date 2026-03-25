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
            assert g.node_features[i, 22] == 1.0  # is_hand_node


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
        assert feat[8] == 1.0  # WHITE
        assert feat[9] == 0.0  # BLACK
        # is_on_ground
        assert feat[10] == 1.0
        # is_on_top
        assert feat[11] == 1.0
        # stack_height / 4
        assert feat[12] == pytest.approx(0.25)
        # is_queen
        assert feat[13] == 1.0
        # No occupied neighbors for a single piece
        assert feat[15] == 0.0
        # All 6 directions empty
        for d in range(6):
            assert feat[16 + d] == 1.0

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
            assert g.node_features[i, 15] == pytest.approx(1.0 / 6.0)

    def test_empty_dir_mask(self, encoder):
        s = self._two_piece_state()
        g = encoder.encode(s)
        # Each piece has 5 empty directions (1 occupied neighbor)
        for i in range(2):
            empty_dirs = g.node_features[i, 16:22].sum()
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
        # Full hand: 11 pieces / 14.0
        assert g.global_features[4] == pytest.approx(11.0 / 14.0)
        assert g.global_features[5] == pytest.approx(11.0 / 14.0)

    def test_hand_decreases_after_placement(self, encoder):
        s = GameState()
        s = _place(s, PieceType.QUEEN, ORIGIN)
        g = encoder.encode(s)
        assert g.global_features[4] == pytest.approx(10.0 / 14.0)  # White played 1

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
            assert feat[22] == 1.0  # is_hand_node
            assert feat[23] > 0.0  # count_remaining > 0

    def test_hand_nodes_have_no_spatial_features(self, encoder, empty_state):
        g = encoder.encode(empty_state)
        for i in range(g.node_features.shape[0]):
            feat = g.node_features[i]
            assert feat[10] == 0.0  # is_on_ground
            assert feat[11] == 0.0  # is_on_top
            assert feat[12] == 0.0  # stack_height
            assert feat[15] == 0.0  # num_occupied_neighbors
            # empty_dir_mask all zeros for hand
            assert feat[16:22].sum() == 0.0

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
            if feat[PieceType.ANT.value] == 1.0 and feat[8] == 1.0:  # ANT + WHITE
                # Was 3, now 2 → 2/3
                assert feat[23] == pytest.approx(2.0 / 3.0)
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

    def test_stack_position_zero_for_ground(self, encoder):
        """Ground-level pieces should have stack_position = 0."""
        s = self._small_hive()
        g = encoder.encode(s)
        for i in range(g.num_piece_nodes):
            assert g.node_features[i, 24] == 0.0  # all on ground


# ---------------------------------------------------------------------------
# TestStackedPieces
# ---------------------------------------------------------------------------


class TestStackedPieces:
    """Graph encoding with beetles creating stacked positions."""

    def _stacked_state(self):
        """Create a state with a beetle on top of a piece.

        Board after setup:
            ORIGIN.W: White Beetle (moved on top of WQ at ORIGIN)
            ORIGIN: White Queen (bottom) + White Beetle (top) -- STACK of 2
            ORIGIN.E: Black Queen
            ORIGIN.E.E: Black Ant
        """
        s = GameState()
        # Turn 0: White Queen at origin
        s = _place(s, PieceType.QUEEN, ORIGIN)
        # Turn 1: Black Queen east
        e = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        s = _place(s, PieceType.QUEEN, e)
        # Turn 2: White Beetle west of origin
        w = ORIGIN.neighbor(ALL_DIRECTIONS[3])
        s = _place(s, PieceType.BEETLE, w)
        # Turn 3: Black Ant east of black queen
        ee = e.neighbor(ALL_DIRECTIONS[0])
        s = _place(s, PieceType.ANT, ee)
        # Turn 4: Move white beetle from W onto origin (on top of queen)
        beetle = None
        for p in s.board.grid[w]:
            if p.piece_type == PieceType.BEETLE:
                beetle = p
                break
        move = Move(move_type=MoveType.MOVE, piece=beetle, to=ORIGIN)
        s.apply_move(move)
        return s

    def test_stacked_creates_extra_nodes(self, encoder):
        """Stacked position should produce 2 piece nodes instead of 1."""
        s = self._stacked_state()
        g = encoder.encode(s)
        # We have 3 positions: ORIGIN (stack of 2), E (1), EE (1)
        # Total piece nodes = 2 + 1 + 1 = 4
        assert g.num_piece_nodes == 4

    def test_stacked_positions_shared(self, encoder):
        """Both pieces in stack should have same (row, col) position."""
        s = self._stacked_state()
        g = encoder.encode(s)
        # Find which nodes share positions (stacked pieces)
        positions = [(g.node_positions[i, 0], g.node_positions[i, 1])
                     for i in range(g.num_piece_nodes)]
        # At least two nodes should share the same position
        from collections import Counter
        pos_counts = Counter(positions)
        stacked_positions = [pos for pos, cnt in pos_counts.items() if cnt > 1]
        assert len(stacked_positions) >= 1

    def test_vertical_edges_exist(self, encoder):
        """Stacked pieces should have vertical edges (is_stacked=1.0)."""
        s = self._stacked_state()
        g = encoder.encode(s)
        stacked_edges = g.edge_features[:, 8] == 1.0
        # Should have exactly 2 vertical edges (bidirectional)
        assert stacked_edges.sum() == 2

    def test_vertical_edge_features(self, encoder):
        """Vertical edges should have dq=0, dr=0 and no direction one-hot."""
        s = self._stacked_state()
        g = encoder.encode(s)
        for i in range(g.edge_features.shape[0]):
            if g.edge_features[i, 8] == 1.0:  # is_stacked
                assert g.edge_features[i, 0] == 0.0  # dq
                assert g.edge_features[i, 1] == 0.0  # dr
                assert g.edge_features[i, 2:8].sum() == 0.0  # no direction

    def test_stack_position_feature(self, encoder):
        """Bottom piece should have stack_position=0, top should have 0.25."""
        s = self._stacked_state()
        g = encoder.encode(s)
        # Find the stacked pair by position
        positions = [(g.node_positions[i, 0], g.node_positions[i, 1])
                     for i in range(g.num_piece_nodes)]
        from collections import Counter
        pos_counts = Counter(positions)
        for pos, cnt in pos_counts.items():
            if cnt == 2:
                # Find the two nodes at this position
                indices = [i for i in range(g.num_piece_nodes)
                           if (g.node_positions[i, 0], g.node_positions[i, 1]) == pos]
                # Bottom piece (first in iteration order)
                assert g.node_features[indices[0], 24] == pytest.approx(0.0)  # height 0
                # Top piece
                assert g.node_features[indices[1], 24] == pytest.approx(0.25)  # height 1

    def test_is_on_ground_and_top(self, encoder):
        """Bottom piece: is_on_ground=1, is_on_top=0. Top: is_on_ground=0, is_on_top=1."""
        s = self._stacked_state()
        g = encoder.encode(s)
        positions = [(g.node_positions[i, 0], g.node_positions[i, 1])
                     for i in range(g.num_piece_nodes)]
        from collections import Counter
        pos_counts = Counter(positions)
        for pos, cnt in pos_counts.items():
            if cnt == 2:
                indices = [i for i in range(g.num_piece_nodes)
                           if (g.node_positions[i, 0], g.node_positions[i, 1]) == pos]
                # Bottom piece
                assert g.node_features[indices[0], 10] == 1.0   # is_on_ground
                assert g.node_features[indices[0], 11] == 0.0   # NOT is_on_top
                # Top piece
                assert g.node_features[indices[1], 10] == 0.0   # NOT is_on_ground
                assert g.node_features[indices[1], 11] == 1.0   # is_on_top

    def test_empty_dir_mask_only_top(self, encoder):
        """Only the top piece in a stack should have empty_dir_mask set."""
        s = self._stacked_state()
        g = encoder.encode(s)
        positions = [(g.node_positions[i, 0], g.node_positions[i, 1])
                     for i in range(g.num_piece_nodes)]
        from collections import Counter
        pos_counts = Counter(positions)
        for pos, cnt in pos_counts.items():
            if cnt == 2:
                indices = [i for i in range(g.num_piece_nodes)
                           if (g.node_positions[i, 0], g.node_positions[i, 1]) == pos]
                # Bottom piece: no empty_dir_mask
                assert g.node_features[indices[0], 16:22].sum() == 0.0
                # Top piece: may have some empty directions
                # (don't assert specific count, just that bottom is zero)

    def test_piece_types_in_stack(self, encoder):
        """Bottom piece should be queen, top should be beetle."""
        s = self._stacked_state()
        g = encoder.encode(s)
        positions = [(g.node_positions[i, 0], g.node_positions[i, 1])
                     for i in range(g.num_piece_nodes)]
        from collections import Counter
        pos_counts = Counter(positions)
        for pos, cnt in pos_counts.items():
            if cnt == 2:
                indices = [i for i in range(g.num_piece_nodes)
                           if (g.node_positions[i, 0], g.node_positions[i, 1]) == pos]
                # Bottom is queen
                assert g.node_piece_types[indices[0]] == PieceType.QUEEN.value
                # Top is beetle
                assert g.node_piece_types[indices[1]] == PieceType.BEETLE.value
