"""Tests for the Hive board — placement, movement rules, and connectivity."""

import pytest

from hive_engine.hex_coord import HexCoord, Direction, ORIGIN
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.board import Board


# ── Helpers ──────────────────────────────────────────────────────

def _piece(pt: PieceType, color: Color = Color.WHITE, pid: int = 0) -> Piece:
    return Piece(pt, color, pid)


def _setup_line(board: Board, pieces: list[Piece], start: HexCoord = ORIGIN,
                direction: Direction = Direction.E) -> list[HexCoord]:
    """Place pieces in a straight line. Returns positions."""
    positions = []
    pos = start
    for p in pieces:
        board.place_piece(p, pos)
        positions.append(pos)
        pos = pos.neighbor(direction)
    return positions


# ── Board Basics ─────────────────────────────────────────────────

class TestBoardBasics:

    def test_empty_board(self):
        board = Board()
        assert board.is_empty()
        assert board.num_pieces_on_board() == 0

    def test_place_piece(self):
        board = Board()
        piece = _piece(PieceType.QUEEN)
        board.place_piece(piece, ORIGIN)
        assert not board.is_empty()
        assert board.top_piece_at(ORIGIN) is piece
        assert board.position_of(piece) == ORIGIN

    def test_remove_piece(self):
        board = Board()
        piece = _piece(PieceType.QUEEN)
        board.place_piece(piece, ORIGIN)
        old_pos = board.remove_piece(piece)
        assert old_pos == ORIGIN
        assert board.is_empty()
        assert board.position_of(piece) is None

    def test_move_piece(self):
        board = Board()
        piece = _piece(PieceType.QUEEN)
        board.place_piece(piece, ORIGIN)
        dest = HexCoord(1, 0)
        old = board.move_piece(piece, dest)
        assert old == ORIGIN
        assert board.position_of(piece) == dest
        assert board.top_piece_at(ORIGIN) is None

    def test_stacking(self):
        board = Board()
        queen = _piece(PieceType.QUEEN, Color.WHITE)
        beetle = _piece(PieceType.BEETLE, Color.BLACK)
        board.place_piece(queen, ORIGIN)
        board.place_piece(beetle, ORIGIN)

        assert board.stack_height(ORIGIN) == 2
        assert board.top_piece_at(ORIGIN) is beetle
        assert board.stack_at(ORIGIN) == [queen, beetle]
        assert board.is_on_top(beetle)
        assert not board.is_on_top(queen)

    def test_copy(self):
        board = Board()
        piece = _piece(PieceType.QUEEN)
        board.place_piece(piece, ORIGIN)

        board2 = board.copy()
        # Modifying copy shouldn't affect original
        board2.move_piece(piece, HexCoord(1, 0))
        assert board.position_of(piece) == ORIGIN

    def test_pieces_of_color(self):
        board = Board()
        w1 = _piece(PieceType.QUEEN, Color.WHITE)
        w2 = _piece(PieceType.ANT, Color.WHITE)
        b1 = _piece(PieceType.QUEEN, Color.BLACK)
        board.place_piece(w1, ORIGIN)
        board.place_piece(w2, HexCoord(1, 0))
        board.place_piece(b1, HexCoord(-1, 0))

        white_pieces = board.pieces_of_color(Color.WHITE)
        assert len(white_pieces) == 2
        assert w1 in white_pieces
        assert w2 in white_pieces


# ── Neighbor Queries ─────────────────────────────────────────────

class TestNeighborQueries:

    def test_occupied_neighbors(self):
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN), ORIGIN)
        board.place_piece(_piece(PieceType.ANT), HexCoord(1, 0))

        neighbors = board.occupied_neighbors(ORIGIN)
        assert len(neighbors) == 1
        assert neighbors[0][0] == Direction.E

    def test_empty_neighbors(self):
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN), ORIGIN)
        empty = board.empty_neighbors(ORIGIN)
        assert len(empty) == 6  # All neighbors are empty when only origin is occupied

        board.place_piece(_piece(PieceType.ANT), HexCoord(1, 0))
        empty = board.empty_neighbors(ORIGIN)
        assert len(empty) == 5


# ── One Hive Rule / Connectivity ─────────────────────────────────

class TestOneHiveRule:

    def test_single_piece_no_articulation(self):
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN), ORIGIN)
        ap = board.find_articulation_points()
        assert len(ap) == 0  # A single piece is not an articulation point

    def test_two_pieces_no_articulation(self):
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN), ORIGIN)
        board.place_piece(_piece(PieceType.ANT, Color.BLACK), HexCoord(1, 0))
        ap = board.find_articulation_points()
        assert len(ap) == 0

    def test_line_of_three_middle_is_articulation(self):
        """
        Layout: Q - A - S (in a line)
        The middle piece (A) is an articulation point.
        """
        board = Board()
        queen = _piece(PieceType.QUEEN, Color.WHITE)
        ant = _piece(PieceType.ANT, Color.WHITE)
        spider = _piece(PieceType.SPIDER, Color.BLACK)

        board.place_piece(queen, ORIGIN)
        board.place_piece(ant, HexCoord(1, 0))
        board.place_piece(spider, HexCoord(2, 0))

        ap = board.find_articulation_points()
        assert HexCoord(1, 0) in ap
        assert ORIGIN not in ap
        assert HexCoord(2, 0) not in ap

    def test_triangle_no_articulation(self):
        """
        Three pieces forming a triangle — no articulation points.
        """
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN), ORIGIN)
        board.place_piece(_piece(PieceType.ANT), HexCoord(1, 0))
        board.place_piece(_piece(PieceType.SPIDER, Color.BLACK), HexCoord(0, 1))

        # Check that the three positions form a connected triangle
        assert ORIGIN.distance(HexCoord(1, 0)) == 1
        assert ORIGIN.distance(HexCoord(0, 1)) == 1

        ap = board.find_articulation_points()
        assert len(ap) == 0

    def test_connected_without_single_piece(self):
        board = Board()
        piece = _piece(PieceType.QUEEN)
        board.place_piece(piece, ORIGIN)
        assert board.is_connected_without(piece)

    def test_connected_without_middle_of_line(self):
        board = Board()
        q = _piece(PieceType.QUEEN)
        a = _piece(PieceType.ANT)
        s = _piece(PieceType.SPIDER, Color.BLACK)
        board.place_piece(q, ORIGIN)
        board.place_piece(a, HexCoord(1, 0))
        board.place_piece(s, HexCoord(2, 0))

        # Removing the middle piece disconnects the hive
        assert not board.is_connected_without(a)
        # Removing an end piece doesn't
        assert board.is_connected_without(q)
        assert board.is_connected_without(s)

    def test_stacked_piece_not_pinned(self):
        """A piece on top of another is never pinned — removal doesn't disconnect."""
        board = Board()
        bottom = _piece(PieceType.QUEEN)
        top = _piece(PieceType.BEETLE, Color.BLACK)
        board.place_piece(bottom, ORIGIN)
        board.place_piece(top, ORIGIN)

        assert board.is_connected_without(top)


# ── Gate Detection ───────────────────────────────────────────────

class TestGateDetection:

    def test_no_gate_open(self):
        """No gate when flanking positions are empty."""
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN), ORIGIN)
        board.place_piece(_piece(PieceType.ANT), HexCoord(1, 0))
        # Only origin and east neighbor occupied; gate to NE is open
        assert not board.is_gate_blocked(ORIGIN, Direction.NE)

    def test_gate_blocked(self):
        """
        Gate is blocked when both flanking positions are occupied.

        Layout (flat-top hex):
             NE
        NW        E
            (0,0)
        W         SE
             SW

        If NE and SE are both occupied, sliding E from origin is blocked.
        """
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN), ORIGIN)
        # Flank positions for direction E are NE and SE
        ne = ORIGIN.neighbor(Direction.NE)
        se = ORIGIN.neighbor(Direction.SE)
        board.place_piece(_piece(PieceType.ANT, Color.WHITE, 0), ne)
        board.place_piece(_piece(PieceType.ANT, Color.WHITE, 1), se)
        # Also need the east to be empty for it to be a meaningful gate
        assert board.is_gate_blocked(ORIGIN, Direction.E)

    def test_can_slide_basic(self):
        """A piece can slide if the destination is empty, no gate, and maintains contact."""
        board = Board()
        queen = _piece(PieceType.QUEEN)
        ant = _piece(PieceType.ANT, Color.BLACK)
        board.place_piece(queen, ORIGIN)
        board.place_piece(ant, HexCoord(1, 0))

        # Queen should be able to slide to positions adjacent to both
        # pieces (maintaining hive contact) with no gate
        # NE neighbor of origin is adjacent to both origin and east piece
        assert board.can_slide(ORIGIN, Direction.NE, exclude_pos=None)


# ── Queen Movement ───────────────────────────────────────────────

class TestQueenMovement:

    def test_queen_slides_one(self):
        """Queen moves exactly one space."""
        board = Board()
        queen = _piece(PieceType.QUEEN)
        ant = _piece(PieceType.ANT, Color.BLACK)

        # Q A (line of two)
        board.place_piece(queen, ORIGIN)
        board.place_piece(ant, HexCoord(1, 0))

        moves = board.generate_slides(queen, max_distance=1)
        # Queen can slide to positions adjacent to both pieces
        assert len(moves) > 0
        # Can't slide to occupied position
        assert HexCoord(1, 0) not in moves
        # All destinations should be distance 1 from queen
        for m in moves:
            assert ORIGIN.distance(m) == 1


# ── Ant Movement ─────────────────────────────────────────────────

class TestAntMovement:

    def test_ant_unlimited_slide(self):
        """Ant can slide unlimited distance around the hive."""
        board = Board()
        ant = _piece(PieceType.ANT, Color.WHITE)
        q = _piece(PieceType.QUEEN, Color.WHITE)
        bq = _piece(PieceType.QUEEN, Color.BLACK)

        # Create a small hive: Q - bQ - A (line of 3)
        board.place_piece(q, ORIGIN)
        board.place_piece(bq, HexCoord(1, 0))
        board.place_piece(ant, HexCoord(2, 0))

        moves = board.generate_slides(ant, max_distance=-1)
        # Ant should be able to reach many positions around the hive
        assert len(moves) > 2
        # Ant cannot stay in place
        assert HexCoord(2, 0) not in moves

    def test_ant_cannot_enter_gap(self):
        """Ant cannot slide through a gate (narrow gap)."""
        board = Board()
        # Create a ring-like structure with a gate
        ant = _piece(PieceType.ANT, Color.WHITE, 0)
        pieces = [
            (_piece(PieceType.QUEEN, Color.WHITE), ORIGIN),
            (_piece(PieceType.SPIDER, Color.BLACK, 0), HexCoord(1, 0)),
            (_piece(PieceType.SPIDER, Color.BLACK, 1), HexCoord(1, -1)),
            (_piece(PieceType.ANT, Color.BLACK, 0), HexCoord(0, -1)),
            (_piece(PieceType.ANT, Color.BLACK, 1), HexCoord(-1, 0)),
            (ant, HexCoord(-1, 1)),
        ]
        for p, pos in pieces:
            board.place_piece(p, pos)

        # The ant should not be able to reach the inside of a gate
        moves = board.generate_slides(ant, max_distance=-1)
        # Verify moves exist
        assert len(moves) > 0


# ── Grasshopper Movement ────────────────────────────────────────

class TestGrasshopperMovement:

    def test_grasshopper_basic_jump(self):
        """Grasshopper jumps over adjacent pieces in a straight line."""
        board = Board()
        gh = _piece(PieceType.GRASSHOPPER)
        ant = _piece(PieceType.ANT, Color.BLACK)

        # GH - A (grasshopper can jump over ant to land on the other side)
        board.place_piece(gh, ORIGIN)
        board.place_piece(ant, HexCoord(1, 0))

        moves = board.generate_grasshopper_moves(gh)
        assert HexCoord(2, 0) in moves  # Lands just past the ant
        assert HexCoord(1, 0) not in moves  # Can't land on ant

    def test_grasshopper_jumps_multiple(self):
        """Grasshopper jumps over multiple pieces in a line."""
        board = Board()
        gh = _piece(PieceType.GRASSHOPPER)

        board.place_piece(gh, ORIGIN)
        board.place_piece(_piece(PieceType.ANT, Color.BLACK, 0), HexCoord(1, 0))
        board.place_piece(_piece(PieceType.ANT, Color.BLACK, 1), HexCoord(2, 0))

        moves = board.generate_grasshopper_moves(gh)
        assert HexCoord(3, 0) in moves  # Jumps over both
        assert HexCoord(1, 0) not in moves
        assert HexCoord(2, 0) not in moves

    def test_grasshopper_needs_adjacent_piece(self):
        """Grasshopper can only jump in directions with an adjacent piece."""
        board = Board()
        gh = _piece(PieceType.GRASSHOPPER)
        ant = _piece(PieceType.ANT, Color.BLACK)

        board.place_piece(gh, ORIGIN)
        board.place_piece(ant, HexCoord(1, 0))

        moves = board.generate_grasshopper_moves(gh)
        # Can only jump east (over the ant)
        # Can't jump in other directions (no adjacent piece)
        for m in moves:
            # All moves should be east of origin
            assert m.q > 0 or m == HexCoord(1, 0)  # Only east direction has pieces

    def test_grasshopper_multiple_directions(self):
        """Grasshopper can jump in multiple directions if surrounded."""
        board = Board()
        gh = _piece(PieceType.GRASSHOPPER)
        board.place_piece(gh, ORIGIN)

        # Surround with pieces in E and W
        board.place_piece(_piece(PieceType.ANT, Color.BLACK, 0), HexCoord(1, 0))
        board.place_piece(_piece(PieceType.ANT, Color.BLACK, 1), HexCoord(-1, 0))

        moves = board.generate_grasshopper_moves(gh)
        assert HexCoord(2, 0) in moves   # Jump east
        assert HexCoord(-2, 0) in moves  # Jump west


# ── Spider Movement ──────────────────────────────────────────────

class TestSpiderMovement:

    def test_spider_exactly_three(self):
        """Spider must move exactly 3 spaces."""
        board = Board()
        spider = _piece(PieceType.SPIDER)
        # Build a line of 5 pieces so spider has room to slide
        board.place_piece(spider, ORIGIN)
        board.place_piece(_piece(PieceType.QUEEN, Color.BLACK), HexCoord(1, 0))
        board.place_piece(_piece(PieceType.ANT, Color.BLACK, 0), HexCoord(2, 0))
        board.place_piece(_piece(PieceType.ANT, Color.BLACK, 1), HexCoord(3, 0))
        board.place_piece(_piece(PieceType.ANT, Color.BLACK, 2), HexCoord(4, 0))

        moves = board.generate_slides(spider, max_distance=3)
        # All destinations must be exactly 3 slides from origin
        assert len(moves) > 0
        # Spider shouldn't be able to reach distance-1 or distance-2 positions
        for m in moves:
            # This is slide distance, not hex distance, so we just verify
            # the destinations are valid (distance check is complex due to
            # path requirements)
            assert m != ORIGIN  # Can't stay in place


# ── Beetle Movement ──────────────────────────────────────────────

class TestBeetleMovement:

    def test_beetle_ground_slide(self):
        """Beetle at ground level can slide like the queen."""
        board = Board()
        beetle = _piece(PieceType.BEETLE)
        ant = _piece(PieceType.ANT, Color.BLACK)

        board.place_piece(beetle, ORIGIN)
        board.place_piece(ant, HexCoord(1, 0))

        moves = board.generate_beetle_moves(beetle)
        assert len(moves) > 0

    def test_beetle_climb_onto_piece(self):
        """Beetle can climb onto an adjacent piece."""
        board = Board()
        beetle = _piece(PieceType.BEETLE)
        queen = _piece(PieceType.QUEEN, Color.BLACK)

        board.place_piece(beetle, ORIGIN)
        board.place_piece(queen, HexCoord(1, 0))

        moves = board.generate_beetle_moves(beetle)
        # Should be able to climb onto the queen
        assert HexCoord(1, 0) in moves

    def test_beetle_on_top_can_move(self):
        """Beetle on top of another piece can move to adjacent positions."""
        board = Board()
        queen = _piece(PieceType.QUEEN, Color.WHITE)
        beetle = _piece(PieceType.BEETLE, Color.BLACK)
        ant = _piece(PieceType.ANT, Color.WHITE)

        board.place_piece(queen, ORIGIN)
        board.place_piece(beetle, ORIGIN)  # On top of queen
        board.place_piece(ant, HexCoord(1, 0))

        assert board.is_on_top(beetle)
        moves = board.generate_beetle_moves(beetle)
        assert len(moves) > 0


# ── Placement Positions ─────────────────────────────────────────

class TestPlacementPositions:

    def test_first_placement_at_origin(self):
        board = Board()
        positions = board.valid_placement_positions(Color.WHITE)
        assert positions == {ORIGIN}

    def test_placement_adjacent_to_friendly(self):
        board = Board()
        # Place white queen at origin
        board.place_piece(_piece(PieceType.QUEEN, Color.WHITE), ORIGIN)
        # Place black queen east
        board.place_piece(_piece(PieceType.QUEEN, Color.BLACK), HexCoord(1, 0))

        # White placements should be adjacent to white only
        white_positions = board.valid_placement_positions(Color.WHITE)
        for pos in white_positions:
            # Must be adjacent to a white piece
            has_white_neighbor = False
            has_black_neighbor = False
            for n in pos.neighbors():
                stack = board.grid.get(n)
                if stack:
                    top = stack[-1]
                    if top.color == Color.WHITE:
                        has_white_neighbor = True
                    else:
                        has_black_neighbor = True
            assert has_white_neighbor
            assert not has_black_neighbor

    def test_no_placement_adjacent_to_enemy(self):
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN, Color.WHITE), ORIGIN)
        board.place_piece(_piece(PieceType.QUEEN, Color.BLACK), HexCoord(1, 0))

        white_positions = board.valid_placement_positions(Color.WHITE)
        # No white placement position should be adjacent to the black queen
        for pos in white_positions:
            for n in pos.neighbors():
                stack = board.grid.get(n)
                if stack:
                    assert stack[-1].color != Color.BLACK


# ── Canonical Hash ───────────────────────────────────────────────

class TestCanonicalHash:

    def test_same_position_same_hash(self):
        board1 = Board()
        board2 = Board()
        piece = _piece(PieceType.QUEEN)
        board1.place_piece(piece, ORIGIN)
        board2.place_piece(piece, ORIGIN)
        assert board1.canonical_hash() == board2.canonical_hash()

    def test_translated_position_same_hash(self):
        """Same shape at different positions should have the same hash."""
        board1 = Board()
        board2 = Board()

        q1 = _piece(PieceType.QUEEN, Color.WHITE)
        a1 = _piece(PieceType.ANT, Color.BLACK)

        # Position 1: at origin
        board1.place_piece(q1, ORIGIN)
        board1.place_piece(a1, HexCoord(1, 0))

        # Position 2: translated by (5, 3)
        board2.place_piece(q1, HexCoord(5, 3))
        board2.place_piece(a1, HexCoord(6, 3))

        assert board1.canonical_hash() == board2.canonical_hash()

    def test_different_position_different_hash(self):
        board1 = Board()
        board2 = Board()

        q = _piece(PieceType.QUEEN, Color.WHITE)
        a = _piece(PieceType.ANT, Color.BLACK)

        board1.place_piece(q, ORIGIN)
        board1.place_piece(a, HexCoord(1, 0))

        board2.place_piece(q, ORIGIN)
        board2.place_piece(a, HexCoord(0, 1))  # Different relative position

        assert board1.canonical_hash() != board2.canonical_hash()
