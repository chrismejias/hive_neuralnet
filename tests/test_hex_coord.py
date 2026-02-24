"""Tests for hex coordinate system."""

import pytest

from hive_engine.hex_coord import HexCoord, Direction, ORIGIN, ALL_DIRECTIONS


class TestHexCoord:
    """Tests for HexCoord basic operations."""

    def test_creation(self):
        h = HexCoord(1, 2)
        assert h.q == 1
        assert h.r == 2
        assert h.s == -3

    def test_cube_constraint(self):
        """q + r + s must always equal 0."""
        for q in range(-5, 6):
            for r in range(-5, 6):
                h = HexCoord(q, r)
                assert h.q + h.r + h.s == 0

    def test_equality(self):
        assert HexCoord(1, 2) == HexCoord(1, 2)
        assert HexCoord(1, 2) != HexCoord(1, 3)
        assert HexCoord(0, 0) == ORIGIN

    def test_hashing(self):
        """Equal coords should have the same hash."""
        a = HexCoord(3, -1)
        b = HexCoord(3, -1)
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    def test_immutability(self):
        h = HexCoord(1, 2)
        with pytest.raises(AttributeError):
            h.q = 5

    def test_addition(self):
        a = HexCoord(1, 2)
        b = HexCoord(3, -1)
        c = a + b
        assert c == HexCoord(4, 1)

    def test_subtraction(self):
        a = HexCoord(3, 1)
        b = HexCoord(1, 2)
        c = a - b
        assert c == HexCoord(2, -1)

    def test_negation(self):
        a = HexCoord(3, -2)
        assert -a == HexCoord(-3, 2)

    def test_ordering(self):
        coords = [HexCoord(2, 1), HexCoord(1, 2), HexCoord(1, 1)]
        sorted_coords = sorted(coords)
        assert sorted_coords == [HexCoord(1, 1), HexCoord(1, 2), HexCoord(2, 1)]


class TestDirection:
    """Tests for direction operations."""

    def test_opposite(self):
        assert Direction.E.opposite() == Direction.W
        assert Direction.NE.opposite() == Direction.SW
        assert Direction.NW.opposite() == Direction.SE

    def test_clockwise(self):
        # E -> SE -> SW -> W -> NW -> NE -> E
        assert Direction.E.clockwise() == Direction.SE
        assert Direction.SE.clockwise() == Direction.SW
        assert Direction.SW.clockwise() == Direction.W
        assert Direction.W.clockwise() == Direction.NW
        assert Direction.NW.clockwise() == Direction.NE
        assert Direction.NE.clockwise() == Direction.E

    def test_counter_clockwise(self):
        assert Direction.E.counter_clockwise() == Direction.NE
        assert Direction.NE.counter_clockwise() == Direction.NW

    def test_full_rotation(self):
        """Six clockwise steps should return to start."""
        d = Direction.E
        for _ in range(6):
            d = d.clockwise()
        assert d == Direction.E


class TestNeighbors:
    """Tests for neighbor calculations."""

    def test_origin_has_six_neighbors(self):
        neighbors = ORIGIN.neighbors()
        assert len(neighbors) == 6

    def test_neighbors_are_distance_1(self):
        for n in ORIGIN.neighbors():
            assert ORIGIN.distance(n) == 1

    def test_neighbors_are_unique(self):
        neighbors = ORIGIN.neighbors()
        assert len(set(neighbors)) == 6

    def test_neighbor_and_back(self):
        """Going in a direction and then the opposite should return to start."""
        for d in ALL_DIRECTIONS:
            n = ORIGIN.neighbor(d)
            back = n.neighbor(d.opposite())
            assert back == ORIGIN

    def test_specific_neighbors(self):
        """Verify specific neighbor offsets for flat-top orientation."""
        h = HexCoord(0, 0)
        assert h.neighbor(Direction.E) == HexCoord(1, 0)
        assert h.neighbor(Direction.W) == HexCoord(-1, 0)
        assert h.neighbor(Direction.NE) == HexCoord(1, -1)
        assert h.neighbor(Direction.NW) == HexCoord(0, -1)
        assert h.neighbor(Direction.SE) == HexCoord(0, 1)
        assert h.neighbor(Direction.SW) == HexCoord(-1, 1)


class TestDistance:
    """Tests for hex distance calculations."""

    def test_distance_to_self(self):
        assert ORIGIN.distance(ORIGIN) == 0

    def test_distance_symmetric(self):
        a = HexCoord(2, -1)
        b = HexCoord(-1, 3)
        assert a.distance(b) == b.distance(a)

    def test_distance_adjacent(self):
        for d in ALL_DIRECTIONS:
            n = ORIGIN.neighbor(d)
            assert ORIGIN.distance(n) == 1

    def test_distance_two_steps(self):
        # Two steps east
        h = HexCoord(2, 0)
        assert ORIGIN.distance(h) == 2


class TestRotation:
    """Tests for hex rotation."""

    def test_rotate_cw_six_times(self):
        """Six 60-degree CW rotations = identity."""
        center = ORIGIN
        point = HexCoord(1, 0)
        current = point
        for _ in range(6):
            current = current.rotate_cw_around(center)
        assert current == point

    def test_rotate_ccw_six_times(self):
        """Six 60-degree CCW rotations = identity."""
        center = ORIGIN
        point = HexCoord(1, 0)
        current = point
        for _ in range(6):
            current = current.rotate_ccw_around(center)
        assert current == point

    def test_cw_then_ccw(self):
        """CW then CCW should return to start."""
        center = ORIGIN
        point = HexCoord(2, -1)
        rotated = point.rotate_cw_around(center)
        back = rotated.rotate_ccw_around(center)
        assert back == point

    def test_rotate_cw_visits_all_neighbors(self):
        """Rotating a neighbor CW around origin visits all 6 neighbors."""
        center = ORIGIN
        point = HexCoord(1, 0)
        visited = set()
        current = point
        for _ in range(6):
            visited.add(current)
            current = current.rotate_cw_around(center)
        assert visited == set(center.neighbors())


class TestDirectionTo:
    """Tests for direction_to."""

    def test_adjacent_direction(self):
        for d in ALL_DIRECTIONS:
            n = ORIGIN.neighbor(d)
            assert ORIGIN.direction_to(n) == d

    def test_non_adjacent_returns_none(self):
        far = HexCoord(3, 0)
        assert ORIGIN.direction_to(far) is None


class TestRing:
    """Tests for ring generation."""

    def test_ring_0(self):
        assert ORIGIN.ring(0) == [ORIGIN]

    def test_ring_1(self):
        ring = ORIGIN.ring(1)
        assert len(ring) == 6
        assert all(ORIGIN.distance(h) == 1 for h in ring)

    def test_ring_2(self):
        ring = ORIGIN.ring(2)
        assert len(ring) == 12
        assert all(ORIGIN.distance(h) == 2 for h in ring)
