"""
Hexagonal coordinate system using cube coordinates.

Uses the cube coordinate system (q, r, s) where q + r + s = 0.
This enables clean distance calculation, rotation, and neighbor finding.
See: https://www.redblobgames.com/grids/hexagons/

We use flat-top orientation with the six directions:
    E, NE, NW, W, SW, SE
"""

from __future__ import annotations

from enum import IntEnum
from functools import cached_property
from typing import Iterator


class Direction(IntEnum):
    """Six hex directions in clockwise order (flat-top orientation)."""
    E = 0
    NE = 1
    NW = 2
    W = 3
    SW = 4
    SE = 5

    def opposite(self) -> Direction:
        """Return the opposite direction."""
        return Direction((self.value + 3) % 6)

    def clockwise(self) -> Direction:
        """Return the next direction clockwise."""
        return Direction((self.value + 5) % 6)

    def counter_clockwise(self) -> Direction:
        """Return the next direction counter-clockwise."""
        return Direction((self.value + 1) % 6)


# Direction offset vectors in cube coordinates (q, r, s)
# Flat-top hex orientation
DIRECTION_OFFSETS: dict[Direction, tuple[int, int, int]] = {
    Direction.E:  (+1, 0, -1),
    Direction.NE: (+1, -1, 0),
    Direction.NW: (0, -1, +1),
    Direction.W:  (-1, 0, +1),
    Direction.SW: (-1, +1, 0),
    Direction.SE: (0, +1, -1),
}

ALL_DIRECTIONS: list[Direction] = list(Direction)

# Pre-computed offset tuples for fast neighbor iteration (q_offset, r_offset)
_OFFSET_LIST: list[tuple[int, int]] = [
    (+1, 0),   # E
    (+1, -1),  # NE
    (0, -1),   # NW
    (-1, 0),   # W
    (-1, +1),  # SW
    (0, +1),   # SE
]


class HexCoord:
    """
    Immutable hexagonal coordinate using cube coordinates.

    The cube coordinate system uses three axes (q, r, s) with the
    constraint q + r + s = 0. This makes distance, neighbor, and
    rotation calculations elegant and efficient.

    Attributes:
        q: Column axis
        r: Row axis (diagonal)
        s: Derived axis (s = -q - r)
    """

    __slots__ = ("q", "r", "_hash")

    def __init__(self, q: int, r: int) -> None:
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "_hash", hash((q, r)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("HexCoord is immutable")

    @property
    def s(self) -> int:
        """Third cube coordinate, derived from q + r + s = 0."""
        return -self.q - self.r

    # ── Arithmetic ──────────────────────────────────────────────

    def __add__(self, other: HexCoord) -> HexCoord:
        return HexCoord(self.q + other.q, self.r + other.r)

    def __sub__(self, other: HexCoord) -> HexCoord:
        return HexCoord(self.q - other.q, self.r - other.r)

    def __neg__(self) -> HexCoord:
        return HexCoord(-self.q, -self.r)

    # ── Comparison / Hashing ────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HexCoord):
            return NotImplemented
        return self.q == other.q and self.r == other.r

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"HexCoord({self.q}, {self.r})"

    def __lt__(self, other: HexCoord) -> bool:
        """Ordering for canonical sorting."""
        return (self.q, self.r) < (other.q, other.r)

    # ── Geometry ────────────────────────────────────────────────

    def neighbor(self, direction: Direction) -> HexCoord:
        """Return the adjacent hex in the given direction."""
        dq, dr, _ = DIRECTION_OFFSETS[direction]
        return HexCoord(self.q + dq, self.r + dr)

    def neighbors(self) -> list[HexCoord]:
        """Return all six adjacent hexes."""
        q, r = self.q, self.r
        return [
            HexCoord(q + dq, r + dr) for dq, dr in _OFFSET_LIST
        ]

    def distance(self, other: HexCoord) -> int:
        """Hex distance (minimum number of steps between two hexes)."""
        dq = abs(self.q - other.q)
        dr = abs(self.r - other.r)
        ds = abs(self.s - other.s)
        return max(dq, dr, ds)

    def direction_to(self, other: HexCoord) -> Direction | None:
        """
        Return the direction from self to other if they are adjacent.
        Returns None if not adjacent.
        """
        dq = other.q - self.q
        dr = other.r - self.r
        for d, (oq, odr, _) in DIRECTION_OFFSETS.items():
            if dq == oq and dr == odr:
                return d
        return None

    def rotate_cw_around(self, center: HexCoord) -> HexCoord:
        """Rotate 60 degrees clockwise around a center hex."""
        # Vector from center
        vq = self.q - center.q
        vr = self.r - center.r
        vs = -vq - vr
        # CW rotation: (q, r, s) -> (-s, -q, -r)
        nq, nr = -vs, -vq
        return HexCoord(nq + center.q, nr + center.r)

    def rotate_ccw_around(self, center: HexCoord) -> HexCoord:
        """Rotate 60 degrees counter-clockwise around a center hex."""
        vq = self.q - center.q
        vr = self.r - center.r
        vs = -vq - vr
        # CCW rotation: (q, r, s) -> (-r, -s, -q)
        nq, nr = -vr, -vs
        return HexCoord(nq + center.q, nr + center.r)

    def ring(self, radius: int) -> list[HexCoord]:
        """Return all hexes exactly `radius` steps away, in order."""
        if radius == 0:
            return [self]
        results = []
        # Start at the hex `radius` steps in the SW direction
        current = self
        for _ in range(radius):
            current = current.neighbor(Direction.SW)
        # Walk around the ring
        for d in ALL_DIRECTIONS:
            for _ in range(radius):
                results.append(current)
                current = current.neighbor(d)
        return results


# Pre-built origin constant
ORIGIN = HexCoord(0, 0)
