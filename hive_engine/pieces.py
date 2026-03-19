"""
Piece definitions for the Hive board game.

Base game (11 pieces per player):
  - 1 Queen Bee
  - 2 Spiders
  - 2 Beetles
  - 3 Grasshoppers
  - 3 Soldier Ants

Expansion pieces (1 each per player, configurable):
  - 1 Mosquito
  - 1 Ladybug
  - 1 Pillbug
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Color(IntEnum):
    """Player color."""
    WHITE = 0
    BLACK = 1

    def other(self) -> Color:
        return Color(1 - self.value)

    def __repr__(self) -> str:
        return f"Color.{self.name}"


class PieceType(IntEnum):
    """Types of pieces in Hive (base + expansion)."""
    QUEEN = 0
    ANT = 1
    GRASSHOPPER = 2
    SPIDER = 3
    BEETLE = 4
    MOSQUITO = 5
    LADYBUG = 6
    PILLBUG = 7

    @property
    def short(self) -> str:
        """Single-character abbreviation."""
        return _SHORT_NAMES[self]

    @property
    def count_per_player(self) -> int:
        """How many of this piece each player starts with."""
        return _PIECE_COUNTS[self]

    @property
    def is_expansion(self) -> bool:
        """True if this is an expansion piece type."""
        return self in EXPANSION_PIECE_TYPES

    def __repr__(self) -> str:
        return f"PieceType.{self.name}"


_SHORT_NAMES: dict[PieceType, str] = {
    PieceType.QUEEN: "Q",
    PieceType.ANT: "A",
    PieceType.GRASSHOPPER: "G",
    PieceType.SPIDER: "S",
    PieceType.BEETLE: "B",
    PieceType.MOSQUITO: "M",
    PieceType.LADYBUG: "L",
    PieceType.PILLBUG: "P",
}

_PIECE_COUNTS: dict[PieceType, int] = {
    PieceType.QUEEN: 1,
    PieceType.ANT: 3,
    PieceType.GRASSHOPPER: 3,
    PieceType.SPIDER: 2,
    PieceType.BEETLE: 2,
    PieceType.MOSQUITO: 1,
    PieceType.LADYBUG: 1,
    PieceType.PILLBUG: 1,
}

# Piece type groupings
BASE_PIECE_TYPES = frozenset({
    PieceType.QUEEN, PieceType.ANT, PieceType.GRASSHOPPER,
    PieceType.SPIDER, PieceType.BEETLE,
})
EXPANSION_PIECE_TYPES = frozenset({
    PieceType.MOSQUITO, PieceType.LADYBUG, PieceType.PILLBUG,
})

# Total piece type counts
NUM_PIECE_TYPES_BASE = 5
NUM_PIECE_TYPES_ALL = 8

# Base game pieces per player (no expansions)
PIECES_PER_PLAYER = 11
# Maximum pieces per player (all expansions enabled)
MAX_PIECES_PER_PLAYER = 14
TOTAL_PIECES = PIECES_PER_PLAYER * 2  # 22 (base only, kept for compat)


@dataclass(frozen=True)
class ExpansionConfig:
    """Configuration for which expansion pieces are enabled."""
    mosquito: bool = False
    ladybug: bool = False
    pillbug: bool = False

    @property
    def expansion_mask(self) -> int:
        """3-bit mask for GPU: bit 0=Mosquito, 1=Ladybug, 2=Pillbug."""
        return (
            (int(self.mosquito) << 0)
            | (int(self.ladybug) << 1)
            | (int(self.pillbug) << 2)
        )

    @property
    def pieces_per_player(self) -> int:
        """Number of pieces per player with this config."""
        return PIECES_PER_PLAYER + sum([
            self.mosquito, self.ladybug, self.pillbug,
        ])

    @property
    def enabled_types(self) -> frozenset[PieceType]:
        """Set of enabled expansion piece types."""
        types: set[PieceType] = set()
        if self.mosquito:
            types.add(PieceType.MOSQUITO)
        if self.ladybug:
            types.add(PieceType.LADYBUG)
        if self.pillbug:
            types.add(PieceType.PILLBUG)
        return frozenset(types)

    @property
    def all_types(self) -> frozenset[PieceType]:
        """All piece types available with this config (base + enabled expansions)."""
        return BASE_PIECE_TYPES | self.enabled_types

    @staticmethod
    def from_string(s: str) -> ExpansionConfig:
        """Parse from a string like 'MLP', 'ML', 'P', or '' for base only."""
        s = s.upper()
        return ExpansionConfig(
            mosquito="M" in s,
            ladybug="L" in s,
            pillbug="P" in s,
        )


# Common configs
NO_EXPANSIONS = ExpansionConfig()
ALL_EXPANSIONS = ExpansionConfig(mosquito=True, ladybug=True, pillbug=True)


class Piece:
    """
    A single Hive piece.

    Attributes:
        piece_type: The type of bug.
        color: Which player owns this piece.
        piece_id: Unique index within the player's pieces of this type (0-indexed).
    """

    __slots__ = ("piece_type", "color", "piece_id", "_hash")

    def __init__(self, piece_type: PieceType, color: Color, piece_id: int = 0) -> None:
        self.piece_type = piece_type
        self.color = color
        self.piece_id = piece_id
        self._hash = hash((piece_type, color, piece_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Piece):
            return NotImplemented
        return (
            self.piece_type == other.piece_type
            and self.color == other.color
            and self.piece_id == other.piece_id
        )

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        color_char = "w" if self.color == Color.WHITE else "b"
        return f"{color_char}{self.piece_type.short}{self.piece_id + 1}"

    @property
    def label(self) -> str:
        """Human-readable label like 'wQ1' or 'bA2'."""
        color_char = "w" if self.color == Color.WHITE else "b"
        return f"{color_char}{self.piece_type.short}{self.piece_id + 1}"


def create_player_pieces(
    color: Color, expansions: ExpansionConfig | None = None,
) -> list[Piece]:
    """Create the set of pieces for one player.

    Args:
        color: Player color.
        expansions: Which expansion pieces to include. None = base only.
    """
    if expansions is None:
        expansions = NO_EXPANSIONS
    pieces = []
    for pt in PieceType:
        if pt.is_expansion and pt not in expansions.enabled_types:
            continue
        for i in range(pt.count_per_player):
            pieces.append(Piece(pt, color, i))
    return pieces
