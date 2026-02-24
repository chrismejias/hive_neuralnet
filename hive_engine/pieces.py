"""
Piece definitions for the Hive board game.

Each player has 11 base pieces:
  - 1 Queen Bee
  - 2 Spiders
  - 2 Beetles
  - 3 Grasshoppers
  - 3 Soldier Ants
"""

from __future__ import annotations

from enum import IntEnum, auto


class Color(IntEnum):
    """Player color."""
    WHITE = 0
    BLACK = 1

    def other(self) -> Color:
        return Color(1 - self.value)

    def __repr__(self) -> str:
        return f"Color.{self.name}"


class PieceType(IntEnum):
    """Types of pieces in base Hive."""
    QUEEN = 0
    ANT = 1
    GRASSHOPPER = 2
    SPIDER = 3
    BEETLE = 4

    @property
    def short(self) -> str:
        """Single-character abbreviation."""
        return _SHORT_NAMES[self]

    @property
    def count_per_player(self) -> int:
        """How many of this piece each player starts with."""
        return _PIECE_COUNTS[self]

    def __repr__(self) -> str:
        return f"PieceType.{self.name}"


_SHORT_NAMES: dict[PieceType, str] = {
    PieceType.QUEEN: "Q",
    PieceType.ANT: "A",
    PieceType.GRASSHOPPER: "G",
    PieceType.SPIDER: "S",
    PieceType.BEETLE: "B",
}

_PIECE_COUNTS: dict[PieceType, int] = {
    PieceType.QUEEN: 1,
    PieceType.ANT: 3,
    PieceType.GRASSHOPPER: 3,
    PieceType.SPIDER: 2,
    PieceType.BEETLE: 2,
}

# Total pieces per player
PIECES_PER_PLAYER = sum(_PIECE_COUNTS.values())  # 11
TOTAL_PIECES = PIECES_PER_PLAYER * 2  # 22


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


def create_player_pieces(color: Color) -> list[Piece]:
    """Create the full set of pieces for one player."""
    pieces = []
    for pt in PieceType:
        for i in range(pt.count_per_player):
            pieces.append(Piece(pt, color, i))
    return pieces
