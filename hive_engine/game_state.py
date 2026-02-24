"""
Hive game state management.

Tracks the full state of a Hive game including:
  - Board position
  - Pieces in each player's hand
  - Turn count and current player
  - Legal move generation
  - Win/draw detection
  - State serialization for MCTS
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Iterator

from hive_engine.hex_coord import HexCoord, ORIGIN
from hive_engine.pieces import (
    Color,
    PieceType,
    Piece,
    create_player_pieces,
    PIECES_PER_PLAYER,
)
from hive_engine.board import Board


class MoveType(IntEnum):
    """Types of moves in Hive."""
    PLACE = 0       # Place a piece from hand onto the board
    MOVE = 1        # Move a piece already on the board
    PASS = 2        # Pass (only when no legal moves exist)


@dataclass(frozen=True)
class Move:
    """
    A single move in Hive.

    Attributes:
        move_type: Whether this is a placement, movement, or pass.
        piece: The piece being placed or moved (None for PASS).
        to: Destination hex coordinate (None for PASS).
        from_pos: Source position for MOVE type (None for PLACE/PASS).
    """
    move_type: MoveType
    piece: Piece | None = None
    to: HexCoord | None = None
    from_pos: HexCoord | None = None

    def __repr__(self) -> str:
        if self.move_type == MoveType.PASS:
            return "Move(PASS)"
        elif self.move_type == MoveType.PLACE:
            return f"Move(PLACE {self.piece} -> ({self.to.q}, {self.to.r}))"
        else:
            return (
                f"Move(MOVE {self.piece} "
                f"({self.from_pos.q}, {self.from_pos.r}) -> "
                f"({self.to.q}, {self.to.r}))"
            )


class GameResult(IntEnum):
    """Game outcome."""
    IN_PROGRESS = 0
    WHITE_WINS = 1
    BLACK_WINS = 2
    DRAW = 3


class GameState:
    """
    Complete state of a Hive game.

    This is the main interface for playing the game. It tracks the board,
    player hands, turn count, and provides legal move generation.

    Usage:
        state = GameState()
        while state.result == GameResult.IN_PROGRESS:
            moves = state.legal_moves()
            state.apply_move(moves[0])  # or pick intelligently
    """

    def __init__(self) -> None:
        self.board = Board()
        self.turn: int = 0  # Incremented after each move (0 = White's first)
        self.result: GameResult = GameResult.IN_PROGRESS

        # Pieces in each player's hand (not yet placed on the board)
        self._hands: dict[Color, list[Piece]] = {
            Color.WHITE: create_player_pieces(Color.WHITE),
            Color.BLACK: create_player_pieces(Color.BLACK),
        }

        # Track whether each player's queen is on the board
        self._queen_placed: dict[Color, bool] = {
            Color.WHITE: False,
            Color.BLACK: False,
        }

        # Move history for undo support
        self._history: list[tuple[Move, dict]] = []

        # Legal moves cache (invalidated on apply_move / undo_move)
        self._legal_moves_cache: list[Move] | None = None

    @property
    def current_player(self) -> Color:
        """The player whose turn it is."""
        return Color(self.turn % 2)

    @property
    def player_turn_number(self) -> int:
        """
        Which turn this is for the current player (0-indexed).
        Turn 0 = first placement, Turn 3 = fourth action (queen must be down).
        """
        return self.turn // 2

    def copy(self) -> GameState:
        """Create a deep copy for MCTS simulation."""
        gs = GameState.__new__(GameState)
        gs.board = self.board.copy()
        gs.turn = self.turn
        gs.result = self.result
        gs._hands = {c: list(h) for c, h in self._hands.items()}
        gs._queen_placed = dict(self._queen_placed)
        gs._history = []  # Don't copy history for rollouts
        gs._legal_moves_cache = None  # Don't copy cache — copies diverge
        return gs

    # ── Hand Management ─────────────────────────────────────────

    def hand(self, color: Color) -> list[Piece]:
        """Return the pieces still in a player's hand."""
        return self._hands[color]

    def hand_piece_types(self, color: Color) -> set[PieceType]:
        """Return the distinct piece types available in a player's hand."""
        return {p.piece_type for p in self._hands[color]}

    def pieces_in_hand(self, color: Color, piece_type: PieceType) -> list[Piece]:
        """Return all pieces of a given type in a player's hand."""
        return [p for p in self._hands[color] if p.piece_type == piece_type]

    def has_piece_in_hand(self, color: Color, piece_type: PieceType) -> bool:
        """Check if a player has a piece of this type in hand."""
        return any(p.piece_type == piece_type for p in self._hands[color])

    def _remove_from_hand(self, piece: Piece) -> None:
        """Remove a specific piece from its owner's hand."""
        self._hands[piece.color].remove(piece)

    def _add_to_hand(self, piece: Piece) -> None:
        """Add a piece back to its owner's hand."""
        self._hands[piece.color].append(piece)

    # ── Legal Move Generation ───────────────────────────────────

    def legal_moves(self) -> list[Move]:
        """
        Generate all legal moves for the current player.

        This is the core method for AI integration. Returns a list of all
        valid moves the current player can make. Results are cached until
        the state changes (apply_move / undo_move invalidates the cache).

        Rules enforced:
        1. First move: place any non-Queen piece at origin
        2. Second move (opponent): place any non-Queen adjacent to first piece
        3. Queen must be placed by player's 4th turn (turn 3, 0-indexed)
        4. Placement: adjacent to friendly, not adjacent to enemy (after turn 1)
        5. Movement: piece must be on top, not pinned, type-specific rules
        6. If no moves available: must pass
        """
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache

        if self.result != GameResult.IN_PROGRESS:
            self._legal_moves_cache = []
            return self._legal_moves_cache

        color = self.current_player
        moves: list[Move] = []

        # Generate placement moves
        placement_moves = self._generate_placements(color)
        moves.extend(placement_moves)

        # Generate movement moves (only if queen is placed)
        if self._queen_placed[color]:
            movement_moves = self._generate_movements(color)
            moves.extend(movement_moves)

        # If no moves available, must pass
        if not moves:
            moves.append(Move(MoveType.PASS))

        self._legal_moves_cache = moves
        return moves

    def _generate_placements(self, color: Color) -> list[Move]:
        """Generate all legal piece placements for the current player."""
        hand = self._hands[color]
        if not hand:
            return []

        player_turn = self.player_turn_number

        # Queen must be placed by turn 3 (4th action)
        must_place_queen = (
            player_turn == 3
            and not self._queen_placed[color]
        )

        # Determine which piece types can be placed
        if must_place_queen:
            placeable_types = {PieceType.QUEEN}
        else:
            placeable_types = {p.piece_type for p in hand}

        # Get valid positions
        positions = self._get_placement_positions(color)
        if not positions:
            return []

        moves = []
        seen_types: set[PieceType] = set()
        for piece in hand:
            if piece.piece_type not in placeable_types:
                continue
            # Only generate one move per piece type (they're fungible before placement)
            if piece.piece_type in seen_types:
                continue
            seen_types.add(piece.piece_type)

            for pos in positions:
                moves.append(Move(MoveType.PLACE, piece, pos))

        return moves

    def _get_placement_positions(self, color: Color) -> set[HexCoord]:
        """Get valid placement positions considering the turn."""
        if self.turn == 0:
            # First move: place at origin
            return {ORIGIN}

        if self.turn == 1:
            # Second move: adjacent to the first piece (opponent adjacency OK)
            return set(ORIGIN.neighbors()) & self._all_empty_neighbors()

        # Normal placement: adjacent to friendly, not adjacent to enemy
        return self.board.valid_placement_positions(color)

    def _all_empty_neighbors(self) -> set[HexCoord]:
        """Return all empty positions adjacent to any occupied position."""
        result: set[HexCoord] = set()
        for pos in self.board.occupied_positions():
            for n in pos.neighbors():
                if n not in self.board.grid:
                    result.add(n)
        return result

    def _generate_movements(self, color: Color) -> list[Move]:
        """Generate all legal piece movements for the current player."""
        moves: list[Move] = []

        # Find articulation points (pinned pieces)
        articulation_points = self.board.find_articulation_points()

        for piece in self.board.pieces_of_color(color):
            pos = self.board.position_of(piece)
            if pos is None:
                continue

            # Piece must be on top of its stack to move
            if not self.board.is_on_top(piece):
                continue

            # Check if piece is pinned (One Hive rule)
            # A piece is pinned if it's at an articulation point AND alone in its stack
            if pos in articulation_points and self.board.stack_height(pos) == 1:
                continue

            # Generate type-specific destinations
            destinations = self.board.generate_piece_moves(piece)
            for dest in destinations:
                moves.append(Move(MoveType.MOVE, piece, dest, from_pos=pos))

        return moves

    # ── Apply / Undo ────────────────────────────────────────────

    def apply_move(self, move: Move) -> None:
        """
        Apply a move to the game state.

        This modifies the state in place. Use copy() first if you need
        to preserve the original state.
        """
        undo_info: dict = {"turn": self.turn, "result": self.result}

        if move.move_type == MoveType.PLACE:
            assert move.piece is not None and move.to is not None
            self._remove_from_hand(move.piece)
            self.board.place_piece(move.piece, move.to)
            if move.piece.piece_type == PieceType.QUEEN:
                self._queen_placed[move.piece.color] = True
            undo_info["placed"] = True

        elif move.move_type == MoveType.MOVE:
            assert move.piece is not None and move.to is not None
            self.board.move_piece(move.piece, move.to)
            undo_info["placed"] = False
            undo_info["from_pos"] = move.from_pos

        # MoveType.PASS: do nothing to the board

        self._history.append((move, undo_info))
        self.turn += 1
        self._legal_moves_cache = None  # Invalidate cache
        self._check_game_over()

    def undo_move(self) -> Move:
        """Undo the last move. Returns the move that was undone."""
        self._legal_moves_cache = None  # Invalidate cache
        move, undo_info = self._history.pop()
        self.turn = undo_info["turn"]
        self.result = undo_info["result"]

        if move.move_type == MoveType.PLACE:
            assert move.piece is not None and move.to is not None
            self.board.remove_piece(move.piece)
            self._add_to_hand(move.piece)
            if move.piece.piece_type == PieceType.QUEEN:
                self._queen_placed[move.piece.color] = False

        elif move.move_type == MoveType.MOVE:
            assert move.piece is not None and move.to is not None
            self.board.move_piece(move.piece, undo_info["from_pos"])

        return move

    # ── Win Detection ───────────────────────────────────────────

    def _check_game_over(self) -> None:
        """
        Check if the game is over.

        A player loses when their Queen Bee is completely surrounded
        (all 6 neighbors occupied). If both queens are surrounded
        simultaneously, the game is a draw.
        """
        white_queen_surrounded = self._is_queen_surrounded(Color.WHITE)
        black_queen_surrounded = self._is_queen_surrounded(Color.BLACK)

        if white_queen_surrounded and black_queen_surrounded:
            self.result = GameResult.DRAW
        elif white_queen_surrounded:
            self.result = GameResult.BLACK_WINS
        elif black_queen_surrounded:
            self.result = GameResult.WHITE_WINS

    def _is_queen_surrounded(self, color: Color) -> bool:
        """Check if a player's queen is completely surrounded."""
        if not self._queen_placed[color]:
            return False

        # Find the queen
        for piece, pos in self.board.piece_positions.items():
            if piece.color == color and piece.piece_type == PieceType.QUEEN:
                return self.board.num_occupied_neighbors(pos) == 6
        return False

    def queen_surrounded_count(self, color: Color) -> int:
        """Count how many of the 6 neighbors around a queen are occupied."""
        if not self._queen_placed[color]:
            return 0
        for piece, pos in self.board.piece_positions.items():
            if piece.color == color and piece.piece_type == PieceType.QUEEN:
                return self.board.num_occupied_neighbors(pos)
        return 0

    # ── Serialization ───────────────────────────────────────────

    def canonical_hash(self) -> int:
        """
        Translation-invariant hash of the complete game state.
        Includes board position, hands, turn, and queen placement status.
        """
        board_hash = self.board.canonical_hash()
        hand_state = tuple(
            (c.value, tuple(sorted((p.piece_type.value, p.piece_id) for p in h)))
            for c, h in sorted(self._hands.items())
        )
        return hash((board_hash, hand_state, self.turn % 2))

    def to_dict(self) -> dict:
        """Serialize the full game state to a dictionary."""
        return {
            "board": self.board.to_dict(),
            "turn": self.turn,
            "result": self.result.value,
            "white_hand": [(p.piece_type.value, p.piece_id) for p in self._hands[Color.WHITE]],
            "black_hand": [(p.piece_type.value, p.piece_id) for p in self._hands[Color.BLACK]],
            "white_queen_placed": self._queen_placed[Color.WHITE],
            "black_queen_placed": self._queen_placed[Color.BLACK],
        }

    # ── Display ─────────────────────────────────────────────────

    def display(self) -> str:
        """Human-readable representation of the game state."""
        lines = [
            f"Turn {self.turn} ({self.current_player.name}'s move, "
            f"player turn #{self.player_turn_number})",
            f"Result: {self.result.name}",
            f"White hand: {[repr(p) for p in self._hands[Color.WHITE]]}",
            f"Black hand: {[repr(p) for p in self._hands[Color.BLACK]]}",
            f"Board:",
            repr(self.board),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GameState(turn={self.turn}, player={self.current_player.name}, "
            f"result={self.result.name}, pieces_on_board={self.board.num_pieces_on_board()})"
        )
