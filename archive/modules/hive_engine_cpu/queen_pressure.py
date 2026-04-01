"""
Queen-pressure curriculum learning utilities.

Generates single-move training examples that reward the engine for
immobilising the enemy queen or adding a newly-adjacent piece to an
already-immobile enemy queen.

Win conditions (checked after a move is applied):
    A. Enemy queen was mobile before the move and is immobile after.
    B. Enemy queen was already immobile, AND the moved/placed piece is
       NOW adjacent to the queen, AND it was NOT adjacent before.

These are pure functions with no trainer dependencies.
"""

from __future__ import annotations

import random

from hive_engine.board import Board
from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.hex_coord import HexCoord, _OFFSET_LIST
from hive_engine.pieces import Color, PieceType


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def queen_is_mobile(game_state: GameState, color: Color) -> bool:
    """
    Return True if the queen of *color* has at least one legal sliding move.

    Returns False when:
    - The queen is not yet on the board.
    - The queen is covered by a beetle (not on top of its stack).
    - The queen is pinned (sole piece at an articulation point).
    - The queen has no legal sliding destinations.
    """
    if not game_state._queen_placed[color]:
        return False

    board = game_state.board
    for piece, pos in board.piece_positions.items():
        if piece.color != color or piece.piece_type != PieceType.QUEEN:
            continue
        # Covered by a beetle?
        if not board.is_on_top(piece):
            return False
        # Pinned (sole piece at articulation point)?
        art_pts = board.find_articulation_points()
        if pos in art_pts and board.stack_height(pos) == 1:
            return False
        return len(board.generate_piece_moves(piece)) > 0

    return False  # queen piece not found on board (shouldn't happen)


def piece_adjacent_to_queen(
    pos: HexCoord, board: Board, enemy_color: Color
) -> bool:
    """
    Return True if *pos* is adjacent to the enemy queen's position.

    Uses tuple arithmetic against *_OFFSET_LIST* to avoid HexCoord allocation.
    """
    pq, pr = pos.q, pos.r
    for piece, q_pos in board.piece_positions.items():
        if piece.color != enemy_color or piece.piece_type != PieceType.QUEEN:
            continue
        qq, qr = q_pos.q, q_pos.r
        for dq, dr in _OFFSET_LIST:
            if pq + dq == qq and pr + dr == qr:
                return True
        return False  # found the queen, pos is not adjacent
    return False  # queen not on board


def find_queen_pressure_wins(game_state: GameState) -> list[Move]:
    """
    Return every legal move that satisfies the queen-pressure win condition.

    For each candidate move the function applies it, evaluates both win
    conditions, then undoes it — cheaper than deep-copying the state.

    Conditions (OR):
        A. Enemy queen mobile before → immobile after.
        B. Enemy queen already immobile AND the piece is NEWLY adjacent
           (was not adjacent at *from_pos* before the move).
    """
    enemy = (
        Color.BLACK if game_state.current_player == Color.WHITE else Color.WHITE
    )
    was_mobile = queen_is_mobile(game_state, enemy)

    winning: list[Move] = []
    for move in game_state.legal_moves():
        if move.move_type == MoveType.PASS:
            continue

        # Condition B pre-check: was this piece already adjacent to the
        # enemy queen before the move?
        # PLACE: piece comes from hand → never adjacent. was_adjacent = False.
        # MOVE: check whether from_pos is adjacent to the enemy queen.
        was_adjacent = (
            move.move_type == MoveType.MOVE
            and move.from_pos is not None
            and piece_adjacent_to_queen(move.from_pos, game_state.board, enemy)
        )

        game_state.apply_move(move)

        now_mobile = queen_is_mobile(game_state, enemy)
        # Condition A: queen newly immobilised
        cond_a = was_mobile and not now_mobile
        # Condition B: queen stays immobile, piece is newly adjacent
        cond_b = (
            not was_mobile
            and move.to is not None
            and not was_adjacent
            and piece_adjacent_to_queen(move.to, game_state.board, enemy)
        )

        game_state.undo_move()

        if cond_a or cond_b:
            winning.append(move)

    return winning


def generate_queen_pressure_state(
    min_moves: int = 10,
    max_moves: int = 40,
    max_attempts: int = 100,
) -> GameState | None:
    """
    Generate a random mid-game state suitable for queen-pressure curriculum.

    Plays a random rollout of *n* moves (n ∈ [min_moves, max_moves]) and
    returns the state only if:
    - The game is still in progress.
    - Both queens have been placed on the board.
    - The current player has at least one legal move.

    Returns None if no valid state is found within *max_attempts*.
    """
    for _ in range(max_attempts):
        g = GameState()
        n = random.randint(min_moves, max_moves)
        aborted = False
        for _ in range(n):
            moves = g.legal_moves()
            if not moves or g.result != GameResult.IN_PROGRESS:
                aborted = True
                break
            g.apply_move(random.choice(moves))

        if (
            not aborted
            and g.result == GameResult.IN_PROGRESS
            and g._queen_placed[Color.WHITE]
            and g._queen_placed[Color.BLACK]
            and bool(g.legal_moves())
        ):
            return g

    return None
