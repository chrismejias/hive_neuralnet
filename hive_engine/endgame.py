"""
Endgame position generator for bootstrap training.

Generates random Hive board states where both queens are nearly
surrounded (4–5 of 6 neighbors filled). Games starting from these
positions resolve quickly with decisive results, providing strong
value signal for early training iterations.

Usage:
    from hive_engine.endgame import generate_endgame
    import numpy as np

    rng = np.random.RandomState(42)
    game_state = generate_endgame(rng)
    # game_state has both queens placed, 4-5 neighbors each filled
    # game is IN_PROGRESS with legal moves available
"""

from __future__ import annotations

from collections import deque

import numpy as np

from hive_engine.board import Board
from hive_engine.game_state import GameState, GameResult
from hive_engine.hex_coord import HexCoord, Direction, ALL_DIRECTIONS
from hive_engine.pieces import Color, PieceType, Piece, create_player_pieces


# Fixed queen positions — close enough to share neighbors, far enough
# that the clusters don't overlap too much.
_WHITE_QUEEN_POS = HexCoord(0, 0)
_BLACK_QUEEN_POS = HexCoord(3, 0)

# The path between the two queens (shared hex neighbors).
# HexCoord(1, 0) and HexCoord(2, 0) sit between them.
_BRIDGE_POSITIONS = [HexCoord(1, 0), HexCoord(2, 0)]

# Maximum generation attempts before giving up.
_MAX_ATTEMPTS = 50


def generate_endgame(
    rng: np.random.RandomState | None = None,
    min_surround: int = 4,
    max_surround: int = 5,
) -> GameState:
    """
    Generate a random endgame position with both queens nearly surrounded.

    Creates a valid Hive GameState where:
    - Both queens are on the board
    - Each queen has `min_surround` to `max_surround` occupied neighbors
    - Neither queen is fully surrounded (game is still in progress)
    - All pieces form a single connected hive
    - Legal moves exist for the current player

    Args:
        rng: Random state for reproducibility. If None, creates one.
        min_surround: Minimum neighbors per queen (default 4).
        max_surround: Maximum neighbors per queen (default 5).

    Returns:
        A GameState ready for self-play.

    Raises:
        RuntimeError: If unable to generate a valid position after max attempts.
    """
    if rng is None:
        rng = np.random.RandomState()

    for attempt in range(_MAX_ATTEMPTS):
        gs = _try_generate(rng, min_surround, max_surround)
        if gs is not None:
            return gs

    raise RuntimeError(
        f"Failed to generate a valid endgame position after {_MAX_ATTEMPTS} attempts"
    )


def _try_generate(
    rng: np.random.RandomState,
    min_surround: int,
    max_surround: int,
) -> GameState | None:
    """
    Single attempt to generate an endgame position. Returns None on failure.
    """
    gs = GameState()
    board = gs.board

    # ── Step 1: Place queens ──────────────────────────────────────

    wq = _find_piece(gs, Color.WHITE, PieceType.QUEEN, 0)
    bq = _find_piece(gs, Color.BLACK, PieceType.QUEEN, 0)

    _place(gs, wq, _WHITE_QUEEN_POS)
    _place(gs, bq, _BLACK_QUEEN_POS)

    # ── Step 2: Choose surround targets ───────────────────────────

    w_surround = rng.randint(min_surround, max_surround + 1)
    b_surround = rng.randint(min_surround, max_surround + 1)

    # Pick which directions to fill for each queen
    w_dirs = list(rng.choice(
        len(ALL_DIRECTIONS), size=w_surround, replace=False
    ))
    b_dirs = list(rng.choice(
        len(ALL_DIRECTIONS), size=b_surround, replace=False
    ))

    w_targets = {_WHITE_QUEEN_POS.neighbor(ALL_DIRECTIONS[d]) for d in w_dirs}
    b_targets = {_BLACK_QUEEN_POS.neighbor(ALL_DIRECTIONS[d]) for d in b_dirs}

    # Positions that need a piece (excluding queen positions themselves)
    needed_positions = (w_targets | b_targets) - {_WHITE_QUEEN_POS, _BLACK_QUEEN_POS}

    # ── Step 3: Ensure connectivity ───────────────────────────────
    # Add bridge positions between the queens if needed to connect clusters
    all_positions = needed_positions | {_WHITE_QUEEN_POS, _BLACK_QUEEN_POS}
    for bridge_pos in _BRIDGE_POSITIONS:
        if bridge_pos not in all_positions:
            # Check if adding this bridge helps connectivity
            # Always add bridges — they're cheap and ensure connectivity
            needed_positions.add(bridge_pos)
            all_positions.add(bridge_pos)

    # Remove queen positions from the needed set (already placed)
    needed_positions -= {_WHITE_QUEEN_POS, _BLACK_QUEEN_POS}

    # ── Step 4: Build piece pool ──────────────────────────────────
    # Collect available non-queen pieces from both hands, alternating colors
    white_pieces = [
        p for p in gs.hand(Color.WHITE)
        if p.piece_type != PieceType.QUEEN
    ]
    black_pieces = [
        p for p in gs.hand(Color.BLACK)
        if p.piece_type != PieceType.QUEEN
    ]

    rng.shuffle(white_pieces)
    rng.shuffle(black_pieces)

    # Interleave pieces from both colors for variety
    pool: list[Piece] = []
    wi, bi = 0, 0
    while wi < len(white_pieces) or bi < len(black_pieces):
        if wi < len(white_pieces):
            pool.append(white_pieces[wi])
            wi += 1
        if bi < len(black_pieces):
            pool.append(black_pieces[bi])
            bi += 1

    # Shuffle the pool for randomness
    pool_list = list(pool)
    rng.shuffle(pool_list)

    # ── Step 5: Place pieces at needed positions ──────────────────
    needed_list = list(needed_positions)
    rng.shuffle(needed_list)

    if len(needed_list) > len(pool_list):
        return None  # Not enough pieces

    for pos in needed_list:
        piece = pool_list.pop(0)
        _place(gs, piece, pos)

    # ── Step 6: Set turn count and verify ─────────────────────────
    # Turn = total pieces placed (each placement is one turn)
    total_placed = len(board.piece_positions)
    gs.turn = total_placed

    # Ensure it's not already game over
    gs.result = GameResult.IN_PROGRESS
    gs._check_game_over()
    if gs.result != GameResult.IN_PROGRESS:
        return None  # Queen got fully surrounded

    # Verify surround counts are in range
    w_count = gs.queen_surrounded_count(Color.WHITE)
    b_count = gs.queen_surrounded_count(Color.BLACK)
    if not (min_surround <= w_count <= max_surround):
        return None
    if not (min_surround <= b_count <= max_surround):
        return None

    # Verify connectivity (BFS from any position)
    if not _is_hive_connected(board):
        return None

    # Verify legal moves exist
    moves = gs.legal_moves()
    if not moves:
        return None

    return gs


def _find_piece(
    gs: GameState, color: Color, piece_type: PieceType, piece_id: int
) -> Piece:
    """Find a specific piece in a player's hand."""
    for p in gs.hand(color):
        if p.piece_type == piece_type and p.piece_id == piece_id:
            return p
    raise ValueError(f"Piece {color.name} {piece_type.name} #{piece_id} not in hand")


def _place(gs: GameState, piece: Piece, pos: HexCoord) -> None:
    """Place a piece directly, bypassing move validation."""
    gs.board.place_piece(piece, pos)
    gs._remove_from_hand(piece)
    if piece.piece_type == PieceType.QUEEN:
        gs._queen_placed[piece.color] = True


def _is_hive_connected(board: Board) -> bool:
    """Check that all pieces on the board form a single connected component."""
    positions = set(board.grid.keys())
    if len(positions) <= 1:
        return True

    # BFS from any starting position
    start = next(iter(positions))
    visited: set[HexCoord] = set()
    queue: deque[HexCoord] = deque([start])
    visited.add(start)

    while queue:
        pos = queue.popleft()
        for neighbor in pos.neighbors():
            if neighbor in positions and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(positions)
