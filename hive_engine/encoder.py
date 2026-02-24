"""
State encoder for neural network input.

Converts Hive game states into fixed-size tensors suitable for
convolutional neural network processing, and maps between Move
objects and flat action indices for the policy head.

Tensor layout: (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
  Channels 0-4:   White pieces at ground level (Q, A, G, S, B)
  Channels 5-9:   Black pieces at ground level (Q, A, G, S, B)
  Channels 10-14: White pieces at stacked level (top piece when stack >= 2)
  Channels 15-19: Black pieces at stacked level
  Channels 20-25: Meta information (current player, turn, queens, hands)

Action space: flat index over placements, movements, and pass.
  [0, 845):        Placement actions (5 piece types × 169 grid positions)
  [845, 29406):    Movement actions (169 source × 169 destination)
  [29406]:         Pass action
"""

from __future__ import annotations

import numpy as np

from hive_engine.hex_coord import HexCoord, ORIGIN
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.board import Board
from hive_engine.game_state import GameState, Move, MoveType


class HiveEncoder:
    """
    Encodes Hive game states as tensors for neural network input.

    The encoder handles three key tasks:
    1. Board state → fixed-size tensor (for CNN input)
    2. Move ↔ action index (for policy head output)
    3. Legal move masking (to constrain policy to valid actions)

    The board is mapped to a 13×13 grid centered on the hive centroid.
    Axial hex coordinates (q, r) map directly to grid (col, row).

    Performance: centroid computation is cached per board state via
    _cached_center(). Methods that need the centroid multiple times
    (e.g., get_legal_action_mask encoding many moves) compute it once.

    Usage:
        encoder = HiveEncoder()
        tensor = encoder.encode_state(game_state)          # shape: (26, 13, 13)
        action = encoder.encode_action(move, game_state)   # int
        move = encoder.decode_action(action, game_state)   # Move
        mask = encoder.get_legal_action_mask(game_state)   # shape: (29407,)
    """

    # ── Board Grid Constants ────────────────────────────────────

    BOARD_SIZE: int = 13
    HALF_BOARD: int = 6  # BOARD_SIZE // 2

    # ── Channel Layout ──────────────────────────────────────────

    NUM_PIECE_TYPES: int = 5
    NUM_COLORS: int = 2
    NUM_STACK_LEVELS: int = 2  # ground (0) and stacked (1+)

    NUM_PIECE_CHANNELS: int = NUM_PIECE_TYPES * NUM_COLORS * NUM_STACK_LEVELS  # 20
    NUM_META_CHANNELS: int = 6
    NUM_CHANNELS: int = NUM_PIECE_CHANNELS + NUM_META_CHANNELS  # 26

    # Meta channel offsets
    CH_CURRENT_PLAYER: int = 20
    CH_TURN_NUMBER: int = 21
    CH_WHITE_QUEEN_PLACED: int = 22
    CH_BLACK_QUEEN_PLACED: int = 23
    CH_WHITE_HAND_COUNT: int = 24
    CH_BLACK_HAND_COUNT: int = 25

    # ── Action Space Constants ──────────────────────────────────

    NUM_GRID_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 169

    # Placement: piece_type (5) × grid_position (169)
    NUM_PLACEMENT_ACTIONS: int = NUM_PIECE_TYPES * NUM_GRID_CELLS  # 845

    # Movement: source_position (169) × destination_position (169)
    NUM_MOVEMENT_ACTIONS: int = NUM_GRID_CELLS * NUM_GRID_CELLS  # 28,561

    # Total action space
    ACTION_SPACE_SIZE: int = (
        NUM_PLACEMENT_ACTIONS + NUM_MOVEMENT_ACTIONS + 1  # +1 for pass
    )  # 29,407

    PASS_ACTION_INDEX: int = ACTION_SPACE_SIZE - 1  # 29,406

    # Offset where movement actions begin
    _MOVEMENT_OFFSET: int = NUM_PLACEMENT_ACTIONS  # 845

    def __init__(self) -> None:
        # Centroid cache: (board_piece_count, board_id) → (center_q, center_r)
        # Uses a simple single-entry cache since calls are typically
        # clustered for the same board state.
        self._center_cache_key: tuple[int, int] | None = None
        self._center_cache_val: tuple[int, int] = (0, 0)

    def _cached_center(self, board: Board) -> tuple[int, int]:
        """
        Return centroid, cached across calls for the same board.

        The cache key is (len(board.grid), id(board.grid)), which is
        invalidated whenever the grid dict object changes. This avoids
        the O(N) centroid scan on repeated calls within MCTS expansion
        (encode_state + get_legal_action_mask + decode all hit the
        centroid for the same board).
        """
        key = (len(board.grid), id(board.grid))
        if key != self._center_cache_key:
            self._center_cache_val = self._compute_center(board)
            self._center_cache_key = key
        return self._center_cache_val

    # ── State Encoding ──────────────────────────────────────────

    def encode_state(self, game_state: GameState) -> np.ndarray:
        """
        Convert a game state to a tensor of shape (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE).

        The board is centered on the centroid of all occupied positions.
        Piece planes encode binary presence at ground level and stacked level.
        Meta planes encode scalar game-state information as constant fills.

        Args:
            game_state: The current game state to encode.

        Returns:
            Float32 tensor of shape (26, 13, 13).
        """
        tensor = np.zeros(
            (self.NUM_CHANNELS, self.BOARD_SIZE, self.BOARD_SIZE),
            dtype=np.float32,
        )

        board = game_state.board
        center_q, center_r = self._cached_center(board)

        # ── Encode piece planes ─────────────────────────────────
        for pos, stack in board.grid.items():
            grid = self._hex_to_grid(pos.q, pos.r, center_q, center_r)
            if grid is None:
                continue  # Off-grid (clipped)
            row, col = grid

            # Ground level: piece at stack index 0
            ground_piece = stack[0]
            ch = ground_piece.color.value * 5 + ground_piece.piece_type.value
            tensor[ch, row, col] = 1.0

            # Stacked level: top piece when stack height >= 2
            if len(stack) >= 2:
                top_piece = stack[-1]
                ch = 10 + top_piece.color.value * 5 + top_piece.piece_type.value
                tensor[ch, row, col] = 1.0

        # ── Encode meta planes ──────────────────────────────────
        # Current player
        tensor[self.CH_CURRENT_PLAYER, :, :] = (
            1.0 if game_state.current_player == Color.WHITE else 0.0
        )

        # Turn number (normalized)
        tensor[self.CH_TURN_NUMBER, :, :] = min(game_state.turn / 100.0, 1.0)

        # Queen placement status
        tensor[self.CH_WHITE_QUEEN_PLACED, :, :] = (
            1.0 if game_state._queen_placed[Color.WHITE] else 0.0
        )
        tensor[self.CH_BLACK_QUEEN_PLACED, :, :] = (
            1.0 if game_state._queen_placed[Color.BLACK] else 0.0
        )

        # Hand counts (normalized)
        tensor[self.CH_WHITE_HAND_COUNT, :, :] = (
            len(game_state.hand(Color.WHITE)) / 11.0
        )
        tensor[self.CH_BLACK_HAND_COUNT, :, :] = (
            len(game_state.hand(Color.BLACK)) / 11.0
        )

        return tensor

    # ── Action Encoding ─────────────────────────────────────────

    def encode_action(self, move: Move, game_state: GameState) -> int:
        """
        Convert a Move to a flat action index.

        The action index is computed using the same centroid centering as
        encode_state, ensuring consistency between the board tensor and
        action indices.

        Args:
            move: The move to encode.
            game_state: The game state (needed for centroid computation).

        Returns:
            Integer action index in [0, ACTION_SPACE_SIZE).

        Raises:
            ValueError: If the move maps to an out-of-bounds grid position.
        """
        if move.move_type == MoveType.PASS:
            return self.PASS_ACTION_INDEX

        center_q, center_r = self._cached_center(game_state.board)

        if move.move_type == MoveType.PLACE:
            grid = self._hex_to_grid(move.to.q, move.to.r, center_q, center_r)
            if grid is None:
                raise ValueError(
                    f"Placement target {move.to} is outside the grid "
                    f"(center: ({center_q}, {center_r}))"
                )
            row, col = grid
            pos_idx = row * self.BOARD_SIZE + col
            return move.piece.piece_type.value * self.NUM_GRID_CELLS + pos_idx

        # MoveType.MOVE
        src_grid = self._hex_to_grid(
            move.from_pos.q, move.from_pos.r, center_q, center_r
        )
        dst_grid = self._hex_to_grid(
            move.to.q, move.to.r, center_q, center_r
        )
        if src_grid is None or dst_grid is None:
            raise ValueError(
                f"Movement {move.from_pos} -> {move.to} is outside the grid "
                f"(center: ({center_q}, {center_r}))"
            )
        src_row, src_col = src_grid
        dst_row, dst_col = dst_grid
        src_idx = src_row * self.BOARD_SIZE + src_col
        dst_idx = dst_row * self.BOARD_SIZE + dst_col
        return self._MOVEMENT_OFFSET + src_idx * self.NUM_GRID_CELLS + dst_idx

    def decode_action(self, action_index: int, game_state: GameState) -> Move:
        """
        Convert a flat action index back to a Move.

        Args:
            action_index: The action index to decode.
            game_state: The current game state (needed for piece lookup and centering).

        Returns:
            The corresponding Move object.

        Raises:
            ValueError: If the action index is out of range.
        """
        if action_index == self.PASS_ACTION_INDEX:
            return Move(MoveType.PASS)

        if action_index < 0 or action_index > self.PASS_ACTION_INDEX:
            raise ValueError(
                f"Action index {action_index} out of range "
                f"[0, {self.ACTION_SPACE_SIZE})"
            )

        center_q, center_r = self._cached_center(game_state.board)

        if action_index < self.NUM_PLACEMENT_ACTIONS:
            # Decode placement
            piece_type_val = action_index // self.NUM_GRID_CELLS
            pos_idx = action_index % self.NUM_GRID_CELLS
            row = pos_idx // self.BOARD_SIZE
            col = pos_idx % self.BOARD_SIZE
            hex_q, hex_r = self._grid_to_hex(row, col, center_q, center_r)

            piece_type = PieceType(piece_type_val)
            color = game_state.current_player

            # Pick the first available piece of this type from hand
            pieces = game_state.pieces_in_hand(color, piece_type)
            if not pieces:
                raise ValueError(
                    f"No {piece_type.name} available in "
                    f"{color.name}'s hand for placement"
                )
            piece = pieces[0]

            return Move(MoveType.PLACE, piece, HexCoord(hex_q, hex_r))

        # Decode movement
        move_idx = action_index - self._MOVEMENT_OFFSET
        src_idx = move_idx // self.NUM_GRID_CELLS
        dst_idx = move_idx % self.NUM_GRID_CELLS

        src_row = src_idx // self.BOARD_SIZE
        src_col = src_idx % self.BOARD_SIZE
        dst_row = dst_idx // self.BOARD_SIZE
        dst_col = dst_idx % self.BOARD_SIZE

        src_q, src_r = self._grid_to_hex(src_row, src_col, center_q, center_r)
        dst_q, dst_r = self._grid_to_hex(dst_row, dst_col, center_q, center_r)

        src_hex = HexCoord(src_q, src_r)
        dst_hex = HexCoord(dst_q, dst_r)

        # The moving piece is the top piece at the source position
        piece = game_state.board.top_piece_at(src_hex)
        if piece is None:
            raise ValueError(
                f"No piece at source position {src_hex} for movement"
            )

        return Move(MoveType.MOVE, piece, dst_hex, from_pos=src_hex)

    # ── Legal Move Masking ──────────────────────────────────────

    def get_legal_action_mask(
        self,
        game_state: GameState,
        legal_moves: list[Move] | None = None,
    ) -> np.ndarray:
        """
        Return a binary mask of shape (ACTION_SPACE_SIZE,) for legal actions.

        Args:
            game_state: The current game state.
            legal_moves: Pre-computed legal moves (avoids recomputation in MCTS).
                If None, calls game_state.legal_moves().

        Returns:
            Float32 array of shape (29407,) with 1.0 for legal actions.
        """
        if legal_moves is None:
            legal_moves = game_state.legal_moves()

        mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)

        for move in legal_moves:
            try:
                idx = self.encode_action(move, game_state)
                mask[idx] = 1.0
            except ValueError:
                # Move maps to out-of-bounds grid position (extremely rare)
                pass

        return mask

    # ── Grid ↔ Hex Conversion ───────────────────────────────────

    @staticmethod
    def _compute_center(board: Board) -> tuple[int, int]:
        """
        Compute the centroid of all occupied board positions.

        Returns the centroid as (center_q, center_r) rounded to the
        nearest integer. For an empty board, returns (0, 0).
        """
        grid = board.grid
        if not grid:
            return (0, 0)

        total_q = 0
        total_r = 0
        count = 0
        for pos in grid:
            total_q += pos.q
            total_r += pos.r
            count += 1

        # Use Python's built-in round() which does banker's rounding
        return (round(total_q / count), round(total_r / count))

    @staticmethod
    def _hex_to_grid(
        q: int, r: int, center_q: int, center_r: int
    ) -> tuple[int, int] | None:
        """
        Map hex coordinates (q, r) to grid coordinates (row, col).

        Returns None if the position falls outside the 13×13 grid.

        Mapping:
            col = q - center_q + HALF_BOARD
            row = r - center_r + HALF_BOARD
        """
        col = q - center_q + HiveEncoder.HALF_BOARD
        row = r - center_r + HiveEncoder.HALF_BOARD
        if 0 <= row < HiveEncoder.BOARD_SIZE and 0 <= col < HiveEncoder.BOARD_SIZE:
            return (row, col)
        return None

    @staticmethod
    def _grid_to_hex(
        row: int, col: int, center_q: int, center_r: int
    ) -> tuple[int, int]:
        """
        Map grid coordinates (row, col) back to hex coordinates (q, r).

        Inverse of _hex_to_grid:
            q = col - HALF_BOARD + center_q
            r = row - HALF_BOARD + center_r
        """
        q = col - HiveEncoder.HALF_BOARD + center_q
        r = row - HiveEncoder.HALF_BOARD + center_r
        return (q, r)
