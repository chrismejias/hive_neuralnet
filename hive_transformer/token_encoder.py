"""
Token encoder for converting Hive game states to transformer-ready sequences.

Converts a GameState into a HiveTokenSequence. Each piece on the board
becomes one token, each (color, piece_type) in hand becomes one hand token,
and a CLS token is prepended for the value head.

Token features are identical to GNN node features (21 dims).
Position encoding uses grid index: row * 13 + col for board pieces,
169 for CLS and hand tokens.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hive_engine.board import Board
from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState
from hive_engine.hex_coord import ALL_DIRECTIONS, HexCoord
from hive_engine.pieces import Color, PieceType

from hive_transformer.token_types import (
    TOKEN_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    TOKEN_TYPE_CLS,
    TOKEN_TYPE_BOARD,
    TOKEN_TYPE_HAND,
    OFF_BOARD_POSITION,
    HiveTokenSequence,
)


class TokenEncoder:
    """
    Converts a Hive GameState into a HiveTokenSequence for transformer input.

    Token ordering: [CLS, board_piece₁, ..., board_pieceₙ, hand₁, ..., handₘ]

    Board piece features and hand node features are identical to the GNN
    graph_encoder.py implementation (21-dim per token).

    Usage:
        encoder = TokenEncoder()
        seq = encoder.encode(game_state)
    """

    def __init__(self) -> None:
        self._center_cache_key: tuple[int, int] | None = None
        self._center_cache_val: tuple[int, int] = (0, 0)

    def _cached_center(self, board: Board) -> tuple[int, int]:
        """Return the board centroid, cached across calls."""
        key = (len(board.grid), id(board.grid))
        if key != self._center_cache_key:
            self._center_cache_val = HiveEncoder._compute_center(board)
            self._center_cache_key = key
        return self._center_cache_val

    def encode(self, game_state: GameState) -> HiveTokenSequence:
        """
        Convert a GameState to a HiveTokenSequence.

        Args:
            game_state: The current game state.

        Returns:
            A HiveTokenSequence with token features, positions, types,
            and global features.
        """
        board = game_state.board
        center_q, center_r = self._cached_center(board)

        features_list: list[np.ndarray] = []
        positions_list: list[int] = []
        types_list: list[int] = []

        # -----------------------------------------------------------------
        # 1. CLS token
        # -----------------------------------------------------------------
        features_list.append(np.zeros(TOKEN_FEAT_DIM, dtype=np.float32))
        positions_list.append(OFF_BOARD_POSITION)
        types_list.append(TOKEN_TYPE_CLS)

        # -----------------------------------------------------------------
        # 2. Board piece tokens (one per occupied position, top piece)
        # -----------------------------------------------------------------
        num_board_tokens = 0

        for pos, stack in board.grid.items():
            grid = HiveEncoder._hex_to_grid(pos.q, pos.r, center_q, center_r)
            if grid is None:
                continue

            row, col = grid
            top_piece = stack[-1]
            stack_height = len(stack)

            feat = np.zeros(TOKEN_FEAT_DIM, dtype=np.float32)

            # [0:5] piece_type one-hot
            feat[top_piece.piece_type.value] = 1.0

            # [5:7] color one-hot
            feat[5 + top_piece.color.value] = 1.0

            # [7] is_on_ground
            feat[7] = 1.0 if stack_height == 1 else 0.0

            # [8] is_on_top
            feat[8] = 1.0

            # [9] stack_height / 4.0
            feat[9] = stack_height / 4.0

            # [10] is_queen
            feat[10] = 1.0 if top_piece.piece_type == PieceType.QUEEN else 0.0

            # [11] queen_neighbor_count / 6.0
            feat[11] = game_state.queen_surrounded_count(top_piece.color) / 6.0

            # [12] num_occupied_neighbors / 6.0
            feat[12] = board.num_occupied_neighbors(pos) / 6.0

            # [13:19] empty_dir_mask
            for d in ALL_DIRECTIONS:
                neighbor = pos.neighbor(d)
                if neighbor not in board.grid:
                    feat[13 + d.value] = 1.0

            # [19] is_hand_node = 0 for board pieces
            feat[19] = 0.0

            # [20] count_remaining = 0 for board pieces
            feat[20] = 0.0

            features_list.append(feat)
            positions_list.append(row * 13 + col)  # grid position index
            types_list.append(TOKEN_TYPE_BOARD)
            num_board_tokens += 1

        # -----------------------------------------------------------------
        # 3. Hand tokens (one per piece type per player with count)
        # -----------------------------------------------------------------
        for color in (Color.WHITE, Color.BLACK):
            hand = game_state.hand(color)
            type_counts: dict[PieceType, int] = defaultdict(int)
            for p in hand:
                type_counts[p.piece_type] += 1

            for pt, count in type_counts.items():
                feat = np.zeros(TOKEN_FEAT_DIM, dtype=np.float32)

                feat[pt.value] = 1.0
                feat[5 + color.value] = 1.0
                feat[7] = 0.0   # not on ground
                feat[8] = 0.0   # not on top
                feat[9] = 0.0   # no stack height
                feat[10] = 1.0 if pt == PieceType.QUEEN else 0.0
                feat[11] = 0.0  # no queen neighbor count
                feat[12] = 0.0  # no occupied neighbors
                # [13:19] empty_dir_mask all zeros
                feat[19] = 1.0  # is_hand_node
                feat[20] = count / pt.count_per_player

                features_list.append(feat)
                positions_list.append(OFF_BOARD_POSITION)
                types_list.append(TOKEN_TYPE_HAND)

        # -----------------------------------------------------------------
        # 4. Global features
        # -----------------------------------------------------------------
        global_features = np.array(
            [
                1.0 if game_state.current_player == Color.WHITE else 0.0,
                min(game_state.turn / 100.0, 1.0),
                1.0 if game_state._queen_placed[Color.WHITE] else 0.0,
                1.0 if game_state._queen_placed[Color.BLACK] else 0.0,
                len(game_state.hand(Color.WHITE)) / 11.0,
                len(game_state.hand(Color.BLACK)) / 11.0,
            ],
            dtype=np.float32,
        )

        # -----------------------------------------------------------------
        # 5. Assemble
        # -----------------------------------------------------------------
        token_features = np.stack(features_list, axis=0)
        token_positions = np.array(positions_list, dtype=np.int32)
        token_types = np.array(types_list, dtype=np.int32)

        return HiveTokenSequence(
            token_features=token_features,
            token_positions=token_positions,
            token_types=token_types,
            num_board_tokens=num_board_tokens,
            global_features=global_features,
        )
