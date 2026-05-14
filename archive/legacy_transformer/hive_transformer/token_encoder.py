"""Compatibility wrapper for the archived legacy transformer token encoder."""

from archive.legacy_transformer.hive_transformer.token_encoder import TokenEncoder

<<<<<<<< HEAD:hive_transformer/token_encoder.py
__all__ = ["TokenEncoder"]
========
Token features are identical to GNN node features (25 dims).
Position encoding uses grid index: row * 13 + col for board pieces
(stacked pieces share the same position index), 169 for CLS and hand tokens.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hive_engine.board import Board
from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState
from hive_engine.hex_coord import HexCoord, _OFFSET_LIST
from hive_engine.pieces import Color, PieceType

from hive_common.token_types import (
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

    Board piece tokens include all pieces in each stack (bottom to top),
    not just the top piece. Features are 25-dim per token, matching the
    GNN graph_encoder.py implementation.

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

        # -----------------------------------------------------------------
        # Pre-allocate output buffers.
        # Upper bound: 1 CLS + all pieces across all stacks + hand tokens.
        # Avoids per-token np.zeros() + np.stack() at the end.
        # -----------------------------------------------------------------
        n_grid = len(board.grid)
        max_tokens = 1 + n_grid * 4 + 22  # conservative upper bound
        token_features = np.zeros((max_tokens, TOKEN_FEAT_DIM), dtype=np.float32)
        token_positions = np.empty(max_tokens, dtype=np.int32)
        token_types    = np.empty(max_tokens, dtype=np.int32)
        tok = 0  # running token index

        # -----------------------------------------------------------------
        # 1. CLS token (features already zero from np.zeros above)
        # -----------------------------------------------------------------
        token_positions[tok] = OFF_BOARD_POSITION
        token_types[tok]     = TOKEN_TYPE_CLS
        tok += 1

        # -----------------------------------------------------------------
        # 2. Board piece tokens (one per piece, including stacked)
        # -----------------------------------------------------------------
        num_board_tokens = 0

        # Build a tuple set for O(1) neighbor checks without HexCoord allocation.
        # queen_surrounded_count() is the same for all pieces of a given color,
        # so compute it once here rather than once per piece.
        grid_tuples: set[tuple[int, int]] = {(p.q, p.r) for p in board.grid}
        queen_surround_w = game_state.queen_surrounded_count(Color.WHITE) / 6.0
        queen_surround_b = game_state.queen_surrounded_count(Color.BLACK) / 6.0

        for pos, stack in board.grid.items():
            grid = HiveEncoder._hex_to_grid(pos.q, pos.r, center_q, center_r)
            if grid is None:
                continue

            row, col = grid
            stack_height = len(stack)
            grid_pos = row * 13 + col

            # --- Compute per-position neighbor info in one unrolled pass ---
            # Inlining _OFFSET_LIST avoids Python loop + tuple-unpack overhead.
            # Simultaneously computes occ_count (feat[12]) and the 6-bit
            # empty_dir_mask (feat[13:19]) so we only scan neighbors once.
            pq, pr = pos.q, pos.r
            e_occ  = int((pq + 1, pr    ) in grid_tuples)
            ne_occ = int((pq + 1, pr - 1) in grid_tuples)
            nw_occ = int((pq,     pr - 1) in grid_tuples)
            w_occ  = int((pq - 1, pr    ) in grid_tuples)
            sw_occ = int((pq - 1, pr + 1) in grid_tuples)
            se_occ = int((pq,     pr + 1) in grid_tuples)
            occ_count = e_occ + ne_occ + nw_occ + w_occ + sw_occ + se_occ

            for height, piece in enumerate(stack):
                is_top = (height == stack_height - 1)
                f = token_features[tok]  # direct view — no allocation

                # [0:8] piece_type one-hot
                f[piece.piece_type.value] = 1.0

                # [8:10] color one-hot
                f[8 + piece.color.value] = 1.0

                # [10] is_on_ground
                if height == 0:
                    f[10] = 1.0

                # [11] is_on_top
                if is_top:
                    f[11] = 1.0

                # [12] stack_height / 4.0
                f[12] = stack_height * 0.25

                # [13] is_queen
                if piece.piece_type == PieceType.QUEEN:
                    f[13] = 1.0

                # [14] queen_neighbor_count / 6.0  (hoisted: same for all pieces of this color)
                f[14] = (
                    queen_surround_w
                    if piece.color == Color.WHITE
                    else queen_surround_b
                )

                # [15] num_occupied_neighbors / 6.0  (hoisted: same for all pieces at this pos)
                f[15] = occ_count * (1.0 / 6.0)

                # [16:22] empty_dir_mask — only for top piece.
                # Unrolled from the occ booleans computed above; no extra lookup.
                if is_top:
                    if not e_occ:  f[16] = 1.0
                    if not ne_occ: f[17] = 1.0
                    if not nw_occ: f[18] = 1.0
                    if not w_occ:  f[19] = 1.0
                    if not sw_occ: f[20] = 1.0
                    if not se_occ: f[21] = 1.0

                # [22] is_hand_node = 0 (already zero from np.zeros)
                # [23] count_remaining = 0 (already zero)
                # [24] stack_position
                f[24] = height * 0.25

                token_positions[tok] = grid_pos
                token_types[tok]     = TOKEN_TYPE_BOARD
                tok += 1
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
                f = token_features[tok]  # already zero

                f[pt.value] = 1.0
                f[8 + color.value] = 1.0
                # [10][11][12] stay 0: not on ground, not on top, no stack height
                if pt == PieceType.QUEEN:
                    f[13] = 1.0
                # [14][15][16:22] stay 0: hand token
                f[22] = 1.0  # is_hand_node
                f[23] = count / pt.count_per_player
                # [24] stays 0

                token_positions[tok] = OFF_BOARD_POSITION
                token_types[tok]     = TOKEN_TYPE_HAND
                tok += 1

        # -----------------------------------------------------------------
        # 4. Global features
        # -----------------------------------------------------------------
        global_features = np.array(
            [
                1.0 if game_state.current_player == Color.WHITE else 0.0,
                min(game_state.turn / 100.0, 1.0),
                1.0 if game_state._queen_placed[Color.WHITE] else 0.0,
                1.0 if game_state._queen_placed[Color.BLACK] else 0.0,
                len(game_state.hand(Color.WHITE)) / 14.0,
                len(game_state.hand(Color.BLACK)) / 14.0,
            ],
            dtype=np.float32,
        )

        # -----------------------------------------------------------------
        # 5. Return slices of the pre-allocated buffers (no copy needed).
        # -----------------------------------------------------------------
        return HiveTokenSequence(
            token_features=token_features[:tok],
            token_positions=token_positions[:tok],
            token_types=token_types[:tok],
            num_board_tokens=num_board_tokens,
            global_features=global_features,
        )
>>>>>>>> 7c7d146 (Refactor legacy transformer and MC packages):archive/legacy_transformer/hive_transformer/token_encoder.py
