"""
Graph encoder for converting Hive game states to GNN-ready graphs.

Converts a GameState into a HiveGraph suitable for message-passing
neural networks. Each piece on the board becomes a node; hand pieces
are grouped by type into aggregate hand nodes. Edges connect
spatially adjacent board positions with directional features.

Node features (21 dims):
    [0:5]   piece_type one-hot (Q=0, A=1, G=2, S=3, B=4)
    [5:7]   color one-hot (W=0, B=1)
    [7]     is_on_ground (1 if stack height == 1)
    [8]     is_on_top (always 1 for piece nodes)
    [9]     stack_height / 4.0
    [10]    is_queen (1 if queen)
    [11]    queen_neighbor_count / 6.0
    [12]    num_occupied_neighbors / 6.0
    [13:19] empty_dir_mask (6 directions, 1 if neighbor is unoccupied)
    [19]    is_hand_node (0 for board, 1 for hand)
    [20]    count_remaining / count_per_player (for hand nodes)

Edge features (9 dims):
    [0]     dq (neighbor.q - pos.q)
    [1]     dr (neighbor.r - pos.r)
    [2:8]   direction one-hot (6 dims)
    [8]     is_stacked (0.0 for spatial edges)

Global features (6 dims):
    [0]     current_player == WHITE ? 1.0 : 0.0
    [1]     min(turn / 100, 1.0)
    [2]     white_queen_placed
    [3]     black_queen_placed
    [4]     len(white_hand) / 11.0
    [5]     len(black_hand) / 11.0
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hive_engine.board import Board
from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState
from hive_engine.hex_coord import Direction, ALL_DIRECTIONS, HexCoord
from hive_engine.pieces import Color, PieceType, Piece

from hive_gnn.graph_types import HiveGraph


class GraphEncoder:
    """
    Converts a Hive GameState into a HiveGraph for GNN consumption.

    Each occupied board position produces one piece node (for the top piece).
    Each piece type still in a player's hand produces one hand node with
    a count feature. Edges are bidirectional between spatially adjacent
    board positions.

    The encoder caches the board centroid across repeated calls on the
    same board state, matching the caching strategy of HiveEncoder.

    Usage:
        encoder = GraphEncoder()
        graph = encoder.encode(game_state)
    """

    def __init__(self) -> None:
        self._center_cache_key: tuple[int, int] | None = None
        self._center_cache_val: tuple[int, int] = (0, 0)

    def _cached_center(self, board: Board) -> tuple[int, int]:
        """Return the board centroid, cached across calls for the same board."""
        key = (len(board.grid), id(board.grid))
        if key != self._center_cache_key:
            self._center_cache_val = HiveEncoder._compute_center(board)
            self._center_cache_key = key
        return self._center_cache_val

    def encode(self, game_state: GameState) -> HiveGraph:
        """
        Convert a GameState to a HiveGraph.

        Args:
            game_state: The current game state.

        Returns:
            A HiveGraph with node features, edge index/features, global
            features, and auxiliary position/type arrays.
        """
        board = game_state.board
        center_q, center_r = self._cached_center(board)

        # -----------------------------------------------------------------
        # 1. Build piece nodes (one per occupied board position, top piece)
        # -----------------------------------------------------------------
        node_features_list: list[np.ndarray] = []
        node_positions_list: list[tuple[int, int]] = []
        node_piece_types_list: list[int] = []

        # Map from HexCoord -> node index (for edge construction)
        pos_to_node: dict[HexCoord, int] = {}
        node_idx = 0

        for pos, stack in board.grid.items():
            grid = HiveEncoder._hex_to_grid(pos.q, pos.r, center_q, center_r)
            if grid is None:
                continue  # off-grid, skip

            row, col = grid
            top_piece = stack[-1]
            stack_height = len(stack)

            feat = np.zeros(21, dtype=np.float32)

            # [0:5] piece_type one-hot
            feat[top_piece.piece_type.value] = 1.0

            # [5:7] color one-hot
            feat[5 + top_piece.color.value] = 1.0

            # [7] is_on_ground
            feat[7] = 1.0 if stack_height == 1 else 0.0

            # [8] is_on_top (always 1 for board piece nodes)
            feat[8] = 1.0

            # [9] stack_height / 4.0
            feat[9] = stack_height / 4.0

            # [10] is_queen
            feat[10] = 1.0 if top_piece.piece_type == PieceType.QUEEN else 0.0

            # [11] queen_neighbor_count / 6.0
            feat[11] = game_state.queen_surrounded_count(top_piece.color) / 6.0

            # [12] num_occupied_neighbors / 6.0
            feat[12] = board.num_occupied_neighbors(pos) / 6.0

            # [13:19] empty_dir_mask (Direction enum order: E=0..SE=5)
            for d in ALL_DIRECTIONS:
                neighbor = pos.neighbor(d)
                if neighbor not in board.grid:
                    feat[13 + d.value] = 1.0

            # [19] is_hand_node
            feat[19] = 0.0

            # [20] count_remaining (0 for board nodes)
            feat[20] = 0.0

            node_features_list.append(feat)
            node_positions_list.append((row, col))
            node_piece_types_list.append(top_piece.piece_type.value)
            pos_to_node[pos] = node_idx
            node_idx += 1

        num_piece_nodes = node_idx

        # -----------------------------------------------------------------
        # 2. Build hand nodes (one per piece type per player with count)
        # -----------------------------------------------------------------
        for color in (Color.WHITE, Color.BLACK):
            hand = game_state.hand(color)
            # Group by piece type
            type_counts: dict[PieceType, int] = defaultdict(int)
            for p in hand:
                type_counts[p.piece_type] += 1

            for pt, count in type_counts.items():
                feat = np.zeros(21, dtype=np.float32)

                # [0:5] piece_type one-hot
                feat[pt.value] = 1.0

                # [5:7] color one-hot
                feat[5 + color.value] = 1.0

                # [7] is_on_ground
                feat[7] = 0.0

                # [8] is_on_top
                feat[8] = 0.0

                # [9] stack_height / 4.0
                feat[9] = 0.0

                # [10] is_queen
                feat[10] = 1.0 if pt == PieceType.QUEEN else 0.0

                # [11] queen_neighbor_count / 6.0
                feat[11] = 0.0

                # [12] num_occupied_neighbors / 6.0
                feat[12] = 0.0

                # [13:19] empty_dir_mask (all zeros for hand nodes)
                # already zero

                # [19] is_hand_node
                feat[19] = 1.0

                # [20] count_remaining / count_per_player
                feat[20] = count / pt.count_per_player

                node_features_list.append(feat)
                node_idx += 1

        # -----------------------------------------------------------------
        # 3. Build edges (bidirectional between adjacent board positions)
        # -----------------------------------------------------------------
        src_list: list[int] = []
        dst_list: list[int] = []
        edge_feat_list: list[np.ndarray] = []

        for pos, src_idx in pos_to_node.items():
            for d in ALL_DIRECTIONS:
                neighbor = pos.neighbor(d)
                if neighbor in pos_to_node:
                    dst_idx = pos_to_node[neighbor]

                    ef = np.zeros(9, dtype=np.float32)
                    # [0] dq
                    ef[0] = float(neighbor.q - pos.q)
                    # [1] dr
                    ef[1] = float(neighbor.r - pos.r)
                    # [2:8] direction one-hot
                    ef[2 + d.value] = 1.0
                    # [8] is_stacked
                    ef[8] = 0.0

                    src_list.append(src_idx)
                    dst_list.append(dst_idx)
                    edge_feat_list.append(ef)

        # -----------------------------------------------------------------
        # 4. Assemble arrays
        # -----------------------------------------------------------------
        if node_features_list:
            node_features = np.stack(node_features_list, axis=0)
        else:
            node_features = np.zeros((0, 21), dtype=np.float32)

        if src_list:
            edge_index = np.array([src_list, dst_list], dtype=np.int64)
            edge_features = np.stack(edge_feat_list, axis=0)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 9), dtype=np.float32)

        if node_positions_list:
            node_positions = np.array(node_positions_list, dtype=np.int32)
        else:
            node_positions = np.zeros((0, 2), dtype=np.int32)

        if node_piece_types_list:
            node_piece_types = np.array(node_piece_types_list, dtype=np.int32)
        else:
            node_piece_types = np.zeros((0,), dtype=np.int32)

        # -----------------------------------------------------------------
        # 5. Global features
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

        return HiveGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=global_features,
            num_piece_nodes=num_piece_nodes,
            node_positions=node_positions,
            node_piece_types=node_piece_types,
        )
