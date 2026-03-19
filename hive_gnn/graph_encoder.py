"""
Graph encoder for converting Hive game states to GNN-ready graphs.

Converts a GameState into a HiveGraph suitable for message-passing
neural networks. Each piece on the board becomes a node (including
pieces sandwiched under beetles); hand pieces are grouped by type
into aggregate hand nodes. Edges connect spatially adjacent board
positions (between top pieces) and vertically within stacks.

Node features (25 dims):
    [0:8]   piece_type one-hot (Q=0, A=1, G=2, S=3, B=4, M=5, L=6, P=7)
    [8:10]  color one-hot (W=0, B=1)
    [10]    is_on_ground (1 if piece height == 0)
    [11]    is_on_top (1 if piece is top of stack)
    [12]    stack_height / 4.0
    [13]    is_queen (1 if queen)
    [14]    queen_neighbor_count / 6.0
    [15]    num_occupied_neighbors / 6.0
    [16:22] empty_dir_mask (6 directions, 1 if neighbor unoccupied; top piece only)
    [22]    is_hand_node (0 for board, 1 for hand)
    [23]    count_remaining / count_per_player (for hand nodes)
    [24]    stack_position (piece height / 4.0)

Edge features (9 dims):
    [0]     dq (neighbor.q - pos.q)  — 0 for vertical edges
    [1]     dr (neighbor.r - pos.r)  — 0 for vertical edges
    [2:8]   direction one-hot (6 dims) — all zeros for vertical edges
    [8]     is_stacked (1.0 for vertical edges, 0.0 for spatial)

Global features (6 dims):
    [0]     current_player == WHITE ? 1.0 : 0.0
    [1]     min(turn / 100, 1.0)
    [2]     white_queen_placed
    [3]     black_queen_placed
    [4]     len(white_hand) / 14.0
    [5]     len(black_hand) / 14.0
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hive_engine.board import Board
from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState
from hive_engine.hex_coord import HexCoord, _OFFSET_LIST
from hive_engine.pieces import Color, PieceType, Piece

from hive_gnn.graph_types import HiveGraph


class GraphEncoder:
    """
    Converts a Hive GameState into a HiveGraph for GNN consumption.

    Each piece on the board produces one piece node — including pieces
    buried under beetles. Hand pieces are grouped by (color, type) into
    aggregate hand nodes. Spatial edges connect top pieces at adjacent
    positions; vertical edges connect consecutive pieces in the same stack.

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
        # Pre-allocate output buffers.
        # Upper bound: all pieces across all stacks + hand nodes + CLS.
        # Avoids per-node np.zeros(22) + np.stack() at the end.
        # -----------------------------------------------------------------
        n_grid = len(board.grid)
        max_nodes = n_grid * 4 + 22   # conservative: stacks ≤ 4, hand ≤ 22
        max_edges = n_grid * 12 + 20  # 6 dirs × 2 (bidirectional) + stack edges

        node_features    = np.zeros((max_nodes, 25), dtype=np.float32)
        node_positions   = np.empty((n_grid * 4, 2), dtype=np.int32)  # board nodes only
        node_piece_types = np.empty(n_grid * 4,      dtype=np.int32)  # board nodes only

        edge_index_buf = np.empty((2, max_edges), dtype=np.int64)
        edge_features  = np.zeros((max_edges, 9), dtype=np.float32)

        node_idx        = 0
        board_node_count = 0  # how many board-piece nodes added (= num_piece_nodes)
        n_edges         = 0

        # -----------------------------------------------------------------
        # 1. Build piece nodes (one per piece on board, including stacked)
        # -----------------------------------------------------------------
        # Tuple-keyed dicts avoid HexCoord allocation during edge building.
        top_node_for_pos: dict[tuple[int, int], int]        = {}
        stack_nodes:      dict[tuple[int, int], list[int]]  = {}

        # Hoist expensive per-piece calls to once per encode call:
        # queen_surrounded_count() is identical for all pieces of the same color.
        grid_tuples: set[tuple[int, int]] = {(p.q, p.r) for p in board.grid}
        queen_surround_w = game_state.queen_surrounded_count(Color.WHITE) / 6.0
        queen_surround_b = game_state.queen_surrounded_count(Color.BLACK) / 6.0

        for pos, stack in board.grid.items():
            grid = HiveEncoder._hex_to_grid(pos.q, pos.r, center_q, center_r)
            if grid is None:
                continue  # off-grid, skip

            row, col = grid
            stack_height = len(stack)

            # --- Compute per-position neighbor info in one unrolled pass ---
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

            pos_node_indices: list[int] = []

            for height, piece in enumerate(stack):
                is_top = (height == stack_height - 1)
                f = node_features[node_idx]  # direct view — no allocation

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

                # [14] queen_neighbor_count / 6.0  (hoisted: same per color per call)
                f[14] = queen_surround_w if piece.color == Color.WHITE else queen_surround_b

                # [15] num_occupied_neighbors / 6.0  (hoisted: same for all pieces at pos)
                f[15] = occ_count * (1.0 / 6.0)

                # [16:22] empty_dir_mask — only for top piece.
                # Derived from the unrolled occ booleans above; no extra neighbor scan.
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

                node_positions[board_node_count, 0] = row
                node_positions[board_node_count, 1] = col
                node_piece_types[board_node_count]  = piece.piece_type.value

                pos_node_indices.append(node_idx)
                board_node_count += 1
                node_idx += 1

            pos_key = (pq, pr)
            top_node_for_pos[pos_key] = pos_node_indices[-1]
            stack_nodes[pos_key]      = pos_node_indices

        num_piece_nodes = node_idx

        # -----------------------------------------------------------------
        # 2. Build hand nodes (one per piece type per player with count)
        # -----------------------------------------------------------------
        for color in (Color.WHITE, Color.BLACK):
            hand = game_state.hand(color)
            type_counts: dict[PieceType, int] = defaultdict(int)
            for p in hand:
                type_counts[p.piece_type] += 1

            for pt, count in type_counts.items():
                f = node_features[node_idx]  # already zero

                f[pt.value] = 1.0
                f[8 + color.value] = 1.0
                # [10][11][12] stay 0: not on ground, not on top, no stack height
                if pt == PieceType.QUEEN:
                    f[13] = 1.0
                # [14][15][16:22] stay 0: hand node
                f[22] = 1.0  # is_hand_node
                f[23] = count / pt.count_per_player
                # [24] stays 0

                node_idx += 1

        # -----------------------------------------------------------------
        # 3. Build edges
        # -----------------------------------------------------------------
        # 3a. Spatial edges: bidirectional between top pieces at adjacent positions.
        # Using _OFFSET_LIST avoids HexCoord.neighbor() allocation; dq/dr are
        # already known so we skip the neighbor.q - pos.q computation too.
        for (pos_q, pos_r), src_idx in top_node_for_pos.items():
            for i, (dq, dr) in enumerate(_OFFSET_LIST):
                nq, nr = pos_q + dq, pos_r + dr
                if (nq, nr) in top_node_for_pos:
                    dst_idx = top_node_for_pos[(nq, nr)]

                    ef = edge_features[n_edges]  # direct view
                    ef[0] = float(dq)
                    ef[1] = float(dr)
                    ef[2 + i] = 1.0
                    # ef[8] stays 0.0 (is_stacked = False)
                    edge_index_buf[0, n_edges] = src_idx
                    edge_index_buf[1, n_edges] = dst_idx
                    n_edges += 1

        # 3b. Vertical edges: bidirectional between consecutive pieces in stack
        for indices in stack_nodes.values():
            if len(indices) < 2:
                continue
            for i in range(len(indices) - 1):
                lower_idx = indices[i]
                upper_idx = indices[i + 1]

                # Edge from lower to upper
                edge_features[n_edges, 8] = 1.0  # is_stacked
                edge_index_buf[0, n_edges] = lower_idx
                edge_index_buf[1, n_edges] = upper_idx
                n_edges += 1

                # Edge from upper to lower
                edge_features[n_edges, 8] = 1.0  # is_stacked
                edge_index_buf[0, n_edges] = upper_idx
                edge_index_buf[1, n_edges] = lower_idx
                n_edges += 1

        # -----------------------------------------------------------------
        # 4. Assemble arrays — slice pre-allocated buffers (no copy needed)
        # -----------------------------------------------------------------
        if node_idx > 0:
            final_node_features = node_features[:node_idx]
        else:
            final_node_features = np.zeros((0, 25), dtype=np.float32)

        if n_edges > 0:
            edge_index        = edge_index_buf[:, :n_edges]
            final_edge_features = edge_features[:n_edges]
        else:
            edge_index          = np.zeros((2, 0), dtype=np.int64)
            final_edge_features = np.zeros((0, 9), dtype=np.float32)

        if board_node_count > 0:
            final_node_positions   = node_positions[:board_node_count]
            final_node_piece_types = node_piece_types[:board_node_count]
        else:
            final_node_positions   = np.zeros((0, 2), dtype=np.int32)
            final_node_piece_types = np.zeros((0,),   dtype=np.int32)

        # -----------------------------------------------------------------
        # 5. Global features
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

        return HiveGraph(
            node_features=final_node_features,
            edge_index=edge_index,
            edge_features=final_edge_features,
            global_features=global_features,
            num_piece_nodes=num_piece_nodes,
            node_positions=final_node_positions,
            node_piece_types=final_node_piece_types,
        )
