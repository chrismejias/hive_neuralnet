"""CPU graph encoder for the hybrid GNN value trunk."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hive_engine.board import Board
from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState
from hive_engine.game_state import MoveType
from hive_engine.hex_coord import _OFFSET_LIST
from hive_engine.pieces import Color, PieceType
from hive_hybrid_gnn.graph_types import HybridGraph, edge_feat_dim_for_radius


def _hex_distance(dq: int, dr: int) -> int:
    return max(abs(dq), abs(dr), abs(dq + dr))


def radius_offsets(radius: int) -> list[tuple[int, int]]:
    """All nonzero axial offsets with hex distance <= radius."""
    offsets: list[tuple[int, int]] = []
    for dq in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if dq == 0 and dr == 0:
                continue
            dist = _hex_distance(dq, dr)
            if 1 <= dist <= radius:
                offsets.append((dq, dr))
    offsets.sort(key=lambda x: (_hex_distance(x[0], x[1]), x[0], x[1]))
    return offsets


class HybridGraphEncoder:
    """Encode a Python GameState into a graph for the hybrid value trunk.

    Board pieces become nodes, including buried pieces. Hand pieces are grouped
    by color/type into aggregate hand nodes. Spatial edges connect top pieces
    whose occupied cells are within ``radius`` hexes; stack edges connect
    adjacent levels inside beetle stacks.
    """

    def __init__(self, radius: int = 2) -> None:
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.radius = int(radius)
        self.offsets = radius_offsets(self.radius)
        self.offset_to_index = {offset: i for i, offset in enumerate(self.offsets)}
        self.edge_feat_dim = edge_feat_dim_for_radius(self.radius)
        self._center_cache_key: tuple[int, int] | None = None
        self._center_cache_val: tuple[int, int] = (0, 0)

    def _cached_center(self, board: Board) -> tuple[int, int]:
        key = (len(board.grid), id(board.grid))
        if key != self._center_cache_key:
            self._center_cache_val = HiveEncoder._compute_center(board)
            self._center_cache_key = key
        return self._center_cache_val

    def _write_spatial_edge(
        self,
        edge_index: np.ndarray,
        edge_features: np.ndarray,
        n_edges: int,
        src_idx: int,
        dst_idx: int,
        dq: int,
        dr: int,
    ) -> int:
        edge_index[0, n_edges] = src_idx
        edge_index[1, n_edges] = dst_idx
        feat = edge_features[n_edges]
        feat[0] = float(dq) / float(self.radius)
        feat[1] = float(dr) / float(self.radius)
        feat[2 + self.offset_to_index[(dq, dr)]] = 1.0
        return n_edges + 1

    def encode(self, game_state: GameState) -> HybridGraph:
        board = game_state.board
        center_q, center_r = self._cached_center(board)

        n_grid = len(board.grid)
        max_stack = max((len(stack) for stack in board.grid.values()), default=1)
        max_nodes = n_grid * max_stack + 22
        max_edges = n_grid * len(self.offsets) + max(0, n_grid * max_stack * 2)

        node_features = np.zeros((max_nodes, 27), dtype=np.float32)
        node_positions = np.empty((n_grid * max_stack, 2), dtype=np.int32)
        node_piece_types = np.empty((n_grid * max_stack,), dtype=np.int32)
        edge_index = np.empty((2, max_edges), dtype=np.int64)
        edge_features = np.zeros((max_edges, self.edge_feat_dim), dtype=np.float32)

        node_idx = 0
        board_node_count = 0
        n_edges = 0
        top_node_for_pos: dict[tuple[int, int], int] = {}
        stack_nodes: dict[tuple[int, int], list[int]] = {}
        grid_tuples = {(p.q, p.r) for p in board.grid}
        articulation_positions = board.find_articulation_points()
        move_sources = {
            (move.from_pos.q, move.from_pos.r)
            for move in game_state.legal_moves()
            if move.move_type == MoveType.MOVE and move.from_pos is not None
        }
        queen_surround_w = game_state.queen_surrounded_count(Color.WHITE) / 6.0
        queen_surround_b = game_state.queen_surrounded_count(Color.BLACK) / 6.0

        for pos, stack in board.grid.items():
            grid = HiveEncoder._hex_to_grid(pos.q, pos.r, center_q, center_r)
            if grid is None:
                continue
            row, col = grid
            stack_height = len(stack)
            pq, pr = pos.q, pos.r
            occ_by_dir = [int((pq + dq, pr + dr) in grid_tuples) for dq, dr in _OFFSET_LIST]
            occ_count = sum(occ_by_dir)
            pos_nodes: list[int] = []

            for height, piece in enumerate(stack):
                is_top = height == stack_height - 1
                feat = node_features[node_idx]
                feat[piece.piece_type.value] = 1.0
                feat[8 + piece.color.value] = 1.0
                feat[10] = 1.0 if height == 0 else 0.0
                feat[11] = 1.0 if is_top else 0.0
                feat[12] = stack_height * 0.25
                feat[13] = 1.0 if piece.piece_type == PieceType.QUEEN else 0.0
                feat[14] = queen_surround_w if piece.color == Color.WHITE else queen_surround_b
                feat[15] = occ_count / 6.0
                if is_top:
                    for d, occupied in enumerate(occ_by_dir):
                        feat[16 + d] = 0.0 if occupied else 1.0
                    feat[25] = 1.0 if (pq, pr) in move_sources else 0.0
                    if stack_height == 1 and pos in articulation_positions:
                        feat[26] = 1.0
                feat[24] = height * 0.25

                node_positions[board_node_count, 0] = row
                node_positions[board_node_count, 1] = col
                node_piece_types[board_node_count] = piece.piece_type.value
                pos_nodes.append(node_idx)
                board_node_count += 1
                node_idx += 1

            key = (pq, pr)
            top_node_for_pos[key] = pos_nodes[-1]
            stack_nodes[key] = pos_nodes

        num_piece_nodes = node_idx

        for color in (Color.WHITE, Color.BLACK):
            counts: dict[PieceType, int] = defaultdict(int)
            for piece in game_state.hand(color):
                counts[piece.piece_type] += 1
            for piece_type, count in counts.items():
                feat = node_features[node_idx]
                feat[piece_type.value] = 1.0
                feat[8 + color.value] = 1.0
                feat[13] = 1.0 if piece_type == PieceType.QUEEN else 0.0
                feat[22] = 1.0
                feat[23] = count / piece_type.count_per_player
                node_idx += 1

        for (pos_q, pos_r), src_idx in top_node_for_pos.items():
            for dq, dr in self.offsets:
                dst_idx = top_node_for_pos.get((pos_q + dq, pos_r + dr))
                if dst_idx is not None:
                    n_edges = self._write_spatial_edge(
                        edge_index, edge_features, n_edges, src_idx, dst_idx, dq, dr,
                    )

        stacked_bucket = self.edge_feat_dim - 1
        for nodes in stack_nodes.values():
            for i in range(len(nodes) - 1):
                lower = nodes[i]
                upper = nodes[i + 1]
                edge_index[0, n_edges] = lower
                edge_index[1, n_edges] = upper
                edge_features[n_edges, stacked_bucket] = 1.0
                n_edges += 1
                edge_index[0, n_edges] = upper
                edge_index[1, n_edges] = lower
                edge_features[n_edges, stacked_bucket] = 1.0
                n_edges += 1

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

        return HybridGraph(
            node_features=node_features[:node_idx],
            edge_index=edge_index[:, :n_edges] if n_edges else np.zeros((2, 0), dtype=np.int64),
            edge_features=(
                edge_features[:n_edges]
                if n_edges
                else np.zeros((0, self.edge_feat_dim), dtype=np.float32)
            ),
            global_features=global_features,
            num_piece_nodes=num_piece_nodes,
            node_positions=(
                node_positions[:board_node_count]
                if board_node_count
                else np.zeros((0, 2), dtype=np.int32)
            ),
            node_piece_types=(
                node_piece_types[:board_node_count]
                if board_node_count
                else np.zeros((0,), dtype=np.int32)
            ),
        )
