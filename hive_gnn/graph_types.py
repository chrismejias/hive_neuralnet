"""Graph data structures for the Hive GNN encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_FEAT_DIM = 21
EDGE_FEAT_DIM = 9
GLOBAL_FEAT_DIM = 6

# ---------------------------------------------------------------------------
# Single-graph representation (numpy, CPU)
# ---------------------------------------------------------------------------


@dataclass
class HiveGraph:
    """Numpy-based graph representation for a single Hive game state.

    The first ``num_piece_nodes`` rows of ``node_features`` correspond to
    pieces on the board; the remaining rows are hand (off-board) nodes.
    """

    node_features: np.ndarray       # (N, 21)  float32
    edge_index: np.ndarray          # (2, E)   int64  – COO edge list
    edge_features: np.ndarray       # (E, 9)   float32
    global_features: np.ndarray     # (6,)     float32
    num_piece_nodes: int            # count of board-piece nodes
    node_positions: np.ndarray      # (num_piece_nodes, 2)  int32  – (row, col)
    node_piece_types: np.ndarray    # (num_piece_nodes,)    int32


# ---------------------------------------------------------------------------
# Batched graph representation (torch, GPU-ready)
# ---------------------------------------------------------------------------


@dataclass
class HiveGraphBatch:
    """Torch-based batched graph representation for GPU training.

    Created via :meth:`collate` from a list of :class:`HiveGraph` objects.
    """

    node_features: torch.Tensor         # (total_N, 21)  float32
    edge_index: torch.Tensor            # (2, total_E)   int64
    edge_features: torch.Tensor         # (total_E, 9)   float32
    global_features: torch.Tensor       # (B, 6)         float32
    num_piece_nodes: int                # total piece nodes across the batch
    node_positions: torch.Tensor        # (total_piece_nodes, 2) int32
    node_piece_types: torch.Tensor      # (total_piece_nodes,)   int32
    batch: torch.Tensor                 # (total_N,)     int64
    piece_node_batch: torch.Tensor      # (total_piece_nodes,) int64

    # ---- factory -----------------------------------------------------------

    @staticmethod
    def collate(graphs: list[HiveGraph]) -> HiveGraphBatch:
        """Merge multiple :class:`HiveGraph` instances into one batch.

        Edge indices are offset by the cumulative node count so that they
        reference the correct rows in the concatenated node-feature matrix.
        """
        node_features_list: list[torch.Tensor] = []
        edge_index_list: list[torch.Tensor] = []
        edge_features_list: list[torch.Tensor] = []
        global_features_list: list[torch.Tensor] = []
        node_positions_list: list[torch.Tensor] = []
        node_piece_types_list: list[torch.Tensor] = []
        batch_list: list[torch.Tensor] = []
        piece_node_batch_list: list[torch.Tensor] = []

        cumulative_nodes = 0

        for idx, g in enumerate(graphs):
            num_nodes = g.node_features.shape[0]

            node_features_list.append(
                torch.from_numpy(g.node_features)
            )

            # Offset edge indices by cumulative node count
            ei = torch.from_numpy(g.edge_index)
            if ei.numel() > 0:
                ei = ei + cumulative_nodes
            edge_index_list.append(ei)

            edge_features_list.append(
                torch.from_numpy(g.edge_features)
            )

            global_features_list.append(
                torch.from_numpy(g.global_features)
            )

            node_positions_list.append(
                torch.from_numpy(g.node_positions)
            )

            node_piece_types_list.append(
                torch.from_numpy(g.node_piece_types)
            )

            # Batch vector: every node in this graph maps to *idx*
            batch_list.append(
                torch.full((num_nodes,), idx, dtype=torch.int64)
            )

            # Piece-node batch vector
            piece_node_batch_list.append(
                torch.full((g.num_piece_nodes,), idx, dtype=torch.int64)
            )

            cumulative_nodes += num_nodes

        # Concatenate / stack
        node_features = torch.cat(node_features_list, dim=0)

        if any(ei.numel() > 0 for ei in edge_index_list):
            edge_index = torch.cat(
                [ei for ei in edge_index_list if ei.numel() > 0], dim=1
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.int64)

        if any(ef.shape[0] > 0 for ef in edge_features_list):
            edge_features = torch.cat(
                [ef for ef in edge_features_list if ef.shape[0] > 0], dim=0
            )
        else:
            edge_features = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)

        global_features = torch.stack(global_features_list, dim=0)  # (B, 6)

        if any(np_.shape[0] > 0 for np_ in node_positions_list):
            node_positions = torch.cat(
                [np_ for np_ in node_positions_list if np_.shape[0] > 0], dim=0
            )
        else:
            node_positions = torch.zeros((0, 2), dtype=torch.int32)

        if any(pt.shape[0] > 0 for pt in node_piece_types_list):
            node_piece_types = torch.cat(
                [pt for pt in node_piece_types_list if pt.shape[0] > 0], dim=0
            )
        else:
            node_piece_types = torch.zeros((0,), dtype=torch.int32)

        batch = torch.cat(batch_list, dim=0)

        if any(pb.shape[0] > 0 for pb in piece_node_batch_list):
            piece_node_batch = torch.cat(
                [pb for pb in piece_node_batch_list if pb.shape[0] > 0], dim=0
            )
        else:
            piece_node_batch = torch.zeros((0,), dtype=torch.int64)

        total_piece_nodes = sum(g.num_piece_nodes for g in graphs)

        return HiveGraphBatch(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=global_features,
            num_piece_nodes=total_piece_nodes,
            node_positions=node_positions,
            node_piece_types=node_piece_types,
            batch=batch,
            piece_node_batch=piece_node_batch,
        )

    # ---- device transfer ---------------------------------------------------

    def to(self, device: torch.device) -> HiveGraphBatch:
        """Return a new :class:`HiveGraphBatch` with all tensors on *device*."""
        return HiveGraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            global_features=self.global_features.to(device),
            num_piece_nodes=self.num_piece_nodes,
            node_positions=self.node_positions.to(device),
            node_piece_types=self.node_piece_types.to(device),
            batch=self.batch.to(device),
            piece_node_batch=self.piece_node_batch.to(device),
        )
