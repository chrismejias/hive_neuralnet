"""Tests for hive_gnn.graph_types data structures."""

import numpy as np
import pytest
import torch

from hive_gnn.graph_types import (
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    NODE_FEAT_DIM,
    HiveGraph,
    HiveGraphBatch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    num_piece_nodes: int,
    num_hand_nodes: int,
    num_edges: int,
) -> HiveGraph:
    """Create a minimal HiveGraph with the given sizes."""
    num_nodes = num_piece_nodes + num_hand_nodes

    node_features = np.random.randn(num_nodes, NODE_FEAT_DIM).astype(np.float32)

    if num_edges > 0:
        src = np.random.randint(0, max(num_nodes, 1), size=num_edges).astype(np.int64)
        dst = np.random.randint(0, max(num_nodes, 1), size=num_edges).astype(np.int64)
        edge_index = np.stack([src, dst], axis=0)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    edge_features = np.random.randn(num_edges, EDGE_FEAT_DIM).astype(np.float32)
    global_features = np.random.randn(GLOBAL_FEAT_DIM).astype(np.float32)
    node_positions = np.random.randint(0, 20, size=(num_piece_nodes, 2)).astype(np.int32)
    node_piece_types = np.random.randint(0, 5, size=(num_piece_nodes,)).astype(np.int32)

    return HiveGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        global_features=global_features,
        num_piece_nodes=num_piece_nodes,
        node_positions=node_positions,
        node_piece_types=node_piece_types,
    )


# ---------------------------------------------------------------------------
# TestHiveGraph
# ---------------------------------------------------------------------------


class TestHiveGraph:
    """Tests for the numpy-based HiveGraph dataclass."""

    def test_create_simple(self):
        """Create a HiveGraph with 3 nodes (2 piece + 1 hand), 4 edges."""
        g = _make_graph(num_piece_nodes=2, num_hand_nodes=1, num_edges=4)

        assert g.node_features.shape == (3, NODE_FEAT_DIM)
        assert g.node_features.dtype == np.float32

        assert g.edge_index.shape == (2, 4)
        assert g.edge_index.dtype == np.int64

        assert g.edge_features.shape == (4, EDGE_FEAT_DIM)
        assert g.edge_features.dtype == np.float32

        assert g.global_features.shape == (GLOBAL_FEAT_DIM,)
        assert g.global_features.dtype == np.float32

        assert g.num_piece_nodes == 2

        assert g.node_positions.shape == (2, 2)
        assert g.node_positions.dtype == np.int32

        assert g.node_piece_types.shape == (2,)
        assert g.node_piece_types.dtype == np.int32

    def test_empty_graph(self):
        """0 piece nodes, 1 hand node, 0 edges -- all shapes must be valid."""
        g = _make_graph(num_piece_nodes=0, num_hand_nodes=1, num_edges=0)

        assert g.node_features.shape == (1, NODE_FEAT_DIM)
        assert g.edge_index.shape == (2, 0)
        assert g.edge_features.shape == (0, EDGE_FEAT_DIM)
        assert g.global_features.shape == (GLOBAL_FEAT_DIM,)
        assert g.num_piece_nodes == 0
        assert g.node_positions.shape == (0, 2)
        assert g.node_piece_types.shape == (0,)


# ---------------------------------------------------------------------------
# TestHiveGraphBatch
# ---------------------------------------------------------------------------


class TestHiveGraphBatch:
    """Tests for the torch-based HiveGraphBatch and its collate method."""

    def test_collate_single(self):
        """Collating a single graph should preserve shapes."""
        g = _make_graph(num_piece_nodes=3, num_hand_nodes=1, num_edges=5)
        batch = HiveGraphBatch.collate([g])

        assert batch.node_features.shape == (4, NODE_FEAT_DIM)
        assert batch.edge_index.shape == (2, 5)
        assert batch.edge_features.shape == (5, EDGE_FEAT_DIM)
        assert batch.global_features.shape == (1, GLOBAL_FEAT_DIM)
        assert batch.num_piece_nodes == 3
        assert batch.node_positions.shape == (3, 2)
        assert batch.node_piece_types.shape == (3,)
        assert batch.batch.shape == (4,)
        assert (batch.batch == 0).all()
        assert batch.piece_node_batch.shape == (3,)
        assert (batch.piece_node_batch == 0).all()

    def test_collate_multiple(self):
        """Collate 3 graphs with different sizes and verify offsets."""
        g0 = _make_graph(num_piece_nodes=2, num_hand_nodes=1, num_edges=3)  # 3 nodes
        g1 = _make_graph(num_piece_nodes=4, num_hand_nodes=2, num_edges=5)  # 6 nodes
        g2 = _make_graph(num_piece_nodes=1, num_hand_nodes=0, num_edges=2)  # 1 node

        batch = HiveGraphBatch.collate([g0, g1, g2])

        # Total nodes = 3 + 6 + 1 = 10
        assert batch.node_features.shape == (10, NODE_FEAT_DIM)

        # Total edges = 3 + 5 + 2 = 10
        assert batch.edge_index.shape == (2, 10)
        assert batch.edge_features.shape == (10, EDGE_FEAT_DIM)

        # Edge index offsets: g0 edges in [0,2], g1 edges offset by 3, g2 offset by 9
        # Verify g1 edges are offset by g0's node count (3)
        g1_edge_index_raw = torch.from_numpy(g1.edge_index)
        # The edges for g1 occupy indices 3..7 in the concatenated edge_index
        g1_edges_in_batch = batch.edge_index[:, 3:8]
        expected_g1_edges = g1_edge_index_raw + 3  # offset by g0 node count
        assert torch.equal(g1_edges_in_batch, expected_g1_edges)

        # Verify g2 edges are offset by g0+g1 node count (3+6=9)
        g2_edge_index_raw = torch.from_numpy(g2.edge_index)
        g2_edges_in_batch = batch.edge_index[:, 8:10]
        expected_g2_edges = g2_edge_index_raw + 9
        assert torch.equal(g2_edges_in_batch, expected_g2_edges)

        # Batch vector
        assert batch.batch.shape == (10,)
        assert (batch.batch[:3] == 0).all()
        assert (batch.batch[3:9] == 1).all()
        assert (batch.batch[9:10] == 2).all()

        # Global features stacked
        assert batch.global_features.shape == (3, GLOBAL_FEAT_DIM)

        # Node positions concatenated: 2 + 4 + 1 = 7
        assert batch.node_positions.shape == (7, 2)

        # Node piece types concatenated
        assert batch.node_piece_types.shape == (7,)

        # Piece-node batch vector
        assert batch.piece_node_batch.shape == (7,)
        assert (batch.piece_node_batch[:2] == 0).all()
        assert (batch.piece_node_batch[2:6] == 1).all()
        assert (batch.piece_node_batch[6:7] == 2).all()

    def test_collate_with_empty_edges(self):
        """A graph with zero edges should collate without errors."""
        g_normal = _make_graph(num_piece_nodes=2, num_hand_nodes=1, num_edges=4)
        g_empty = _make_graph(num_piece_nodes=1, num_hand_nodes=1, num_edges=0)

        batch = HiveGraphBatch.collate([g_normal, g_empty])

        # 3 + 2 = 5 nodes
        assert batch.node_features.shape == (5, NODE_FEAT_DIM)
        # Only g_normal's 4 edges
        assert batch.edge_index.shape == (2, 4)
        assert batch.edge_features.shape == (4, EDGE_FEAT_DIM)
        assert batch.global_features.shape == (2, GLOBAL_FEAT_DIM)
        assert batch.batch.shape == (5,)
        assert (batch.batch[:3] == 0).all()
        assert (batch.batch[3:5] == 1).all()

    def test_to_device(self):
        """Verify .to(device) moves all tensors to the target device."""
        g = _make_graph(num_piece_nodes=2, num_hand_nodes=1, num_edges=3)
        batch = HiveGraphBatch.collate([g])

        cpu = torch.device("cpu")
        batch_cpu = batch.to(cpu)

        assert batch_cpu.node_features.device == cpu
        assert batch_cpu.edge_index.device == cpu
        assert batch_cpu.edge_features.device == cpu
        assert batch_cpu.global_features.device == cpu
        assert batch_cpu.node_positions.device == cpu
        assert batch_cpu.node_piece_types.device == cpu
        assert batch_cpu.batch.device == cpu
        assert batch_cpu.piece_node_batch.device == cpu

        # Scalar field preserved
        assert batch_cpu.num_piece_nodes == 2
