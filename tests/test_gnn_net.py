"""Tests for hive_gnn.gnn_net — GNN network architecture."""

import numpy as np
import pytest
import torch

from hive_gnn.graph_types import (
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    HiveGraph,
    HiveGraphBatch,
)
from hive_gnn.gnn_net import (
    GNNNetConfig,
    MessagePassingLayer,
    HybridPolicyHead,
    ValueHead,
    HiveGNN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    num_piece_nodes: int = 3,
    num_hand_nodes: int = 2,
    num_edges: int = 4,
) -> HiveGraph:
    """Create a minimal HiveGraph with valid shapes."""
    num_nodes = num_piece_nodes + num_hand_nodes
    node_features = np.random.randn(num_nodes, NODE_FEAT_DIM).astype(np.float32)

    if num_edges > 0 and num_nodes > 0:
        src = np.random.randint(0, num_nodes, size=num_edges).astype(np.int64)
        dst = np.random.randint(0, num_nodes, size=num_edges).astype(np.int64)
        edge_index = np.stack([src, dst], axis=0)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    edge_features = np.random.randn(num_edges, EDGE_FEAT_DIM).astype(np.float32)
    global_features = np.random.randn(GLOBAL_FEAT_DIM).astype(np.float32)

    # Positions within the 13x13 grid
    node_positions = np.random.randint(0, 13, size=(num_piece_nodes, 2)).astype(np.int32)
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


def _make_batch(graphs: list[HiveGraph] | None = None) -> HiveGraphBatch:
    """Create a batched graph. If None, uses two default graphs."""
    if graphs is None:
        graphs = [_make_graph(3, 2, 4), _make_graph(2, 1, 2)]
    return HiveGraphBatch.collate(graphs)


# ---------------------------------------------------------------------------
# TestGNNNetConfig
# ---------------------------------------------------------------------------


class TestGNNNetConfig:
    def test_default_config(self):
        cfg = GNNNetConfig()
        assert cfg.hidden_dim == 128
        assert cfg.num_mp_layers == 6
        assert cfg.action_space_size == 29407
        assert cfg.board_size == 13

    def test_small_preset(self):
        cfg = GNNNetConfig.small()
        assert cfg.hidden_dim == 128
        assert cfg.num_mp_layers == 6

    def test_large_preset(self):
        cfg = GNNNetConfig.large()
        assert cfg.hidden_dim == 256
        assert cfg.num_mp_layers == 8


# ---------------------------------------------------------------------------
# TestMessagePassingLayer
# ---------------------------------------------------------------------------


class TestMessagePassingLayer:
    def test_output_shape(self):
        h_dim = 64
        layer = MessagePassingLayer(h_dim, EDGE_FEAT_DIM)
        node_feat = torch.randn(5, h_dim)
        edge_idx = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
        edge_feat = torch.randn(3, EDGE_FEAT_DIM)

        out = layer(node_feat, edge_idx, edge_feat)
        assert out.shape == (5, h_dim)

    def test_no_edges(self):
        h_dim = 64
        layer = MessagePassingLayer(h_dim, EDGE_FEAT_DIM)
        node_feat = torch.randn(3, h_dim)
        edge_idx = torch.zeros((2, 0), dtype=torch.int64)
        edge_feat = torch.zeros((0, EDGE_FEAT_DIM))

        out = layer(node_feat, edge_idx, edge_feat)
        assert out.shape == (3, h_dim)

    def test_gradients_flow(self):
        h_dim = 32
        layer = MessagePassingLayer(h_dim, EDGE_FEAT_DIM)
        node_feat = torch.randn(4, h_dim, requires_grad=True)
        edge_idx = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
        edge_feat = torch.randn(2, EDGE_FEAT_DIM)

        out = layer(node_feat, edge_idx, edge_feat)
        loss = out.sum()
        loss.backward()
        assert node_feat.grad is not None
        assert node_feat.grad.shape == (4, h_dim)


# ---------------------------------------------------------------------------
# TestValueHead
# ---------------------------------------------------------------------------


class TestValueHead:
    def test_output_shape(self):
        h_dim = 64
        head = ValueHead(h_dim, GLOBAL_FEAT_DIM)
        node_emb = torch.randn(8, h_dim)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int64)
        global_feat = torch.randn(2, GLOBAL_FEAT_DIM)

        out = head(node_emb, batch, global_feat, batch_size=2)
        assert out.shape == (2, 1)

    def test_output_range(self):
        h_dim = 64
        head = ValueHead(h_dim, GLOBAL_FEAT_DIM)
        node_emb = torch.randn(5, h_dim)
        batch = torch.zeros(5, dtype=torch.int64)
        global_feat = torch.randn(1, GLOBAL_FEAT_DIM)

        out = head(node_emb, batch, global_feat, batch_size=1)
        assert -1.0 <= out.item() <= 1.0

    def test_single_node(self):
        h_dim = 32
        head = ValueHead(h_dim, GLOBAL_FEAT_DIM)
        node_emb = torch.randn(1, h_dim)
        batch = torch.zeros(1, dtype=torch.int64)
        global_feat = torch.randn(1, GLOBAL_FEAT_DIM)

        out = head(node_emb, batch, global_feat, batch_size=1)
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# TestHiveGNN
# ---------------------------------------------------------------------------


class TestHiveGNN:
    """Tests for the full HiveGNN network."""

    def test_forward_output_shapes(self):
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        batch = _make_batch()

        policy, value = net(batch)
        batch_size = batch.global_features.size(0)

        assert policy.shape == (batch_size, cfg.action_space_size)
        assert value.shape == (batch_size, 1)

    def test_forward_single_graph(self):
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        g = _make_graph(4, 2, 6)
        batch = HiveGraphBatch.collate([g])

        policy, value = net(batch)
        assert policy.shape == (1, cfg.action_space_size)
        assert value.shape == (1, 1)

    def test_forward_empty_edges(self):
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        g = _make_graph(num_piece_nodes=2, num_hand_nodes=3, num_edges=0)
        batch = HiveGraphBatch.collate([g])

        policy, value = net(batch)
        assert policy.shape == (1, cfg.action_space_size)
        assert value.shape == (1, 1)

    def test_forward_only_hand_nodes(self):
        """Test with no piece nodes (empty board)."""
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        g = _make_graph(num_piece_nodes=0, num_hand_nodes=10, num_edges=0)
        batch = HiveGraphBatch.collate([g])

        policy, value = net(batch)
        assert policy.shape == (1, cfg.action_space_size)
        assert value.shape == (1, 1)

    def test_value_in_range(self):
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        batch = _make_batch()

        _, value = net(batch)
        for v in value.flatten():
            assert -1.0 <= v.item() <= 1.0

    def test_gradients_flow_through(self):
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        batch = _make_batch()

        policy, value = net(batch)
        loss = policy.sum() + value.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_count_parameters(self):
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        count = net.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_predict_interface(self):
        """Test MCTS predict interface."""
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        g = _make_graph(3, 2, 4)
        legal_mask = np.zeros(cfg.action_space_size, dtype=np.float32)
        legal_mask[0] = 1.0
        legal_mask[100] = 1.0
        legal_mask[1000] = 1.0

        probs, val = net.predict(g, legal_mask)

        assert probs.shape == (cfg.action_space_size,)
        assert probs.dtype == np.float32
        # Probabilities sum to ~1 over legal actions
        assert abs(probs.sum() - 1.0) < 1e-5
        # Only legal actions have nonzero probability
        assert probs[0] > 0.0
        assert probs[100] > 0.0
        assert probs[1000] > 0.0
        # Illegal actions are zero
        assert probs[1] == 0.0

        assert isinstance(val, float)
        assert -1.0 <= val <= 1.0

    def test_predict_no_legal_moves(self):
        """Edge case: no legal moves."""
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        g = _make_graph(2, 1, 2)
        legal_mask = np.zeros(cfg.action_space_size, dtype=np.float32)

        probs, val = net.predict(g, legal_mask)
        assert probs.sum() == 0.0

    def test_default_config_instantiation(self):
        net = HiveGNN()
        assert net.config.hidden_dim == 128
        assert net.config.num_mp_layers == 6

    def test_batch_size_three(self):
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(cfg)
        graphs = [_make_graph(2, 1, 2), _make_graph(4, 3, 6), _make_graph(1, 1, 0)]
        batch = HiveGraphBatch.collate(graphs)

        policy, value = net(batch)
        assert policy.shape == (3, cfg.action_space_size)
        assert value.shape == (3, 1)


# ---------------------------------------------------------------------------
# TestHybridPolicyHead
# ---------------------------------------------------------------------------


class TestHybridPolicyHead:
    def test_forward_with_grid_shape(self):
        cfg = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        head = HybridPolicyHead(cfg)
        grid = torch.randn(2, 32, 13, 13)
        global_feat = torch.randn(2, GLOBAL_FEAT_DIM)

        out = head.forward_with_grid(grid, global_feat)
        assert out.shape == (2, cfg.action_space_size)

    def test_single_batch(self):
        cfg = GNNNetConfig(hidden_dim=64, num_mp_layers=2)
        head = HybridPolicyHead(cfg)
        grid = torch.randn(1, 64, 13, 13)
        global_feat = torch.randn(1, GLOBAL_FEAT_DIM)

        out = head.forward_with_grid(grid, global_feat)
        assert out.shape == (1, cfg.action_space_size)
