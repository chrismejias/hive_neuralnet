"""Tests for KataGo-inspired auxiliary heads, loss, and playout cap."""

import numpy as np
import pytest
import torch

from hive_engine.game_state import GameState, GameResult
from hive_engine.neural_net import compute_gnn_loss
from hive_engine.pieces import Color, PieceType

from hive_gnn.gnn_encoder import GNNEncoder
from hive_gnn.gnn_net import (
    GNNNetConfig,
    MobilityHead,
    QueenSurroundHead,
    FinalMobilityHead,
    HiveGNN,
)
from hive_gnn.gnn_replay_buffer import (
    GNNTrainingExample,
    GNNTrainingBatch,
    GraphReplayBuffer,
)
from hive_gnn.graph_types import (
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    HiveGraph,
    HiveGraphBatch,
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


def _make_example(num_piece_nodes: int = 3) -> GNNTrainingExample:
    """Create a valid GNNTrainingExample with all auxiliary fields."""
    graph = _make_graph(num_piece_nodes=num_piece_nodes)
    return GNNTrainingExample(
        graph=graph,
        policy_target=np.random.dirichlet(np.ones(29914)).astype(np.float32),
        value_target=np.random.choice([-1.0, 0.0, 1.0]),
        mobility_target=np.random.randint(0, 2, size=num_piece_nodes).astype(np.float32),
        queen_surround_target=np.random.randint(0, 2, size=(num_piece_nodes, 2)).astype(np.float32),
        queen_surround_mask=np.array([1.0, 1.0], dtype=np.float32),
        final_mobility_target=np.random.randint(0, 2, size=num_piece_nodes).astype(np.float32),
        use_for_value=True,
    )


# ---------------------------------------------------------------------------
# Auxiliary head output shapes
# ---------------------------------------------------------------------------


class TestMobilityHead:
    def test_output_shape(self):
        head = MobilityHead(hidden_dim=64)
        x = torch.randn(10, 64)
        out = head(x)
        assert out.shape == (10, 1)

    def test_single_node(self):
        head = MobilityHead(hidden_dim=32)
        x = torch.randn(1, 32)
        out = head(x)
        assert out.shape == (1, 1)


class TestQueenSurroundHead:
    def test_output_shape(self):
        head = QueenSurroundHead(hidden_dim=64)
        x = torch.randn(10, 64)
        out = head(x)
        assert out.shape == (10, 2)


class TestFinalMobilityHead:
    def test_output_shape(self):
        head = FinalMobilityHead(hidden_dim=64)
        x = torch.randn(10, 64)
        out = head(x)
        assert out.shape == (10, 1)


# ---------------------------------------------------------------------------
# HiveGNN forward with auxiliary heads
# ---------------------------------------------------------------------------


class TestHiveGNNForward:
    def test_forward_returns_three_tuple(self):
        config = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(config)
        graph = _make_graph(num_piece_nodes=4, num_hand_nodes=2, num_edges=6)
        batch = HiveGraphBatch.collate([graph])
        result = net(batch)
        assert len(result) == 3
        policy, value, aux = result
        assert policy.shape == (1, 29914)
        assert value.shape == (1, 1)
        assert "mobility_logits" in aux
        assert "queen_surround_logits" in aux
        assert "final_mobility_logits" in aux
        assert aux["mobility_logits"].shape == (4, 1)
        assert aux["queen_surround_logits"].shape == (4, 2)
        assert aux["final_mobility_logits"].shape == (4, 1)

    def test_forward_aux_disabled(self):
        config = GNNNetConfig(
            hidden_dim=32, num_mp_layers=2,
            aux_mobility_enabled=False,
            aux_queen_surround_enabled=False,
            aux_final_mobility_enabled=False,
        )
        net = HiveGNN(config)
        graph = _make_graph()
        batch = HiveGraphBatch.collate([graph])
        policy, value, aux = net(batch)
        assert len(aux) == 0

    def test_forward_batched(self):
        config = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(config)
        g1 = _make_graph(num_piece_nodes=3, num_hand_nodes=1, num_edges=4)
        g2 = _make_graph(num_piece_nodes=5, num_hand_nodes=2, num_edges=8)
        batch = HiveGraphBatch.collate([g1, g2])
        policy, value, aux = net(batch)
        assert policy.shape == (2, 29914)
        assert value.shape == (2, 1)
        total_piece = 3 + 5
        assert aux["mobility_logits"].shape == (total_piece, 1)
        assert aux["queen_surround_logits"].shape == (total_piece, 2)
        assert aux["final_mobility_logits"].shape == (total_piece, 1)


class TestPredict:
    def test_predict_still_works(self):
        """predict() should work unchanged (ignores aux outputs)."""
        config = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(config)
        graph = _make_graph(num_piece_nodes=3)
        mask = np.zeros(29914, dtype=np.float32)
        mask[:10] = 1.0
        probs, value = net.predict(graph, mask)
        assert probs.shape == (29914,)
        assert -1.0 <= value <= 1.0
        assert abs(probs[:10].sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class TestComputeGNNLoss:
    def _make_loss_inputs(self, batch_size=2, total_piece_nodes=6):
        policy_logits = torch.randn(batch_size, 29914)
        value_pred = torch.randn(batch_size, 1)
        target_policy = torch.softmax(torch.randn(batch_size, 29914), dim=1)
        target_value = torch.randn(batch_size, 1).clamp(-1, 1)

        mob_logits = torch.randn(total_piece_nodes, 1)
        qs_logits = torch.randn(total_piece_nodes, 2)
        fm_logits = torch.randn(total_piece_nodes, 1)
        aux_outputs = {
            "mobility_logits": mob_logits,
            "queen_surround_logits": qs_logits,
            "final_mobility_logits": fm_logits,
        }

        mobility_targets = torch.randint(0, 2, (total_piece_nodes,)).float()
        qs_targets = torch.randint(0, 2, (total_piece_nodes, 2)).float()
        qs_mask = torch.ones(batch_size, 2)
        fm_targets = torch.randint(0, 2, (total_piece_nodes,)).float()
        value_mask = torch.ones(batch_size)
        piece_node_batch = torch.tensor([0, 0, 0, 1, 1, 1])

        return (
            policy_logits, value_pred, target_policy, target_value,
            aux_outputs, mobility_targets, qs_targets, qs_mask,
            fm_targets, value_mask, piece_node_batch,
        )

    def test_all_losses_present(self):
        args = self._make_loss_inputs()
        total, loss_dict = compute_gnn_loss(*args)
        assert "policy_loss" in loss_dict
        assert "value_loss" in loss_dict
        assert "mobility_loss" in loss_dict
        assert "queen_surround_loss" in loss_dict
        assert "final_mobility_loss" in loss_dict
        assert "total_loss" in loss_dict
        for v in loss_dict.values():
            assert v.item() > 0 or v.item() == 0  # not NaN
            assert not torch.isnan(v)

    def test_value_mask_zeros_out_value_loss(self):
        args = list(self._make_loss_inputs())
        args[9] = torch.zeros(2)  # value_mask all zeros
        total, loss_dict = compute_gnn_loss(*args)
        assert loss_dict["value_loss"].item() == 0.0

    def test_queen_surround_mask_zeros(self):
        args = list(self._make_loss_inputs())
        args[7] = torch.zeros(2, 2)  # qs_mask all zeros
        total, loss_dict = compute_gnn_loss(*args)
        assert loss_dict["queen_surround_loss"].item() == 0.0

    def test_no_aux_outputs(self):
        args = list(self._make_loss_inputs())
        args[4] = {}  # empty aux_outputs
        total, loss_dict = compute_gnn_loss(*args)
        assert "mobility_loss" not in loss_dict
        assert "queen_surround_loss" not in loss_dict
        assert "final_mobility_loss" not in loss_dict


# ---------------------------------------------------------------------------
# Replay buffer with new fields
# ---------------------------------------------------------------------------


class TestReplayBufferAux:
    def test_sample_batch_returns_training_batch(self):
        buf = GraphReplayBuffer(max_size=100)
        for _ in range(10):
            buf.add_examples([_make_example(num_piece_nodes=3)])
        batch = buf.sample_batch(4)
        assert isinstance(batch, GNNTrainingBatch)
        assert batch.policy_targets.shape[1] == 29914
        assert batch.value_targets.shape[1] == 1
        assert batch.mobility_targets.ndim == 1
        assert batch.queen_surround_targets.ndim == 2
        assert batch.queen_surround_targets.shape[1] == 2
        assert batch.final_mobility_targets.ndim == 1
        assert batch.queen_surround_mask.shape[1] == 2
        assert batch.value_mask.ndim == 1

    def test_variable_piece_node_counts(self):
        """Examples with different num_piece_nodes should collate correctly."""
        buf = GraphReplayBuffer(max_size=100)
        buf.add_examples([_make_example(num_piece_nodes=2)])
        buf.add_examples([_make_example(num_piece_nodes=5)])
        buf.add_examples([_make_example(num_piece_nodes=3)])
        batch = buf.sample_batch(3)
        total_piece = batch.graph_batch.num_piece_nodes
        assert batch.mobility_targets.shape[0] == total_piece
        assert batch.queen_surround_targets.shape[0] == total_piece
        assert batch.final_mobility_targets.shape[0] == total_piece

    def test_to_device(self):
        buf = GraphReplayBuffer(max_size=100)
        buf.add_examples([_make_example()])
        batch = buf.sample_batch(1)
        batch_cpu = batch.to(torch.device("cpu"))
        assert batch_cpu.mobility_targets.device.type == "cpu"
        assert batch_cpu.final_mobility_targets.device.type == "cpu"


# ---------------------------------------------------------------------------
# Gradients flow through aux heads
# ---------------------------------------------------------------------------


class TestGradients:
    def test_gradients_flow_through_all_aux_heads(self):
        config = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        net = HiveGNN(config)
        graph = _make_graph(num_piece_nodes=4, num_hand_nodes=2, num_edges=6)
        batch = HiveGraphBatch.collate([graph])

        policy, value, aux = net(batch)

        # Create dummy targets
        total_piece = 4
        mobility_targets = torch.randint(0, 2, (total_piece,)).float()
        qs_targets = torch.randint(0, 2, (total_piece, 2)).float()
        fm_targets = torch.randint(0, 2, (total_piece,)).float()
        qs_mask = torch.ones(1, 2)
        value_mask = torch.ones(1)
        target_policy = torch.softmax(torch.randn(1, 29914), dim=1)
        target_value = torch.zeros(1, 1)

        total_loss, _ = compute_gnn_loss(
            policy, value, target_policy, target_value,
            aux, mobility_targets, qs_targets, qs_mask,
            fm_targets, value_mask, batch.piece_node_batch,
        )
        total_loss.backward()

        # Check that auxiliary head parameters received gradients
        for name, param in net.named_parameters():
            if "mobility_head" in name or "queen_surround_head" in name or "final_mobility_head" in name:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ---------------------------------------------------------------------------
# Mobility target from real game state
# ---------------------------------------------------------------------------


class TestMobilityTarget:
    def test_initial_position_no_mobility(self):
        """At move 0, no pieces on board → all mobility zeros."""
        from hive_gnn.gnn_trainer import GNNTrainer, GNNTrainConfig

        trainer = GNNTrainer(
            config=GNNTrainConfig(device="cpu"),
            net_config=GNNNetConfig(hidden_dim=32, num_mp_layers=2),
        )
        game = GameState()
        encoder = GNNEncoder()
        graph = encoder.encode_state(game)
        mob = trainer._compute_mobility_target(game, graph)
        # No pieces on board at start → no piece nodes → empty array
        assert mob.shape[0] == graph.num_piece_nodes
        assert mob.shape[0] == 0  # no pieces placed yet


# ---------------------------------------------------------------------------
# End-to-end mini training
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.slow
    def test_mini_training_completes(self):
        """One iteration with 2 games and minimal sims should run without error."""
        from hive_gnn.gnn_trainer import GNNTrainer, GNNTrainConfig

        config = GNNTrainConfig(
            num_iterations=1,
            games_per_iteration=2,
            num_epochs=1,
            batch_size=4,
            mcts_simulations=3,
            arena_games=2,
            arena_mcts_simulations=3,
            max_game_length=10,
            device="cpu",
            use_amp=False,
            checkpoint_dir="test_checkpoints_aux",
        )
        net_config = GNNNetConfig(hidden_dim=32, num_mp_layers=2)
        trainer = GNNTrainer(config=config, net_config=net_config)
        trainer.run()

        # Clean up
        import shutil
        shutil.rmtree("test_checkpoints_aux", ignore_errors=True)
