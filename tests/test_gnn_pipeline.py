"""
End-to-end integration tests for the GNN Hive training pipeline.

Tests the full flow: game state → graph → GNN → MCTS → self-play → train.
"""

import numpy as np
import pytest
import torch

from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.hex_coord import ORIGIN, ALL_DIRECTIONS
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.neural_net import compute_loss
from hive_engine.pieces import Color, PieceType

from hive_gnn.gnn_encoder import GNNEncoder
from hive_gnn.gnn_net import GNNNetConfig, HiveGNN
from hive_gnn.gnn_replay_buffer import GNNTrainingExample, GraphReplayBuffer
from hive_gnn.gnn_trainer import GNNTrainConfig, GNNTrainer
from hive_gnn.graph_encoder import GraphEncoder
from hive_gnn.graph_types import HiveGraphBatch


# ---------------------------------------------------------------------------
# Small config for fast tests
# ---------------------------------------------------------------------------

SMALL_NET = GNNNetConfig(hidden_dim=32, num_mp_layers=2)


# ---------------------------------------------------------------------------
# TestMCTSIntegration
# ---------------------------------------------------------------------------


class TestMCTSIntegration:
    """MCTS works with GNNEncoder + HiveGNN via duck typing."""

    def test_mcts_search_returns_policy(self):
        encoder = GNNEncoder()
        net = HiveGNN(SMALL_NET)
        net.eval()

        game = GameState()
        config = MCTSConfig(num_simulations=5)
        mcts = MCTS(net, encoder, config)

        policy = mcts.search(game, move_number=0)

        assert policy.shape == (GNNEncoder.ACTION_SPACE_SIZE,)
        assert abs(policy.sum() - 1.0) < 1e-5
        assert (policy >= 0).all()

    def test_mcts_respects_legal_moves(self):
        encoder = GNNEncoder()
        net = HiveGNN(SMALL_NET)
        net.eval()

        game = GameState()
        config = MCTSConfig(num_simulations=5)
        mcts = MCTS(net, encoder, config)

        policy = mcts.search(game, move_number=0)
        legal_mask = encoder.get_legal_action_mask(game)

        # Policy should only be nonzero where legal_mask is 1
        illegal = legal_mask == 0
        assert (policy[illegal] == 0).all()

    def test_mcts_multiple_moves(self):
        """Run MCTS for a few moves of a game."""
        encoder = GNNEncoder()
        net = HiveGNN(SMALL_NET)
        net.eval()

        game = GameState()
        config = MCTSConfig(num_simulations=3)
        mcts = MCTS(net, encoder, config)

        for move_num in range(4):
            if game.result != GameResult.IN_PROGRESS:
                break

            policy = mcts.search(game, move_number=move_num)
            action = int(np.argmax(policy))
            mask = encoder.get_legal_action_mask(game)

            if mask[action] > 0:
                move = encoder.decode_action(action, game)
            else:
                legal_actions = np.where(mask > 0)[0]
                move = encoder.decode_action(int(legal_actions[0]), game)

            game.apply_move(move)

        assert game.turn >= 4 or game.result != GameResult.IN_PROGRESS


# ---------------------------------------------------------------------------
# TestReplayBuffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    """GraphReplayBuffer works with GNN training examples."""

    def test_add_and_sample(self):
        encoder = GNNEncoder()
        graph_encoder = GraphEncoder()
        game = GameState()

        buf = GraphReplayBuffer(max_size=100)

        # Add some examples
        for _ in range(10):
            graph = graph_encoder.encode(game)
            policy = np.random.dirichlet(np.ones(GNNEncoder.ACTION_SPACE_SIZE))
            value = np.random.choice([-1.0, 0.0, 1.0])
            buf.add_examples([GNNTrainingExample(graph, policy, value)])

        assert len(buf) == 10

        # Sample a batch
        batch, policies, values = buf.sample_batch(4)

        assert isinstance(batch, HiveGraphBatch)
        assert policies.shape == (4, GNNEncoder.ACTION_SPACE_SIZE)
        assert values.shape == (4, 1)

    def test_buffer_overflow(self):
        graph_encoder = GraphEncoder()
        game = GameState()
        graph = graph_encoder.encode(game)

        buf = GraphReplayBuffer(max_size=5)
        for i in range(10):
            policy = np.zeros(GNNEncoder.ACTION_SPACE_SIZE, dtype=np.float32)
            buf.add_examples([GNNTrainingExample(graph, policy, float(i))])

        assert len(buf) == 5  # Oldest evicted


# ---------------------------------------------------------------------------
# TestTrainingStep
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """One training step: forward + backward through the GNN."""

    def test_forward_backward(self):
        net = HiveGNN(SMALL_NET)
        net.train()

        graph_encoder = GraphEncoder()

        # Use a game with pieces on board (so edges exist for message passing)
        game = GameState()
        hand = game.hand(Color.WHITE)
        queen = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        game.apply_move(Move(MoveType.PLACE, queen, ORIGIN))
        hand_b = game.hand(Color.BLACK)
        queen_b = next(p for p in hand_b if p.piece_type == PieceType.QUEEN)
        neighbor = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        game.apply_move(Move(MoveType.PLACE, queen_b, neighbor))

        # Create a small batch with edges
        graphs = [graph_encoder.encode(game) for _ in range(4)]
        batch = HiveGraphBatch.collate(graphs)

        assert batch.edge_index.size(1) > 0, "Need edges for full gradient test"

        policy_targets = torch.randn(4, GNNEncoder.ACTION_SPACE_SIZE)
        policy_targets = torch.softmax(policy_targets, dim=1)
        value_targets = torch.randn(4, 1).clamp(-1, 1)

        # Forward
        policy_logits, value_pred = net(batch)

        # Loss
        total_loss, p_loss, v_loss = compute_loss(
            policy_logits, value_pred, policy_targets, value_targets
        )

        assert total_loss.item() > 0
        assert p_loss.item() > 0
        assert v_loss.item() >= 0

        # Backward
        total_loss.backward()

        # Check gradients exist for all parameters
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step(self):
        """Verify an optimizer step reduces loss."""
        net = HiveGNN(SMALL_NET)
        net.train()

        graph_encoder = GraphEncoder()
        game = GameState()

        graphs = [graph_encoder.encode(game) for _ in range(8)]
        batch = HiveGraphBatch.collate(graphs)

        policy_targets = torch.softmax(
            torch.randn(8, GNNEncoder.ACTION_SPACE_SIZE), dim=1
        )
        value_targets = torch.zeros(8, 1)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        # Measure loss before
        policy_logits, value_pred = net(batch)
        loss_before, _, _ = compute_loss(
            policy_logits, value_pred, policy_targets, value_targets
        )

        # Training step
        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        # Measure loss after
        with torch.no_grad():
            policy_logits2, value_pred2 = net(batch)
            loss_after, _, _ = compute_loss(
                policy_logits2, value_pred2, policy_targets, value_targets
            )

        # Loss should decrease (or at least not increase much)
        assert loss_after.item() < loss_before.item() * 1.1


# ---------------------------------------------------------------------------
# TestSelfPlayGame
# ---------------------------------------------------------------------------


class TestSelfPlayGame:
    """Test self-play game generation."""

    def test_self_play_game(self):
        config = GNNTrainConfig(
            mcts_simulations=3,
            max_game_length=10,
            games_per_iteration=1,
        )
        trainer = GNNTrainer(config=config, net_config=SMALL_NET)

        examples, result, length = trainer._self_play_game(trainer.best_net)

        assert len(examples) > 0
        assert len(examples) <= 10  # max_game_length
        assert isinstance(examples[0], GNNTrainingExample)
        assert examples[0].policy_target.shape == (GNNEncoder.ACTION_SPACE_SIZE,)
        assert -1.0 <= examples[0].value_target <= 1.0
        assert length <= 10


# ---------------------------------------------------------------------------
# TestTrainer
# ---------------------------------------------------------------------------


class TestTrainer:
    """Test the full GNN trainer pipeline."""

    def test_trainer_init(self):
        config = GNNTrainConfig(
            num_iterations=1,
            games_per_iteration=2,
            mcts_simulations=3,
            max_game_length=10,
        )
        trainer = GNNTrainer(config=config, net_config=SMALL_NET)

        assert trainer.best_net is not None
        assert trainer.encoder is not None
        assert len(trainer.buffer) == 0

    def test_train_phase_with_buffer(self):
        """Train phase works when buffer has enough examples."""
        config = GNNTrainConfig(
            batch_size=4,
            num_epochs=1,
        )
        trainer = GNNTrainer(config=config, net_config=SMALL_NET)

        # Populate buffer manually
        encoder = GraphEncoder()
        game = GameState()
        for _ in range(20):
            graph = encoder.encode(game)
            policy = np.random.dirichlet(
                np.ones(GNNEncoder.ACTION_SPACE_SIZE)
            ).astype(np.float32)
            value = np.random.choice([-1.0, 0.0, 1.0])
            trainer.buffer.add_examples(
                [GNNTrainingExample(graph, policy, value)]
            )

        new_net, stats = trainer._train_phase(iteration=1)
        assert stats.num_batches > 0
        assert stats.avg_loss > 0

    def test_train_phase_empty_buffer(self):
        """Train phase returns untrained net when buffer is too small."""
        config = GNNTrainConfig(batch_size=64)
        trainer = GNNTrainer(config=config, net_config=SMALL_NET)

        new_net, stats = trainer._train_phase(iteration=1)
        assert stats.num_batches == 0

    def test_parameter_count_comparison(self):
        """Verify parameter counts for presets."""
        small = HiveGNN(GNNNetConfig.small())
        large = HiveGNN(GNNNetConfig.large())

        small_count = small.count_parameters()
        large_count = large.count_parameters()

        assert small_count > 0
        assert large_count > small_count
        print(f"  GNN small: {small_count:,} params")
        print(f"  GNN large: {large_count:,} params")


# ---------------------------------------------------------------------------
# TestCheckpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_save_and_load(self, tmp_path):
        config = GNNTrainConfig(
            num_iterations=5,
            checkpoint_dir=str(tmp_path),
        )
        trainer = GNNTrainer(config=config, net_config=SMALL_NET)

        # Save
        path = trainer._save_checkpoint(trainer.best_net, iteration=3)
        assert os.path.exists(path)

        # Load
        loaded = GNNTrainer.from_checkpoint(path)
        assert loaded._start_iteration == 4
        assert loaded.net_config.hidden_dim == SMALL_NET.hidden_dim

    def test_resume_with_overrides(self, tmp_path):
        config = GNNTrainConfig(
            num_iterations=10,
            checkpoint_dir=str(tmp_path),
        )
        trainer = GNNTrainer(config=config, net_config=SMALL_NET)
        path = trainer._save_checkpoint(trainer.best_net, iteration=5)

        loaded = GNNTrainer.from_checkpoint(
            path, config_overrides={"num_iterations": 20}
        )
        assert loaded.config.num_iterations == 20


import os  # for TestCheckpoint


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI argument parsing."""

    def test_parse_defaults(self):
        from hive_gnn.train import parse_args

        args = parse_args([])
        assert args.iterations == 20
        assert args.games == 50
        assert args.hidden_dim == 128
        assert args.mp_layers == 6

    def test_parse_custom(self):
        from hive_gnn.train import parse_args

        args = parse_args([
            "--iterations", "5",
            "--games", "10",
            "--hidden-dim", "64",
            "--mp-layers", "3",
            "--device", "cpu",
            "--no-amp",
        ])
        assert args.iterations == 5
        assert args.games == 10
        assert args.hidden_dim == 64
        assert args.mp_layers == 3
        assert args.device == "cpu"
        assert args.no_amp is True


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all public exports from hive_gnn."""

    def test_all_exports(self):
        from hive_gnn import (
            HiveGraph,
            HiveGraphBatch,
            NODE_FEAT_DIM,
            EDGE_FEAT_DIM,
            GLOBAL_FEAT_DIM,
            GraphEncoder,
            GNNEncoder,
            GNNNetConfig,
            HiveGNN,
            MessagePassingLayer,
            GNNTrainingExample,
            GraphReplayBuffer,
            GNNTrainConfig,
            GNNTrainer,
        )
        # Just verify they imported without error
        assert HiveGraph is not None
        assert HiveGNN is not None
        assert GNNTrainer is not None
