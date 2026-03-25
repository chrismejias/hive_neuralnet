"""
End-to-end integration tests for the NNUE Hive training pipeline.

Tests the full flow: game state -> features -> NNUE MLP -> MCTS -> self-play -> train.
"""

import os

import numpy as np
import pytest
import torch

from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.hex_coord import ORIGIN, ALL_DIRECTIONS
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.neural_net import compute_loss
from hive_engine.pieces import Color, PieceType

from hive_nnue.nnue_encoder import NNUEEncoder, FEATURE_DIM
from hive_nnue.nnue_net import NNUEConfig, HiveNNUE
from hive_nnue.nnue_replay_buffer import NNUETrainingExample, NNUEReplayBuffer
from hive_nnue.nnue_trainer import NNUETrainConfig, NNUETrainer


# ---------------------------------------------------------------------------
# Small config for fast tests
# ---------------------------------------------------------------------------

SMALL_NET = NNUEConfig(hidden_dims=[64, 32], dropout=0.0)


# ---------------------------------------------------------------------------
# TestMCTSIntegration
# ---------------------------------------------------------------------------


class TestMCTSIntegration:
    """MCTS works with NNUEEncoder + HiveNNUE via duck typing."""

    def test_mcts_search_returns_policy(self):
        encoder = NNUEEncoder()
        net = HiveNNUE(SMALL_NET)
        net.eval()

        game = GameState()
        config = MCTSConfig(num_simulations=5)
        mcts = MCTS(net, encoder, config)

        policy = mcts.search(game, move_number=0)

        assert policy.shape == (NNUEEncoder.ACTION_SPACE_SIZE,)
        assert abs(policy.sum() - 1.0) < 1e-5
        assert (policy >= 0).all()

    def test_mcts_respects_legal_moves(self):
        encoder = NNUEEncoder()
        net = HiveNNUE(SMALL_NET)
        net.eval()

        game = GameState()
        config = MCTSConfig(num_simulations=5)
        mcts = MCTS(net, encoder, config)

        policy = mcts.search(game, move_number=0)
        legal_mask = encoder.get_legal_action_mask(game)

        illegal = legal_mask == 0
        assert (policy[illegal] == 0).all()

    def test_mcts_multiple_moves(self):
        """Run MCTS for a few moves of a game."""
        encoder = NNUEEncoder()
        net = HiveNNUE(SMALL_NET)
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
    """NNUEReplayBuffer works with training examples."""

    def test_add_and_sample(self):
        buf = NNUEReplayBuffer(max_size=100)

        for _ in range(10):
            features = np.random.randn(FEATURE_DIM).astype(np.float32)
            policy = np.random.dirichlet(np.ones(NNUEEncoder.ACTION_SPACE_SIZE)).astype(np.float32)
            value = np.random.choice([-1.0, 0.0, 1.0])
            buf.add_examples([NNUETrainingExample(features, policy, value)])

        assert len(buf) == 10

        features_batch, policies, values = buf.sample_batch(4)
        assert features_batch.shape == (4, FEATURE_DIM)
        assert policies.shape == (4, NNUEEncoder.ACTION_SPACE_SIZE)
        assert values.shape == (4, 1)

    def test_buffer_overflow(self):
        buf = NNUEReplayBuffer(max_size=5)
        for i in range(10):
            features = np.zeros(FEATURE_DIM, dtype=np.float32)
            policy = np.zeros(NNUEEncoder.ACTION_SPACE_SIZE, dtype=np.float32)
            buf.add_examples([NNUETrainingExample(features, policy, float(i))])

        assert len(buf) == 5


# ---------------------------------------------------------------------------
# TestTrainingStep
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """One training step: forward + backward through the NNUE MLP."""

    def test_forward_backward(self):
        net = HiveNNUE(SMALL_NET)
        net.train()

        features = torch.randn(4, FEATURE_DIM)
        policy_targets = torch.softmax(
            torch.randn(4, NNUEEncoder.ACTION_SPACE_SIZE), dim=1
        )
        value_targets = torch.randn(4, 1).clamp(-1, 1)

        policy_logits, value_pred = net(features)
        total_loss, p_loss, v_loss = compute_loss(
            policy_logits, value_pred, policy_targets, value_targets
        )

        assert total_loss.item() > 0
        assert p_loss.item() > 0
        assert v_loss.item() >= 0

        total_loss.backward()

        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step(self):
        """Verify an optimizer step reduces loss."""
        net = HiveNNUE(SMALL_NET)
        net.train()

        features = torch.randn(8, FEATURE_DIM)
        policy_targets = torch.softmax(
            torch.randn(8, NNUEEncoder.ACTION_SPACE_SIZE), dim=1
        )
        value_targets = torch.zeros(8, 1)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        # Loss before
        policy_logits, value_pred = net(features)
        loss_before, _, _ = compute_loss(
            policy_logits, value_pred, policy_targets, value_targets
        )

        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        # Loss after
        with torch.no_grad():
            policy_logits2, value_pred2 = net(features)
            loss_after, _, _ = compute_loss(
                policy_logits2, value_pred2, policy_targets, value_targets
            )

        assert loss_after.item() < loss_before.item() * 1.1


# ---------------------------------------------------------------------------
# TestSelfPlayGame
# ---------------------------------------------------------------------------


class TestSelfPlayGame:
    """Test self-play game generation."""

    def test_self_play_game(self):
        config = NNUETrainConfig(
            mcts_simulations=3,
            max_game_length=10,
            games_per_iteration=1,
        )
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)

        examples, result, length = trainer._self_play_game(trainer.best_net)

        assert len(examples) > 0
        assert len(examples) <= 10
        assert isinstance(examples[0], NNUETrainingExample)
        assert examples[0].features.shape == (FEATURE_DIM,)
        assert examples[0].policy_target.shape == (NNUEEncoder.ACTION_SPACE_SIZE,)
        assert -1.0 <= examples[0].value_target <= 1.0
        assert length <= 10


# ---------------------------------------------------------------------------
# TestTrainer
# ---------------------------------------------------------------------------


class TestTrainer:
    """Test the full NNUE trainer pipeline."""

    def test_trainer_init(self):
        config = NNUETrainConfig(
            num_iterations=1,
            games_per_iteration=2,
            mcts_simulations=3,
            max_game_length=10,
        )
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)

        assert trainer.best_net is not None
        assert trainer.encoder is not None
        assert len(trainer.buffer) == 0

    def test_train_phase_with_buffer(self):
        """Train phase works when buffer has enough examples."""
        config = NNUETrainConfig(
            batch_size=4,
            num_epochs=1,
        )
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)

        for _ in range(20):
            features = np.random.randn(FEATURE_DIM).astype(np.float32)
            policy = np.random.dirichlet(
                np.ones(NNUEEncoder.ACTION_SPACE_SIZE)
            ).astype(np.float32)
            value = np.random.choice([-1.0, 0.0, 1.0])
            trainer.buffer.add_examples(
                [NNUETrainingExample(features, policy, value)]
            )

        new_net, stats = trainer._train_phase(iteration=1)
        assert stats.num_batches > 0
        assert stats.avg_loss > 0

    def test_train_phase_empty_buffer(self):
        """Train phase returns untrained net when buffer is too small."""
        config = NNUETrainConfig(batch_size=64)
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)

        new_net, stats = trainer._train_phase(iteration=1)
        assert stats.num_batches == 0

    def test_learning_rate_schedule(self):
        """Verify LR schedule produces expected values."""
        config = NNUETrainConfig(
            num_iterations=10,
            learning_rate=1e-3,
            lr_warmup_iterations=2,
            lr_schedule="cosine",
            lr_min=1e-5,
        )
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)

        # Warmup: should ramp up
        lr1 = trainer._get_learning_rate(1)
        lr2 = trainer._get_learning_rate(2)
        assert lr1 < lr2

        # After warmup: cosine decay
        lr3 = trainer._get_learning_rate(3)
        lr10 = trainer._get_learning_rate(10)
        assert lr3 > lr10  # Should decrease

    def test_learning_rate_constant(self):
        """Constant LR schedule returns base LR."""
        config = NNUETrainConfig(
            learning_rate=1e-3,
            lr_schedule="constant",
            lr_warmup_iterations=0,
        )
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)

        lr = trainer._get_learning_rate(5)
        assert abs(lr - 1e-3) < 1e-8


# ---------------------------------------------------------------------------
# TestCheckpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_save_and_load(self, tmp_path):
        config = NNUETrainConfig(
            num_iterations=5,
            checkpoint_dir=str(tmp_path),
        )
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)

        path = trainer._save_checkpoint(trainer.best_net, iteration=3)
        assert os.path.exists(path)

        loaded = NNUETrainer.from_checkpoint(path)
        assert loaded._start_iteration == 4
        assert loaded.net_config.hidden_dims == SMALL_NET.hidden_dims

    def test_resume_with_overrides(self, tmp_path):
        config = NNUETrainConfig(
            num_iterations=10,
            checkpoint_dir=str(tmp_path),
        )
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)
        path = trainer._save_checkpoint(trainer.best_net, iteration=5)

        loaded = NNUETrainer.from_checkpoint(
            path, config_overrides={"num_iterations": 20}
        )
        assert loaded.config.num_iterations == 20

    def test_checkpoint_preserves_weights(self, tmp_path):
        """Network weights should be identical after save/load."""
        config = NNUETrainConfig(checkpoint_dir=str(tmp_path))
        trainer = NNUETrainer(config=config, net_config=SMALL_NET)

        path = trainer._save_checkpoint(trainer.best_net, iteration=1)
        loaded = NNUETrainer.from_checkpoint(path)

        for (n1, p1), (n2, p2) in zip(
            trainer.best_net.named_parameters(),
            loaded.best_net.named_parameters(),
        ):
            assert n1 == n2
            torch.testing.assert_close(p1.cpu(), p2.cpu())


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI argument parsing."""

    def test_parse_defaults(self):
        from hive_nnue.train import parse_args

        args = parse_args([])
        assert args.iterations == 20
        assert args.games == 50
        assert args.preset is None
        assert args.hidden_dims is None
        assert args.dropout == 0.1

    def test_parse_custom(self):
        from hive_nnue.train import parse_args

        args = parse_args([
            "--iterations", "5",
            "--games", "10",
            "--preset", "large",
            "--dropout", "0.2",
            "--device", "cpu",
            "--no-amp",
        ])
        assert args.iterations == 5
        assert args.games == 10
        assert args.preset == "large"
        assert args.dropout == 0.2
        assert args.device == "cpu"
        assert args.no_amp is True

    def test_parse_hidden_dims(self):
        from hive_nnue.train import parse_args

        args = parse_args(["--hidden-dims", "256", "128", "64"])
        assert args.hidden_dims == [256, 128, 64]

    def test_build_net_config_preset(self):
        from hive_nnue.train import parse_args, _build_net_config

        args = parse_args(["--preset", "large"])
        config = _build_net_config(args)
        assert config.hidden_dims == [1024, 512, 256]

    def test_build_net_config_custom_dims(self):
        from hive_nnue.train import parse_args, _build_net_config

        args = parse_args(["--hidden-dims", "256", "128"])
        config = _build_net_config(args)
        assert config.hidden_dims == [256, 128]

    def test_build_net_config_default(self):
        from hive_nnue.train import parse_args, _build_net_config

        args = parse_args([])
        config = _build_net_config(args)
        assert config.hidden_dims == [512, 256]  # small preset


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all public exports from hive_nnue."""

    def test_all_exports(self):
        from hive_nnue import (
            NNUEFeatureEncoder,
            NNUEEncoder,
            FEATURE_DIM,
            FEATURES_PER_PIECE,
            NUM_PIECES_PER_PLAYER,
            NUM_DISTANCE_BUCKETS,
            NUM_GLOBAL_FEATURES,
            NNUEConfig,
            HiveNNUE,
            NNUETrainingExample,
            NNUEReplayBuffer,
            NNUETrainConfig,
            NNUETrainer,
        )
        assert NNUEFeatureEncoder is not None
        assert HiveNNUE is not None
        assert NNUETrainer is not None
        assert FEATURE_DIM == 428
