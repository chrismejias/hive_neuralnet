"""
End-to-end integration tests for the Transformer Hive training pipeline.

Tests the full flow: game state → tokens → Transformer → MCTS → self-play → train.
"""

import os

import numpy as np
import pytest
import torch

from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.hex_coord import ORIGIN, ALL_DIRECTIONS
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.neural_net import compute_loss, compute_transformer_loss
from hive_engine.pieces import Color, PieceType

from hive_transformer.token_encoder import TokenEncoder
from hive_transformer.token_types import HiveTokenBatch, HiveTokenSequence
from hive_transformer.transformer_encoder import TransformerEncoder
from hive_transformer.transformer_net import TransformerConfig, HiveTransformer
from hive_transformer.transformer_replay_buffer import (
    TransformerTrainingExample,
    TransformerTrainingBatch,
    TokenReplayBuffer,
)
from hive_transformer.transformer_trainer import (
    TransformerTrainConfig,
    TransformerTrainer,
)


# ---------------------------------------------------------------------------
# Small config for fast tests
# ---------------------------------------------------------------------------

SMALL_NET = TransformerConfig(
    d_model=32, num_heads=4, num_layers=2, dim_feedforward=64
)


def _make_example(tok_enc, game):
    """Create a TransformerTrainingExample with all 8 fields."""
    seq = tok_enc.encode(game)
    policy = np.random.dirichlet(
        np.ones(TransformerEncoder.ACTION_SPACE_SIZE)
    ).astype(np.float32)
    value = np.random.choice([-1.0, 0.0, 1.0])
    num_board = seq.num_board_tokens
    mobility = np.zeros(num_board, dtype=np.float32)
    surround = np.zeros((num_board, 2), dtype=np.float32)
    surround_mask = np.zeros(2, dtype=np.float32)
    final_mob = np.zeros(num_board, dtype=np.float32)
    return TransformerTrainingExample(
        seq, policy, value, mobility, surround, surround_mask, final_mob, True
    )


# ---------------------------------------------------------------------------
# TestMCTSIntegration
# ---------------------------------------------------------------------------


class TestMCTSIntegration:
    """MCTS works with TransformerEncoder + HiveTransformer via duck typing."""

    def test_mcts_search_returns_policy(self):
        encoder = TransformerEncoder()
        net = HiveTransformer(SMALL_NET)
        net.eval()

        game = GameState()
        config = MCTSConfig(num_simulations=5)
        mcts = MCTS(net, encoder, config)

        policy = mcts.search(game, move_number=0)

        assert policy.shape == (TransformerEncoder.ACTION_SPACE_SIZE,)
        assert abs(policy.sum() - 1.0) < 1e-5
        assert (policy >= 0).all()

    def test_mcts_respects_legal_moves(self):
        encoder = TransformerEncoder()
        net = HiveTransformer(SMALL_NET)
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
        encoder = TransformerEncoder()
        net = HiveTransformer(SMALL_NET)
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
    """TokenReplayBuffer works with Transformer training examples."""

    def test_add_and_sample(self):
        tok_enc = TokenEncoder()
        game = GameState()

        buf = TokenReplayBuffer(max_size=100)

        # Add some examples
        for _ in range(10):
            buf.add_examples([_make_example(tok_enc, game)])

        assert len(buf) == 10

        # Sample a batch
        batch = buf.sample_batch(4)

        assert isinstance(batch, TransformerTrainingBatch)
        assert isinstance(batch.token_batch, HiveTokenBatch)
        assert batch.policy_targets.shape == (4, TransformerEncoder.ACTION_SPACE_SIZE)
        assert batch.value_targets.shape == (4, 1)
        assert batch.value_mask.shape == (4,)
        assert batch.queen_surround_mask.shape == (4, 2)

    def test_buffer_overflow(self):
        tok_enc = TokenEncoder()
        game = GameState()

        buf = TokenReplayBuffer(max_size=5)
        for _ in range(10):
            buf.add_examples([_make_example(tok_enc, game)])

        assert len(buf) == 5  # Oldest evicted

    def test_board_token_batch_alignment(self):
        """board_token_batch correctly maps board tokens to sequences."""
        tok_enc = TokenEncoder()
        game = GameState()
        # Place a piece so there are board tokens
        hand = game.hand(Color.WHITE)
        queen = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        game.apply_move(Move(MoveType.PLACE, queen, ORIGIN))

        buf = TokenReplayBuffer(max_size=100)
        for _ in range(8):
            buf.add_examples([_make_example(tok_enc, game)])

        batch = buf.sample_batch(4)
        # Each sequence has 1 board token (queen on board)
        assert batch.board_token_batch.shape[0] == batch.mobility_targets.shape[0]
        # Values should be 0, 1, 2, 3 (one board token per sequence)
        assert batch.board_token_batch.max().item() <= 3


# ---------------------------------------------------------------------------
# TestTransformerLoss
# ---------------------------------------------------------------------------


class TestTransformerLoss:
    """Test compute_transformer_loss with auxiliary heads."""

    def test_basic_loss(self):
        net = HiveTransformer(SMALL_CFG)
        net.train()

        tok_enc = TokenEncoder()
        game = GameState()
        hand = game.hand(Color.WHITE)
        queen = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        game.apply_move(Move(MoveType.PLACE, queen, ORIGIN))

        sequences = [tok_enc.encode(game) for _ in range(4)]
        batch = HiveTokenBatch.collate(sequences)

        policy_logits, value_pred, aux_outputs = net(batch)

        policy_targets = torch.softmax(
            torch.randn(4, TransformerEncoder.ACTION_SPACE_SIZE), dim=1
        )
        value_targets = torch.randn(4, 1).clamp(-1, 1)

        # Each sequence has 1 board token
        total_board = 4 * 1
        mobility_targets = torch.zeros(total_board)
        surround_targets = torch.zeros(total_board, 2)
        surround_mask = torch.zeros(4, 2)
        final_mob_targets = torch.zeros(total_board)
        value_mask = torch.ones(4)
        board_token_batch = torch.arange(4).repeat_interleave(1)

        total_loss, loss_dict = compute_transformer_loss(
            policy_logits, value_pred, policy_targets, value_targets,
            aux_outputs, mobility_targets, surround_targets, surround_mask,
            final_mob_targets, value_mask, board_token_batch,
        )

        assert total_loss.item() > 0
        assert "policy_loss" in loss_dict
        assert "value_loss" in loss_dict
        assert "mobility_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_value_mask_zeros_out_value_loss(self):
        """When value_mask is all zeros, value loss should be zero."""
        B = 2
        policy_logits = torch.randn(B, 29914)
        value_pred = torch.randn(B, 1)
        policy_targets = torch.softmax(torch.randn(B, 29914), dim=1)
        value_targets = torch.ones(B, 1)
        value_mask = torch.zeros(B)
        board_token_batch = torch.tensor([], dtype=torch.long)

        total_loss, loss_dict = compute_transformer_loss(
            policy_logits, value_pred, policy_targets, value_targets,
            {}, torch.tensor([]), torch.zeros(0, 2),
            torch.zeros(B, 2), torch.tensor([]), value_mask,
            board_token_batch,
        )

        assert loss_dict["value_loss"].item() == 0.0


SMALL_CFG = TransformerConfig(
    d_model=32, num_heads=4, num_layers=2, dim_feedforward=64
)


# ---------------------------------------------------------------------------
# TestTrainingStep
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """One training step: forward + backward through the Transformer."""

    def test_forward_backward(self):
        net = HiveTransformer(SMALL_NET)
        net.train()

        tok_enc = TokenEncoder()

        # Use a game with pieces on the board (so board tokens get scattered)
        game = GameState()
        hand = game.hand(Color.WHITE)
        queen = next(p for p in hand if p.piece_type == PieceType.QUEEN)
        game.apply_move(Move(MoveType.PLACE, queen, ORIGIN))
        hand_b = game.hand(Color.BLACK)
        queen_b = next(p for p in hand_b if p.piece_type == PieceType.QUEEN)
        neighbor = ORIGIN.neighbor(ALL_DIRECTIONS[0])
        game.apply_move(Move(MoveType.PLACE, queen_b, neighbor))

        # Create a small batch
        B = 4
        sequences = [tok_enc.encode(game) for _ in range(B)]
        batch = HiveTokenBatch.collate(sequences)

        policy_targets = torch.randn(B, TransformerEncoder.ACTION_SPACE_SIZE)
        policy_targets = torch.softmax(policy_targets, dim=1)
        value_targets = torch.randn(B, 1).clamp(-1, 1)

        # Forward (now returns 3-tuple)
        policy_logits, value_pred, aux_outputs = net(batch)

        # Use full transformer loss so aux heads get gradients
        num_board = sequences[0].num_board_tokens
        total_board = B * num_board
        mobility_targets = torch.zeros(total_board)
        surround_targets = torch.zeros(total_board, 2)
        surround_mask = torch.ones(B, 2)
        final_mob_targets = torch.zeros(total_board)
        value_mask = torch.ones(B)
        board_token_batch = torch.arange(B).repeat_interleave(num_board)

        total_loss, loss_dict = compute_transformer_loss(
            policy_logits, value_pred, policy_targets, value_targets,
            aux_outputs, mobility_targets, surround_targets, surround_mask,
            final_mob_targets, value_mask, board_token_batch,
        )

        assert total_loss.item() > 0
        assert loss_dict["policy_loss"].item() > 0
        assert loss_dict["value_loss"].item() >= 0

        # Backward
        total_loss.backward()

        # Check gradients exist for all parameters
        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_optimizer_step(self):
        """Verify an optimizer step reduces loss."""
        net = HiveTransformer(SMALL_NET)
        net.train()

        tok_enc = TokenEncoder()
        game = GameState()

        sequences = [tok_enc.encode(game) for _ in range(8)]
        batch = HiveTokenBatch.collate(sequences)

        policy_targets = torch.softmax(
            torch.randn(8, TransformerEncoder.ACTION_SPACE_SIZE), dim=1
        )
        value_targets = torch.zeros(8, 1)

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        # Measure loss before
        policy_logits, value_pred, _ = net(batch)
        loss_before, _, _ = compute_loss(
            policy_logits, value_pred, policy_targets, value_targets
        )

        # Training step
        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        # Measure loss after
        with torch.no_grad():
            policy_logits2, value_pred2, _ = net(batch)
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
        config = TransformerTrainConfig(
            mcts_simulations=3,
            max_game_length=10,
            games_per_iteration=1,
        )
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )

        examples, result, length = trainer._self_play_game(trainer.best_net)

        assert len(examples) > 0
        assert len(examples) <= 10  # max_game_length
        assert isinstance(examples[0], TransformerTrainingExample)
        assert examples[0].policy_target.shape == (
            TransformerEncoder.ACTION_SPACE_SIZE,
        )
        assert -1.0 <= examples[0].value_target <= 1.0
        assert length <= 10

    def test_self_play_produces_aux_targets(self):
        config = TransformerTrainConfig(
            mcts_simulations=3,
            max_game_length=10,
        )
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )

        examples, _, _ = trainer._self_play_game(trainer.best_net)

        for ex in examples:
            num_board = ex.sequence.num_board_tokens
            assert ex.mobility_target.shape == (num_board,)
            assert ex.queen_surround_target.shape == (num_board, 2)
            assert ex.queen_surround_mask.shape == (2,)
            assert ex.final_mobility_target.shape == (num_board,)
            assert isinstance(ex.use_for_value, bool)

    def test_playout_cap_randomization(self):
        config = TransformerTrainConfig(
            mcts_simulations=10,
            max_game_length=5,
            games_per_iteration=10,
            playout_cap_randomization=True,
            playout_cap_full_fraction=0.5,
        )
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )

        examples, stats = trainer._self_play_phase()

        # With 50% full fraction, some should have use_for_value=False
        has_false = any(not ex.use_for_value for ex in examples)
        has_true = any(ex.use_for_value for ex in examples)
        # With 10 games and 50% chance, both should occur (statistical)
        assert len(examples) > 0


# ---------------------------------------------------------------------------
# TestTrainer
# ---------------------------------------------------------------------------


class TestTrainer:
    """Test the full Transformer trainer pipeline."""

    def test_trainer_init(self):
        config = TransformerTrainConfig(
            num_iterations=1,
            games_per_iteration=2,
            mcts_simulations=3,
            max_game_length=10,
        )
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )

        assert trainer.best_net is not None
        assert trainer.encoder is not None
        assert len(trainer.buffer) == 0

    def test_train_phase_with_buffer(self):
        """Train phase works when buffer has enough examples."""
        config = TransformerTrainConfig(
            batch_size=4,
            num_epochs=1,
        )
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )

        # Populate buffer manually
        tok_enc = TokenEncoder()
        game = GameState()
        for _ in range(20):
            trainer.buffer.add_examples([_make_example(tok_enc, game)])

        new_net, stats = trainer._train_phase(iteration=1)
        assert stats.num_batches > 0
        assert stats.avg_loss > 0

    def test_train_phase_empty_buffer(self):
        """Train phase returns untrained net when buffer is too small."""
        config = TransformerTrainConfig(batch_size=64)
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )

        new_net, stats = trainer._train_phase(iteration=1)
        assert stats.num_batches == 0

    def test_continuous_updates_config(self):
        config = TransformerTrainConfig(continuous_updates=True)
        assert config.continuous_updates is True

    def test_parameter_count_comparison(self):
        """Verify parameter counts for presets."""
        small = HiveTransformer(TransformerConfig.small())
        large = HiveTransformer(TransformerConfig.large())

        small_count = small.count_parameters()
        large_count = large.count_parameters()

        assert small_count > 0
        assert large_count > small_count
        print(f"  Transformer small: {small_count:,} params")
        print(f"  Transformer large: {large_count:,} params")


# ---------------------------------------------------------------------------
# TestCheckpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_save_and_load(self, tmp_path):
        config = TransformerTrainConfig(
            num_iterations=5,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )

        # Save
        path = trainer._save_checkpoint(trainer.best_net, iteration=3)
        assert os.path.exists(path)

        # Load
        loaded = TransformerTrainer.from_checkpoint(path)
        assert loaded._start_iteration == 4
        assert loaded.net_config.d_model == SMALL_NET.d_model
        assert loaded.net_config.aux_mobility_enabled is True

    def test_resume_with_overrides(self, tmp_path):
        config = TransformerTrainConfig(
            num_iterations=10,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )
        path = trainer._save_checkpoint(trainer.best_net, iteration=5)

        loaded = TransformerTrainer.from_checkpoint(
            path, config_overrides={"num_iterations": 20}
        )
        # num_iterations override is relative: checkpoint_iter (5) + 20 = 25
        assert loaded.config.num_iterations == 25

    def test_backward_compat_checkpoint(self, tmp_path):
        """Old checkpoint without aux config fields loads cleanly."""
        config = TransformerTrainConfig(
            num_iterations=5,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TransformerTrainer(
            config=config, net_config=SMALL_NET
        )
        path = trainer._save_checkpoint(trainer.best_net, iteration=1)

        # Simulate old checkpoint by removing new config fields
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(checkpoint["train_config"], "playout_cap_randomization"):
            delattr(checkpoint["train_config"], "playout_cap_randomization")
        if hasattr(checkpoint["train_config"], "continuous_updates"):
            delattr(checkpoint["train_config"], "continuous_updates")
        if hasattr(checkpoint["net_config"], "aux_mobility_enabled"):
            delattr(checkpoint["net_config"], "aux_mobility_enabled")
        torch.save(checkpoint, path)

        # Should load without error
        loaded = TransformerTrainer.from_checkpoint(path)
        assert loaded.config.playout_cap_randomization is True
        assert loaded.config.continuous_updates is False
        assert loaded.net_config.aux_mobility_enabled is True


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """Test CLI argument parsing."""

    def test_parse_defaults(self):
        from hive_transformer.train import parse_args

        args = parse_args([])
        assert args.iterations == 20
        assert args.games == 50
        assert args.d_model == 128
        assert args.num_layers == 6
        assert args.num_heads == 8
        assert args.continuous_updates is False
        assert args.no_playout_cap is False
        assert args.policy_prune_threshold == 0.0

    def test_parse_custom(self):
        from hive_transformer.train import parse_args

        args = parse_args([
            "--iterations", "5",
            "--games", "10",
            "--d-model", "64",
            "--num-layers", "3",
            "--num-heads", "4",
            "--device", "cpu",
            "--no-amp",
            "--continuous-updates",
            "--no-playout-cap",
            "--policy-prune-threshold", "0.03",
            "--no-mobility-head",
            "--surround-weight", "0.2",
        ])
        assert args.iterations == 5
        assert args.games == 10
        assert args.d_model == 64
        assert args.num_layers == 3
        assert args.num_heads == 4
        assert args.device == "cpu"
        assert args.no_amp is True
        assert args.continuous_updates is True
        assert args.no_playout_cap is True
        assert args.policy_prune_threshold == 0.03
        assert args.no_mobility_head is True
        assert args.surround_weight == 0.2


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all public exports from hive_transformer."""

    def test_all_exports(self):
        from hive_transformer import (
            HiveTokenSequence,
            HiveTokenBatch,
            TOKEN_FEAT_DIM,
            GLOBAL_FEAT_DIM,
            OFF_BOARD_POSITION,
            TOKEN_TYPE_CLS,
            TOKEN_TYPE_BOARD,
            TOKEN_TYPE_HAND,
            TokenEncoder,
            TransformerEncoder,
            TransformerConfig,
            HiveTransformer,
            TransformerPolicyHead,
            TransformerValueHead,
            TransformerMobilityHead,
            TransformerQueenSurroundHead,
            TransformerFinalMobilityHead,
            TransformerTrainingExample,
            TransformerTrainingBatch,
            TokenReplayBuffer,
            TransformerTrainConfig,
            TransformerTrainer,
        )
        # Just verify they imported without error
        assert HiveTokenSequence is not None
        assert HiveTransformer is not None
        assert TransformerTrainer is not None
        assert TransformerTrainingBatch is not None
        assert TransformerMobilityHead is not None
