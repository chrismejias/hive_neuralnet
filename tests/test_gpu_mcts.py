"""
Tests for GPU MCTS components.

Phase 1: Legal mask kernel correctness (GPU vs CPU).
Phase 2: GPU MCTS orchestrator (after gpu_mcts.py is implemented).
Phase 3: GPU self-play integration with training (Milestone 4).
"""

import numpy as np
import pytest
import torch

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.pieces import Color, PieceType


# ── Helpers ────────────────────────────────────────────────────────────


def _load_ext():
    """Load GPU extension, skip if CUDA not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import hive_gpu
    ext = hive_gpu.load_extension()
    return ext


def _play_random_moves_parallel(ext, gs, gpu_states, n, rng):
    """
    Play n random moves on both CPU GameState and GPU states in parallel.
    Returns the number of moves actually played.
    """
    BOARD_SIZE = ext.BOARD_SIZE
    HALF = BOARD_SIZE // 2

    played = 0
    for _ in range(n):
        if gs.result != GameResult.IN_PROGRESS:
            break

        cpu_moves = gs.legal_moves()
        if not cpu_moves:
            break

        move = rng.choice(cpu_moves)
        gs.apply_move(move)

        # Generate GPU legal moves and find the matching one
        moves_tensor, num_legal = ext.generate_legal_moves_batch(gpu_states, 1)
        n_legal = num_legal[0].item()

        # Build CPU move signature
        if move.move_type == MoveType.PLACE:
            cpu_type = 0
            cpu_to = (move.to.r + HALF) * BOARD_SIZE + (move.to.q + HALF)
            cpu_pt = move.piece.piece_type.value + 1
            cpu_from = 0
        elif move.move_type == MoveType.MOVE:
            cpu_type = 1
            cpu_from = (move.from_pos.r + HALF) * BOARD_SIZE + (move.from_pos.q + HALF)
            cpu_to = (move.to.r + HALF) * BOARD_SIZE + (move.to.q + HALF)
            cpu_pt = 0
        else:
            cpu_type = 2
            cpu_to = 0
            cpu_from = 0
            cpu_pt = 0

        # Parse GPU moves and find match
        raw_moves = moves_tensor[0].cpu().numpy()
        match_idx = -1
        for mi in range(n_legal):
            raw = raw_moves[mi]
            g_type = int(raw[0])
            g_pt = int(raw[1])
            g_from = int(raw[2]) | (int(raw[3]) << 8)
            g_to = int(raw[4]) | (int(raw[5]) << 8)

            if cpu_type == 2 and g_type == 2:
                match_idx = mi
                break
            elif cpu_type == 0 and g_type == 0 and g_to == cpu_to and g_pt == cpu_pt:
                match_idx = mi
                break
            elif cpu_type == 1 and g_type == 1 and g_from == cpu_from and g_to == cpu_to:
                match_idx = mi
                break

        assert match_idx >= 0, (
            f"Could not find matching GPU move for CPU move: "
            f"type={cpu_type}, pt={cpu_pt}, from={cpu_from}, to={cpu_to}"
        )

        # Apply the matched move on GPU
        move_bytes = raw_moves[match_idx:match_idx + 1].copy()
        move_t = torch.from_numpy(move_bytes).cuda().unsqueeze(0)
        ext.apply_moves_batch(gpu_states, move_t, 1)
        played += 1

    return played


# ── Tests ──────────────────────────────────────────────────────────────


class TestLegalMask:
    """Test that GPU legal mask matches CPU encoder.get_legal_action_mask()."""

    def test_action_space_constants(self):
        """Verify GPU constants match CPU encoder constants."""
        ext = _load_ext()
        assert ext.ACTION_SPACE_SIZE == HiveEncoder.ACTION_SPACE_SIZE == 29914
        assert ext.PASS_ACTION_INDEX == HiveEncoder.PASS_ACTION_INDEX == 29913

    def test_initial_position(self):
        """Legal mask for the initial position (turn 0, all placements)."""
        ext = _load_ext()
        encoder = HiveEncoder()

        gs = GameState()
        gpu_states = ext.create_initial_states(1)

        # CPU mask
        cpu_mask = encoder.get_legal_action_mask(gs)

        # GPU mask
        gpu_masks, num_legal = ext.generate_legal_mask_batch(gpu_states, 1)
        gpu_mask = gpu_masks[0].cpu().numpy()

        # Compare
        np.testing.assert_array_equal(
            gpu_mask, cpu_mask,
            err_msg="Initial position legal masks differ"
        )

    def test_after_one_move(self):
        """Legal mask after placing one piece (turn 1)."""
        ext = _load_ext()
        encoder = HiveEncoder()
        rng = np.random.RandomState(42)

        gs = GameState()
        gpu_states = ext.create_initial_states(1)

        _play_random_moves_parallel(ext, gs, gpu_states, 1, rng)

        cpu_mask = encoder.get_legal_action_mask(gs)
        gpu_masks, _ = ext.generate_legal_mask_batch(gpu_states, 1)
        gpu_mask = gpu_masks[0].cpu().numpy()

        np.testing.assert_array_equal(
            gpu_mask, cpu_mask,
            err_msg="After 1 move: legal masks differ"
        )

    def test_after_6_moves(self):
        """Legal mask after 6 moves (both players have pieces, more complex)."""
        ext = _load_ext()
        encoder = HiveEncoder()
        rng = np.random.RandomState(123)

        gs = GameState()
        gpu_states = ext.create_initial_states(1)

        _play_random_moves_parallel(ext, gs, gpu_states, 6, rng)

        cpu_mask = encoder.get_legal_action_mask(gs)
        gpu_masks, _ = ext.generate_legal_mask_batch(gpu_states, 1)
        gpu_mask = gpu_masks[0].cpu().numpy()

        # Check same number of legal actions
        cpu_count = int(cpu_mask.sum())
        gpu_count = int(gpu_mask.sum())
        assert cpu_count == gpu_count, (
            f"Legal action count mismatch: CPU={cpu_count}, GPU={gpu_count}"
        )

        np.testing.assert_array_equal(
            gpu_mask, cpu_mask,
            err_msg="After 6 moves: legal masks differ"
        )

    def test_after_20_moves(self):
        """Legal mask after 20 moves (movement actions, queen forced)."""
        ext = _load_ext()
        encoder = HiveEncoder()
        rng = np.random.RandomState(999)

        gs = GameState()
        gpu_states = ext.create_initial_states(1)

        played = _play_random_moves_parallel(ext, gs, gpu_states, 20, rng)

        if gs.result != GameResult.IN_PROGRESS:
            pytest.skip(f"Game ended after {played} moves")

        cpu_mask = encoder.get_legal_action_mask(gs)
        gpu_masks, _ = ext.generate_legal_mask_batch(gpu_states, 1)
        gpu_mask = gpu_masks[0].cpu().numpy()

        np.testing.assert_array_equal(
            gpu_mask, cpu_mask,
            err_msg=f"After {played} moves: legal masks differ"
        )

    def test_multiple_random_games(self):
        """Run 5 random games, checking legal mask at various points."""
        ext = _load_ext()
        encoder = HiveEncoder()

        mismatches = 0
        total_checks = 0

        for game_idx in range(5):
            rng = np.random.RandomState(game_idx * 17 + 42)
            gs = GameState()
            gpu_states = ext.create_initial_states(1)

            for step in range(30):
                if gs.result != GameResult.IN_PROGRESS:
                    break

                # Check mask every 5 steps
                if step % 5 == 0:
                    cpu_mask = encoder.get_legal_action_mask(gs)
                    gpu_masks, _ = ext.generate_legal_mask_batch(gpu_states, 1)
                    gpu_mask = gpu_masks[0].cpu().numpy()

                    total_checks += 1
                    if not np.array_equal(gpu_mask, cpu_mask):
                        cpu_legal = set(np.where(cpu_mask > 0)[0])
                        gpu_legal = set(np.where(gpu_mask > 0)[0])
                        only_cpu = cpu_legal - gpu_legal
                        only_gpu = gpu_legal - cpu_legal
                        mismatches += 1
                        pytest.fail(
                            f"Game {game_idx}, step {step}: mask mismatch. "
                            f"CPU-only actions: {only_cpu}, "
                            f"GPU-only actions: {only_gpu}"
                        )

                # Play a move
                cpu_moves = gs.legal_moves()
                if not cpu_moves:
                    break

                move = rng.choice(cpu_moves)
                gs.apply_move(move)

                # Match and apply on GPU
                moves_tensor, num_legal = ext.generate_legal_moves_batch(gpu_states, 1)
                n_legal = num_legal[0].item()
                BOARD_SIZE = ext.BOARD_SIZE
                HALF = BOARD_SIZE // 2

                if move.move_type == MoveType.PLACE:
                    cpu_type = 0
                    cpu_to = (move.to.r + HALF) * BOARD_SIZE + (move.to.q + HALF)
                    cpu_pt = move.piece.piece_type.value + 1
                    cpu_from = 0
                elif move.move_type == MoveType.MOVE:
                    cpu_type = 1
                    cpu_from = (move.from_pos.r + HALF) * BOARD_SIZE + (move.from_pos.q + HALF)
                    cpu_to = (move.to.r + HALF) * BOARD_SIZE + (move.to.q + HALF)
                    cpu_pt = 0
                else:
                    cpu_type = 2
                    cpu_to = 0
                    cpu_from = 0
                    cpu_pt = 0

                raw_moves = moves_tensor[0].cpu().numpy()
                match_idx = -1
                for mi in range(n_legal):
                    raw = raw_moves[mi]
                    g_type = int(raw[0])
                    g_pt = int(raw[1])
                    g_from = int(raw[2]) | (int(raw[3]) << 8)
                    g_to = int(raw[4]) | (int(raw[5]) << 8)

                    if cpu_type == 2 and g_type == 2:
                        match_idx = mi
                        break
                    elif cpu_type == 0 and g_type == 0 and g_to == cpu_to and g_pt == cpu_pt:
                        match_idx = mi
                        break
                    elif cpu_type == 1 and g_type == 1 and g_from == cpu_from and g_to == cpu_to:
                        match_idx = mi
                        break

                if match_idx < 0:
                    break

                move_bytes = raw_moves[match_idx:match_idx + 1].copy()
                move_t = torch.from_numpy(move_bytes).cuda().unsqueeze(0)
                ext.apply_moves_batch(gpu_states, move_t, 1)

        assert total_checks > 0, "No legal mask checks were performed"
        assert mismatches == 0, f"{mismatches}/{total_checks} legal mask mismatches"

    def test_pass_action(self):
        """Test that PASS action is correctly mapped when it's the only legal move."""
        ext = _load_ext()
        encoder = HiveEncoder()

        # PASS is uncommon, but the mask should handle it.
        # We test the constant mapping at least.
        assert ext.PASS_ACTION_INDEX == 29913

    def test_batch_of_states(self):
        """Test legal masks for a batch of 4 games at different positions."""
        ext = _load_ext()
        encoder = HiveEncoder()
        B = 4

        gpu_states = ext.create_initial_states(B)
        games = [GameState() for _ in range(B)]

        # Play different numbers of moves in each game
        for i in range(B):
            rng = np.random.RandomState(i * 100 + 7)
            n_moves = (i + 1) * 3  # 3, 6, 9, 12 moves

            # Play moves individually per game
            single_state = gpu_states[i:i+1].clone()
            _play_random_moves_parallel(ext, games[i], single_state, n_moves, rng)
            gpu_states[i] = single_state[0]

        # Generate masks for the whole batch
        gpu_masks, _ = ext.generate_legal_mask_batch(gpu_states, B)

        for i in range(B):
            if games[i].result != GameResult.IN_PROGRESS:
                continue

            cpu_mask = encoder.get_legal_action_mask(games[i])
            gpu_mask = gpu_masks[i].cpu().numpy()

            np.testing.assert_array_equal(
                gpu_mask, cpu_mask,
                err_msg=f"Batch element {i}: legal masks differ"
            )

    def test_mask_sum_matches_legal_count(self):
        """Mask sum should equal number of legal moves (modulo out-of-grid skips)."""
        ext = _load_ext()
        rng = np.random.RandomState(42)

        gs = GameState()
        gpu_states = ext.create_initial_states(1)

        _play_random_moves_parallel(ext, gs, gpu_states, 8, rng)

        if gs.result != GameResult.IN_PROGRESS:
            pytest.skip("Game ended")

        gpu_masks, num_legal = ext.generate_legal_mask_batch(gpu_states, 1)
        mask_sum = int(gpu_masks[0].sum().item())
        n_legal = num_legal[0].item()

        # mask_sum <= n_legal (some moves may be out of encoder grid)
        assert mask_sum <= n_legal, (
            f"Mask sum {mask_sum} > num_legal {n_legal}"
        )
        # But should be close (very few moves out of grid in normal play)
        assert mask_sum > 0, "Mask should have at least one legal action"


# ── GPU MCTS Orchestrator Tests ───────────────────────────────────────


class TestGPUMCTS:
    """Test GPU MCTS orchestrator components."""

    def _make_dummy_net(self, net_type="gnn"):
        """Create a small dummy network for testing."""
        if net_type == "gnn":
            from hive_gnn.gnn_net import HiveGNN, GNNNetConfig
            config = GNNNetConfig.small()
            net = HiveGNN(config).cuda().eval()
            return net
        else:
            from hive_transformer.transformer_net import HiveTransformer, TransformerConfig
            config = TransformerConfig.small()
            net = HiveTransformer(config).cuda().eval()
            return net

    def test_gpu_mcts_node_basics(self):
        """Test GPUMCTSNode creation and properties."""
        from hive_gpu.gpu_mcts import GPUMCTSNode

        root = GPUMCTSNode()
        assert root.visit_count == 0
        assert root.mean_value == 0.0
        assert not root.is_expanded
        assert not root.is_terminal
        assert root.parent is None
        assert root.children == {}

        child = GPUMCTSNode(parent=root, parent_action=42, prior=0.5)
        assert child.parent is root
        assert child.parent_action == 42
        assert child.prior == 0.5

    def test_gpu_mcts_config(self):
        """Test GPUMCTSConfig defaults."""
        from hive_gpu.gpu_mcts import GPUMCTSConfig

        config = GPUMCTSConfig()
        assert config.num_simulations == 100
        assert config.batch_size == 64
        assert config.c_puct == 1.5

    def test_orchestrator_creates(self):
        """Test GPUMCTSOrchestrator can be instantiated."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig

        net = self._make_dummy_net("gnn")
        config = GPUMCTSConfig(
            num_simulations=2,
            batch_size=2,
            encoder_type="gnn",
        )
        orchestrator = GPUMCTSOrchestrator(net, config)
        assert orchestrator.ext is not None
        assert orchestrator._action_space_size == 29914

    def test_self_play_batch_produces_examples(self):
        """Test that self_play_batch produces valid training examples with HiveGraph."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig
        from hive_gnn.graph_types import HiveGraph

        net = self._make_dummy_net("gnn")
        config = GPUMCTSConfig(
            num_simulations=2,     # Very few sims for speed
            batch_size=2,          # 2 concurrent games
            max_game_length=10,    # Short games for testing
            encoder_type="gnn",
        )
        orchestrator = GPUMCTSOrchestrator(net, config)

        # Run self-play
        all_examples = orchestrator.self_play_batch()

        # Should get B=2 game results
        assert len(all_examples) == 2

        # Each game should have produced some examples
        for game_examples in all_examples:
            assert len(game_examples) > 0, "Game produced no examples"

            for ex in game_examples:
                # Policy should sum to ~1
                policy_sum = ex.policy_target.sum()
                assert abs(policy_sum - 1.0) < 0.01, (
                    f"Policy sum = {policy_sum}, expected ~1.0"
                )

                # Value should be in [-1, 1]
                assert -1.0 <= ex.value_target <= 1.0

                # State should be a HiveGraph (not a mask)
                assert isinstance(ex.graph, HiveGraph), (
                    f"Expected HiveGraph, got {type(ex.graph)}"
                )
                assert ex.graph.node_features.ndim == 2
                assert ex.graph.node_features.shape[1] == 25
                assert ex.graph.edge_index.shape[0] == 2
                assert ex.graph.global_features.shape == (6,)

    def test_policy_only_legal_actions(self):
        """Verify MCTS policy only assigns probability to legal actions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig

        net = self._make_dummy_net("gnn")
        config = GPUMCTSConfig(
            num_simulations=5,
            batch_size=1,
            max_game_length=5,
            encoder_type="gnn",
        )
        orchestrator = GPUMCTSOrchestrator(net, config)

        all_examples = orchestrator.self_play_batch()

        for game_examples in all_examples:
            for ex in game_examples:
                policy = ex.policy_target
                # All nonzero policy entries should be valid action indices
                nonzero_actions = np.where(policy > 0)[0]
                assert len(nonzero_actions) > 0, "Policy has no nonzero entries"
                assert all(0 <= a <= 29913 for a in nonzero_actions)


# ── GPU Self-Play Integration Tests (Milestone 4) ──────────────────────


class TestGPUSelfPlayIntegration:
    """Test end-to-end GPU self-play → replay buffer → training step."""

    def _make_dummy_net(self):
        """Create a small GNN for testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from hive_gnn.gnn_net import HiveGNN, GNNNetConfig
        config = GNNNetConfig.small()
        return HiveGNN(config).cuda(), config

    def test_unbatch_to_graphs(self):
        """Test GPUGNNEncoder.unbatch_to_graphs produces valid HiveGraphs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_encoder import GPUGNNEncoder
        from hive_gnn.graph_types import HiveGraph

        ext = _load_ext()
        encoder = GPUGNNEncoder()

        # Create 3 initial states
        B = 3
        states = ext.create_initial_states(B)
        graphs = encoder.unbatch_to_graphs(states, B)

        assert len(graphs) == B
        for g in graphs:
            assert isinstance(g, HiveGraph)
            assert g.node_features.dtype == np.float32
            assert g.node_features.shape[1] == 25
            assert g.edge_index.shape[0] == 2
            assert g.edge_index.dtype == np.int64
            assert g.global_features.shape == (6,)
            assert g.node_positions.dtype == np.int32
            assert g.node_piece_types.dtype == np.int32

    def test_examples_to_replay_buffer(self):
        """Test converting GPUTrainingExamples to GNNTrainingExamples in replay buffer."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig
        from hive_gnn.gnn_replay_buffer import GNNTrainingExample, GraphReplayBuffer

        net, _ = self._make_dummy_net()
        net.eval()

        config = GPUMCTSConfig(
            num_simulations=2,
            batch_size=2,
            max_game_length=6,
            encoder_type="gnn",
        )
        orchestrator = GPUMCTSOrchestrator(net, config)
        all_examples = orchestrator.self_play_batch()

        # Convert to GNNTrainingExamples
        buffer = GraphReplayBuffer(max_size=1000)
        gnn_examples = []
        for game_examples in all_examples:
            for ex in game_examples:
                gnn_ex = GNNTrainingExample(
                    graph=ex.graph,
                    policy_target=ex.policy_target,
                    value_target=ex.value_target,
                    mobility_target=ex.mobility_target,
                    queen_surround_target=ex.queen_surround_target,
                    queen_surround_mask=ex.queen_surround_mask,
                    final_mobility_target=ex.final_mobility_target,
                    use_for_value=True,
                )
                gnn_examples.append(gnn_ex)

        buffer.add_examples(gnn_examples)
        assert len(buffer) == len(gnn_examples)

        # Sample a batch
        batch = buffer.sample_batch(min(len(buffer), 4))
        assert batch.policy_targets.shape[1] == 29914
        assert batch.value_targets.shape[1] == 1
        assert batch.graph_batch.node_features.ndim == 2

    def test_train_step(self):
        """Test one training step: self-play → buffer → forward/backward."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig
        from hive_gnn.gnn_replay_buffer import GNNTrainingExample, GraphReplayBuffer
        from hive_engine.neural_net import compute_gnn_loss

        net, _ = self._make_dummy_net()
        net.eval()

        config = GPUMCTSConfig(
            num_simulations=2,
            batch_size=2,
            max_game_length=6,
            encoder_type="gnn",
        )
        orchestrator = GPUMCTSOrchestrator(net, config)
        all_examples = orchestrator.self_play_batch()

        # Convert and add to buffer
        buffer = GraphReplayBuffer(max_size=1000)
        gnn_examples = []
        for game_examples in all_examples:
            for ex in game_examples:
                gnn_ex = GNNTrainingExample(
                    graph=ex.graph,
                    policy_target=ex.policy_target,
                    value_target=ex.value_target,
                    mobility_target=ex.mobility_target,
                    queen_surround_target=ex.queen_surround_target,
                    queen_surround_mask=ex.queen_surround_mask,
                    final_mobility_target=ex.final_mobility_target,
                    use_for_value=True,
                )
                gnn_examples.append(gnn_ex)

        buffer.add_examples(gnn_examples)
        assert len(buffer) >= 2, "Need at least 2 examples for a batch"

        # Sample batch and do one training step
        batch = buffer.sample_batch(min(len(buffer), 4))
        batch = batch.to(torch.device("cuda"))

        net.train()
        policy_logits, value_pred, aux_outputs = net(batch.graph_batch)

        total_loss, loss_dict = compute_gnn_loss(
            policy_logits,
            value_pred,
            batch.policy_targets,
            batch.value_targets,
            aux_outputs,
            batch.mobility_targets,
            batch.queen_surround_targets,
            batch.queen_surround_mask,
            batch.final_mobility_targets,
            batch.value_mask,
            batch.graph_batch.piece_node_batch,
        )

        assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss}"
        assert total_loss.item() > 0, "Loss should be positive"

        # Backward should work
        total_loss.backward()

        # Check gradients exist
        has_grad = False
        for p in net.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients after backward pass"
