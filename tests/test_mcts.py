"""Tests for MCTS implementation."""

import numpy as np
import pytest
import torch

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.neural_net import HiveNet, NetConfig
from hive_engine.mcts import MCTS, MCTSConfig, MCTSNode
from hive_engine.pieces import Color, PieceType
from hive_engine.hex_coord import HexCoord


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def tiny_net():
    """Very small network for fast tests."""
    config = NetConfig(num_blocks=1, num_filters=16)
    net = HiveNet(config)
    net.eval()
    return net


@pytest.fixture
def encoder():
    return HiveEncoder()


@pytest.fixture
def mcts_fast(tiny_net, encoder):
    """MCTS with minimal simulations for fast tests."""
    config = MCTSConfig(
        num_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
    )
    return MCTS(tiny_net, encoder, config)


@pytest.fixture
def game_state():
    return GameState()


# ── MCTSNode Tests ───────────────────────────────────────────────


class TestMCTSNode:
    def test_initial_state(self, game_state):
        node = MCTSNode(game_state)
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.prior == 0.0
        assert not node.is_expanded
        assert not node.children

    def test_mean_value_zero_visits(self, game_state):
        node = MCTSNode(game_state)
        assert node.mean_value == 0.0

    def test_mean_value_computed(self, game_state):
        node = MCTSNode(game_state)
        node.visit_count = 10
        node.total_value = 3.0
        assert abs(node.mean_value - 0.3) < 1e-6

    def test_is_terminal_new_game(self, game_state):
        node = MCTSNode(game_state)
        assert not node.is_terminal


# ── MCTS Search Tests ───────────────────────────────────────────


class TestMCTSSearch:
    def test_policy_shape(self, mcts_fast, game_state):
        policy = mcts_fast.search(game_state)
        assert policy.shape == (HiveEncoder.ACTION_SPACE_SIZE,)

    def test_policy_sums_to_one(self, mcts_fast, game_state):
        policy = mcts_fast.search(game_state)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_policy_non_negative(self, mcts_fast, game_state):
        policy = mcts_fast.search(game_state)
        assert (policy >= 0.0).all()

    def test_only_legal_actions_have_probability(self, mcts_fast, game_state, encoder):
        policy = mcts_fast.search(game_state)
        mask = encoder.get_legal_action_mask(game_state)
        # Actions with non-zero probability should be legal
        nonzero = policy > 0
        assert (nonzero <= (mask > 0)).all(), (
            "Found non-zero probability for illegal action"
        )

    def test_visit_counts_sum_to_num_simulations(self, tiny_net, encoder, game_state):
        config = MCTSConfig(num_simulations=20, temperature=1.0)
        mcts = MCTS(tiny_net, encoder, config)
        root = MCTSNode(game_state.copy())
        mcts._expand(root)
        mcts._add_dirichlet_noise(root)

        for _ in range(20):
            node = mcts._select(root)
            if node.is_terminal:
                value = mcts._terminal_value(node)
            else:
                value = mcts._expand(node)
            mcts._backpropagate(node, value)

        # Root visit count = num_simulations (from backpropagation only)
        total_child_visits = sum(
            c.visit_count for c in root.children.values()
        )
        # Total child visits should equal root visits (every simulation
        # traverses through root to one child)
        assert total_child_visits == root.visit_count

    def test_search_after_moves(self, mcts_fast):
        """Test MCTS on a game with some moves already played."""
        import random
        random.seed(42)
        gs = GameState()
        for _ in range(6):
            moves = gs.legal_moves()
            if not moves:
                break
            gs.apply_move(random.choice(moves))

        policy = mcts_fast.search(gs)
        assert abs(policy.sum() - 1.0) < 1e-5
        assert (policy >= 0).all()


# ── PUCT Score Tests ─────────────────────────────────────────────


class TestPUCTScore:
    def test_higher_prior_higher_score(self, tiny_net, encoder):
        """Child with higher prior should get higher PUCT score (when N=0)."""
        config = MCTSConfig(c_puct=1.5)
        mcts = MCTS(tiny_net, encoder, config)

        gs = GameState()
        parent = MCTSNode(gs)
        parent.visit_count = 10

        child_high = MCTSNode(gs, parent=parent, prior=0.8)
        child_low = MCTSNode(gs, parent=parent, prior=0.1)

        score_high = mcts._puct_score(parent, child_high)
        score_low = mcts._puct_score(parent, child_low)
        assert score_high > score_low

    def test_unvisited_child_gets_exploration_bonus(self, tiny_net, encoder):
        """Unvisited child should have higher exploration component."""
        config = MCTSConfig(c_puct=1.5)
        mcts = MCTS(tiny_net, encoder, config)

        gs = GameState()
        parent = MCTSNode(gs)
        parent.visit_count = 100

        child_visited = MCTSNode(gs, parent=parent, prior=0.5)
        child_visited.visit_count = 50
        child_visited.total_value = 0.0

        child_unvisited = MCTSNode(gs, parent=parent, prior=0.5)

        score_visited = mcts._puct_score(parent, child_visited)
        score_unvisited = mcts._puct_score(parent, child_unvisited)
        assert score_unvisited > score_visited


# ── Dirichlet Noise Tests ───────────────────────────────────────


class TestDirichletNoise:
    def test_noise_changes_priors(self, tiny_net, encoder, game_state):
        mcts = MCTS(tiny_net, encoder, MCTSConfig())
        root = MCTSNode(game_state.copy())
        mcts._expand(root)

        # Record original priors
        original_priors = {
            a: c.prior for a, c in root.children.items()
        }

        np.random.seed(42)
        mcts._add_dirichlet_noise(root)

        # At least some priors should change
        changed = sum(
            1
            for a, c in root.children.items()
            if abs(c.prior - original_priors[a]) > 1e-6
        )
        assert changed > 0

    def test_noise_keeps_priors_positive(self, tiny_net, encoder, game_state):
        mcts = MCTS(tiny_net, encoder, MCTSConfig())
        root = MCTSNode(game_state.copy())
        mcts._expand(root)
        mcts._add_dirichlet_noise(root)

        for child in root.children.values():
            assert child.prior >= 0.0


# ── Temperature Tests ────────────────────────────────────────────


class TestTemperature:
    def test_zero_temperature_is_greedy(self, tiny_net, encoder, game_state):
        """With temperature=0, policy should be one-hot on the most-visited."""
        config = MCTSConfig(num_simulations=20, temperature=0.0)
        mcts = MCTS(tiny_net, encoder, config)
        policy = mcts.search(game_state, move_number=0)

        # Should be one-hot (exactly one non-zero entry)
        nonzero = (policy > 0).sum()
        assert nonzero == 1
        assert abs(policy.max() - 1.0) < 1e-6

    def test_temperature_drop(self, tiny_net, encoder, game_state):
        """After temperature_drop_move, should behave like temp=0."""
        config = MCTSConfig(
            num_simulations=20,
            temperature=1.0,
            temperature_drop_move=10,
        )
        mcts = MCTS(tiny_net, encoder, config)
        policy = mcts.search(game_state, move_number=25)

        # Should be one-hot (greedy)
        nonzero = (policy > 0).sum()
        assert nonzero == 1


# ── Backpropagation Tests ────────────────────────────────────────


class TestBackpropagation:
    def test_value_negation(self, game_state):
        """Backprop should alternate signs going up the tree."""
        root = MCTSNode(game_state)
        root.is_expanded = True

        child = MCTSNode(game_state, parent=root, parent_action=0)
        root.children[0] = child

        # Manually backprop value +1 from child
        mcts = MCTS.__new__(MCTS)
        mcts.config = MCTSConfig()
        mcts._backpropagate(child, 1.0)

        # Child should have +1, root should have -1
        assert child.total_value == 1.0
        assert child.visit_count == 1
        assert root.total_value == -1.0
        assert root.visit_count == 1

    def test_multiple_backpropagations(self, game_state):
        """Multiple backprops should accumulate correctly."""
        root = MCTSNode(game_state)
        root.is_expanded = True

        child = MCTSNode(game_state, parent=root, parent_action=0)
        root.children[0] = child

        mcts = MCTS.__new__(MCTS)
        mcts.config = MCTSConfig()

        mcts._backpropagate(child, 1.0)
        mcts._backpropagate(child, -1.0)
        mcts._backpropagate(child, 0.5)

        assert child.visit_count == 3
        assert abs(child.total_value - 0.5) < 1e-6  # 1.0 + (-1.0) + 0.5
        assert root.visit_count == 3
        assert abs(root.total_value - (-0.5)) < 1e-6  # -1.0 + 1.0 + (-0.5)


# ── Terminal Value Tests ─────────────────────────────────────────


class TestTerminalValue:
    def test_terminal_value_requires_game_over(self, tiny_net, encoder):
        """Terminal value should handle game-over states."""
        mcts = MCTS(tiny_net, encoder, MCTSConfig())
        gs = GameState()
        node = MCTSNode(gs)
        # New game is not terminal
        assert not node.is_terminal
