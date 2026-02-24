"""
Monte Carlo Tree Search (MCTS) with PUCT for Hive.

Implements the AlphaZero-style MCTS algorithm:
  1. SELECT: Traverse tree using PUCT scores
  2. EXPAND: At leaf node, evaluate with neural network
  3. BACKPROPAGATE: Update visit counts and values up the tree

Value convention: each node stores its value from the perspective
of the player whose turn it is at that node. During selection, child
values are negated (since the child's "good" is the parent's "bad").
During backpropagation, the value alternates sign at each level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.pieces import Color


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_drop_move: int = 20


# ── MCTS Node ──────────────────────────────────────────────────────


class MCTSNode:
    """
    A node in the MCTS search tree.

    Each node stores:
      - The game state at this node
      - Visit count (N), total value (W), and prior probability (P)
      - Children indexed by action index

    Value convention: W and Q are from the perspective of the player
    whose turn it is at this node.
    """

    __slots__ = (
        "game_state",
        "parent",
        "parent_action",
        "children",
        "visit_count",
        "total_value",
        "prior",
        "is_expanded",
        "_legal_moves",
        "_legal_mask",
    )

    def __init__(
        self,
        game_state: GameState,
        parent: MCTSNode | None = None,
        parent_action: int | None = None,
        prior: float = 0.0,
    ) -> None:
        self.game_state = game_state
        self.parent = parent
        self.parent_action = parent_action
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False
        self._legal_moves: list[Move] | None = None
        self._legal_mask: np.ndarray | None = None

    @property
    def mean_value(self) -> float:
        """Q = W / N. Returns 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_terminal(self) -> bool:
        """Check if this node represents a finished game."""
        return self.game_state.result != GameResult.IN_PROGRESS


# ── MCTS Search ────────────────────────────────────────────────────


class MCTS:
    """
    Monte Carlo Tree Search with PUCT selection and neural network evaluation.

    Usage:
        mcts = MCTS(net, encoder, config)
        policy = mcts.search(game_state)
        # policy is shape (29407,) summing to ~1.0
    """

    def __init__(
        self,
        net,  # HiveNet — type hint omitted to avoid circular import
        encoder: HiveEncoder,
        config: MCTSConfig | None = None,
    ) -> None:
        self.net = net
        self.encoder = encoder
        self.config = config or MCTSConfig()

    def search(
        self,
        game_state: GameState,
        move_number: int = 0,
    ) -> np.ndarray:
        """
        Run MCTS from the given game state and return a policy vector.

        Args:
            game_state: The current game state to search from.
            move_number: Current move number in the game (for temperature).

        Returns:
            Policy vector of shape (ACTION_SPACE_SIZE,), sums to ~1.0.
        """
        # Create root node
        root = MCTSNode(game_state.copy())

        # Expand root
        self._expand(root)

        # Add Dirichlet noise to root priors for exploration
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = self._select(root)

            if node.is_terminal:
                value = self._terminal_value(node)
            else:
                value = self._expand(node)

            self._backpropagate(node, value)

        # Build policy from visit counts
        return self._get_policy(root, move_number)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a leaf node by traversing the tree with PUCT.

        Starting from the given node, repeatedly pick the child with
        the highest PUCT score until reaching an unexpanded or terminal node.
        """
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            # Pick child with highest PUCT score
            best_action = -1
            best_score = float("-inf")
            for action, child in node.children.items():
                score = self._puct_score(node, child)
                if score > best_score:
                    best_score = score
                    best_action = action
            node = node.children[best_action]
        return node

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand a node: evaluate with neural network, create children.

        Returns the value estimate from the neural network, from the
        perspective of the node's current player.
        """
        if node.is_terminal:
            return self._terminal_value(node)

        # Get legal moves and mask
        legal_moves = node.game_state.legal_moves()
        node._legal_moves = legal_moves
        legal_mask = self.encoder.get_legal_action_mask(
            node.game_state, legal_moves
        )
        node._legal_mask = legal_mask

        # Neural network evaluation
        state_tensor = self.encoder.encode_state(node.game_state)
        action_probs, value = self.net.predict(state_tensor, legal_mask)

        # Create child nodes for all legal actions
        for move in legal_moves:
            try:
                action_idx = self.encoder.encode_action(move, node.game_state)
            except ValueError:
                continue  # Skip moves that don't fit in grid

            if action_idx not in node.children:
                child_state = node.game_state.copy()
                child_state.apply_move(move)
                child = MCTSNode(
                    game_state=child_state,
                    parent=node,
                    parent_action=action_idx,
                    prior=action_probs[action_idx],
                )
                node.children[action_idx] = child

        node.is_expanded = True
        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate the value from a leaf node to the root.

        The value alternates sign at each level because what's good
        for one player is bad for the other.

        Convention: `value` is from the perspective of the player at
        `node`. As we go up, we negate it for the parent (who is the
        opponent).
        """
        current = node
        v = value
        while current is not None:
            current.visit_count += 1
            current.total_value += v
            v = -v  # Flip perspective for parent
            current = current.parent

    def _puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """
        Compute the PUCT score for a child node.

        PUCT = -Q_child + c_puct * P_child * sqrt(N_parent) / (1 + N_child)

        We negate Q because the child's value is from the child player's
        perspective, but the parent wants to pick the child that is
        worst for the child player (= best for the parent player).
        """
        q = -child.mean_value if child.visit_count > 0 else 0.0
        exploration = (
            self.config.c_puct
            * child.prior
            * math.sqrt(parent.visit_count)
            / (1 + child.visit_count)
        )
        return q + exploration

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """
        Add Dirichlet noise to root node priors for exploration.

        New prior = (1 - epsilon) * prior + epsilon * Dir(alpha)
        """
        if not node.children:
            return

        actions = list(node.children.keys())
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(actions)
        )

        eps = self.config.dirichlet_epsilon
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - eps) * child.prior + eps * noise[i]

    def _terminal_value(self, node: MCTSNode) -> float:
        """
        Get the value of a terminal node from the current player's perspective.

        Returns:
            +1 if the current player won
            -1 if the current player lost
             0 if draw
        """
        gs = node.game_state
        result = gs.result

        if result == GameResult.DRAW:
            return 0.0

        # Determine who won
        if result == GameResult.WHITE_WINS:
            winner = Color.WHITE
        elif result == GameResult.BLACK_WINS:
            winner = Color.BLACK
        else:
            return 0.0  # Shouldn't reach here for terminal nodes

        if winner == gs.current_player:
            return 1.0
        return -1.0

    def _get_policy(
        self, root: MCTSNode, move_number: int
    ) -> np.ndarray:
        """
        Build a policy vector from the root's child visit counts.

        Uses temperature to control exploration vs exploitation:
        - temperature > 0: proportional to N^(1/temp)
        - temperature = 0 (or >= temperature_drop_move): one-hot on argmax N
        """
        policy = np.zeros(
            self.encoder.ACTION_SPACE_SIZE, dtype=np.float32
        )

        if not root.children:
            return policy

        # Determine effective temperature
        temp = self.config.temperature
        if move_number >= self.config.temperature_drop_move:
            temp = 0.0

        if temp == 0.0:
            # Greedy: pick action with most visits
            best_action = max(
                root.children,
                key=lambda a: root.children[a].visit_count,
            )
            policy[best_action] = 1.0
        else:
            # Proportional to visit count raised to 1/temp
            visits = np.array(
                [
                    root.children[a].visit_count
                    for a in sorted(root.children.keys())
                ],
                dtype=np.float64,
            )
            actions = sorted(root.children.keys())

            if temp != 1.0:
                visits = visits ** (1.0 / temp)

            total = visits.sum()
            if total > 0:
                visits /= total
                for i, action in enumerate(actions):
                    policy[action] = visits[i]
            else:
                # Fallback: uniform
                for action in actions:
                    policy[action] = 1.0 / len(actions)

        return policy
