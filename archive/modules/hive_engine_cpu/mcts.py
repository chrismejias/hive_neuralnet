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

Performance features:
  - Virtual loss: each selected path gets a temporary penalty of -1
    so that concurrent threads diverge into different subtrees.
    The penalty is undone exactly during backpropagation.
  - Lazy child game states: child MCTSNodes are created without a
    copied GameState.  The state is computed on first expansion by
    copying the parent state and applying the stored move, saving
    O(branching_factor) copies per expansion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from archive.modules.hive_engine_cpu.encoder import HiveEncoder
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
    policy_prune_threshold: float = 0.0  # 0.0 = disabled


# ── MCTS Node ──────────────────────────────────────────────────────


class MCTSNode:
    """
    A node in the MCTS search tree.

    Each node stores:
      - The game state at this node (may be None until first expansion
        for non-root nodes — see lazy initialisation below)
      - Visit count (N), total value (W), and prior probability (P)
      - Children indexed by action index

    Value convention: W and Q are from the perspective of the player
    whose turn it is at this node.

    Lazy game states
    ----------------
    Child nodes created during expansion do not immediately copy the
    parent's GameState.  Instead they store a reference to their
    parent node and the Move that leads to them.  The actual
    GameState is computed and cached on the first call to
    ``ensure_game_state()``, which happens at the start of
    ``MCTS._expand()``.  This saves O(branching_factor) deep-copies
    per expansion; in practice only 5–15 % of children are ever
    selected, so most copies are avoided entirely.
    """

    __slots__ = (
        "game_state",
        "parent",
        "parent_action",
        "_parent_move",
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
        game_state: GameState | None = None,
        parent: MCTSNode | None = None,
        parent_action: int | None = None,
        prior: float = 0.0,
        parent_move: Move | None = None,
    ) -> None:
        self.game_state = game_state      # None iff this is a lazy child node
        self.parent = parent
        self.parent_action = parent_action
        self._parent_move = parent_move   # Move stored for lazy state init
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False
        self._legal_moves: list[Move] | None = None
        self._legal_mask: np.ndarray | None = None

    # ── Lazy state initialisation ──────────────────────────────────

    def ensure_game_state(self) -> GameState:
        """
        Return this node's GameState, computing it lazily if needed.

        For the root node the state is always pre-initialised.
        For child nodes created during expansion the state is computed
        here by copying the parent's state and applying ``_parent_move``.
        The result is cached so subsequent calls are free.
        """
        if self.game_state is None:
            assert self.parent is not None, "Lazy node must have a parent"
            parent_gs = self.parent.game_state
            assert parent_gs is not None, (
                "Parent game state must be initialised before child"
            )
            self.game_state = parent_gs.copy()
            self.game_state.apply_move(self._parent_move)
        return self.game_state

    # ── Properties ────────────────────────────────────────────────

    @property
    def mean_value(self) -> float:
        """Q = W / N. Returns 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_terminal(self) -> bool:
        """
        Check if this node represents a finished game.

        Returns False for lazy nodes whose game state has not yet been
        computed (they cannot be terminal until expanded).
        """
        if self.game_state is None:
            return False
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

        # Expand root (single-threaded; no virtual loss needed here)
        self._expand(root)

        # Add Dirichlet noise to root priors for exploration
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node, vl_path = self._select(root)

            if node.is_terminal:
                value = self._terminal_value(node)
            else:
                value = self._expand(node)

            self._backpropagate(node, value, vl_path)

        # Build policy from visit counts
        return self._get_policy(root, move_number)

    def _select(
        self, node: MCTSNode
    ) -> tuple[MCTSNode, list[MCTSNode]]:
        """
        Select a leaf node by traversing the tree with PUCT.

        Applies a virtual loss of -1 to every node moved through so
        that concurrent MCTS threads are steered toward different
        subtrees.  The virtual loss is undone in ``_backpropagate``.

        Returns:
            (leaf_node, vl_path) where vl_path contains every node
            that received a virtual loss and must be cleaned up.
        """
        vl_path: list[MCTSNode] = []

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

            # Virtual loss: make this path look pessimistic to other
            # threads that may be selecting concurrently.
            node.visit_count += 1
            node.total_value -= 1.0
            vl_path.append(node)

        return node, vl_path

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand a node: evaluate with neural network, create children.

        Computes the node's game state lazily if not yet initialised,
        then calls the neural network and creates lazy child nodes
        (children store only their parent reference and the move — no
        game-state copy is made until a child is itself expanded).

        Returns the value estimate from the neural network, from the
        perspective of the node's current player.
        """
        # Materialise lazy game state (no-op for the root)
        node.ensure_game_state()

        if node.is_terminal:
            node.is_expanded = True
            return self._terminal_value(node)

        # Guard against double-expansion by concurrent threads
        if node.is_expanded:
            return 0.0

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

        # Create child nodes lazily: store parent + move instead of
        # copying the game state now.  The copy is deferred until the
        # child is first selected for expansion.
        for move in legal_moves:
            try:
                action_idx = self.encoder.encode_action(move, node.game_state)
            except ValueError:
                continue  # Skip moves that don't fit in grid

            if action_idx not in node.children:
                child = MCTSNode(
                    game_state=None,           # Lazy: computed on first expand
                    parent=node,
                    parent_action=action_idx,
                    prior=action_probs[action_idx],
                    parent_move=move,          # Stored for lazy state init
                )
                node.children[action_idx] = child

        node.is_expanded = True
        return value

    def _backpropagate(
        self,
        node: MCTSNode,
        value: float,
        vl_path: list[MCTSNode],
    ) -> None:
        """
        Backpropagate the value from a leaf node to the root.

        First removes the virtual loss from every node on ``vl_path``
        (restoring the counts to their pre-selection state), then
        applies the real value using the standard alternating-sign
        convention.

        The value alternates sign at each level because what's good
        for one player is bad for the other.
        """
        # Undo virtual loss applied during _select
        for vl_node in vl_path:
            vl_node.visit_count -= 1
            vl_node.total_value += 1.0

        # Standard backpropagation: propagate value up to root
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

        # Policy target pruning: zero out low-probability actions, renormalize
        if self.config.policy_prune_threshold > 0.0:
            below = policy < self.config.policy_prune_threshold
            policy[below] = 0.0
            total = policy.sum()
            if total > 0:
                policy /= total

        return policy
