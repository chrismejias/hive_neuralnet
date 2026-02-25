"""
GNN encoder wrapper providing the same interface as HiveEncoder.

This thin wrapper allows MCTS and the trainer to use the GNN
transparently via duck typing. The action encoding/decoding is
delegated to HiveEncoder (pure coordinate math, shared between
CNN and GNN).

Usage:
    encoder = GNNEncoder()
    graph = encoder.encode_state(game_state)       # returns HiveGraph
    action = encoder.encode_action(move, game_state)  # returns int
    move = encoder.decode_action(action_index, game_state)  # returns Move
    mask = encoder.get_legal_action_mask(game_state)  # returns np.ndarray
"""

from __future__ import annotations

import numpy as np

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, Move

from hive_gnn.graph_encoder import GraphEncoder
from hive_gnn.graph_types import HiveGraph


class GNNEncoder:
    """
    Encoder wrapper for the GNN pipeline.

    Provides the same interface as HiveEncoder but returns HiveGraph
    objects for state encoding. Action encoding/decoding and legal
    mask generation are delegated to HiveEncoder.
    """

    # Re-export constants for compatibility
    ACTION_SPACE_SIZE: int = HiveEncoder.ACTION_SPACE_SIZE
    BOARD_SIZE: int = HiveEncoder.BOARD_SIZE
    NUM_CHANNELS: int = HiveEncoder.NUM_CHANNELS
    PASS_ACTION_INDEX: int = HiveEncoder.PASS_ACTION_INDEX

    def __init__(self) -> None:
        self._graph_encoder = GraphEncoder()
        self._hive_encoder = HiveEncoder()

    def encode_state(self, game_state: GameState) -> HiveGraph:
        """
        Encode a game state as a HiveGraph (instead of a tensor).

        Args:
            game_state: The current game state.

        Returns:
            A HiveGraph with node/edge features, positions, etc.
        """
        return self._graph_encoder.encode(game_state)

    def encode_action(self, move: Move, game_state: GameState) -> int:
        """
        Encode a move as a flat action index.

        Delegates to HiveEncoder — same coordinate-based encoding.

        Args:
            move: The move to encode.
            game_state: The current game state (for centroid).

        Returns:
            Integer action index in [0, ACTION_SPACE_SIZE).
        """
        return self._hive_encoder.encode_action(move, game_state)

    def decode_action(
        self, action_index: int, game_state: GameState
    ) -> Move:
        """
        Decode a flat action index back to a Move.

        Delegates to HiveEncoder.

        Args:
            action_index: Integer action index.
            game_state: The current game state.

        Returns:
            The corresponding Move object.
        """
        return self._hive_encoder.decode_action(action_index, game_state)

    def get_legal_action_mask(
        self,
        game_state: GameState,
        legal_moves: list[Move] | None = None,
    ) -> np.ndarray:
        """
        Get a binary mask over the action space for legal moves.

        Delegates to HiveEncoder.

        Args:
            game_state: The current game state.
            legal_moves: Optional pre-computed legal moves.

        Returns:
            Float32 array of shape (ACTION_SPACE_SIZE,), 1.0 for legal.
        """
        return self._hive_encoder.get_legal_action_mask(
            game_state, legal_moves
        )
