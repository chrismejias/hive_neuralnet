"""
Transformer encoder wrapper providing the same interface as HiveEncoder.

Allows MCTS and the trainer to use the transformer transparently
via duck typing. Action encoding/decoding is delegated to HiveEncoder.
"""

from __future__ import annotations

import numpy as np

from hive_engine.encoder import HiveEncoder
from hive_engine.game_state import GameState, Move

from hive_transformer.token_encoder import TokenEncoder
from hive_transformer.token_types import HiveTokenSequence


class TransformerEncoder:
    """
    Encoder wrapper for the transformer pipeline.

    Provides the same interface as HiveEncoder / GNNEncoder but returns
    HiveTokenSequence objects for state encoding.
    """

    ACTION_SPACE_SIZE: int = HiveEncoder.ACTION_SPACE_SIZE
    BOARD_SIZE: int = HiveEncoder.BOARD_SIZE
    NUM_CHANNELS: int = HiveEncoder.NUM_CHANNELS
    PASS_ACTION_INDEX: int = HiveEncoder.PASS_ACTION_INDEX

    def __init__(self) -> None:
        self._token_encoder = TokenEncoder()
        self._hive_encoder = HiveEncoder()

    def encode_state(self, game_state: GameState) -> HiveTokenSequence:
        """Encode a game state as a token sequence."""
        return self._token_encoder.encode(game_state)

    def encode_action(self, move: Move, game_state: GameState) -> int:
        """Encode a move as a flat action index. Delegates to HiveEncoder."""
        return self._hive_encoder.encode_action(move, game_state)

    def decode_action(
        self, action_index: int, game_state: GameState
    ) -> Move:
        """Decode a flat action index to a Move. Delegates to HiveEncoder."""
        return self._hive_encoder.decode_action(action_index, game_state)

    def get_legal_action_mask(
        self,
        game_state: GameState,
        legal_moves: list[Move] | None = None,
    ) -> np.ndarray:
        """Get binary mask over actions for legal moves. Delegates to HiveEncoder."""
        return self._hive_encoder.get_legal_action_mask(
            game_state, legal_moves
        )
