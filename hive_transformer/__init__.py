"""
hive_transformer — Transformer package for Hive AI.

An alternative to the CNN-based hive_engine neural network and the
GNN-based hive_gnn, using self-attention over token sequences for
global context and optimal GPU tensor core utilization.
"""

from hive_transformer.token_types import (
    HiveTokenSequence,
    HiveTokenBatch,
    TOKEN_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    OFF_BOARD_POSITION,
    TOKEN_TYPE_CLS,
    TOKEN_TYPE_BOARD,
    TOKEN_TYPE_HAND,
)
from hive_transformer.token_encoder import TokenEncoder
from hive_transformer.transformer_encoder import TransformerEncoder
from hive_transformer.transformer_net import (
    TransformerConfig,
    HiveTransformer,
    TransformerPolicyHead,
    TransformerValueHead,
)
from hive_transformer.transformer_replay_buffer import (
    TransformerTrainingExample,
    TokenReplayBuffer,
)
from hive_transformer.transformer_trainer import (
    TransformerTrainConfig,
    TransformerTrainer,
)

__all__ = [
    "HiveTokenSequence",
    "HiveTokenBatch",
    "TOKEN_FEAT_DIM",
    "GLOBAL_FEAT_DIM",
    "OFF_BOARD_POSITION",
    "TOKEN_TYPE_CLS",
    "TOKEN_TYPE_BOARD",
    "TOKEN_TYPE_HAND",
    "TokenEncoder",
    "TransformerEncoder",
    "TransformerConfig",
    "HiveTransformer",
    "TransformerPolicyHead",
    "TransformerValueHead",
    "TransformerTrainingExample",
    "TokenReplayBuffer",
    "TransformerTrainConfig",
    "TransformerTrainer",
]
