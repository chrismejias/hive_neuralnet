"""
hive_transformer — Transformer package for Hive AI.

GPU-native transformer using self-attention over token sequences for
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
from hive_transformer.transformer_net import (
    TransformerConfig,
    HiveTransformer,
    TransformerPolicyHead,
    TransformerValueHead,
    TransformerMobilityHead,
    TransformerQueenSurroundHead,
    TransformerFinalMobilityHead,
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
    "TransformerConfig",
    "HiveTransformer",
    "TransformerPolicyHead",
    "TransformerValueHead",
    "TransformerMobilityHead",
    "TransformerQueenSurroundHead",
    "TransformerFinalMobilityHead",
]
