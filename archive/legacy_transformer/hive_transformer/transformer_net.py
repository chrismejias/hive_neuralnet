"""Compatibility wrapper for the archived legacy transformer network."""

<<<<<<<< HEAD:hive_transformer/transformer_net.py
from archive.legacy_transformer.hive_transformer.transformer_net import (
    TransformerConfig,
    HiveTransformer,
    TransformerPolicyHead,
    TransformerValueHead,
    TransformerMobilityHead,
    TransformerQueenSurroundHead,
    TransformerFinalMobilityHead,
========
Architecture:
    HiveTokenBatch → token/position/type embeddings →
    TransformerEncoder (self-attention × N) →
        ├── PolicyHead (scatter to grid → conv → FC) → logits (29407,)
        ├── ValueHead (CLS token → FC → tanh) → scalar in [-1, 1]
        ├── MobilityHead (per-board-token MLP → binary logit) [optional]
        ├── QueenSurroundHead (per-board-token MLP → 2 logits) [optional]
        └── FinalMobilityHead (per-board-token MLP → binary logit) [optional]

Uses standard nn.TransformerEncoder with batch_first=True for
optimal GPU tensor core utilization and torch.compile() compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_engine.encoder import HiveEncoder

from hive_common.token_types import (
    TOKEN_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    OFF_BOARD_POSITION,
    TOKEN_TYPE_BOARD,
    HiveTokenSequence,
    HiveTokenBatch,
>>>>>>>> 7c7d146 (Refactor legacy transformer and MC packages):archive/legacy_transformer/hive_transformer/transformer_net.py
)

__all__ = [
    "TransformerConfig",
    "HiveTransformer",
    "TransformerPolicyHead",
    "TransformerValueHead",
    "TransformerMobilityHead",
    "TransformerQueenSurroundHead",
    "TransformerFinalMobilityHead",
]
