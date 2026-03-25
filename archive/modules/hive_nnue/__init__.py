"""
hive_nnue — NNUE-style evaluation network for Hive AI.

A lightweight MLP architecture using hand-crafted queen-relative
distance features, inspired by Stockfish NNUE.
"""

from hive_nnue.nnue_encoder import (
    NNUEFeatureEncoder,
    NNUEEncoder,
    FEATURE_DIM,
    FEATURES_PER_PIECE,
    NUM_PIECES_PER_PLAYER,
    NUM_DISTANCE_BUCKETS,
    NUM_GLOBAL_FEATURES,
)
from hive_nnue.nnue_net import NNUEConfig, HiveNNUE
from hive_nnue.nnue_replay_buffer import NNUETrainingExample, NNUEReplayBuffer
from hive_nnue.nnue_trainer import NNUETrainConfig, NNUETrainer

__all__ = [
    "NNUEFeatureEncoder",
    "NNUEEncoder",
    "FEATURE_DIM",
    "FEATURES_PER_PIECE",
    "NUM_PIECES_PER_PLAYER",
    "NUM_DISTANCE_BUCKETS",
    "NUM_GLOBAL_FEATURES",
    "NNUEConfig",
    "HiveNNUE",
    "NNUETrainingExample",
    "NNUEReplayBuffer",
    "NNUETrainConfig",
    "NNUETrainer",
]
