"""
hive_gnn — Graph Neural Network package for Hive AI.

An alternative to the CNN-based hive_engine neural network,
using message-passing GNNs with directional edge features to
preserve hex spatial geometry.
"""

from hive_gnn.graph_types import (
    HiveGraph,
    HiveGraphBatch,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
)
from hive_gnn.graph_encoder import GraphEncoder
from hive_gnn.gnn_encoder import GNNEncoder
from hive_gnn.gnn_net import GNNNetConfig, HiveGNN, MessagePassingLayer
from hive_gnn.gnn_replay_buffer import GNNTrainingExample, GraphReplayBuffer
from hive_gnn.gnn_trainer import GNNTrainConfig, GNNTrainer

__all__ = [
    "HiveGraph",
    "HiveGraphBatch",
    "NODE_FEAT_DIM",
    "EDGE_FEAT_DIM",
    "GLOBAL_FEAT_DIM",
    "GraphEncoder",
    "GNNEncoder",
    "GNNNetConfig",
    "HiveGNN",
    "MessagePassingLayer",
    "GNNTrainingExample",
    "GraphReplayBuffer",
    "GNNTrainConfig",
    "GNNTrainer",
]
