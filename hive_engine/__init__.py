"""Hive Board Game Engine - A fast, AI-friendly implementation."""

from hive_engine.hex_coord import HexCoord
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.board import Board
from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.encoder import HiveEncoder
from hive_engine.endgame import generate_endgame
from hive_engine.neural_net import HiveNet, NetConfig, compute_loss
from hive_engine.mcts import MCTS, MCTSConfig, MCTSNode
from hive_engine.metrics import IterationMetrics, MetricsLogger
from hive_engine.elo import EloTracker
from hive_engine.tb_logger import TBLogger
from hive_engine.augment import augment_example, rotate_state, rotate_policy
from hive_engine.curriculum import CurriculumConfig, CurriculumPhase, run_curriculum
from hive_engine.device import get_device, device_summary
from hive_engine.trainer import (
    Trainer,
    TrainConfig,
    TrainingExample,
    ReplayBuffer,
    SelfPlayStats,
    TrainStats,
)

__all__ = [
    # Core engine
    "HexCoord",
    "Color",
    "PieceType",
    "Piece",
    "Board",
    "GameState",
    "GameResult",
    "Move",
    "MoveType",
    # Encoder
    "HiveEncoder",
    # Endgame
    "generate_endgame",
    # Neural network
    "HiveNet",
    "NetConfig",
    "compute_loss",
    # MCTS
    "MCTS",
    "MCTSConfig",
    "MCTSNode",
    # Metrics
    "IterationMetrics",
    "MetricsLogger",
    # ELO
    "EloTracker",
    # TensorBoard
    "TBLogger",
    # Augmentation
    "augment_example",
    "rotate_state",
    "rotate_policy",
    # Curriculum
    "CurriculumConfig",
    "CurriculumPhase",
    "run_curriculum",
    # Device
    "get_device",
    "device_summary",
    # Training
    "Trainer",
    "TrainConfig",
    "TrainingExample",
    "ReplayBuffer",
    "SelfPlayStats",
    "TrainStats",
]
