"""Hive Board Game Engine - A fast, AI-friendly implementation."""

from hive_engine.hex_coord import HexCoord
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.board import Board
from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.neural_net import HiveNet, NetConfig, compute_loss
from hive_engine.elo import EloTracker
from hive_engine.device import get_device, device_summary

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
    # Neural network
    "HiveNet",
    "NetConfig",
    "compute_loss",
    # ELO
    "EloTracker",
    # Device
    "get_device",
    "device_summary",
]
