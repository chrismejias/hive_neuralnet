import numpy as np
import torch

from hive_engine.game_state import GameState, MoveType
from hive_engine.hex_coord import HexCoord
from hive_engine.pieces import PieceType
from hive_fnn.fnn_cpu_player import extract_fnn_features_cpu
from hive_fnn.fnn_features import FEAT_DIM
from hive_fnn.fnn_network import HiveFNN


def _place(game: GameState, piece_type: PieceType, destination: HexCoord) -> None:
    move = next(
        move
        for move in game.legal_moves()
        if move.move_type == MoveType.PLACE
        and move.piece is not None
        and move.piece.piece_type == piece_type
        and move.to == destination
    )
    game.apply_move(move)


def test_tactical_features_are_not_applicable_before_queen_placement() -> None:
    features = extract_fnn_features_cpu(GameState())

    assert features.shape == (FEAT_DIM,)
    assert FEAT_DIM == 140
    np.testing.assert_array_equal(features[122:140], np.zeros(18))


def test_tactical_features_are_one_hot_after_both_queens_are_placed() -> None:
    game = GameState()
    _place(game, PieceType.ANT, HexCoord(0, 0))
    _place(game, PieceType.ANT, HexCoord(1, 0))
    _place(game, PieceType.QUEEN, HexCoord(-1, 0))
    _place(game, PieceType.QUEEN, HexCoord(2, 0))

    features = extract_fnn_features_cpu(game)
    buckets = (
        features[122:125],
        features[125:128],
        features[128:131],
        features[131:134],
        features[134:137],
        features[137:140],
    )

    for bucket in buckets:
        assert bucket.sum() == 1.0
        assert set(bucket.tolist()) <= {0.0, 1.0}


def test_old_queen_throwable_weights_are_not_reused() -> None:
    old = torch.arange(2 * 124, dtype=torch.float32).reshape(2, 124)
    template = torch.empty((2, 140), dtype=torch.float32)

    migrated = HiveFNN._pad_loaded_weight(old, template, noise_scale=0.0)

    torch.testing.assert_close(migrated[:, :122], old[:, :122])
    assert torch.count_nonzero(migrated[:, 122:]) == 0
