"""CPU iterative-deepening alpha-beta player backed by the native Hive engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
import torch

import hive_cpu_native
from hive_engine.game_state import GameState, Move, MoveType
from hive_engine.hex_coord import HexCoord
from hive_engine.pieces import PieceType
from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_gpu.endgame_generator import gamestate_to_gpu_bytes


_TURN_OFFSET: Final = 3412
_BOARD_SIZE: Final = 23
_HALF_BOARD: Final = 11
_MATE_SCORE: Final = 10.0
_INF: Final = 1000.0


class _Bound(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


@dataclass
class AlphaBetaConfig:
    """Search controls for the CPU alpha-beta player.

    ``node_budget`` bounds one move search.  The player returns the best move
    from the deepest *fully completed* iterative-deepening pass.
    """

    node_budget: int = 100_000
    max_depth: int = 32
    torch_threads: int | None = 1
    tt_max_entries: int = 500_000


@dataclass
class AlphaBetaStats:
    completed_depth: int = 0
    nodes: int = 0
    cutoffs: int = 0
    transposition_hits: int = 0
    value: float = 0.0


@dataclass
class _TTEntry:
    depth: int
    value: float
    bound: _Bound
    best_move: bytes | None


@dataclass
class _Expansion:
    moves: np.ndarray
    child_states: list[bytes]
    child_results: np.ndarray
    root_features: np.ndarray
    child_features: np.ndarray
    priors: np.ndarray


class _SearchAborted(RuntimeError):
    pass


class FNNAlphaBetaPlayer:
    """Depth-first negamax alpha-beta using the FNN value network at leaves.

    All rule work stays in ``hive_cpu_native``.  The FNN policy head is used
    only for move ordering; the value head supplies the leaf evaluation.
    """

    def __init__(self, net: HiveFNN, config: AlphaBetaConfig | None = None) -> None:
        self.ext = hive_cpu_native.load_extension()
        self.net = net.to("cpu").eval()
        self.config = config or AlphaBetaConfig()
        self.tt: dict[bytes, _TTEntry] = {}
        self._expansion_cache: dict[bytes, _Expansion] = {}
        self._value_cache: dict[bytes, float] = {}
        self.last_stats = AlphaBetaStats()
        if self.config.torch_threads is not None:
            torch.set_num_threads(max(1, int(self.config.torch_threads)))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        config: AlphaBetaConfig | None = None,
    ) -> "FNNAlphaBetaPlayer":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        net_config = checkpoint.get("net_config")
        if not isinstance(net_config, FNNConfig):
            net_config = FNNConfig.large()
        state_dict = (
            checkpoint.get("ema_state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("champion_state_dict")
        )
        if state_dict is None:
            raise KeyError("checkpoint missing model state")
        net = HiveFNN(net_config)
        net.load_state_dict(state_dict)
        return cls(net, config=config)

    def close(self) -> None:
        """Match the CPU-MCTS player interface used by the GUI."""

    def state_bytes_from_game(self, game: GameState) -> bytes:
        return gamestate_to_gpu_bytes(game)

    @staticmethod
    def _turn_from_state(state: bytes) -> int:
        return int(state[_TURN_OFFSET]) | (int(state[_TURN_OFFSET + 1]) << 8)

    def _terminal_value(self, state: bytes, result: int, ply: int) -> float:
        if result in (0, 3):
            return 0.0
        turn_is_white = (self._turn_from_state(state) & 1) == 0
        side_to_move_won = (result == 1 and turn_is_white) or (
            result == 2 and not turn_is_white
        )
        score = _MATE_SCORE - min(ply, 100) * 0.01
        return score if side_to_move_won else -score

    def _consume_node(self) -> None:
        self.last_stats.nodes += 1
        if self.last_stats.nodes > max(1, int(self.config.node_budget)):
            raise _SearchAborted

    def _expand(self, state: bytes) -> _Expansion:
        cached = self._expansion_cache.get(state)
        if cached is not None:
            return cached

        moves_pad, n_legal, root_features = self.ext.legal_moves_and_fnn_features(state)
        n_legal = int(n_legal)
        moves = np.asarray(moves_pad, dtype=np.uint8)[:n_legal].copy()
        root_features_np = np.asarray(root_features, dtype=np.float32).copy()
        if n_legal <= 0:
            expansion = _Expansion(
                moves=moves,
                child_states=[],
                child_results=np.zeros((0,), dtype=np.int32),
                root_features=root_features_np,
                child_features=np.zeros((0, root_features_np.shape[0]), dtype=np.float32),
                priors=np.zeros((0,), dtype=np.float32),
            )
            self._expansion_cache[state] = expansion
            return expansion

        child_features, child_states, child_results = self.ext.successor_features(
            state, moves, n_legal,
        )
        child_features_np = np.asarray(child_features, dtype=np.float32).copy()
        child_results_np = np.asarray(child_results, dtype=np.int32).copy()

        with torch.inference_mode():
            root_t = torch.from_numpy(root_features_np).unsqueeze(0)
            child_t = torch.from_numpy(child_features_np)
            root_embed = self.net.encode(root_t)
            child_embed = self.net.encode(child_t)
            logits = self.net.score_actions(
                root_embed.expand(n_legal, -1),
                child_embed,
                root_t.expand(n_legal, -1),
                child_t,
            )
            priors = torch.softmax(logits.float(), dim=0).cpu().numpy().astype(np.float32)

        expansion = _Expansion(
            moves=moves,
            child_states=list(child_states),
            child_results=child_results_np,
            root_features=root_features_np,
            child_features=child_features_np,
            priors=priors,
        )
        self._expansion_cache[state] = expansion
        return expansion

    def _leaf_value(self, state: bytes) -> float:
        cached = self._value_cache.get(state)
        if cached is not None:
            return cached
        expansion = self._expand(state)
        with torch.inference_mode():
            features = torch.from_numpy(expansion.root_features).unsqueeze(0)
            value = float(self.net.value_head(self.net.encode(features)).item())
        self._value_cache[state] = value
        return value

    @staticmethod
    def _move_key(move: np.ndarray) -> bytes:
        return bytes(np.asarray(move, dtype=np.uint8).tolist())

    def _ordered_indices(
        self,
        state: bytes,
        expansion: _Expansion,
        tt_move: bytes | None,
    ) -> list[int]:
        n = len(expansion.moves)
        if n == 0:
            return []
        scores = expansion.priors.astype(np.float64, copy=True)
        # A known terminal win must be tried before learned ordering.
        for idx, (child_state, result) in enumerate(
            zip(expansion.child_states, expansion.child_results)
        ):
            if int(result) == 0:
                continue
            if self._terminal_value(child_state, int(result), 0) < 0.0:
                scores[idx] += 100.0
        if tt_move is not None:
            for idx, move in enumerate(expansion.moves):
                if self._move_key(move) == tt_move:
                    scores[idx] += 1000.0
                    break
        return np.argsort(-scores, kind="stable").astype(np.int64).tolist()

    def _store_tt(
        self,
        state: bytes,
        depth: int,
        value: float,
        alpha_start: float,
        beta: float,
        best_move: bytes | None,
    ) -> None:
        if len(self.tt) >= max(1, int(self.config.tt_max_entries)):
            self.tt.clear()
        if value <= alpha_start:
            bound = _Bound.UPPER
        elif value >= beta:
            bound = _Bound.LOWER
        else:
            bound = _Bound.EXACT
        self.tt[state] = _TTEntry(depth, value, bound, best_move)

    def _negamax(
        self,
        state: bytes,
        result: int,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        self._consume_node()
        if result != 0:
            return self._terminal_value(state, result, ply)
        if depth <= 0:
            return self._leaf_value(state)

        tt_entry = self.tt.get(state)
        if tt_entry is not None and tt_entry.depth >= depth:
            self.last_stats.transposition_hits += 1
            if tt_entry.bound == _Bound.EXACT:
                return tt_entry.value
            if tt_entry.bound == _Bound.LOWER:
                alpha = max(alpha, tt_entry.value)
            else:
                beta = min(beta, tt_entry.value)
            if alpha >= beta:
                return tt_entry.value

        expansion = self._expand(state)
        if not expansion.child_states:
            return self._leaf_value(state)

        alpha_start = alpha
        best_value = -_INF
        best_move: bytes | None = None
        for idx in self._ordered_indices(state, expansion, tt_entry.best_move if tt_entry else None):
            child_value = -self._negamax(
                expansion.child_states[idx],
                int(expansion.child_results[idx]),
                depth - 1,
                -beta,
                -alpha,
                ply + 1,
            )
            if child_value > best_value:
                best_value = child_value
                best_move = self._move_key(expansion.moves[idx])
            alpha = max(alpha, best_value)
            if alpha >= beta:
                self.last_stats.cutoffs += 1
                break

        self._store_tt(state, depth, best_value, alpha_start, beta, best_move)
        return best_value

    def _search_root(
        self,
        state: bytes,
        result: int,
        depth: int,
        previous_best: bytes | None,
    ) -> tuple[bytes | None, float]:
        self._consume_node()
        if result != 0:
            return None, self._terminal_value(state, result, 0)
        expansion = self._expand(state)
        if not expansion.child_states:
            return None, self._leaf_value(state)

        tt_entry = self.tt.get(state)
        tt_move = previous_best or (tt_entry.best_move if tt_entry else None)
        best_move = None
        best_value = -_INF
        alpha = -_INF
        beta = _INF
        for idx in self._ordered_indices(state, expansion, tt_move):
            value = -self._negamax(
                expansion.child_states[idx],
                int(expansion.child_results[idx]),
                depth - 1,
                -beta,
                -alpha,
                1,
            )
            if value > best_value:
                best_value = value
                best_move = self._move_key(expansion.moves[idx])
            alpha = max(alpha, best_value)
        self.tt[state] = _TTEntry(depth, best_value, _Bound.EXACT, best_move)
        return best_move, best_value

    def choose_move_bytes(self, state: bytes) -> np.ndarray:
        result = int(self.ext.check_result(state))
        self.last_stats = AlphaBetaStats()
        self._expansion_cache.clear()
        self._value_cache.clear()

        root = self._expand(state)
        if not root.child_states:
            return np.zeros((int(self.ext.SIZEOF_GPU_MOVE),), dtype=np.uint8)
        fallback = self._move_key(root.moves[int(np.argmax(root.priors))])
        best_move = fallback
        best_value = self._leaf_value(state)
        for depth in range(1, max(1, int(self.config.max_depth)) + 1):
            try:
                move, value = self._search_root(state, result, depth, best_move)
            except _SearchAborted:
                break
            if move is not None:
                best_move = move
                best_value = value
                self.last_stats.completed_depth = depth
                self.last_stats.value = value
            if abs(value) >= _MATE_SCORE - 1.0:
                break

        for move in root.moves:
            if self._move_key(move) == best_move:
                return move.copy()
        return root.moves[0].copy()

    @staticmethod
    def _cpu_move_bytes(move: Move) -> bytes:
        if move.move_type == MoveType.PASS:
            return bytes((2, 0, 0, 0, 0, 0))
        if move.piece is None:
            raise ValueError("non-pass move is missing its piece")
        piece_type = int(move.piece.piece_type.value) + 1
        to_cell = (move.to.r + _HALF_BOARD) * _BOARD_SIZE + (move.to.q + _HALF_BOARD)
        if move.move_type == MoveType.PLACE:
            return bytes((0, piece_type, 0, 0, to_cell & 0xFF, to_cell >> 8))
        from_cell = (
            (move.from_pos.r + _HALF_BOARD) * _BOARD_SIZE
            + (move.from_pos.q + _HALF_BOARD)
        )
        return bytes((
            1,
            piece_type,
            from_cell & 0xFF,
            from_cell >> 8,
            to_cell & 0xFF,
            to_cell >> 8,
        ))

    def choose_move(self, game: GameState) -> Move:
        legal_moves = game.legal_moves()
        if not legal_moves:
            return Move(MoveType.PASS, None, HexCoord(0, 0))
        selected = bytes(self.choose_move_bytes(self.state_bytes_from_game(game)).tolist())
        for move in legal_moves:
            if self._cpu_move_bytes(move) == selected:
                return move
        return legal_moves[0]
