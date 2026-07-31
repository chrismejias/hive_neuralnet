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
_PVS_EPSILON: Final = 1e-4


def normalize_alpha_beta_score(raw_score: float) -> tuple[float, int | None]:
    """Map native root scores to outcome value and optional mate distance.

    Learned/search values occupy [-1, 1]. Scores with magnitude >= 9 are
    proven terminal lines in the native +/-10, 0.01-per-ply mate band.
    """
    raw = float(raw_score)
    if abs(raw) >= 9.0:
        plies = max(0, int(round((10.0 - min(abs(raw), 10.0)) * 100.0)))
        return (1.0 if raw > 0.0 else -1.0), plies
    return max(-1.0, min(1.0, raw)), None


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
    pvs: bool = False
    native_tree: bool = True
    quiescence_plies: int = 0
    quiescence_budget_fraction: float = 0.2
    tactical_immobilization: bool = True
    tactical_opponent_surround: bool = True
    tactical_own_relief: bool = True
    tactical_queen_threat: bool = True
    recursive_threat_qsearch: bool = False
    forced_extensions: bool = False
    forced_extension_max_chain: int = 2

    @classmethod
    def from_profile(cls, name: str, **kwargs: object) -> "AlphaBetaConfig":
        profile = name.strip().lower()
        if profile in ("plain", "legacy"):
            return cls(
                quiescence_plies=0, tactical_immobilization=False,
                tactical_opponent_surround=False, tactical_own_relief=False,
                tactical_queen_threat=False, **kwargs,
            )
        if profile in ("baseline", "value-only"):
            return cls(**kwargs)
        if profile == "quiescence":
            return cls(quiescence_plies=1, **kwargs)
        if profile in ("threat", "full"):
            return cls(
                quiescence_plies=4, recursive_threat_qsearch=True,
                forced_extensions=True, **kwargs,
            )
        raise ValueError(
            f"unknown CPU alpha-beta profile {name!r}; expected plain, "
            "baseline, quiescence, threat, or full"
        )


@dataclass
class AlphaBetaStats:
    completed_depth: int = 0
    nodes: int = 0
    cutoffs: int = 0
    transposition_hits: int = 0
    pvs_researches: int = 0
    value: float = 0.0
    qnodes: int = 0
    tactical_moves: int = 0
    forced_extensions: int = 0


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


class _SearchAborted(RuntimeError):
    pass


class FNNAlphaBetaPlayer:
    """Depth-first negamax alpha-beta using the FNN value network at leaves.

    All rule work stays in ``hive_cpu_native``.  The FNN supplies only the
    scalar leaf evaluation; move ordering is search-derived or handcrafted.
    """

    def __init__(self, net: HiveFNN, config: AlphaBetaConfig | None = None) -> None:
        self.ext = hive_cpu_native.load_extension()
        self.net = net.to("cpu").eval()
        self.config = config or AlphaBetaConfig()
        self.tt: dict[bytes, _TTEntry] = {}
        self._expansion_cache: dict[bytes, _Expansion] = {}
        self._feature_cache: dict[bytes, np.ndarray] = {}
        self._embed_cache: dict[bytes, torch.Tensor] = {}
        self._value_cache: dict[bytes, float] = {}
        self._killers: dict[int, tuple[bytes, bytes | None]] = {}
        self._history: dict[bytes, int] = {}
        self.last_stats = AlphaBetaStats()
        state_dict = self.net.state_dict()
        weight_order = (
            "fc1.weight", "fc1.bias", "ln1.weight", "ln1.bias",
            "fc2.weight", "fc2.bias", "value_fc.weight", "value_fc.bias",
        )
        self._native_weights = np.concatenate([
            state_dict[name].detach().numpy().ravel() for name in weight_order
        ]).astype(np.float32, copy=False)
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
        root_features_np = self._feature_cache.get(state)
        if root_features_np is None:
            root_features_np = np.asarray(root_features, dtype=np.float32).copy()
            self._feature_cache[state] = root_features_np
        if n_legal <= 0:
            expansion = _Expansion(
                moves=moves,
                child_states=[],
                child_results=np.zeros((0,), dtype=np.int32),
                root_features=root_features_np,
            )
        else:
            child_states, child_results = self.ext.successors(
                state, moves, n_legal,
            )
            expansion = _Expansion(
                moves=moves,
                child_states=list(child_states),
                child_results=np.asarray(
                    child_results, dtype=np.int32,
                ).copy(),
                root_features=root_features_np,
            )
        self._expansion_cache[state] = expansion
        return expansion

    def _leaf_value(self, state: bytes) -> float:
        cached = self._value_cache.get(state)
        if cached is not None:
            return cached
        features_np = self._feature_cache.get(state)
        if features_np is None:
            _moves, _n_legal, features = self.ext.legal_moves_and_fnn_features(state)
            features_np = np.asarray(features, dtype=np.float32).copy()
            self._feature_cache[state] = features_np
        with torch.inference_mode():
            features_t = torch.from_numpy(features_np).unsqueeze(0)
            embed = self.net.encode(features_t)
            self._embed_cache[state] = embed
            value = float(self.net.value_head(embed).item())
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
        ply: int,
    ) -> list[int]:
        n = len(expansion.moves)
        if n == 0:
            return []
        scores = np.zeros((n,), dtype=np.float64)
        killer_pair = self._killers.get(ply)
        for idx, move in enumerate(expansion.moves):
            key = self._move_key(move)
            scores[idx] += float(self._history.get(key, 0))
            if killer_pair is not None:
                if key == killer_pair[0]:
                    scores[idx] += 1_000_000.0
                elif key == killer_pair[1]:
                    scores[idx] += 900_000.0
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
        for rank, idx in enumerate(self._ordered_indices(
            state, expansion, tt_entry.best_move if tt_entry else None, ply,
        )):
            if self.config.pvs and rank > 0:
                child_value = -self._negamax(
                    expansion.child_states[idx],
                    int(expansion.child_results[idx]),
                    depth - 1,
                    -alpha - _PVS_EPSILON,
                    -alpha,
                    ply + 1,
                )
                if alpha < child_value < beta:
                    self.last_stats.pvs_researches += 1
                    child_value = -self._negamax(
                        expansion.child_states[idx],
                        int(expansion.child_results[idx]),
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                    )
            else:
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
                cutoff_move = self._move_key(expansion.moves[idx])
                first, _second = self._killers.get(ply, (cutoff_move, None))
                if cutoff_move != first:
                    self._killers[ply] = (cutoff_move, first)
                else:
                    self._killers[ply] = (first, _second)
                self._history[cutoff_move] = min(
                    1_000_000,
                    self._history.get(cutoff_move, 0) + depth * depth,
                )
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
        for rank, idx in enumerate(self._ordered_indices(state, expansion, tt_move, 0)):
            if self.config.pvs and rank > 0:
                value = -self._negamax(
                    expansion.child_states[idx],
                    int(expansion.child_results[idx]),
                    depth - 1,
                    -alpha - _PVS_EPSILON,
                    -alpha,
                    1,
                )
                if value > alpha:
                    self.last_stats.pvs_researches += 1
                    value = -self._negamax(
                        expansion.child_states[idx],
                        int(expansion.child_results[idx]),
                        depth - 1,
                        -beta,
                        -alpha,
                        1,
                    )
            else:
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
        self._feature_cache.clear()
        self._embed_cache.clear()
        self._value_cache.clear()
        self._killers.clear()
        self._history.clear()

        if self.config.native_tree and result == 0:
            move, stats = self.ext.native_alpha_beta(
                state,
                self._native_weights,
                int(self.net.config.hidden_dim),
                int(self.net.config.embed_dim),
                int(self.config.node_budget),
                int(self.config.max_depth),
                int(self.config.quiescence_plies),
                float(self.config.quiescence_budget_fraction),
                ((1 if self.config.tactical_immobilization else 0)
                 | (2 if self.config.tactical_opponent_surround else 0)
                 | (4 if self.config.tactical_own_relief else 0)
                 | (8 if self.config.tactical_queen_threat else 0)),
                bool(self.config.recursive_threat_qsearch),
                (int(self.config.forced_extension_max_chain)
                 if self.config.forced_extensions else 0),
            )
            self.last_stats = AlphaBetaStats(
                completed_depth=int(stats["depth"]),
                nodes=int(stats["nodes"]),
                cutoffs=int(stats["cutoffs"]),
                transposition_hits=int(stats["tt_hits"]),
                value=float(stats["value"]),
                qnodes=int(stats["qnodes"]),
                tactical_moves=int(stats["tactical_moves"]),
                forced_extensions=int(stats["forced_extensions"]),
            )
            return np.asarray(move, dtype=np.uint8).copy()

        root = self._expand(state)
        if not root.child_states:
            return np.zeros((int(self.ext.SIZEOF_GPU_MOVE),), dtype=np.uint8)
        fallback = self._move_key(root.moves[0])
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
