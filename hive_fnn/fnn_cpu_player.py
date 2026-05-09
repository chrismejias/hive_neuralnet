"""CPU-only single-game FNN player.

This module provides a pure-Python / PyTorch-CPU fallback for playing or
analyzing individual Hive games with the FNN model. It does not touch the
GPU-native self-play training pipeline.

Design:
  - Recreates the 110-dim FNN feature vector from the Python ``GameState``.
  - Expands one game tree at a time using CPU PUCT.
  - Batches successor-state scoring through the existing FNN network.

This is intentionally a single-game engine. It is not meant to replace the
GPU training/search path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Hashable
import multiprocessing as mp

import numpy as np
import torch

from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.hex_coord import HexCoord
from hive_engine.pieces import Color, PieceType, Piece, ExpansionConfig
from hive_fnn.fnn_features import FEAT_DIM
from hive_fnn.fnn_network import FNNConfig, HiveFNN


_DRAW_MOVE_LIMIT = 200
_PIECE_TYPES: tuple[PieceType, ...] = tuple(PieceType)
_CPU_WORKER_PLAYER: "FNNCPUPlayer | None" = None
_CPU_WORKER_ROOT: "_Node | None" = None


def _type_color_index(piece_type: PieceType, color: Color) -> int:
    return piece_type.value * 2 + int(color)


def _serialize_piece(piece: Piece) -> tuple[int, int, int]:
    return (piece.piece_type.value, int(piece.color), int(piece.piece_id))


def _deserialize_piece(spec: tuple[int, int, int]) -> Piece:
    pt, color, piece_id = spec
    return Piece(PieceType(pt), Color(color), piece_id)


def _serialize_game_state(game: GameState) -> dict:
    return {
        "expansions": (
            bool(game.expansions.mosquito),
            bool(game.expansions.ladybug),
            bool(game.expansions.pillbug),
        ),
        "turn": int(game.turn),
        "result": int(game.result),
        "queen_placed": (
            bool(game._queen_placed[Color.WHITE]),
            bool(game._queen_placed[Color.BLACK]),
        ),
        "board": [
            (
                (pos.q, pos.r),
                [_serialize_piece(piece) for piece in stack],
            )
            for pos, stack in game.board.grid.items()
        ],
        "hands": {
            int(color): [_serialize_piece(piece) for piece in game.hand(color)]
            for color in (Color.WHITE, Color.BLACK)
        },
    }


def _deserialize_game_state(payload: dict) -> GameState:
    expansions = ExpansionConfig(
        mosquito=bool(payload["expansions"][0]),
        ladybug=bool(payload["expansions"][1]),
        pillbug=bool(payload["expansions"][2]),
    )
    game = GameState(expansions=expansions)
    game.turn = int(payload["turn"])
    game.result = GameResult(int(payload["result"]))
    game._queen_placed = {
        Color.WHITE: bool(payload["queen_placed"][0]),
        Color.BLACK: bool(payload["queen_placed"][1]),
    }
    game._hands = {
        Color.WHITE: [_deserialize_piece(spec) for spec in payload["hands"][int(Color.WHITE)]],
        Color.BLACK: [_deserialize_piece(spec) for spec in payload["hands"][int(Color.BLACK)]],
    }
    game.board.grid = {}
    game.board.piece_positions = {}
    game.board._ap_cache = None
    game.board._ap_dirty = True
    game.board._bct_blocks = []
    game.board._bct_vertex_blocks = {}
    game.board._bct_cut_vertices = set()
    game.board._bct_valid = False
    for pos_spec, stack_specs in payload["board"]:
        pos = HexCoord(int(pos_spec[0]), int(pos_spec[1]))
        stack = [_deserialize_piece(spec) for spec in stack_specs]
        game.board.grid[pos] = stack
        for piece in stack:
            game.board.piece_positions[piece] = pos
    game._legal_moves_cache = None
    return game


def _queen_positions(game: GameState) -> dict[Color, object | None]:
    pos_by_color: dict[Color, object | None] = {
        Color.WHITE: None,
        Color.BLACK: None,
    }
    for pos, stack in game.board.grid.items():
        for piece in stack:
            if piece.piece_type == PieceType.QUEEN:
                pos_by_color[piece.color] = pos
    return pos_by_color


def _state_cache_key(game: GameState) -> Hashable:
    board_items = []
    for pos in sorted(game.board.grid.keys()):
        stack = game.board.grid[pos]
        board_items.append(
            (
                pos.q,
                pos.r,
                tuple(_serialize_piece(piece) for piece in stack),
            )
        )
    hand_items = (
        tuple(sorted(_serialize_piece(piece) for piece in game.hand(Color.WHITE))),
        tuple(sorted(_serialize_piece(piece) for piece in game.hand(Color.BLACK))),
    )
    return (
        int(game.turn),
        int(game.result),
        bool(game.expansions.mosquito),
        bool(game.expansions.ladybug),
        bool(game.expansions.pillbug),
        bool(game._queen_placed[Color.WHITE]),
        bool(game._queen_placed[Color.BLACK]),
        tuple(board_items),
        hand_items,
    )


def extract_fnn_features_cpu(
    game: GameState,
    legal_moves: list[Move] | None = None,
) -> np.ndarray:
    """Recreate the 110-dim FNN feature vector on CPU."""
    if legal_moves is None:
        legal_moves = game.legal_moves()

    board = game.board
    features = np.zeros(FEAT_DIM, dtype=np.float32)

    queen_pos = _queen_positions(game)
    ap = board.find_articulation_points()
    dist_sum = np.zeros((2, len(_PIECE_TYPES)), dtype=np.float32)
    dist_count = np.zeros((2, len(_PIECE_TYPES)), dtype=np.int32)
    pillbug_capable_cells: dict[Color, list] = {
        Color.WHITE: [],
        Color.BLACK: [],
    }

    # count_on_board / queen_neighbors / avg_dist_to_opp_q /
    # articulation_count / num_single / queen_covered / pillbug_capable
    for pos, stack in board.grid.items():
        top = stack[-1]
        idx = _type_color_index(top.piece_type, top.color)
        features[idx] += 1.0

        if board.num_occupied_neighbors(pos) == 0:
            features[96 + int(top.color)] += 1.0

        opp_q = queen_pos[top.color.other()]
        if opp_q is not None:
            if any(n == opp_q for n in pos.neighbors()):
                features[32 + idx] += 1.0
            dist_sum[int(top.color), top.piece_type.value] += pos.distance(opp_q)
            dist_count[int(top.color), top.piece_type.value] += 1

        if len(stack) == 1 and pos in ap:
            features[80 + idx] += 1.0

        if top.piece_type != PieceType.QUEEN:
            for below in stack[:-1]:
                if below.piece_type == PieceType.QUEEN:
                    features[98 + int(below.color)] = 1.0

        is_capable = top.piece_type == PieceType.PILLBUG
        if (
            not is_capable
            and top.piece_type == PieceType.MOSQUITO
            and len(stack) == 1
        ):
            for n in pos.neighbors():
                n_top = board.top_piece_at(n)
                if n_top is not None and n_top.piece_type == PieceType.PILLBUG:
                    is_capable = True
                    break
        if is_capable:
            features[104 + int(top.color)] = 1.0
            pillbug_capable_cells[top.color].append(pos)

    # avg_dist_to_opp_q
    for color in (Color.WHITE, Color.BLACK):
        ci = int(color)
        for pt in _PIECE_TYPES:
            count = dist_count[ci, pt.value]
            if count > 0:
                features[48 + _type_color_index(pt, color)] = (
                    dist_sum[ci, pt.value] / float(count) / 10.0
                )

    # count_in_hand
    for color in (Color.WHITE, Color.BLACK):
        hand_counts = {pt: 0 for pt in _PIECE_TYPES}
        for piece in game.hand(color):
            hand_counts[piece.piece_type] += 1
        for pt in _PIECE_TYPES:
            denom = max(1, pt.count_per_player)
            features[16 + _type_color_index(pt, color)] = (
                hand_counts[pt] / float(denom)
            )

    # can_move_count / num_placement_pos, based only on current legal moves
    movable_pieces: dict[tuple[PieceType, Color], set[Piece]] = {}
    seen_place_dest: set = set()
    cur = game.current_player
    for move in legal_moves:
        if move.move_type == MoveType.MOVE and move.piece is not None:
            key = (move.piece.piece_type, move.piece.color)
            movable_pieces.setdefault(key, set()).add(move.piece)
        elif move.move_type == MoveType.PLACE and move.to is not None:
            seen_place_dest.add(move.to)

    for (piece_type, color), pieces in movable_pieces.items():
        features[64 + _type_color_index(piece_type, color)] = float(len(pieces))
    features[100 + int(cur)] = float(len(seen_place_dest)) / 10.0

    # draw and move clocks
    moves_left = max(0, _DRAW_MOVE_LIMIT - game.turn)
    features[102] = moves_left / float(_DRAW_MOVE_LIMIT)
    features[103] = min(game.turn / 100.0, 1.0)

    # throwable_own / throwable_opp
    for color in (Color.WHITE, Color.BLACK):
        for pos in pillbug_capable_cells[color]:
            for n in pos.neighbors():
                top = board.top_piece_at(n)
                if top is None:
                    continue
                if top.color == color:
                    features[106 + int(color)] += 1.0
                else:
                    features[108 + int(top.color)] += 1.0

    return features


@dataclass
class FNNCPUMCTSConfig:
    num_simulations: int = 256
    c_puct: float = 1.25
    c_visit: float = 50.0
    c_scale: float = 1.0
    temperature: float = 0.0
    temperature_drop_move: int = 20
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0
    use_gumbel_root: bool = False
    gumbel_considered: int = 16
    gumbel_noise_scale: float = 1.0
    root_parallel_workers: int = 2
    torch_threads: int | None = None


class _Node:
    __slots__ = (
        "state",
        "parent",
        "move_from_parent",
        "state_key",
        "root_feature",
        "prior",
        "visit_count",
        "total_value",
        "children",
        "legal_moves",
        "expanded",
    )

    def __init__(
        self,
        state: GameState | None,
        *,
        parent: _Node | None = None,
        move_from_parent: Move | None = None,
        state_key: Hashable | None = None,
        root_feature: np.ndarray | None = None,
        prior: float = 0.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.state_key = state_key
        self.root_feature = root_feature
        self.prior = float(prior)
        self.visit_count = 0
        self.total_value = 0.0
        self.children: list[_Node] = []
        self.legal_moves: list[Move] | None = None
        self.expanded = False

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ensure_state(self) -> GameState:
        if self.state is not None:
            return self.state
        if self.parent is None or self.move_from_parent is None:
            raise RuntimeError("Lazy node missing parent or move")
        parent_state = self.parent.ensure_state()
        state = parent_state.copy()
        state.apply_move(self.move_from_parent)
        self.state = state
        if self.state_key is None:
            self.state_key = _state_cache_key(state)
        return state


class _CandidateWorker:
    """Persistent worker process for one candidate root subtree at a time."""

    def __init__(
        self,
        checkpoint_path: str,
        torch_threads: int | None,
        c_puct: float,
    ) -> None:
        parent_conn, child_conn = mp.Pipe()
        self._conn = parent_conn
        self._proc = mp.Process(
            target=_candidate_worker_main,
            args=(child_conn, checkpoint_path, torch_threads, c_puct),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()

    def reset(self, state_payload: dict) -> None:
        self._conn.send(("reset", state_payload))
        msg = self._conn.recv()
        if msg[0] != "ready":
            raise RuntimeError(f"Unexpected worker response: {msg}")

    def run(self, extra_sims: int) -> tuple[float, float]:
        self._conn.send(("run", int(extra_sims)))
        msg, visits, q_parent = self._conn.recv()
        if msg != "stats":
            raise RuntimeError(f"Unexpected worker response: {msg}")
        return float(visits), float(q_parent)

    def stats(self) -> tuple[float, float]:
        self._conn.send(("stats",))
        msg, visits, q_parent = self._conn.recv()
        if msg != "stats":
            raise RuntimeError(f"Unexpected worker response: {msg}")
        return float(visits), float(q_parent)

    def close(self) -> None:
        try:
            if self._proc.is_alive():
                self._conn.send(("close",))
        except (BrokenPipeError, EOFError, OSError):
            pass
        try:
            self._conn.close()
        except OSError:
            pass
        self._proc.join(timeout=1.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)


class FNNCPUPlayer:
    """Single-game CPU FNN player with batched successor scoring."""

    def __init__(
        self,
        net: HiveFNN,
        config: FNNCPUMCTSConfig | None = None,
    ) -> None:
        self.net = net.to("cpu")
        self.net.eval()
        self.config = config or FNNCPUMCTSConfig()
        self._checkpoint_path: str | None = None
        self._feature_cache: dict[Hashable, np.ndarray] = {}
        self._candidate_worker_pool: list[_CandidateWorker] = []
        if self.config.torch_threads is not None:
            torch.set_num_threads(max(1, int(self.config.torch_threads)))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        config: FNNCPUMCTSConfig | None = None,
    ) -> FNNCPUPlayer:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        net_config = ckpt.get("net_config")
        if not isinstance(net_config, FNNConfig):
            net_config = FNNConfig.large()
        net = HiveFNN(net_config)
        net.load_state_dict(ckpt["model_state_dict"])
        player = cls(net, config=config)
        player._checkpoint_path = checkpoint_path
        return player

    def _ensure_candidate_worker_pool(self, size: int) -> None:
        if self._checkpoint_path is None or size <= 0:
            return
        while len(self._candidate_worker_pool) < size:
            self._candidate_worker_pool.append(
                _CandidateWorker(
                    self._checkpoint_path,
                    self.config.torch_threads,
                    float(self.config.c_puct),
                )
            )

    def close(self) -> None:
        for worker in self._candidate_worker_pool:
            worker.close()
        self._candidate_worker_pool.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _terminal_value(self, state: GameState) -> float:
        if state.result == GameResult.DRAW:
            return 0.0
        winner = (
            Color.WHITE if state.result == GameResult.WHITE_WINS else Color.BLACK
        )
        return 1.0 if state.current_player == winner else -1.0

    def _search_root_value(
        self,
        state: GameState,
        *,
        num_simulations: int | None = None,
        c_puct: float | None = None,
    ) -> tuple[int, float]:
        """Run a standalone PUCT search from ``state`` and return (visits, Q).

        Q is from the state's current-player perspective.
        """
        cfg = self.config
        n_sims = int(num_simulations if num_simulations is not None else cfg.num_simulations)
        if state.result != GameResult.IN_PROGRESS:
            return 0, self._terminal_value(state)

        state_copy = state.copy()
        root = _Node(
            state_copy,
            state_key=_state_cache_key(state_copy),
        )
        local_c_puct = float(c_puct if c_puct is not None else cfg.c_puct)
        saved = cfg.c_puct
        cfg.c_puct = local_c_puct
        try:
            self._run_search_on_existing_root(root, n_sims)
        finally:
            cfg.c_puct = saved
        return root.visit_count, root.mean_value

    def _run_search_on_existing_root(self, root: _Node, num_simulations: int) -> None:
        """Continue searching an existing root in-place."""
        if num_simulations <= 0:
            return
        if not root.expanded:
            self._expand(root)
        for _ in range(int(num_simulations)):
            leaf = self._select(root)
            value = self._expand(leaf)
            self._backpropagate(leaf, value)

    def _evaluate_state(
        self,
        node: _Node,
    ) -> tuple[list[Move], np.ndarray, float, list[np.ndarray], list[Hashable]]:
        state = node.ensure_state()
        legal_moves = state.legal_moves()
        if node.state_key is None:
            node.state_key = _state_cache_key(state)
        root_feat = node.root_feature
        if root_feat is None:
            root_feat = self._feature_cache.get(node.state_key)
        if root_feat is None:
            root_feat = extract_fnn_features_cpu(state, legal_moves)
            self._feature_cache[node.state_key] = root_feat
        node.root_feature = root_feat

        succ_features: list[np.ndarray] = []
        child_keys: list[Hashable] = []
        for move in legal_moves:
            child = state.copy()
            child.apply_move(move)
            child_key = _state_cache_key(child)
            child_keys.append(child_key)
            child_feat = self._feature_cache.get(child_key)
            if child_feat is None:
                child_legal = child.legal_moves()
                child_feat = extract_fnn_features_cpu(child, child_legal)
                self._feature_cache[child_key] = child_feat
            succ_features.append(child_feat)

        with torch.inference_mode():
            root_t = torch.from_numpy(root_feat).unsqueeze(0)
            root_emb = self.net.encode(root_t)
            root_value = float(self.net.value_head(root_emb).item())

            if succ_features:
                succ_t = torch.from_numpy(np.stack(succ_features, axis=0))
                succ_emb = self.net.encode(succ_t)
                root_rep = root_emb.expand(succ_emb.shape[0], -1)
                logits = self.net.score_actions(root_rep, succ_emb)
                priors = torch.softmax(logits, dim=0).cpu().numpy()
            else:
                priors = np.zeros((0,), dtype=np.float32)

        return legal_moves, priors.astype(np.float32), root_value, succ_features, child_keys

    def _expand(self, node: _Node) -> float:
        state = node.ensure_state()
        if state.result != GameResult.IN_PROGRESS:
            node.expanded = True
            node.legal_moves = []
            return self._terminal_value(state)

        if node.expanded:
            return 0.0

        legal_moves, priors, value, succ_features, child_keys = self._evaluate_state(node)
        node.legal_moves = legal_moves
        node.children = [
            _Node(
                None,
                parent=node,
                move_from_parent=move,
                state_key=child_key,
                root_feature=child_feat,
                prior=float(prior),
            )
            for move, prior, child_feat, child_key in zip(
                legal_moves, priors, succ_features, child_keys,
            )
        ]
        node.expanded = True
        if node.parent is None and self.config.dirichlet_epsilon > 0.0 and node.children:
            noise = np.random.dirichlet(
                [self.config.dirichlet_alpha] * len(node.children),
            )
            eps = self.config.dirichlet_epsilon
            for child, n in zip(node.children, noise):
                child.prior = (1.0 - eps) * child.prior + eps * float(n)
        return value

    def _select(self, root: _Node) -> _Node:
        node = root
        while node.expanded and node.children and node.state.result == GameResult.IN_PROGRESS:
            sqrt_parent = math.sqrt(max(1, node.visit_count))
            best_child = None
            best_score = float("-inf")
            for child in node.children:
                q = -child.mean_value
                u = self.config.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
        return node

    def _gumbel_values_for_candidate_roots(
        self,
        candidate_roots: list[_Node],
        incremental_sims: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (visits, q_parent) for each candidate subtree root.

        Searches continue from the existing candidate roots in place. q_parent
        is from the original parent/root player's perspective, so it is the
        negated candidate-root Q.
        """
        visits = np.zeros(len(candidate_roots), dtype=np.float32)
        q_parent = np.zeros(len(candidate_roots), dtype=np.float32)

        for i, (root, extra_sims) in enumerate(zip(candidate_roots, incremental_sims)):
            self._run_search_on_existing_root(root, int(extra_sims))
            visits[i] = float(root.visit_count)
            q_parent[i] = float(-root.mean_value)

        return visits, q_parent

    def _backpropagate(self, node: _Node, value: float) -> None:
        cur = node
        v = value
        while cur is not None:
            cur.visit_count += 1
            cur.total_value += v
            v = -v
            cur = cur.parent

    def search(self, state: GameState) -> tuple[list[Move], np.ndarray]:
        if self.config.use_gumbel_root:
            return self._search_gumbel(state)

        state_copy = state.copy()
        root = _Node(
            state_copy,
            state_key=_state_cache_key(state_copy),
        )
        self._run_search_on_existing_root(root, self.config.num_simulations)

        if not root.children:
            return [Move(MoveType.PASS)], np.array([1.0], dtype=np.float32)

        visits = np.array([child.visit_count for child in root.children], dtype=np.float32)
        temperature = self.config.temperature
        if temperature <= 1e-8:
            probs = np.zeros_like(visits)
            probs[int(np.argmax(visits))] = 1.0
        else:
            scaled = np.power(np.maximum(visits, 1e-8), 1.0 / temperature)
            probs = scaled / np.maximum(scaled.sum(), 1e-8)
        return root.legal_moves or [], probs

    def _search_gumbel(self, state: GameState) -> tuple[list[Move], np.ndarray]:
        """CPU Gumbel-root search with non-root PUCT."""
        if state.result != GameResult.IN_PROGRESS:
            return [Move(MoveType.PASS)], np.array([1.0], dtype=np.float32)

        root_state = state.copy()
        root_node = _Node(root_state, state_key=_state_cache_key(root_state))
        legal_moves, priors, _value, succ_features, child_keys = self._evaluate_state(root_node)
        if not legal_moves:
            return [Move(MoveType.PASS)], np.array([1.0], dtype=np.float32)

        k = min(int(self.config.gumbel_considered), len(legal_moves))
        rounds = max(1, math.ceil(math.log2(max(k, 1))))
        sims_per_round = max(1, int(self.config.num_simulations) // rounds)

        log_priors = np.log(np.clip(priors, 1e-30, None))
        gumbel = (
            -np.log(-np.log(np.clip(np.random.random(len(legal_moves)), 1e-12, 1.0 - 1e-12)))
            * float(self.config.gumbel_noise_scale)
        ).astype(np.float32)
        base_score = log_priors + gumbel

        candidate_idx = np.argsort(base_score)[-k:][::-1].tolist()
        candidate_roots = {
            idx: _Node(
                None,
                parent=root_node,
                move_from_parent=legal_moves[idx],
                state_key=child_keys[idx],
                root_feature=succ_features[idx],
                prior=float(priors[idx]),
            )
            for idx in candidate_idx
        }
        candidate_workers: dict[int, _CandidateWorker] = {}

        use_parallel_workers = (
            self.config.root_parallel_workers > 0
            and self._checkpoint_path is not None
        )

        if use_parallel_workers:
            pool_size = min(int(self.config.root_parallel_workers), len(candidate_idx))
            self._ensure_candidate_worker_pool(pool_size)
            for worker, idx in zip(self._candidate_worker_pool[:pool_size], candidate_idx):
                worker.reset(_serialize_game_state(candidate_roots[idx].ensure_state()))
                candidate_workers[idx] = worker

        try:
            for _round in range(rounds):
                if len(candidate_idx) <= 1:
                    break
                sims_per_candidate = max(1, sims_per_round // len(candidate_idx))

                if use_parallel_workers:
                    cand_visits = np.zeros(len(candidate_idx), dtype=np.float32)
                    cand_q = np.zeros(len(candidate_idx), dtype=np.float32)
                    active_workers = len(candidate_workers)
                    if active_workers < len(candidate_idx):
                        use_parallel_workers = False
                    else:
                        for i, idx in enumerate(candidate_idx):
                            cand_visits[i], cand_q[i] = candidate_workers[idx].run(sims_per_candidate)
                else:
                    cand_roots = [candidate_roots[idx] for idx in candidate_idx]
                    cand_alloc = [sims_per_candidate for _ in candidate_idx]
                    cand_visits, cand_q = self._gumbel_values_for_candidate_roots(cand_roots, cand_alloc)

                sigma_norm = (float(self.config.c_visit) + float(cand_visits.max())) * float(self.config.c_scale)
                cand_score = np.array(
                    [base_score[idx] for idx in candidate_idx],
                    dtype=np.float32,
                ) + sigma_norm * cand_q

                keep = max(1, len(candidate_idx) // 2)
                order = np.argsort(cand_score)[-keep:][::-1]
                candidate_idx = [candidate_idx[i] for i in order]

            if len(candidate_idx) > 1:
                if use_parallel_workers:
                    cand_visits = np.zeros(len(candidate_idx), dtype=np.float32)
                    cand_q = np.zeros(len(candidate_idx), dtype=np.float32)
                    for i, idx in enumerate(candidate_idx):
                        cand_visits[i], cand_q[i] = candidate_workers[idx].stats()
                else:
                    cand_roots = [candidate_roots[idx] for idx in candidate_idx]
                    cand_visits = np.array([root.visit_count for root in cand_roots], dtype=np.float32)
                    cand_q = np.array([-root.mean_value for root in cand_roots], dtype=np.float32)
                sigma_norm = (float(self.config.c_visit) + float(cand_visits.max())) * float(self.config.c_scale)
                cand_score = np.array(
                    [base_score[idx] for idx in candidate_idx],
                    dtype=np.float32,
                ) + sigma_norm * cand_q
                chosen_local = int(np.argmax(cand_score))
                chosen_idx = candidate_idx[chosen_local]
            else:
                chosen_idx = candidate_idx[0]
        finally:
            pass

        probs = np.zeros(len(legal_moves), dtype=np.float32)
        probs[chosen_idx] = 1.0
        return legal_moves, probs

    def choose_move(self, state: GameState) -> Move:
        legal_moves, probs = self.search(state)
        if not legal_moves:
            return Move(MoveType.PASS)
        idx = int(np.argmax(probs))
        return legal_moves[idx]


def load_fnn_cpu_player(
    checkpoint_path: str,
    *,
    num_simulations: int = 256,
    c_puct: float = 1.25,
    temperature: float = 0.0,
    torch_threads: int | None = None,
) -> FNNCPUPlayer:
    """Convenience loader for the CPU single-game player."""
    return FNNCPUPlayer.from_checkpoint(
        checkpoint_path,
        config=FNNCPUMCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            torch_threads=torch_threads,
        ),
    )

def _candidate_worker_main(
    conn,
    checkpoint_path: str,
    torch_threads: int | None,
    c_puct: float,
) -> None:
    global _CPU_WORKER_PLAYER, _CPU_WORKER_ROOT
    _CPU_WORKER_PLAYER = FNNCPUPlayer.from_checkpoint(
        checkpoint_path,
        config=FNNCPUMCTSConfig(
            num_simulations=1,
            c_puct=c_puct,
            temperature=0.0,
            torch_threads=torch_threads,
        ),
    )
    _CPU_WORKER_ROOT = None

    try:
        while True:
            msg = conn.recv()
            op = msg[0]
            if op == "reset":
                state = _deserialize_game_state(msg[1])
                state_copy = state.copy()
                _CPU_WORKER_ROOT = _Node(
                    state_copy,
                    state_key=_state_cache_key(state_copy),
                )
                conn.send(("ready",))
            elif op == "run":
                if _CPU_WORKER_ROOT is None:
                    raise RuntimeError("Worker root not initialized; call reset first")
                extra_sims = int(msg[1])
                _CPU_WORKER_PLAYER._run_search_on_existing_root(_CPU_WORKER_ROOT, extra_sims)
                conn.send(("stats", float(_CPU_WORKER_ROOT.visit_count), float(-_CPU_WORKER_ROOT.mean_value)))
            elif op == "stats":
                if _CPU_WORKER_ROOT is None:
                    raise RuntimeError("Worker root not initialized; call reset first")
                conn.send(("stats", float(_CPU_WORKER_ROOT.visit_count), float(-_CPU_WORKER_ROOT.mean_value)))
            elif op == "close":
                break
            else:
                raise RuntimeError(f"Unknown worker op: {op}")
    finally:
        try:
            conn.close()
        except OSError:
            pass
