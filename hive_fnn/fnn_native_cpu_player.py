from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable

import numpy as np
import torch

import hive_cpu_native
from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_gpu.endgame_generator import gamestate_to_gpu_bytes


@dataclass
class FNNNativeCPUConfig:
    num_simulations: int = 512
    c_puct: float = 1.25
    c_visit: float = 50.0
    c_scale: float = 1.0
    use_gumbel_root: bool = True
    gumbel_considered: int = 16
    gumbel_noise_scale: float = 0.0
    torch_threads: int | None = 1


class _NativeNode:
    __slots__ = (
        "state_bytes",
        "state_key",
        "root_feature",
        "parent",
        "move_from_parent",
        "prior",
        "visit_count",
        "total_value",
        "children",
        "legal_moves",
        "expanded",
        "result",
    )

    def __init__(
        self,
        state_bytes: bytes,
        *,
        parent: "_NativeNode | None" = None,
        move_from_parent: np.ndarray | None = None,
        root_feature: np.ndarray | None = None,
        prior: float = 0.0,
        result: int = 0,
    ) -> None:
        self.state_bytes = state_bytes
        self.state_key: Hashable = state_bytes
        self.root_feature = root_feature
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.prior = float(prior)
        self.visit_count = 0
        self.total_value = 0.0
        self.children: list[_NativeNode] = []
        self.legal_moves: np.ndarray | None = None
        self.expanded = False
        self.result = int(result)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class FNNNativeCPUPlayer:
    """CPU-only FNN search using the packed native HiveState backend."""

    def __init__(self, net: HiveFNN, config: FNNNativeCPUConfig | None = None) -> None:
        self.ext = hive_cpu_native.load_extension()
        self.net = net.to("cpu").eval()
        self.config = config or FNNNativeCPUConfig()
        self._feature_cache: dict[Hashable, np.ndarray] = {}
        if self.config.torch_threads is not None:
            torch.set_num_threads(max(1, int(self.config.torch_threads)))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        config: FNNNativeCPUConfig | None = None,
    ) -> "FNNNativeCPUPlayer":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        net_config = ckpt.get("net_config")
        if not isinstance(net_config, FNNConfig):
            net_config = FNNConfig.large()
        net = HiveFNN(net_config)
        state = ckpt.get("model_state_dict", ckpt.get("model_state"))
        if state is None:
            raise KeyError("checkpoint missing model state")
        net.load_state_dict(state)
        return cls(net, config=config)

    @classmethod
    def random(
        cls,
        net_config: FNNConfig,
        *,
        seed: int,
        config: FNNNativeCPUConfig | None = None,
    ) -> "FNNNativeCPUPlayer":
        torch.manual_seed(seed)
        return cls(HiveFNN(net_config), config=config)

    def state_bytes_from_game(self, game) -> bytes:
        return gamestate_to_gpu_bytes(game)

    def _result_to_value(self, result: int, turn: int) -> float:
        if result == 0 or result == 3:
            return 0.0
        is_white = (turn % 2 == 0)
        root_won = (result == 1 and is_white) or (result == 2 and not is_white)
        return 1.0 if root_won else -1.0

    def _turn_from_state_bytes(self, state_bytes: bytes) -> int:
        return int(state_bytes[3412]) | (int(state_bytes[3413]) << 8)

    def _evaluate_state(
        self,
        node: _NativeNode,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, list[bytes], np.ndarray]:
        moves_pad, n_legal, root_feature = self.ext.legal_moves_and_fnn_features(node.state_bytes)
        moves = np.asarray(moves_pad, dtype=np.uint8)[: int(n_legal)].copy()
        root_feat = np.asarray(root_feature, dtype=np.float32).copy()
        node.root_feature = root_feat
        self._feature_cache[node.state_key] = root_feat

        if n_legal <= 0:
            return moves, np.zeros(0, dtype=np.float32), 0.0, np.zeros((0, 110), dtype=np.float32), [], np.zeros(0, dtype=np.int32)

        succ_features, child_states, child_results = self.ext.successor_features(
            node.state_bytes,
            moves,
            int(n_legal),
        )
        succ_features_np = np.asarray(succ_features, dtype=np.float32)
        child_results_np = np.asarray(child_results, dtype=np.int32)

        with torch.inference_mode():
            root_t = torch.from_numpy(root_feat).unsqueeze(0)
            succ_t = torch.from_numpy(succ_features_np)
            root_emb = self.net.encode(root_t)
            succ_emb = self.net.encode(succ_t)
            root_value = float(self.net.value_head(root_emb).squeeze().item())
            root_rep = root_emb.expand(succ_emb.shape[0], -1)
            logits = self.net.score_actions(root_rep, succ_emb).float()
            priors = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)

        return moves, priors, root_value, succ_features_np, list(child_states), child_results_np

    def _expand(self, node: _NativeNode) -> float:
        if node.result != 0:
            turn = self._turn_from_state_bytes(node.state_bytes)
            return self._result_to_value(node.result, turn)

        moves, priors, value, succ_features, child_states, child_results = self._evaluate_state(node)
        node.legal_moves = moves
        node.children = []
        for i, move in enumerate(moves):
            child = _NativeNode(
                child_states[i],
                parent=node,
                move_from_parent=move.copy(),
                root_feature=succ_features[i],
                prior=float(priors[i]),
                result=int(child_results[i]),
            )
            node.children.append(child)
        node.expanded = True
        return value

    def _select_child(self, node: _NativeNode) -> _NativeNode:
        total = max(1, node.visit_count)
        sqrt_total = math.sqrt(total)
        best_score = -1e30
        best_child = node.children[0]
        for child in node.children:
            q = -child.mean_value if child.visit_count > 0 else 0.0
            u = self.config.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backpropagate(self, node: _NativeNode, value: float) -> None:
        cur: _NativeNode | None = node
        v = value
        while cur is not None:
            cur.visit_count += 1
            cur.total_value += v
            v = -v
            cur = cur.parent

    def _run_search_on_existing_root(self, root: _NativeNode, simulations: int) -> None:
        for _ in range(max(0, int(simulations))):
            node = root
            while node.expanded and node.children:
                node = self._select_child(node)
            value = self._expand(node) if not node.expanded else node.mean_value
            self._backpropagate(node, value)

    def _gumbel_search(self, root: _NativeNode) -> int:
        moves, priors, _value, succ_features, child_states, child_results = self._evaluate_state(root)
        root.legal_moves = moves
        if len(moves) == 0:
            return -1

        root.children = [
            _NativeNode(
                child_states[i],
                parent=root,
                move_from_parent=moves[i].copy(),
                root_feature=succ_features[i],
                prior=float(priors[i]),
                result=int(child_results[i]),
            )
            for i in range(len(moves))
        ]
        root.expanded = True

        k = min(int(self.config.gumbel_considered), len(moves))
        rounds = max(1, math.ceil(math.log2(max(k, 1))))
        sims_per_round = max(1, int(self.config.num_simulations) // rounds)
        log_priors = np.log(np.clip(priors, 1e-30, None))
        gumbel = (
            -np.log(-np.log(np.clip(np.random.random(len(moves)), 1e-12, 1.0 - 1e-12)))
            * float(self.config.gumbel_noise_scale)
        ).astype(np.float32)
        base_score = log_priors + gumbel
        candidate_idx = np.argsort(base_score)[-k:][::-1].tolist()

        for _round in range(rounds):
            if len(candidate_idx) <= 1:
                break
            sims_per_candidate = max(1, sims_per_round // len(candidate_idx))
            cand_visits = np.zeros(len(candidate_idx), dtype=np.float32)
            cand_q = np.zeros(len(candidate_idx), dtype=np.float32)
            for i, idx in enumerate(candidate_idx):
                child = root.children[idx]
                self._run_search_on_existing_root(child, sims_per_candidate)
                cand_visits[i] = float(child.visit_count)
                cand_q[i] = float(-child.mean_value)

            sigma_norm = (float(self.config.c_visit) + float(cand_visits.max())) * float(self.config.c_scale)
            cand_score = np.array([base_score[idx] for idx in candidate_idx], dtype=np.float32) + sigma_norm * cand_q
            keep = max(1, len(candidate_idx) // 2)
            order = np.argsort(cand_score)[-keep:][::-1]
            candidate_idx = [candidate_idx[i] for i in order]

        if len(candidate_idx) == 1:
            return int(candidate_idx[0])

        cand_visits = np.array([root.children[idx].visit_count for idx in candidate_idx], dtype=np.float32)
        cand_q = np.array([-root.children[idx].mean_value for idx in candidate_idx], dtype=np.float32)
        sigma_norm = (float(self.config.c_visit) + float(cand_visits.max())) * float(self.config.c_scale)
        cand_score = np.array([base_score[idx] for idx in candidate_idx], dtype=np.float32) + sigma_norm * cand_q
        return int(candidate_idx[int(np.argmax(cand_score))])

    def choose_move_bytes(self, state_bytes: bytes) -> np.ndarray:
        result = int(self.ext.check_result(state_bytes))
        root = _NativeNode(state_bytes, result=result)
        if self.config.use_gumbel_root:
            idx = self._gumbel_search(root)
            if idx < 0 or root.legal_moves is None:
                return np.zeros((self.ext.SIZEOF_GPU_MOVE,), dtype=np.uint8)
            return root.legal_moves[idx].copy()

        self._run_search_on_existing_root(root, self.config.num_simulations)
        if not root.children or root.legal_moves is None:
            return np.zeros((self.ext.SIZEOF_GPU_MOVE,), dtype=np.uint8)
        idx = int(np.argmax([c.visit_count for c in root.children]))
        return root.legal_moves[idx].copy()
