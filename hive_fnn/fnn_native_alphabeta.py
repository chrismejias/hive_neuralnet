"""Launcher for the CUDA-resident FNN alpha-beta baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from enum import IntEnum
import weakref

import torch

import hive_gpu
from hive_fnn.fnn_network import HiveFNN


class AlphaBetaBound(IntEnum):
    UNSEARCHED = 0
    EXACT = 1
    LOWER = 2
    UPPER = 3


@dataclass(frozen=True)
class AlphaBetaSearchConfig:
    """Runtime-tunable native alpha-beta controls."""

    aspiration_window: float = 0.0
    lmr_min_depth: int = 4
    lmr_min_move: int = 4
    lmr_reduction: int = 1
    quiescence_plies: int = 0
    quiescence_budget_fraction: float = 0.2
    force_win_probes: bool = True
    tactical_immobilization: bool = True
    tactical_opponent_surround: bool = True
    tactical_own_relief: bool = True
    tactical_queen_threat: bool = True
    branching_allocation: float = 0.0
    early_stop_score: float = 9.0
    early_stop_min_depth: int = 1
    recursive_threat_qsearch: bool = False
    forced_extensions: bool = False
    forced_extension_max_chain: int = 2
    singular_extensions: bool = False
    singular_min_depth: int = 6
    singular_margin: float = 0.25
    proof_search: bool = False
    proof_max_plies: int = 10
    proof_budget_fraction: float = 0.2
    proof_trigger_surround: int = 4
    persistent_tt: bool = False
    countermove_ordering: bool = False
    continuation_history: bool = False
    internal_heuristic_ordering: bool = False

    @classmethod
    def from_profile(cls, name: str) -> "AlphaBetaSearchConfig":
        """Construct a stable ablation profile."""
        profile = name.strip().lower()
        baseline = cls()
        if profile in ("baseline", "value-only"):
            # ``value-only`` is retained as a legacy alias; all search is value-only.
            return baseline
        if profile == "quiescence":
            return replace(baseline, quiescence_plies=1)
        if profile == "threat":
            return replace(baseline, recursive_threat_qsearch=True,
                           quiescence_plies=4, forced_extensions=True)
        if profile == "proof":
            return replace(baseline, proof_search=True)
        if profile == "ordering":
            return replace(baseline, countermove_ordering=True,
                           continuation_history=True,
                           internal_heuristic_ordering=True)
        if profile == "full":
            return replace(
                baseline, recursive_threat_qsearch=True, quiescence_plies=4,
                forced_extensions=True, singular_extensions=True,
                proof_search=True, persistent_tt=True,
                countermove_ordering=True, continuation_history=True,
                internal_heuristic_ordering=True,
            )
        raise ValueError(
            f"unknown alpha-beta profile {name!r}; expected baseline, "
            "threat, proof, ordering, or full"
        )

    @classmethod
    def from_metadata(cls, values: dict[str, object]) -> "AlphaBetaSearchConfig":
        """Load current or legacy checkpoint/search metadata."""
        normalized = dict(values)
        if (
            "internal_policy_ordering" in normalized
            and "internal_heuristic_ordering" not in normalized
        ):
            normalized["internal_heuristic_ordering"] = normalized[
                "internal_policy_ordering"
            ]
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in normalized.items() if key in allowed})

    def metadata(self) -> dict[str, object]:
        return asdict(self)

    def packed(self, device: torch.device | str = "cuda") -> torch.Tensor:
        tactical_mask = (
            (1 if self.tactical_immobilization else 0)
            | (2 if self.tactical_opponent_surround else 0)
            | (4 if self.tactical_own_relief else 0)
            | (8 if self.tactical_queen_threat else 0)
        )
        return torch.tensor(
            [
                max(0.0, float(self.aspiration_window)),
                max(1, int(self.lmr_min_depth)),
                max(1, int(self.lmr_min_move)),
                max(0, int(self.lmr_reduction)),
                max(0, int(self.quiescence_plies)),
                min(0.95, max(0.0, float(self.quiescence_budget_fraction))),
                1.0 if self.force_win_probes else 0.0,
                float(tactical_mask),
                min(0.75, max(-0.75, float(self.branching_allocation))),
                min(9.99, max(1.0, float(self.early_stop_score))),
                max(1, int(self.early_stop_min_depth)),
                1.0 if self.recursive_threat_qsearch else 0.0,
                1.0 if self.forced_extensions else 0.0,
                max(0, int(self.forced_extension_max_chain)),
                1.0 if self.singular_extensions else 0.0,
                max(2, int(self.singular_min_depth)),
                max(0.0, float(self.singular_margin)),
                1.0 if self.proof_search else 0.0,
                max(1, int(self.proof_max_plies)),
                min(0.75, max(0.0, float(self.proof_budget_fraction))),
                min(5, max(0, int(self.proof_trigger_surround))),
                1.0 if self.persistent_tt else 0.0,
                1.0 if self.countermove_ordering else 0.0,
                1.0 if self.continuation_history else 0.0,
                1.0 if self.internal_heuristic_ordering else 0.0,
            ],
            dtype=torch.float32,
            device=device,
        )


@dataclass(frozen=True)
class AlphaBetaTeacherBatch:
    selected_moves: torch.Tensor
    search_values: torch.Tensor
    stats: torch.Tensor
    raw_values: torch.Tensor
    legal_moves: torch.Tensor
    num_legal: torch.Tensor
    root_scores: torch.Tensor
    root_bounds: torch.Tensor
    selected_indices: torch.Tensor
    pv_moves: torch.Tensor
    pv_lengths: torch.Tensor


_PACKED_WEIGHT_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_CONFIG_CACHE: dict[tuple[AlphaBetaSearchConfig, int], torch.Tensor] = {}


def _parameter_signature(net: HiveFNN) -> tuple[tuple[int, int], ...]:
    signature = []
    for parameter in net.parameters():
        try:
            version = parameter._version
        except RuntimeError:
            # Tensors created inside inference_mode are immutable and do not
            # carry a version counter; their storage identity is sufficient.
            version = -1
        signature.append((parameter.data_ptr(), version))
    return tuple(signature)


def pack_fnn_weights(net: HiveFNN) -> torch.Tensor:
    """Pack weights once, refreshing automatically after an optimizer update."""

    signature = _parameter_signature(net)
    cached = _PACKED_WEIGHT_CACHE.get(net)
    if cached is not None and cached[0] == signature:
        return cached[1]
    state = net.state_dict()
    order = (
        "fc1.weight", "fc1.bias", "ln1.weight", "ln1.bias",
        "fc2.weight", "fc2.bias", "value_fc.weight", "value_fc.bias",
    )
    packed = torch.cat(
        [state[key].detach().float().flatten() for key in order]
    ).cuda().contiguous()
    _PACKED_WEIGHT_CACHE[net] = (signature, packed)
    return packed


def _packed_search_config(
    config: AlphaBetaSearchConfig, device: torch.device,
) -> torch.Tensor:
    index = torch.cuda.current_device() if device.index is None else device.index
    key = (config, index)
    packed = _CONFIG_CACHE.get(key)
    if packed is None:
        packed = config.packed(device)
        _CONFIG_CACHE[key] = packed
    return packed


class AlphaBetaGPUContext:
    """Reusable packed inputs, TT, and scratch storage for ordinary play."""

    def __init__(
        self,
        net: HiveFNN,
        *,
        capacity: int,
        search_config: AlphaBetaSearchConfig | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.net = net
        self.capacity = int(capacity)
        self.search_config = search_config or AlphaBetaSearchConfig()
        self._workspace: list[torch.Tensor] | None = None
        self._generation = 0

    def _prepare(
        self, states: torch.Tensor,
    ) -> tuple[object, object, torch.Tensor, torch.Tensor, int]:
        if states.shape[0] > self.capacity:
            raise ValueError(
                f"batch {states.shape[0]} exceeds context capacity {self.capacity}"
            )
        ext = hive_gpu.load_extension()
        if self._workspace is None:
            self._workspace = list(
                ext.fnn_alphabeta_workspace(states, self.capacity)
            )
        if not self.search_config.persistent_tt:
            self._generation += 1
        elif self._generation == 0:
            self._generation = 1
        if self._generation >= 2_000_000_000:
            self._workspace[5].zero_()
            self._generation = 1
        return (
            ext, self.net.config, pack_fnn_weights(self.net),
            _packed_search_config(self.search_config, states.device),
            self._generation,
        )

    def search(
        self, states: torch.Tensor, *, node_budget: int, max_depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ext, cfg, weights, native_config, generation = self._prepare(states)
        return ext.fnn_alphabeta_batch_reuse(
            states.contiguous(), weights,
            cfg.hidden_dim, cfg.embed_dim, cfg.action_hidden, native_config,
            int(node_budget), int(max_depth), self._workspace, generation,
        )

    def search_teacher(
        self,
        states: torch.Tensor,
        *,
        node_budget: int,
        max_depth: int,
        root_exact_count: int,
    ) -> AlphaBetaTeacherBatch:
        ext, cfg, weights, native_config, generation = self._prepare(states)
        outputs = ext.fnn_alphabeta_teacher_batch_reuse(
            states.contiguous(), weights,
            cfg.hidden_dim, cfg.embed_dim, cfg.action_hidden, native_config,
            int(node_budget), int(max_depth), max(1, int(root_exact_count)),
            self._workspace, generation,
        )
        return AlphaBetaTeacherBatch(*outputs)


def search_batch(
    net: HiveFNN,
    states: torch.Tensor,
    *,
    node_budget: int,
    max_depth: int = 32,
    search_config: AlphaBetaSearchConfig | None = None,
    context: AlphaBetaGPUContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return moves, values, and search counters.

    Counters are [depth, nodes, cutoffs, tt_hits, pvs_researches,
    lmr_reductions, q_probes, forced_win_probes, tactical_moves]. Quiescence
    charges every candidate it applies, including candidates later rejected
    as quiet, so its configured budget bounds actual tactical work.
    """

    if states.device.type != "cuda" or states.dtype != torch.uint8:
        raise ValueError("states must be CUDA uint8 packed HiveState rows")
    cfg = net.config
    if max(cfg.hidden_dim, cfg.embed_dim, cfg.action_hidden) > 64:
        raise ValueError("native alpha-beta supports FNN dimensions up to 64")
    if context is not None:
        if context.net is not net:
            raise ValueError("context belongs to a different network")
        if search_config is not None and search_config != context.search_config:
            raise ValueError("search_config does not match the reusable context")
        return context.search(
            states, node_budget=node_budget, max_depth=max_depth,
        )
    ext = hive_gpu.load_extension()
    native_config = _packed_search_config(
        search_config or AlphaBetaSearchConfig(), states.device,
    )
    return ext.fnn_alphabeta_batch(
        states.contiguous(), pack_fnn_weights(net), cfg.hidden_dim, cfg.embed_dim,
        cfg.action_hidden, native_config, int(node_budget), int(max_depth),
    )


def search_batch_resumable(
    net: HiveFNN,
    states: torch.Tensor,
    *,
    node_budget: int,
    max_depth: int = 32,
    search_config: AlphaBetaSearchConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact experimental cross-game resumable traversal."""

    if states.device.type != "cuda" or states.dtype != torch.uint8:
        raise ValueError("states must be CUDA uint8 packed HiveState rows")
    cfg = net.config
    if max(cfg.hidden_dim, cfg.embed_dim, cfg.action_hidden) > 64:
        raise ValueError("native alpha-beta supports FNN dimensions up to 64")
    ext = hive_gpu.load_extension()
    native_config = _packed_search_config(
        search_config or AlphaBetaSearchConfig(), states.device,
    )
    return ext.fnn_alphabeta_resumable_batch(
        states.contiguous(), pack_fnn_weights(net), cfg.hidden_dim, cfg.embed_dim,
        cfg.action_hidden, native_config, int(node_budget), int(max_depth),
    )


def search_teacher_batch(
    net: HiveFNN,
    states: torch.Tensor,
    *,
    node_budget: int,
    max_depth: int = 32,
    root_exact_count: int = 1,
    search_config: AlphaBetaSearchConfig | None = None,
    context: AlphaBetaGPUContext | None = None,
) -> AlphaBetaTeacherBatch:
    """Return records from only the deepest fully completed iteration."""

    if states.device.type != "cuda" or states.dtype != torch.uint8:
        raise ValueError("states must be CUDA uint8 packed HiveState rows")
    cfg = net.config
    if max(cfg.hidden_dim, cfg.embed_dim, cfg.action_hidden) > 64:
        raise ValueError("native alpha-beta supports FNN dimensions up to 64")
    if context is not None:
        if context.net is not net:
            raise ValueError("context belongs to a different network")
        if search_config is not None and search_config != context.search_config:
            raise ValueError("search_config does not match the reusable context")
        return context.search_teacher(
            states, node_budget=node_budget, max_depth=max_depth,
            root_exact_count=root_exact_count,
        )
    ext = hive_gpu.load_extension()
    native_config = _packed_search_config(
        search_config or AlphaBetaSearchConfig(), states.device,
    )
    outputs = ext.fnn_alphabeta_teacher_batch(
        states.contiguous(), pack_fnn_weights(net), cfg.hidden_dim, cfg.embed_dim,
        cfg.action_hidden, native_config, int(node_budget), int(max_depth),
        max(1, int(root_exact_count)),
    )
    return AlphaBetaTeacherBatch(*outputs)
