"""Launcher for the CUDA-resident FNN alpha-beta baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

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
    quiescence_plies: int = 2
    quiescence_budget_fraction: float = 0.5
    force_win_probes: bool = True
    tactical_immobilization: bool = True
    tactical_opponent_surround: bool = True
    tactical_own_relief: bool = True
    tactical_queen_threat: bool = True
    policy_ordering_weight: float = 1.0
    tactical_ordering_weight: float = 0.0
    branching_allocation: float = 0.0
    early_stop_score: float = 9.0
    early_stop_min_depth: int = 1

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
                max(0.0, float(self.policy_ordering_weight)),
                max(0.0, float(self.tactical_ordering_weight)),
                min(0.75, max(-0.75, float(self.branching_allocation))),
                min(9.99, max(1.0, float(self.early_stop_score))),
                max(1, int(self.early_stop_min_depth)),
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


def pack_fnn_weights(net: HiveFNN) -> torch.Tensor:
    """Pack parameters in the device evaluator's documented linear order."""

    state = net.state_dict()
    order = (
        "fc1.weight", "fc1.bias", "ln1.weight", "ln1.bias",
        "fc2.weight", "fc2.bias", "value_fc.weight", "value_fc.bias",
        "action_fc1.weight", "action_fc1.bias", "action_fc2.weight", "action_fc2.bias",
    )
    return torch.cat([state[key].detach().float().flatten() for key in order]).cuda().contiguous()


def search_batch(
    net: HiveFNN,
    states: torch.Tensor,
    *,
    node_budget: int,
    max_depth: int = 32,
    search_config: AlphaBetaSearchConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return moves, values, and search counters.

    Counters are [depth, nodes, cutoffs, tt_hits, pvs_researches,
    lmr_reductions, qnodes, forced_win_probes, tactical_moves].
    """

    if states.device.type != "cuda" or states.dtype != torch.uint8:
        raise ValueError("states must be CUDA uint8 packed HiveState rows")
    cfg = net.config
    if max(cfg.hidden_dim, cfg.embed_dim, cfg.action_hidden) > 64:
        raise ValueError("native alpha-beta supports FNN dimensions up to 64")
    ext = hive_gpu.load_extension()
    native_config = (search_config or AlphaBetaSearchConfig()).packed(states.device)
    return ext.fnn_alphabeta_batch(
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
) -> AlphaBetaTeacherBatch:
    """Return records from only the deepest fully completed iteration."""

    if states.device.type != "cuda" or states.dtype != torch.uint8:
        raise ValueError("states must be CUDA uint8 packed HiveState rows")
    cfg = net.config
    if max(cfg.hidden_dim, cfg.embed_dim, cfg.action_hidden) > 64:
        raise ValueError("native alpha-beta supports FNN dimensions up to 64")
    ext = hive_gpu.load_extension()
    native_config = (search_config or AlphaBetaSearchConfig()).packed(states.device)
    outputs = ext.fnn_alphabeta_teacher_batch(
        states.contiguous(), pack_fnn_weights(net), cfg.hidden_dim, cfg.embed_dim,
        cfg.action_hidden, native_config, int(node_budget), int(max_depth),
        max(1, int(root_exact_count)),
    )
    return AlphaBetaTeacherBatch(*outputs)
