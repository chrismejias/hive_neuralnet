"""Paired-arena SPSA tuning for the native FNN alpha-beta search."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import hive_gpu
from hive_fnn.fnn_native_alphabeta import (
    AlphaBetaGPUContext,
    AlphaBetaSearchConfig,
    search_batch,
)
from hive_fnn.fnn_network import HiveFNN


@dataclass(frozen=True)
class TunableParameter:
    name: str
    low: float
    high: float
    kind: Literal["float", "int", "bool"]
    default: float

    def decode(self, normalized: float) -> float | int | bool:
        value = self.low + min(1.0, max(0.0, normalized)) * (
            self.high - self.low
        )
        if self.kind == "bool":
            return normalized >= 0.5
        if self.kind == "int":
            return int(round(value))
        return float(value)

    def encode_default(self) -> float:
        if self.kind == "bool":
            # Center categorical defaults so early SPSA perturbations test
            # both enabled and disabled forms.
            return 0.5 if self.default else 0.499999
        return (float(self.default) - self.low) / (self.high - self.low)


SEARCH_PARAMETER_SPACE = (
    TunableParameter("aspiration_window", 0.0, 1.0, "float", 0.0),
    TunableParameter("lmr_min_depth", 3, 8, "int", 4),
    TunableParameter("lmr_min_move", 2, 12, "int", 4),
    TunableParameter("lmr_reduction", 0, 2, "int", 1),
    TunableParameter("quiescence_plies", 0, 1, "int", 1),
    TunableParameter(
        "quiescence_budget_fraction", 0.05, 0.35, "float", 0.2,
    ),
    TunableParameter("force_win_probes", 0, 1, "bool", 1),
    TunableParameter("tactical_immobilization", 0, 1, "bool", 1),
    TunableParameter("tactical_opponent_surround", 0, 1, "bool", 1),
    TunableParameter("tactical_own_relief", 0, 1, "bool", 1),
    TunableParameter("tactical_queen_threat", 0, 1, "bool", 1),
    TunableParameter("policy_ordering_weight", 0.1, 3.0, "float", 1.0),
    TunableParameter("tactical_ordering_weight", 0.0, 2.0, "float", 0.0),
    TunableParameter("branching_allocation", -0.5, 0.5, "float", 0.0),
    TunableParameter("early_stop_score", 8.5, 9.95, "float", 9.0),
    TunableParameter("early_stop_min_depth", 1, 6, "int", 1),
)


def default_normalized_parameters() -> np.ndarray:
    return np.asarray(
        [parameter.encode_default() for parameter in SEARCH_PARAMETER_SPACE],
        dtype=np.float64,
    )


def decode_search_config(values: np.ndarray) -> AlphaBetaSearchConfig:
    if values.shape != (len(SEARCH_PARAMETER_SPACE),):
        raise ValueError("normalized alpha-beta parameter vector has wrong shape")
    decoded = {
        parameter.name: parameter.decode(float(value))
        for parameter, value in zip(SEARCH_PARAMETER_SPACE, values)
    }
    return AlphaBetaSearchConfig(**decoded)


def load_search_config(path: str | Path) -> AlphaBetaSearchConfig:
    payload = json.loads(Path(path).read_text())
    values = payload.get("search_config", payload)
    profile = values.get("profile") if isinstance(values, dict) else None
    base = AlphaBetaSearchConfig.from_profile(profile) if profile else AlphaBetaSearchConfig()
    allowed = {field.name for field in fields(AlphaBetaSearchConfig)}
    overrides = {
        name: value for name, value in values.items() if name in allowed
    }
    return AlphaBetaSearchConfig(**{**asdict(base), **overrides})


@dataclass(frozen=True)
class PairedArenaResult:
    plus_score: float
    games: int
    plus_wins: int
    minus_wins: int
    draws: int
    plus_nodes: int
    minus_nodes: int
    plus_moves: int
    minus_moves: int

    @property
    def plus_nodes_per_move(self) -> float:
        return self.plus_nodes / max(1, self.plus_moves)

    @property
    def minus_nodes_per_move(self) -> float:
        return self.minus_nodes / max(1, self.minus_moves)


def _paired_random_openings(
    states: torch.Tensor,
    *,
    pair_count: int,
    opening_plies: int,
    rng: np.random.Generator,
) -> None:
    ext = hive_gpu.load_extension()
    games = pair_count * 2
    for _ in range(max(0, int(opening_plies))):
        legal_moves, num_legal = ext.generate_legal_moves_batch(states, games)
        selected = legal_moves[:, 0].clone()
        counts = num_legal.cpu().numpy()
        for pair in range(pair_count):
            even = pair * 2
            count = int(counts[even])
            if count <= 0:
                continue
            slot = int(rng.integers(0, count))
            selected[even] = legal_moves[even, slot]
            selected[even + 1] = legal_moves[even, slot]
        ext.apply_moves_batch(states, selected, games)


@torch.inference_mode()
def run_paired_alpha_beta_arena(
    net: HiveFNN,
    plus_config: AlphaBetaSearchConfig,
    minus_config: AlphaBetaSearchConfig,
    *,
    pairs: int,
    node_budget: int,
    max_depth: int,
    max_plies: int,
    opening_plies: int,
    expansion_mask: int,
    seed: int,
    min_search_batch: int = 0,
) -> PairedArenaResult:
    """Compare two configurations on identical color-swapped openings."""

    if pairs <= 0:
        raise ValueError("paired arena requires at least one pair")
    ext = hive_gpu.load_extension()
    games = pairs * 2
    rng = np.random.default_rng(seed)
    states = ext.create_initial_states(games, expansion_mask)
    _paired_random_openings(
        states,
        pair_count=pairs,
        opening_plies=opening_plies,
        rng=rng,
    )
    plies = np.full(games, opening_plies, dtype=np.int32)
    opening_results = ext.check_results_batch(states, games).cpu().numpy()
    active = (opening_results == 0) & (plies < max_plies)
    plus_is_white = np.zeros(games, dtype=bool)
    plus_is_white[::2] = True
    plus_nodes = minus_nodes = plus_moves = minus_moves = 0
    persistent_contexts = {
        True: AlphaBetaGPUContext(
            net, capacity=pairs, search_config=plus_config,
        ) if plus_config.persistent_tt else None,
        False: AlphaBetaGPUContext(
            net, capacity=pairs, search_config=minus_config,
        ) if minus_config.persistent_tt else None,
    }

    while bool(active.any()):
        turns = states[:, 3412].cpu().numpy().astype(np.int32)
        turns |= states[:, 3413].cpu().numpy().astype(np.int32) << 8
        plus_turn = active & (((turns & 1) == 0) == plus_is_white)
        for is_plus, mask, config in (
            (True, plus_turn, plus_config),
            (False, active & ~plus_turn, minus_config),
        ):
            rows = np.flatnonzero(mask)
            if not rows.size:
                continue
            row_tensor = torch.from_numpy(
                rows.astype(np.int64, copy=False),
            ).cuda()
            context = persistent_contexts[is_plus]
            if context is not None:
                # Keep one stable workspace slot per paired opening. Exactly
                # one game in each live pair is normally this configuration's
                # turn; inactive slots search a harmless duplicate and their
                # outputs are discarded.
                slot_rows = np.empty(pairs, dtype=np.int64)
                active_slots = []
                active_rows = []
                fallback = int(rows[0])
                for pair in range(pairs):
                    first = pair * 2
                    second = first + 1
                    selected = (
                        first if mask[first] else
                        second if mask[second] else fallback
                    )
                    slot_rows[pair] = selected
                    if mask[selected]:
                        active_slots.append(pair)
                        active_rows.append(selected)
                slot_tensor = torch.from_numpy(slot_rows).cuda()
                slot_states = states.index_select(0, slot_tensor).contiguous()
                all_moves, _values, all_stats = search_batch(
                    net, slot_states, node_budget=node_budget,
                    max_depth=max_depth, search_config=config,
                    context=context,
                )
                active_slot_tensor = torch.tensor(
                    active_slots, device="cuda", dtype=torch.int64,
                )
                moves = all_moves.index_select(0, active_slot_tensor)
                stats = all_stats.index_select(0, active_slot_tensor)
                rows = np.asarray(active_rows, dtype=np.int64)
                row_tensor = torch.from_numpy(rows).cuda()
            else:
                sub_states = states.index_select(0, row_tensor).contiguous()
                real_count = int(rows.size)
                padded_states = sub_states
                target_count = max(real_count, int(min_search_batch))
                if target_count > real_count:
                    padding = sub_states[:1].expand(
                        target_count - real_count, -1,
                    ).contiguous()
                    padded_states = torch.cat((sub_states, padding), dim=0)
                all_moves, _values, all_stats = search_batch(
                    net, padded_states, node_budget=node_budget,
                    max_depth=max_depth, search_config=config,
                )
                moves = all_moves[:real_count]
                stats = all_stats[:real_count]
            sub_states = states.index_select(0, row_tensor).contiguous()
            ext.apply_moves_batch(sub_states, moves, int(rows.size))
            states.index_copy_(0, row_tensor, sub_states)
            plies[rows] += 1
            searched_nodes = int(stats[:, 1].sum().item())
            if is_plus:
                plus_nodes += searched_nodes
                plus_moves += int(rows.size)
            else:
                minus_nodes += searched_nodes
                minus_moves += int(rows.size)

        moved = np.flatnonzero(active)
        if not moved.size:
            break
        row_tensor = torch.from_numpy(
            moved.astype(np.int64, copy=False),
        ).cuda()
        results = ext.check_results_batch(
            states.index_select(0, row_tensor), int(moved.size),
        ).cpu().numpy()
        active[moved] = (results == 0) & (plies[moved] < max_plies)

    results = ext.check_results_batch(states, games).cpu().numpy()
    plus_wins = minus_wins = draws = 0
    for game, result in enumerate(results):
        plus_won = (
            (result == 1 and plus_is_white[game])
            or (result == 2 and not plus_is_white[game])
        )
        minus_won = (
            (result == 2 and plus_is_white[game])
            or (result == 1 and not plus_is_white[game])
        )
        if plus_won:
            plus_wins += 1
        elif minus_won:
            minus_wins += 1
        else:
            draws += 1
    plus_score = (plus_wins + 0.5 * draws) / games
    return PairedArenaResult(
        plus_score=plus_score,
        games=games,
        plus_wins=plus_wins,
        minus_wins=minus_wins,
        draws=draws,
        plus_nodes=plus_nodes,
        minus_nodes=minus_nodes,
        plus_moves=plus_moves,
        minus_moves=minus_moves,
    )


@dataclass(frozen=True)
class SPSAConfig:
    learning_rate: float = 0.08
    perturbation: float = 0.20
    alpha: float = 0.602
    gamma: float = 0.101
    stability: float = 10.0
    node_cost_penalty: float = 0.02


class AlphaBetaSPSATuner:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        config: SPSAConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or SPSAConfig()
        self.rng = np.random.default_rng(seed)
        self.theta = default_normalized_parameters()
        self.iteration = 0

    @property
    def state_path(self) -> Path:
        return self.output_dir / "spsa_state_latest.json"

    @property
    def history_path(self) -> Path:
        return self.output_dir / "spsa_history.jsonl"

    def load(self, path: str | Path | None = None) -> None:
        payload = json.loads(Path(path or self.state_path).read_text())
        self.iteration = int(payload["iteration"])
        self.theta = np.asarray(payload["theta"], dtype=np.float64)
        self.rng.bit_generator.state = payload["rng_state"]

    def _save(self) -> None:
        payload = {
            "format": "fnn_alphabeta_spsa_v1",
            "iteration": self.iteration,
            "theta": self.theta.tolist(),
            "search_config": asdict(decode_search_config(self.theta)),
            "spsa_config": asdict(self.config),
            "rng_state": self.rng.bit_generator.state,
        }
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(self.state_path)

    def ask(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        k = self.iteration + 1
        c_k = self.config.perturbation / (k ** self.config.gamma)
        a_k = self.config.learning_rate / (
            (self.config.stability + k) ** self.config.alpha
        )
        delta = self.rng.choice(
            np.asarray([-1.0, 1.0]), size=self.theta.shape,
        )
        plus = np.clip(self.theta + c_k * delta, 0.0, 1.0)
        minus = np.clip(self.theta - c_k * delta, 0.0, 1.0)
        return plus, minus, delta, a_k, c_k

    def tell(
        self,
        plus: np.ndarray,
        minus: np.ndarray,
        delta: np.ndarray,
        a_k: float,
        c_k: float,
        arena: PairedArenaResult,
    ) -> dict:
        node_ratio = arena.plus_nodes_per_move / max(
            arena.minus_nodes_per_move, 1e-9,
        )
        outcome_difference = 2.0 * arena.plus_score - 1.0
        objective_difference = (
            outcome_difference
            - self.config.node_cost_penalty * math.log(max(node_ratio, 1e-9))
        )
        denominator = np.maximum(np.abs(plus - minus), 1e-6)
        gradient = objective_difference * delta / denominator
        self.theta = np.clip(self.theta + a_k * gradient, 0.0, 1.0)
        self.iteration += 1
        record = {
            "iteration": self.iteration,
            "plus": asdict(decode_search_config(plus)),
            "minus": asdict(decode_search_config(minus)),
            "incumbent": asdict(decode_search_config(self.theta)),
            "plus_score": arena.plus_score,
            "plus_wins": arena.plus_wins,
            "minus_wins": arena.minus_wins,
            "draws": arena.draws,
            "plus_nodes_per_move": arena.plus_nodes_per_move,
            "minus_nodes_per_move": arena.minus_nodes_per_move,
            "outcome_difference": outcome_difference,
            "objective_difference": objective_difference,
            "a_k": a_k,
            "c_k": c_k,
        }
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._save()
        return record
