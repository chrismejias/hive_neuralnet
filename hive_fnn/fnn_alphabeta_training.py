"""Alpha-beta teacher records, self-play generation, and FNN training losses."""

from __future__ import annotations

import random
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F

import hive_gpu
from hive_fnn.fnn_native_alphabeta import (
    AlphaBetaBound,
    AlphaBetaGPUContext,
    AlphaBetaSearchConfig,
    AlphaBetaTeacherBatch,
    search_teacher_batch,
)
from hive_fnn.fnn_network import HiveFNN

_TURN_OFFSET = 3412


@dataclass(frozen=True)
class AlphaBetaRecord:
    """One root record from a fully completed iterative-deepening pass."""

    state: torch.Tensor
    legal_moves: torch.Tensor
    raw_value: float
    search_value: float
    selected_index: int
    pv_moves: torch.Tensor
    root_scores: torch.Tensor
    root_bounds: torch.Tensor
    completed_depth: int
    nodes: int
    final_result: float
    depth_value_delta: float = 0.0
    depth_move_changed: bool = False

    @property
    def selected_move(self) -> torch.Tensor:
        return self.legal_moves[self.selected_index]


@dataclass(frozen=True)
class AlphaBetaGenerationConfig:
    games: int = 256
    game_node_budget: int = 1_000
    teacher_node_budget: int = 6_000
    teacher_relabel_fraction: float = 0.25
    max_depth: int = 32
    max_game_length: int = 300
    expansion_mask: int = 0
    relabel_batch_size: int = 64
    opening_diversity_plies: int = 6
    opening_diversity_candidates: int = 4
    opening_diversity_value_window: float = 0.12
    opening_diversity_temperature: float = 0.0
    search_config: AlphaBetaSearchConfig = field(
        default_factory=AlphaBetaSearchConfig,
    )
    seed: int = 0
    endgame_fraction: float = 0.0
    endgame_min_surround: int = 4
    endgame_max_surround: int = 5
    endgame_mixed_pair: bool = True
    retain_truncated_teacher_records: bool = False


def alpha_beta_expansion_runs(
    games: int, expansion_mask: int,
) -> list[tuple[int, int]]:
    """Distribute ``-1`` evenly over all eight Base/L/M/P subsets."""
    if games <= 0:
        return []
    if expansion_mask >= 0:
        return [(expansion_mask, games)]
    base, remainder = divmod(games, 8)
    return [
        (mask, base + (1 if mask < remainder else 0))
        for mask in range(8)
        if base + (1 if mask < remainder else 0) > 0
    ]


@dataclass(frozen=True)
class AlphaBetaLossConfig:
    search_value_weight: float = 0.25
    value_loss_weight: float = 1.0


@dataclass(frozen=True)
class AlphaBetaReplayPriorityConfig:
    """Bounded hybrid replay priorities computed when records are generated."""

    enabled: bool = True
    alpha: float = 0.6
    uniform_fraction: float = 0.25
    value_surprise_weight: float = 1.0
    depth_change_weight: float = 1.0
    outcome_surprise_weight: float = 0.5
    complexity_weight: float = 0.25
    epsilon: float = 0.05


@dataclass(frozen=True)
class AlphaBetaTrainingBatch:
    states: torch.Tensor
    legal_moves: torch.Tensor
    num_legal: torch.Tensor
    search_values: torch.Tensor
    final_results: torch.Tensor


class AlphaBetaReplayBuffer:
    """Bound-aware replay storage kept separate from MCTS policy targets."""

    def __init__(
        self,
        capacity: int = 75_000,
        priority_config: AlphaBetaReplayPriorityConfig | None = None,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.priority_config = priority_config or AlphaBetaReplayPriorityConfig()
        self._records: list[AlphaBetaRecord] = []
        self._priorities: list[float] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._records)

    def add(self, records: list[AlphaBetaRecord]) -> None:
        for record in records:
            if record.completed_depth <= 0:
                continue
            priority = self._record_priority(record)
            if len(self._records) < self.capacity:
                self._records.append(record)
                self._priorities.append(priority)
            else:
                self._records[self._next] = record
                self._priorities[self._next] = priority
            self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int, *, seed: int | None = None) -> list[AlphaBetaRecord]:
        if not self._records:
            raise ValueError("cannot sample an empty alpha-beta replay buffer")
        rng = random.Random(seed)
        count = int(batch_size)
        cfg = self.priority_config
        if not cfg.enabled:
            return [rng.choice(self._records) for _ in range(count)]
        uniform_fraction = max(0.0, min(1.0, cfg.uniform_fraction))
        weights = [
            max(cfg.epsilon, priority) ** max(0.0, cfg.alpha)
            for priority in self._priorities
        ]
        uniform_count = sum(
            rng.random() < uniform_fraction for _ in range(count)
        )
        samples = [
            rng.choice(self._records) for _ in range(uniform_count)
        ]
        samples.extend(rng.choices(
            self._records,
            weights=weights,
            k=count - uniform_count,
        ))
        rng.shuffle(samples)
        return samples

    def configure_priorities(
        self,
        config: AlphaBetaReplayPriorityConfig,
    ) -> None:
        self.priority_config = config
        self._priorities = [
            self._record_priority(record) for record in self._records
        ]

    def _record_priority(self, record: AlphaBetaRecord) -> float:
        cfg = self.priority_config
        if not cfg.enabled:
            return 1.0

        value_surprise = min(
            1.0, abs(float(record.raw_value) - float(record.search_value)) / 2.0,
        )
        outcome_surprise = (
            min(1.0, abs(float(record.search_value) -
                         float(record.final_result)) / 2.0)
            if math.isfinite(float(record.final_result)) else 0.0
        )
        depth_change = min(
            1.0,
            abs(float(getattr(record, "depth_value_delta", 0.0))) / 2.0,
        )
        if bool(getattr(record, "depth_move_changed", False)):
            depth_change = min(1.0, depth_change + 0.5)

        legal_count = max(1, int(record.legal_moves.shape[0]))
        branching = math.log1p(legal_count) / math.log1p(256)
        bounds = record.root_bounds
        exact_fraction = (
            float((bounds == int(AlphaBetaBound.EXACT)).float().mean())
            if bounds.numel()
            else 0.0
        )
        bound_uncertainty = 1.0 - exact_fraction
        depth_difficulty = 1.0 / max(1.0, float(record.completed_depth))
        complexity = (branching + bound_uncertainty + depth_difficulty) / 3.0

        return max(
            cfg.epsilon,
            cfg.epsilon
            + cfg.value_surprise_weight * value_surprise
            + cfg.depth_change_weight * depth_change
            + cfg.outcome_surprise_weight * outcome_surprise
            + cfg.complexity_weight * complexity,
        )

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "next": self._next,
            "records": self._records,
            "priorities": self._priorities,
            "priority_config": asdict(self.priority_config),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.capacity = max(1, int(payload["capacity"]))
        self._records = list(payload["records"])
        if len(self._records) > self.capacity:
            self._records = self._records[-self.capacity:]
        saved_config = payload.get("priority_config")
        if saved_config:
            self.priority_config = AlphaBetaReplayPriorityConfig(**saved_config)
        saved_priorities = payload.get("priorities")
        if saved_priorities and len(saved_priorities) == len(self._records):
            self._priorities = [float(value) for value in saved_priorities]
        else:
            self._priorities = [
                self._record_priority(record) for record in self._records
            ]
        self._next = int(payload["next"]) % self.capacity

    @classmethod
    def load(cls, path: str | Path) -> "AlphaBetaReplayBuffer":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        buffer = cls(int(payload["capacity"]))
        buffer.load_state_dict(payload)
        return buffer


def _turn_from_state(state: torch.Tensor) -> int:
    return int(state[_TURN_OFFSET]) | (int(state[_TURN_OFFSET + 1]) << 8)


def _outcome_for_turn(result: int, turn: int) -> float:
    if result not in (1, 2):
        return 0.0
    player_is_white = (turn & 1) == 0
    player_won = (result == 1 and player_is_white) or (
        result == 2 and not player_is_white
    )
    return 1.0 if player_won else -1.0


def _record_from_teacher_row(
    state: torch.Tensor,
    teacher: AlphaBetaTeacherBatch,
    row: int,
    *,
    final_result: float = 0.0,
) -> AlphaBetaRecord | None:
    completed_depth = int(teacher.stats[row, 0])
    selected_index = int(teacher.selected_indices[row])
    n = int(teacher.num_legal[row])
    if completed_depth <= 0 or selected_index < 0 or selected_index >= n:
        return None
    pv_length = int(teacher.pv_lengths[row])
    return AlphaBetaRecord(
        state=state.detach().to(device="cpu", dtype=torch.uint8).clone(),
        legal_moves=teacher.legal_moves[row, :n].detach().cpu().clone(),
        raw_value=float(teacher.raw_values[row]),
        search_value=float(teacher.search_values[row]),
        selected_index=selected_index,
        pv_moves=teacher.pv_moves[row, :pv_length].detach().cpu().clone(),
        root_scores=teacher.root_scores[row, :n].detach().cpu().clone(),
        root_bounds=teacher.root_bounds[row, :n].detach().cpu().clone(),
        completed_depth=completed_depth,
        nodes=int(teacher.stats[row, 1]),
        final_result=float(final_result),
    )


def _teacher_batch_to_cpu(
    teacher: AlphaBetaTeacherBatch,
) -> AlphaBetaTeacherBatch:
    """Transfer each dense teacher output once instead of once per record."""

    return AlphaBetaTeacherBatch(*(
        tensor.detach().cpu() for tensor in (
            teacher.selected_moves,
            teacher.search_values,
            teacher.stats,
            teacher.raw_values,
            teacher.legal_moves,
            teacher.num_legal,
            teacher.root_scores,
            teacher.root_bounds,
            teacher.selected_indices,
            teacher.pv_moves,
            teacher.pv_lengths,
        )
    ))


def opening_diversity_candidates(
    root_scores: torch.Tensor,
    root_bounds: torch.Tensor,
    selected_index: int,
    *,
    max_candidates: int = 4,
    value_window: float = 0.12,
) -> list[int]:
    """Return exact near-best root moves, ordered by searched value."""

    if root_scores.ndim != 1 or root_bounds.ndim != 1:
        raise ValueError("root scores and bounds must be one-dimensional")
    if root_scores.numel() != root_bounds.numel():
        raise ValueError("root scores and bounds must have the same length")
    if not 0 <= selected_index < root_scores.numel():
        return []
    if (
        int(root_bounds[selected_index]) != int(AlphaBetaBound.EXACT)
        or not torch.isfinite(root_scores[selected_index])
    ):
        return []

    exact = root_bounds == int(AlphaBetaBound.EXACT)
    finite = torch.isfinite(root_scores)
    eligible = (exact & finite).nonzero().flatten().tolist()
    if not eligible:
        return []

    best_score = max(float(root_scores[index]) for index in eligible)
    threshold = best_score - max(0.0, float(value_window))
    candidates = [
        index for index in eligible
        if float(root_scores[index]) >= threshold
    ]
    candidates.sort(key=lambda index: (-float(root_scores[index]), index))
    return candidates[:max(1, int(max_candidates))]


def sample_opening_move_index(
    candidates: list[int],
    root_scores: torch.Tensor,
    rng: random.Random,
    *,
    temperature: float = 0.0,
) -> int:
    """Sample an exact candidate uniformly or with a mild score softmax."""

    if not candidates:
        raise ValueError("cannot sample from an empty candidate list")
    if len(candidates) == 1:
        return candidates[0]
    if temperature <= 0.0:
        return rng.choice(candidates)

    scale = max(float(temperature), 1e-6)
    best = max(float(root_scores[index]) for index in candidates)
    weights = [
        math.exp((float(root_scores[index]) - best) / scale)
        for index in candidates
    ]
    return rng.choices(candidates, weights=weights, k=1)[0]


@torch.inference_mode()
def generate_alpha_beta_records(
    net: HiveFNN,
    config: AlphaBetaGenerationConfig | None = None,
) -> tuple[list[AlphaBetaRecord], dict[str, int]]:
    """Generate alpha-beta self-play and optionally relabel sampled roots."""

    cfg = config or AlphaBetaGenerationConfig()
    torch.manual_seed(cfg.seed)
    ext = hive_gpu.load_extension()
    net = net.cuda().eval()
    expansion_runs = alpha_beta_expansion_runs(cfg.games, cfg.expansion_mask)
    states = torch.cat([
        ext.create_initial_states(count, mask)
        for mask, count in expansion_runs
    ], dim=0)
    endgame_count = min(
        cfg.games, round(cfg.games * max(0.0, min(1.0, cfg.endgame_fraction))),
    )
    if endgame_count:
        from hive_gpu.endgame_generator import (
            generate_endgame_positions, positions_to_tensor,
            rebalance_side_to_move,
        )
        positions = generate_endgame_positions(
            endgame_count, expansion_mask=cfg.expansion_mask,
            min_surround=cfg.endgame_min_surround,
            max_surround=cfg.endgame_max_surround,
            mixed_pair=cfg.endgame_mixed_pair, verbose=False,
        )
        positions = rebalance_side_to_move(positions, random.Random(cfg.seed))
        states[:endgame_count] = positions_to_tensor(positions)
    search_context = AlphaBetaGPUContext(
        net,
        capacity=max(cfg.games, cfg.relabel_batch_size),
        search_config=cfg.search_config,
    )
    active = torch.ones(cfg.games, dtype=torch.bool)
    histories: list[list[AlphaBetaRecord]] = [[] for _ in range(cfg.games)]
    game_rngs = [
        random.Random(cfg.seed + 0x9E3779B1 * (game + 1))
        for game in range(cfg.games)
    ]
    diverse_moves = 0
    candidate_positions = 0
    candidate_total = 0

    for ply in range(cfg.max_game_length):
        rows = active.nonzero().flatten()
        if rows.numel() == 0:
            break
        row_gpu = rows.to(device="cuda")
        roots = states.index_select(0, row_gpu).contiguous()
        teacher = search_teacher_batch(
            net,
            roots,
            node_budget=cfg.game_node_budget,
            max_depth=cfg.max_depth,
            root_exact_count=(
                cfg.opening_diversity_candidates
                if ply < cfg.opening_diversity_plies
                else 1
            ),
            search_config=cfg.search_config,
            context=search_context,
        )
        roots_cpu = roots.cpu()
        teacher_cpu = _teacher_batch_to_cpu(teacher)
        played_moves = teacher.selected_moves.clone()
        for local_row, game in enumerate(rows.tolist()):
            record = _record_from_teacher_row(
                roots_cpu[local_row], teacher_cpu, local_row,
            )
            if record is not None:
                histories[game].append(record)
            if ply >= cfg.opening_diversity_plies:
                continue
            n = int(teacher_cpu.num_legal[local_row])
            candidates = opening_diversity_candidates(
                teacher_cpu.root_scores[local_row, :n],
                teacher_cpu.root_bounds[local_row, :n],
                int(teacher_cpu.selected_indices[local_row]),
                max_candidates=cfg.opening_diversity_candidates,
                value_window=cfg.opening_diversity_value_window,
            )
            if not candidates:
                continue
            candidate_positions += 1
            candidate_total += len(candidates)
            sampled_index = sample_opening_move_index(
                candidates,
                teacher_cpu.root_scores[local_row],
                game_rngs[game],
                temperature=cfg.opening_diversity_temperature,
            )
            if sampled_index != int(teacher_cpu.selected_indices[local_row]):
                diverse_moves += 1
            played_moves[local_row] = teacher.legal_moves[local_row, sampled_index]

        ext.apply_moves_batch(
            roots, played_moves, int(rows.numel()),
        )
        states.index_copy_(0, row_gpu, roots)
        results = ext.check_results_batch(roots, int(rows.numel())).cpu()
        for local_row, game in enumerate(rows.tolist()):
            if int(results[local_row]) != 0:
                active[game] = False

    final_results = ext.check_results_batch(states, cfg.games).cpu()
    records: list[AlphaBetaRecord] = []
    for game, history in enumerate(histories):
        result = int(final_results[game])
        if result == 0 and not cfg.retain_truncated_teacher_records:
            # A move-limit cutoff is unknown, not a rule draw.
            continue
        for record in history:
            turn = _turn_from_state(record.state)
            records.append(replace(
                record,
                final_result=(
                    _outcome_for_turn(result, turn)
                    if result != 0 else float("nan")
                ),
            ))

    relabel_count = min(
        len(records),
        round(len(records) * max(0.0, min(1.0, cfg.teacher_relabel_fraction))),
    )
    if relabel_count and cfg.teacher_node_budget > cfg.game_node_budget:
        rng = random.Random(cfg.seed)
        relabel_indices = rng.sample(range(len(records)), relabel_count)
        for start in range(0, relabel_count, cfg.relabel_batch_size):
            indices = relabel_indices[start : start + cfg.relabel_batch_size]
            batch_states_cpu = torch.stack(
                [records[index].state for index in indices],
            )
            batch_states = batch_states_cpu.cuda()
            teacher = search_teacher_batch(
                net,
                batch_states,
                node_budget=cfg.teacher_node_budget,
                max_depth=cfg.max_depth,
                search_config=cfg.search_config,
                context=search_context,
            )
            teacher_cpu = _teacher_batch_to_cpu(teacher)
            for row, index in enumerate(indices):
                relabeled = _record_from_teacher_row(
                    batch_states_cpu[row],
                    teacher_cpu,
                    row,
                    final_result=records[index].final_result,
                )
                if relabeled is not None:
                    previous = records[index]
                    records[index] = replace(
                        relabeled,
                        depth_value_delta=abs(
                            relabeled.search_value - previous.search_value,
                        ),
                        depth_move_changed=not torch.equal(
                            relabeled.selected_move,
                            previous.selected_move,
                        ),
                    )

    stats = {
        "games": cfg.games,
        "records": len(records),
        "endgame_starts": endgame_count,
        "truncated_records_retained": sum(
            math.isnan(record.final_result) for record in records
        ),
        "relabel_requested": relabel_count,
        "white_wins": int((final_results == 1).sum()),
        "black_wins": int((final_results == 2).sum()),
        "draws": int((final_results == 3).sum()),
        "truncated_games": int((final_results == 0).sum()),
        "opening_candidate_positions": candidate_positions,
        "opening_candidates_total": candidate_total,
        "opening_diverse_moves": diverse_moves,
    }
    return records, stats


def collate_alpha_beta_records(
    records: list[AlphaBetaRecord],
    *,
    device: torch.device | str = "cuda",
) -> AlphaBetaTrainingBatch:
    if not records:
        raise ValueError("cannot collate an empty alpha-beta record list")
    target = torch.device(device)
    batch_size = len(records)
    max_legal = int(hive_gpu.load_extension().MAX_LEGAL_MOVES)
    move_size = records[0].legal_moves.shape[-1]
    states = torch.stack([record.state for record in records])
    legal_moves = torch.zeros(
        (batch_size, max_legal, move_size), dtype=torch.uint8,
    )
    num_legal = torch.empty(batch_size, dtype=torch.int64)
    for row, record in enumerate(records):
        n = record.legal_moves.shape[0]
        legal_moves[row, :n] = record.legal_moves
        num_legal[row] = n
    return AlphaBetaTrainingBatch(
        states=states.to(target),
        legal_moves=legal_moves.to(target),
        num_legal=num_legal.to(target),
        search_values=torch.tensor(
            [record.search_value for record in records],
            dtype=torch.float32,
            device=target,
        ),
        final_results=torch.tensor(
            [record.final_result for record in records],
            dtype=torch.float32,
            device=target,
        ),
    )


def _root_values(
    net: HiveFNN,
    batch: AlphaBetaTrainingBatch,
) -> torch.Tensor:
    """Evaluate root values without encoding or scoring successors."""
    ext = hive_gpu.load_extension()
    batch_size = batch.states.shape[0]
    root_features = ext.extract_fnn_features_batch(
        batch.states, batch.legal_moves, batch.num_legal.to(torch.int32), batch_size,
    )
    return net.value_head(net.encode(root_features)).squeeze(-1)


def alpha_beta_value_targets(
    search_values: torch.Tensor,
    final_results: torch.Tensor,
    search_value_weight: float = 0.25,
) -> torch.Tensor:
    """Blend player-to-move search values with player-to-move WDL outcomes."""

    weight = max(0.0, min(1.0, float(search_value_weight)))
    search = search_values.clamp(-1.0, 1.0)
    blended = weight * search + (1.0 - weight) * final_results
    return torch.where(torch.isfinite(final_results), blended, search)


def compute_alpha_beta_loss(
    net: HiveFNN,
    batch: AlphaBetaTrainingBatch,
    config: AlphaBetaLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the scalar leaf value used by alpha-beta search."""

    cfg = config or AlphaBetaLossConfig()
    root_values = _root_values(net, batch)
    value_targets = alpha_beta_value_targets(
        batch.search_values,
        batch.final_results,
        cfg.search_value_weight,
    )
    value_loss = F.mse_loss(root_values.float(), value_targets)
    total = cfg.value_loss_weight * value_loss
    return total, {
        "value_loss": value_loss,
        "value_target_mean": value_targets.mean(),
    }


def train_alpha_beta_batch(
    net: HiveFNN,
    optimizer: torch.optim.Optimizer,
    records: list[AlphaBetaRecord],
    config: AlphaBetaLossConfig | None = None,
    *,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    net.train()
    batch = collate_alpha_beta_records(records, device=next(net.parameters()).device)
    optimizer.zero_grad(set_to_none=True)
    loss, components = compute_alpha_beta_loss(net, batch, config)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        **{name: float(value.detach()) for name, value in components.items()},
    }
