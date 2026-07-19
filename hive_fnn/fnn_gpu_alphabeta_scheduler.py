"""Batched GPU evaluation service for cooperative alpha-beta searches.

Alpha-beta remains depth-first *within* one game, but many games can pause at
an expansion/leaf boundary.  This module batches those boundaries into one
GPU legal-move, successor-feature, policy-ordering, and value evaluation.
It deliberately owns no tree: a search worker retains its transposition table
and alpha/beta window, then resumes after receiving ``GPUExpansion``.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import TypeVar

import torch

import hive_gpu
from hive_fnn.fnn_network import HiveFNN


@dataclass(frozen=True)
class GPUExpansion:
    """One evaluated state, with legal moves ordered best-first by policy."""

    state: torch.Tensor
    legal_moves: torch.Tensor
    num_legal: int
    root_features: torch.Tensor
    root_value: float
    priors: torch.Tensor
    child_values: torch.Tensor
    order: torch.Tensor
    child_states: torch.Tensor
    child_results: torch.Tensor


@dataclass(frozen=True)
class GPUExpansionRequest:
    """A worker pauses by yielding one or more CUDA HiveState rows."""

    states: torch.Tensor


@dataclass(frozen=True)
class GPUAlphaBetaConfig:
    node_budget: int = 10_000
    max_depth: int = 32
    mate_score: float = 10.0


@dataclass
class GPUAlphaBetaResult:
    move: torch.Tensor
    value: float
    completed_depth: int
    nodes: int
    cutoffs: int


T = TypeVar("T")


class GPUAlphaBetaBatchScheduler:
    """Cooperatively batches GPU evaluation requests from alpha-beta workers.

    A worker is a generator that yields ``GPUExpansionRequest`` and receives
    a list of matching ``GPUExpansion`` objects.  Workers may yield again as
    soon as their depth-first traversal reaches another unknown node.  The
    scheduler collects one request from every runnable worker per round,
    evaluates the flattened frontier on GPU, then resumes each worker.
    """

    def __init__(self, net: HiveFNN, *, max_batch_states: int = 4096) -> None:
        self.ext = hive_gpu.load_extension()
        self.net = net.cuda().eval()
        self.max_batch_states = max(1, int(max_batch_states))
        self._max_legal = int(self.ext.MAX_LEGAL_MOVES)
        self._slots = torch.arange(self._max_legal, device="cuda", dtype=torch.int64)

    @torch.inference_mode()
    def evaluate(self, states: torch.Tensor) -> list[GPUExpansion]:
        """Evaluate a CUDA batch without returning results to the CPU mid-pass."""

        if states.ndim != 2:
            raise ValueError("states must have shape [batch, sizeof(HiveState)]")
        if states.device.type != "cuda":
            states = states.to(device="cuda", non_blocking=True)
        if states.dtype != torch.uint8:
            raise TypeError("states must be uint8 packed HiveState rows")

        out: list[GPUExpansion] = []
        for chunk in states.split(self.max_batch_states, dim=0):
            count = int(chunk.shape[0])
            moves, num_legal, root_features = (
                self.ext.generate_legal_moves_and_fnn_features_batch(chunk, count)
            )
            valid = self._slots.unsqueeze(0) < num_legal.to(torch.int64).unsqueeze(1)
            action_to_root = torch.arange(count, device="cuda", dtype=torch.int64).unsqueeze(1)
            action_to_root = action_to_root.expand_as(valid)[valid]
            move_indices = self._slots.unsqueeze(0).expand_as(valid)[valid]
            total_actions = int(action_to_root.numel())

            root_embed = self.net.encode(root_features)
            root_values = self.net.value_head(root_embed).squeeze(-1).float()
            priors = torch.zeros((count, self._max_legal), device="cuda", dtype=torch.float32)
            child_values = torch.zeros_like(priors)
            if total_actions:
                successor_features = self.ext.fnn_successor_features_batch(
                    chunk, moves, action_to_root, move_indices, total_actions,
                )
                successor_embed = self.net.encode(successor_features)
                logits = self.net.score_actions(
                    root_embed[action_to_root], successor_embed,
                    root_features[action_to_root], successor_features,
                ).float()
                padded_logits = torch.full_like(priors, -1e30)
                padded_logits[valid] = logits
                priors = torch.softmax(padded_logits, dim=1).masked_fill(~valid, 0.0)
                child_values[valid] = self.net.value_head(successor_embed).squeeze(-1).float()

            order = torch.argsort(priors, dim=1, descending=True, stable=True)
            # Small per-state handles are intentionally materialized only after
            # all neural work for the batch has completed.
            child_states = chunk.index_select(0, action_to_root).clone()
            child_moves = moves[action_to_root, move_indices].contiguous()
            if total_actions:
                self.ext.apply_moves_batch(child_states, child_moves, total_actions)
                child_results = self.ext.check_results_batch(child_states, total_actions)
            else:
                child_results = torch.empty(0, device="cuda", dtype=torch.int32)
            counts = num_legal.to(device="cpu", dtype=torch.int64).tolist()
            child_offset = 0
            for row, n in enumerate(counts):
                out.append(GPUExpansion(
                    state=chunk[row], legal_moves=moves[row, :n], num_legal=n,
                    root_features=root_features[row], root_value=root_values[row],
                    priors=priors[row, :n], child_values=child_values[row, :n],
                    order=order[row, :n],
                    child_states=child_states[child_offset:child_offset + n],
                    child_results=child_results[child_offset:child_offset + n],
                ))
                child_offset += n
        return out

    def run(self, workers: Iterable[Generator[GPUExpansionRequest, list[GPUExpansion], T]]) -> list[T]:
        """Run independent cooperative workers, batching one frontier per worker."""

        pending: list[tuple[Generator[GPUExpansionRequest, list[GPUExpansion], T], GPUExpansionRequest]] = []
        results: list[T] = []
        for worker in workers:
            try:
                pending.append((worker, next(worker)))
            except StopIteration as done:
                results.append(done.value)

        while pending:
            flat = torch.cat([request.states for _, request in pending], dim=0)
            expansions = self.evaluate(flat)
            next_pending: list[tuple[Generator[GPUExpansionRequest, list[GPUExpansion], T], GPUExpansionRequest]] = []
            offset = 0
            for worker, request in pending:
                count = int(request.states.shape[0])
                reply = expansions[offset:offset + count]
                offset += count
                try:
                    next_pending.append((worker, worker.send(reply)))
                except StopIteration as done:
                    results.append(done.value)
            pending = next_pending
        return results


class GPUAlphaBetaWorker:
    """One cooperative iterative-deepening alpha-beta search.

    It has no direct CUDA calls.  Every unknown node is yielded to
    ``GPUAlphaBetaBatchScheduler`` so arenas can batch searches from many
    games.  A transposition table is intentionally omitted in this first GPU
    path: hashing packed device states would otherwise introduce a device to
    host synchronization at every node.
    """

    _INF = 1_000.0

    def __init__(
        self,
        root_state: torch.Tensor,
        *,
        root_turn: int,
        config: GPUAlphaBetaConfig | None = None,
    ) -> None:
        if root_state.ndim != 1:
            raise ValueError("root_state must be one packed HiveState row")
        self.root_state = root_state
        self.root_turn = int(root_turn)
        self.config = config or GPUAlphaBetaConfig()
        self.nodes = 0
        self.cutoffs = 0

    def _consume_node(self) -> None:
        self.nodes += 1
        if self.nodes > max(1, int(self.config.node_budget)):
            raise RuntimeError("alpha-beta node budget exhausted")

    def _terminal_value(self, result: int, ply: int) -> float:
        if result in (0, 3):
            return 0.0
        side_is_white = ((self.root_turn + ply) & 1) == 0
        side_won = (result == 1 and side_is_white) or (result == 2 and not side_is_white)
        score = float(self.config.mate_score) - min(ply, 100) * 0.01
        return score if side_won else -score

    def _expand(self, state: torch.Tensor) -> Generator[GPUExpansionRequest, list[GPUExpansion], GPUExpansion]:
        reply = yield GPUExpansionRequest(state.unsqueeze(0))
        return reply[0]

    def _negamax(
        self,
        state: torch.Tensor,
        result: int,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> Generator[GPUExpansionRequest, list[GPUExpansion], float]:
        self._consume_node()
        if result:
            return self._terminal_value(result, ply)
        expansion = yield from self._expand(state)
        if depth <= 0 or expansion.num_legal == 0:
            return float(expansion.root_value.item())

        best = -self._INF
        for index in expansion.order.tolist():
            child_result = int(expansion.child_results[index].item())
            value = -(yield from self._negamax(
                expansion.child_states[index], child_result, depth - 1,
                -beta, -alpha, ply + 1,
            ))
            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta:
                self.cutoffs += 1
                break
        return best

    def run(self) -> Generator[GPUExpansionRequest, list[GPUExpansion], GPUAlphaBetaResult]:
        root = yield from self._expand(self.root_state)
        if root.num_legal == 0:
            return GPUAlphaBetaResult(
                move=torch.empty(0, dtype=torch.uint8, device="cuda"), value=0.0,
                completed_depth=0, nodes=0, cutoffs=0,
            )
        fallback = root.legal_moves[int(root.order[0].item())]
        best_move = fallback
        best_value = float(root.root_value.item())
        completed = 0
        for depth in range(1, max(1, int(self.config.max_depth)) + 1):
            try:
                self._consume_node()
                alpha = -self._INF
                value = -self._INF
                move = best_move
                for index in root.order.tolist():
                    child_result = int(root.child_results[index].item())
                    child_value = -(yield from self._negamax(
                        root.child_states[index], child_result, depth - 1,
                        -self._INF, -alpha, 1,
                    ))
                    if child_value > value:
                        value = child_value
                        move = root.legal_moves[index]
                    alpha = max(alpha, value)
                best_move, best_value, completed = move, value, depth
                if abs(value) >= float(self.config.mate_score) - 1.0:
                    break
            except RuntimeError as exc:
                if str(exc) != "alpha-beta node budget exhausted":
                    raise
                break
        return GPUAlphaBetaResult(
            move=best_move, value=best_value, completed_depth=completed,
            nodes=self.nodes, cutoffs=self.cutoffs,
        )
