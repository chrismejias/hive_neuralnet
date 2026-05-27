"""Tensor-backed replay buffer for FNN-style training examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

import hive_gpu
from hive_fnn.fnn_features import FEAT_DIM

_OFF_TURN = 3412
_HYBRID_NODE_FEAT_DIM = 26
_HYBRID_GLOBAL_FEAT_DIM = 6
_HYBRID_MAX_PIECE_TOKENS = 28
_HYBRID_MOVE_FEAT_DIM = 25


class FNNTrainingExample(NamedTuple):
    """Root-state training example with policy over its legal move list."""

    state_bytes: np.ndarray
    policy_target: np.ndarray
    value_target: float
    use_for_policy: bool = True
    use_for_value: bool = True
    value_weight: float = 1.0
    surprise_weight: float = 1.0


@dataclass
class FNNTrainingBatch:
    state_bytes: torch.Tensor
    policy_targets: torch.Tensor
    num_actions: torch.Tensor
    value_targets: torch.Tensor
    policy_mask: torch.Tensor
    value_mask: torch.Tensor
    value_weights: torch.Tensor
    root_features: torch.Tensor | None = None
    legal_moves: torch.Tensor | None = None
    token_features: torch.Tensor | None = None
    token_q: torch.Tensor | None = None
    token_r: torch.Tensor | None = None
    token_z: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    global_features: torch.Tensor | None = None
    move_features: torch.Tensor | None = None

    def to(self, device: torch.device, non_blocking: bool = False) -> FNNTrainingBatch:
        return FNNTrainingBatch(
            state_bytes=self.state_bytes.to(device, non_blocking=non_blocking),
            policy_targets=self.policy_targets.to(device, non_blocking=non_blocking),
            num_actions=self.num_actions.to(device, non_blocking=non_blocking),
            value_targets=self.value_targets.to(device, non_blocking=non_blocking),
            policy_mask=self.policy_mask.to(device, non_blocking=non_blocking),
            value_mask=self.value_mask.to(device, non_blocking=non_blocking),
            value_weights=self.value_weights.to(device, non_blocking=non_blocking),
            root_features=(
                None if self.root_features is None
                else self.root_features.to(device, non_blocking=non_blocking)
            ),
            legal_moves=(
                None if self.legal_moves is None
                else self.legal_moves.to(device, non_blocking=non_blocking)
            ),
            token_features=(
                None if self.token_features is None
                else self.token_features.to(device, non_blocking=non_blocking)
            ),
            token_q=(
                None if self.token_q is None
                else self.token_q.to(device, non_blocking=non_blocking)
            ),
            token_r=(
                None if self.token_r is None
                else self.token_r.to(device, non_blocking=non_blocking)
            ),
            token_z=(
                None if self.token_z is None
                else self.token_z.to(device, non_blocking=non_blocking)
            ),
            token_mask=(
                None if self.token_mask is None
                else self.token_mask.to(device, non_blocking=non_blocking)
            ),
            global_features=(
                None if self.global_features is None
                else self.global_features.to(device, non_blocking=non_blocking)
            ),
            move_features=(
                None if self.move_features is None
                else self.move_features.to(device, non_blocking=non_blocking)
            ),
        )


class FNNReplayBuffer:
    """Contiguous replay storage with optional cached GPU features."""

    def __init__(
        self,
        max_size: int = 100_000,
        *,
        device: torch.device | str | None = None,
        cache_root_features: bool = True,
        cache_hybrid_root_features: bool = False,
        gpu_sampling: bool = True,
        merge_opening_value_examples: bool = True,
        opening_value_merge_plies: int = 4,
    ) -> None:
        self.max_size = int(max_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_hybrid_root_features = bool(cache_hybrid_root_features)
        self.cache_root_features = bool(cache_root_features or self.cache_hybrid_root_features)
        self.gpu_sampling = bool(gpu_sampling and self.device.type == "cuda")
        self.merge_opening_value_examples = bool(merge_opening_value_examples)
        self.opening_value_merge_plies = max(0, int(opening_value_merge_plies))
        self.ext = hive_gpu.load_extension()
        self.max_actions = int(self.ext.MAX_LEGAL_MOVES)
        self.move_size = int(self.ext.SIZEOF_GPU_MOVE)
        self.state_size: int | None = None

        self._size = 0
        self._next_idx = 0
        self._uniform_weights = True

        self._states_cpu: torch.Tensor | None = None
        self._policy_cpu: torch.Tensor | None = None
        self._num_actions_cpu: torch.Tensor | None = None
        self._values_cpu: torch.Tensor | None = None
        self._policy_mask_cpu: torch.Tensor | None = None
        self._value_mask_cpu: torch.Tensor | None = None
        self._weights_cpu: torch.Tensor | None = None
        self._value_weights_cpu: torch.Tensor | None = None
        self._root_features_cpu: torch.Tensor | None = None
        self._legal_moves_cpu: torch.Tensor | None = None
        self._token_features_cpu: torch.Tensor | None = None
        self._token_q_cpu: torch.Tensor | None = None
        self._token_r_cpu: torch.Tensor | None = None
        self._token_z_cpu: torch.Tensor | None = None
        self._token_mask_cpu: torch.Tensor | None = None
        self._global_features_cpu: torch.Tensor | None = None
        self._move_features_cpu: torch.Tensor | None = None

        self._states_gpu: torch.Tensor | None = None
        self._policy_gpu: torch.Tensor | None = None
        self._num_actions_gpu: torch.Tensor | None = None
        self._values_gpu: torch.Tensor | None = None
        self._policy_mask_gpu: torch.Tensor | None = None
        self._value_mask_gpu: torch.Tensor | None = None
        self._weights_gpu: torch.Tensor | None = None
        self._value_weights_gpu: torch.Tensor | None = None
        self._root_features_gpu: torch.Tensor | None = None
        self._legal_moves_gpu: torch.Tensor | None = None
        self._token_features_gpu: torch.Tensor | None = None
        self._token_q_gpu: torch.Tensor | None = None
        self._token_r_gpu: torch.Tensor | None = None
        self._token_z_gpu: torch.Tensor | None = None
        self._token_mask_gpu: torch.Tensor | None = None
        self._global_features_gpu: torch.Tensor | None = None
        self._move_features_gpu: torch.Tensor | None = None

    def __len__(self) -> int:
        return self._size

    def _allocate_storage(self, state_size: int) -> None:
        self.state_size = int(state_size)
        pin = True
        self._states_cpu = torch.empty(
            (self.max_size, state_size), dtype=torch.uint8, pin_memory=pin,
        )
        self._policy_cpu = torch.zeros(
            (self.max_size, self.max_actions), dtype=torch.float16, pin_memory=pin,
        )
        self._num_actions_cpu = torch.zeros(
            (self.max_size,), dtype=torch.int64, pin_memory=pin,
        )
        self._values_cpu = torch.zeros(
            (self.max_size, 1), dtype=torch.float32, pin_memory=pin,
        )
        self._policy_mask_cpu = torch.zeros(
            (self.max_size, 1), dtype=torch.float32, pin_memory=pin,
        )
        self._value_mask_cpu = torch.zeros(
            (self.max_size, 1), dtype=torch.float32, pin_memory=pin,
        )
        self._weights_cpu = torch.ones(
            (self.max_size,), dtype=torch.float32, pin_memory=pin,
        )
        self._value_weights_cpu = torch.ones(
            (self.max_size, 1), dtype=torch.float32, pin_memory=pin,
        )
        if self.cache_root_features:
            self._root_features_cpu = torch.zeros(
                (self.max_size, FEAT_DIM), dtype=torch.float16, pin_memory=pin,
            )
            self._legal_moves_cpu = torch.zeros(
                (self.max_size, self.max_actions, self.move_size),
                dtype=torch.uint8,
                pin_memory=pin,
            )
        if self.cache_hybrid_root_features:
            self._token_features_cpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS, _HYBRID_NODE_FEAT_DIM),
                dtype=torch.float16,
                pin_memory=pin,
            )
            self._token_q_cpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS), dtype=torch.int16, pin_memory=pin,
            )
            self._token_r_cpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS), dtype=torch.int16, pin_memory=pin,
            )
            self._token_z_cpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS), dtype=torch.int16, pin_memory=pin,
            )
            self._token_mask_cpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS), dtype=torch.bool, pin_memory=pin,
            )
            self._global_features_cpu = torch.zeros(
                (self.max_size, _HYBRID_GLOBAL_FEAT_DIM), dtype=torch.float16, pin_memory=pin,
            )
            self._move_features_cpu = torch.zeros(
                (self.max_size, self.max_actions, _HYBRID_MOVE_FEAT_DIM),
                dtype=torch.float16,
                pin_memory=pin,
            )

        if not self.gpu_sampling:
            return

        dev = self.device
        self._states_gpu = torch.empty((self.max_size, state_size), dtype=torch.uint8, device=dev)
        self._policy_gpu = torch.zeros((self.max_size, self.max_actions), dtype=torch.float16, device=dev)
        self._num_actions_gpu = torch.zeros((self.max_size,), dtype=torch.int64, device=dev)
        self._values_gpu = torch.zeros((self.max_size, 1), dtype=torch.float32, device=dev)
        self._policy_mask_gpu = torch.zeros((self.max_size, 1), dtype=torch.float32, device=dev)
        self._value_mask_gpu = torch.zeros((self.max_size, 1), dtype=torch.float32, device=dev)
        self._weights_gpu = torch.ones((self.max_size,), dtype=torch.float32, device=dev)
        self._value_weights_gpu = torch.ones((self.max_size, 1), dtype=torch.float32, device=dev)
        if self.cache_root_features:
            self._root_features_gpu = torch.zeros(
                (self.max_size, FEAT_DIM), dtype=torch.float16, device=dev,
            )
            self._legal_moves_gpu = torch.zeros(
                (self.max_size, self.max_actions, self.move_size),
                dtype=torch.uint8,
                device=dev,
            )
        if self.cache_hybrid_root_features:
            self._token_features_gpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS, _HYBRID_NODE_FEAT_DIM),
                dtype=torch.float16,
                device=dev,
            )
            self._token_q_gpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS), dtype=torch.int16, device=dev,
            )
            self._token_r_gpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS), dtype=torch.int16, device=dev,
            )
            self._token_z_gpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS), dtype=torch.int16, device=dev,
            )
            self._token_mask_gpu = torch.zeros(
                (self.max_size, _HYBRID_MAX_PIECE_TOKENS), dtype=torch.bool, device=dev,
            )
            self._global_features_gpu = torch.zeros(
                (self.max_size, _HYBRID_GLOBAL_FEAT_DIM), dtype=torch.float16, device=dev,
            )
            self._move_features_gpu = torch.zeros(
                (self.max_size, self.max_actions, _HYBRID_MOVE_FEAT_DIM),
                dtype=torch.float16,
                device=dev,
            )

    def _target_slices(self, start: int, count: int) -> list[tuple[int, int]]:
        if count <= 0:
            return []
        end = start + count
        if end <= self.max_size:
            return [(start, end)]
        return [(start, self.max_size), (0, end - self.max_size)]

    def add_examples(self, examples: list[FNNTrainingExample]) -> None:
        if not examples:
            return
        examples = self._compact_opening_value_examples(examples)
        if self.state_size is None:
            self._allocate_storage(int(examples[0].state_bytes.shape[0]))
        assert self._states_cpu is not None
        count = len(examples)
        offset = 0
        for dst_lo, dst_hi in self._target_slices(self._next_idx, count):
            chunk = examples[offset: offset + (dst_hi - dst_lo)]
            self._write_chunk(chunk, dst_lo)
            offset += dst_hi - dst_lo

        self._next_idx = (self._next_idx + count) % self.max_size
        self._size = min(self.max_size, self._size + count)

    @staticmethod
    def _state_turn(state_bytes: np.ndarray) -> int:
        return int(state_bytes[_OFF_TURN]) | (int(state_bytes[_OFF_TURN + 1]) << 8)

    def _compact_opening_value_examples(
        self,
        examples: list[FNNTrainingExample],
    ) -> list[FNNTrainingExample]:
        if not self.merge_opening_value_examples or self.opening_value_merge_plies <= 0:
            return examples

        merged: list[FNNTrainingExample] = []
        opening_groups: dict[bytes, list[FNNTrainingExample]] = {}

        for ex in examples:
            if (
                not ex.use_for_value
                or self._state_turn(ex.state_bytes) >= self.opening_value_merge_plies
            ):
                merged.append(ex)
                continue
            opening_groups.setdefault(ex.state_bytes.tobytes(), []).append(ex)

        for grouped in opening_groups.values():
            if len(grouped) == 1:
                merged.append(grouped[0])
                continue

            for ex in grouped:
                merged.append(ex._replace(use_for_value=False))

            total_weight = sum(float(ex.value_weight) for ex in grouped)
            if total_weight <= 0.0:
                total_weight = float(len(grouped))
            avg_value = sum(
                float(ex.value_target) * float(ex.value_weight) for ex in grouped
            ) / total_weight
            rep = grouped[0]
            merged.append(FNNTrainingExample(
                state_bytes=rep.state_bytes,
                policy_target=rep.policy_target,
                value_target=float(avg_value),
                use_for_policy=False,
                use_for_value=True,
                value_weight=float(total_weight),
                surprise_weight=rep.surprise_weight,
            ))

        return merged

    def _write_chunk(self, examples: list[FNNTrainingExample], dst_lo: int) -> None:
        if not examples:
            return
        B = len(examples)
        assert self._states_cpu is not None
        assert self._policy_cpu is not None
        assert self._num_actions_cpu is not None
        assert self._values_cpu is not None
        assert self._policy_mask_cpu is not None
        assert self._value_mask_cpu is not None
        assert self._weights_cpu is not None
        assert self._value_weights_cpu is not None

        states_np = np.stack([ex.state_bytes for ex in examples], axis=0).astype(np.uint8, copy=False)
        policy_np = np.zeros((B, self.max_actions), dtype=np.float16)
        num_actions_np = np.zeros((B,), dtype=np.int64)
        values_np = np.zeros((B, 1), dtype=np.float32)
        policy_mask_np = np.zeros((B, 1), dtype=np.float32)
        value_mask_np = np.zeros((B, 1), dtype=np.float32)
        weights_np = np.ones((B,), dtype=np.float32)
        value_weights_np = np.ones((B, 1), dtype=np.float32)
        self._uniform_weights = self._uniform_weights and all(
            ex.surprise_weight == 1.0 for ex in examples
        )

        for i, ex in enumerate(examples):
            n = int(ex.policy_target.shape[0])
            policy_np[i, :n] = ex.policy_target.astype(np.float16, copy=False)
            num_actions_np[i] = n
            values_np[i, 0] = ex.value_target
            policy_mask_np[i, 0] = float(ex.use_for_policy)
            value_mask_np[i, 0] = float(ex.use_for_value)
            weights_np[i] = ex.surprise_weight
            value_weights_np[i, 0] = ex.value_weight

        dst = slice(dst_lo, dst_lo + B)
        states_cpu = torch.from_numpy(states_np)
        policy_cpu = torch.from_numpy(policy_np)
        num_actions_cpu = torch.from_numpy(num_actions_np)
        values_cpu = torch.from_numpy(values_np)
        policy_mask_cpu = torch.from_numpy(policy_mask_np)
        value_mask_cpu = torch.from_numpy(value_mask_np)
        weights_cpu = torch.from_numpy(weights_np)
        value_weights_cpu = torch.from_numpy(value_weights_np)

        self._states_cpu[dst].copy_(states_cpu, non_blocking=False)
        self._policy_cpu[dst].copy_(policy_cpu, non_blocking=False)
        self._num_actions_cpu[dst].copy_(num_actions_cpu, non_blocking=False)
        self._values_cpu[dst].copy_(values_cpu, non_blocking=False)
        self._policy_mask_cpu[dst].copy_(policy_mask_cpu, non_blocking=False)
        self._value_mask_cpu[dst].copy_(value_mask_cpu, non_blocking=False)
        self._weights_cpu[dst].copy_(weights_cpu, non_blocking=False)
        self._value_weights_cpu[dst].copy_(value_weights_cpu, non_blocking=False)

        states_gpu = states_cpu.to(self.device, non_blocking=True) if self.device.type == "cuda" else states_cpu
        legal_moves_gpu = None
        root_features_gpu = None
        if self.cache_hybrid_root_features and self.device.type == "cuda":
            (
                legal_moves_gpu,
                num_legal_gpu,
                root_features_gpu,
                token_features_gpu,
                token_q_gpu,
                token_r_gpu,
                token_z_gpu,
                token_mask_gpu,
                global_features_gpu,
                move_features_gpu,
            ) = self.ext.generate_legal_moves_and_hybrid_root_features_batch(states_gpu, B)
            assert self._root_features_cpu is not None
            assert self._legal_moves_cpu is not None
            assert self._token_features_cpu is not None
            assert self._token_q_cpu is not None
            assert self._token_r_cpu is not None
            assert self._token_z_cpu is not None
            assert self._token_mask_cpu is not None
            assert self._global_features_cpu is not None
            assert self._move_features_cpu is not None
            self._root_features_cpu[dst].copy_(
                root_features_gpu.to(dtype=torch.float16, device="cpu", non_blocking=True),
            )
            self._legal_moves_cpu[dst].copy_(
                legal_moves_gpu.to(device="cpu", non_blocking=True),
            )
            self._token_features_cpu[dst].copy_(
                token_features_gpu.to(dtype=torch.float16, device="cpu", non_blocking=True),
            )
            self._token_q_cpu[dst].copy_(
                token_q_gpu.to(dtype=torch.int16, device="cpu", non_blocking=True),
            )
            self._token_r_cpu[dst].copy_(
                token_r_gpu.to(dtype=torch.int16, device="cpu", non_blocking=True),
            )
            self._token_z_cpu[dst].copy_(
                token_z_gpu.to(dtype=torch.int16, device="cpu", non_blocking=True),
            )
            self._token_mask_cpu[dst].copy_(
                token_mask_gpu.to(dtype=torch.bool, device="cpu", non_blocking=True),
            )
            self._global_features_cpu[dst].copy_(
                global_features_gpu.to(dtype=torch.float16, device="cpu", non_blocking=True),
            )
            self._move_features_cpu[dst].copy_(
                move_features_gpu.to(dtype=torch.float16, device="cpu", non_blocking=True),
            )
            self._num_actions_cpu[dst].copy_(
                num_legal_gpu.to(dtype=torch.int64, device="cpu", non_blocking=True),
            )
        elif self.cache_root_features and self.device.type == "cuda":
            legal_moves_gpu, num_legal_gpu, root_features_gpu = (
                self.ext.generate_legal_moves_and_fnn_features_batch(states_gpu, B)
            )
            assert self._root_features_cpu is not None
            assert self._legal_moves_cpu is not None
            self._root_features_cpu[dst].copy_(
                root_features_gpu.to(dtype=torch.float16, device="cpu", non_blocking=True),
            )
            self._legal_moves_cpu[dst].copy_(
                legal_moves_gpu.to(device="cpu", non_blocking=True),
            )
            self._num_actions_cpu[dst].copy_(
                num_legal_gpu.to(dtype=torch.int64, device="cpu", non_blocking=True),
            )

        if not self.gpu_sampling:
            return

        assert self._states_gpu is not None
        assert self._policy_gpu is not None
        assert self._num_actions_gpu is not None
        assert self._values_gpu is not None
        assert self._policy_mask_gpu is not None
        assert self._value_mask_gpu is not None
        assert self._weights_gpu is not None
        assert self._value_weights_gpu is not None
        self._states_gpu[dst].copy_(states_gpu, non_blocking=True)
        self._policy_gpu[dst].copy_(policy_cpu.to(self.device, non_blocking=True), non_blocking=True)
        self._num_actions_gpu[dst].copy_(num_actions_cpu.to(self.device, non_blocking=True), non_blocking=True)
        self._values_gpu[dst].copy_(values_cpu.to(self.device, non_blocking=True), non_blocking=True)
        self._policy_mask_gpu[dst].copy_(policy_mask_cpu.to(self.device, non_blocking=True), non_blocking=True)
        self._value_mask_gpu[dst].copy_(value_mask_cpu.to(self.device, non_blocking=True), non_blocking=True)
        self._weights_gpu[dst].copy_(weights_cpu.to(self.device, non_blocking=True), non_blocking=True)
        self._value_weights_gpu[dst].copy_(value_weights_cpu.to(self.device, non_blocking=True), non_blocking=True)
        if self.cache_root_features:
            assert legal_moves_gpu is not None and root_features_gpu is not None
            assert self._root_features_gpu is not None
            assert self._legal_moves_gpu is not None
            self._root_features_gpu[dst].copy_(
                root_features_gpu.to(dtype=torch.float16),
                non_blocking=True,
            )
            self._legal_moves_gpu[dst].copy_(legal_moves_gpu, non_blocking=True)
        if self.cache_hybrid_root_features:
            assert self._token_features_gpu is not None
            assert self._token_q_gpu is not None
            assert self._token_r_gpu is not None
            assert self._token_z_gpu is not None
            assert self._token_mask_gpu is not None
            assert self._global_features_gpu is not None
            assert self._move_features_gpu is not None
            self._token_features_gpu[dst].copy_(
                token_features_gpu.to(dtype=torch.float16),
                non_blocking=True,
            )
            self._token_q_gpu[dst].copy_(token_q_gpu.to(dtype=torch.int16), non_blocking=True)
            self._token_r_gpu[dst].copy_(token_r_gpu.to(dtype=torch.int16), non_blocking=True)
            self._token_z_gpu[dst].copy_(token_z_gpu.to(dtype=torch.int16), non_blocking=True)
            self._token_mask_gpu[dst].copy_(token_mask_gpu.to(dtype=torch.bool), non_blocking=True)
            self._global_features_gpu[dst].copy_(
                global_features_gpu.to(dtype=torch.float16),
                non_blocking=True,
            )
            self._move_features_gpu[dst].copy_(
                move_features_gpu.to(dtype=torch.float16),
                non_blocking=True,
            )

    def _sample_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        size = self._size
        if self._uniform_weights:
            return torch.randint(0, size, (batch_size,), device=device, dtype=torch.int64)
        if device.type == "cuda" and self._weights_gpu is not None:
            weights = self._weights_gpu[:size]
            return torch.multinomial(weights / weights.sum(), batch_size, replacement=True)
        assert self._weights_cpu is not None
        weights = self._weights_cpu[:size]
        probs = weights / weights.sum()
        return torch.multinomial(probs, batch_size, replacement=True).to(device=device)

    def sample_batch(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> FNNTrainingBatch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        target_device = torch.device(device or "cpu")
        if self.gpu_sampling and target_device.type == "cuda":
            return self._sample_batch_gpu(batch_size, target_device)
        return self._sample_batch_cpu(batch_size).to(target_device, non_blocking=non_blocking)

    def _sample_batch_gpu(self, batch_size: int, device: torch.device) -> FNNTrainingBatch:
        assert self._states_gpu is not None
        assert self._policy_gpu is not None
        assert self._num_actions_gpu is not None
        assert self._values_gpu is not None
        assert self._policy_mask_gpu is not None
        assert self._value_mask_gpu is not None
        assert self._value_weights_gpu is not None
        idx = self._sample_indices(batch_size, device)
        root_features = None
        legal_moves = None
        token_features = None
        token_q = None
        token_r = None
        token_z = None
        token_mask = None
        global_features = None
        move_features = None
        if self.cache_root_features:
            assert self._root_features_gpu is not None
            assert self._legal_moves_gpu is not None
            root_features = self._root_features_gpu.index_select(0, idx).float()
            legal_moves = self._legal_moves_gpu.index_select(0, idx)
        if self.cache_hybrid_root_features:
            assert self._token_features_gpu is not None
            assert self._token_q_gpu is not None
            assert self._token_r_gpu is not None
            assert self._token_z_gpu is not None
            assert self._token_mask_gpu is not None
            assert self._global_features_gpu is not None
            assert self._move_features_gpu is not None
            token_features = self._token_features_gpu.index_select(0, idx).float()
            token_q = self._token_q_gpu.index_select(0, idx)
            token_r = self._token_r_gpu.index_select(0, idx)
            token_z = self._token_z_gpu.index_select(0, idx)
            token_mask = self._token_mask_gpu.index_select(0, idx)
            global_features = self._global_features_gpu.index_select(0, idx).float()
            move_features = self._move_features_gpu.index_select(0, idx).float()
        return FNNTrainingBatch(
            state_bytes=self._states_gpu.index_select(0, idx),
            policy_targets=self._policy_gpu.index_select(0, idx).float(),
            num_actions=self._num_actions_gpu.index_select(0, idx),
            value_targets=self._values_gpu.index_select(0, idx),
            policy_mask=self._policy_mask_gpu.index_select(0, idx),
            value_mask=self._value_mask_gpu.index_select(0, idx),
            value_weights=self._value_weights_gpu.index_select(0, idx),
            root_features=root_features,
            legal_moves=legal_moves,
            token_features=token_features,
            token_q=token_q,
            token_r=token_r,
            token_z=token_z,
            token_mask=token_mask,
            global_features=global_features,
            move_features=move_features,
        )

    def _sample_batch_cpu(self, batch_size: int) -> FNNTrainingBatch:
        assert self._states_cpu is not None
        assert self._policy_cpu is not None
        assert self._num_actions_cpu is not None
        assert self._values_cpu is not None
        assert self._policy_mask_cpu is not None
        assert self._value_mask_cpu is not None
        assert self._value_weights_cpu is not None
        idx = self._sample_indices(batch_size, torch.device("cpu"))
        root_features = None
        legal_moves = None
        token_features = None
        token_q = None
        token_r = None
        token_z = None
        token_mask = None
        global_features = None
        move_features = None
        if self.cache_root_features:
            assert self._root_features_cpu is not None
            assert self._legal_moves_cpu is not None
            root_features = self._root_features_cpu.index_select(0, idx).float()
            legal_moves = self._legal_moves_cpu.index_select(0, idx)
        if self.cache_hybrid_root_features:
            assert self._token_features_cpu is not None
            assert self._token_q_cpu is not None
            assert self._token_r_cpu is not None
            assert self._token_z_cpu is not None
            assert self._token_mask_cpu is not None
            assert self._global_features_cpu is not None
            assert self._move_features_cpu is not None
            token_features = self._token_features_cpu.index_select(0, idx).float()
            token_q = self._token_q_cpu.index_select(0, idx)
            token_r = self._token_r_cpu.index_select(0, idx)
            token_z = self._token_z_cpu.index_select(0, idx)
            token_mask = self._token_mask_cpu.index_select(0, idx)
            global_features = self._global_features_cpu.index_select(0, idx).float()
            move_features = self._move_features_cpu.index_select(0, idx).float()
        return FNNTrainingBatch(
            state_bytes=self._states_cpu.index_select(0, idx),
            policy_targets=self._policy_cpu.index_select(0, idx).float(),
            num_actions=self._num_actions_cpu.index_select(0, idx),
            value_targets=self._values_cpu.index_select(0, idx),
            policy_mask=self._policy_mask_cpu.index_select(0, idx),
            value_mask=self._value_mask_cpu.index_select(0, idx),
            value_weights=self._value_weights_cpu.index_select(0, idx),
            root_features=root_features,
            legal_moves=legal_moves,
            token_features=token_features,
            token_q=token_q,
            token_r=token_r,
            token_z=token_z,
            token_mask=token_mask,
            global_features=global_features,
            move_features=move_features,
        )
