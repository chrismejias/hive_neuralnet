"""Tensor-backed replay buffer for PRS v2 / v3 training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

import hive_gpu
from hive_prs.action_space import MAX_BOARD
from hive_prs.prs_aux_targets import compute_articulation_target
from hive_prs.prs_c6_augment import rotate_moves_batch, rotate_states_batch
from hive_prs.prs_encoder import PRSEncoder, PRSTokenBatch, PRS_MAX_SEQ_LEN, PRS_OFF_BOARD
from hive_prs.slot_map import N_SLOTS

_TOKEN_FEAT_DIM = 25


class PRSTrainingExampleV2(NamedTuple):
    token_features: np.ndarray
    token_positions: np.ndarray
    token_types: np.ndarray
    num_board_tokens: int
    global_features: np.ndarray
    seq_length: int
    state_bytes: np.ndarray
    legal_moves: np.ndarray
    visit_counts: np.ndarray
    nlegal: int
    occupied_cells: np.ndarray
    num_occupied: int
    slot_target: np.ndarray
    legal_mask: np.ndarray
    articulation_target: np.ndarray
    articulation_mask: np.ndarray
    value_target: float
    use_for_value: bool


@dataclass
class PRSTrainingBatchV2:
    prs_batch: PRSTokenBatch
    state_bytes: torch.Tensor
    legal_moves_raw: torch.Tensor
    nlegal_raw: torch.Tensor
    slot_targets: torch.Tensor
    legal_masks: torch.Tensor
    value_targets: torch.Tensor
    value_masks: torch.Tensor
    articulation_targets: torch.Tensor
    articulation_mask: torch.Tensor
    augmentation_k: int = 0

    def to(self, device, non_blocking: bool = False) -> "PRSTrainingBatchV2":
        return PRSTrainingBatchV2(
            prs_batch=self.prs_batch.to(device, non_blocking=non_blocking),
            state_bytes=self.state_bytes.to(device, non_blocking=non_blocking),
            legal_moves_raw=self.legal_moves_raw.to(device, non_blocking=non_blocking),
            nlegal_raw=self.nlegal_raw.to(device, non_blocking=non_blocking),
            slot_targets=self.slot_targets.to(device, non_blocking=non_blocking),
            legal_masks=self.legal_masks.to(device, non_blocking=non_blocking),
            value_targets=self.value_targets.to(device, non_blocking=non_blocking),
            value_masks=self.value_masks.to(device, non_blocking=non_blocking),
            articulation_targets=self.articulation_targets.to(device, non_blocking=non_blocking),
            articulation_mask=self.articulation_mask.to(device, non_blocking=non_blocking),
            augmentation_k=self.augmentation_k,
        )


_encoder: PRSEncoder | None = None


def _get_encoder() -> PRSEncoder:
    global _encoder
    if _encoder is None:
        _encoder = PRSEncoder()
    return _encoder


class PRSReplayBufferV2:
    """Contiguous CPU replay storage with optional rotation augmentation."""

    def __init__(self, max_size: int = 100_000) -> None:
        self.max_size = int(max_size)
        self.ext = hive_gpu.load_extension()
        self.max_legal = int(self.ext.MAX_LEGAL_MOVES)
        self.move_size = int(self.ext.SIZEOF_GPU_MOVE)
        self.state_size: int | None = None
        self._size = 0
        self._next_idx = 0

        self._token_features: torch.Tensor | None = None
        self._token_positions: torch.Tensor | None = None
        self._token_types: torch.Tensor | None = None
        self._num_board_tokens: torch.Tensor | None = None
        self._global_features: torch.Tensor | None = None
        self._seq_lengths: torch.Tensor | None = None
        self._occupied_cells: torch.Tensor | None = None
        self._num_occupied: torch.Tensor | None = None
        self._state_bytes: torch.Tensor | None = None
        self._legal_moves: torch.Tensor | None = None
        self._visit_counts: torch.Tensor | None = None
        self._nlegal: torch.Tensor | None = None
        self._slot_targets: torch.Tensor | None = None
        self._legal_masks: torch.Tensor | None = None
        self._articulation_targets: torch.Tensor | None = None
        self._articulation_masks: torch.Tensor | None = None
        self._value_targets: torch.Tensor | None = None
        self._value_masks: torch.Tensor | None = None

    def __len__(self) -> int:
        return self._size

    def _allocate_storage(self, state_size: int) -> None:
        self.state_size = int(state_size)
        pin = True
        self._token_features = torch.zeros(
            (self.max_size, PRS_MAX_SEQ_LEN, _TOKEN_FEAT_DIM),
            dtype=torch.float32,
            pin_memory=pin,
        )
        self._token_positions = torch.full(
            (self.max_size, PRS_MAX_SEQ_LEN),
            PRS_OFF_BOARD,
            dtype=torch.int64,
            pin_memory=pin,
        )
        self._token_types = torch.zeros(
            (self.max_size, PRS_MAX_SEQ_LEN), dtype=torch.int64, pin_memory=pin,
        )
        self._num_board_tokens = torch.zeros(
            (self.max_size,), dtype=torch.int64, pin_memory=pin,
        )
        self._global_features = torch.zeros(
            (self.max_size, 6), dtype=torch.float32, pin_memory=pin,
        )
        self._seq_lengths = torch.zeros(
            (self.max_size,), dtype=torch.int64, pin_memory=pin,
        )
        self._occupied_cells = torch.full(
            (self.max_size, MAX_BOARD), -1, dtype=torch.int32, pin_memory=pin,
        )
        self._num_occupied = torch.zeros(
            (self.max_size,), dtype=torch.int32, pin_memory=pin,
        )
        self._state_bytes = torch.zeros(
            (self.max_size, state_size), dtype=torch.uint8, pin_memory=pin,
        )
        self._legal_moves = torch.zeros(
            (self.max_size, self.max_legal, self.move_size),
            dtype=torch.uint8,
            pin_memory=pin,
        )
        self._visit_counts = torch.zeros(
            (self.max_size, self.max_legal), dtype=torch.float32, pin_memory=pin,
        )
        self._nlegal = torch.zeros(
            (self.max_size,), dtype=torch.int32, pin_memory=pin,
        )
        self._slot_targets = torch.zeros(
            (self.max_size, N_SLOTS), dtype=torch.float32, pin_memory=pin,
        )
        self._legal_masks = torch.zeros(
            (self.max_size, N_SLOTS), dtype=torch.bool, pin_memory=pin,
        )
        self._articulation_targets = torch.zeros(
            (self.max_size, MAX_BOARD), dtype=torch.float32, pin_memory=pin,
        )
        self._articulation_masks = torch.zeros(
            (self.max_size, MAX_BOARD), dtype=torch.float32, pin_memory=pin,
        )
        self._value_targets = torch.zeros(
            (self.max_size, 1), dtype=torch.float32, pin_memory=pin,
        )
        self._value_masks = torch.zeros(
            (self.max_size, 1), dtype=torch.float32, pin_memory=pin,
        )

    def _target_slices(self, start: int, count: int) -> list[tuple[int, int]]:
        if count <= 0:
            return []
        end = start + count
        if end <= self.max_size:
            return [(start, end)]
        return [(start, self.max_size), (0, end - self.max_size)]

    def add_examples(self, examples: list[PRSTrainingExampleV2]) -> None:
        if not examples:
            return
        if self.state_size is None:
            self._allocate_storage(int(examples[0].state_bytes.shape[0]))
        count = len(examples)
        offset = 0
        for dst_lo, dst_hi in self._target_slices(self._next_idx, count):
            self._write_chunk(examples[offset: offset + (dst_hi - dst_lo)], dst_lo)
            offset += dst_hi - dst_lo
        self._next_idx = (self._next_idx + count) % self.max_size
        self._size = min(self.max_size, self._size + count)

    def _write_chunk(self, examples: list[PRSTrainingExampleV2], dst_lo: int) -> None:
        if not examples:
            return
        B = len(examples)
        dst = slice(dst_lo, dst_lo + B)
        assert self._token_features is not None
        assert self._token_positions is not None
        assert self._token_types is not None
        assert self._num_board_tokens is not None
        assert self._global_features is not None
        assert self._seq_lengths is not None
        assert self._occupied_cells is not None
        assert self._num_occupied is not None
        assert self._state_bytes is not None
        assert self._legal_moves is not None
        assert self._visit_counts is not None
        assert self._nlegal is not None
        assert self._slot_targets is not None
        assert self._legal_masks is not None
        assert self._articulation_targets is not None
        assert self._articulation_masks is not None
        assert self._value_targets is not None
        assert self._value_masks is not None

        token_features = np.zeros((B, PRS_MAX_SEQ_LEN, _TOKEN_FEAT_DIM), dtype=np.float32)
        token_positions = np.full((B, PRS_MAX_SEQ_LEN), PRS_OFF_BOARD, dtype=np.int64)
        token_types = np.zeros((B, PRS_MAX_SEQ_LEN), dtype=np.int64)
        num_board = np.zeros((B,), dtype=np.int64)
        global_features = np.zeros((B, 6), dtype=np.float32)
        seq_lengths = np.zeros((B,), dtype=np.int64)
        occupied = np.full((B, MAX_BOARD), -1, dtype=np.int32)
        num_occupied = np.zeros((B,), dtype=np.int32)
        state_bytes = np.zeros((B, self.state_size), dtype=np.uint8)
        legal_moves = np.zeros((B, self.max_legal, self.move_size), dtype=np.uint8)
        visit_counts = np.zeros((B, self.max_legal), dtype=np.float32)
        nlegal = np.zeros((B,), dtype=np.int32)
        slot_targets = np.zeros((B, N_SLOTS), dtype=np.float32)
        legal_masks = np.zeros((B, N_SLOTS), dtype=bool)
        articulation_targets = np.zeros((B, MAX_BOARD), dtype=np.float32)
        articulation_masks = np.zeros((B, MAX_BOARD), dtype=np.float32)
        value_targets = np.zeros((B, 1), dtype=np.float32)
        value_masks = np.zeros((B, 1), dtype=np.float32)

        for i, ex in enumerate(examples):
            S = int(ex.seq_length)
            n = int(ex.nlegal)
            token_features[i, :S] = ex.token_features[:S]
            token_positions[i, :S] = ex.token_positions[:S]
            token_types[i, :S] = ex.token_types[:S]
            num_board[i] = ex.num_board_tokens
            global_features[i] = ex.global_features
            seq_lengths[i] = S
            no = int(ex.num_occupied)
            occupied[i, :no] = ex.occupied_cells[:no]
            num_occupied[i] = no
            state_bytes[i] = ex.state_bytes
            if n > 0:
                legal_moves[i, :n] = ex.legal_moves[:n]
                visit_counts[i, :n] = ex.visit_counts[:n]
            nlegal[i] = n
            slot_targets[i] = ex.slot_target
            legal_masks[i] = ex.legal_mask
            articulation_targets[i] = ex.articulation_target
            articulation_masks[i] = ex.articulation_mask
            value_targets[i, 0] = ex.value_target
            value_masks[i, 0] = float(ex.use_for_value)

        self._token_features[dst].copy_(torch.from_numpy(token_features))
        self._token_positions[dst].copy_(torch.from_numpy(token_positions))
        self._token_types[dst].copy_(torch.from_numpy(token_types))
        self._num_board_tokens[dst].copy_(torch.from_numpy(num_board))
        self._global_features[dst].copy_(torch.from_numpy(global_features))
        self._seq_lengths[dst].copy_(torch.from_numpy(seq_lengths))
        self._occupied_cells[dst].copy_(torch.from_numpy(occupied))
        self._num_occupied[dst].copy_(torch.from_numpy(num_occupied))
        self._state_bytes[dst].copy_(torch.from_numpy(state_bytes))
        self._legal_moves[dst].copy_(torch.from_numpy(legal_moves))
        self._visit_counts[dst].copy_(torch.from_numpy(visit_counts))
        self._nlegal[dst].copy_(torch.from_numpy(nlegal))
        self._slot_targets[dst].copy_(torch.from_numpy(slot_targets))
        self._legal_masks[dst].copy_(torch.from_numpy(legal_masks))
        self._articulation_targets[dst].copy_(torch.from_numpy(articulation_targets))
        self._articulation_masks[dst].copy_(torch.from_numpy(articulation_masks))
        self._value_targets[dst].copy_(torch.from_numpy(value_targets))
        self._value_masks[dst].copy_(torch.from_numpy(value_masks))

    def sample_batch(self, batch_size: int, augment_prob: float = 0.0) -> PRSTrainingBatchV2:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        idx = torch.randint(0, self._size, (batch_size,), dtype=torch.int64)
        k = random.randint(1, 5) if augment_prob > 0.0 and random.random() < augment_prob else 0
        if k == 0:
            return self._sample_noaug(idx)
        return self._sample_rotated(idx, k)

    def _sample_noaug(self, idx: torch.Tensor) -> PRSTrainingBatchV2:
        assert self._token_features is not None
        assert self._token_positions is not None
        assert self._token_types is not None
        assert self._num_board_tokens is not None
        assert self._global_features is not None
        assert self._seq_lengths is not None
        assert self._occupied_cells is not None
        assert self._num_occupied is not None
        assert self._state_bytes is not None
        assert self._legal_moves is not None
        assert self._nlegal is not None
        assert self._slot_targets is not None
        assert self._legal_masks is not None
        assert self._articulation_targets is not None
        assert self._articulation_masks is not None
        assert self._value_targets is not None
        assert self._value_masks is not None

        seq_lengths = self._seq_lengths.index_select(0, idx)
        max_seq = int(seq_lengths.max().item())
        attn = (
            torch.arange(max_seq, dtype=torch.int64).unsqueeze(0)
            < seq_lengths.unsqueeze(1)
        ).pin_memory()
        prs = PRSTokenBatch(
            token_features=self._token_features.index_select(0, idx)[:, :max_seq].pin_memory(),
            token_positions=self._token_positions.index_select(0, idx)[:, :max_seq].pin_memory(),
            token_types=self._token_types.index_select(0, idx)[:, :max_seq].pin_memory(),
            attention_mask=attn,
            num_board_tokens=self._num_board_tokens.index_select(0, idx).pin_memory(),
            global_features=self._global_features.index_select(0, idx).pin_memory(),
            seq_lengths=seq_lengths.pin_memory(),
            occupied_cells=self._occupied_cells.index_select(0, idx).pin_memory(),
            num_occupied=self._num_occupied.index_select(0, idx).pin_memory(),
        )
        return PRSTrainingBatchV2(
            prs_batch=prs,
            state_bytes=self._state_bytes.index_select(0, idx).pin_memory(),
            legal_moves_raw=self._legal_moves.index_select(0, idx).pin_memory(),
            nlegal_raw=self._nlegal.index_select(0, idx).pin_memory(),
            slot_targets=self._slot_targets.index_select(0, idx).pin_memory(),
            legal_masks=self._legal_masks.index_select(0, idx).pin_memory(),
            value_targets=self._value_targets.index_select(0, idx).pin_memory(),
            value_masks=self._value_masks.index_select(0, idx).pin_memory(),
            articulation_targets=self._articulation_targets.index_select(0, idx).pin_memory(),
            articulation_mask=self._articulation_masks.index_select(0, idx).pin_memory(),
            augmentation_k=0,
        )

    def _sample_rotated(self, idx: torch.Tensor, k: int) -> PRSTrainingBatchV2:
        assert self._state_bytes is not None
        assert self._legal_moves is not None
        assert self._visit_counts is not None
        assert self._nlegal is not None
        assert self._value_targets is not None
        assert self._value_masks is not None

        idx_cpu = idx.to(torch.int64)
        sb_orig = self._state_bytes.index_select(0, idx_cpu).numpy()
        nlegal_np = self._nlegal.index_select(0, idx_cpu).numpy()
        moves_pad = self._legal_moves.index_select(0, idx_cpu).numpy()
        visits_pad = self._visit_counts.index_select(0, idx_cpu).numpy()
        B = sb_orig.shape[0]
        max_L = max(int(nlegal_np.max()), 1)

        sb_rot = rotate_states_batch(sb_orig, k)
        moves_rot = rotate_moves_batch(moves_pad[:, :max_L], nlegal_np, k)

        states_gpu = torch.from_numpy(sb_rot).cuda()
        encoder = _get_encoder()
        prs_batch_gpu = encoder.encode_batch(states_gpu, B)
        nlegal_gpu = torch.from_numpy(nlegal_np).cuda()
        moves_gpu = torch.from_numpy(moves_rot).cuda()
        kernel_out = self.ext.prs_v2_classify_batch(
            states_gpu, moves_gpu, nlegal_gpu, B, max_L,
        )
        slot_of_legal = kernel_out[8].cpu().numpy()

        slot_t = np.zeros((B, N_SLOTS), dtype=np.float32)
        lmask = np.zeros((B, N_SLOTS), dtype=bool)
        for i in range(B):
            n = int(nlegal_np[i])
            if n == 0:
                continue
            for j in range(n):
                s = int(slot_of_legal[i, j])
                if s < 0:
                    continue
                slot_t[i, s] += float(visits_pad[i, j])
                lmask[i, s] = True
            ssum = slot_t[i].sum()
            if ssum > 0:
                slot_t[i] /= ssum
            else:
                nleg = int(lmask[i].sum())
                if nleg > 0:
                    slot_t[i, lmask[i]] = 1.0 / nleg

        art_t = np.zeros((B, MAX_BOARD), dtype=np.float32)
        art_m = np.zeros((B, MAX_BOARD), dtype=np.float32)
        for i in range(B):
            t, m = compute_articulation_target(sb_rot[i], MAX_BOARD)
            art_t[i] = t
            art_m[i] = m

        prs = PRSTokenBatch(
            token_features=prs_batch_gpu.token_features.cpu().pin_memory(),
            token_positions=prs_batch_gpu.token_positions.cpu().pin_memory(),
            token_types=prs_batch_gpu.token_types.cpu().pin_memory(),
            attention_mask=prs_batch_gpu.attention_mask.cpu().pin_memory(),
            num_board_tokens=prs_batch_gpu.num_board_tokens.cpu().pin_memory(),
            global_features=prs_batch_gpu.global_features.cpu().pin_memory(),
            seq_lengths=prs_batch_gpu.seq_lengths.cpu().pin_memory(),
            occupied_cells=prs_batch_gpu.occupied_cells.cpu().pin_memory(),
            num_occupied=prs_batch_gpu.num_occupied.cpu().pin_memory(),
        )
        return PRSTrainingBatchV2(
            prs_batch=prs,
            state_bytes=torch.from_numpy(sb_rot).pin_memory(),
            legal_moves_raw=torch.from_numpy(moves_rot).pin_memory(),
            nlegal_raw=torch.from_numpy(nlegal_np.astype(np.int32, copy=False)).pin_memory(),
            slot_targets=torch.from_numpy(slot_t).pin_memory(),
            legal_masks=torch.from_numpy(lmask).pin_memory(),
            value_targets=self._value_targets.index_select(0, idx_cpu).pin_memory(),
            value_masks=self._value_masks.index_select(0, idx_cpu).pin_memory(),
            articulation_targets=torch.from_numpy(art_t).pin_memory(),
            articulation_mask=torch.from_numpy(art_m).pin_memory(),
            augmentation_k=k,
        )
