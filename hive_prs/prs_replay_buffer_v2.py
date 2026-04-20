"""Replay buffer for PRS v2 training examples (813-slot structured targets).

Stores the raw HiveState bytes + legal moves + MCTS visit counts alongside
the token encoding so the trainer can either:

  a) use the cached tokens/slot_target/legal_mask (k=0, fast path), or
  b) pick a C6 rotation k∈{1..5}, rotate state + moves, re-encode tokens
     and rebuild slot_target/legal_mask on the fly (training-time hex-grid
     symmetry augmentation).

Slot indices are NOT rotation-invariant (they depend on ascending cell-id
rank within the state's legal move/place-cell set), so rotation must go
through: rotate state → rotate moves → re-run SlotMapper → rebuild target.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

import hive_gpu
from hive_prs.prs_encoder import PRSEncoder, PRSTokenBatch, PRS_OFF_BOARD
from hive_prs.slot_map import N_SLOTS, map_legal_moves
from hive_prs.action_space import MAX_BOARD
from hive_prs.prs_c6_augment import rotate_states_batch, rotate_moves_batch


class PRSTrainingExampleV2(NamedTuple):
    """One self-play position, v2 (structured 813-slot policy target)."""

    # Token encoding (cached for k=0 fast path; invalidated on rotation)
    token_features:   np.ndarray   # (S, 25) float32
    token_positions:  np.ndarray   # (S,) int32
    token_types:      np.ndarray   # (S,) int8
    num_board_tokens: int
    global_features:  np.ndarray   # (6,) float32
    seq_length:       int

    # Raw state bytes + legal moves + visit counts (for rotation augmentation)
    state_bytes:      np.ndarray   # (SIZEOF_HIVE_STATE,) uint8
    legal_moves:      np.ndarray   # (nlegal, SIZEOF_GPU_MOVE) uint8
    visit_counts:     np.ndarray   # (nlegal,) float32 — raw MCTS visit counts
    nlegal:           int

    # Also kept for compatibility with anything that consumes PRSTokenBatch
    occupied_cells:   np.ndarray
    num_occupied:     int

    # Pre-baked targets (used on k=0 fast path — saves SlotMapper calls)
    slot_target:      np.ndarray   # (N_SLOTS,) float32 — MCTS visit distribution
    legal_mask:       np.ndarray   # (N_SLOTS,) bool    — True for legal slots

    value_target:     float


@dataclass
class PRSTrainingBatchV2:
    """Batched data for one PRS v2 training step."""

    prs_batch:       PRSTokenBatch
    state_bytes:     np.ndarray       # (B, SIZEOF_HIVE_STATE) uint8 — CPU (head bridge runs CPU-side)
    slot_targets:    torch.Tensor     # (B, N_SLOTS) float32
    legal_masks:     torch.Tensor     # (B, N_SLOTS) bool
    value_targets:   torch.Tensor     # (B, 1) float32
    augmentation_k:  int = 0          # rotation applied to this batch (0..5)

    def to(self, device, non_blocking: bool = False) -> "PRSTrainingBatchV2":
        return PRSTrainingBatchV2(
            prs_batch      = self.prs_batch.to(device, non_blocking=non_blocking),
            state_bytes    = self.state_bytes,   # stays CPU for the bridge
            slot_targets   = self.slot_targets.to(device, non_blocking=non_blocking),
            legal_masks    = self.legal_masks.to(device, non_blocking=non_blocking),
            value_targets  = self.value_targets.to(device, non_blocking=non_blocking),
            augmentation_k = self.augmentation_k,
        )


class PRSReplayBufferV2:
    """Circular replay buffer for PRSTrainingExampleV2 instances.

    `sample_batch(batch_size, augment_prob)`: with probability `augment_prob`,
    picks k∈{1..5} uniformly and applies a C6 rotation to the whole batch.
    """

    def __init__(self, max_size: int = 100_000) -> None:
        self.buffer: deque[PRSTrainingExampleV2] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_examples(self, examples: list[PRSTrainingExampleV2]) -> None:
        self.buffer.extend(examples)

    def sample_batch(
        self, batch_size: int, augment_prob: float = 0.0,
    ) -> PRSTrainingBatchV2:
        buf = list(self.buffer)
        indices = np.random.randint(0, len(buf), size=batch_size)
        samples = [buf[i] for i in indices]

        if augment_prob > 0.0 and random.random() < augment_prob:
            k = random.randint(1, 5)
        else:
            k = 0

        if k == 0:
            return _collate_noaug(samples)
        return _collate_rotated(samples, k)


# ── k=0 fast path ─────────────────────────────────────────────────────

def _collate_noaug(samples: list[PRSTrainingExampleV2]) -> PRSTrainingBatchV2:
    B = len(samples)
    max_seq = max(s.seq_length for s in samples)
    state_size = samples[0].state_bytes.shape[0]

    feat    = np.zeros((B, max_seq, 25),  dtype=np.float32)
    pos     = np.full( (B, max_seq),      PRS_OFF_BOARD, dtype=np.int64)
    types   = np.zeros((B, max_seq),      dtype=np.int64)
    mask    = np.zeros((B, max_seq),      dtype=bool)
    gf      = np.zeros((B, 6),            dtype=np.float32)
    nb      = np.zeros(B,                 dtype=np.int64)
    sl      = np.zeros(B,                 dtype=np.int64)
    occ     = np.full( (B, MAX_BOARD),    -1, dtype=np.int32)
    nocc    = np.zeros(B,                 dtype=np.int32)
    sb      = np.zeros((B, state_size),   dtype=np.uint8)
    slot_t  = np.zeros((B, N_SLOTS),      dtype=np.float32)
    lmask   = np.zeros((B, N_SLOTS),      dtype=bool)
    value   = np.zeros((B, 1),            dtype=np.float32)

    for i, s in enumerate(samples):
        S = s.seq_length
        feat[i, :S]   = s.token_features[:S]
        pos[i,  :S]   = s.token_positions[:S]
        types[i, :S]  = s.token_types[:S]
        mask[i, :S]   = True
        gf[i]         = s.global_features
        nb[i]         = s.num_board_tokens
        sl[i]         = S
        no = s.num_occupied
        occ[i, :no]   = s.occupied_cells[:no]
        nocc[i]       = no
        sb[i]         = s.state_bytes
        slot_t[i]     = s.slot_target
        lmask[i]      = s.legal_mask
        value[i, 0]   = s.value_target

    prs = PRSTokenBatch(
        token_features   = torch.from_numpy(feat).pin_memory(),
        token_positions  = torch.from_numpy(pos).pin_memory(),
        token_types      = torch.from_numpy(types).pin_memory(),
        attention_mask   = torch.from_numpy(mask).pin_memory(),
        num_board_tokens = torch.from_numpy(nb).pin_memory(),
        global_features  = torch.from_numpy(gf).pin_memory(),
        seq_lengths      = torch.from_numpy(sl).pin_memory(),
        occupied_cells   = torch.from_numpy(occ).pin_memory(),
        num_occupied     = torch.from_numpy(nocc).pin_memory(),
    )

    return PRSTrainingBatchV2(
        prs_batch      = prs,
        state_bytes    = sb,
        slot_targets   = torch.from_numpy(slot_t).pin_memory(),
        legal_masks    = torch.from_numpy(lmask).pin_memory(),
        value_targets  = torch.from_numpy(value).pin_memory(),
        augmentation_k = 0,
    )


# ── k>0 rotated path ──────────────────────────────────────────────────

# Cached encoder instance (cheap — just holds the extension handle)
_encoder: PRSEncoder | None = None


def _get_encoder() -> PRSEncoder:
    global _encoder
    if _encoder is None:
        _encoder = PRSEncoder()
    return _encoder


def _collate_rotated(
    samples: list[PRSTrainingExampleV2], k: int,
) -> PRSTrainingBatchV2:
    """C6-augmented collate. Rotates state+moves, re-encodes tokens, and
    rebuilds slot_target/legal_mask via SlotMapper."""
    assert 1 <= k <= 5
    ext = hive_gpu.load_extension()
    B = len(samples)
    state_size = samples[0].state_bytes.shape[0]
    move_sz    = samples[0].legal_moves.shape[1]

    # Stack raw states + moves
    sb_orig = np.stack([s.state_bytes for s in samples], axis=0)  # (B, S)
    nlegal_np = np.array([s.nlegal for s in samples], dtype=np.int32)

    # Variable-length moves: pad to a common max_L for rotate_moves_batch
    max_L = int(nlegal_np.max()) if nlegal_np.size > 0 else 0
    max_L = max(max_L, 1)
    moves_pad = np.zeros((B, max_L, move_sz), dtype=np.uint8)
    visits_pad = np.zeros((B, max_L), dtype=np.float32)
    for i, s in enumerate(samples):
        n = s.nlegal
        if n > 0:
            moves_pad[i, :n] = s.legal_moves
            visits_pad[i, :n] = s.visit_counts

    # Rotate state + moves
    sb_rot = rotate_states_batch(sb_orig, k)                     # (B, S) uint8
    moves_rot = rotate_moves_batch(moves_pad, nlegal_np, k)      # (B, max_L, sz) uint8

    # Upload rotated states to GPU and re-encode tokens
    states_gpu = torch.from_numpy(sb_rot).cuda()
    encoder = _get_encoder()
    prs_batch_gpu = encoder.encode_batch(states_gpu, B)

    # Use the CUDA classify kernel to map rotated legal moves → slot ids in
    # the rotated state's slot space. Much faster than 256 × SlotMapper.
    nlegal_gpu = torch.from_numpy(nlegal_np).cuda()
    moves_gpu  = torch.from_numpy(moves_rot).cuda()
    kernel_out = ext.prs_v2_classify_batch(
        states_gpu, moves_gpu, nlegal_gpu, B, max_L,
    )
    # kernel_out[8] is slot_of_legal: (B, max_L) int32 (see prs_v2_bridge)
    slot_of_legal = kernel_out[8].cpu().numpy()                  # (B, max_L)

    # Build fresh slot_target / legal_mask from rotated visits
    slot_t = np.zeros((B, N_SLOTS), dtype=np.float32)
    lmask  = np.zeros((B, N_SLOTS), dtype=bool)
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

    # Bring tokens back to CPU (the trainer's .to(device) path expects CPU
    # tensors that it then moves to device — mirrors the k=0 path).
    prs_batch = PRSTokenBatch(
        token_features   = prs_batch_gpu.token_features.cpu(),
        token_positions  = prs_batch_gpu.token_positions.cpu(),
        token_types      = prs_batch_gpu.token_types.cpu(),
        attention_mask   = prs_batch_gpu.attention_mask.cpu(),
        num_board_tokens = prs_batch_gpu.num_board_tokens.cpu(),
        global_features  = prs_batch_gpu.global_features.cpu(),
        seq_lengths      = prs_batch_gpu.seq_lengths.cpu(),
        occupied_cells   = prs_batch_gpu.occupied_cells.cpu(),
        num_occupied     = prs_batch_gpu.num_occupied.cpu(),
    )

    value = np.array([[s.value_target] for s in samples], dtype=np.float32)

    return PRSTrainingBatchV2(
        prs_batch      = prs_batch,
        state_bytes    = sb_rot,
        slot_targets   = torch.from_numpy(slot_t),
        legal_masks    = torch.from_numpy(lmask),
        value_targets  = torch.from_numpy(value),
        augmentation_k = k,
    )
