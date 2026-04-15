from __future__ import annotations

from dataclasses import dataclass

import torch

import hive_gpu
from hive_gpu.gpu_encoder import GPUTransformerEncoder
from hive_transformer.token_types import HiveTokenBatch

# Engine coordinate system (23x23 board)
ENGINE_BOARD_SIZE = 23
ENGINE_CELLS = ENGINE_BOARD_SIZE ** 2  # 529
ENGINE_OFF_BOARD = ENGINE_CELLS        # 529, sentinel for hand pieces
ENGINE_MAX_POSITIONS = ENGINE_CELLS + 1  # 530


@dataclass
class MoveFeatures:
    """Compact features extracted from engine move bytes."""

    action_to_root: torch.Tensor   # (N,) int64
    num_actions: torch.Tensor      # (B,) int64
    move_types: torch.Tensor       # (N,) int64, 0=place 1=move
    piece_types: torch.Tensor      # (N,) int64, 0-7
    from_positions: torch.Tensor   # (N,) int64, engine cell 0-528 or 529
    to_positions: torch.Tensor     # (N,) int64, engine cell 0-528

    def to(self, device: torch.device) -> MoveFeatures:
        return MoveFeatures(
            action_to_root=self.action_to_root.to(device),
            num_actions=self.num_actions.to(device),
            move_types=self.move_types.to(device),
            piece_types=self.piece_types.to(device),
            from_positions=self.from_positions.to(device),
            to_positions=self.to_positions.to(device),
        )


@dataclass
class MoveConditionedBatch:
    """Root states with encoded successor states and compact move features."""

    root_batch: HiveTokenBatch
    action_batch: HiveTokenBatch
    move_features: MoveFeatures
    legal_moves: torch.Tensor  # (B, MAX_LEGAL, move_size) uint8

    def to(self, device: torch.device) -> MoveConditionedBatch:
        return MoveConditionedBatch(
            root_batch=self.root_batch.to(device),
            action_batch=self.action_batch.to(device),
            move_features=self.move_features.to(device),
            legal_moves=self.legal_moves.to(device),
        )


def parse_move_features(
    legal_moves: torch.Tensor,
    num_legal: torch.Tensor,
    device: torch.device,
) -> MoveFeatures:
    """Extract compact features from move bytes without encoding any states."""
    num_actions = num_legal.to(torch.int64)
    # Use bool check to avoid .item() sync for the zero case
    if num_actions.shape[0] == 0 or not num_actions.any():
        z = torch.zeros((0,), dtype=torch.int64, device=device)
        return MoveFeatures(
            action_to_root=z,
            num_actions=num_actions,
            move_types=z,
            piece_types=z,
            from_positions=z,
            to_positions=z,
        )

    max_legal = legal_moves.shape[1]
    slot_idx = torch.arange(max_legal, device=device, dtype=torch.int64).unsqueeze(0)
    valid = slot_idx < num_actions.unsqueeze(1)
    # Use boolean indexing instead of nonzero to avoid sync
    action_to_root = torch.arange(
        num_actions.shape[0], device=device, dtype=torch.int64
    ).unsqueeze(1).expand_as(valid)[valid]
    move_indices = slot_idx.expand_as(valid)[valid]

    moves = legal_moves[action_to_root, move_indices].to(torch.int64)
    raw_move_types = moves[:, 0]
    move_types = raw_move_types.clamp(0, 2)
    piece_types = ((moves[:, 1] & 0x0F) - 1).clamp(0, 7)
    from_cells = moves[:, 2] | (moves[:, 3] << 8)
    to_cells = moves[:, 4] | (moves[:, 5] << 8)

    from_positions = torch.where(
        move_types == 1,
        from_cells.clamp(0, ENGINE_CELLS - 1),
        torch.full_like(from_cells, ENGINE_OFF_BOARD),
    )
    to_positions = torch.where(
        move_types == 2,
        torch.full_like(to_cells, ENGINE_OFF_BOARD),
        to_cells.clamp(0, ENGINE_CELLS - 1),
    )

    return MoveFeatures(
        action_to_root=action_to_root,
        num_actions=num_actions,
        move_types=move_types,
        piece_types=piece_types,
        from_positions=from_positions,
        to_positions=to_positions,
    )


def build_move_conditioned_batch(
    states: torch.Tensor,
    batch_size: int,
    *,
    encoder: GPUTransformerEncoder | None = None,
    ext=None,
) -> MoveConditionedBatch:
    """Encode root states plus all legal successor states (for training)."""
    if ext is None:
        ext = hive_gpu.load_extension()
    if encoder is None:
        encoder = GPUTransformerEncoder()

    root_batch = encoder.encode_batch(states, batch_size)
    legal_moves, num_legal = ext.generate_legal_moves_batch(states, batch_size)
    move_features = parse_move_features(legal_moves, num_legal, states.device)

    # Single .item() sync — needed for batch_size arg to CUDA kernels
    total_actions = int(move_features.action_to_root.shape[0])

    if total_actions == 0:
        action_batch = encoder.encode_batch(states[:0].clone(), 0)
        return MoveConditionedBatch(
            root_batch=root_batch,
            action_batch=action_batch,
            move_features=move_features,
            legal_moves=legal_moves,
        )

    child_states = states[move_features.action_to_root].clone()
    # Reuse action_to_root and pre-computed indices from parse_move_features
    max_legal = legal_moves.shape[1]
    slot_idx = torch.arange(max_legal, device=states.device, dtype=torch.int64).unsqueeze(0)
    valid = slot_idx < move_features.num_actions.unsqueeze(1)
    move_indices = slot_idx.expand_as(valid)[valid]
    moves = legal_moves[move_features.action_to_root, move_indices]
    ext.apply_moves_batch(child_states, moves, total_actions)
    action_batch = encoder.encode_batch(child_states, total_actions)

    return MoveConditionedBatch(
        root_batch=root_batch,
        action_batch=action_batch,
        move_features=move_features,
        legal_moves=legal_moves,
    )


def build_successor_batch(
    states: torch.Tensor,
    legal_moves: torch.Tensor,
    game_indices: torch.Tensor,
    move_col_indices: torch.Tensor,
    *,
    encoder: GPUTransformerEncoder | None = None,
    ext=None,
) -> HiveTokenBatch:
    """Encode only selected successor states (for self-play top-k)."""
    if ext is None:
        ext = hive_gpu.load_extension()
    if encoder is None:
        encoder = GPUTransformerEncoder()

    n = game_indices.shape[0]
    if n == 0:
        return encoder.encode_batch(states[:0].clone(), 0)

    child_states = states[game_indices].clone()
    moves = legal_moves[game_indices, move_col_indices]
    ext.apply_moves_batch(child_states, moves, n)
    return encoder.encode_batch(child_states, n)


def flat_to_padded(
    flat_values: torch.Tensor,
    counts: torch.Tensor,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Convert a flat ragged vector into a padded [B, max_count] tensor."""
    B = counts.shape[0]
    if B == 0:
        return flat_values.new_full((0, 0), pad_value)
    max_count = int(counts.max().item())
    if max_count == 0 or flat_values.numel() == 0:
        return flat_values.new_full((B, max_count), pad_value)

    # Build (B, max_count) mask and scatter flat values via boolean indexing
    col_idx = torch.arange(max_count, device=counts.device, dtype=torch.int64).unsqueeze(0)
    mask = col_idx < counts.unsqueeze(1)  # (B, max_count)
    out = flat_values.new_full((B, max_count), pad_value)
    out[mask] = flat_values
    return out


def padded_to_flat(
    padded_values: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Flatten a padded [B, max_count] tensor back into ragged form."""
    if counts.numel() == 0 or padded_values.numel() == 0:
        return padded_values.new_zeros((0,), dtype=padded_values.dtype)
    max_count = padded_values.shape[1]
    col_idx = torch.arange(max_count, device=counts.device, dtype=torch.int64).unsqueeze(0)
    mask = col_idx < counts.unsqueeze(1)
    return padded_values[mask]
