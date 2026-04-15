"""
GPU encoder adapter for the Piece-Relative Space (PRS) model.

Differences from the existing GPUTransformerEncoder:
  1. Token positions use ABSOLUTE 23×23 cell indices (0-528) instead of
     centroid-centred 17×17 grid positions.  This eliminates all clipping.
  2. Off-board position index = BOARD_CELLS = 529.
  3. Also outputs the sorted occupied_cells array needed by the action-space
     encoder to convert move bytes → piece-relative action indices.

Everything else (token features, types, attention mask, global features)
is identical to the existing encoder.
"""

from __future__ import annotations

import numpy as np
import torch

import hive_gpu
from hive_transformer.token_types import (
    HiveTokenBatch,
    TOKEN_TYPE_CLS,
    TOKEN_TYPE_BOARD,
    TOKEN_TYPE_HAND,
)
from hive_prs.action_space import BOARD_CELLS  # 529

# Off-board sentinel for the PRS model
PRS_OFF_BOARD = BOARD_CELLS  # 529  (position table has 530 entries)

# Max sequence length (same as existing model)
PRS_MAX_SEQ_LEN = 60  # 28 board + 28 hand + 1 CLS + 3 margin


class PRSTokenBatch:
    """Like HiveTokenBatch but also carries occupied_cells for action encoding."""

    def __init__(
        self,
        token_features:    torch.Tensor,   # (B, S, 25) float32
        token_positions:   torch.Tensor,   # (B, S) int64  — 0-528 board, 529 off-board
        token_types:       torch.Tensor,   # (B, S) int64
        attention_mask:    torch.Tensor,   # (B, S) bool
        num_board_tokens:  torch.Tensor,   # (B,) int64
        global_features:   torch.Tensor,   # (B, 6) float32
        seq_lengths:       torch.Tensor,   # (B,) int64
        occupied_cells:    torch.Tensor,   # (B, MAX_BOARD) int32  — cell indices, -1 = padding
        num_occupied:      torch.Tensor,   # (B,) int32
    ):
        self.token_features   = token_features
        self.token_positions  = token_positions
        self.token_types      = token_types
        self.attention_mask   = attention_mask
        self.num_board_tokens = num_board_tokens
        self.global_features  = global_features
        self.seq_lengths      = seq_lengths
        self.occupied_cells   = occupied_cells
        self.num_occupied     = num_occupied

    def to(self, device, non_blocking: bool = False) -> PRSTokenBatch:
        return PRSTokenBatch(
            token_features   = self.token_features.to(device, non_blocking=non_blocking),
            token_positions  = self.token_positions.to(device, non_blocking=non_blocking),
            token_types      = self.token_types.to(device, non_blocking=non_blocking),
            attention_mask   = self.attention_mask.to(device, non_blocking=non_blocking),
            num_board_tokens = self.num_board_tokens.to(device, non_blocking=non_blocking),
            global_features  = self.global_features.to(device, non_blocking=non_blocking),
            seq_lengths      = self.seq_lengths.to(device, non_blocking=non_blocking),
            occupied_cells   = self.occupied_cells.to(device, non_blocking=non_blocking),
            num_occupied     = self.num_occupied.to(device, non_blocking=non_blocking),
        )

    def slice_batch(self, start: int, end: int) -> PRSTokenBatch:
        return PRSTokenBatch(
            token_features   = self.token_features[start:end],
            token_positions  = self.token_positions[start:end],
            token_types      = self.token_types[start:end],
            attention_mask   = self.attention_mask[start:end],
            num_board_tokens = self.num_board_tokens[start:end],
            global_features  = self.global_features[start:end],
            seq_lengths      = self.seq_lengths[start:end],
            occupied_cells   = self.occupied_cells[start:end],
            num_occupied     = self.num_occupied[start:end],
        )


class PRSEncoder:
    """
    Encodes GPU HiveStates into PRSTokenBatch for the PRS transformer.

    Uses the same CUDA kernel as GPUTransformerEncoder but remaps positions
    to absolute 23×23 cell indices.
    """

    _BOARD_SIZE = 23  # internal game board width

    def __init__(self):
        self.ext = hive_gpu.load_extension()

    def encode_batch(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> PRSTokenBatch:
        """Encode a batch of GPU HiveStates into a PRSTokenBatch."""
        B = batch_size
        raw = self.ext.encode_states_batch(states_tensor, B)
        (
            node_features, node_grid_pos, node_piece_types,
            global_features, num_nodes, num_board_nodes,
            _edge_index, _edge_features, _num_edges,
        ) = raw

        device = node_features.device
        max_nodes  = num_nodes.max().item()
        max_seq    = max_nodes + 1  # +1 for CLS

        seq_lens = (num_nodes + 1).to(torch.int64)
        nb       = num_board_nodes.to(torch.int64)

        pos_idx  = torch.arange(max_seq,    device=device).unsqueeze(0)
        node_idx = torch.arange(max_nodes,  device=device).unsqueeze(0)
        nb_col   = nb.unsqueeze(1)
        sl_col   = seq_lens.unsqueeze(1)

        # ── Token features ──
        token_features = torch.zeros((B, max_seq, 25), dtype=torch.float32, device=device)
        token_features[:, 1:1 + max_nodes] = node_features[:, :max_nodes]

        # ── Attention mask ──
        attention_mask = pos_idx < sl_col

        # ── Token types ──
        is_board = (pos_idx >= 1) & (pos_idx <= nb_col)
        is_hand  = (pos_idx >  nb_col) & (pos_idx < sl_col)
        token_types = torch.zeros((B, max_seq), dtype=torch.int64, device=device)
        token_types[is_board] = TOKEN_TYPE_BOARD
        token_types[is_hand]  = TOKEN_TYPE_HAND

        # ── Token positions: ABSOLUTE 23×23 cell index ──
        # node_grid_pos contains (row_enc, col_enc) in the 17×17 centroid-centred
        # grid: enc = abs - centroid + 8, where centroid is the per-game piece
        # centroid in absolute 23×23 space.
        #
        # compute_centroids_batch returns (B, 2) int32: [col_offset, row_offset]
        # where col_offset = Cc - 11, row_offset = Cr - 11 (offset from board centre).
        # Therefore:
        #   abs_row = enc_row + centroid[1] + 3   (= enc_row - 8 + Cr)
        #   abs_col = enc_col + centroid[0] + 3   (= enc_col - 8 + Cc)

        centroids = self.ext.compute_centroids_batch(states_tensor, B)  # (B, 2) int32 on GPU
        # centroid[b, 0] = col_offset, centroid[b, 1] = row_offset
        cent_col = centroids[:, 0].to(torch.int64).view(B, 1)  # (B, 1)
        cent_row = centroids[:, 1].to(torch.int64).view(B, 1)  # (B, 1)

        token_positions = torch.full(
            (B, max_seq), PRS_OFF_BOARD, dtype=torch.int64, device=device,
        )

        # node_grid_pos: (B, max_nodes, 2) — (row_enc, col_enc) in 17×17
        board_rows = node_grid_pos[:, :max_nodes, 0].to(torch.int64)  # (B, max_nodes)
        board_cols = node_grid_pos[:, :max_nodes, 1].to(torch.int64)

        # Mask non-board nodes
        valid_board = node_idx < nb_col  # (B, max_nodes) bool

        ENC_TO_ABS = 3   # = BOARD_HALF(11) - ENC_HALF(8)

        abs_rows = board_rows + cent_row + ENC_TO_ABS  # (B, max_nodes)
        abs_cols = board_cols + cent_col + ENC_TO_ABS

        # Clamp to [0, 22] to handle any edge cases
        abs_rows = abs_rows.clamp(0, 22)
        abs_cols = abs_cols.clamp(0, 22)

        abs_cell = abs_rows * self._BOARD_SIZE + abs_cols  # (B, max_nodes)

        # Replace off-board / hand / padding with PRS_OFF_BOARD
        abs_cell = torch.where(valid_board, abs_cell,
                               torch.full_like(abs_cell, PRS_OFF_BOARD))

        token_positions[:, 1:1 + max_nodes] = abs_cell

        # ── Occupied cells (for action encoding) ──
        # Board tokens (token index 1..nb_i) correspond to occupied cells in
        # ascending cell-index order (CUDA emits them sorted).
        from hive_prs.action_space import MAX_BOARD
        occupied_cells = torch.full((B, MAX_BOARD), -1, dtype=torch.int32, device=device)
        num_occupied   = nb.to(torch.int32)

        # abs_cell[:, :max_nodes] already has sorted board cell indices for board tokens;
        # hand tokens have PRS_OFF_BOARD — we take only the first nb_i per game.
        max_nb = int(nb.max().item())
        if max_nb > 0:
            fill_len = min(max_nb, MAX_BOARD)
            occupied_cells[:, :fill_len] = abs_cell[:, :fill_len].to(torch.int32)

        return PRSTokenBatch(
            token_features   = token_features,
            token_positions  = token_positions,
            token_types      = token_types,
            attention_mask   = attention_mask,
            num_board_tokens = nb,
            global_features  = global_features,
            seq_lengths      = seq_lens,
            occupied_cells   = occupied_cells,
            num_occupied     = num_occupied,
        )
