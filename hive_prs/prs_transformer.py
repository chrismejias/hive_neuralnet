"""
Piece-Relative Space (PRS) Transformer for Hive.

Architecture:
    PRSTokenBatch → token/position/type embeddings →
    TransformerEncoder (self-attention × N) →
        ├── PRSPolicyHead (bilinear over board tokens) → logits (6,841)
        └── ValueHead (CLS token → FC → tanh)         → scalar in [-1, 1]

Policy head produces piece-relative logits:
    MOVE    : score(actor_tok, ref_tok, direction)  using factored bilinear
    PLACE   : score(piece_type, ref_tok, direction) using piece-type embedding
    FIRST   : score(piece_type)                     using CLS → FC
    PASS    : score()                               using CLS → FC

Position embeddings index over the full 23×23 board (529 cells) plus one
off-board sentinel (530 entries total), so no clipping occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_prs.action_space import (
    ACTION_SPACE_SIZE,
    MAX_BOARD,
    NUM_TYPES,
    DIRECTIONS,
    BOARD_CELLS,
    MOVE_ACTIONS,
    PLACE_ACTIONS,
    FIRST_PLACE_ACTIONS,
)
from hive_prs.prs_encoder import PRSTokenBatch, PRS_OFF_BOARD


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class PRSConfig:
    """Configuration for the PRS Transformer."""

    d_model:         int   = 128
    num_heads:       int   = 8
    num_layers:      int   = 6
    dim_feedforward: int   = 512
    dropout:         float = 0.1

    # Token input dims (match CUDA kernel output)
    token_feat_dim:  int   = 25
    global_feat_dim: int   = 6
    num_token_types: int   = 3   # CLS=0, board=1, hand=2

    # Position table: 529 board cells + 1 off-board sentinel
    max_positions:   int   = BOARD_CELLS + 1   # 530

    # PRS v3 relative attention bias. The clip value buckets offsets outside
    # [-clip, clip] in row/col space, plus one bucket for off-board/hand/pad.
    relative_position_clip: int = 8

    # Bilinear key dimension for the policy head
    d_key:           int   = 64

    # Action space (from action_space constants)
    action_space_size: int = ACTION_SPACE_SIZE   # 6,841

    @classmethod
    def small(cls) -> PRSConfig:
        return cls(d_model=128, num_heads=8, num_layers=6, dim_feedforward=512)

    @classmethod
    def large(cls) -> PRSConfig:
        return cls(d_model=256, num_heads=8, num_layers=12, dim_feedforward=1024)


# ── Bilinear Policy Head ───────────────────────────────────────────────────────


class PRSPolicyHead(nn.Module):
    """
    Bilinear policy head for piece-relative action space.

    Factored trilinear scoring:
        MOVE  score[actor, ref, dir] = (W_actor @ h_actor) · ((W_ref @ h_ref) ⊙ dir_emb[dir])
        PLACE score[type, ref, dir]  = type_emb[type]      · ((W_ref @ h_ref) ⊙ dir_emb[dir])
        FIRST score[type]            = fc_first(cls)[type]
        PASS  score                  = fc_pass(cls)

    Memory: O(B · MAX_BOARD² · d_key) ≈ small for MAX_BOARD=28, d_key=64.
    """

    def __init__(self, config: PRSConfig) -> None:
        super().__init__()
        d = config.d_model
        dk = config.d_key
        self.max_board  = MAX_BOARD
        self.num_types  = NUM_TYPES
        self.directions = DIRECTIONS

        # Actor and reference projections for MOVE
        self.actor_proj = nn.Linear(d, dk, bias=False)
        self.ref_proj   = nn.Linear(d, dk, bias=False)

        # Direction embeddings (7: 6 hex neighbours + 1 stack)
        self.dir_emb = nn.Embedding(DIRECTIONS, dk)

        # Piece-type embeddings for PLACE (actor role)
        self.type_emb = nn.Embedding(NUM_TYPES, dk)

        # First-placement head: CLS → NUM_TYPES logits
        self.fc_first = nn.Linear(d, NUM_TYPES)

        # Pass head: CLS → 1 logit
        self.fc_pass  = nn.Linear(d, 1)

    def forward(
        self,
        h: torch.Tensor,           # (B, S, d_model)  transformer output
        cls_h: torch.Tensor,       # (B, d_model)      CLS token embedding
    ) -> torch.Tensor:
        """Return policy logits (B, ACTION_SPACE_SIZE = 6841)."""
        B = h.size(0)
        dk = self.dir_emb.weight.size(1)
        MB = self.max_board
        device = h.device

        # Board token embeddings: tokens 1..MAX_BOARD  (pad positions are zero)
        # Slice up to MB tokens; pad with zeros if sequence is shorter.
        raw_board = h[:, 1:1 + MB, :]       # (B, min(MB, S-1), d)
        if raw_board.size(1) < MB:
            pad = torch.zeros(B, MB - raw_board.size(1), raw_board.size(2),
                              dtype=raw_board.dtype, device=device)
            board_h = torch.cat([raw_board, pad], dim=1)   # (B, MB, d)
        else:
            board_h = raw_board               # (B, MB, d)

        # ── Project board tokens ──
        actor_keys = self.actor_proj(board_h)   # (B, MB, dk)
        ref_keys   = self.ref_proj(board_h)     # (B, MB, dk)

        dir_all   = self.dir_emb.weight          # (7, dk)
        dir_place = dir_all[:DIRECTIONS - 1]     # (6, dk)  — no stacking for placements

        # ── MOVE scores: (B, MB, MB, 7) → (B, MB*MB*7) ──
        # actor_dir[b,i,k,:] = actor_keys[b,i,:] * dir_all[k,:]  → (B, MB, 7, dk)
        actor_dir = actor_keys.unsqueeze(2) * dir_all.unsqueeze(0).unsqueeze(0)
        #   (B, MB, 1, dk) * (1, 1, 7, dk) → (B, MB, 7, dk)

        # score[b, i, k, j] = actor_dir[b,i,k,:] · ref_keys[b,j,:]
        actor_dir_flat = actor_dir.view(B, MB * self.directions, dk)  # (B, MB*7, dk)
        move_scores = torch.bmm(actor_dir_flat, ref_keys.transpose(1, 2))
        #   (B, MB*7, dk) @ (B, dk, MB) → (B, MB*7, MB)
        move_scores = move_scores.view(B, MB, self.directions, MB)
        move_scores = move_scores.permute(0, 1, 3, 2).contiguous()    # (B, MB, MB, 7)
        move_scores = move_scores.view(B, -1)                          # (B, 5488)

        # ── PLACE scores: (B, 8, MB, 6) → (B, 8*MB*6 = 1344) ──
        type_keys = self.type_emb.weight                               # (8, dk)
        type_keys = type_keys.unsqueeze(0).expand(B, -1, -1)          # (B, 8, dk)

        # type_dir[b,t,k,:] = type_keys[b,t,:] * dir_place[k,:]  → (B, 8, 6, dk)
        type_dir = type_keys.unsqueeze(2) * dir_place.unsqueeze(0).unsqueeze(0)

        type_dir_flat = type_dir.view(B, self.num_types * (self.directions - 1), dk)
        place_scores = torch.bmm(type_dir_flat, ref_keys.transpose(1, 2))
        #   (B, 8*6, dk) @ (B, dk, MB) → (B, 8*6, MB)
        place_scores = place_scores.view(B, self.num_types, self.directions - 1, MB)
        place_scores = place_scores.permute(0, 1, 3, 2).contiguous()  # (B, 8, MB, 6)
        place_scores = place_scores.view(B, -1)                        # (B, 1344)

        # ── FIRST-PLACEMENT scores: (B, 8) ──
        first_scores = self.fc_first(cls_h)                            # (B, 8)

        # ── PASS score: (B, 1) ──
        pass_score = self.fc_pass(cls_h)                               # (B, 1)

        # ── Concatenate → (B, 6841) ──
        return torch.cat([move_scores, place_scores, first_scores, pass_score], dim=1)


# ── Value Head ─────────────────────────────────────────────────────────────────


class PRSValueHead(nn.Module):
    def __init__(self, d_model: int, global_feat_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model + global_feat_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(
        self,
        cls_embedding: torch.Tensor,    # (B, d)
        global_features: torch.Tensor,  # (B, 6)
    ) -> torch.Tensor:
        combined = torch.cat([cls_embedding, global_features], dim=1)
        v = F.relu(self.fc1(combined))
        return torch.tanh(self.fc2(v))   # (B, 1)


# ── HivePRSTransformer ─────────────────────────────────────────────────────────


class HivePRSTransformer(nn.Module):
    """
    PRS Transformer: 23×23 input positions, piece-relative policy output.

    Forward:
        PRSTokenBatch → (policy_logits (B, 6841), value (B, 1))

    The policy head uses a factored bilinear score over board token pairs
    rather than a spatial convolution, so the output is naturally
    permutation-equivariant with respect to the board token ordering.
    """

    def __init__(self, config: PRSConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = PRSConfig()
        self.config = config
        d = config.d_model

        # Token feature projection (25 → d)
        self.token_proj = nn.Linear(config.token_feat_dim, d)

        # Position embedding: 530 entries (0-528 board + 529 off-board)
        self.position_embedding = nn.Embedding(config.max_positions, d)

        # Token type embedding: 3 types
        self.type_embedding = nn.Embedding(config.num_token_types, d)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Heads
        self.policy_head = PRSPolicyHead(config)
        self.value_head  = PRSValueHead(d, config.global_feat_dim)

    def forward(
        self, batch: PRSTokenBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: PRSTokenBatch — all tensors on the same device.

        Returns:
            policy_logits : (B, ACTION_SPACE_SIZE)
            value         : (B, 1) in [-1, 1]
        """
        padding_mask = ~batch.attention_mask   # True = pad (PyTorch convention)

        # 1. Embed tokens
        h = self.token_proj(batch.token_features)
        h = h + self.position_embedding(batch.token_positions)
        h = h + self.type_embedding(batch.token_types)

        # 2. Transformer
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)
        # h: (B, S, d)

        # 3. Heads
        cls_h = h[:, 0, :]                                # (B, d)
        policy_logits = self.policy_head(h, cls_h)        # (B, 6841)
        value         = self.value_head(cls_h, batch.global_features)  # (B, 1)

        return policy_logits, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
