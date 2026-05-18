"""Compatibility wrapper for the archived move-conditioned transformer."""

<<<<<<<< HEAD:hive_mc/mc_transformer.py
from archive.legacy_mc.hive_mc.mc_transformer import *
========
Two-stage architecture:
1. Screening head: lightweight scoring of all legal moves from compact features
2. Action head: precise scoring of selected candidates via full successor encoding

During self-play, only top-k candidates from screening are fully encoded,
saving ~3-5x compute. During training, all successors are encoded for both heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_engine.encoder import HiveEncoder
from archive.legacy_mc.hive_mc.mc_utils import (
    MoveConditionedBatch,
    MoveFeatures,
    ENGINE_MAX_POSITIONS,
)
from hive_common.token_types import HiveTokenBatch, GLOBAL_FEAT_DIM, TOKEN_FEAT_DIM

NUM_PIECE_TYPES = 8


@dataclass
class MCTransformerConfig:
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_positions: int = HiveEncoder.BOARD_SIZE ** 2 + 1  # 290
    num_token_types: int = 3
    token_feat_dim: int = TOKEN_FEAT_DIM
    global_feat_dim: int = GLOBAL_FEAT_DIM
    screen_embed_dim: int = 16
    max_candidates: int = 16

    @classmethod
    def small(cls) -> MCTransformerConfig:
        return cls(d_model=128, num_heads=8, num_layers=3, dim_feedforward=512)

    @classmethod
    def large(cls) -> MCTransformerConfig:
        return cls(d_model=256, num_heads=8, num_layers=6, dim_feedforward=1024)


class MCScreeningHead(nn.Module):
    """Lightweight head scoring moves from root CLS + compact move features."""

    def __init__(self, d_model: int, global_feat_dim: int, screen_embed_dim: int = 16) -> None:
        super().__init__()
        self.piece_emb = nn.Embedding(NUM_PIECE_TYPES, screen_embed_dim)
        self.move_type_emb = nn.Embedding(3, screen_embed_dim)  # PLACE=0, MOVE=1, PASS=2
        self.from_pos_emb = nn.Embedding(ENGINE_MAX_POSITIONS, screen_embed_dim)
        self.to_pos_emb = nn.Embedding(ENGINE_MAX_POSITIONS, screen_embed_dim)
        input_dim = d_model + global_feat_dim + screen_embed_dim * 4
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(
        self,
        root_cls: torch.Tensor,
        root_global: torch.Tensor,
        piece_types: torch.Tensor,
        move_types: torch.Tensor,
        from_positions: torch.Tensor,
        to_positions: torch.Tensor,
    ) -> torch.Tensor:
        p = self.piece_emb(piece_types)
        m = self.move_type_emb(move_types)
        f = self.from_pos_emb(from_positions)
        t = self.to_pos_emb(to_positions)
        x = torch.cat([root_cls, root_global, p, m, f, t], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class MCActionHead(nn.Module):
    """Score a successor state conditioned on the root state embedding."""

    def __init__(self, d_model: int, global_feat_dim: int) -> None:
        super().__init__()
        fused_dim = d_model * 4 + global_feat_dim * 2
        self.fc1 = nn.Linear(fused_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.ln = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 1)

    def forward(
        self,
        root_cls: torch.Tensor,
        action_cls: torch.Tensor,
        root_global: torch.Tensor,
        action_global: torch.Tensor,
    ) -> torch.Tensor:
        feats = torch.cat(
            [
                root_cls,
                action_cls,
                action_cls - root_cls,
                action_cls * root_cls,
                root_global,
                action_global,
            ],
            dim=1,
        )
        x = F.relu(self.fc1(feats))
        x = self.ln(F.relu(self.fc2(x)))
        return self.fc3(x).squeeze(-1)


class MCValueHead(nn.Module):
    def __init__(self, d_model: int, global_feat_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model + global_feat_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, cls: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cls, global_features], dim=1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class HiveMoveTransformer(nn.Module):
    """Two-stage move-conditioned transformer.

    Self-play: encode root → screen all moves → top-k → encode successors → Q-values
    Training:  encode root + all successors → screening loss + action loss + value loss
    """

    def __init__(self, config: MCTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or MCTransformerConfig.small()
        d = self.config.d_model

        self.token_proj = nn.Linear(self.config.token_feat_dim, d)
        self.position_embedding = nn.Embedding(self.config.max_positions, d)
        self.type_embedding = nn.Embedding(self.config.num_token_types, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
        )
        self.final_ln = nn.LayerNorm(d)

        self.screening_head = MCScreeningHead(
            d, self.config.global_feat_dim, self.config.screen_embed_dim,
        )
        self.action_head = MCActionHead(d, self.config.global_feat_dim)
        self.value_head = MCValueHead(d, self.config.global_feat_dim)

    def encode(self, batch: HiveTokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared trunk. Returns (all_tokens, cls_token)."""
        x = self.token_proj(batch.token_features)
        x = x + self.position_embedding(batch.token_positions)
        x = x + self.type_embedding(batch.token_types)
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=~batch.attention_mask,
        )
        x = self.final_ln(x)
        cls = x[:, 0]
        return x, cls

    def screen(
        self,
        root_cls: torch.Tensor,
        root_global: torch.Tensor,
        move_features: MoveFeatures,
    ) -> torch.Tensor:
        """Score all legal moves cheaply. Returns (N_actions,) logits."""
        if move_features.action_to_root.shape[0] == 0:
            return root_cls.new_zeros((0,))
        gathered_cls = root_cls[move_features.action_to_root]
        gathered_global = root_global[move_features.action_to_root]
        return self.screening_head(
            gathered_cls,
            gathered_global,
            move_features.piece_types,
            move_features.move_types,
            move_features.from_positions,
            move_features.to_positions,
        )

    def forward(
        self,
        batch: MoveConditionedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward for training (encodes root + all successors).

        Returns:
            screening_logits: (N_actions,) from screening head
            action_logits:    (N_actions,) from action head on successors
            root_values:      (B, 1)
            action_values:    (N_actions, 1) from value head on successors
        """
        _root_tokens, root_cls = self.encode(batch.root_batch)
        root_values = self.value_head(root_cls, batch.root_batch.global_features)

        mf = batch.move_features
        screening_logits = self.screen(root_cls, batch.root_batch.global_features, mf)

        if mf.action_to_root.shape[0] == 0:
            empty = root_cls.new_zeros((0,))
            return screening_logits, empty, root_values, empty.unsqueeze(1)

        _action_tokens, action_cls = self.encode(batch.action_batch)
        action_values = self.value_head(action_cls, batch.action_batch.global_features)

        gathered_root_cls = root_cls[mf.action_to_root]
        gathered_root_global = batch.root_batch.global_features[mf.action_to_root]
        action_logits = self.action_head(
            gathered_root_cls,
            action_cls,
            gathered_root_global,
            batch.action_batch.global_features,
        )
        return screening_logits, action_logits, root_values, action_values

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
>>>>>>>> 7c7d146 (Refactor legacy transformer and MC packages):archive/legacy_mc/hive_mc/mc_transformer.py
