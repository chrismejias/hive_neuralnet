"""
Transformer neural network for Hive.

Architecture:
    HiveTokenBatch → token/position/type embeddings →
    TransformerEncoder (self-attention × N) →
        ├── PolicyHead (scatter to grid → conv → FC) → logits (29407,)
        └── ValueHead (CLS token → FC → tanh) → scalar in [-1, 1]

Uses standard nn.TransformerEncoder with batch_first=True for
optimal GPU tensor core utilization and torch.compile() compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_engine.encoder import HiveEncoder

from hive_transformer.token_types import (
    TOKEN_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    OFF_BOARD_POSITION,
    TOKEN_TYPE_BOARD,
    HiveTokenSequence,
    HiveTokenBatch,
)


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class TransformerConfig:
    """Configuration for HiveTransformer architecture."""

    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_positions: int = 170        # 169 grid cells + 1 off-board
    num_token_types: int = 3        # CLS=0, board=1, hand=2
    token_feat_dim: int = TOKEN_FEAT_DIM   # 21
    global_feat_dim: int = GLOBAL_FEAT_DIM  # 6
    policy_conv_channels: int = 32
    action_space_size: int = HiveEncoder.ACTION_SPACE_SIZE  # 29407
    board_size: int = HiveEncoder.BOARD_SIZE                # 13

    @classmethod
    def small(cls) -> TransformerConfig:
        """Small preset (~3M parameters)."""
        return cls(d_model=128, num_heads=8, num_layers=6, dim_feedforward=512)

    @classmethod
    def large(cls) -> TransformerConfig:
        """Large preset (~12M parameters)."""
        return cls(d_model=256, num_heads=8, num_layers=12, dim_feedforward=1024)


# ── Policy Head ────────────────────────────────────────────────────


class TransformerPolicyHead(nn.Module):
    """
    Hybrid policy head: scatter board token embeddings into a spatial
    grid, then use conv + FC to produce action logits.

    Same approach as the GNN HybridPolicyHead.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        d = config.d_model
        bs = config.board_size
        pc = config.policy_conv_channels

        self.conv1 = nn.Conv2d(d, pc, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pc)
        self.conv2 = nn.Conv2d(pc, 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2)

        flatten_size = 2 * bs * bs  # 338
        self.fc1 = nn.Linear(flatten_size + config.global_feat_dim, 256)
        self.fc2 = nn.Linear(256, config.action_space_size)

        self._d_model = d
        self._board_size = bs

    def forward(
        self,
        grid: torch.Tensor,               # (B, d_model, 13, 13)
        global_features: torch.Tensor,     # (B, 6)
    ) -> torch.Tensor:
        """
        Compute policy logits from spatial grid.

        Returns:
            Policy logits (B, action_space_size).
        """
        p = F.relu(self.bn1(self.conv1(grid)))
        p = F.relu(self.bn2(self.conv2(p)))
        p = p.view(p.size(0), -1)  # (B, 338)
        p = torch.cat([p, global_features], dim=1)  # (B, 344)
        p = F.relu(self.fc1(p))
        return self.fc2(p)  # (B, 29407)


# ── Value Head ─────────────────────────────────────────────────────


class TransformerValueHead(nn.Module):
    """
    Value head using CLS token embedding + global features.

    CLS token → concat with global → FC → ReLU → FC → tanh
    """

    def __init__(self, d_model: int, global_feat_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model + global_feat_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(
        self,
        cls_embedding: torch.Tensor,      # (B, d_model)
        global_features: torch.Tensor,    # (B, global_feat_dim)
    ) -> torch.Tensor:
        """
        Compute value from CLS token.

        Returns:
            Value predictions (B, 1) in [-1, 1].
        """
        combined = torch.cat([cls_embedding, global_features], dim=1)
        v = F.relu(self.fc1(combined))
        return torch.tanh(self.fc2(v))


# ── HiveTransformer ───────────────────────────────────────────────


class HiveTransformer(nn.Module):
    """
    Transformer network for Hive, producing the same output interface
    as HiveNet and HiveGNN: policy logits (B, 29407) and value (B, 1).

    Architecture:
        1. Linear projection of token features (21 → d_model)
        2. Add learned position embedding (170 positions)
        3. Add token type embedding (3 types: CLS, board, hand)
        4. TransformerEncoder with self-attention (num_layers)
        5. PolicyHead: scatter board tokens to grid → conv → FC → logits
        6. ValueHead: CLS token → FC → tanh → value
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config

        d = config.d_model

        # Token feature projection
        self.token_proj = nn.Linear(config.token_feat_dim, d)

        # Position embedding: 170 positions (169 grid + 1 off-board)
        self.position_embedding = nn.Embedding(config.max_positions, d)

        # Token type embedding: 3 types (CLS=0, board=1, hand=2)
        self.type_embedding = nn.Embedding(config.num_token_types, d)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Policy head
        self.policy_head = TransformerPolicyHead(config)

        # Value head
        self.value_head = TransformerValueHead(d, config.global_feat_dim)

        self._d_model = d
        self._board_size = config.board_size

    def forward(
        self, batch: HiveTokenBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on a padded token batch.

        Args:
            batch: HiveTokenBatch with all tensors on the same device.

        Returns:
            (policy_logits, value) where:
                policy_logits: shape (B, 29407)
                value: shape (B, 1), in [-1, 1]
        """
        B = batch.global_features.size(0)
        device = batch.token_features.device

        # 1. Embed tokens
        h = self.token_proj(batch.token_features)        # (B, S, d)
        h = h + self.position_embedding(batch.token_positions)  # + pos
        h = h + self.type_embedding(batch.token_types)          # + type

        # 2. Create padding mask for transformer
        # PyTorch TransformerEncoder uses src_key_padding_mask where
        # True means "ignore this position" (opposite of our attention_mask)
        padding_mask = ~batch.attention_mask  # (B, S) True=pad

        # 3. Transformer encoder
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)
        # h: (B, S, d)

        # 4. Build spatial grid for policy head
        grid = self._scatter_to_grid(h, batch, B, device)

        # 5. Policy logits
        policy_logits = self.policy_head(grid, batch.global_features)

        # 6. Value from CLS token (always at position 0)
        cls_embedding = h[:, 0, :]  # (B, d)
        value = self.value_head(cls_embedding, batch.global_features)

        return policy_logits, value

    def _scatter_to_grid(
        self,
        embeddings: torch.Tensor,       # (B, S, d)
        batch: HiveTokenBatch,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Scatter board token embeddings into a spatial grid for policy head.

        Board tokens have token_type == 1 and valid grid positions.
        We place their embeddings at the corresponding (row, col) in a
        (B, d_model, board_size, board_size) tensor.

        Args:
            embeddings: Transformer output (B, S, d).
            batch: Token batch data.
            batch_size: Number of sequences in batch.
            device: Target device.

        Returns:
            Grid tensor (B, d_model, board_size, board_size).
        """
        d = self._d_model
        bs = self._board_size

        grid = torch.zeros(batch_size, d, bs, bs, device=device)

        # Find board tokens: token_types == TOKEN_TYPE_BOARD
        board_mask = batch.token_types == TOKEN_TYPE_BOARD  # (B, S)

        if not board_mask.any():
            return grid

        # Get indices of board tokens
        b_idx, s_idx = torch.where(board_mask)  # both (num_board_total,)

        # Get embeddings and positions for board tokens
        board_embeddings = embeddings[b_idx, s_idx]  # (num_board_total, d)
        board_positions = batch.token_positions[b_idx, s_idx]  # (num_board_total,)

        # Convert flat position to (row, col)
        rows = (board_positions // bs).clamp(0, bs - 1)
        cols = (board_positions % bs).clamp(0, bs - 1)

        # Scatter into grid using index_put_ with accumulate
        ch = torch.arange(d, device=device).unsqueeze(0).expand(b_idx.size(0), -1)
        b_exp = b_idx.unsqueeze(1).expand(-1, d)
        r_exp = rows.unsqueeze(1).expand(-1, d)
        c_exp = cols.unsqueeze(1).expand(-1, d)

        grid.index_put_(
            (b_exp, ch, r_exp, c_exp),
            board_embeddings,
            accumulate=True,
        )

        return grid

    @torch.no_grad()
    def predict(
        self,
        sequence: HiveTokenSequence,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Single-state inference for MCTS.

        Creates a single-element batch, runs forward, applies legal mask,
        returns action probabilities and value.

        Args:
            sequence: A HiveTokenSequence (numpy-based).
            legal_mask: Shape (29407,), float32. 1.0 for legal actions.

        Returns:
            (action_probs, value) where:
                action_probs: Shape (29407,), sums to ~1.0 over legal.
                value: Scalar in [-1, 1].
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        # Create single-element batch
        batch = HiveTokenBatch.collate([sequence]).to(device)
        mask = torch.from_numpy(legal_mask).to(device)

        policy_logits, value_tensor = self.forward(batch)

        # Apply legal mask
        policy_logits = policy_logits.squeeze(0)  # (29407,)
        policy_logits = policy_logits.masked_fill(mask == 0, float("-inf"))

        if mask.sum() > 0:
            action_probs = F.softmax(policy_logits, dim=0)
        else:
            action_probs = torch.zeros_like(policy_logits)

        if was_training:
            self.train()

        return (
            action_probs.cpu().numpy(),
            value_tensor.item(),
        )

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
