"""PRS v3 transformer: PRS v2 head with relative hex-offset attention bias.

PRS v3 keeps the v2 tokenization, value head, auxiliary heads, and structured
813-slot policy head. The only architectural change is in the trunk: each
self-attention layer receives a learned per-head bias based on the relative
23x23 board offset between token pairs.

The bias table is zero-initialized, so v2 checkpoints can be migrated with
strict=False and initially behave close to v2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_prs.prs_encoder import PRSTokenBatch
from hive_prs.prs_transformer import PRSConfig
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2


_BOARD_SIZE = 23
_BOARD_CELLS = _BOARD_SIZE * _BOARD_SIZE


class RelativeTransformerEncoderLayer(nn.Module):
    """Pre-norm TransformerEncoderLayer with additive relative-position bias.

    Parameter names intentionally mirror PyTorch's TransformerEncoderLayer so
    existing v2 checkpoint tensors for self-attention, feedforward, norms, and
    dropouts load without renaming. The new `relative_bias` parameter is the
    only fresh tensor.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        rel_clip: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.num_heads = nhead
        self.rel_clip = int(rel_clip)
        n_offsets = (2 * self.rel_clip + 1) ** 2
        self.offboard_bucket = n_offsets
        self.relative_bias = nn.Parameter(torch.zeros(nhead, n_offsets + 1))

    def _relative_attn_mask(self, bucket: torch.Tensor) -> torch.Tensor:
        B, S, _ = bucket.shape
        flat_bucket = bucket.reshape(-1)
        bias = self.relative_bias[:, flat_bucket]
        bias = bias.view(self.num_heads, B, S, S).permute(1, 0, 2, 3)
        return bias.reshape(B * self.num_heads, S, S)

    def forward(
        self,
        src: torch.Tensor,
        relative_bucket: torch.Tensor,
        expanded_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = src
        y = self.norm1(x)
        attn_bias = self._relative_attn_mask(relative_bucket).to(dtype=y.dtype)
        if expanded_pad_mask is not None:
            attn_bias = attn_bias.masked_fill(expanded_pad_mask, float("-inf"))
        y, _ = self.self_attn(
            y,
            y,
            y,
            attn_mask=attn_bias,
            key_padding_mask=None,
            need_weights=False,
        )
        x = x + self.dropout1(y)

        y = self.norm2(x)
        y = self.linear2(self.dropout(F.relu(self.linear1(y))))
        x = x + self.dropout2(y)
        return x


class RelativeTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        rel_clip: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RelativeTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rel_clip=rel_clip,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        src: torch.Tensor,
        token_positions: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S = token_positions.shape
        pos = token_positions.to(torch.int64)
        valid = (pos >= 0) & (pos < _BOARD_CELLS)
        row = torch.div(pos.clamp(0, _BOARD_CELLS - 1), _BOARD_SIZE, rounding_mode="floor")
        col = pos.clamp(0, _BOARD_CELLS - 1) % _BOARD_SIZE
        c = self.layers[0].rel_clip
        dr = row.unsqueeze(1) - row.unsqueeze(2)
        dc = col.unsqueeze(1) - col.unsqueeze(2)
        dr = dr.clamp(-c, c) + c
        dc = dc.clamp(-c, c) + c
        relative_bucket = dr * (2 * c + 1) + dc
        offboard_bucket = self.layers[0].offboard_bucket
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
        relative_bucket = torch.where(
            pair_valid,
            relative_bucket,
            torch.full_like(relative_bucket, offboard_bucket),
        )

        expanded_pad_mask = None
        if src_key_padding_mask is not None:
            num_heads = self.layers[0].num_heads
            expanded_pad_mask = src_key_padding_mask[:, None, None, :].expand(
                B, num_heads, S, S,
            ).reshape(B * num_heads, S, S)

        h = src
        for layer in self.layers:
            h = layer(
                h,
                relative_bucket=relative_bucket,
                expanded_pad_mask=expanded_pad_mask,
            )
        return h


class HivePRSTransformerV3(HivePRSTransformerV2):
    """PRS v3: v2 structured head plus per-head relative attention bias."""

    model_version = "v3"

    def __init__(self, config: PRSConfig | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        rel_clip = int(getattr(cfg, "relative_position_clip", 8))
        self.transformer_encoder = RelativeTransformerEncoder(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            num_layers=cfg.num_layers,
            rel_clip=rel_clip,
        )

    def _forward_trunk_tensors(
        self,
        token_features: torch.Tensor,
        token_positions: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        padding_mask = ~attention_mask
        h = self.token_proj(token_features)
        h = h + self.position_embedding(token_positions)
        h = h + self.type_embedding(token_types)
        h = self.transformer_encoder(
            h,
            token_positions=token_positions,
            src_key_padding_mask=padding_mask,
        )

        cls_h = h[:, 0, :]
        B = h.size(0)
        device = h.device
        raw_board = h[:, 1:1 + 28, :]
        if raw_board.size(1) < 28:
            pad = torch.zeros(
                B,
                28 - raw_board.size(1),
                raw_board.size(2),
                dtype=raw_board.dtype,
                device=device,
            )
            board_h = torch.cat([raw_board, pad], dim=1)
        else:
            board_h = raw_board

        value = self.value_head(cls_h, global_features)
        return board_h, cls_h, h, value
