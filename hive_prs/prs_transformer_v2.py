"""
PRS v2 transformer: trunk + structured 813-slot policy head.

The trunk body (token embeddings + TransformerEncoder + value head) is
identical to v1; only the policy head is replaced. This module exposes a
split API because the new head needs per-state SlotMapper inputs that the
caller builds on CPU from raw state bytes:

    board_h, cls_h, value = net.forward_trunk(prs_batch)
    inp = build_head_inputs_from_states(state_bytes, board_h, cls_h)
    policy_logits_813 = net.head(inp)

The convenience `forward(prs_batch, state_bytes)` does all three in one
call when the caller already has state bytes on hand.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hive_prs.prs_encoder import PRSTokenBatch
from hive_prs.prs_transformer import PRSConfig, PRSValueHead
from hive_prs.prs_v2_head import PRSv2PolicyHead, MB
from hive_prs.prs_v2_bridge import build_head_inputs_from_states


class HivePRSTransformerV2(nn.Module):
    """PRS transformer with the v2 structured 813-slot policy head.

    Trunk weights are compatible with v1; only the policy head differs.
    """

    def __init__(self, config: PRSConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = PRSConfig()
        self.config = config
        d = config.d_model

        # ── Trunk (same layout as v1) ─────────────────────────────────
        self.token_proj = nn.Linear(config.token_feat_dim, d)
        self.position_embedding = nn.Embedding(config.max_positions, d)
        self.type_embedding = nn.Embedding(config.num_token_types, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,
        )

        # ── Heads ─────────────────────────────────────────────────────
        self.head = PRSv2PolicyHead(d)
        self.value_head = PRSValueHead(d, config.global_feat_dim)

    # ── Trunk only (for orchestrator / evaluator) ────────────────────

    def forward_trunk(
        self, batch: PRSTokenBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (board_h [B, MB, d], cls_h [B, d], full_h [B, S, d], value [B, 1]).

        `board_h` is the per-occupied-cell token slice (positions 1..MB),
        zero-padded when S-1 < MB.

        `full_h` is the complete trunk output sequence, used by the head to
        gather per-(color, type) hand tokens via `hand_token_idx` from the
        CUDA bridge.
        """
        padding_mask = ~batch.attention_mask
        h = self.token_proj(batch.token_features)
        h = h + self.position_embedding(batch.token_positions)
        h = h + self.type_embedding(batch.token_types)
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        cls_h = h[:, 0, :]                            # (B, d)

        B = h.size(0)
        device = h.device
        raw_board = h[:, 1:1 + MB, :]
        if raw_board.size(1) < MB:
            pad = torch.zeros(B, MB - raw_board.size(1), raw_board.size(2),
                              dtype=raw_board.dtype, device=device)
            board_h = torch.cat([raw_board, pad], dim=1)
        else:
            board_h = raw_board

        value = self.value_head(cls_h, batch.global_features)
        return board_h, cls_h, h, value

    # ── Full forward, given state bytes ──────────────────────────────

    def forward(
        self,
        batch: PRSTokenBatch,
        state_bytes_cpu,   # np.ndarray (B, SIZEOF_HIVE_STATE) uint8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits [B, 813], value [B, 1])."""
        board_h, cls_h, full_h, value = self.forward_trunk(batch)
        inp, _ = build_head_inputs_from_states(
            state_bytes_cpu, board_h, cls_h, full_h,
        )
        policy_logits = self.head(inp)
        return policy_logits, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
