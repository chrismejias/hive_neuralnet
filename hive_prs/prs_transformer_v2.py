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

import os

import torch
import torch.nn as nn

from hive_prs.prs_encoder import PRSTokenBatch
from hive_prs.prs_transformer import PRSConfig, PRSValueHead
from hive_prs.prs_v2_head import PRSv2PolicyHead, MB
from hive_prs.prs_v2_bridge import (
    build_head_inputs_from_kernel,
    build_head_inputs_from_states,
)


class _PerBoardHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, board_embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(board_embeddings)


class HivePRSTransformerV2(nn.Module):
    """PRS transformer with the v2 structured 813-slot policy head.

    Trunk weights are compatible with v1; only the policy head differs.
    """

    def __init__(self, config: PRSConfig | None = None) -> None:
        super().__init__()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
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
        self.queen_surround_head = _PerBoardHead(d, out_dim=2)
        self.final_mobility_head = _PerBoardHead(d, out_dim=1)
        self._compiled_trunk = None
        self._compiled_head = None
        self._compile_warned = False

    def _extract_board_embeddings(
        self,
        board_h: torch.Tensor,
        batch: PRSTokenBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = batch.num_board_tokens.to(torch.int64)
        if counts.numel() == 0:
            empty = torch.zeros(0, board_h.size(-1), device=board_h.device, dtype=board_h.dtype)
            return empty, torch.zeros(0, device=board_h.device, dtype=torch.int64)
        max_board = board_h.size(1)
        token_idx = torch.arange(max_board, device=board_h.device).unsqueeze(0)
        valid = token_idx < counts.unsqueeze(1)
        if not valid.any():
            empty = torch.zeros(0, board_h.size(-1), device=board_h.device, dtype=board_h.dtype)
            return empty, torch.zeros(0, device=board_h.device, dtype=torch.int64)
        batch_idx, board_idx = torch.where(valid)
        return board_h[batch_idx, board_idx], batch_idx

    def enable_compiled_forward(self, enabled: bool = True) -> None:
        """Compile tensor-only trunk/head paths when available.

        The CUDA bridge that builds `PRSv2HeadInputs` stays outside Dynamo; it
        is an extension call with dynamic state-derived outputs. If compile
        fails at runtime, `forward_trunk`/`forward_head` fall back once to eager.
        """
        if not enabled or not hasattr(torch, "compile"):
            object.__setattr__(self, "_compiled_trunk", None)
            object.__setattr__(self, "_compiled_head", None)
            return
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        try:
            import torch._inductor.config as inductor_config
            inductor_config.compile_threads = 1
        except Exception:
            pass
        object.__setattr__(
            self,
            "_compiled_trunk",
            torch.compile(
                self._forward_trunk_tensors,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            ),
        )
        object.__setattr__(
            self,
            "_compiled_head",
            torch.compile(
                self.head,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            ),
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

        value = self.value_head(cls_h, global_features)
        return board_h, cls_h, h, value

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
        fn = self._compiled_trunk or self._forward_trunk_tensors
        try:
            return fn(
                batch.token_features,
                batch.token_positions,
                batch.token_types,
                batch.attention_mask,
                batch.global_features,
            )
        except Exception as exc:
            if self._compiled_trunk is None:
                raise
            if not self._compile_warned:
                print(f"[PRS_V2_COMPILE_FALLBACK] trunk: {type(exc).__name__}: {exc}")
                self._compile_warned = True
            object.__setattr__(self, "_compiled_trunk", None)
            return self._forward_trunk_tensors(
                batch.token_features,
                batch.token_positions,
                batch.token_types,
                batch.attention_mask,
                batch.global_features,
            )

    def forward_head(self, inp) -> torch.Tensor:
        fn = self._compiled_head or self.head
        try:
            return fn(inp)
        except Exception as exc:
            if self._compiled_head is None:
                raise
            if not self._compile_warned:
                print(f"[PRS_V2_COMPILE_FALLBACK] head: {type(exc).__name__}: {exc}")
                self._compile_warned = True
            object.__setattr__(self, "_compiled_head", None)
            return self.head(inp)

    def forward_from_kernel(
        self,
        batch: PRSTokenBatch,
        kernel_out: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits [B, 813], value [B, 1]) using CUDA bridge output."""
        board_h, cls_h, full_h, value = self.forward_trunk(batch)
        inp, _ = build_head_inputs_from_kernel(board_h, cls_h, full_h, kernel_out)
        return self.forward_head(inp), value

    def forward_train_from_kernel(
        self,
        batch: PRSTokenBatch,
        kernel_out: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        board_h, cls_h, full_h, value = self.forward_trunk(batch)
        inp, _ = build_head_inputs_from_kernel(board_h, cls_h, full_h, kernel_out)
        policy_logits = self.forward_head(inp)
        board_embeddings, _ = self._extract_board_embeddings(board_h, batch)
        aux_outputs: dict[str, torch.Tensor] = {}
        if board_embeddings.numel() > 0:
            aux_outputs["queen_surround_logits"] = self.queen_surround_head(board_embeddings)
            aux_outputs["final_mobility_logits"] = self.final_mobility_head(board_embeddings)
        return policy_logits, value, aux_outputs

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
        policy_logits = self.forward_head(inp)
        return policy_logits, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
