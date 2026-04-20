"""PRS v2 policy head: 813-slot structured output — rich variant.

The four block outputs are:
  * dir head    — direction-local moves for Q/B/G/P/M (48 slots)
  * throw head  — pillbug/mosquito throws (60 slots)
  * long head   — cross-attention piece × move-cell (448 slots)
  * hand head   — cross-attention hand-type × place-cell (256 slots)
  * PASS logit  — 1 slot

See `slot_map.py` for slot layout and `PRSv2HeadInputs` for tensor shapes.

This revision adds four enrichments over v1 of the head:

  1. **Real hand-token context.** Hand-type queries come from the trunk
     itself (gathered from `full_h` via `hand_token_idx`) instead of a
     learned 16-vector embedding. Absent types (hand count==0) fall
     back to a small learned table; their slots are masked out downstream.

  2. **Attention-pooled cell tokens + cell-geometry embeddings.** Each
     move/place cell is represented by an E_cell[cell_id] query that
     attends over the ≤6 adjacent *board* tokens, rather than a plain
     mean-pool. This gives the head sharp per-cell features.

  3. **FiLM-style CLS injection.** Each of the four blocks receives a
     γ/β modulation from cls_h so that global context (turn count,
     queens-surround pressure, etc.) can gate local scores.

  4. **Destination-conditioned dir/throw scoring.** Instead of
     `Linear(piece, 6)`, the dir head scores every (piece, direction)
     by a bilinear `<piece_tok, E_cell[dest] + W·board_h[dest]>`, with
     `dest` provided per-(piece, dir) by the CUDA bridge (grasshopper
     walks included). Throw head is analogous with E_cell[throw_dest].

All the new bridge fields (cell IDs, dest cells, dest board indices,
and hand-token positions) are computed on GPU in
`ext.prs_v2_classify_batch` and passed in via `PRSv2HeadInputs`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hive_prs.slot_map import (
    N_SLOTS, DIR_OFFSET, THROW_OFFSET, LONG_OFFSET, HAND_OFFSET, PASS_SLOT,
    C_MOVE, C_HAND,
)

MB = 28        # max board tokens per state
N_DIR_PIECES   = 8  # Q, B1, B2, G1, G2, G3, P, M
N_THROW_PIECES = 2  # P, M
N_LONG_PIECES  = 7  # A1, A2, A3, S1, S2, L, M
N_HAND_TYPES   = 8  # 8 piece types
N_CELLS_WITH_PAD = 530   # 529 board cells + 1 "off-board / pad" sentinel (index 529)
PAD_CELL_ID = 529

# Canonical piece-type indices into `piece_type_embed` (rows 0..7).
# Order: Q=0, A=1, G=2, S=3, B=4, M=5, L=6, P=7  (also matches hand head layout)
_DIR_TYPE_IDX   = (0, 4, 4, 2, 2, 2, 7, 5)   # Q, B1, B2, G1, G2, G3, P, M
_THROW_TYPE_IDX = (7, 5)                     # P, M
_LONG_TYPE_IDX  = (1, 1, 1, 3, 3, 6, 5)      # A1, A2, A3, S1, S2, L, M


@dataclass
class PRSv2HeadInputs:
    """Bundle of tensors needed by the v2 head. All batched (B, …)."""
    # Trunk outputs
    board_h: torch.Tensor                 # (B, MB, d)    per-occupied-cell tokens
    cls_h:   torch.Tensor                 # (B, d)        CLS summary
    full_h:  torch.Tensor                 # (B, S, d)     full trunk seq, for hand-token gather

    # Piece-instance → board-token index (-1 if piece not in play)
    dir_piece_idx:   torch.Tensor         # (B, 8)   int64
    throw_piece_idx: torch.Tensor         # (B, 2)   int64
    long_piece_idx:  torch.Tensor         # (B, 7)   int64

    # Cell-token → adjacent board-token indices; -1 = empty/pad
    move_nbrs:   torch.Tensor             # (B, 64, 6)  int64
    place_nbrs:  torch.Tensor             # (B, 32, 6)  int64
    move_mask:   torch.Tensor             # (B, 64)     bool — True = active cell
    place_mask:  torch.Tensor             # (B, 32)     bool

    # Per-cell absolute cell IDs (0..528); -1 for padding
    move_cell_ids:  torch.Tensor          # (B, 64)     int32
    place_cell_ids: torch.Tensor          # (B, 32)     int32

    # Destination info for dir / throw heads
    dir_dest_cell:      torch.Tensor      # (B, 8, 6)   int32 cell or -1
    dir_dest_board_idx: torch.Tensor      # (B, 8, 6)   int64 board-token idx or -1
    throw_dest_cell:    torch.Tensor      # (B, 2, 30)  int32 cell or -1

    # Trunk seq position per (color*8 + type) hand slot; -1 if hand count==0
    hand_token_idx: torch.Tensor          # (B, 16)  int64

    # Current color (0=white, 1=black), shape (B,)
    current_color: torch.Tensor           # (B,)  int64


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_gather(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows from h (B, N_tok, d) using idx of arbitrary trailing shape.

    `idx` may be any shape (..., ) with int64 entries; -1 → zero vector.
    Returns (..., d)."""
    B, _, d = h.shape
    flat = idx.reshape(B, -1)                                 # (B, K)
    safe = flat.clamp(min=0).to(torch.long)
    gathered = h.gather(1, safe.unsqueeze(-1).expand(-1, -1, d))  # (B, K, d)
    mask = (flat >= 0).unsqueeze(-1).to(h.dtype)               # (B, K, 1)
    gathered = gathered * mask
    return gathered.reshape(*idx.shape, d)


def _film(x: torch.Tensor, cls_h: torch.Tensor, mod: nn.Linear) -> torch.Tensor:
    """Apply FiLM gating: y = (1+γ)·x + β, where (γ, β) = mod(cls_h).

    `x` is (..., d); cls_h is (B, d). The γ/β are broadcast over any
    intermediate axes by expanding `cls_h` via `.view(B, 1..., d)`.
    """
    B, d = cls_h.shape
    gb = mod(cls_h)                         # (B, 2d)
    gamma, beta = gb.chunk(2, dim=-1)       # (B, d), (B, d)
    # Broadcast over all middle dims of x
    view_shape = [B] + [1] * (x.dim() - 2) + [d]
    gamma = gamma.view(*view_shape)
    beta  = beta.view(*view_shape)
    return (1.0 + gamma) * x + beta


# ── Head module ────────────────────────────────────────────────────────────

class PRSv2PolicyHead(nn.Module):
    """Structured 813-slot policy head (see module docstring)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d = d_model

        # Cell-geometry embedding (529 real cells + 1 pad sentinel)
        self.E_cell = nn.Embedding(N_CELLS_WITH_PAD, d_model)

        # Per-piece-type bias added to piece tokens before dir/throw/long
        # projections. Shared across instances (A1/A2/A3 get the same type
        # vector) so instance symmetry is preserved while giving each kind
        # its own learned role (ants vs ladybugs behave very differently).
        self.piece_type_embed = nn.Embedding(N_HAND_TYPES, d_model)
        nn.init.normal_(self.piece_type_embed.weight, std=0.02)
        self.register_buffer(
            "dir_type_idx",
            torch.tensor(_DIR_TYPE_IDX, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "throw_type_idx",
            torch.tensor(_THROW_TYPE_IDX, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "long_type_idx",
            torch.tensor(_LONG_TYPE_IDX, dtype=torch.long),
            persistent=False,
        )

        # Hand-type fallback when current hand count == 0
        # 16 rows = 2 colors × 8 types
        self.hand_fallback = nn.Embedding(16, d_model)

        # Attention-pool for cell tokens (shared move + place)
        self.pool_wq = nn.Linear(d_model, d_model, bias=False)
        self.pool_wk = nn.Linear(d_model, d_model, bias=False)
        self.pool_wv = nn.Linear(d_model, d_model, bias=False)

        # Destination / query projections for the four blocks
        self.dir_dest_board   = nn.Linear(d_model, d_model, bias=False)
        self.throw_dest_proj  = nn.Linear(d_model, d_model, bias=False)
        self.long_q           = nn.Linear(d_model, d_model, bias=False)
        self.long_k           = nn.Linear(d_model, d_model, bias=False)
        self.hand_q           = nn.Linear(d_model, d_model, bias=False)
        self.hand_k           = nn.Linear(d_model, d_model, bias=False)

        # FiLM modulators: cls_h → (γ, β) per block. Zero-init weights so
        # γ≈0, β≈0 at start → head behaves like trunk-only at init.
        self.film_dir   = nn.Linear(d_model, 2 * d_model)
        self.film_throw = nn.Linear(d_model, 2 * d_model)
        self.film_long  = nn.Linear(d_model, 2 * d_model)
        self.film_hand  = nn.Linear(d_model, 2 * d_model)
        for m in (self.film_dir, self.film_throw, self.film_long, self.film_hand):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # PASS logit
        self.pass_head = nn.Linear(d_model, 1)

    # ── Attention-pool over ≤6 neighbor board tokens ──

    def _attn_pool(
        self,
        board_h: torch.Tensor,       # (B, MB, d)
        cell_ids: torch.Tensor,      # (B, C)  int (cell index or -1)
        nbrs: torch.Tensor,          # (B, C, 6) int64
    ) -> torch.Tensor:
        """Attention-pool adjacent board tokens, with E_cell[cell_id] as query.

        Returns (B, C, d). Inactive cells (cell_id == -1) produce zero vectors."""
        B, C = cell_ids.shape
        d = self.d

        # Query from cell geometry embedding
        safe_ids = cell_ids.clamp(min=0).to(torch.long)
        q_cell = self.E_cell(safe_ids)                         # (B, C, d)

        # Gather neighbor vectors: -1 → zero
        K_nbrs = _safe_gather(board_h, nbrs)                   # (B, C, 6, d)

        Q = self.pool_wq(q_cell)                               # (B, C, d)
        K = self.pool_wk(K_nbrs)                               # (B, C, 6, d)
        V = self.pool_wv(K_nbrs)                               # (B, C, 6, d)

        scores = (Q.unsqueeze(2) * K).sum(-1) / (d ** 0.5)     # (B, C, 6)
        scores = scores.masked_fill(nbrs < 0, float("-inf"))

        # For cells with no valid neighbors (shouldn't happen for active
        # cells, but guard anyway), softmax of all -inf is NaN — guard by
        # adding a tiny epsilon for fully-masked rows.
        all_masked = (nbrs < 0).all(dim=-1, keepdim=True)      # (B, C, 1)
        scores = torch.where(
            all_masked.expand_as(scores),
            torch.zeros_like(scores),
            scores,
        )
        weights = scores.softmax(dim=-1)                       # (B, C, 6)
        pooled = (weights.unsqueeze(-1) * V).sum(dim=2)        # (B, C, d)

        # Add cell geometry residual so pure-pad cells still have signal
        pooled = pooled + q_cell

        # Zero out inactive cells (cell_id == -1)
        active = (cell_ids >= 0).to(pooled.dtype).unsqueeze(-1)
        return pooled * active

    # ── Hand-type token gather ──

    def _hand_tokens(
        self,
        full_h: torch.Tensor,        # (B, S, d)
        hand_token_idx: torch.Tensor,# (B, 16)   int64; -1 if count==0
        current_color: torch.Tensor, # (B,)  int64
    ) -> torch.Tensor:
        """Return (B, 8, d) — per-hand-type tokens for the current color.

        When a type has count == 0 (idx == -1), fall back to the learned
        `hand_fallback` embedding for (color, type). These rows will be
        masked out by legality filtering downstream, but a sane vector
        keeps gradients / softmax well-behaved."""
        B, _S, d = full_h.shape

        # Pick 8 slots [color*8 : color*8+8] per batch row
        base = current_color.unsqueeze(1) * N_HAND_TYPES           # (B, 1)
        type_range = torch.arange(N_HAND_TYPES, device=full_h.device, dtype=torch.long)
        sel = base + type_range.unsqueeze(0)                       # (B, 8)
        idx = hand_token_idx.gather(1, sel)                        # (B, 8) int64

        # Gather real tokens; -1 slots produce zero
        real_tok = _safe_gather(full_h, idx)                       # (B, 8, d)

        # Fallback embedding for missing types
        fb = self.hand_fallback(sel)                               # (B, 8, d)

        present = (idx >= 0).unsqueeze(-1).to(full_h.dtype)        # (B, 8, 1)
        return real_tok * present + fb * (1.0 - present)

    # ── Main forward ──

    def forward(self, inp: PRSv2HeadInputs) -> torch.Tensor:
        """Return policy logits (B, N_SLOTS=813)."""
        B = inp.board_h.size(0)
        d = self.d
        device = inp.board_h.device
        scale = d ** -0.5

        # ── Piece-instance tokens from board_h ──
        dir_tok   = _safe_gather(inp.board_h, inp.dir_piece_idx)    # (B, 8, d)
        throw_tok = _safe_gather(inp.board_h, inp.throw_piece_idx)  # (B, 2, d)
        long_tok  = _safe_gather(inp.board_h, inp.long_piece_idx)   # (B, 7, d)

        # Add per-type bias — shared across instances, distinct across kinds
        dir_tok   = dir_tok   + self.piece_type_embed(self.dir_type_idx)   # broadcast (8,d)
        throw_tok = throw_tok + self.piece_type_embed(self.throw_type_idx) # broadcast (2,d)
        long_tok  = long_tok  + self.piece_type_embed(self.long_type_idx)  # broadcast (7,d)

        # ── Cell tokens via attention-pool ──
        move_cell = self._attn_pool(
            inp.board_h, inp.move_cell_ids.to(torch.long), inp.move_nbrs,
        )                                                            # (B, 64, d)
        place_cell = self._attn_pool(
            inp.board_h, inp.place_cell_ids.to(torch.long), inp.place_nbrs,
        )                                                            # (B, 32, d)

        # ── Hand-type tokens (real trunk gather + fallback) ──
        hand_tok = self._hand_tokens(
            inp.full_h, inp.hand_token_idx, inp.current_color,
        )                                                            # (B, 8, d)

        # ── Head 1: direction bilinear ──
        # dir_dest_cell: (B, 8, 6) int32 ; -1 → pad
        # dir_dest_board_idx: (B, 8, 6) int64 ; -1 → empty dest (no board token)
        ddc = inp.dir_dest_cell.to(torch.long)                       # (B, 8, 6)
        # Map -1 → PAD_CELL_ID so E_cell has a stable pad row
        ddc_safe = torch.where(ddc >= 0, ddc, torch.full_like(ddc, PAD_CELL_ID))
        dest_cell_emb = self.E_cell(ddc_safe)                        # (B, 8, 6, d)

        # Optional board-token contribution (for beetles climbing occupied cells)
        dest_board = _safe_gather(inp.board_h, inp.dir_dest_board_idx)  # (B, 8, 6, d)
        dest_vec = dest_cell_emb + self.dir_dest_board(dest_board)      # (B, 8, 6, d)
        dest_vec = _film(dest_vec, inp.cls_h, self.film_dir)

        dir_l = (dir_tok.unsqueeze(2) * dest_vec).sum(-1) * scale    # (B, 8, 6)
        # Zero out invalid (off-board) destinations
        dir_valid = (ddc >= 0).to(dir_l.dtype)
        dir_l = dir_l * dir_valid

        # ── Head 2: throw bilinear ──
        tdc = inp.throw_dest_cell.to(torch.long)                     # (B, 2, 30)
        tdc_safe = torch.where(tdc >= 0, tdc, torch.full_like(tdc, PAD_CELL_ID))
        throw_dest_emb = self.E_cell(tdc_safe)                       # (B, 2, 30, d)
        throw_dest_vec = self.throw_dest_proj(throw_dest_emb)        # (B, 2, 30, d)
        throw_dest_vec = _film(throw_dest_vec, inp.cls_h, self.film_throw)

        throw_l = (throw_tok.unsqueeze(2) * throw_dest_vec).sum(-1) * scale  # (B, 2, 30)
        throw_valid = (tdc >= 0).to(throw_l.dtype)
        throw_l = throw_l * throw_valid

        # ── Head 3: long-range cross-attention (piece × move-cell) ──
        Km = self.long_k(move_cell)                                  # (B, 64, d)
        Qp = self.long_q(long_tok)                                   # (B, 7, d)
        # FiLM inject on the query side
        Qp = _film(Qp, inp.cls_h, self.film_long)
        long_l = torch.einsum("bpd,bcd->bpc", Qp, Km) * scale        # (B, 7, 64)
        long_l = long_l.masked_fill(~inp.move_mask.unsqueeze(1), 0.0)

        # ── Head 4: placement cross-attention (hand-type × place-cell) ──
        Kp = self.hand_k(place_cell)                                 # (B, 32, d)
        Qh = self.hand_q(hand_tok)                                   # (B, 8, d)
        Qh = _film(Qh, inp.cls_h, self.film_hand)
        hand_l = torch.einsum("btd,bcd->btc", Qh, Kp) * scale        # (B, 8, 32)
        hand_l = hand_l.masked_fill(~inp.place_mask.unsqueeze(1), 0.0)

        # ── PASS logit ──
        pass_l = self.pass_head(inp.cls_h)                           # (B, 1)

        # ── Flatten in canonical slot order ──
        out = torch.empty(B, N_SLOTS, device=device, dtype=dir_l.dtype)
        out[:, DIR_OFFSET:DIR_OFFSET + 48]      = dir_l.reshape(B, 48)
        out[:, THROW_OFFSET:THROW_OFFSET + 60]  = throw_l.reshape(B, 60)
        out[:, LONG_OFFSET:LONG_OFFSET + 448]   = long_l.reshape(B, 448)
        out[:, HAND_OFFSET:HAND_OFFSET + 256]   = hand_l.reshape(B, 256)
        out[:, PASS_SLOT:PASS_SLOT + 1]         = pass_l
        return out
