"""Hybrid relative-transformer + FNN network.

Policy keeps the FNN successor-state feature path, but scores each action with
both the root transformer summary and the root/successor FNN embeddings. Value
uses the same root transformer context through a piece-only relative-position
trunk over the 28 board pieces.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hive_fnn.fnn_features import FEAT_DIM
from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_fnn_transformer.graph_types import (
    GLOBAL_FEAT_DIM,
    MAX_PIECE_TOKENS,
    NODE_FEAT_DIM,
    HybridPieceTensorBatch,
)

MOVE_FEAT_DIM = 25


@dataclass
class HybridGNNConfig:
    fnn_config: FNNConfig | None = None
    graph_hidden_dim: int = 64
    graph_layers: int = 4
    graph_radius: int = 2
    graph_mlp_hidden: int = 96
    num_heads: int = 8
    max_piece_tokens: int = MAX_PIECE_TOKENS
    rel_coord_clip: int = 8
    rel_height_clip: int = 2
    global_pool_bias: bool = True
    value_hidden: int = 128
    move_feat_dim: int = MOVE_FEAT_DIM
    node_feat_dim: int = NODE_FEAT_DIM
    global_feat_dim: int = GLOBAL_FEAT_DIM

    @classmethod
    def small(cls) -> "HybridGNNConfig":
        return cls(
            fnn_config=FNNConfig.medium(),
            graph_hidden_dim=64,
            graph_layers=4,
            graph_radius=2,
            graph_mlp_hidden=96,
            value_hidden=96,
        )

    @classmethod
    def large(cls) -> "HybridGNNConfig":
        return cls(
            fnn_config=FNNConfig.large(),
            graph_hidden_dim=96,
            graph_layers=6,
            graph_radius=2,
            graph_mlp_hidden=96,
            value_hidden=160,
        )

    @property
    def edge_feat_dim(self) -> int:
        return 3


def _mean_max_pool(
    node_h: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    hidden = node_h.size(1)
    mean_pool = torch.zeros(batch_size, hidden, device=node_h.device, dtype=node_h.dtype)
    counts = torch.zeros(batch_size, 1, device=node_h.device, dtype=node_h.dtype)
    mean_pool.scatter_add_(0, batch.unsqueeze(1).expand_as(node_h), node_h)
    counts.scatter_add_(
        0,
        batch.unsqueeze(1),
        torch.ones(batch.size(0), 1, device=node_h.device, dtype=node_h.dtype),
    )
    mean_pool = mean_pool / counts.clamp_min(1.0)

    max_pool = torch.full(
        (batch_size, hidden),
        -torch.inf,
        device=node_h.device,
        dtype=node_h.dtype,
    )
    max_pool.scatter_reduce_(
        0,
        batch.unsqueeze(1).expand_as(node_h),
        node_h,
        reduce="amax",
        include_self=False,
    )
    max_pool = max_pool.masked_fill(max_pool == -torch.inf, 0.0)
    return torch.cat([mean_pool, max_pool], dim=1)


class HybridMessagePassingLayer(nn.Module):
    """Legacy message-passing block retained for archived diagnostics."""

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        mlp_hidden: int,
        global_pool_bias: bool,
    ) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.global_pool_bias = bool(global_pool_bias)
        if self.global_pool_bias:
            self.global_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            out = self.norm(node_h)
        else:
            src = edge_index[0]
            dst = edge_index[1]
            msg_in = torch.cat([node_h[dst], node_h[src], edge_features], dim=1)
            messages = self.message_mlp(msg_in)
            aggregated = torch.zeros_like(node_h)
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
            update = self.update_mlp(torch.cat([node_h, aggregated], dim=1))
            out = self.norm(node_h + update)

        if self.global_pool_bias:
            out = out + self.global_proj(_mean_max_pool(out, batch, batch_size))[batch]
        return out

class RelativeSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, rel_coord_clip: int, rel_height_clip: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rel_coord_clip = rel_coord_clip
        self.rel_height_clip = rel_height_clip

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dq_bias = nn.Embedding(rel_coord_clip * 2 + 1, num_heads)
        self.dr_bias = nn.Embedding(rel_coord_clip * 2 + 1, num_heads)
        self.dz_bias = nn.Embedding(rel_height_clip * 2 + 1, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        token_q: torch.Tensor,
        token_r: torch.Tensor,
        token_z: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        dq = (token_q.unsqueeze(1) - token_q.unsqueeze(2)).clamp(
            -self.rel_coord_clip, self.rel_coord_clip,
        ) + self.rel_coord_clip
        dr = (token_r.unsqueeze(1) - token_r.unsqueeze(2)).clamp(
            -self.rel_coord_clip, self.rel_coord_clip,
        ) + self.rel_coord_clip
        dz = (token_z.unsqueeze(1) - token_z.unsqueeze(2)).clamp(
            -self.rel_height_clip, self.rel_height_clip,
        ) + self.rel_height_clip
        rel_bias = (
            self.dq_bias(dq)
            + self.dr_bias(dr)
            + self.dz_bias(dz)
        ).permute(0, 3, 1, 2)
        scores = scores + rel_bias.to(scores.dtype)

        key_mask = token_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(out)


class RelativeTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_hidden: int, rel_coord_clip: int, rel_height_clip: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RelativeSelfAttention(hidden_dim, num_heads, rel_coord_clip, rel_height_clip)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        token_q: torch.Tensor,
        token_r: torch.Tensor,
        token_z: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), token_q, token_r, token_z, token_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class HybridRelativeTransformerTrunk(nn.Module):
    def __init__(self, config: HybridGNNConfig) -> None:
        super().__init__()
        h = config.graph_hidden_dim
        self.max_piece_tokens = config.max_piece_tokens
        self.token_proj = nn.Linear(config.node_feat_dim, h)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
        self.layers = nn.ModuleList([
            RelativeTransformerBlock(
                h,
                config.num_heads,
                config.graph_mlp_hidden,
                config.rel_coord_clip,
                config.rel_height_clip,
            )
            for _ in range(config.graph_layers)
        ])
        self.norm = nn.LayerNorm(h)
        self.out_dim = h

    def forward(self, piece_batch: HybridPieceTensorBatch) -> torch.Tensor:
        token_h = self.token_proj(piece_batch.token_features)
        token_h = token_h * piece_batch.token_mask.unsqueeze(-1).to(token_h.dtype)

        batch_size = token_h.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, token_h], dim=1)

        zeros = torch.zeros(
            batch_size, 1,
            dtype=piece_batch.token_q.dtype,
            device=piece_batch.token_q.device,
        )
        token_q = torch.cat([zeros, piece_batch.token_q], dim=1)
        token_r = torch.cat([zeros, piece_batch.token_r], dim=1)
        token_z = torch.cat([zeros, piece_batch.token_z], dim=1)
        token_mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=piece_batch.token_mask.device),
                piece_batch.token_mask,
            ],
            dim=1,
        )

        piece_mask_f = piece_batch.token_mask.unsqueeze(-1).to(token_h.dtype)
        for layer in self.layers:
            x = layer(x, token_q, token_r, token_z, token_mask)
            x[:, 1:] = x[:, 1:] * piece_mask_f
        x = self.norm(x)

        return x[:, 0]


class HiveHybridGNN(nn.Module):
    """FNN successor features plus transformer-aware policy/value heads."""

    model_version = "fnn_transformer"

    def __init__(self, config: HybridGNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridGNNConfig()
        fnn_config = self.config.fnn_config or FNNConfig.large()
        if fnn_config.feat_dim != FEAT_DIM:
            raise ValueError(f"FNN feature dim must be {FEAT_DIM}, got {fnn_config.feat_dim}")

        self.fnn = HiveFNN(fnn_config)
        self.graph_trunk = HybridRelativeTransformerTrunk(self.config)
        self.policy_graph_proj = nn.Linear(self.graph_trunk.out_dim, fnn_config.embed_dim)
        self.policy_move_proj = nn.Linear(self.config.move_feat_dim, fnn_config.embed_dim)
        self.policy_fc1 = nn.Linear(
            fnn_config.embed_dim * 4,
            fnn_config.action_hidden,
        )
        self.policy_fc2 = nn.Linear(fnn_config.action_hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(
                fnn_config.embed_dim + self.graph_trunk.out_dim + self.config.global_feat_dim,
                self.config.value_hidden,
            ),
            nn.ReLU(),
            nn.Linear(self.config.value_hidden, 1),
        )

    def forward(
        self,
        root_features: torch.Tensor,
        successor_features: torch.Tensor,
        action_to_root: torch.Tensor,
        num_actions: torch.Tensor,
        piece_batch: HybridPieceTensorBatch,
        move_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transformer-aware action logits and transformer-enhanced root values."""
        root_emb = self.fnn.encode(root_features)
        graph_summary = self.graph_trunk(piece_batch)

        if action_to_root.shape[0] == 0:
            action_logits = root_features.new_zeros((0,))
        else:
            succ_emb = self.fnn.encode(successor_features)
            action_logits = self.score_hybrid_actions(
                root_emb[action_to_root],
                graph_summary[action_to_root],
                succ_emb,
                move_features,
            )

        value_in = torch.cat([root_emb, graph_summary, piece_batch.global_features], dim=1)
        values = torch.tanh(self.value_head(value_in))
        return action_logits, values

    def score_hybrid_actions(
        self,
        root_emb: torch.Tensor,
        root_graph_emb: torch.Tensor,
        successor_emb: torch.Tensor,
        move_features: torch.Tensor,
    ) -> torch.Tensor:
        graph_ctx = self.policy_graph_proj(root_graph_emb)
        move_ctx = self.policy_move_proj(move_features)
        combined = torch.cat([root_emb, graph_ctx, successor_emb, move_ctx], dim=1)
        x = torch.sigmoid(self.policy_fc1(combined))
        return self.policy_fc2(x).squeeze(-1)

    def policy_forward(
        self,
        root_features: torch.Tensor,
        successor_features: torch.Tensor,
        action_to_root: torch.Tensor,
        piece_batch: HybridPieceTensorBatch,
        move_features: torch.Tensor,
    ) -> torch.Tensor:
        root_emb = self.fnn.encode(root_features)
        if action_to_root.shape[0] == 0:
            return root_features.new_zeros((0,))
        succ_emb = self.fnn.encode(successor_features)
        graph_summary = self.graph_trunk(piece_batch)
        return self.score_hybrid_actions(
            root_emb[action_to_root],
            graph_summary[action_to_root],
            succ_emb,
            move_features,
        )

    def value_forward(
        self,
        root_features: torch.Tensor,
        piece_batch: HybridPieceTensorBatch,
    ) -> torch.Tensor:
        root_emb = self.fnn.encode(root_features)
        graph_summary = self.graph_trunk(piece_batch)
        value_in = torch.cat([root_emb, graph_summary, piece_batch.global_features], dim=1)
        return torch.tanh(self.value_head(value_in))

    def load_fnn_policy_weights(self, fnn: HiveFNN) -> None:
        """Initialize the shared FNN encoder from a trained HiveFNN.

        The hybrid policy tower is transformer-aware and is not weight-compatible
        with the FNN action tower.
        """
        self.fnn.load_state_dict(fnn.state_dict())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


HiveFNNTransformer = HiveHybridGNN
