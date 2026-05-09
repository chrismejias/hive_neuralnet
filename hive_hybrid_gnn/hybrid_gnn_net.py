"""Hybrid GNN + FNN network.

Policy stays exactly in the FNN style: encode root and successor feature
vectors, then score each successor relative to the root. Value gets additional
full-board graph context through a small message-passing trunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_fnn.fnn_features import FEAT_DIM
from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_hybrid_gnn.graph_types import (
    GLOBAL_FEAT_DIM,
    NODE_FEAT_DIM,
    HybridGraphBatch,
    HybridGraphTensorBatch,
    edge_feat_dim_for_radius,
)


@dataclass
class HybridGNNConfig:
    fnn_config: FNNConfig | None = None
    graph_hidden_dim: int = 64
    graph_layers: int = 3
    graph_radius: int = 2
    graph_mlp_hidden: int = 64
    global_pool_bias: bool = True
    value_hidden: int = 128
    node_feat_dim: int = NODE_FEAT_DIM
    global_feat_dim: int = GLOBAL_FEAT_DIM

    @classmethod
    def small(cls) -> "HybridGNNConfig":
        return cls(
            fnn_config=FNNConfig.medium(),
            graph_hidden_dim=48,
            graph_layers=3,
            graph_radius=2,
            graph_mlp_hidden=48,
            value_hidden=96,
        )

    @classmethod
    def large(cls) -> "HybridGNNConfig":
        return cls(
            fnn_config=FNNConfig.large(),
            graph_hidden_dim=80,
            graph_layers=5,
            graph_radius=2,
            graph_mlp_hidden=80,
            value_hidden=160,
        )

    @property
    def edge_feat_dim(self) -> int:
        return edge_feat_dim_for_radius(self.graph_radius)


class HybridMessagePassingLayer(nn.Module):
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

    def forward_padded(
        self,
        node_h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = node_h.size(-1)
        src_h = torch.gather(
            node_h,
            1,
            edge_src.unsqueeze(-1).expand(-1, -1, hidden),
        )
        dst_h = torch.gather(
            node_h,
            1,
            edge_dst.unsqueeze(-1).expand(-1, -1, hidden),
        )
        msg_in = torch.cat([dst_h, src_h, edge_features], dim=-1)
        messages = self.message_mlp(msg_in)
        messages = messages.to(node_h.dtype)
        messages = messages * edge_mask.unsqueeze(-1).to(messages.dtype)

        aggregated = torch.zeros_like(node_h)
        aggregated.scatter_add_(
            1,
            edge_dst.unsqueeze(-1).expand(-1, -1, hidden),
            messages,
        )
        update = self.update_mlp(torch.cat([node_h, aggregated], dim=-1))
        out = self.norm(node_h + update)
        out = out * node_mask.unsqueeze(-1).to(out.dtype)

        if self.global_pool_bias:
            pooled = _mean_max_pool_padded(out, node_mask)
            out = out + self.global_proj(pooled).unsqueeze(1)
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out


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


def _mean_max_pool_padded(node_h: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    mask = node_mask.unsqueeze(-1).to(node_h.dtype)
    counts = mask.sum(dim=1).clamp_min(1.0)
    mean_pool = (node_h * mask).sum(dim=1) / counts
    masked_h = node_h.masked_fill(~node_mask.unsqueeze(-1), -torch.inf)
    max_pool = masked_h.max(dim=1).values
    max_pool = max_pool.masked_fill(max_pool == -torch.inf, 0.0)
    return torch.cat([mean_pool, max_pool], dim=1)


class HybridGraphValueTrunk(nn.Module):
    def __init__(self, config: HybridGNNConfig) -> None:
        super().__init__()
        h = config.graph_hidden_dim
        self.node_proj = nn.Linear(config.node_feat_dim, h)
        self.layers = nn.ModuleList([
            HybridMessagePassingLayer(
                h,
                config.edge_feat_dim,
                config.graph_mlp_hidden,
                config.global_pool_bias,
            )
            for _ in range(config.graph_layers)
        ])
        self.out_dim = h * 2

    def forward(self, graph: HybridGraphBatch | HybridGraphTensorBatch) -> torch.Tensor:
        if isinstance(graph, HybridGraphTensorBatch):
            h = F.relu(self.node_proj(graph.node_features))
            h = h * graph.node_mask.unsqueeze(-1).to(h.dtype)
            for layer in self.layers:
                h = layer.forward_padded(
                    h,
                    graph.edge_src,
                    graph.edge_dst,
                    graph.edge_features,
                    graph.node_mask,
                    graph.edge_mask,
                )
            return _mean_max_pool_padded(h, graph.node_mask)

        h = F.relu(self.node_proj(graph.node_features))
        batch_size = graph.global_features.size(0)
        for layer in self.layers:
            h = layer(
                h,
                graph.edge_index,
                graph.edge_features,
                graph.batch,
                batch_size,
            )
        return _mean_max_pool(h, graph.batch, batch_size)


class HiveHybridGNN(nn.Module):
    """FNN policy plus graph-enhanced value network."""

    model_version = "hybrid_gnn"

    def __init__(self, config: HybridGNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridGNNConfig()
        fnn_config = self.config.fnn_config or FNNConfig.large()
        if fnn_config.feat_dim != FEAT_DIM:
            raise ValueError(f"FNN feature dim must be {FEAT_DIM}, got {fnn_config.feat_dim}")

        self.fnn = HiveFNN(fnn_config)
        self.graph_trunk = HybridGraphValueTrunk(self.config)
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
        graph_batch: HybridGraphBatch | HybridGraphTensorBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FNN-style action logits and graph-enhanced root values."""
        root_emb = self.fnn.encode(root_features)

        if action_to_root.shape[0] == 0:
            action_logits = root_features.new_zeros((0,))
        else:
            succ_emb = self.fnn.encode(successor_features)
            action_logits = self.fnn.score_actions(root_emb[action_to_root], succ_emb)

        graph_summary = self.graph_trunk(graph_batch)
        value_in = torch.cat([root_emb, graph_summary, graph_batch.global_features], dim=1)
        values = torch.tanh(self.value_head(value_in))
        return action_logits, values

    def policy_forward(
        self,
        root_features: torch.Tensor,
        successor_features: torch.Tensor,
        action_to_root: torch.Tensor,
    ) -> torch.Tensor:
        root_emb = self.fnn.encode(root_features)
        if action_to_root.shape[0] == 0:
            return root_features.new_zeros((0,))
        succ_emb = self.fnn.encode(successor_features)
        return self.fnn.score_actions(root_emb[action_to_root], succ_emb)

    def value_forward(
        self,
        root_features: torch.Tensor,
        graph_batch: HybridGraphBatch | HybridGraphTensorBatch,
    ) -> torch.Tensor:
        root_emb = self.fnn.encode(root_features)
        graph_summary = self.graph_trunk(graph_batch)
        value_in = torch.cat([root_emb, graph_summary, graph_batch.global_features], dim=1)
        return torch.tanh(self.value_head(value_in))

    def load_fnn_policy_weights(self, fnn: HiveFNN) -> None:
        """Initialize the policy-side FNN from a trained HiveFNN."""
        self.fnn.load_state_dict(fnn.state_dict())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
