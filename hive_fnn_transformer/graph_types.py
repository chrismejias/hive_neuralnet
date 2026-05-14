"""Piece-token data structures for the hybrid FNN transformer trunk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

NODE_FEAT_DIM = 25
GLOBAL_FEAT_DIM = 6
MAX_PIECE_TOKENS = 28


@dataclass
class HybridPieceTensorBatch:
    """Padded piece-token tensors produced by the CUDA encoder."""

    token_features: torch.Tensor
    token_q: torch.Tensor
    token_r: torch.Tensor
    token_z: torch.Tensor
    token_mask: torch.Tensor
    global_features: torch.Tensor
    num_tokens: torch.Tensor

    def to(self, device: torch.device | str) -> "HybridPieceTensorBatch":
        return HybridPieceTensorBatch(
            token_features=self.token_features.to(device),
            token_q=self.token_q.to(device),
            token_r=self.token_r.to(device),
            token_z=self.token_z.to(device),
            token_mask=self.token_mask.to(device),
            global_features=self.global_features.to(device),
            num_tokens=self.num_tokens.to(device),
        )


def edge_feat_dim_for_radius(radius: int) -> int:
    """Legacy helper retained for archived graph diagnostics."""
    if radius < 1:
        raise ValueError("radius must be >= 1")
    return 3


@dataclass
class HybridGraph:
    """Legacy sparse-graph structure retained for archived diagnostics."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    global_features: np.ndarray
    num_piece_nodes: int
    node_positions: np.ndarray
    node_piece_types: np.ndarray


@dataclass
class HybridGraphBatch:
    """Legacy collated sparse-graph batch retained for archived diagnostics."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    global_features: torch.Tensor
    num_piece_nodes: int
    node_positions: torch.Tensor
    node_piece_types: torch.Tensor
    batch: torch.Tensor
    piece_node_batch: torch.Tensor

    @staticmethod
    def collate(graphs: list[HybridGraph]) -> "HybridGraphBatch":
        node_features_list: list[torch.Tensor] = []
        edge_index_list: list[torch.Tensor] = []
        edge_features_list: list[torch.Tensor] = []
        global_features_list: list[torch.Tensor] = []
        node_positions_list: list[torch.Tensor] = []
        node_piece_types_list: list[torch.Tensor] = []
        batch_list: list[torch.Tensor] = []
        piece_node_batch_list: list[torch.Tensor] = []

        cumulative_nodes = 0
        for idx, graph in enumerate(graphs):
            num_nodes = graph.node_features.shape[0]
            node_features_list.append(torch.from_numpy(graph.node_features))

            edge_index = torch.from_numpy(graph.edge_index)
            if edge_index.numel() > 0:
                edge_index = edge_index + cumulative_nodes
            edge_index_list.append(edge_index)
            edge_features_list.append(torch.from_numpy(graph.edge_features))
            global_features_list.append(torch.from_numpy(graph.global_features))
            node_positions_list.append(torch.from_numpy(graph.node_positions))
            node_piece_types_list.append(torch.from_numpy(graph.node_piece_types))
            batch_list.append(torch.full((num_nodes,), idx, dtype=torch.int64))
            piece_node_batch_list.append(
                torch.full((graph.num_piece_nodes,), idx, dtype=torch.int64)
            )
            cumulative_nodes += num_nodes

        node_features = torch.cat(node_features_list, dim=0)
        edge_index = (
            torch.cat([ei for ei in edge_index_list if ei.numel() > 0], dim=1)
            if any(ei.numel() > 0 for ei in edge_index_list)
            else torch.zeros((2, 0), dtype=torch.int64)
        )
        edge_features = (
            torch.cat([ef for ef in edge_features_list if ef.shape[0] > 0], dim=0)
            if any(ef.shape[0] > 0 for ef in edge_features_list)
            else torch.zeros((0, graphs[0].edge_features.shape[1]), dtype=torch.float32)
        )
        node_positions = (
            torch.cat([p for p in node_positions_list if p.shape[0] > 0], dim=0)
            if any(p.shape[0] > 0 for p in node_positions_list)
            else torch.zeros((0, 2), dtype=torch.int32)
        )
        node_piece_types = (
            torch.cat([pt for pt in node_piece_types_list if pt.shape[0] > 0], dim=0)
            if any(pt.shape[0] > 0 for pt in node_piece_types_list)
            else torch.zeros((0,), dtype=torch.int32)
        )
        piece_node_batch = (
            torch.cat([pb for pb in piece_node_batch_list if pb.shape[0] > 0], dim=0)
            if any(pb.shape[0] > 0 for pb in piece_node_batch_list)
            else torch.zeros((0,), dtype=torch.int64)
        )

        return HybridGraphBatch(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=torch.stack(global_features_list, dim=0),
            num_piece_nodes=sum(g.num_piece_nodes for g in graphs),
            node_positions=node_positions,
            node_piece_types=node_piece_types,
            batch=torch.cat(batch_list, dim=0),
            piece_node_batch=piece_node_batch,
        )

    def to(self, device: torch.device | str) -> "HybridGraphBatch":
        return HybridGraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            global_features=self.global_features.to(device),
            num_piece_nodes=self.num_piece_nodes,
            node_positions=self.node_positions.to(device),
            node_piece_types=self.node_piece_types.to(device),
            batch=self.batch.to(device),
            piece_node_batch=self.piece_node_batch.to(device),
        )
