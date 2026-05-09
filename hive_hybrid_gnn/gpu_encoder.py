"""GPU graph encoder for the hybrid GNN value trunk."""

from __future__ import annotations

import torch

import hive_gpu
from hive_hybrid_gnn.graph_types import HybridGraphTensorBatch, edge_feat_dim_for_radius


class HybridGraphGPUEncoder:
    """Encode opaque GPU HiveState batches into padded graph tensors."""

    def __init__(self, radius: int = 2) -> None:
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.radius = int(radius)
        self._ext = None

    @property
    def ext(self):
        if self._ext is None:
            self._ext = hive_gpu.load_extension()
        return self._ext

    def encode_batch(
        self,
        states: torch.Tensor,
        batch_size: int | None = None,
        legal_moves: torch.Tensor | None = None,
        num_legal: torch.Tensor | None = None,
    ) -> HybridGraphTensorBatch:
        if not states.is_cuda:
            raise ValueError("HybridGraphGPUEncoder expects CUDA HiveState tensors")
        if batch_size is None:
            batch_size = int(states.shape[0])
        if legal_moves is None or num_legal is None:
            legal_moves, num_legal = self.ext.generate_legal_moves_batch(states, int(batch_size))
        (
            node_features,
            edge_src,
            edge_dst,
            edge_features,
            node_mask,
            edge_mask,
            global_features,
            num_nodes,
            num_edges,
        ) = self.ext.hybrid_gnn_encode_with_moves_batch(
            states,
            legal_moves,
            num_legal,
            int(batch_size),
            self.radius,
        )
        edge_features = edge_features[..., :edge_feat_dim_for_radius(self.radius)]
        return HybridGraphTensorBatch(
            node_features=node_features,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_features=edge_features,
            node_mask=node_mask,
            edge_mask=edge_mask,
            global_features=global_features,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )
