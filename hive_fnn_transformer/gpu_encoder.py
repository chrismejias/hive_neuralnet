"""GPU piece-token encoder for the hybrid FNN transformer trunk."""

from __future__ import annotations

import torch

import hive_gpu
from hive_fnn_transformer.graph_types import HybridPieceTensorBatch


class HybridTransformerGPUEncoder:
    """Encode opaque GPU HiveState batches into padded piece-token tensors."""

    def __init__(self) -> None:
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
    ) -> HybridPieceTensorBatch:
        if not states.is_cuda:
            raise ValueError("HybridTransformerGPUEncoder expects CUDA HiveState tensors")
        if batch_size is None:
            batch_size = int(states.shape[0])
        if legal_moves is None or num_legal is None:
            legal_moves, num_legal = self.ext.generate_legal_moves_batch(states, int(batch_size))
        (
            token_features,
            token_q,
            token_r,
            token_z,
            token_mask,
            global_features,
            num_tokens,
        ) = self.ext.hybrid_transformer_encode_with_moves_batch(
            states,
            legal_moves,
            num_legal,
            int(batch_size),
        )
        return HybridPieceTensorBatch(
            token_features=token_features,
            token_q=token_q,
            token_r=token_r,
            token_z=token_z,
            token_mask=token_mask,
            global_features=global_features,
            num_tokens=num_tokens,
        )


# Backward-compatible alias for older imports.
HybridGraphGPUEncoder = HybridTransformerGPUEncoder
