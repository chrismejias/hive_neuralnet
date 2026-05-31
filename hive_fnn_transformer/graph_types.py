"""Piece-token data structures for the active hybrid FNN transformer trunk."""

from __future__ import annotations

from dataclasses import dataclass

import torch

BASE_NODE_FEAT_DIM = 26
ARTICULATION_NODE_FEAT_DIM = 27
NODE_FEAT_DIM = BASE_NODE_FEAT_DIM
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
