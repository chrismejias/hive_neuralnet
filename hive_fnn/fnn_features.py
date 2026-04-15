"""
HiveGo-style FNN feature extraction via CUDA kernel.

Produces 88-dim features directly from HiveState + legal moves in a single
CUDA kernel call, bypassing the full encode_states_batch pipeline.

Feature layout (FEAT_DIM = 88):
  [0:16]  count_on_board     -- visible top pieces per type(8) x color(2)
  [16:32] count_in_hand      -- hand piece counts per type(8) x color(2)
  [32:48] queen_neighbors    -- pieces adjacent to opponent queen per type(8) x color(2)
  [48:64] avg_dist_to_opp_q  -- avg hex distance to opponent queen per type(8) x color(2)
  [64:80] can_move           -- piece type has >=1 legal MOVE per type(8) x color(2)
  [80:82] num_single         -- pieces with 0 occupied neighbors per color(2)
  [82:84] queen_covered      -- queen not on top (beetle covering) per color(2)
  [84:86] num_placement_pos  -- unique placement destinations per color(2)
  [86]    moves_to_draw      -- normalized turn count
  [87]    move_number         -- turn / 100
"""

from __future__ import annotations

import torch

import hive_gpu

FEAT_DIM = 88


class FNNFeatureEncoder:
    """Encode GPU HiveStates into fixed-size board feature vectors via CUDA."""

    def __init__(self) -> None:
        self.ext = hive_gpu.load_extension()

    def encode_features(
        self,
        states: torch.Tensor,
        batch_size: int,
        legal_moves: torch.Tensor | None = None,
        num_legal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of GPU HiveStates into feature vectors.

        Args:
            states: (batch_size, SIZEOF_HIVE_STATE) uint8 GPU tensor.
            batch_size: number of games.
            legal_moves: (batch_size, MAX_LEGAL_MOVES) from generate_legal_moves_batch.
            num_legal: (batch_size,) int32 legal move counts.

        Returns:
            (batch_size, FEAT_DIM) float32 tensor on GPU.
        """
        if batch_size == 0:
            return torch.zeros((0, FEAT_DIM), dtype=torch.float32, device=states.device)

        if legal_moves is None or num_legal is None:
            legal_moves, num_legal = self.ext.generate_legal_moves_batch(
                states, batch_size,
            )

        return self.ext.extract_fnn_features_batch(
            states, legal_moves, num_legal, batch_size,
        )
