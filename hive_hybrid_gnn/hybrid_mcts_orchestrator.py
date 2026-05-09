"""GPU-native Gumbel MCTS orchestration for the hybrid GNN.

The policy path is the same successor-conditioned FNN policy used by
``FNNMCTSOrchestrator``.  The value path evaluates root/leaf states with the
hybrid graph value trunk, using the same precomputed legal moves that produced
the FNN features so graph token mobility is not recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hive_fnn.fnn_mcts_orchestrator import FNNMCTSConfig, FNNMCTSOrchestrator
from hive_hybrid_gnn.gpu_encoder import HybridGraphGPUEncoder
from hive_hybrid_gnn.hybrid_gnn_net import HiveHybridGNN


@dataclass
class HybridMCTSConfig(FNNMCTSConfig):
    graph_radius: int = 2


class HybridMCTSOrchestrator(FNNMCTSOrchestrator):
    """FNN-policy MCTS with hybrid graph value evaluation."""

    def __init__(
        self,
        net: HiveHybridGNN,
        config: HybridMCTSConfig | None = None,
    ) -> None:
        cfg = config or HybridMCTSConfig()
        super().__init__(net, cfg)
        self.net: HiveHybridGNN = net
        self.config: HybridMCTSConfig = cfg
        self.graph_encoder = HybridGraphGPUEncoder(radius=cfg.graph_radius)

    def _eval_states(
        self,
        states: torch.Tensor,
        legal_moves: torch.Tensor,
        num_legal: torch.Tensor,
        total: int,
        root_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode states, score legal successors, and value roots.

        Child initial Q is intentionally zero.  Computing exact hybrid child
        values would require graph-encoding every legal successor state, which
        is far more expensive than the FNN successor feature pass.
        """
        dev = "cuda"
        max_legal = self._max_legal
        n64 = num_legal.to(torch.int64)

        if root_features is None:
            root_features = self.ext.extract_fnn_features_batch(
                states, legal_moves, num_legal, total,
            )

        if total == 0:
            empty = torch.zeros(0, max_legal, dtype=torch.float32, device=dev)
            return empty, empty.new_zeros((0,)), empty

        slot_idx = self._slot_idx
        valid = slot_idx < n64.unsqueeze(1)
        action_to_root = self._row_indices(total).expand_as(valid)[valid]
        move_indices = slot_idx.expand_as(valid)[valid]
        n_total = int(action_to_root.shape[0])

        graph_batch = self.graph_encoder.encode_batch(
            states,
            total,
            legal_moves=legal_moves,
            num_legal=num_legal,
        )

        with torch.inference_mode():
            root_emb = self.net.fnn.encode(root_features)
            graph_summary = self.net.graph_trunk(graph_batch)
            value_in = torch.cat(
                [root_emb, graph_summary, graph_batch.global_features],
                dim=1,
            )
            root_values = torch.tanh(self.net.value_head(value_in)).squeeze(-1).float()

            if n_total == 0:
                priors_per_legal = torch.zeros(
                    total, max_legal, dtype=torch.float32, device=dev,
                )
                child_q_per_legal = torch.zeros_like(priors_per_legal)
                return priors_per_legal, root_values, child_q_per_legal

            succ_features = self.ext.fnn_successor_features_batch(
                states, legal_moves, action_to_root, move_indices, n_total,
            )
            succ_emb = self.net.fnn.encode(succ_features)
            gathered_root = root_emb[action_to_root]
            action_logits = self.net.fnn.score_actions(
                gathered_root, succ_emb,
            ).float()

        legal_logits = torch.full(
            (total, max_legal), -1e30, dtype=torch.float32, device=dev,
        )
        legal_logits[valid] = action_logits

        priors_per_legal = torch.softmax(legal_logits, dim=1)
        priors_per_legal = priors_per_legal.masked_fill(~valid, 0.0)

        child_q_per_legal = torch.zeros(
            total, max_legal, dtype=torch.float32, device=dev,
        )
        return priors_per_legal, root_values, child_q_per_legal
