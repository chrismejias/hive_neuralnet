"""
NNUE-style MLP network for Hive.

Architecture:
    Feature vector (428) → [FC + ReLU] × N → Split:
        ├── Policy head → FC → logits (29407,)
        └── Value head  → FC → tanh → scalar in [-1, 1]

Compared to the CNN (conv towers) and GNN (message passing), this
architecture is radically simpler: just fully-connected layers on a
hand-crafted feature vector. The quality of the features — especially
the queen-relative distance buckets — is what allows the network to
learn strategic positional evaluation.

The network is small and fast, making it well-suited for rapid MCTS
rollouts where inference speed matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_engine.encoder import HiveEncoder

from hive_nnue.nnue_encoder import FEATURE_DIM


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class NNUEConfig:
    """Configuration for NNUE network architecture."""

    feature_dim: int = FEATURE_DIM                                  # 428
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    action_space_size: int = HiveEncoder.ACTION_SPACE_SIZE          # 29407

    @classmethod
    def small(cls) -> NNUEConfig:
        """Small network (~8M params, dominated by policy head)."""
        return cls(hidden_dims=[512, 256])

    @classmethod
    def large(cls) -> NNUEConfig:
        """Larger network with deeper MLP (~12M params)."""
        return cls(hidden_dims=[1024, 512, 256])


# ── NNUE Network ──────────────────────────────────────────────────


class HiveNNUE(nn.Module):
    """
    NNUE-style MLP for Hive with AlphaZero dual heads.

    Takes a flat feature vector (428 dimensions) and produces:
        - Policy logits over the 29,407-action space
        - Value scalar in [-1, 1]

    The backbone is a simple MLP with configurable hidden layers.
    The policy and value heads branch off the final hidden layer.

    Input:
        x: Tensor of shape (batch, feature_dim) or a batched dict
           containing 'features' key.

    Output:
        (policy_logits, value) where:
            policy_logits: shape (batch, 29407)
            value: shape (batch, 1), in [-1, 1]
    """

    def __init__(self, config: NNUEConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = NNUEConfig()
        self.config = config

        # ── Build backbone MLP ────────────────────────────────
        layers: list[nn.Module] = []
        in_dim = config.feature_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        backbone_out = config.hidden_dims[-1]  # last hidden dim

        # ── Policy head ───────────────────────────────────────
        # Extra FC layer to bridge from backbone to large action space
        self.policy_fc1 = nn.Linear(backbone_out, 256)
        self.policy_fc2 = nn.Linear(256, config.action_space_size)

        # ── Value head ────────────────────────────────────────
        self.value_fc1 = nn.Linear(backbone_out, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Feature tensor of shape (batch, feature_dim).

        Returns:
            (policy_logits, value) where:
                policy_logits: shape (batch, 29407)
                value: shape (batch, 1), in [-1, 1]
        """
        # Backbone
        h = self.backbone(x)  # (batch, backbone_out)

        # Policy head
        p = F.relu(self.policy_fc1(h))
        policy_logits = self.policy_fc2(p)  # (batch, 29407)

        # Value head
        v = F.relu(self.value_fc1(h))
        value = torch.tanh(self.value_fc2(v))  # (batch, 1)

        return policy_logits, value

    @torch.no_grad()
    def predict(
        self,
        feature_vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Single-state inference for MCTS.

        Takes a numpy feature vector and legal mask, runs forward in
        eval mode, applies the legal mask, and returns probabilities
        and value.

        Args:
            feature_vector: Shape (feature_dim,), float32.
            legal_mask: Shape (29407,), float32. 1.0 for legal actions.

        Returns:
            (action_probs, value) where:
                action_probs: Shape (29407,), sums to ~1.0 over legal.
                value: Scalar in [-1, 1].
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        # Prepare input: (1, feature_dim)
        x = torch.from_numpy(feature_vector).unsqueeze(0).to(device)
        mask = torch.from_numpy(legal_mask).to(device)

        policy_logits, value_tensor = self.forward(x)

        # Apply legal mask
        policy_logits = policy_logits.squeeze(0)  # (29407,)
        policy_logits = policy_logits.masked_fill(mask == 0, float("-inf"))

        if mask.sum() > 0:
            action_probs = F.softmax(policy_logits, dim=0)
        else:
            action_probs = torch.zeros_like(policy_logits)

        if was_training:
            self.train()

        return (
            action_probs.cpu().numpy(),
            value_tensor.item(),
        )

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
