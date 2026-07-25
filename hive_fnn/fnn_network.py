"""
HiveGo-style feedforward neural network for Hive.

Architecture mirrors HiveGo's AlphaZeroFNN:
  - Shared board encoder: features -> hidden -> embedding (sigmoid, LayerNorm)
  - Value head: embedding -> Linear -> tanh
  - Action scoring: concat [root_emb, successor_emb] -> tower -> logit

The same encoder is used for both root and successor states. Policy is
computed by encoding ALL legal successor states and scoring each one
against the root embedding -- no action-space enumeration needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hive_fnn.fnn_features import FEAT_DIM


_WHITE_MOVE_SLICE = slice(64, 72)
_BLACK_MOVE_SLICE = slice(72, 80)
_WHITE_PLACE_IDX = 100
_BLACK_PLACE_IDX = 101
_WHITE_SURROUND_SLICE = slice(110, 116)
_BLACK_SURROUND_SLICE = slice(116, 122)
_ACTION_SURROUND_ABS_DIM = 24


@dataclass
class FNNConfig:
    feat_dim: int = 140  # FNN_FEAT_DIM from CUDA kernel
    hidden_dim: int = 64
    embed_dim: int = 64
    action_hidden: int = 64

    @classmethod
    def small(cls) -> FNNConfig:
        """~600 params, closest to HiveGo's 454-param model."""
        return cls(hidden_dim=8, embed_dim=8, action_hidden=8)

    @classmethod
    def medium(cls) -> FNNConfig:
        """~5k params."""
        return cls(hidden_dim=32, embed_dim=32, action_hidden=32)

    @classmethod
    def large(cls) -> FNNConfig:
        """~15k params."""
        return cls(hidden_dim=64, embed_dim=64, action_hidden=64)


class HiveFNN(nn.Module):
    """HiveGo-style tiny feedforward network with successor-state scoring.

    Self-play:  encode root + ALL successors -> score actions -> Gumbel search
    Training:   encode root + ALL successors -> policy loss + value loss
    """

    def __init__(self, config: FNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or FNNConfig.large()
        if self.config.feat_dim != FEAT_DIM:
            self.config.feat_dim = FEAT_DIM
        c = self.config

        # ---- Shared board encoder (root & successor) ----
        self.fc1 = nn.Linear(c.feat_dim, c.hidden_dim)
        self.ln1 = nn.LayerNorm(c.hidden_dim)
        self.fc2 = nn.Linear(c.hidden_dim, c.embed_dim)

        # ---- Value head ----
        self.value_fc = nn.Linear(c.embed_dim, 1)

        # ---- Action tower (scores a successor relative to root) ----
        self.action_fc1 = nn.Linear(c.embed_dim * 2 + _ACTION_SURROUND_ABS_DIM, c.action_hidden)
        self.action_fc2 = nn.Linear(c.action_hidden, 1)

    @staticmethod
    def _pad_loaded_weight(
        old: torch.Tensor,
        new_template: torch.Tensor,
        *,
        noise_scale: float = 1e-3,
    ) -> torch.Tensor:
        new = torch.empty_like(new_template)
        new.normal_(mean=0.0, std=noise_scale)
        rows = min(old.shape[0], new.shape[0])
        cols = min(old.shape[1], new.shape[1])
        # The old 124-input layout ended with two own-queen-throwable bits.
        # They were removed rather than reinterpreted as the first new feature
        # buckets, so migrate only the 122 unchanged columns.
        if old.shape[1] == 124 and new.shape[1] == 140:
            cols = 122
        new[:rows, :cols] = old[:rows, :cols]
        return new

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state = dict(state_dict)
        cur_state = self.state_dict()
        for key in ("fc1.weight", "action_fc1.weight"):
            if key in state and key in cur_state and state[key].shape != cur_state[key].shape:
                old = state[key]
                state[key] = self._pad_loaded_weight(old, cur_state[key])
        return super().load_state_dict(state, strict=strict, assign=assign)

    @staticmethod
    def _infer_current_player_is_white(features: torch.Tensor) -> torch.Tensor:
        white_activity = (
            features[:, _WHITE_MOVE_SLICE].sum(dim=1) + features[:, _WHITE_PLACE_IDX]
        )
        black_activity = (
            features[:, _BLACK_MOVE_SLICE].sum(dim=1) + features[:, _BLACK_PLACE_IDX]
        )
        return white_activity >= black_activity

    def _action_surround_features(
        self,
        root_features: torch.Tensor,
        successor_features: torch.Tensor,
    ) -> torch.Tensor:
        root_is_white = self._infer_current_player_is_white(root_features)
        succ_is_white = self._infer_current_player_is_white(successor_features)
        ambiguous = (
            (root_features[:, _WHITE_MOVE_SLICE].sum(dim=1) + root_features[:, _WHITE_PLACE_IDX])
            == (root_features[:, _BLACK_MOVE_SLICE].sum(dim=1) + root_features[:, _BLACK_PLACE_IDX])
        )
        root_is_white = torch.where(ambiguous, ~succ_is_white, root_is_white)

        root_own = torch.where(
            root_is_white.unsqueeze(1),
            root_features[:, _WHITE_SURROUND_SLICE],
            root_features[:, _BLACK_SURROUND_SLICE],
        )
        root_opp = torch.where(
            root_is_white.unsqueeze(1),
            root_features[:, _BLACK_SURROUND_SLICE],
            root_features[:, _WHITE_SURROUND_SLICE],
        )
        succ_own = torch.where(
            root_is_white.unsqueeze(1),
            successor_features[:, _WHITE_SURROUND_SLICE],
            successor_features[:, _BLACK_SURROUND_SLICE],
        )
        succ_opp = torch.where(
            root_is_white.unsqueeze(1),
            successor_features[:, _BLACK_SURROUND_SLICE],
            successor_features[:, _WHITE_SURROUND_SLICE],
        )
        return torch.cat([root_own, root_opp, succ_own, succ_opp], dim=1)

    # ---- Shared encoder ----

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode board features into an embedding.

        Args:
            features: (N, feat_dim)

        Returns:
            (N, embed_dim)
        """
        x = torch.sigmoid(self.fc1(features))
        x = self.ln1(x)
        x = self.fc2(x)
        return x

    # ---- Value head ----

    def value_head(self, embed: torch.Tensor) -> torch.Tensor:
        """Predict state value from embedding.

        Args:
            embed: (B, embed_dim)

        Returns:
            (B, 1) in [-1, 1]
        """
        return torch.tanh(self.value_fc(embed))

    # ---- Action scoring ----

    def score_actions(
        self,
        root_embed: torch.Tensor,
        successor_embed: torch.Tensor,
        root_features: torch.Tensor,
        successor_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score successor states relative to the root.

        Args:
            root_embed: (N_actions, embed_dim) -- gathered per action
            successor_embed: (N_actions, embed_dim)

        Returns:
            (N_actions,) logits
        """
        surround_features = self._action_surround_features(
            root_features,
            successor_features,
        )
        combined = torch.cat([root_embed, successor_embed, surround_features], dim=1)
        x = torch.sigmoid(self.action_fc1(combined))
        return self.action_fc2(x).squeeze(-1)

    # ---- Full forward (training) ----

    def forward(
        self,
        root_features: torch.Tensor,
        successor_features: torch.Tensor,
        action_to_root: torch.Tensor,
        num_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full training forward pass.

        Args:
            root_features: (B, feat_dim)
            successor_features: (N_total, feat_dim)
            action_to_root: (N_total,) int64 mapping each action to its root
            num_actions: (B,) int64 legal-move count per root

        Returns:
            action_logits: (N_total,) from action tower
            root_values: (B, 1) from value head
        """
        root_emb = self.encode(root_features)               # (B, embed_dim)
        root_values = self.value_head(root_emb)              # (B, 1)

        if action_to_root.shape[0] == 0:
            empty = root_features.new_zeros((0,))
            return empty, root_values

        succ_emb = self.encode(successor_features)           # (N, embed_dim)
        gathered_root = root_emb[action_to_root]             # (N, embed_dim)
        gathered_root_features = root_features[action_to_root]
        action_logits = self.score_actions(
            gathered_root,
            succ_emb,
            gathered_root_features,
            successor_features,
        )  # (N,)
        return action_logits, root_values

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
