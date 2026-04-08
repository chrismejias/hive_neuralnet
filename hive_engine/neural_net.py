"""
AlphaZero-style dual-headed residual neural network for Hive.

Architecture:
    Input (26, 13, 13) → Conv stem → [ResidualBlock × N] →
        ├── Policy head → logits (29407,)
        └── Value head  → scalar in [-1, 1]

The policy head outputs raw logits over the full action space.
Legal move masking + softmax is applied in predict() for MCTS.
The value head outputs a tanh-activated scalar estimating the
expected outcome from the current player's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_engine.encoder import HiveEncoder


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class NetConfig:
    """Configuration for HiveNet architecture."""

    num_blocks: int = 5
    num_filters: int = 128
    input_channels: int = HiveEncoder.NUM_CHANNELS  # 26
    action_space_size: int = HiveEncoder.ACTION_SPACE_SIZE  # 29407
    board_size: int = HiveEncoder.BOARD_SIZE  # 13

    @classmethod
    def small(cls) -> NetConfig:
        """Small network preset (~1.6M parameters)."""
        return cls(num_blocks=5, num_filters=128)

    @classmethod
    def large(cls) -> NetConfig:
        """Large network preset (~10M parameters)."""
        return cls(num_blocks=10, num_filters=256)


# ── Residual Block ─────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """
    Standard residual block: two convolutions with batch norm and skip connection.

    Architecture:
        x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+x) → ReLU
    """

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


# ── HiveNet ────────────────────────────────────────────────────────


class HiveNet(nn.Module):
    """
    AlphaZero-style dual-headed network for Hive.

    Inputs:
        x: Tensor of shape (batch, 26, 13, 13)

    Outputs:
        policy_logits: Tensor of shape (batch, 29407) — raw logits
        value: Tensor of shape (batch, 1) — in [-1, 1]
    """

    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = NetConfig()
        self.config = config

        f = config.num_filters
        bs = config.board_size
        grid_cells = bs * bs  # 169

        # ── Convolutional Stem ──
        self.conv_stem = nn.Conv2d(
            config.input_channels, f, kernel_size=3, padding=1, bias=False
        )
        self.bn_stem = nn.BatchNorm2d(f)

        # ── Residual Tower ──
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(f) for _ in range(config.num_blocks)]
        )

        # ── Policy Head ──
        # Use 2 conv channels to keep parameter count manageable
        # with our large action space (29407).
        self.policy_conv = nn.Conv2d(f, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        policy_flatten_size = 2 * grid_cells  # 338
        self.policy_fc1 = nn.Linear(policy_flatten_size, 256)
        self.policy_fc2 = nn.Linear(256, config.action_space_size)

        # ── Value Head ──
        self.value_conv = nn.Conv2d(f, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(grid_cells, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 26, 13, 13).

        Returns:
            (policy_logits, value) where:
                policy_logits: shape (batch, 29407)
                value: shape (batch, 1), in [-1, 1]
        """
        # Stem
        s = F.relu(self.bn_stem(self.conv_stem(x)))

        # Residual tower
        s = self.res_blocks(s)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(p.size(0), -1)  # (batch, 2 * 169 = 338)
        p = F.relu(self.policy_fc1(p))  # (batch, 256)
        policy_logits = self.policy_fc2(p)  # (batch, 29407)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1)  # (batch, 169)
        v = F.relu(self.value_fc1(v))  # (batch, 256)
        value = torch.tanh(self.value_fc2(v))  # (batch, 1)

        return policy_logits, value

    @torch.no_grad()
    def predict(
        self,
        state_tensor: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Single-state inference for MCTS.

        Takes a numpy state tensor and legal mask, runs the network in
        eval mode with no gradient tracking, applies the legal mask to
        the policy logits, and returns a probability distribution over
        actions plus a scalar value estimate.

        Args:
            state_tensor: Shape (26, 13, 13), float32.
            legal_mask: Shape (29407,), float32. 1.0 for legal actions.

        Returns:
            (action_probs, value) where:
                action_probs: Shape (29407,), sums to ~1.0 over legal actions.
                value: Scalar in [-1, 1].
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        # Prepare input: (1, 26, 13, 13)
        x = torch.from_numpy(state_tensor).unsqueeze(0).to(device)
        mask = torch.from_numpy(legal_mask).to(device)

        policy_logits, value_tensor = self.forward(x)

        # Apply legal mask: set illegal actions to -inf
        policy_logits = policy_logits.squeeze(0)  # (29407,)
        policy_logits = policy_logits.masked_fill(mask == 0, float("-inf"))

        # Softmax to get probabilities
        if mask.sum() > 0:
            action_probs = F.softmax(policy_logits, dim=0)
        else:
            # Edge case: no legal moves (shouldn't happen in practice)
            action_probs = torch.zeros_like(policy_logits)

        # Restore training mode if needed
        if was_training:
            self.train()

        return (
            action_probs.cpu().numpy(),
            value_tensor.item(),
        )

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss Function ──────────────────────────────────────────────────


def compute_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the AlphaZero combined loss.

    Args:
        policy_logits: Raw logits, shape (batch, 29407).
        value_pred: Predicted values, shape (batch, 1).
        target_policy: Target policy distribution, shape (batch, 29407).
            Should sum to ~1.0 per sample (MCTS visit distribution).
        target_value: Target values, shape (batch, 1). In [-1, 1].

    Returns:
        (total_loss, policy_loss, value_loss) where all are scalar tensors.
        total_loss = policy_loss + value_loss
    """
    # Policy loss: cross-entropy masked to legal actions only.
    legal_mask = target_policy > 0
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1).masked_fill(~legal_mask, 0.0)
    policy_loss = -torch.mean(torch.sum(target_policy * log_probs, dim=1))

    # Value loss: mean squared error
    value_loss = F.mse_loss(value_pred.squeeze(-1), target_value.squeeze(-1))

    total_loss = policy_loss + value_loss

    return total_loss, policy_loss, value_loss


# ── GNN Loss with Auxiliary Heads ──────────────────────────────────


def compute_gnn_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    aux_outputs: dict[str, torch.Tensor],
    mobility_targets: torch.Tensor,
    queen_surround_targets: torch.Tensor,
    queen_surround_mask: torch.Tensor,
    final_mobility_targets: torch.Tensor,
    value_mask: torch.Tensor,
    piece_node_batch: torch.Tensor,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    mobility_weight: float = 0.15,
    queen_surround_weight: float = 0.15,
    final_mobility_weight: float = 0.15,
    policy_softening: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the combined loss with auxiliary heads and playout cap masking.

    Args:
        policy_logits: Raw logits, shape (B, 29407).
        value_pred: Predicted values, shape (B, 1).
        target_policy: Target policy distribution, shape (B, 29407).
        target_value: Target values, shape (B, 1). In [-1, 1].
        aux_outputs: Dict from HiveGNN.forward() with optional keys
            ``mobility_logits``, ``queen_surround_logits``, and
            ``final_mobility_logits``.
        mobility_targets: Per-piece binary targets, shape (total_piece_nodes,).
        queen_surround_targets: Per-piece queen adjacency targets,
            shape (total_piece_nodes, 2).
        queen_surround_mask: Per-graph mask indicating which queens are
            on the board, shape (B, 2). 1.0 if placed, 0.0 otherwise.
        final_mobility_targets: Per-piece binary targets for end-of-game
            mobility, shape (total_piece_nodes,).
        value_mask: Per-graph mask for playout cap, shape (B,). 1.0 for
            full-playout games that contribute to value loss.
        piece_node_batch: Graph index for each piece node,
            shape (total_piece_nodes,).
        policy_weight: Weight for policy loss.
        value_weight: Weight for value loss.
        mobility_weight: Weight for mobility auxiliary loss.
        queen_surround_weight: Weight for queen surround auxiliary loss.
        final_mobility_weight: Weight for final mobility auxiliary loss.

    Returns:
        (total_loss, loss_dict) where loss_dict maps loss names to scalars.
    """
    # Policy softening: mix MCTS target with uniform over legal moves.
    # target_policy has epsilon on all legal actions (not just Gumbel-visited),
    # so (target_policy > 0) correctly identifies the full legal action set.
    legal_mask = target_policy > 0
    if policy_softening > 0.0:
        legal = legal_mask.float()
        num_legal = legal.sum(dim=1, keepdim=True).clamp(min=1)
        uniform = legal / num_legal
        target_policy = (1.0 - policy_softening) * target_policy + policy_softening * uniform

    # Policy loss: cross-entropy masked to legal actions only.
    # Masking logits to -inf for illegal actions before log_softmax concentrates
    # the entire gradient signal on legal moves, eliminating the ~52% waste on
    # illegal action suppression that occurs with an unmasked softmax.
    # We zero out log_probs for illegal positions after log_softmax to avoid
    # 0 * -inf = NaN (target_policy is 0 there, so the zero is mathematically
    # correct and the gradient contribution is also zero).
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1).masked_fill(~legal_mask, 0.0)
    policy_loss = -torch.mean(torch.sum(target_policy * log_probs, dim=1))

    # Value loss: Gaussian NLL if uncertainty head present, else MSE; both masked by playout cap
    if "value_logvar" in aux_outputs:
        log_var = aux_outputs["value_logvar"].squeeze(-1)  # (B,)
        var = torch.exp(log_var)
        diff_sq = (value_pred.squeeze(-1) - target_value.squeeze(-1)) ** 2
        per_sample = 0.5 * diff_sq / var + 0.5 * log_var
        if value_mask.sum() > 0:
            value_loss = (per_sample * value_mask).sum() / value_mask.sum()
        else:
            value_loss = torch.tensor(0.0, device=policy_logits.device)
    else:
        value_diff = (value_pred.squeeze(-1) - target_value.squeeze(-1)) ** 2
        if value_mask.sum() > 0:
            value_loss = (value_diff * value_mask).sum() / value_mask.sum()
        else:
            value_loss = torch.tensor(0.0, device=policy_logits.device)

    loss_dict: dict[str, torch.Tensor] = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }

    total = policy_weight * policy_loss + value_weight * value_loss

    # Mobility auxiliary loss (current-state mobility)
    if "mobility_logits" in aux_outputs and mobility_targets.numel() > 0:
        mob_logits = aux_outputs["mobility_logits"].squeeze(-1)  # (total_piece_nodes,)
        mob_loss = F.binary_cross_entropy_with_logits(mob_logits, mobility_targets)
        loss_dict["mobility_loss"] = mob_loss
        total = total + mobility_weight * mob_loss

    # Queen surround auxiliary loss
    if "queen_surround_logits" in aux_outputs and queen_surround_targets.numel() > 0:
        qs_logits = aux_outputs["queen_surround_logits"]  # (total_piece_nodes, 2)

        # Expand queen_surround_mask from (B, 2) to (total_piece_nodes, 2)
        per_node_mask = queen_surround_mask[piece_node_batch]  # (total_piece_nodes, 2)

        # Binary cross-entropy with masking
        qs_bce = F.binary_cross_entropy_with_logits(
            qs_logits, queen_surround_targets, reduction="none"
        )  # (total_piece_nodes, 2)
        qs_bce = qs_bce * per_node_mask

        if per_node_mask.sum() > 0:
            qs_loss = qs_bce.sum() / per_node_mask.sum()
        else:
            qs_loss = torch.tensor(0.0, device=policy_logits.device)

        loss_dict["queen_surround_loss"] = qs_loss
        total = total + queen_surround_weight * qs_loss

    # Final mobility auxiliary loss (end-of-game mobility)
    if "final_mobility_logits" in aux_outputs and final_mobility_targets.numel() > 0:
        fm_logits = aux_outputs["final_mobility_logits"].squeeze(-1)
        fm_loss = F.binary_cross_entropy_with_logits(fm_logits, final_mobility_targets)
        loss_dict["final_mobility_loss"] = fm_loss
        total = total + final_mobility_weight * fm_loss

    loss_dict["total_loss"] = total
    return total, loss_dict


# ── Transformer Loss with Auxiliary Heads ─────────────────────────


def compute_transformer_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    aux_outputs: dict[str, torch.Tensor],
    mobility_targets: torch.Tensor,
    queen_surround_targets: torch.Tensor,
    queen_surround_mask: torch.Tensor,
    final_mobility_targets: torch.Tensor,
    value_mask: torch.Tensor,
    board_token_batch: torch.Tensor,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    mobility_weight: float = 0.15,
    queen_surround_weight: float = 0.15,
    final_mobility_weight: float = 0.15,
    policy_softening: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the combined loss with auxiliary heads and playout cap masking.

    Same structure as compute_gnn_loss() but uses board_token_batch
    instead of piece_node_batch for per-token target indexing.

    Args:
        policy_logits: Raw logits, shape (B, 29407).
        value_pred: Predicted values, shape (B, 1).
        target_policy: Target policy distribution, shape (B, 29407).
        target_value: Target values, shape (B, 1). In [-1, 1].
        aux_outputs: Dict from HiveTransformer.forward() with optional keys
            ``mobility_logits``, ``queen_surround_logits``, and
            ``final_mobility_logits``.
        mobility_targets: Per-board-token binary targets, shape (total_board_tokens,).
        queen_surround_targets: Per-board-token queen adjacency targets,
            shape (total_board_tokens, 2).
        queen_surround_mask: Per-sequence mask indicating which queens are
            on the board, shape (B, 2). 1.0 if placed, 0.0 otherwise.
        final_mobility_targets: Per-board-token binary targets for end-of-game
            mobility, shape (total_board_tokens,).
        value_mask: Per-sequence mask for playout cap, shape (B,). 1.0 for
            full-playout games that contribute to value loss.
        board_token_batch: Sequence index for each board token,
            shape (total_board_tokens,).
        policy_weight: Weight for policy loss.
        value_weight: Weight for value loss.
        mobility_weight: Weight for mobility auxiliary loss.
        queen_surround_weight: Weight for queen surround auxiliary loss.
        final_mobility_weight: Weight for final mobility auxiliary loss.

    Returns:
        (total_loss, loss_dict) where loss_dict maps loss names to scalars.
    """
    # Policy softening: mix MCTS target with uniform over legal moves.
    # target_policy has epsilon on all legal actions (not just Gumbel-visited),
    # so (target_policy > 0) correctly identifies the full legal action set.
    legal_mask = target_policy > 0
    if policy_softening > 0.0:
        legal = legal_mask.float()
        num_legal = legal.sum(dim=1, keepdim=True).clamp(min=1)
        uniform = legal / num_legal
        target_policy = (1.0 - policy_softening) * target_policy + policy_softening * uniform

    # Policy loss: cross-entropy masked to legal actions only.
    # Masking logits to -inf for illegal actions before log_softmax concentrates
    # the entire gradient signal on legal moves, eliminating the ~52% waste on
    # illegal action suppression that occurs with an unmasked softmax.
    # We zero out log_probs for illegal positions after log_softmax to avoid
    # 0 * -inf = NaN (target_policy is 0 there, so the zero is mathematically
    # correct and the gradient contribution is also zero).
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1).masked_fill(~legal_mask, 0.0)
    policy_loss = -torch.mean(torch.sum(target_policy * log_probs, dim=1))

    # Value loss: Gaussian NLL if uncertainty head present, else MSE; both masked by playout cap
    if "value_logvar" in aux_outputs:
        log_var = aux_outputs["value_logvar"].squeeze(-1)  # (B,)
        var = torch.exp(log_var)
        diff_sq = (value_pred.squeeze(-1) - target_value.squeeze(-1)) ** 2
        per_sample = 0.5 * diff_sq / var + 0.5 * log_var
        if value_mask.sum() > 0:
            value_loss = (per_sample * value_mask).sum() / value_mask.sum()
        else:
            value_loss = torch.tensor(0.0, device=policy_logits.device)
    else:
        value_diff = (value_pred.squeeze(-1) - target_value.squeeze(-1)) ** 2
        if value_mask.sum() > 0:
            value_loss = (value_diff * value_mask).sum() / value_mask.sum()
        else:
            value_loss = torch.tensor(0.0, device=policy_logits.device)

    loss_dict: dict[str, torch.Tensor] = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }

    total = policy_weight * policy_loss + value_weight * value_loss

    # Mobility auxiliary loss (current-state mobility)
    if "mobility_logits" in aux_outputs and mobility_targets.numel() > 0:
        mob_logits = aux_outputs["mobility_logits"].squeeze(-1)
        mob_loss = F.binary_cross_entropy_with_logits(mob_logits, mobility_targets)
        loss_dict["mobility_loss"] = mob_loss
        total = total + mobility_weight * mob_loss

    # Queen surround auxiliary loss
    if "queen_surround_logits" in aux_outputs and queen_surround_targets.numel() > 0:
        qs_logits = aux_outputs["queen_surround_logits"]  # (total_board_tokens, 2)

        # Expand queen_surround_mask from (B, 2) to (total_board_tokens, 2)
        per_token_mask = queen_surround_mask[board_token_batch]  # (total_board_tokens, 2)

        # Binary cross-entropy with masking
        qs_bce = F.binary_cross_entropy_with_logits(
            qs_logits, queen_surround_targets, reduction="none"
        )  # (total_board_tokens, 2)
        qs_bce = qs_bce * per_token_mask

        if per_token_mask.sum() > 0:
            qs_loss = qs_bce.sum() / per_token_mask.sum()
        else:
            qs_loss = torch.tensor(0.0, device=policy_logits.device)

        loss_dict["queen_surround_loss"] = qs_loss
        total = total + queen_surround_weight * qs_loss

    # Final mobility auxiliary loss (end-of-game mobility)
    if "final_mobility_logits" in aux_outputs and final_mobility_targets.numel() > 0:
        fm_logits = aux_outputs["final_mobility_logits"].squeeze(-1)
        fm_loss = F.binary_cross_entropy_with_logits(fm_logits, final_mobility_targets)
        loss_dict["final_mobility_loss"] = fm_loss
        total = total + final_mobility_weight * fm_loss

    loss_dict["total_loss"] = total
    return total, loss_dict
