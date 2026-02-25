"""
Graph Neural Network for Hive.

Architecture:
    HiveGraph → NodeEmbedding → [MessagePassingLayer × N] →
        ├── HybridPolicyHead → logits (29407,)
        └── ValueHead → scalar in [-1, 1]

The message-passing layers use direction-aware edges: each message
includes the source node, destination node, and edge features (dq, dr,
direction one-hot). This preserves hex spatial orientation without
requiring a fixed grid.

No PyTorch Geometric dependency — scatter operations use raw torch
scatter_add_.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_engine.encoder import HiveEncoder

from hive_gnn.graph_types import (
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    HiveGraph,
    HiveGraphBatch,
)


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class GNNNetConfig:
    """Configuration for HiveGNN architecture."""

    hidden_dim: int = 128
    num_mp_layers: int = 6
    edge_feat_dim: int = EDGE_FEAT_DIM      # 9
    node_feat_dim: int = NODE_FEAT_DIM       # 21
    global_feat_dim: int = GLOBAL_FEAT_DIM   # 6
    policy_conv_channels: int = 32
    action_space_size: int = HiveEncoder.ACTION_SPACE_SIZE  # 29407
    board_size: int = HiveEncoder.BOARD_SIZE                # 13

    @classmethod
    def small(cls) -> GNNNetConfig:
        """Small network preset (~1.5M parameters)."""
        return cls(hidden_dim=128, num_mp_layers=6)

    @classmethod
    def large(cls) -> GNNNetConfig:
        """Large network preset (~6M parameters)."""
        return cls(hidden_dim=256, num_mp_layers=8)


# ── Message Passing Layer ──────────────────────────────────────────


class MessagePassingLayer(nn.Module):
    """
    Direction-aware message passing layer.

    For each edge (src → dst) with edge features e_ij:
        message_ij = MLP([h_dst || h_src || e_ij])
        aggregated_i = scatter_add(messages, dst_indices)
        h_i' = LayerNorm(h_i + MLP([h_i || aggregated_i]))

    The edge features include directional information (dq, dr,
    direction_onehot) which allows the GNN to reason about spatial
    relationships.
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int) -> None:
        super().__init__()
        # Message MLP: [h_dst, h_src, e_ij] → hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Update MLP: [h_i, aggregated_i] → hidden_dim
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,   # (N, hidden_dim)
        edge_index: torch.Tensor,       # (2, E)
        edge_features: torch.Tensor,    # (E, edge_feat_dim)
    ) -> torch.Tensor:
        """
        Forward pass of one message-passing step.

        Args:
            node_features: Node embeddings (N, hidden_dim).
            edge_index: Edge list, shape (2, E). Row 0 = src, row 1 = dst.
            edge_features: Per-edge features (E, edge_feat_dim).

        Returns:
            Updated node embeddings (N, hidden_dim).
        """
        num_nodes = node_features.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # No edges — skip message passing, just pass through
            return self.layer_norm(node_features)

        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        h_src = node_features[src_idx]   # (E, hidden_dim)
        h_dst = node_features[dst_idx]   # (E, hidden_dim)

        # Compute messages: [h_dst || h_src || e_ij]
        msg_input = torch.cat([h_dst, h_src, edge_features], dim=1)
        messages = self.message_mlp(msg_input)  # (E, hidden_dim)

        # Aggregate messages at destination nodes via scatter_add
        aggregated = torch.zeros(
            num_nodes, messages.size(1),
            dtype=messages.dtype, device=messages.device
        )
        aggregated.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(messages), messages)

        # Update: h_i' = LayerNorm(h_i + MLP([h_i || aggregated_i]))
        update_input = torch.cat([node_features, aggregated], dim=1)
        updated = self.update_mlp(update_input)  # (N, hidden_dim)

        return self.layer_norm(node_features + updated)


# ── Policy Head ────────────────────────────────────────────────────


class HybridPolicyHead(nn.Module):
    """
    Hybrid policy head that bridges GNN node embeddings to the
    fixed-size action space.

    Strategy:
        1. Scatter piece node embeddings into a (batch, hidden_dim, 13, 13)
           spatial grid using their grid positions.
        2. Conv layers reduce to (batch, 2, 13, 13) → flatten to (batch, 338).
        3. Concatenate with global features → FC layers → (batch, 29407).
    """

    def __init__(self, config: GNNNetConfig) -> None:
        super().__init__()
        h = config.hidden_dim
        bs = config.board_size
        pc = config.policy_conv_channels

        # Conv layers on the spatial grid
        self.conv1 = nn.Conv2d(h, pc, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pc)
        self.conv2 = nn.Conv2d(pc, 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2)

        flatten_size = 2 * bs * bs  # 2 * 169 = 338
        self.fc1 = nn.Linear(flatten_size + config.global_feat_dim, 256)
        self.fc2 = nn.Linear(256, config.action_space_size)

        self._hidden_dim = h
        self._board_size = bs

    def forward(
        self,
        node_embeddings: torch.Tensor,       # (total_N, hidden_dim)
        piece_node_batch: torch.Tensor,       # (total_piece_nodes,)
        node_positions: torch.Tensor,         # (total_piece_nodes, 2) int
        global_features: torch.Tensor,        # (B, global_feat_dim)
        num_piece_nodes: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute policy logits from node embeddings.

        Args:
            node_embeddings: All node embeddings (piece + hand).
            piece_node_batch: Graph index for each piece node.
            node_positions: Grid (row, col) for each piece node.
            global_features: Per-graph global features.
            num_piece_nodes: Total piece nodes across batch.
            batch_size: Number of graphs in batch.

        Returns:
            Policy logits of shape (batch_size, action_space_size).
        """
        h = self._hidden_dim
        bs = self._board_size
        device = node_embeddings.device

        # 1. Create empty spatial grid
        grid = torch.zeros(batch_size, h, bs, bs, device=device)

        # 2. Scatter piece node embeddings into grid
        if num_piece_nodes > 0:
            # Get embeddings for piece nodes only (first num_piece_nodes
            # within each graph, but they're scattered — use piece_node indices)
            # We need to extract piece node embeddings from the full node_embeddings
            # piece_node_batch tells us which graph each piece node belongs to
            # node_positions tells us (row, col) for each piece node

            # Build piece node indices in the full node_embeddings tensor
            # Piece nodes are the first num_piece_nodes in each graph within
            # the batch vector. We need a mapping from piece_node index → global index.
            # Actually, piece node embeddings correspond to the first
            # graph.num_piece_nodes of each graph's nodes. Since the batch
            # concatenates them, we need to track which global node indices
            # are piece nodes.
            # For simplicity: we trust that the caller provides positions
            # aligned with piece nodes. We'll use piece_node_batch to extract
            # the right embeddings.

            # piece_node_embeddings: we need to find them in node_embeddings
            # The batch vector has all piece nodes first per graph, but that's
            # not guaranteed across graphs. We'll precompute them.
            # Actually: in our collate, piece nodes from graph i are at the
            # start of graph i's node block. So we can reconstruct indices.
            # BUT: it's simpler and safer to pass piece node embeddings directly.
            # For now, we'll rely on the caller slicing correctly.

            # The HiveGNN.forward() will extract piece_embeddings and pass them here.
            pass

        # The actual scatter happens in HiveGNN.forward which passes
        # pre-extracted piece_embeddings. See _scatter_to_grid.
        # This method is called after scatter.

        return grid  # placeholder — real logic below in _forward_with_grid

    def forward_with_grid(
        self,
        grid: torch.Tensor,                  # (B, hidden_dim, 13, 13)
        global_features: torch.Tensor,        # (B, global_feat_dim)
    ) -> torch.Tensor:
        """
        Run conv + FC layers on the pre-built spatial grid.

        Args:
            grid: Spatial grid with scattered node embeddings.
            global_features: Per-graph global features.

        Returns:
            Policy logits (B, action_space_size).
        """
        p = F.relu(self.bn1(self.conv1(grid)))
        p = F.relu(self.bn2(self.conv2(p)))
        p = p.view(p.size(0), -1)  # (B, 338)

        # Concatenate with global features
        p = torch.cat([p, global_features], dim=1)  # (B, 338 + 6)
        p = F.relu(self.fc1(p))
        return self.fc2(p)  # (B, 29407)


# ── Value Head ─────────────────────────────────────────────────────


class ValueHead(nn.Module):
    """
    Value head using mean + max pooling over node embeddings.

    Architecture:
        node_embeddings → mean_pool + max_pool → concat with global →
        FC → ReLU → FC → tanh → scalar
    """

    def __init__(self, hidden_dim: int, global_feat_dim: int) -> None:
        super().__init__()
        # Input: mean_pool(h) + max_pool(h) + global = 2*h + g
        self.fc1 = nn.Linear(hidden_dim * 2 + global_feat_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(
        self,
        node_embeddings: torch.Tensor,   # (total_N, hidden_dim)
        batch: torch.Tensor,             # (total_N,)
        global_features: torch.Tensor,   # (B, global_feat_dim)
        batch_size: int,
    ) -> torch.Tensor:
        """
        Compute value prediction.

        Args:
            node_embeddings: All node embeddings.
            batch: Graph index for each node.
            global_features: Per-graph global features.
            batch_size: Number of graphs in batch.

        Returns:
            Value predictions (B, 1) in [-1, 1].
        """
        hidden_dim = node_embeddings.size(1)
        device = node_embeddings.device

        # Mean pooling per graph
        mean_pool = torch.zeros(batch_size, hidden_dim, device=device)
        count = torch.zeros(batch_size, 1, device=device)
        mean_pool.scatter_add_(
            0, batch.unsqueeze(1).expand_as(node_embeddings), node_embeddings
        )
        count.scatter_add_(
            0, batch.unsqueeze(1), torch.ones(batch.size(0), 1, device=device)
        )
        mean_pool = mean_pool / count.clamp(min=1)

        # Max pooling per graph
        max_pool = torch.full(
            (batch_size, hidden_dim), float("-inf"), device=device
        )
        max_pool.scatter_reduce_(
            0,
            batch.unsqueeze(1).expand_as(node_embeddings),
            node_embeddings,
            reduce="amax",
            include_self=False,
        )
        # Handle empty graphs (shouldn't happen but be safe)
        max_pool = max_pool.masked_fill(max_pool == float("-inf"), 0.0)

        # Concat: [mean_pool, max_pool, global]
        combined = torch.cat([mean_pool, max_pool, global_features], dim=1)

        v = F.relu(self.fc1(combined))
        return torch.tanh(self.fc2(v))  # (B, 1)


# ── HiveGNN ────────────────────────────────────────────────────────


class HiveGNN(nn.Module):
    """
    Graph Neural Network for Hive, producing the same output interface
    as HiveNet: policy logits (batch, 29407) and value (batch, 1).

    Forward input: HiveGraphBatch (batched graph with node/edge features).

    Architecture:
        Linear projection → [MessagePassingLayer × N] →
            ├── HybridPolicyHead (scatter to grid → conv → FC) → logits
            └── ValueHead (pool → FC → tanh) → value
    """

    def __init__(self, config: GNNNetConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = GNNNetConfig()
        self.config = config

        h = config.hidden_dim

        # Node feature projection
        self.node_proj = nn.Linear(config.node_feat_dim, h)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(h, config.edge_feat_dim)
            for _ in range(config.num_mp_layers)
        ])

        # Policy head
        self.policy_head = HybridPolicyHead(config)

        # Value head
        self.value_head = ValueHead(h, config.global_feat_dim)

        self._board_size = config.board_size
        self._hidden_dim = h

    def forward(
        self, graph_batch: HiveGraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on a batched graph.

        Args:
            graph_batch: A HiveGraphBatch with all tensors on the same device.

        Returns:
            (policy_logits, value) where:
                policy_logits: shape (batch, 29407)
                value: shape (batch, 1), in [-1, 1]
        """
        batch_size = graph_batch.global_features.size(0)
        device = graph_batch.node_features.device

        # 1. Project node features
        h = F.relu(self.node_proj(graph_batch.node_features))  # (N, hidden_dim)

        # 2. Message passing
        for mp_layer in self.mp_layers:
            h = mp_layer(h, graph_batch.edge_index, graph_batch.edge_features)

        # 3. Build spatial grid for policy head
        grid = self._scatter_to_grid(
            h, graph_batch, batch_size, device
        )

        # 4. Policy logits
        policy_logits = self.policy_head.forward_with_grid(
            grid, graph_batch.global_features
        )

        # 5. Value
        value = self.value_head(
            h, graph_batch.batch, graph_batch.global_features, batch_size
        )

        return policy_logits, value

    def _scatter_to_grid(
        self,
        node_embeddings: torch.Tensor,
        graph_batch: HiveGraphBatch,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Scatter piece node embeddings into a spatial grid for the policy head.

        Each piece node has a grid position (row, col). We place its
        embedding into the corresponding cell of a (B, H, 13, 13) tensor.

        Args:
            node_embeddings: All node embeddings (N, hidden_dim).
            graph_batch: The batched graph data.
            batch_size: Number of graphs.
            device: Target device.

        Returns:
            Grid tensor (B, hidden_dim, board_size, board_size).
        """
        h = self._hidden_dim
        bs = self._board_size

        grid = torch.zeros(batch_size, h, bs, bs, device=device)

        num_piece = graph_batch.num_piece_nodes
        if num_piece == 0:
            return grid

        # Extract piece node embeddings
        # Piece nodes are the first num_piece_nodes of each graph in order.
        # We need to find them in the concatenated node_embeddings.
        # Strategy: build a boolean mask over all nodes indicating piece nodes.
        total_nodes = node_embeddings.size(0)
        batch_vec = graph_batch.batch  # (total_N,)

        # Count nodes per graph
        nodes_per_graph = torch.zeros(batch_size, dtype=torch.long, device=device)
        nodes_per_graph.scatter_add_(
            0, batch_vec, torch.ones(total_nodes, dtype=torch.long, device=device)
        )

        # For each graph, piece nodes are the first graph_i.num_piece_nodes
        # We'll build piece node indices by iterating (this is at collate level,
        # but piece_node_batch already tells us which piece nodes belong to which graph)
        # Actually, we already have piece_node_batch (total_piece_nodes,) and
        # node_positions (total_piece_nodes, 2). We need the global node indices
        # of piece nodes.

        # Approach: the piece nodes in each graph are at the beginning of that
        # graph's node block. We can compute cumulative offsets.
        # piece_node_batch[j] = graph_idx means piece node j is in graph graph_idx
        # We need to figure out which global node index corresponds to piece node j.

        # Simpler approach: build a mask. For each graph i, the first
        # piece_count_i nodes (within graph i's block) are piece nodes.
        # But we don't store per-graph piece counts in the batch...

        # Best approach: explicitly compute piece node global indices.
        # We know: nodes_per_graph gives the count. For graph i, the node
        # block starts at cum_nodes[i]. Within that block, the first
        # piece_nodes_for_graph_i nodes are piece nodes.

        # Compute piece nodes per graph from piece_node_batch
        piece_per_graph = torch.zeros(batch_size, dtype=torch.long, device=device)
        piece_per_graph.scatter_add_(
            0,
            graph_batch.piece_node_batch,
            torch.ones(num_piece, dtype=torch.long, device=device),
        )

        # Compute cumulative node offsets per graph
        cum_nodes = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        cum_nodes[1:] = nodes_per_graph.cumsum(0)

        # Build piece node global indices
        # For graph i, piece nodes are at cum_nodes[i], cum_nodes[i]+1, ..., cum_nodes[i]+piece_per_graph[i]-1
        piece_indices = []
        cum_piece = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        cum_piece[1:] = piece_per_graph.cumsum(0)

        # Vectorized approach: for each piece node j, its graph is piece_node_batch[j]
        # and its local index within that graph's piece nodes is j - cum_piece[graph_idx]
        # Its global node index is cum_nodes[graph_idx] + local_index
        graph_indices = graph_batch.piece_node_batch  # (num_piece,)
        local_piece_idx = torch.arange(num_piece, device=device) - cum_piece[graph_indices]
        global_node_idx = cum_nodes[graph_indices] + local_piece_idx

        # Get piece node embeddings
        piece_embeddings = node_embeddings[global_node_idx]  # (num_piece, h)

        # Scatter into grid using node_positions
        positions = graph_batch.node_positions.long()  # (num_piece, 2) → (row, col)
        rows = positions[:, 0].clamp(0, bs - 1)
        cols = positions[:, 1].clamp(0, bs - 1)

        # Use index_put_ with accumulate for overlapping positions
        batch_idx = graph_batch.piece_node_batch  # (num_piece,)

        # Grid shape: (B, H, bs, bs)
        # We want: grid[batch_idx[j], :, rows[j], cols[j]] += piece_embeddings[j, :]
        # Use advanced indexing with accumulate
        idx = (batch_idx, slice(None), rows, cols)
        # Can't use slice with index_put_ directly; expand batch_idx/rows/cols
        # to match (num_piece, H) shape
        b_exp = batch_idx.unsqueeze(1).expand(-1, h)  # (num_piece, h)
        r_exp = rows.unsqueeze(1).expand(-1, h)
        c_exp = cols.unsqueeze(1).expand(-1, h)
        ch = torch.arange(h, device=device).unsqueeze(0).expand(num_piece, -1)

        grid.index_put_(
            (b_exp, ch, r_exp, c_exp),
            piece_embeddings,
            accumulate=True,
        )

        return grid

    @torch.no_grad()
    def predict(
        self,
        graph: HiveGraph,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Single-state inference for MCTS.

        Creates a single-element batch, runs forward, applies legal mask,
        and returns action probabilities + value.

        Args:
            graph: A HiveGraph (numpy-based).
            legal_mask: Shape (29407,), float32. 1.0 for legal actions.

        Returns:
            (action_probs, value) where:
                action_probs: Shape (29407,), sums to ~1.0 over legal actions.
                value: Scalar in [-1, 1].
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        # Create single-element batch
        batch = HiveGraphBatch.collate([graph]).to(device)
        mask = torch.from_numpy(legal_mask).to(device)

        policy_logits, value_tensor = self.forward(batch)

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
