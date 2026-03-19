"""
GPU-accelerated state encoder for GNN and Transformer networks.

Converts GPU-resident HiveState tensors into HiveGraphBatch / HiveTokenBatch
formats ready for NN forward passes, without round-tripping through Python.
"""

from __future__ import annotations

import numpy as np
import torch

from hive_gnn.graph_types import HiveGraph, HiveGraphBatch
from hive_transformer.token_types import (
    HiveTokenBatch,
    HiveTokenSequence,
    OFF_BOARD_POSITION,
    TOKEN_TYPE_CLS,
    TOKEN_TYPE_BOARD,
    TOKEN_TYPE_HAND,
)

import hive_gpu

# Raw encode output: 9-tuple returned by ext.encode_states_batch
_RawEncoded = tuple[
    torch.Tensor,  # node_features     [B, MAX_NODES, 25]
    torch.Tensor,  # node_grid_pos     [B, MAX_NODES, 2]
    torch.Tensor,  # node_piece_types  [B, MAX_NODES]
    torch.Tensor,  # global_features   [B, 6]
    torch.Tensor,  # num_nodes         [B]
    torch.Tensor,  # num_board_nodes   [B]
    torch.Tensor,  # edge_index_raw    [B, MAX_EDGES, 2]
    torch.Tensor,  # edge_features_raw [B, MAX_EDGES, 9]
    torch.Tensor,  # num_edges         [B]
]


class GPUGNNEncoder:
    """Encode GPU HiveStates into HiveGraphBatch for GNN inference."""

    def __init__(self):
        self.ext = hive_gpu.load_extension()

    # ── Private helpers ────────────────────────────────────────────────

    def _raw_encode(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> _RawEncoded:
        """Call the CUDA kernel once; return the raw 9-tuple."""
        return self.ext.encode_states_batch(states_tensor, batch_size)

    def _assemble_batch(self, raw: _RawEncoded, batch_size: int) -> HiveGraphBatch:
        """Assemble a HiveGraphBatch from raw kernel output (no kernel call)."""
        (
            node_features, node_grid_pos, node_piece_types,
            global_features, num_nodes, num_board_nodes,
            edge_index_raw, edge_features_raw, num_edges,
        ) = raw

        nn = num_nodes.cpu()
        nb = num_board_nodes.cpu()
        ne = num_edges.cpu()

        all_nf, all_ei, all_ef = [], [], []
        all_pos, all_pt, all_batch, all_piece_batch = [], [], [], []
        cumulative_nodes = 0
        total_piece_nodes = 0

        for i in range(batch_size):
            n_i = nn[i].item()
            nb_i = nb[i].item()
            ne_i = ne[i].item()

            all_nf.append(node_features[i, :n_i])

            if ne_i > 0:
                ei = edge_index_raw[i, :ne_i] + cumulative_nodes
                all_ei.append(ei.T.to(torch.int64))
                all_ef.append(edge_features_raw[i, :ne_i])

            if nb_i > 0:
                all_pos.append(node_grid_pos[i, :nb_i].to(torch.int32))
                all_pt.append(node_piece_types[i, :nb_i].to(torch.int32))

            all_batch.append(
                torch.full((n_i,), i, dtype=torch.int64, device=node_features.device)
            )
            all_piece_batch.append(
                torch.full((nb_i,), i, dtype=torch.int64, device=node_features.device)
            )

            cumulative_nodes += n_i
            total_piece_nodes += nb_i

        dev = node_features.device
        cat_nf = torch.cat(all_nf, dim=0) if all_nf else torch.zeros((0, 25), dtype=torch.float32, device=dev)
        cat_ei = torch.cat(all_ei, dim=1) if all_ei else torch.zeros((2, 0), dtype=torch.int64, device=dev)
        cat_ef = torch.cat(all_ef, dim=0) if all_ef else torch.zeros((0, 9), dtype=torch.float32, device=dev)
        cat_pos = torch.cat(all_pos, dim=0) if all_pos else torch.zeros((0, 2), dtype=torch.int32, device=dev)
        cat_pt = torch.cat(all_pt, dim=0) if all_pt else torch.zeros((0,), dtype=torch.int32, device=dev)
        cat_batch = torch.cat(all_batch, dim=0)
        cat_piece_batch = torch.cat(all_piece_batch, dim=0)

        return HiveGraphBatch(
            node_features=cat_nf,
            edge_index=cat_ei,
            edge_features=cat_ef,
            global_features=global_features,
            num_piece_nodes=total_piece_nodes,
            node_positions=cat_pos,
            node_piece_types=cat_pt,
            batch=cat_batch,
            piece_node_batch=cat_piece_batch,
        )

    def _assemble_graphs(self, raw: _RawEncoded, batch_size: int) -> list[HiveGraph]:
        """Assemble individual HiveGraphs (CPU numpy) from raw kernel output."""
        (
            node_features, node_grid_pos, node_piece_types,
            global_features, num_nodes, num_board_nodes,
            edge_index_raw, edge_features_raw, num_edges,
        ) = raw

        nn = num_nodes.cpu()
        nb = num_board_nodes.cpu()
        ne = num_edges.cpu()
        nf_cpu = node_features.cpu()
        pos_cpu = node_grid_pos.cpu()
        pt_cpu = node_piece_types.cpu()
        gf_cpu = global_features.cpu()
        ei_cpu = edge_index_raw.cpu()
        ef_cpu = edge_features_raw.cpu()

        graphs = []
        for i in range(batch_size):
            n_i = nn[i].item()
            nb_i = nb[i].item()
            ne_i = ne[i].item()

            nf = nf_cpu[i, :n_i].numpy().astype("float32")

            if ne_i > 0:
                ei = ei_cpu[i, :ne_i].numpy().astype("int64").T  # (2, ne_i)
                ef = ef_cpu[i, :ne_i].numpy().astype("float32")
            else:
                ei = np.zeros((2, 0), dtype=np.int64)
                ef = np.zeros((0, 9), dtype=np.float32)

            gf = gf_cpu[i].numpy().astype("float32")

            if nb_i > 0:
                pos = pos_cpu[i, :nb_i].numpy().astype("int32")
                pt = pt_cpu[i, :nb_i].numpy().astype("int32")
            else:
                pos = np.zeros((0, 2), dtype=np.int32)
                pt = np.zeros((0,), dtype=np.int32)

            graphs.append(HiveGraph(
                node_features=nf,
                edge_index=ei,
                edge_features=ef,
                global_features=gf,
                num_piece_nodes=nb_i,
                node_positions=pos,
                node_piece_types=pt,
            ))

        return graphs

    # ── Public API ─────────────────────────────────────────────────────

    def encode_batch(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> HiveGraphBatch:
        """Encode a batch of GPU HiveStates into a HiveGraphBatch.

        Args:
            states_tensor: [batch_size, sizeof(HiveState)] uint8 GPU tensor
            batch_size: number of games

        Returns:
            HiveGraphBatch ready for HiveGNN.forward()
        """
        return self._assemble_batch(self._raw_encode(states_tensor, batch_size), batch_size)

    def unbatch_to_graphs(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> list[HiveGraph]:
        """Encode a batch of GPU HiveStates and return individual HiveGraph objects.

        Args:
            states_tensor: [batch_size, sizeof(HiveState)] uint8 GPU tensor
            batch_size: number of games

        Returns:
            List of HiveGraph objects (one per game), suitable for GraphReplayBuffer.
        """
        return self._assemble_graphs(self._raw_encode(states_tensor, batch_size), batch_size)

    def encode_batch_with_graphs(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> tuple[HiveGraphBatch, list[HiveGraph]]:
        """Encode once, return both HiveGraphBatch (for NN) and list[HiveGraph] (for buffer).

        Calls the CUDA kernel only once, avoiding the double-encode overhead of
        calling encode_batch() and unbatch_to_graphs() separately.

        Args:
            states_tensor: [batch_size, sizeof(HiveState)] uint8 GPU tensor
            batch_size: number of games

        Returns:
            (HiveGraphBatch, list[HiveGraph]) — same data, two formats.
        """
        raw = self._raw_encode(states_tensor, batch_size)
        return self._assemble_batch(raw, batch_size), self._assemble_graphs(raw, batch_size)


class GPUTransformerEncoder:
    """Encode GPU HiveStates into HiveTokenBatch for Transformer inference."""

    def __init__(self):
        self.ext = hive_gpu.load_extension()
        self._gnn_encoder = GPUGNNEncoder()

    def _raw_encode(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> _RawEncoded:
        return self.ext.encode_states_batch(states_tensor, batch_size)

    def _assemble_token_batch(
        self, raw: _RawEncoded, batch_size: int
    ) -> HiveTokenBatch:
        """Assemble a HiveTokenBatch from raw kernel output.

        Fully vectorized — no Python loops over batch items.
        """
        (
            node_features, node_grid_pos, node_piece_types,
            global_features, num_nodes, num_board_nodes,
            _edge_index, _edge_features, _num_edges,
        ) = raw

        device = node_features.device
        B = batch_size
        max_nodes = num_nodes.max().item()  # single GPU→CPU sync
        max_seq_len = max_nodes + 1  # +1 for CLS token

        # Sequence lengths: n_i + 1 (for CLS)
        seq_lens = (num_nodes + 1).to(torch.int64)   # (B,)
        nb = num_board_nodes.to(torch.int64)          # (B,)

        # Position index grids for broadcasting
        pos_idx = torch.arange(max_seq_len, device=device).unsqueeze(0)  # (1, S)
        node_idx = torch.arange(max_nodes, device=device).unsqueeze(0)   # (1, M)
        nb_col = nb.unsqueeze(1)       # (B, 1)
        sl_col = seq_lens.unsqueeze(1)  # (B, 1)

        # ── Token features: CLS(zero) at 0, then node features at 1..max_nodes
        token_features = torch.zeros(
            (B, max_seq_len, 25), dtype=torch.float32, device=device
        )
        token_features[:, 1:1 + max_nodes] = node_features[:, :max_nodes]

        # ── Attention mask: True for positions 0..seq_len-1
        attention_mask = pos_idx < sl_col  # (B, S)

        # ── Token types: 0=CLS, 1=BOARD (positions 1..nb), 2=HAND (nb+1..n)
        is_board = (pos_idx >= 1) & (pos_idx <= nb_col)
        is_hand = (pos_idx > nb_col) & (pos_idx < sl_col)
        token_types = torch.zeros(
            (B, max_seq_len), dtype=torch.int64, device=device
        )
        token_types[is_board] = TOKEN_TYPE_BOARD
        token_types[is_hand] = TOKEN_TYPE_HAND

        # ── Token positions: OFF_BOARD for CLS/hand, row*13+col for board
        token_positions = torch.full(
            (B, max_seq_len), OFF_BOARD_POSITION,
            dtype=torch.int64, device=device,
        )
        # Compute positions for all nodes; board nodes get row*13+col,
        # hand/padding nodes get OFF_BOARD_POSITION (from invalid grid_pos=-1)
        all_pos = (
            node_grid_pos[:, :max_nodes, 0].to(torch.int64) * 13
            + node_grid_pos[:, :max_nodes, 1].to(torch.int64)
        )  # (B, max_nodes) — hand tokens get -14 from (-1)*13+(-1)
        valid_board_node = node_idx < nb_col  # (B, max_nodes)
        board_pos = torch.where(
            valid_board_node, all_pos,
            torch.tensor(OFF_BOARD_POSITION, device=device),
        )
        token_positions[:, 1:1 + max_nodes] = board_pos

        return HiveTokenBatch(
            token_features=token_features,
            token_positions=token_positions,
            token_types=token_types,
            attention_mask=attention_mask,
            num_board_tokens=nb,
            global_features=global_features,
            seq_lengths=seq_lens,
        )

    def encode_batch(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> HiveTokenBatch:
        """Encode a batch of GPU HiveStates into a HiveTokenBatch."""
        return self._assemble_token_batch(
            self._raw_encode(states_tensor, batch_size), batch_size
        )

    def _unbatch_to_sequences(
        self, raw: _RawEncoded, batch_size: int
    ) -> list[HiveTokenSequence]:
        """Assemble individual HiveTokenSequences (CPU numpy) from raw kernel output."""
        (
            node_features, node_grid_pos, node_piece_types,
            global_features, num_nodes, num_board_nodes,
            _edge_index, _edge_features, _num_edges,
        ) = raw

        nn = num_nodes.cpu()
        nb = num_board_nodes.cpu()
        nf_cpu = node_features.cpu().numpy()
        pos_cpu = node_grid_pos.cpu().numpy()
        gf_cpu = global_features.cpu().numpy()

        sequences = []
        for i in range(batch_size):
            n_i = int(nn[i].item())
            nb_i = int(nb[i].item())
            seq_len = n_i + 1  # +1 for CLS

            # Token features: CLS=zeros at 0, node features at 1..n_i
            tf = np.zeros((seq_len, 25), dtype=np.float32)
            tf[1:n_i + 1] = nf_cpu[i, :n_i]

            # Token positions: OFF_BOARD for CLS and hand, row*13+col for board
            tp = np.full(seq_len, OFF_BOARD_POSITION, dtype=np.int32)
            for j in range(nb_i):
                r, c = int(pos_cpu[i, j, 0]), int(pos_cpu[i, j, 1])
                if r >= 0 and c >= 0:
                    tp[j + 1] = r * 13 + c

            # Token types: 0=CLS, 1=BOARD, 2=HAND
            tt = np.zeros(seq_len, dtype=np.int32)
            tt[1:nb_i + 1] = TOKEN_TYPE_BOARD
            tt[nb_i + 1:n_i + 1] = TOKEN_TYPE_HAND

            sequences.append(HiveTokenSequence(
                token_features=tf,
                token_positions=tp,
                token_types=tt,
                num_board_tokens=nb_i,
                global_features=gf_cpu[i].astype(np.float32),
            ))

        return sequences

    def encode_batch_with_graphs(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> tuple[HiveTokenBatch, list[HiveGraph]]:
        """Encode once, return both HiveTokenBatch and list[HiveGraph].

        The HiveGraph objects are needed for the replay buffer (they store
        per-example node features, edges, etc.).
        """
        raw = self._raw_encode(states_tensor, batch_size)
        token_batch = self._assemble_token_batch(raw, batch_size)
        graphs = self._gnn_encoder._assemble_graphs(raw, batch_size)
        return token_batch, graphs

    def encode_batch_with_graphs_and_seqs(
        self, states_tensor: torch.Tensor, batch_size: int
    ) -> tuple[HiveTokenBatch, list[HiveGraph], list[HiveTokenSequence]]:
        """Encode once, return HiveTokenBatch, list[HiveGraph], and list[HiveTokenSequence].

        Single kernel call for all three representations needed by the transformer
        training pipeline.
        """
        raw = self._raw_encode(states_tensor, batch_size)
        token_batch = self._assemble_token_batch(raw, batch_size)
        graphs = self._gnn_encoder._assemble_graphs(raw, batch_size)
        sequences = self._unbatch_to_sequences(raw, batch_size)
        return token_batch, graphs, sequences
