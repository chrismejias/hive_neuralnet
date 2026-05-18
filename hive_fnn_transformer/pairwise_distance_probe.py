"""Standalone receptive-field probe for the hybrid GNN trunk.

This script removes the FNN, policy head, and value head entirely. It trains
the graph message-passing stack to classify the exact hex distance between
pairs of occupied cells in simplified, connected Hive-like midgame positions.

Positions are synthetic:
  - one shared token type
  - alternating colors
  - no stacks, no hands, no move generation
  - connected placement-only hives

The goal is not game strength. It is to measure how well the graph trunk can
recover pairwise distances from local radius-k edges alone.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hive_engine.hex_coord import _OFFSET_LIST
from hive_engine.pieces import PieceType
from hive_fnn_transformer.graph_encoder import _hex_distance, radius_offsets
from hive_fnn_transformer.graph_types import (
    GLOBAL_FEAT_DIM,
    NODE_FEAT_DIM,
    HybridGraph,
    HybridGraphBatch,
    edge_feat_dim_for_radius,
)
from hive_fnn_transformer.fnn_transformer_net import HybridGNNConfig, HybridMessagePassingLayer


@dataclass
class ProbeExample:
    graph: HybridGraph
    pair_index: np.ndarray
    targets: np.ndarray


@dataclass
class ProbeBatch:
    graph_batch: HybridGraphBatch
    pair_index: torch.Tensor
    targets: torch.Tensor

    def to(self, device: torch.device | str) -> "ProbeBatch":
        return ProbeBatch(
            graph_batch=self.graph_batch.to(device),
            pair_index=self.pair_index.to(device),
            targets=self.targets.to(device),
        )


class PairwiseDistanceNet(nn.Module):
    def __init__(self, config: HybridGNNConfig, max_distance: int) -> None:
        super().__init__()
        self.config = config
        self.max_distance = int(max_distance)
        hidden = config.graph_hidden_dim
        self.node_proj = nn.Linear(config.node_feat_dim, hidden)
        self.layers = nn.ModuleList([
            HybridMessagePassingLayer(
                hidden,
                config.edge_feat_dim,
                config.graph_mlp_hidden,
                config.global_pool_bias,
            )
            for _ in range(config.graph_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, config.value_hidden),
            nn.ReLU(),
            nn.Linear(config.value_hidden, self.max_distance + 1),
        )

    def encode_nodes(self, graph: HybridGraphBatch) -> torch.Tensor:
        h = F.relu(self.node_proj(graph.node_features))
        batch_size = graph.global_features.size(0)
        for layer in self.layers:
            h = layer(
                h,
                graph.edge_index,
                graph.edge_features,
                graph.batch,
                batch_size,
            )
        return h

    def forward(self, graph: HybridGraphBatch, pair_index: torch.Tensor) -> torch.Tensor:
        node_h = self.encode_nodes(graph)
        left = node_h[pair_index[0]]
        right = node_h[pair_index[1]]
        pair_h = torch.cat([left + right, torch.abs(left - right)], dim=1)
        return self.head(pair_h)


_SYN_TYPE_BASE = 0
_SYN_TYPE_DIM = 14
_SYN_COLOR_BASE = 14
_SYN_GROUND_IDX = 16
_SYN_TOP_IDX = 17
_SYN_STACK_HEIGHT_IDX = 18
_SYN_TYPE0_MARKER_IDX = 19
_SYN_OCC_COUNT_IDX = 20
_SYN_EMPTY_NEIGHBOR_BASE = 21


def build_graph_from_positions(
    coords: list[tuple[int, int]],
    colors: list[int],
    type_labels: list[int],
    radius: int,
    piece_type: PieceType = PieceType.QUEEN,
    synthetic_types: int = 1,
) -> HybridGraph:
    offsets = radius_offsets(radius)
    offset_to_index = {offset: i for i, offset in enumerate(offsets)}
    edge_feat_dim = edge_feat_dim_for_radius(radius)
    num_nodes = len(coords)

    node_features = np.zeros((num_nodes, NODE_FEAT_DIM), dtype=np.float32)
    node_positions = np.asarray(coords, dtype=np.int32)
    node_piece_types = np.full((num_nodes,), int(piece_type.value), dtype=np.int32)
    edge_index = np.zeros((2, num_nodes * len(offsets)), dtype=np.int64)
    edge_features = np.zeros((num_nodes * len(offsets), edge_feat_dim), dtype=np.float32)

    occupied = set(coords)
    coord_to_idx = {coord: idx for idx, coord in enumerate(coords)}

    edge_count = 0
    for idx, (q, r) in enumerate(coords):
        occ_by_dir = [int((q + dq, r + dr) in occupied) for dq, dr in _OFFSET_LIST]
        occ_count = sum(occ_by_dir)

        feat = node_features[idx]
        if synthetic_types < 1 or synthetic_types > _SYN_TYPE_DIM:
            raise ValueError(f"synthetic_types must be in [1, {_SYN_TYPE_DIM}]")
        feat[_SYN_TYPE_BASE + type_labels[idx]] = 1.0
        feat[_SYN_COLOR_BASE + colors[idx]] = 1.0
        feat[_SYN_GROUND_IDX] = 1.0
        feat[_SYN_TOP_IDX] = 1.0
        feat[_SYN_STACK_HEIGHT_IDX] = 0.25
        feat[_SYN_TYPE0_MARKER_IDX] = 1.0 if type_labels[idx] == 0 else 0.0
        feat[_SYN_OCC_COUNT_IDX] = occ_count / 6.0
        for d, occupied_dir in enumerate(occ_by_dir):
            feat[_SYN_EMPTY_NEIGHBOR_BASE + d] = 0.0 if occupied_dir else 1.0

        for dq, dr in offsets:
            dst_idx = coord_to_idx.get((q + dq, r + dr))
            if dst_idx is None:
                continue
            edge_index[0, edge_count] = idx
            edge_index[1, edge_count] = dst_idx
            edge_features[edge_count, 0] = float(dq) / float(radius)
            edge_features[edge_count, 1] = float(dr) / float(radius)
            edge_features[edge_count, 2 + offset_to_index[(dq, dr)]] = 1.0
            edge_count += 1

    return HybridGraph(
        node_features=node_features,
        edge_index=edge_index[:, :edge_count],
        edge_features=edge_features[:edge_count],
        global_features=np.zeros((GLOBAL_FEAT_DIM,), dtype=np.float32),
        num_piece_nodes=num_nodes,
        node_positions=node_positions,
        node_piece_types=node_piece_types,
    )


def sample_connected_position(
    rng: random.Random,
    num_pieces: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    coords: list[tuple[int, int]] = [(0, 0)]
    occupied = {coords[0]}
    frontier = {(dq, dr) for dq, dr in _OFFSET_LIST}
    start_color = rng.randrange(2)
    colors = [start_color]

    while len(coords) < num_pieces:
        choices = sorted(frontier)
        pos = choices[rng.randrange(len(choices))]
        coords.append(pos)
        occupied.add(pos)
        colors.append((start_color + len(coords) - 1) & 1)
        frontier.discard(pos)
        q, r = pos
        for dq, dr in _OFFSET_LIST:
            nxt = (q + dq, r + dr)
            if nxt not in occupied:
                frontier.add(nxt)

    order = list(range(num_pieces))
    rng.shuffle(order)
    coords = [coords[i] for i in order]
    colors = [colors[i] for i in order]
    return coords, colors


def sample_type_labels(
    rng: random.Random,
    num_pieces: int,
    synthetic_types: int,
) -> list[int]:
    if synthetic_types < 1 or synthetic_types > _SYN_TYPE_DIM:
        raise ValueError(f"synthetic_types must be in [1, {_SYN_TYPE_DIM}]")
    return [rng.randrange(synthetic_types) for _ in range(num_pieces)]


def build_probe_example(
    rng: random.Random,
    radius: int,
    min_pieces: int,
    max_pieces: int,
    max_distance: int,
    synthetic_types: int,
) -> ProbeExample:
    num_pieces = rng.randint(min_pieces, max_pieces)
    coords, colors = sample_connected_position(rng, num_pieces)
    type_labels = sample_type_labels(rng, num_pieces, synthetic_types)
    graph = build_graph_from_positions(
        coords,
        colors,
        type_labels,
        radius,
        synthetic_types=synthetic_types,
    )

    pairs: list[tuple[int, int]] = []
    targets: list[int] = []
    for i in range(num_pieces):
        qi, ri = coords[i]
        for j in range(i + 1, num_pieces):
            qj, rj = coords[j]
            dist = _hex_distance(qi - qj, ri - rj)
            pairs.append((i, j))
            targets.append(min(dist, max_distance))

    return ProbeExample(
        graph=graph,
        pair_index=np.asarray(pairs, dtype=np.int64).T,
        targets=np.asarray(targets, dtype=np.int64),
    )


def build_dataset(
    seed: int,
    positions: int,
    radius: int,
    min_pieces: int,
    max_pieces: int,
    max_distance: int,
    synthetic_types: int,
) -> list[ProbeExample]:
    rng = random.Random(seed)
    return [
        build_probe_example(
            rng,
            radius,
            min_pieces,
            max_pieces,
            max_distance,
            synthetic_types,
        )
        for _ in range(positions)
    ]


def collate_probe_batch(examples: list[ProbeExample]) -> ProbeBatch:
    graphs = [ex.graph for ex in examples]
    graph_batch = HybridGraphBatch.collate(graphs)

    pair_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    node_offset = 0
    for ex in examples:
        pairs = torch.from_numpy(ex.pair_index).clone()
        pairs += node_offset
        pair_parts.append(pairs)
        target_parts.append(torch.from_numpy(ex.targets))
        node_offset += ex.graph.node_features.shape[0]

    pair_index = torch.cat(pair_parts, dim=1)
    targets = torch.cat(target_parts, dim=0)
    return ProbeBatch(graph_batch=graph_batch, pair_index=pair_index, targets=targets)


def iterate_batches(
    dataset: list[ProbeExample],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[ProbeBatch]:
    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    batches: list[ProbeBatch] = []
    for start in range(0, len(indices), batch_size):
        chunk = [dataset[i] for i in indices[start:start + batch_size]]
        batches.append(collate_probe_batch(chunk))
    return batches


@torch.no_grad()
def evaluate(
    model: PairwiseDistanceNet,
    dataset: list[ProbeExample],
    batch_size: int,
    device: torch.device,
) -> tuple[float, dict[int, tuple[int, int]]]:
    model.eval()
    total = 0
    correct = 0
    by_distance: dict[int, list[int]] = {}
    for batch in iterate_batches(dataset, batch_size=batch_size, shuffle=False, seed=0):
        batch = batch.to(device)
        logits = model(batch.graph_batch, batch.pair_index)
        pred = logits.argmax(dim=1)
        truth = batch.targets
        hit = (pred == truth)
        total += int(truth.numel())
        correct += int(hit.sum().item())
        for dist in truth.unique(sorted=True).tolist():
            mask = truth == dist
            if int(dist) not in by_distance:
                by_distance[int(dist)] = [0, 0]
            by_distance[int(dist)][0] += int((hit & mask).sum().item())
            by_distance[int(dist)][1] += int(mask.sum().item())
    return correct / max(total, 1), {k: (v[0], v[1]) for k, v in sorted(by_distance.items())}


def train_probe(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )
    config = HybridGNNConfig.small() if args.preset == "small" else HybridGNNConfig.large()
    if args.graph_layers is not None:
        config.graph_layers = int(args.graph_layers)
    if args.graph_hidden_dim is not None:
        config.graph_hidden_dim = int(args.graph_hidden_dim)
    if args.graph_mlp_hidden is not None:
        config.graph_mlp_hidden = int(args.graph_mlp_hidden)
    if args.value_hidden is not None:
        config.value_hidden = int(args.value_hidden)
    if args.graph_radius is not None:
        config.graph_radius = int(args.graph_radius)
    if args.global_pool_bias is not None:
        config.global_pool_bias = bool(args.global_pool_bias)
    model = PairwiseDistanceNet(config, max_distance=args.max_distance).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    eval_set = build_dataset(
        seed=args.seed + 10_000,
        positions=args.eval_positions,
        radius=config.graph_radius,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
        max_distance=args.max_distance,
        synthetic_types=args.synthetic_types,
    )

    print(
        f"preset={args.preset} device={device} graph_hidden={config.graph_hidden_dim} "
        f"layers={config.graph_layers} radius={config.graph_radius} "
        f"graph_mlp_hidden={config.graph_mlp_hidden} value_hidden={config.value_hidden} "
        f"global_pool_bias={config.global_pool_bias} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"train_iters={args.iterations} positions_per_iter={args.positions_per_iter} "
        f"eval_positions={args.eval_positions} piece_range=[{args.min_pieces},{args.max_pieces}] "
        f"max_distance={args.max_distance} synthetic_types={args.synthetic_types}"
    )

    start_time = time.time()
    for iteration in range(1, args.iterations + 1):
        dataset = build_dataset(
            seed=args.seed + iteration,
            positions=args.positions_per_iter,
            radius=config.graph_radius,
            min_pieces=args.min_pieces,
            max_pieces=args.max_pieces,
            max_distance=args.max_distance,
            synthetic_types=args.synthetic_types,
        )

        model.train()
        loss_sum = 0.0
        pair_count = 0
        for batch_idx, batch in enumerate(
            iterate_batches(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed + iteration)
        ):
            batch = batch.to(device)
            logits = model(batch.graph_batch, batch.pair_index)
            loss = F.cross_entropy(logits, batch.targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pairs = int(batch.targets.numel())
            loss_sum += float(loss.item()) * pairs
            pair_count += pairs
            if args.max_batches > 0 and batch_idx + 1 >= args.max_batches:
                break

        overall_acc, by_distance = evaluate(model, eval_set, args.batch_size, device)
        mean_loss = loss_sum / max(pair_count, 1)
        elapsed = time.time() - start_time
        print(
            f"iter={iteration} loss={mean_loss:.4f} eval_acc={overall_acc:.4f} "
            f"elapsed={elapsed:.1f}s"
        )
        line = " ".join(
            f"d{dist}:{correct/total:.3f}({correct}/{total})"
            for dist, (correct, total) in by_distance.items()
        )
        print(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a standalone hybrid-GNN receptive-field probe on pairwise distances.",
    )
    p.add_argument("--preset", choices=("small", "large"), default="small")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--positions-per-iter", type=int, default=5000)
    p.add_argument("--eval-positions", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-pieces", type=int, default=10)
    p.add_argument("--max-pieces", type=int, default=18)
    p.add_argument("--synthetic-types", type=int, default=1)
    p.add_argument("--max-distance", type=int, default=18)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--graph-layers", type=int, default=None)
    p.add_argument("--graph-hidden-dim", type=int, default=None)
    p.add_argument("--graph-mlp-hidden", type=int, default=None)
    p.add_argument("--value-hidden", type=int, default=None)
    p.add_argument("--graph-radius", type=int, default=None)
    p.add_argument(
        "--global-pool-bias",
        type=int,
        choices=(0, 1),
        default=None,
        help="Override graph global-pool bias: 1=enabled, 0=disabled.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="For smoke tests only: stop each training iteration early after this many batches.",
    )
    return p.parse_args()


if __name__ == "__main__":
    train_probe(parse_args())
