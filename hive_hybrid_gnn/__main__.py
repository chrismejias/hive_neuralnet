"""Print a small summary for the hybrid GNN model."""

from hive_hybrid_gnn.hybrid_gnn_net import HybridGNNConfig, HiveHybridGNN


def main() -> None:
    for name, cfg in (
        ("small", HybridGNNConfig.small()),
        ("large", HybridGNNConfig.large()),
    ):
        net = HiveHybridGNN(cfg)
        print(f"hybrid_GNN {name}: {net.count_parameters():,} parameters")


if __name__ == "__main__":
    main()
