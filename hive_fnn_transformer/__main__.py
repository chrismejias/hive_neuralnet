"""Print a small summary for the hybrid FNN transformer model."""

from hive_fnn_transformer.fnn_transformer_net import HybridGNNConfig, HiveHybridGNN


def main() -> None:
    for name, cfg in (
        ("small", HybridGNNConfig.small()),
        ("large", HybridGNNConfig.large()),
    ):
        net = HiveHybridGNN(cfg)
        print(f"fnn_transformer {name}: {net.count_parameters():,} parameters")


if __name__ == "__main__":
    main()
