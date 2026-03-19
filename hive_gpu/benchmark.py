"""
GPU MCTS throughput benchmark.

Measures positions-per-second at various batch sizes and reports
a per-phase timing breakdown: encode, legal mask, NN forward, CPU tree ops.

Usage:
    python -m hive_gpu.benchmark
    python -m hive_gpu.benchmark --simulations 20 --batch-size 64
    python -m hive_gpu.benchmark --sweep  # run batch sizes 1,4,16,32,64
"""

from __future__ import annotations

import argparse
import time

import torch

from hive_gnn.gnn_net import GNNNetConfig, HiveGNN
from hive_gpu.gpu_encoder import GPUGNNEncoder
from hive_gpu.gpu_mcts import GPUMCTSConfig, GPUMCTSOrchestrator


def _time(fn):
    """Run fn(), return (result, elapsed_ms)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000.0


def run_benchmark(
    batch_size: int,
    num_simulations: int,
    net: torch.nn.Module,
    warm_up: bool = True,
) -> dict:
    """Run one benchmark trial; return timing dict."""
    import hive_gpu
    ext = hive_gpu.load_extension()
    encoder = GPUGNNEncoder()

    # Warm-up (avoid cold-start CUDA overhead)
    if warm_up:
        states = ext.create_initial_states(batch_size)
        encoder.encode_batch(states, batch_size)
        ext.generate_legal_mask_batch(states, batch_size)
        with torch.no_grad():
            encoded = encoder.encode_batch(states, batch_size)
            net(encoded)

    # --- Phase-level timings (averaged over num_simulations rounds) ---
    states = ext.create_initial_states(batch_size)

    encode_ms_total = 0.0
    legal_mask_ms_total = 0.0
    nn_ms_total = 0.0

    for _ in range(num_simulations):
        _, e_ms = _time(lambda: encoder.encode_batch(states, batch_size))
        encode_ms_total += e_ms

        _, lm_ms = _time(lambda: ext.generate_legal_mask_batch(states, batch_size))
        legal_mask_ms_total += lm_ms

        encoded = encoder.encode_batch(states, batch_size)
        with torch.no_grad():
            _, nn_ms = _time(lambda: net(encoded))
        nn_ms_total += nn_ms

    # --- Full self-play batch timing ---
    config = GPUMCTSConfig(
        num_simulations=num_simulations,
        batch_size=batch_size,
        max_game_length=30,     # short games for benchmark
        temperature=0.0,
        encoder_type="gnn",
    )
    orchestrator = GPUMCTSOrchestrator(net, config)

    all_examples, total_ms = _time(orchestrator.self_play_batch)

    # Count actual NN evaluations from example counts:
    # each game contributes len(game_examples) moves × num_simulations sims + num_simulations root evals
    total_moves = sum(len(g) for g in all_examples)
    games_per_min = (batch_size / (total_ms / 1000.0)) * 60 if total_ms > 0 else 0
    ms_per_game = total_ms / batch_size if batch_size > 0 else 0

    return {
        "batch_size": batch_size,
        "num_simulations": num_simulations,
        "encode_ms_per_sim": encode_ms_total / num_simulations,
        "legal_mask_ms_per_sim": legal_mask_ms_total / num_simulations,
        "nn_ms_per_sim": nn_ms_total / num_simulations,
        "total_batch_ms": total_ms,
        "ms_per_game": ms_per_game,
        "games_per_min": games_per_min,
        "avg_game_len": total_moves / batch_size if batch_size > 0 else 0,
    }


def print_results(results: list[dict]) -> None:
    """Print a summary table of benchmark results."""
    print(f"\n{'='*80}")
    print("GPU MCTS Benchmark Results")
    print(f"{'='*80}")
    print(
        f"{'B':>6}  {'Sims':>5}  "
        f"{'Enc ms':>8}  {'Mask ms':>8}  {'NN ms':>8}  "
        f"{'Total s':>8}  {'ms/game':>8}  {'games/min':>10}  {'avg moves':>9}"
    )
    print(f"{'-'*80}")
    for r in results:
        print(
            f"{r['batch_size']:>6}  {r['num_simulations']:>5}  "
            f"{r['encode_ms_per_sim']:>8.2f}  "
            f"{r['legal_mask_ms_per_sim']:>8.2f}  "
            f"{r['nn_ms_per_sim']:>8.2f}  "
            f"{r['total_batch_ms']/1000:>8.2f}  "
            f"{r['ms_per_game']:>8.0f}  "
            f"{r['games_per_min']:>10.1f}  "
            f"{r['avg_game_len']:>9.1f}"
        )
    print(f"{'='*80}")
    print("(Enc/Mask/NN are per-simulation averages over isolated kernel calls)")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU MCTS throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--simulations", type=int, default=10, help="MCTS simulations per move.")
    parser.add_argument("--batch-size", type=int, default=32, help="Concurrent games.")
    parser.add_argument("--model-size", choices=["small", "large"], default="small")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep batch sizes 1, 4, 16, 32, 64 (ignores --batch-size).",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available — benchmark requires a GPU.")
        return

    print(f"Loading {args.model_size} GNN model...")
    net_config = GNNNetConfig.large() if args.model_size == "large" else GNNNetConfig.small()
    net = HiveGNN(net_config).cuda().eval()
    param_count = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Device:     {next(net.parameters()).device}")
    print(f"  Simulations per move: {args.simulations}")

    batch_sizes = [1, 4, 16, 32, 64] if args.sweep else [args.batch_size]

    results = []
    for bs in batch_sizes:
        print(f"  Benchmarking batch_size={bs}...", end=" ", flush=True)
        r = run_benchmark(bs, args.simulations, net, warm_up=True)
        results.append(r)
        print(f"{r['games_per_min']:.1f} games/min")

    print_results(results)


if __name__ == "__main__":
    main()
