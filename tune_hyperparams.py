"""
Sequential hyperparameter search for GPU trainer.

Runs each candidate config for a fixed number of probe iterations
starting from the same base checkpoint, records loss curves and wall time,
and reports which config learned fastest per unit time.

Usage:
    python tune_hyperparams.py --base-checkpoint <path> [options]

Example:
    python tune_hyperparams.py \\
        --base-checkpoint checkpoints_gpu_transformer/hive_gpu_checkpoint_0024.pt \\
        --probe-iters 5 --games 100 --encoder-type transformer \\
        --sim-counts 100 200 400 --batch-sizes 64 128 256
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from itertools import product


@dataclass
class ProbeResult:
    config: dict
    losses: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    final_loss: float = float("inf")
    loss_drop: float = 0.0        # losses[0] - losses[-1], higher = better
    drop_per_hour: float = 0.0    # loss_drop / (elapsed_s / 3600)
    error: str = ""

    def label(self) -> str:
        return "  ".join(f"{k}={v}" for k, v in self.config.items())


def run_probe(
    base_checkpoint: str,
    probe_dir: str,
    config: dict,
    base_args: list[str],
    probe_iters: int,
) -> ProbeResult:
    """Run one probe: copy base checkpoint -> train N iters -> record losses."""
    result = ProbeResult(config=config)

    os.makedirs(probe_dir, exist_ok=True)
    ckpt_num = re.search(r"_(\d+)\.pt$", base_checkpoint)
    iter_num = int(ckpt_num.group(1)) if ckpt_num else 0
    dest = os.path.join(probe_dir, f"hive_gpu_checkpoint_{iter_num:04d}.pt")
    shutil.copy(base_checkpoint, dest)

    cmd = [
        sys.executable, "-u", "-m", "hive_gpu",
        "--iterations", str(probe_iters),
        "--checkpoint-dir", probe_dir,
        "--resume", dest,
    ] + base_args

    for key, flag in [
        ("lr", "--lr"),
        ("lr_schedule", "--lr-schedule"),
        ("simulations", "--simulations"),
        ("batch_size", "--batch-size"),
        ("epochs", "--epochs"),
        ("games", "--games"),
    ]:
        if key in config:
            cmd += [flag, str(config[key])]

    log_path = os.path.join(probe_dir, "probe.log")
    print(f"\n  Running: {result.label()}", flush=True)
    print(f"  Log: {log_path}", flush=True)
    t0 = time.time()

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

    result.elapsed_s = time.time() - t0

    if proc.returncode != 0:
        result.error = f"exit code {proc.returncode}"
        print(f"  FAILED ({result.error}) in {result.elapsed_s:.0f}s", flush=True)
        return result

    # Parse losses from log
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Training: loss=([-\d.]+)", line)
            if m:
                result.losses.append(float(m.group(1)))

    if result.losses:
        result.final_loss = result.losses[-1]
        result.loss_drop = result.losses[0] - result.losses[-1]
        hours = result.elapsed_s / 3600.0
        result.drop_per_hour = result.loss_drop / hours if hours > 0 else 0.0
        losses_str = ", ".join(f"{l:.3f}" for l in result.losses)
        print(
            f"  Done in {result.elapsed_s:.0f}s  "
            f"losses=[{losses_str}]  "
            f"drop={result.loss_drop:+.3f}  "
            f"drop/hr={result.drop_per_hour:+.2f}",
            flush=True,
        )
    else:
        result.error = "no loss found in log"
        print(f"  No losses found. Check {log_path}", flush=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential hyperparameter search.")
    parser.add_argument("--base-checkpoint", required=True,
                        help="Checkpoint to start all probes from.")
    parser.add_argument("--probe-iters", type=int, default=5,
                        help="Training iterations per probe.")
    parser.add_argument("--output-dir", default="hp_search",
                        help="Root dir for probe checkpoint dirs and results.")
    parser.add_argument("--encoder-type", default="transformer",
                        choices=["transformer", "gnn"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--model-size", default="small")

    # Search space
    parser.add_argument("--lrs", nargs="+", type=float, default=[2e-4],
                        help="Learning rates to try.")
    parser.add_argument("--lr-schedules", nargs="+", default=["constant"],
                        help="LR schedules to try (constant, cosine).")
    parser.add_argument("--sim-counts", nargs="+", type=int, default=None,
                        help="Simulation counts to sweep.")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="Batch sizes to sweep.")
    parser.add_argument("--rank-by", default="drop_per_hour",
                        choices=["drop_per_hour", "final_loss", "loss_drop"],
                        help="Metric to rank configs by.")
    args = parser.parse_args()

    sim_counts = args.sim_counts or [args.simulations]
    batch_sizes = args.batch_sizes or [None]
    lrs = args.lrs
    schedules = args.lr_schedules

    # Build search space (cartesian product)
    search_configs: list[dict] = []
    for lr, sched, sims, bs in product(lrs, schedules, sim_counts, batch_sizes):
        cfg: dict = {}
        if len(lrs) > 1 or lr != 2e-4:
            cfg["lr"] = lr
        if sched != "constant":
            cfg["lr_schedule"] = sched
        if len(sim_counts) > 1:
            cfg["simulations"] = sims
        if bs is not None and (len(batch_sizes) > 1 or bs != 64):
            cfg["batch_size"] = bs
        if not cfg:
            cfg["simulations"] = sims  # always show something
        search_configs.append(cfg)

    base_args = [
        "--encoder-type", args.encoder_type,
        "--games", str(args.games),
        "--simulations", str(args.simulations),
        "--model-size", args.model_size,
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Hyperparameter Search")
    print(f"{'='*60}")
    print(f"Base checkpoint: {args.base_checkpoint}")
    print(f"Probe iterations: {args.probe_iters}")
    print(f"Configs to test: {len(search_configs)}")
    print(f"Ranking by: {args.rank_by}")
    print(f"{'='*60}\n")

    results: list[ProbeResult] = []

    for i, config in enumerate(search_configs):
        probe_dir = os.path.join(args.output_dir, f"probe_{i:03d}")
        print(f"\n[{i+1}/{len(search_configs)}] {ProbeResult(config).label()}")
        result = run_probe(
            args.base_checkpoint,
            probe_dir,
            config,
            base_args,
            args.probe_iters,
        )
        results.append(result)

        # Save incremental results
        summary = [
            {
                "config": r.config,
                "losses": r.losses,
                "elapsed_s": r.elapsed_s,
                "final_loss": r.final_loss,
                "loss_drop": r.loss_drop,
                "drop_per_hour": r.drop_per_hour,
                "error": r.error,
            }
            for r in results
        ]
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Final ranking
    valid = [r for r in results if not r.error and r.losses]

    rank_key = {
        "drop_per_hour": lambda r: -r.drop_per_hour,
        "final_loss":    lambda r:  r.final_loss,
        "loss_drop":     lambda r: -r.loss_drop,
    }[args.rank_by]
    valid.sort(key=rank_key)

    print(f"\n{'='*60}")
    print(f"Results (ranked by {args.rank_by}):")
    print(f"{'='*60}")
    print(f"  {'Config':<30} {'Time':>6}  {'Drop':>6}  {'Drop/hr':>7}  {'Final':>7}")
    print(f"  {'-'*30} {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")
    for r in valid:
        h = r.elapsed_s / 3600
        time_str = f"{h:.2f}h"
        print(
            f"  {r.label():<30} {time_str:>6}  "
            f"{r.loss_drop:+6.3f}  {r.drop_per_hour:+7.2f}  {r.final_loss:7.3f}"
        )

    if valid:
        best = valid[0]
        print(f"\nBest config: {best.label()}")
        print(f"  Final loss: {best.final_loss:.3f}  drop: {best.loss_drop:+.3f}  drop/hr: {best.drop_per_hour:+.2f}")

    failed = [r for r in results if r.error]
    if failed:
        print(f"\nFailed probes:")
        for r in failed:
            print(f"  {r.label()}: {r.error}")

    print(f"\nFull results saved to {os.path.join(args.output_dir, 'results.json')}")


if __name__ == "__main__":
    main()
