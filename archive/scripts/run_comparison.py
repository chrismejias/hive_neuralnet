"""
Run GNN vs Transformer comparison training.

Runs transformer first, then GNN, both with identical hyperparameters.
"""
import subprocess
import sys

COMMON_ARGS = [
    "--games", "100",
    "--simulations", "200",
    "--iterations", "20",
    "--model-size", "small",
]

runs = [
    {
        "name": "Transformer",
        "args": ["--encoder-type", "transformer", "--checkpoint-dir", "checkpoints_gpu_transformer"],
        "log": "training_transformer_compare.log",
    },
    {
        "name": "GNN",
        "args": ["--encoder-type", "gnn", "--checkpoint-dir", "checkpoints_gpu_gnn"],
        "log": "training_gnn_compare.log",
    },
]

for run in runs:
    print(f"\n{'='*60}")
    print(f"Starting {run['name']} training")
    print(f"{'='*60}\n", flush=True)

    cmd = [
        sys.executable, "-u", "run_training.py",
        *COMMON_ARGS,
        *run["args"],
    ]

    with open(run["log"], "w") as logfile:
        result = subprocess.run(cmd, stdout=logfile, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"{run['name']} training failed with code {result.returncode}")
    else:
        print(f"{run['name']} training completed successfully")

print("\nComparison complete.")
