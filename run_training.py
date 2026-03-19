"""
Auto-restarting training wrapper.

Runs GPU training and automatically resumes from the latest checkpoint
if the process crashes (e.g., CUDA OOM or illegal memory access).

Usage:
    python run_training.py [--args passed to hive_gpu trainer...]

Example:
    python run_training.py --encoder-type transformer --games 100 --simulations 600 --iterations 30
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    pattern = os.path.join(checkpoint_dir, "hive_gpu_checkpoint_*.pt")
    checkpoints = sorted(glob.glob(pattern))
    return checkpoints[-1] if checkpoints else None


def parse_checkpoint_dir(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--checkpoint-dir" and i + 1 < len(argv):
            return argv[i + 1]
    return "checkpoints_gpu"


def main() -> None:
    argv = sys.argv[1:]
    checkpoint_dir = parse_checkpoint_dir(argv)

    # Strip any --resume arg the user passed (we manage it ourselves)
    clean_argv = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--resume":
            skip_next = True
            continue
        clean_argv.append(arg)

    base_cmd = [sys.executable, "-u", "-m", "hive_gpu"] + clean_argv

    attempt = 0
    while True:
        attempt += 1
        latest = find_latest_checkpoint(checkpoint_dir)
        cmd = base_cmd + (["--resume", latest] if latest else [])

        print(f"\n{'='*60}")
        print(f"Attempt {attempt} — {'resuming from ' + latest if latest else 'starting fresh'}")
        print(f"{'='*60}\n", flush=True)

        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\nTraining completed successfully.")
            break

        print(f"\nProcess exited with code {result.returncode}. Restarting in 5s...", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
