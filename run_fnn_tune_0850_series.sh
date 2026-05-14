#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hive_neuralnet

mkdir -p checkpoints_fnn_tune0850_base checkpoints_fnn_tune0850_cpuct1 checkpoints_fnn_tune0850_cscale125

python3.11 -u -m hive_fnn.train_fnn \
  --preset large \
  --iterations 855 \
  --games 512 \
  --simulations 2048 \
  --gumbel-considered 16 \
  --batch-size 128 \
  --checkpoint-dir checkpoints_fnn_tune0850_base \
  --checkpoint-keep-every 1 \
  --resume checkpoints_fnn_resume0800_probe_endgame45_1024x2048/hive_fnn_checkpoint_0850.pt \
  --gumbel-wave-parallel \
  --draw-keep-rate 0 \
  --short-forced-win-probe \
  --probe-win-in-one \
  --no-probe-check-opponent-wins \
  --no-probe-win-in-two \
  >> checkpoints_fnn_tune0850_base/training.log 2>&1

python3.11 -u -m hive_fnn.train_fnn \
  --preset large \
  --iterations 860 \
  --games 512 \
  --simulations 2048 \
  --gumbel-considered 16 \
  --batch-size 128 \
  --checkpoint-dir checkpoints_fnn_tune0850_cpuct1 \
  --checkpoint-keep-every 1 \
  --resume checkpoints_fnn_tune0850_base/hive_fnn_checkpoint_0855.pt \
  --gumbel-wave-parallel \
  --draw-keep-rate 0 \
  --short-forced-win-probe \
  --probe-win-in-one \
  --no-probe-check-opponent-wins \
  --no-probe-win-in-two \
  --c-puct 1.0 \
  >> checkpoints_fnn_tune0850_cpuct1/training.log 2>&1

python3.11 -u -m hive_fnn.train_fnn \
  --preset large \
  --iterations 865 \
  --games 512 \
  --simulations 2048 \
  --gumbel-considered 16 \
  --batch-size 128 \
  --checkpoint-dir checkpoints_fnn_tune0850_cscale125 \
  --checkpoint-keep-every 1 \
  --resume checkpoints_fnn_tune0850_cpuct1/hive_fnn_checkpoint_0860.pt \
  --gumbel-wave-parallel \
  --draw-keep-rate 0 \
  --short-forced-win-probe \
  --probe-win-in-one \
  --no-probe-check-opponent-wins \
  --no-probe-win-in-two \
  --c-scale 1.25 \
  >> checkpoints_fnn_tune0850_cscale125/training.log 2>&1
