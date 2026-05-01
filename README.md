# Hive Neural Network

AlphaZero-style self-play training for the board game [Hive](https://en.wikipedia.org/wiki/Hive_(game)), focused on the PRS and FNN models plus a GPU-native game engine.

## Architecture Overview

| Model | Package | Params | Self-play |
|-------|---------|--------|-----------|
| PRS Transformer | `hive_prs` | 1.3M / 9.7M | Gumbel-root MCTS (v2 default) |
| FNN (HiveGo-style) | `hive_fnn` | 1.0K – 18.8K | Gumbel-root MCTS / PUCT MCTS |

All models use the same GPU-native game engine (`hive_gpu`) with CUDA kernels for move generation, state encoding, and legal move lookup.

## Requirements

- Python 3.11+
- PyTorch 2.4+ (tested with 2.10.0+cu128)
- CUDA 12.4+ / driver supporting CUDA 12.8

## Setup

### Option A: Docker / Linux (recommended for cloud, RunPod, or WSL2)

```bash
docker build -t hive-neuralnet .
```

The image compiles the CUDA extension automatically at build time through
`hive_gpu.load_extension()`, using `ninja` inside the container and storing
the compiled artifact under `/workspace/.torch_extensions`.

For WSL2 on Windows, make sure these prerequisites are in place before trying
to build:

- `Virtual Machine Platform` enabled
- BIOS/UEFI virtualization enabled
- a WSL2 distro installed, for example `Ubuntu-24.04`

Typical setup:

```powershell
wsl --install --no-distribution
wsl --install -d Ubuntu-24.04
```

If `wsl --install -d Ubuntu-24.04` fails with `HCS_E_HYPERV_NOT_INSTALLED`,
enable virtualization features first and reboot before retrying.

### Option B: Local

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy ninja pytest

# Compile the CUDA extension
pip install -e hive_gpu/
```

Or if a prebuilt `.pyd`/`.so` is already present in `hive_gpu/`, it loads automatically.

## Long-Running Background Training

Use `python -u`, redirect stdin from `/dev/null`, append both stdout and stderr
to a log, and capture the PID. This is more reliable than a bare `nohup`
because startup output is unbuffered and the process does not inherit an
interactive stdin handle.

PRS example:

```bash
cd /workspace/hive_neuralnet
mkdir -p checkpoints_prs_v2

nohup python3.11 -u -m hive_prs.train_prs \
  --iterations 1000 \
  --games 256 \
  --simulations 256 \
  --max-considered 16 \
  --checkpoint-dir checkpoints_prs_v2 \
  --checkpoint-keep-every 50 \
  --resume checkpoints_prs_v2/prs_v2_iter_0500.pt \
  --wave-parallel \
  --deterministic-non-root \
  --virtual-q-penalty 0.25 \
  >> checkpoints_prs_v2/training.log 2>&1 < /dev/null &

echo $!
```

FNN example:

```bash
cd /workspace/hive_neuralnet
mkdir -p checkpoints_fnn

nohup python3.11 -u -m hive_fnn.train_fnn \
  --preset large \
  --iterations 1000 \
  --games 256 \
  --simulations 2048 \
  --gumbel-considered 16 \
  --checkpoint-dir checkpoints_fnn \
  --checkpoint-keep-every 50 \
  --gumbel-wave-parallel \
  --gumbel-deterministic-non-root \
  >> checkpoints_fnn/training.log 2>&1 < /dev/null &

echo $!
```

Check that exactly one intended trainer is running:

```bash
pgrep -af 'hive_prs.train_prs|train_prs|hive_fnn.train_fnn|train_fnn'
tail -f checkpoints_prs_v2/training.log
```

If no process appears and the log does not update, run the same command in the
foreground first to catch startup errors:

```bash
python3.11 -u -m hive_prs.train_prs \
  --iterations 1000 \
  --games 256 \
  --simulations 256 \
  --max-considered 16 \
  --checkpoint-dir checkpoints_prs_v2 \
  --checkpoint-keep-every 50 \
  --resume checkpoints_prs_v2/prs_v2_iter_0500.pt \
Some managed notebook/agent environments kill detached child processes when the
launch command exits, even with `nohup`. In that case, start training inside a
persistent shell, `tmux`, `screen`, or a long-lived PTY session and keep that
session open. The trainer CLI help for PRS and FNN includes model-specific
background-launch templates.
```
## Search Algorithms

### Gumbel AlphaZero (default)

Based on [Danihelka et al., 2022](https://openreview.net/forum?id=bERaNdoegnO). The active PRS and FNN paths use **Gumbel at the root** with real tree search underneath, rather than following the paper literally at every step.

- **On by default** — no flag needed
- Current PRS/FNN defaults standardize on `k=16`
- Rounds = `ceil(log2(k))` — for the current fixed `k=16`, that is 4 rounds
- Scales well with VRAM: run more games in parallel, not more serial rounds

#### Project Deviations From The Original Algorithm

The current project makes two deliberate changes relative to the original paper:

1. **Non-root nodes use PUCT MCTS instead of pure policy-preserving sampling.**
   The paper's non-root rule is closer to direct policy improvement. In this codebase, the active PRS and FNN search paths use non-root PUCT MCTS because it was empirically stronger than pure policy sampling.

2. **The active FNN trainer uses a visited-only Gumbel policy target.**
   For FNN self-play, legal moves outside the searched Gumbel candidate set receive zero target mass instead of keeping their raw prior. This outperformed the prior-preserving variant empirically.

So the active FNN target is:

- **Visited/searched moves**
  ```
  π_imp(a ∈ S) ∝ exp(log π₀(a) + σ · Q_mcts(a))
  ```
- **Unvisited / not-in-S moves**
  ```
  π_imp(a ∉ S) = 0
  ```

PRS v2 differs slightly here: it still uses a **prior-anchored** target for unsampled legal moves while sharing the same Gumbel-root + non-root PUCT tree-search structure.

## PRS Transformer (Piece-Relative Space, v2 default)

The default PRS training path now uses **PRS v2**: a structured **813-slot legal-masked head** over dynamic legal move structure. This replaced the old fixed 6,841-action v1 head as the default `train_prs` path.

### Architecture

- **Policy head**: Structured 813-slot legal-masked head (dynamic legal mapping per state)
- **Small config** (default): 1.3M parameters — d\_model=128, heads=8, layers=6, dim\_ff=512
- **Large config**: 9.7M parameters — d\_model=256, heads=16, layers=8, dim\_ff=1024

Legacy PRS v1 modules are archived under `archive/legacy_prs_v1`.

### Search Pattern

- **Default:** Gumbel-root MCTS tree search (`PRSMCTSOrchestratorV2`)
- Sequential-halving rounds use final Gumbel sigma scoring (`gumbel + logits + sigma * Q`) for the played move, rather than visit-count argmax.
- Wave-parallel MCTS is enabled by default with a hard-coded per-round schedule: `1, 2, 4, 8`. Use `--no-wave-parallel` for pure serial waves.
- `--deterministic-non-root` enables the paper-style deterministic inner selection, with `--virtual-q-penalty` used to diversify wave-parallel sims.
- The current training/profile defaults use `k=16`.
- PRS enables TF32 matmul precision for the transformer trunk on supported GPUs.
- `--compile-forward` can opt in to `torch.compile` for the tensor-only trunk/head path; it is off by default because Inductor can be unstable on some hosts.
- Current `train_prs` path is v2-only; legacy PRS-v1 search paths are archived.
- Move-cap draws are excluded from value loss, while genuine draws are retained.
- `profile_models.py` exposes `prs-small` and `prs-large` for direct preset comparison.

### Training

```bash
# Quick benchmark (1 iteration, 128 games, 256 sims, k=16)
python -m hive_prs.train_prs \
  --iterations 1 --games 128 --simulations 256 --max-considered 16

# Full training run
python -m hive_prs.train_prs \
  --iterations 1500 --games 128 --simulations 512 --max-considered 16 \
  --checkpoint-dir checkpoints_prs_v2

# Large model
python -m hive_prs.train_prs \
  --d-model 256 --num-heads 16 --num-layers 8 --dim-ff 1024 \
  --iterations 1500 --games 128 --simulations 512 --max-considered 16 \
  --checkpoint-dir checkpoints_prs_large

# Compare small vs large
python -m profile_models --models prs-small prs-large --games 256 --sims 256 --gumbel-k 16
```

### Performance (RTX 4090, small model)

| Config | Self-play | Training | Total |
|--------|-----------|----------|-------|
| 128 games, 256 sims, k=16 | ~60s | ~34s | ~94s/iter |

### PRS Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 1500 | Training iterations |
| `--games` | 128 | Parallel self-play games per iteration |
| `--simulations` | 512 | Gumbel simulation budget per move |
| `--max-considered` | 16 | Root actions considered (k); rounds = ceil(log2(k)) |
| `--wave-parallel` / `--no-wave-parallel` | on | Enable/disable PRS v2 per-round MCTS wave schedule (`1,2,4,8`) |
| `--deterministic-non-root` / `--no-deterministic-non-root` | off | Use paper-style deterministic inner-node selection |
| `--virtual-q-penalty` | 0.25 | Temporary Q penalty used to diversify deterministic wave-parallel sims |
| `--compile-forward` / `--no-compile-forward` | off | Opt in to `torch.compile` for tensor-only trunk/head forward |
| `--d-model` | 128 | Transformer hidden dimension |
| `--num-heads` | 8 | Attention heads |
| `--num-layers` | 6 | Transformer layers |
| `--dim-ff` | 512 | Feed-forward hidden dimension |
| `--expansion-mask` | 7 | Expansion piece mask: 0=base, 1=+Mosquito, 2=+Ladybug, 4=+Pillbug, 7=all |
| `--augment-prob` | 0.5 | Probability of C6 rotational augmentation per training batch |
| `--buffer-size` | 150000 | Replay buffer capacity |
| `--checkpoint-dir` | checkpoints\_prs\_v2 | Checkpoint output directory |
| `--resume` | — | Path to checkpoint to resume from |

---

## FNN (HiveGo-style Feedforward Network)

A tiny feedforward network inspired by HiveGo's AlphaZeroFNN. The key idea: encode root + all successor states independently, then score each action as `f(root_emb ‖ successor_emb)`. No action-space enumeration, no attention — just board features.

### Architecture

- **Board encoder**: 110-dim features → hidden → embedding (Linear + sigmoid + LayerNorm)
- **Value head**: embedding → Linear → tanh
- **Action scoring**: concat [root\_emb, successor\_emb] → tower → logit
- **Small** (~1.0K params): hidden=8, embed=8, action\_hidden=8
- **Medium** (~6.3K params): hidden=32, embed=32, action\_hidden=32
- **Large** (~18.8K params, current default): hidden=64, embed=64, action\_hidden=64

### 110-dim Feature Vector (per state, extracted by CUDA kernel)

For each of 8 piece types × 2 players (16 entries):
- `count_on_board`
- `count_in_hand`
- `queen_neighbors`
- `avg_dist_to_opp_queen`
- `can_move_count`
- `articulation_count`

Additional per-color features:
- `num_single`
- `queen_covered`
- `num_placement_pos`
- `pillbug_capable`
- `throwable_own`
- `throwable_opp`

Global scalar features:
- `moves_to_draw`
- `move_number`

### GPU-Native Self-Play (Fused CUDA Kernel)

The entire Gumbel AlphaZero game loop — move generation, feature extraction, FNN forward pass, Sequential Halving, move application, and game termination — runs in a **single CUDA kernel** with one block per game. This eliminates all Python overhead during self-play.

**Benchmark (RTX 4090, small preset, 32 games, 1024 sims, k=16):**

| Stage | Python orchestrator | CUDA kernel | Speedup |
|-------|--------------------:|------------:|---------|
| Self-play | 3.63s | 0.61s | **5.9×** |

### Search Patterns

- **Default:** Gumbel-root MCTS tree search (`FNNMCTSOrchestrator`)
- Gumbel MCTS uses a hard-coded per-round wave schedule `2, 4, 8, 16` by default; use `--no-gumbel-wave-parallel` for pure serial waves.
- The current Gumbel implementations standardize on `k=16`.
- Within the Gumbel search, inner-node selection uses standard non-root PUCT MCTS.
- The default non-root/root PUCT exploration constant for FNN search is `c_puct = 1.25`.
- Policy target uses the visited-only improved policy used by the strongest FNN nondeterministic checkpoints: unvisited moves receive zero policy mass.
- **Alternative:** plain PUCT MCTS (`FNNPUCTOrchestrator`) via `--puct`
- PUCT MCTS uses wave-parallel virtual-loss search with `--puct-wave-size` (default 16).
- Move-cap draws are excluded from value loss and can be dropped from policy training with `--draw-keep-rate 0.0`; genuine draws are retained.

### Training

```bash
# Quick smoke test (1 iteration, large model)
python -m hive_fnn.train_fnn \
  --preset large --games 32 --simulations 64 --gumbel-considered 16 \
  --iterations 1

# Full training run (large model)
python -m hive_fnn.train_fnn \
  --preset large \
  --games 128 --simulations 1024 --gumbel-considered 16 \
  --iterations 1500 \
  --checkpoint-dir checkpoints_fnn

# Large model with default Gumbel-root + non-root PUCT MCTS
python -m hive_fnn.train_fnn \
  --preset large \
  --games 512 --simulations 2048 --gumbel-considered 16 \
  --gumbel-wave-parallel \
  --buffer-size 200000 \
  --iterations 1000 \
  --checkpoint-dir checkpoints_fnn

# Medium model
python -m hive_fnn.train_fnn \
  --preset medium \
  --games 128 --simulations 512 --gumbel-considered 16 \
  --iterations 1500 \
  --checkpoint-dir checkpoints_fnn_medium
```

The bare `train_fnn` defaults now map to the large configuration (`64/64/64`).

### FNN Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | — | `small` / `medium` / `large` (overrides `--hidden-dim` etc.) |
| `--hidden-dim` | 64 | Board encoder hidden dimension |
| `--embed-dim` | 64 | Board embedding dimension |
| `--action-hidden` | 64 | Action tower hidden dimension |
| `--iterations` | 1500 | Training iterations |
| `--games` | 128 | Parallel self-play games per iteration |
| `--simulations` | 128 | Gumbel simulation budget per move |
| `--gumbel-considered` | 16 | Root actions considered (k) |
| `--gumbel-wave-parallel` / `--no-gumbel-wave-parallel` | on | Enable/disable FNN Gumbel per-round MCTS wave schedule (`2,4,8,16`) |
| `--buffer-size` | 100000 | Replay buffer capacity (200000 recommended for better generalization) |
| `--puct-wave-size` | 16 | Parallel MCTS simulations per wave for plain PUCT |
| `--puct` | off | Use plain PUCT MCTS root policy instead of Gumbel root halving |
| `--checkpoint-dir` | checkpoints\_fnn | Checkpoint output directory |

## Diagnostics

### Win-in-one probe

Measures whether the network correctly identifies forced wins:

```bash
python probe_win_in_one.py
# or target a specific checkpoint:
python probe_win_in_one.py --checkpoint checkpoints/hive_gpu_checkpoint_0124.pt
```

### Value head evaluation

```bash
python eval_value_head.py
```

### FNN MCTS search diagnostics

Inspect FNN non-root search behavior (depth distribution, visit entropy, new-leaf rate):

```bash
python fnn_search_diagnostic_nonroot.py \
    --checkpoint checkpoints_fnn/hive_fnn_checkpoint_0200.pt \
    --sims 512 --positions 8
```

### PRS MCTS search diagnostics

The same comparison is available for PRS v2 checkpoints:

```bash
python prs_search_diagnostic_nonroot.py \
    --checkpoint checkpoints_prs_v2/prs_v2_iter_0600.pt \
    --sims 256 --positions 2
```

### PRS improved-policy mass analysis

Measure how much policy mass lands on MCTS-searched vs unsampled moves, and diagnose value-head calibration relative to Q_mcts:

```bash
# Mass vs sim count (four sweep: 256/512/1024/2048)
python prs_mass_diagnostic.py \
    --checkpoint checkpoints_prs_v2/prs_v2_iter_0600.pt \
    --positions 30
```

### FNN improved-policy mass analysis

Measure how much policy mass lands on MCTS-searched vs unsampled moves, and diagnose value-head calibration relative to Q_mcts:

```bash
# Mass vs sim count (four sweep: 256/512/1024/2048)
python fnn_mass_diagnostic.py \
    --checkpoint checkpoints_fnn/hive_fnn_checkpoint_0200.pt \
    --positions 30

# Compare three fallback strategies for unsampled moves (v_pi / child_init_q / 0.0)
python fnn_mass_fallback_compare.py \
    --checkpoint checkpoints_fnn/hive_fnn_checkpoint_0200.pt \
    --positions 50 --sims 2048

# Verify the prior-anchored policy target vs old v_pi approach
python fnn_mass_new_policy.py \
    --checkpoint checkpoints_fnn/hive_fnn_checkpoint_0200.pt \
    --positions 50 --sims 2048
```

## GUI

Play against the trained AI using pygame (local only):

```bash
# vs AI
python gui.py --checkpoint checkpoints_fnn/hive_fnn_checkpoint_0200.pt

# AI self-play
python gui.py --self-play
```

Requires `pygame-ce`: `pip install pygame-ce`

Controls: click hand panel to select piece, click highlighted hex to place/move, `R` to restart, `Esc` to quit.

## Running Tests

```bash
python -m pytest tests/
```

Test coverage includes the game engine, PRS/FNN paths, and GPU move generation validation.

## Project Structure

```
hive_engine/       # Core game rules (board, pieces, game state, hex grid)
hive_transformer/  # Shared token batch types + archived transformer compatibility wrappers
hive_gpu/          # CUDA extension, GPU-native MCTS/Gumbel, shared kernels
  csrc/            # CUDA C++ source
    game_logic.cu      # Move gen, state apply, feature extraction, fused self-play
    fnn_features.cuh   # 110-dim FNN feature extraction kernel
    fnn_selfplay.cuh   # Fused Gumbel self-play kernel (FNN)
    state_encoder.cuh  # Transformer state encoding
    mcts_tree.cuh      # GPU-native MCTS tree
hive_prs/          # PRS v2 default (structured 813-slot head)
  prs_transformer_v2.py        # HivePRSTransformerV2
  prs_mcts_orchestrator_v2.py  # PRS v2 Gumbel-root MCTS
  prs_trainer_v2.py            # PRS v2 trainer
  train_prs.py                 # CLI entry point (v2 default)
hive_fnn/          # HiveGo-style FNN with multiple search paths
  fnn_network.py       # HiveFNN: shared encoder + value head + action tower
  fnn_mcts_orchestrator.py  # Gumbel-root MCTS tree search
  fnn_puct_orchestrator.py  # Plain PUCT MCTS tree search
  fnn_trainer.py       # FNNTrainer training loop
  train_fnn.py         # CLI entry point
tests/             # Test suite (250+ tests)
archive/           # Archived legacy models, diagnostics, and old CPU training code
Dockerfile         # Container for cloud/RunPod deployment
```
