# Hive Neural Network

AlphaZero-style self-play training for the board game [Hive](https://en.wikipedia.org/wiki/Hive_(game)), with four distinct network architectures and a GPU-native game engine.

## Architecture Overview

| Model | Package | Params | Self-play |
|-------|---------|--------|-----------|
| Transformer | `hive_gpu` | ~9M | GPU-native batched Gumbel |
| PRS Transformer | `hive_prs` | 1.3M / 9.7M | GPU-native batched Gumbel |
| FNN (HiveGo-style) | `hive_fnn` | 0.6K – 15K | **Fused CUDA kernel** (5.9× faster) |
| Move-Conditioned Transformer | `hive_mc` | ~1.1M / ~5M | GPU-native batched Gumbel |

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

## Search Algorithms

### Gumbel AlphaZero (default)

Based on [Danihelka et al., 2022](https://openreview.net/forum?id=bERaNdoegnO). Uses Sequential Halving with Gumbel noise to select and evaluate a fixed budget of candidate actions. All NN evaluations within a round are fully independent and batched — no serial tree traversal bottleneck.

- **On by default** — no flag needed
- Use `--gumbel-considered` to set the number of actions considered at the root (k, default 16)
- Rounds = ceil(log2(k)) — e.g. k=16 → 4 rounds
- Scales well with VRAM: run more games in parallel, not more serial rounds

### Wave-parallel MCTS (opt-out, Transformer only)

GPU-native tree search with W simulations per wave. Virtual loss diversifies selections within each wave. Use `--wave-size` to control parallelism.

- Disable Gumbel and use MCTS with `--no-gumbel`
- Use `--wave-size` to tune parallelism (default 8; increase for larger GPUs)

---

## Transformer (position-based)

The original model — a transformer encoder operating on the full board token sequence, with policy, value, and queen-surround auxiliary heads.

- **Network**: Transformer encoder (~9M parameters) with policy, value, and queen-surround auxiliary heads
- **Action space**: 29,407 position-absolute actions
- **Self-play**: Batched parallel games on GPU; Gumbel AlphaZero by default

### Running Training

```bash
python -m hive_gpu \
  --encoder-type transformer \
  --games 256 \
  --simulations 128 \
  --gumbel-considered 16 \
  --expansion -1 \
  --iterations 100 \
  --checkpoint-dir checkpoints \
  --log-file training.log
```

### Resume from checkpoint

```bash
python -m hive_gpu \
  --encoder-type transformer \
  --games 256 \
  --simulations 128 \
  --gumbel-considered 16 \
  --expansion -1 \
  --iterations 100 \
  --resume checkpoints/hive_gpu_checkpoint_0124.pt \
  --checkpoint-dir checkpoints \
  --log-file training.log
```

### Docker (RunPod / Linux)

```bash
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  hive-neuralnet \
  python -m hive_gpu \
    --encoder-type transformer \
    --games 512 \
    --simulations 128 \
    --gumbel-considered 16 \
    --expansion -1 \
    --endgame-frac 1.0 \
    --endgame-surround 5 \
    --resume checkpoints/hive_gpu_checkpoint_0124.pt \
    --checkpoint-dir checkpoints \
    --log-file checkpoints/training.log
```

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--games` | 64 | Parallel self-play games per iteration |
| `--simulations` | 100 | MCTS simulations per move (or Gumbel simulation budget) |
| `--iterations` | 20 | Total training iterations |
| `--no-gumbel` | — | Disable Gumbel and fall back to wave-parallel MCTS |
| `--wave-size` | 8 | Parallel MCTS sims per GPU wave (only with `--no-gumbel`) |
| `--gumbel-considered` | 16 | Number of root actions considered (k); rounds = ceil(log2(k)) |
| `--expansion` | 0 | Expansion piece mask: 0=base, 1=+Mosquito, 2=+Ladybug, 4=+Pillbug, 7=all, -1=random |
| `--endgame-frac` | 0.0 | Fraction of games starting from endgame positions (0–1) |
| `--draw-keep-rate` | 1.0 | Fraction of drawn games kept for training |
| `--lr` | 0.0002 | Learning rate |
| `--batch-size` | 256 | Training batch size |
| `--resume` | — | Path to checkpoint to resume from |

---

## PRS Transformer (Piece-Relative Space)

A smaller model that operates in a **piece-relative action space** of 6,841 actions instead of the engine's position-absolute actions. Each action is encoded as *(actor token, reference token, direction)* — a representation that is invariant to board translation and rotation, generalising better from limited data.

### Architecture

- **Action space**: 6,841 = 5,488 MOVE (28×28×7) + 1,344 PLACE (8×28×6) + 8 FIRST\_PLACE + 1 PASS
- **Policy head**: Factored bilinear scoring — `score(actor, ref, dir) = (Wₐ hᵢ) · ((W_r hⱼ) ⊙ dir_emb[k])` — two `bmm` calls, O(B × tokens × 7 × dk) memory
- **Small config** (default): 1.3M parameters — d\_model=128, heads=8, layers=6, dim\_ff=512
- **Large config**: 9.7M parameters — d\_model=256, heads=16, layers=8, dim\_ff=1024

### Recent Fixes

- **Multi-representation fix**: Each board position can correspond to multiple PRS actions; policy targets now correctly aggregate probability mass across all representations of the same board move, preventing conflicting gradients
- **Gumbel Q-transform fix**: Corrected inversion sign and temperature sampling logic
- **~2.5× self-play speedup**: Fused NN evaluation, sparse Gumbel topk, GPU-to-GPU state indexing

### Training

```bash
# Quick benchmark (1 iteration, 128 games, 256 sims, k=16)
python -m hive_prs.train_prs \
  --iterations 1 --games 128 --simulations 256 --max-considered 16

# Full training run
python -m hive_prs.train_prs \
  --iterations 1500 --games 128 --simulations 512 --max-considered 16 \
  --checkpoint-dir checkpoints_prs

# Large model
python -m hive_prs.train_prs \
  --d-model 256 --num-heads 16 --num-layers 8 --dim-ff 1024 \
  --iterations 1500 --games 128 --simulations 512 --max-considered 16 \
  --checkpoint-dir checkpoints_prs_large
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
| `--d-model` | 128 | Transformer hidden dimension |
| `--num-heads` | 8 | Attention heads |
| `--num-layers` | 6 | Transformer layers |
| `--dim-ff` | 512 | Feed-forward hidden dimension |
| `--d-key` | 64 | Bilinear key dimension in policy head |
| `--checkpoint-dir` | checkpoints\_prs | Checkpoint output directory |
| `--resume` | — | Path to checkpoint to resume from |

---

## FNN (HiveGo-style Feedforward Network)

A tiny feedforward network inspired by HiveGo's AlphaZeroFNN. The key idea: encode root + all successor states independently, then score each action as `f(root_emb ‖ successor_emb)`. No action-space enumeration, no attention — just board features.

### Architecture

- **Board encoder**: 88-dim features → hidden → embedding (Linear + sigmoid + LayerNorm)
- **Value head**: embedding → Linear → tanh
- **Action scoring**: concat [root\_emb, successor\_emb] → tower → logit
- **Small** (~600 params): hidden=8, embed=8, action\_hidden=8
- **Medium** (~5K params): hidden=32, embed=32, action\_hidden=32
- **Large** (~15K params): hidden=64, embed=64, action\_hidden=64

### 88-dim Feature Vector (per state, extracted by CUDA kernel)

For each of 8 piece types × 2 players (16 entries):
- `count_on_board`, `count_in_hand`, `queen_neighbors`, `avg_dist_to_opp_queen`, `can_move`, `num_single`

Plus 8 global/per-player features:
- `queen_covered`, `num_placement_positions`, `moves_to_draw`, `move_number`

### GPU-Native Self-Play (Fused CUDA Kernel)

The entire Gumbel AlphaZero game loop — move generation, feature extraction, FNN forward pass, Sequential Halving, move application, and game termination — runs in a **single CUDA kernel** with one block per game. This eliminates all Python overhead during self-play.

**Benchmark (RTX 4090, small preset, 32 games, 1024 sims, k=16):**

| Stage | Python orchestrator | CUDA kernel | Speedup |
|-------|--------------------:|------------:|---------|
| Self-play | 3.63s | 0.61s | **5.9×** |

### Training

```bash
# Quick smoke test (1 iteration, small model)
python -m hive_fnn.train_fnn \
  --preset small --games 32 --simulations 64 --gumbel-considered 16 \
  --iterations 1

# Full training run (small model)
python -m hive_fnn.train_fnn \
  --preset small \
  --games 128 --simulations 1024 --gumbel-considered 16 \
  --iterations 1500 \
  --checkpoint-dir checkpoints_fnn

# Medium model
python -m hive_fnn.train_fnn \
  --preset medium \
  --games 128 --simulations 512 --gumbel-considered 16 \
  --iterations 1500 \
  --checkpoint-dir checkpoints_fnn_medium
```

### FNN Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | — | `small` / `medium` / `large` (overrides `--hidden-dim` etc.) |
| `--hidden-dim` | 32 | Board encoder hidden dimension |
| `--embed-dim` | 32 | Board embedding dimension |
| `--action-hidden` | 32 | Action tower hidden dimension |
| `--iterations` | 1500 | Training iterations |
| `--games` | 128 | Parallel self-play games per iteration |
| `--simulations` | 128 | Gumbel simulation budget per move |
| `--gumbel-considered` | 16 | Root actions considered (k) |
| `--checkpoint-dir` | checkpoints\_fnn | Checkpoint output directory |

---

## Move-Conditioned Transformer (MC)

A two-stage transformer that avoids full successor encoding for every candidate action. A lightweight screening head quickly scores all legal moves from compact features; only the top-k candidates are fully encoded by the action head.

### Architecture

- **Trunk**: Standard transformer encoder (token sequence of board pieces)
- **Screening head**: Root CLS + compact move features (piece type, move type, from/to position embeddings) → logit; scores all legal moves cheaply
- **Action head**: Successor CLS vs root CLS comparison → logit; applied only to top-k candidates
- **Small** (~1.1M params): d=128, heads=8, layers=3, ff=512
- **Large** (~5M params): d=256, heads=8, layers=6, ff=1024

This achieves 3–5× compute savings over evaluating all successors with the full transformer.

### Training

```bash
# Quick smoke test
python -m hive_mc.train_mc \
  --iterations 1 --games 32 --simulations 64 --gumbel-considered 16

# Full training run
python -m hive_mc.train_mc \
  --iterations 1500 --games 128 --simulations 256 --gumbel-considered 16 \
  --checkpoint-dir checkpoints_mc

# Large model
python -m hive_mc.train_mc \
  --d-model 256 --num-layers 6 \
  --iterations 1500 --games 128 --simulations 256 --gumbel-considered 16 \
  --checkpoint-dir checkpoints_mc_large
```

### MC Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--d-model` | 128 | Transformer hidden dimension |
| `--num-heads` | 8 | Attention heads |
| `--num-layers` | 3 | Transformer layers |
| `--dim-ff` | 512 | Feed-forward hidden dimension |
| `--max-candidates` | 16 | Top-k candidates passed from screening to action head |
| `--iterations` | 1500 | Training iterations |
| `--games` | 128 | Parallel self-play games per iteration |
| `--simulations` | 128 | Gumbel simulation budget per move |
| `--checkpoint-dir` | checkpoints\_mc | Checkpoint output directory |

---

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

## GUI

Play against the trained AI using pygame (local only):

```bash
# vs AI
python gui.py --checkpoint checkpoints/hive_gpu_checkpoint_0124.pt

# AI self-play
python gui.py --self-play
```

Requires `pygame-ce`: `pip install pygame-ce`

Controls: click hand panel to select piece, click highlighted hex to place/move, `R` to restart, `Esc` to quit.

## Running Tests

```bash
python -m pytest tests/
```

250+ tests covering the game engine, all four networks, and GPU move generation validation.

## Project Structure

```
hive_engine/       # Core game rules (board, pieces, game state, hex grid)
hive_transformer/  # Transformer network and token encoder
hive_gpu/          # CUDA extension, GPU-native MCTS/Gumbel, transformer trainer
  csrc/            # CUDA C++ source
    game_logic.cu      # Move gen, state apply, feature extraction, fused self-play
    fnn_features.cuh   # 88-dim FNN feature extraction kernel
    fnn_selfplay.cuh   # Fused Gumbel self-play kernel (FNN)
    state_encoder.cuh  # Transformer state encoding
    mcts_tree.cuh      # GPU-native MCTS tree
hive_prs/          # PRS Transformer: piece-relative action space (6,841 actions)
  action_space.py      # Action encoding/decoding, vectorised batch conversion
  prs_encoder.py       # PRSEncoder: board state → token sequence
  prs_transformer.py   # HivePRSTransformer + bilinear policy head
  prs_orchestrator.py  # PRSGumbelOrchestrator: batched Gumbel self-play
  prs_trainer.py       # PRSTrainer training loop
  train_prs.py         # CLI entry point
hive_fnn/          # HiveGo-style FNN with GPU-native fused self-play
  fnn_network.py       # HiveFNN: shared encoder + value head + action tower
  fnn_features.py      # Python feature encoder (calls CUDA kernel)
  fnn_orchestrator.py  # FNNCudaOrchestrator: calls fused CUDA self-play kernel
  fnn_trainer.py       # FNNTrainer training loop
  train_fnn.py         # CLI entry point
hive_mc/           # Move-conditioned transformer (two-stage: screen → full)
  mc_transformer.py    # HiveMoveTransformer + screening + action heads
  mc_orchestrator.py   # MCGumbelOrchestrator: batched Gumbel self-play
  mc_trainer.py        # MCTrainer training loop
  train_mc.py          # CLI entry point
tests/             # Test suite (250+ tests)
checkpoints/       # Pretrained checkpoints
archive/           # Archived CPU training code (GNN, NNUE, CPU MCTS)
Dockerfile         # Container for cloud/RunPod deployment
```
