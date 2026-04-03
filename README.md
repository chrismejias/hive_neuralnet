# Hive Neural Network

AlphaZero-style self-play training for the board game [Hive](https://en.wikipedia.org/wiki/Hive_(game)), using a GPU-native transformer with Gumbel AlphaZero search.

## Architecture

- **Network**: Transformer encoder (~9M parameters) with policy, value, and queen-surround auxiliary heads
- **Search**: Gumbel AlphaZero (default) — Sequential Halving with Gumbel noise, fully batched NN evaluations per round; or GPU-native MCTS with CUDA kernels (opt-out via `--no-gumbel`)
- **Self-play**: Batched parallel games entirely on GPU; no Python loop overhead
- **Expansion pieces**: Supports Mosquito, Ladybug, Pillbug (configurable or random)
- **Endgame curriculum**: Optionally start games from near-decided positions to sharpen tactical play

## Requirements

- Python 3.14
- PyTorch 2.10.0
- CUDA 12.8 / cuDNN 9.1.0

## Setup

### Option A: Docker / Linux (recommended for cloud, RunPod, or WSL2)

```bash
docker build -t hive-neuralnet .
```

The image compiles the CUDA extension automatically at build time through
`hive_gpu.load_extension()`, using `ninja` inside the container and storing
the compiled artifact under `/workspace/.torch_extensions`.

If you are moving off Windows because extension builds are unreliable there,
this is the preferred route.

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
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy ninja pytest

# Compile the CUDA extension
pip install -e hive_gpu/
```

Or if a prebuilt `.pyd`/`.so` is already present in `hive_gpu/`, it loads automatically.

## Search Algorithms

### Gumbel AlphaZero (default)

Based on [Danihelka et al., 2022](https://openreview.net/forum?id=bERaNdoegnO). Uses Sequential Halving with Gumbel noise to select and evaluate a fixed budget of candidate actions. All NN evaluations within a round are fully independent and batched — no serial tree traversal bottleneck.

- **On by default** — no flag needed
- Use `--gumbel-considered` to set the number of actions considered at the root (k, default 32)
- Rounds = ceil(log2(k)) — e.g. k=32 → 5 rounds
- Scales well with VRAM: run more games in parallel, not more serial rounds

### Wave-parallel MCTS (opt-out)

GPU-native tree search with W simulations per wave. Virtual loss diversifies selections within each wave. Use `--wave-size` to control parallelism.

- Disable Gumbel and use MCTS with `--no-gumbel`
- Use `--wave-size` to tune parallelism (default 8; increase for larger GPUs)

## Running Training

### Fresh start (local)

```bash
python -m hive_gpu \
  --encoder-type transformer \
  --games 256 \
  --simulations 128 \
  --gumbel-considered 32 \
  --expansion -1 \
  --iterations 100 \
  --checkpoint-dir checkpoints \
  --log-file training.log
```

Gumbel AlphaZero is used by default. Pass `--no-gumbel` to fall back to wave-parallel MCTS.

### Resume from checkpoint (local)

```bash
python -m hive_gpu \
  --encoder-type transformer \
  --games 256 \
  --simulations 128 \
  --gumbel-considered 32 \
  --expansion -1 \
  --iterations 100 \
  --resume checkpoints/hive_gpu_checkpoint_0124.pt \
  --checkpoint-dir checkpoints \
  --log-file training.log
```

The `--resume` flag loads network weights, optimizer state, and learning rate schedule. The replay buffer is not saved and resets on restart.

### Docker (RunPod / Linux)

Mount a local directory for checkpoints so they persist if the container stops:

```bash
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  hive-neuralnet \
  python -m hive_gpu \
    --encoder-type transformer \
    --games 512 \
    --simulations 128 \
    --gumbel-considered 32 \
    --expansion -1 \
    --endgame-frac 1.0 \
    --endgame-surround 5 \
    --resume checkpoints/hive_gpu_checkpoint_0124.pt \
    --checkpoint-dir checkpoints \
    --log-file checkpoints/training.log
```

To smoke-test that the Linux extension build worked before starting a long run:

```bash
docker run --rm --gpus all hive-neuralnet \
  python -c "import hive_gpu; ext = hive_gpu.load_extension(); print(ext.BOARD_SIZE, ext.ACTION_SPACE_SIZE)"
```

To run a single iteration as a smoke test:

```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  hive-neuralnet \
  python -m hive_gpu \
    --encoder-type transformer \
    --games 32 \
    --simulations 64 \
    --gumbel-considered 32 \
    --iterations 1 \
    --checkpoint-dir checkpoints \
    --log-file checkpoints/linux_smoke.log
```

### With endgame curriculum

Start a fraction of games from positions where both queens are nearly surrounded (4–5 neighbors), giving the network strong gradient signal on tactical decisions:

```bash
python -m hive_gpu \
  --encoder-type transformer \
  --games 512 \
  --simulations 128 \
  --gumbel-considered 32 \
  --expansion -1 \
  --iterations 200 \
  --endgame-frac 1.0 \
  --endgame-surround 5 \
  --draw-keep-rate 0.10 \
  --resume checkpoints/hive_gpu_checkpoint_0124.pt \
  --checkpoint-dir checkpoints \
  --log-file training.log
```

## Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--games` | 64 | Parallel self-play games per iteration |
| `--simulations` | 100 | MCTS simulations per move (or Gumbel simulation budget) |
| `--iterations` | 20 | Total training iterations |
| `--no-gumbel` | — | Disable Gumbel and fall back to wave-parallel MCTS |
| `--wave-size` | 8 | Parallel MCTS sims per GPU wave (only with `--no-gumbel`) |
| `--gumbel-considered` | 32 | Number of root actions considered (k); rounds = ceil(log2(k)) |
| `--gumbel-c-visit` | 50.0 | Q-transform visit count scale |
| `--gumbel-c-scale` | 1.0 | Q-transform value scale |
| `--expansion` | 0 | Expansion piece mask: 0=base, 1=+Mosquito, 2=+Ladybug, 4=+Pillbug, 7=all, -1=random |
| `--endgame-frac` | 0.0 | Fraction of games starting from endgame positions (0–1) |
| `--endgame-surround` | 5 | Max queen neighbor count for endgame starts (range 4–surround) |
| `--draw-keep-rate` | 1.0 | Fraction of drawn games kept for training (0.1 = discard 90%) |
| `--lr` | 0.0002 | Learning rate |
| `--buffer-size` | 50000 | Replay buffer capacity |
| `--resume` | — | Path to checkpoint to resume from |
| `--log-file` | — | Append output to log file (in addition to terminal) |

## Diagnostics

### Win-in-one probe

Measures whether the network correctly identifies forced wins. Tests across all 8 expansion subsets:

```bash
python probe_win_in_one.py
# or target a specific checkpoint:
python probe_win_in_one.py --checkpoint checkpoints/hive_gpu_checkpoint_0124.pt
```

Output shows per-subset results and aggregate statistics (win move rank, top-3 rate, mean policy probability on the winning move).

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

## Checkpoints

Saved after every iteration as `hive_gpu_checkpoint_NNNN.pt`. Each checkpoint contains:
- Network weights and config
- Optimizer state
- Iteration number and training metrics

A pretrained checkpoint (iteration 124) is included in `checkpoints/`.

## Running Tests

```bash
python -m pytest tests/
```

241 tests covering the game engine, transformer network, and GPU move generation validation.

## Project Structure

```
hive_engine/       # Core game rules (board, pieces, game state, hex grid)
hive_transformer/  # Transformer network and token encoder
hive_gpu/          # CUDA extension, GPU-native MCTS, trainer, CLI
  csrc/            # CUDA C++ source (move gen, MCTS tree, state encoder)
tests/             # Test suite
checkpoints/       # Pretrained checkpoints
archive/           # Archived CPU training code (GNN, NNUE, CPU MCTS)
Dockerfile         # Container for cloud/RunPod deployment
```
