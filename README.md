# Hive Neural Network

AlphaZero-style self-play training for the board game [Hive](https://en.wikipedia.org/wiki/Hive_(game)), focused on a fast recommended FNN engine, an experimental PRS transformer line, and a shared GPU-native game engine.

## Recommended Engine

**FNN is the default recommended engine.** It is the strongest and most practical training path in this repository today: it is much faster, reaches strong play with far less compute, and benefits from a compact handcrafted state representation that already encodes a lot of useful Hive structure.

**PRS is a work in progress research engine.** It learns a richer board representation from scratch and is therefore slower in two ways:

- each search evaluation is much heavier than FNN
- it has to spend training capacity learning useful board and legality structure that FNN already gets from engineered features

That makes PRS interesting architecturally, but it currently needs much more wall-clock time to approach the same strength.

## Architecture Overview

| Model | Package | Params | Self-play |
|-------|---------|--------|-----------|
| FNN (recommended default) | `hive_fnn` | 1.0K – 18.8K | Gumbel-root MCTS / PUCT MCTS |
| Hybrid GNN (experimental) | `hive_hybrid_gnn` | 125K+ | FNN successor features + graph-aware policy/value |
| PRS Transformer (research path) | `hive_prs` | 1.7M / 9.8M | Gumbel-root MCTS (v3 default trunk) |

All models use the same GPU-native game engine (`hive_gpu`) with CUDA kernels for move generation, state encoding, and legal move lookup.

## Requirements

- Python 3.11+
- PyTorch 2.4+ (tested with 2.10.0+cu128)
- CUDA 12.4+ / driver supporting CUDA 12.8

If you do not have an NVIDIA GPU, the repository also includes a **single-game CPU FNN fallback** for play and analysis. It is much slower than the GPU engine and is not used for self-play training, but it does let you play against the recommended FNN model on CPU-only machines.

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

## Playing Against The Engine

For the recommended GUI path on a machine without an NVIDIA GPU, use the CPU FNN fallback:

```bash
cd /workspace/hive_neuralnet
python3.11 gui.py \
  --engine fnn-cpu \
  --checkpoint checkpoints_fnn/resume0500_24576/hive_fnn_checkpoint_0500.pt \
  --simulations 256 \
  --root-workers 2
```

There is also a text-mode CPU player:

```bash
cd /workspace/hive_neuralnet
python3.11 play_fnn_cpu.py \
  --checkpoint checkpoints_fnn/resume0500_24576/hive_fnn_checkpoint_0500.pt \
  --mode human_vs_ai \
  --simulations 256 \
  --gumbel-root \
  --root-workers 2
```

The old `gui.py --engine legacy` path is still available for the archived GNN / NNUE / transformer checkpoints, but the FNN CPU option is the recommended fallback when CUDA is unavailable.

## Long-Running Background Training

Use `python -u`, redirect stdin from `/dev/null`, append both stdout and stderr
to a log, and capture the PID. This is more reliable than a bare `nohup`
because startup output is unbuffered and the process does not inherit an
interactive stdin handle.

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

PRS example:

```bash
cd /workspace/hive_neuralnet
mkdir -p checkpoints_prs_v3

nohup python3.11 -u -m hive_prs.train_prs \
  --model-version v3 \
  --iterations 1000 \
  --games 256 \
  --simulations 256 \
  --max-considered 16 \
  --checkpoint-dir checkpoints_prs_v3 \
  --checkpoint-keep-every 50 \
  --resume checkpoints_prs_v2/prs_v2_iter_1500.pt \
  --wave-parallel \
  >> checkpoints_prs_v3/training.log 2>&1 < /dev/null &

echo $!
```

Check that exactly one intended trainer is running:

```bash
pgrep -af 'hive_prs.train_prs|train_prs|hive_fnn.train_fnn|train_fnn'
tail -f checkpoints_prs_v3/training.log
```

If no process appears and the log does not update, run the same command in the
foreground first to catch startup errors:

```bash
python3.11 -u -m hive_prs.train_prs \
  --model-version v3 \
  --iterations 1000 \
  --games 256 \
  --simulations 256 \
  --max-considered 16 \
  --checkpoint-dir checkpoints_prs_v3 \
  --checkpoint-keep-every 50 \
  --resume checkpoints_prs_v2/prs_v2_iter_1500.pt \
  --wave-parallel
```

Some managed notebook/agent environments kill detached child processes when the
launch command exits, even with `nohup`. In that case, start training inside a
persistent shell, `tmux`, `screen`, or a long-lived PTY session and keep that
session open. The trainer CLI help for PRS and FNN includes model-specific
background-launch templates.

## Search Algorithms

### Gumbel AlphaZero with sequential Halving (default)

Based on [Danihelka et al., 2022](https://openreview.net/forum?id=bERaNdoegnO). At the root, candidate moves are sampled from the policy prior plus Gumbel noise, typically with `k = 16`. All other legal moves are dropped from search for that move. Search then runs in four sequential-halving rounds:

`16 -> 8 -> 4 -> 2`

Each round gets an equal fraction of the total simulation budget. For example, with `256` simulations total, each round gets `64` simulations, so each of the first `16` candidates receives `4` simulations in round 1, each of the remaining `8` receives `8` in round 2, and so on.

In the original paper, non-root search is deterministic and tries to preserve the root policy improvement semantics as closely as possible. In this codebase, the active PRS and FNN paths instead use non-root **PUCT MCTS** because it produced stronger play empirically. The inner-node exploration score is:

`Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`

where `Q` is mean action value, `P` is the policy prior, and `N_parent` / `N_child` are visit counts.

During sequential halving, surviving root moves are ranked by a Gumbel sigma score of the form:

`g(a) + log π₀(a) + σ * Q_mcts(a)`

where:
- `g(a)` is the sampled Gumbel noise
- `π₀(a)` is the root prior on the legal move
- `Q_mcts(a)` is the search-estimated action value for that root move
- `σ` is a visit-scaled coefficient that increases the influence of search as simulations accumulate

Finally, the training policy target is derived from these improved logits rather than raw visit counts. That is the core Gumbel AlphaZero idea: use search to refine the policy in logit space instead of simply imitating a visit histogram.

#### Project Deviations From The Original Algorithm

The current project makes three deliberate changes relative to the original paper:

1. **Non-root nodes use PUCT MCTS instead of pure policy-preserving sampling.**
   The paper's non-root rule is closer to direct policy improvement. In this codebase, the active PRS and FNN search paths use non-root PUCT MCTS because it was empirically stronger than pure policy sampling.

2. **The active PRS and FNN trainers use a visited-only Gumbel policy target.**
   For PRS and FNN self-play, legal moves outside the searched Gumbel candidate set receive zero target mass instead of keeping their raw prior. This outperformed the prior-preserving variant empirically.

So the active PRS/FNN target is:

- **Visited/searched moves**
  ```
  π_imp(a ∈ S) ∝ exp(log π₀(a) + σ · Q_mcts(a))
  ```
- **Unvisited / not-in-S moves**
  ```
  π_imp(a ∉ S) = 0
  ```
3. **Non-root nodes are explored in an escalating wave-parallel fashion.**
   The paper's serial schedule shrinks the amount of parallel work as halving proceeds. On GPU, that leaves batch efficiency on the table. In this project, each later round increases the number of parallel sims per surviving root move so the total batch stays roughly constant. For example, PRS uses `1,2,4,8` waves per round and FNN uses `2,4,8,16`.
   
## FNN (HiveGo-style Feedforward Network)

This is the **recommended engine** for current use. It is inspired by HiveGo's AlphaZeroFNN. The key idea is to encode root + successor states independently, then score each action as `f(root_emb ‖ successor_emb)`. The network never sees the raw board directly; instead it operates on a compact engineered feature set. That keeps the network tiny, the search fast, and the training signal much more sample-efficient than PRS.

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

### GPU-Native Self-Play

The live path is a Python/Torch orchestrator (`FNNMCTSOrchestrator`) that repeatedly calls batched CUDA kernels for move generation, successor feature extraction, tree select/expand/backprop, and move application, while running the FNN forward pass in PyTorch on GPU.

There is also an experimental fused FNN self-play kernel in `hive_gpu/csrc/fnn_selfplay.cuh`, but it is not the current training path.

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

---

## Hybrid GNN (experimental FNN successor features + graph policy/value)

The hybrid GNN research direction lives in `hive_hybrid_gnn`.

The hybrid model keeps the part of FNN that has worked best:

- root and successor states are encoded as `110`-dim FNN feature vectors
- policy logits compare the root FNN embedding, root graph embedding, and successor FNN embedding
- a trained FNN checkpoint can initialize the shared FNN feature encoder, but the graph-aware policy tower is not weight-compatible with the FNN action tower

It changes both policy and value:

- the value head receives the FNN root embedding
- it also receives a pooled graph embedding over the full board state
- graph nodes include all board pieces, including buried stack pieces, plus grouped hand nodes
- graph edges connect vertical stack neighbors and nearby top pieces

The default graph radius is `2`, so a top piece connects to occupied cells
within the 18 hexes at distance 1 or 2. Radius 1 is still available and gives
the classic 6-neighbor local graph. Radius 2 increases the edge count, but it
should improve message passing because two-step tactical motifs are available
in a single layer rather than requiring an additional round of propagation.
The default small preset now uses `5` graph layers with global-pool bias
enabled; the large preset remains `5` layers at a wider hidden size.

Current status:

- model scaffold is implemented
- CPU `GameState` graph encoding is implemented for prototyping
- CUDA `HiveState` graph encoding is implemented as a padded batched tensor path for training/search integration
- GPU-native self-play and training are implemented through `hive_hybrid_gnn.train_hybrid`
- the trainer reuses the FNN raw-state replay format and rebuilds FNN features, successor features, and graph tensors on GPU
- this is intended as a lighter alternative to giving the value head a full PRS transformer trunk

Quick smoke test:

```bash
python3.11 -m hive_hybrid_gnn.train_hybrid \
  --preset small \
  --iterations 1 \
  --games 32 \
  --simulations 64 \
  --gumbel-considered 16 \
  --checkpoint-dir checkpoints_hybrid_gnn_smoke
```

Quick parameter summary:

```bash
python3.11 -m hive_hybrid_gnn
```

---

## PRS Transformer (Piece-Relative Space, current default trunk: v3)

PRS is the experimental line in this repository. Its goal is to learn a richer
board representation than the FNN, but in practice that means much slower
self-play and training, and much slower strength gain. The PRS policy/value
stack has to discover legality, geometry, and tactical structure from the board
tokens themselves, while FNN already starts from handcrafted features that encode a lot of that structure cheaply. Based on current trends, this model will likely require months worth of RTX 4090 equievlent compute to equal the FNN. It remains to be seen whether it would eventually surpass the FNN.

The active PRS policy head is still the structured **813-slot legal-masked**
head introduced in v2. **PRS v3** keeps that same tokenization, value head,
policy head, trainer, and search path, but replaces the trunk with custom
encoder layers that add a learned per-head relative attention bias based on
pairwise `23x23` board-cell offsets. Absolute cell embeddings are still
present, so v3 is a hybrid absolute+relative design rather than a purely
relative transformer.

The relative bias tensors are zero-initialized, so PRS v2 checkpoints can be
migrated to v3: all old trunk/head weights load, and only the new per-layer
`relative_bias` parameters start fresh.

### Architecture

- **Tokenization:** up to `60` tokens per state
  - `1` CLS token
  - up to `28` board tokens, one per piece in board-stack order
  - up to `28` hand tokens
- **Per-token input features:** `25` floats from the CUDA state encoder
- **Global features:** `6` floats appended to the value head
- **Positions:** absolute `23x23 = 529` board-cell ids plus one off-board sentinel
- **Trunk (v2):** token projection + learned absolute position/type embeddings + Transformer encoder
- **Trunk (v3):** the same, plus learned per-head relative attention bias from clipped board-cell offsets
- **Value head:** CLS embedding + global features -> MLP -> `tanh`
- **Policy head:** structured **813-slot** legal-masked head built around move geometry rather than a flat board-wide action lattice

The important design choice is the split between a generic board encoder and a structured action head. The transformer trunk learns a board representation from scratch, while the policy head does not score all theoretically possible actions. Instead it scores only a compact structured action basis derived from the current legal geometry.

The 813 slots are:
- `48` direction slots for queen / beetle / grasshopper / pillbug / mosquito local moves
- `60` throw slots for pillbug-style throws
- `448` long-move slots for ant / spider / ladybug / mosquito long-range moves
- `256` hand-placement slots
- `1` pass slot

At runtime, the CUDA bridge maps the current legal moves into that 813-slot space, including:
- which piece instances exist
- which move / place cells are active
- neighbor structure for those cells
- destination structure for direction and throw moves

This keeps the policy space compact without giving up a learned board representation.

- **Small config** (default): about `1.66M` parameters for v2, `1.72M` for v3 — `d_model=128`, `heads=8`, `layers=6`, `dim_ff=512`
- **Large config**: `d_model=256`, `heads=8`, `layers=12`, `dim_ff=1024`

Legacy PRS v1 modules are archived under `archive/legacy_prs_v1`.

### Archived Auxiliary Heads

Four auxiliary-head experiments were tried on PRS and all degraded playing
strength, so they are now archived and disabled by default:

- `queen_surround` / endgame pieces surrounding each queen
- `final_mobility` / endgame mobility of each piece on the final two turns
- `slot_legality` / current legal move prediction
- `articulation` / one-hive articulation-point prediction

Those heads were informative as diagnostics, but in training they slowed the
model down and produced weaker checkpoints.

### Search Pattern

- **Default:** Gumbel-root MCTS tree search (`PRSMCTSOrchestratorV2`)
- Sequential-halving rounds use final Gumbel sigma scoring (`gumbel + logits + sigma * Q`) for the played move, rather than visit-count argmax.
- Wave-parallel MCTS is enabled by default with a hard-coded per-round schedule: `1, 2, 4, 8`. Use `--no-wave-parallel` for pure serial waves.
- `--deterministic-non-root` enables the paper-style deterministic inner selection, with `--virtual-q-penalty` used to diversify wave-parallel sims.
- The current training/profile defaults use `k=16`.
- PRS enables TF32 matmul precision for the transformer trunk on supported GPUs.
- `--compile-forward` can opt in to `torch.compile` for the tensor-only trunk/head path; it is off by default because Inductor can be unstable on some hosts.
- Current `train_prs` default is PRS v3; legacy PRS-v1 search paths are archived.
- Policy target now matches FNN's visited-only improved policy: unvisited legal moves receive zero target mass.
- Move-cap draws are excluded from value loss, while genuine draws are retained.
- `profile_models.py` exposes `prs-small` and `prs-large` for direct preset comparison.

### Training

```bash
# Quick benchmark (1 iteration, 128 games, 256 sims, k=16)
python -m hive_prs.train_prs \
  --iterations 1 --games 128 --simulations 256 --max-considered 16

# Default PRS v3 run
python -m hive_prs.train_prs \
  --iterations 1500 --games 128 --simulations 512 --max-considered 16 \
  --checkpoint-dir checkpoints_prs_v3

# PRS v3 run migrated from a v2 checkpoint
python -m hive_prs.train_prs \
  --model-version v3 \
  --iterations 2000 --games 256 --simulations 256 --max-considered 16 \
  --checkpoint-dir checkpoints_prs_v3 \
  --resume checkpoints_prs_v2/prs_v2_iter_1500.pt

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
| `--model-version` | v3 | `v2` standard trunk or `v3` trunk with relative attention bias |
| `--relative-position-clip` | 8 | PRS v3 relative row/col offset clipping radius |
| `--expansion-mask` | 7 | Expansion piece mask: 0=base, 1=+Mosquito, 2=+Ladybug, 4=+Pillbug, 7=all |
| `--augment-prob` | 0.5 | Probability of C6 rotational augmentation per training batch |
| `--buffer-size` | 150000 | Replay buffer capacity |
| `--slot-legality-loss-weight` | 0.0 | Archived auxiliary head; keep at 0 unless explicitly re-testing |
| `--articulation-loss-weight` | 0.0 | Archived auxiliary head; keep at 0 unless explicitly re-testing |
| `--checkpoint-dir` | checkpoints\_prs\_v3 | Checkpoint output directory |
| `--resume` | — | Path to checkpoint to resume from |

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
    game_logic.cu      # Move gen, state apply, feature extraction, tree/search kernels
    fnn_features.cuh   # 110-dim FNN feature extraction kernel
    fnn_selfplay.cuh   # Experimental fused FNN self-play kernel (not the active trainer path)
    state_encoder.cuh  # Transformer state encoding
    mcts_tree.cuh      # GPU-native MCTS tree
hive_prs/          # PRS v2/v3 (structured 813-slot head)
  prs_transformer_v2.py        # HivePRSTransformerV2
  prs_transformer_v3.py        # HivePRSTransformerV3 with relative attention bias
  prs_mcts_orchestrator_v2.py  # PRS v2 Gumbel-root MCTS
  prs_trainer_v2.py            # PRS v2 trainer
  train_prs.py                 # CLI entry point (v3 default trunk)
hive_fnn/          # HiveGo-style FNN with multiple search paths
  fnn_network.py       # HiveFNN: shared encoder + value head + action tower
  fnn_mcts_orchestrator.py  # Gumbel-root MCTS tree search
  fnn_puct_orchestrator.py  # Plain PUCT MCTS tree search
  fnn_trainer.py       # FNNTrainer training loop
  train_fnn.py         # CLI entry point
hive_hybrid_gnn/   # Experimental FNN policy + graph value model
  hybrid_gnn_net.py    # HiveHybridGNN
  graph_encoder.py     # CPU GameState -> radius-1/radius-2 graph encoder
tests/             # Test suite (250+ tests)
archive/           # Archived legacy models, diagnostics, and old CPU training code
Dockerfile         # Container for cloud/RunPod deployment
```
