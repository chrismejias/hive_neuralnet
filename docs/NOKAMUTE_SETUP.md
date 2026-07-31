# Installing Nokamute

Nokamute is included as a pinned Git submodule at `external/nokamute`. A fresh
clone should pull the repository and all submodules together:

```bash
git clone --recurse-submodules https://github.com/chrismejias/hive_neuralnet.git
cd hive_neuralnet
```

For an existing clone, including one that was originally cloned without
submodules:

```bash
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

Install a stable Rust toolchain from <https://rustup.rs/> or through the system
package manager, then build the release binary:

```bash
cargo build --release --manifest-path external/nokamute/Cargo.toml
```

The executable will be at:

```text
external/nokamute/target/release/nokamute
```

Confirm that its UHP server starts:

```bash
external/nokamute/target/release/nokamute uhp
```

It can then be used by the bundled arena:

```bash
python3 nokamute_arena.py \
  --checkpoint checkpoints_fnn_alphabeta_value_only_20k_512/hive_fnn_alphabeta_0167.pt \
  --games 20 --our-nodes 40000 \
  --nokamute-seconds 1 --nokamute-threads 1
```

Nokamute accepts time or depth budgets, not an exact node budget. Use
`--nokamute-depth N` for repeatable fixed-depth comparisons.

## Optional diagnostic score command

The upstream UHP interface does not report an evaluation. Our disagreement
analysis used `patches/nokamute-score-command.patch`, which adds a non-standard
`score` command returning the backed-up value from the last completed search.
Apply and rebuild it only when running those diagnostics:

```bash
git -C external/nokamute apply ../../patches/nokamute-score-command.patch
cargo build --release --manifest-path external/nokamute/Cargo.toml
```

To return the submodule to its pinned upstream source after the experiment:

```bash
git -C external/nokamute apply -R ../../patches/nokamute-score-command.patch
```
