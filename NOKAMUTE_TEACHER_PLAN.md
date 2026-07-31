# Nokamute teacher and search follow-up plan

## Objective

Reduce the value-only alpha-beta network's late-middle-game positional errors
while keeping production search fast, value-only, and independently testable.
The next machine is expected to have a stronger CPU and a cheaper GPU, so CPU
workers should be the default for arenas, teacher labeling, and scalable
self-play unless a new machine-specific crossover benchmark proves otherwise.

## Completed on the current machine

- Built pinned Nokamute and integrated it through the UHP arena.
- Added an optional diagnostic `score` UHP extension and saved it as
  `patches/nokamute-score-command.patch`.
- Ran matched-compute arenas showing the value-only alpha-beta engine is
  competitive with Nokamute.
- Logged our search evaluation through games until a Nokamute win. The largest
  disagreements occurred in late middle games, not only in long endgames.
- Replayed disputed positions at escalating budgets for both engines. Several
  disagreements remained stable, indicating representation/target errors more
  often than a simple search horizon problem.
- Diagnosed queen surround and mobile/free-piece features in a major false
  positive position.
- Ported GPU tactical concepts to the native CPU engine: queen threats,
  queen relief, power-piece immobilization, bounded quiescence, and forced
  extensions, all exposed through profiles and counters.
- Ran a 128-game, 40,000-node quiescence ablation. Quiescence consumed its full
  20% allowance; quiescence-free search won 58-46 with 24 draws. It is now the
  default, while `quiescence` retains the old behavior as a toggle.
- Removed policy evaluation/order dependence from alpha-beta and retained the
  scalar value-only network/search path.
- Profiled GPU alpha-beta. Active-game compaction and expansion grouping were
  already present. The explicit global-frame traversal was about eight times
  slower than the recursive kernel.
- Added a guarded warp-block launch for wide GPU batches at depth <=16. It
  reached about 2.21M nodes/s at 512 games and 40,000 nodes, versus about
  1.87M before. Deeper recursive searches retain the safe serial launch.
- Benchmarked the CPU engine over 512 games with 16 workers, checkpoint 167,
  depth 16, and 40,000 nodes: 1.774M aggregate nodes/s (1.793M search-only),
  approximately 112k nodes/s per worker.
- Added aggregate throughput reporting to the CPU checkpoint arena and a
  reusable-context-only GPU benchmark mode.

## Recommended next-machine workflow

1. Pull with submodules and build Nokamute using `docs/NOKAMUTE_SETUP.md`.
2. Build the Hive CPU/CUDA extensions and run the focused alpha-beta tests.
3. Benchmark CPU worker counts (1, physical-core count, and SMT count) at
   20k, 40k, and 100k nodes on a fixed position suite.
4. Benchmark GPU batches 128/256/512 at depth 16 and the intended training
   budget. Use CPU self-play if it offers better nodes per dollar or wall time.
5. Resume value-only training from checkpoint 167 or its latest preserved
   training-state file; do not assume the checkpoint alone contains replay and
   optimizer state.

## Remaining engine work

- Replace or bound the CUDA recursive traversal before relying on requested
  depth 32: a deterministic middle-game depth-32 probe exposed a pre-existing
  device-stack illegal-memory failure, even with the original serial launch.
- Build a fixed regression corpus from disputed Nokamute games, including raw
  state, expansion set, side to move, result, our raw/search values, Nokamute
  score/PV, and escalating-budget stability.
- Calibrate ordinary Nokamute scores against empirical outcomes rather than
  treating its handcrafted units as win probabilities. Give proven results
  substantially more weight.
- Mine the most surprising 1-5% of self-play positions using student/teacher
  disagreement, shallow/deep instability, search/result disagreement, forced
  results, constrained queens, and asymmetric mobility.
- Test temporary auxiliary training heads for queen liberties/coverage,
  movable and pinned pieces, fillable queen-neighbor cells, and pillbug escape
  potential. Omit those heads from deployed inference.
- Compare, independently and in combination: hard-position sampling,
  calibrated Nokamute targets, auxiliary targets, and selective extensions.
- Generalize the Nokamute arena/teacher pipeline from Base to all supported
  expansion combinations and quarantine legality disagreements.
- Keep paired color-balanced strength arenas, calibration error, move
  stability, forced-result detection, and node throughput as acceptance gates.

## Data that must move separately

Git contains checkpoint 167 and source/diagnostic tooling, but not the large
training-state/replay file. Copy the latest
`alpha_beta_training_state_latest.pt` separately if exact optimizer and replay
continuation is required.
