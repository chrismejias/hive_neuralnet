# Multi-Representation PRS Fix Plan

## Problem

Each physical move (piece A → cell X) can be encoded by multiple PRS indices — one per
occupied neighbour of the destination cell (any neighbour can serve as the reference piece).
Currently only the **canonical** rep (smallest-token-index neighbour) is used in both
training and inference. This causes:

1. **Poor generalisation**: when token ordering changes between game states, the same
   conceptual move gets a different canonical PRS index, so the network can't recognise
   patterns it has already seen.
2. **Under-use of Gumbel budget**: the win move has only 1 lottery ticket in the top-K
   draw instead of n_reps tickets, making it less likely to be evaluated.

## Why the Previous Attempt Failed

The previous commit spread training targets across all reps (`p/n_reps` each) and added
all reps to the CE legal mask — but inference (`prs_mask`) remained canonical-only.
This created a train/inference mismatch: the network learned to distribute logit across
canonical + non-canonical slots, making canonical logits weaker than a random network.
Draw rate climbed to 94% by iteration 5 and kept rising.

**The rule**: training legal mask and inference `prs_mask` must be identical.

## Correct Fix (3 coordinated changes)

### 1. Section 3 — extend `prs_to_j` to all reps

After building the canonical `prs_to_j`, call `moves_to_all_reps` per game and add
non-canonical reps to `prs_to_j` (pointing to the same move column `j` as canonical).
Store the result as `all_reps_per_game` for reuse in sections 8 and `_build_examples`.

```python
# Import already available: moves_to_all_reps from hive_prs.action_space

all_reps_per_game = []
for i in range(B):
    nl_i, no_i = int(nlegal_np[i]), int(nocc_cpu[i])
    all_reps_i = moves_to_all_reps(legal_np[i], nl_i, occ_cpu[i, :no_i], prs_from_move[i])
    all_reps_per_game.append(all_reps_i)
    for can_prs, reps in all_reps_i.items():
        j_can = prs_to_j[i, can_prs]
        if j_can >= 0:
            for ri in reps:
                if ri != can_prs and 0 <= ri < ACTION_SPACE_SIZE and prs_to_j[i, ri] < 0:
                    prs_to_j[i, ri] = j_can

# prs_mask now covers canonical + non-canonical reps automatically
```

### 2. Section 5 — deduplicate Gumbel top-K by physical move

After `torch.topk(perturbed_dense, max_k)`, some slots may be duplicate reps of the
same physical move. Deduplicate in scan order (highest Gumbel score first = best rep
kept), so K slots evaluate K unique physical moves.

```python
topk_j = prs_to_j[np.arange(B)[:, None], topk_actions.cpu().numpy()].astype(np.int32)

dedup_ok = np.ones((B, max_k), dtype=bool)
for b in range(B):
    seen = set()
    for k_i in range(max_k):
        j = int(topk_j[b, k_i])
        if j < 0 or j in seen:
            dedup_ok[b, k_i] = False
        else:
            seen.add(j)

cand_mask = cand_mask & torch.from_numpy(dedup_ok).cuda()
```

Key property: `topk_actions` is already sorted by Gumbel score descending, so the first
occurrence of each physical move is its best-Gumbel-score rep. Duplicates (lower score)
are eliminated. In practice dedup removes very few candidates (most top-K slots cover
distinct physical moves).

### 3. Section 8 + `_build_examples` — spread policy targets across all reps

Store `all_reps_per_game[i]` in history. In `_build_examples`, distribute each topk
move's visit probability equally across all its reps (`p / n_reps` each). Apply epsilon
floor to all reps. Compute surprise-weight KL using canonical-aggregated policy to avoid
`log(p / ~0)` from non-canonical reps that `nn_prior` assigns zero.

This is now correct because training legal mask == inference prs_mask (both include all
reps).

## Expected Benefits

- **More lottery tickets**: win move has n_reps chances to enter top-K instead of 1.
  With avg n_reps≈3 and K=16 from 90 legal reps, expected unique moves in top-K ≈ 16.
- **Better generalisation**: network sees all valid reference-piece encodings, so it
  recognises the same pattern regardless of which neighbour becomes the canonical ref
  in a new game state.
- **No train/inference mismatch**: both sides use the same set of legal PRS indices.

## Files to Change

- `hive_prs/prs_orchestrator.py`: sections 3, 5, 8, and `_build_examples`
- Import `moves_to_all_reps` (already in `action_space.py`)
- No changes needed to `action_space.py`, `prs_trainer.py`, or `prs_replay_buffer.py`
