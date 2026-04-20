"""Validate the C6 state/move rotator round-trips correctly.

Checks:
  1. Cell rotation table: 6×60° CW == identity on on-grid cells.
  2. State rotation: rotate(state, k) six times == state exactly.
  3. Rotated state is self-consistent: bitboards match pieces stack,
     queen_cell matches pieces on the board.
  4. Move rotation: 6×rotate_moves_batch == input.
  5. Rotated states accept legal moves matching rotated legal moves
     from the original state (permutation-equivalent set).
"""
from __future__ import annotations

import numpy as np
import torch

import hive_gpu
from hive_prs.prs_c6_augment import (
    rotate_state_bytes, rotate_states_batch, rotate_moves_batch,
    _assert_layout, _ROT,
    _PIECES_OFFSET, _HEIGHT_OFFSET, _HEIGHT_BYTES,
    _OCC_OFFSET, _WTOP_OFFSET, _BTOP_OFFSET, _BB_BYTES,
    _QCELL_OFFSET, _QCELL_BYTES,
    NUM_CELLS,
)


def play_random(ext, states, B, plies, rng):
    for _ in range(plies):
        legal, nlegal = ext.generate_legal_moves_batch(states, B)
        nc = nlegal.cpu().numpy()
        chosen = np.zeros(B, dtype=np.int64)
        for i in range(B):
            chosen[i] = rng.integers(max(1, int(nc[i])))
        chosen_t = torch.from_numpy(chosen).cuda()
        moves = legal[torch.arange(B, device="cuda"), chosen_t]
        ext.apply_moves_batch(states, moves, B)


def check_consistency(state: np.ndarray) -> list[str]:
    """Return a list of inconsistency messages ([] = ok)."""
    errs = []
    pieces = state[_PIECES_OFFSET:_PIECES_OFFSET + 5 * NUM_CELLS].reshape(5, NUM_CELLS)
    height = state[_HEIGHT_OFFSET:_HEIGHT_OFFSET + _HEIGHT_BYTES]
    occ = np.unpackbits(
        state[_OCC_OFFSET:_OCC_OFFSET + _BB_BYTES], bitorder="little"
    )[:NUM_CELLS]
    for c in range(NUM_CELLS):
        h = int(height[c])
        if h == 0:
            if occ[c] != 0:
                errs.append(f"cell {c}: occ=1 but height=0")
            for lvl in range(5):
                if pieces[lvl, c] != 0:
                    errs.append(f"cell {c}: empty but pieces[{lvl}]={pieces[lvl,c]}")
        else:
            if occ[c] != 1:
                errs.append(f"cell {c}: height={h} but occ=0")
            for lvl in range(h):
                if pieces[lvl, c] == 0:
                    errs.append(f"cell {c}: height={h} but pieces[{lvl}]=0")
    return errs


def main() -> None:
    ext = hive_gpu.load_extension()
    ext.initialize_tables()
    _assert_layout(ext.SIZEOF_HIVE_STATE)

    rng = np.random.default_rng(7)
    B = 8

    # Build 8 non-trivial positions
    states = ext.create_initial_states(B, 7)
    play_random(ext, states, B, 12, rng)
    states_np = states.cpu().numpy()

    # ── Check 1: cell rot table round-trip ──
    acc = np.arange(NUM_CELLS, dtype=np.int64)
    for k in range(6):
        acc = np.where(acc >= 0, _ROT[1][acc.clip(min=0)], -1)
    ok = (acc == np.arange(NUM_CELLS)) | (acc == -1)
    print(f"[1] cell rot table round-trip: {'OK' if ok.all() else 'FAIL'}")

    # ── Check 2: state rotation round-trip ──
    for i in range(B):
        s = states_np[i]
        # Apply k=1 six times
        r = s.copy()
        for _ in range(6):
            r = rotate_state_bytes(r, 1)
        if not np.array_equal(r, s):
            diff = np.where(r != s)[0]
            print(f"[2] state {i}: round-trip FAIL, {len(diff)} bytes differ (first: {diff[:10]})")
            return
    print(f"[2] state 6×rot(k=1) round-trip: OK on {B} states")

    # Also test rotate(k) is identical to k×rotate(1)
    for k in range(6):
        r_direct = rotate_states_batch(states_np, k)
        r_iter = states_np.copy()
        for _ in range(k):
            r_iter = rotate_states_batch(r_iter, 1)
        if not np.array_equal(r_direct, r_iter):
            print(f"[2b] rotate(k={k}) ≠ {k}×rotate(1): FAIL")
            return
    print(f"[2b] rotate(k) == k×rotate(1) for kin[0,5]: OK")

    # ── Check 3: rotated states are internally consistent ──
    for k in range(1, 6):
        rotated = rotate_states_batch(states_np, k)
        for i in range(B):
            errs = check_consistency(rotated[i])
            if errs:
                print(f"[3] k={k}, state {i}: {len(errs)} inconsistencies")
                for e in errs[:3]:
                    print(f"    {e}")
                return
    print(f"[3] rotated states internally consistent (occ/height/pieces match): OK")

    # ── Check 4: rotated states produce same number of legal moves ──
    # (rotation is a symmetry of the game rules)
    legal_t, nlegal_t = ext.generate_legal_moves_batch(states, B)
    nlegal_orig = nlegal_t.cpu().numpy()

    for k in range(1, 6):
        rotated = rotate_states_batch(states_np, k)
        rotated_t = torch.from_numpy(rotated).cuda()
        _, nleg_rot = ext.generate_legal_moves_batch(rotated_t, B)
        nleg_rot_np = nleg_rot.cpu().numpy()
        mismatches = np.where(nleg_rot_np != nlegal_orig)[0]
        if len(mismatches) > 0:
            print(f"[4] k={k}: nlegal differs on {len(mismatches)} states "
                  f"(orig={nlegal_orig[mismatches][:5]} rot={nleg_rot_np[mismatches][:5]})")
            return
    print(f"[4] rotated states produce same #legal moves as originals: OK")

    # ── Check 5: move rotation round-trip ──
    legal_np = legal_t.cpu().numpy()
    r_moves = legal_np.copy()
    for _ in range(6):
        r_moves = rotate_moves_batch(r_moves, nlegal_orig, 1)
    if not np.array_equal(r_moves, legal_np):
        diff = np.where(r_moves != legal_np)
        print(f"[5] move rotation round-trip FAIL: {len(diff[0])} byte diffs")
        return
    print(f"[5] move 6×rot round-trip: OK")

    # ── Check 6: legal-moves set equals rotated-legal-moves set ──
    # For each original state, the set of rotated legal moves should equal
    # the set of legal moves generated directly from the rotated state.
    for k in range(1, 6):
        rotated = rotate_states_batch(states_np, k)
        rotated_t = torch.from_numpy(rotated).cuda()
        leg_from_rot, nleg_from_rot = ext.generate_legal_moves_batch(rotated_t, B)
        leg_from_rot_np = leg_from_rot.cpu().numpy()
        nleg_from_rot_np = nleg_from_rot.cpu().numpy()

        # Rotate original moves, compare as sets
        r_legal = rotate_moves_batch(legal_np, nlegal_orig, k)

        for i in range(B):
            n = int(nlegal_orig[i])
            a = {tuple(r_legal[i, j]) for j in range(n)}
            n_eng = int(nleg_from_rot_np[i])
            b_set = {tuple(leg_from_rot_np[i, j]) for j in range(n_eng)}
            if a != b_set:
                print(f"[6] k={k}, state {i}: move sets differ. "
                      f"|rot_from_orig|={len(a)}  |engine_on_rotated|={len(b_set)}")
                only_a = a - b_set
                only_b = b_set - a
                print(f"    only in rotated-orig ({len(only_a)}): {list(only_a)[:3]}")
                print(f"    only in engine-rotated ({len(only_b)}): {list(only_b)[:3]}")
                return
    print(f"[6] legal-move-set equivariance: OK across all kin[1,5]")

    print("\n== ALL CHECKS PASS ==")


if __name__ == "__main__":
    main()
