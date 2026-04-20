"""Validate the CUDA prs_v2_classify_batch kernel against Python reference.

Compares per-state outputs of `ext.prs_v2_classify_batch` against
`SlotMapper` + `build_head_inputs_from_states` + per-state slot-of-legal
mapping. Runs on a batch of random play-outs at varying ply depths.
"""
from __future__ import annotations

import numpy as np
import torch

import hive_gpu
from hive_prs.slot_map import SlotMapper, map_legal_moves
from hive_prs.prs_v2_bridge import build_head_inputs_from_states


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


def reference_outputs(state_bytes_cpu, legal_np, nlegal_np, B):
    """Compute Python-reference versions of every kernel output."""
    from hive_prs.prs_v2_head import N_DIR_PIECES, N_THROW_PIECES, N_LONG_PIECES
    from hive_prs.slot_map import C_MOVE, C_HAND, N_SLOTS, PASS_SLOT

    # Build a fake board_h/full_h with shape (B, MB, 1) so the bridge is happy;
    # we only need the index outputs, which are independent of the actual h.
    MB = 64
    SEQ = 80
    board_h = torch.zeros(B, MB, 1)
    cls_h   = torch.zeros(B, 1)
    full_h  = torch.zeros(B, SEQ, 1)
    inp, mappers = build_head_inputs_from_states(
        state_bytes_cpu, board_h, cls_h, full_h
    )

    max_l = legal_np.shape[1]
    slot_of_legal = np.full((B, max_l), -1, dtype=np.int64)
    for b in range(B):
        n = int(nlegal_np[b])
        if n == 0:
            continue
        slots, _, _ = map_legal_moves(state_bytes_cpu[b], legal_np[b, :n], n)
        slot_of_legal[b, :n] = slots

    return {
        "dir_piece_idx":      inp.dir_piece_idx.cpu().numpy(),
        "throw_piece_idx":    inp.throw_piece_idx.cpu().numpy(),
        "long_piece_idx":     inp.long_piece_idx.cpu().numpy(),
        "move_nbrs":          inp.move_nbrs.cpu().numpy(),
        "place_nbrs":         inp.place_nbrs.cpu().numpy(),
        "move_mask":          inp.move_mask.cpu().numpy(),
        "place_mask":         inp.place_mask.cpu().numpy(),
        "current_color":      inp.current_color.cpu().numpy(),
        "slot_of_legal":      slot_of_legal,
        "move_cell_ids":      inp.move_cell_ids.cpu().numpy(),
        "place_cell_ids":     inp.place_cell_ids.cpu().numpy(),
        "dir_dest_cell":      inp.dir_dest_cell.cpu().numpy(),
        "dir_dest_board_idx": inp.dir_dest_board_idx.cpu().numpy(),
        "throw_dest_cell":    inp.throw_dest_cell.cpu().numpy(),
        "hand_token_idx":     inp.hand_token_idx.cpu().numpy(),
    }


def kernel_outputs(ext, states_t, legal_t, nlegal_t, B, max_l):
    out = ext.prs_v2_classify_batch(states_t, legal_t, nlegal_t, B, max_l)
    keys = ("dir_piece_idx", "throw_piece_idx", "long_piece_idx",
            "move_nbrs", "place_nbrs", "move_mask", "place_mask",
            "current_color", "slot_of_legal",
            "move_cell_ids", "place_cell_ids",
            "dir_dest_cell", "dir_dest_board_idx", "throw_dest_cell",
            "hand_token_idx")
    return {k: t.cpu().numpy() for k, t in zip(keys, out)}


def diff_report(name, py, cu, b):
    if py.shape != cu.shape:
        print(f"  [{b}] {name}: SHAPE MISMATCH {py.shape} vs {cu.shape}")
        return False
    if np.array_equal(py, cu):
        return True
    diff_idx = np.argwhere(py != cu)
    print(f"  [{b}] {name}: {len(diff_idx)} diffs (showing first 5):")
    for i in diff_idx[:5]:
        print(f"     idx={tuple(i)}  py={py[tuple(i)]} cu={cu[tuple(i)]}")
    return False


def main() -> None:
    ext = hive_gpu.load_extension()
    ext.initialize_tables()
    rng = np.random.default_rng(42)

    B = 64
    total_ok = 0
    total_states = 0
    fail_keys = {}

    depths = [0, 3, 6, 10, 15, 20, 30, 45, 60, 80, 100, 120, 150, 180, 220, 260]
    for trial, depth in enumerate(depths):
        states = ext.create_initial_states(B, 7)
        if depth > 0:
            play_random(ext, states, B, depth, rng)
        legal_t, nlegal_t = ext.generate_legal_moves_batch(states, B)
        max_l = legal_t.shape[1]

        states_cpu = states.cpu().numpy()
        legal_np   = legal_t.cpu().numpy()
        nlegal_np  = nlegal_t.cpu().numpy()

        py = reference_outputs(states_cpu, legal_np, nlegal_np, B)
        cu = kernel_outputs(ext, states, legal_t, nlegal_t, B, max_l)

        print(f"\n-- trial {trial}: depth={depth}, B={B}, max_legal={max_l} --")
        for b in range(B):
            ok_b = True
            for key in py:
                arr_py = py[key][b] if py[key].ndim > 0 else py[key]
                arr_cu = cu[key][b] if cu[key].ndim > 0 else cu[key]
                if key == "slot_of_legal":
                    n = int(nlegal_np[b])
                    arr_py, arr_cu = arr_py[:n], arr_cu[:n]
                if not diff_report(key, arr_py, arr_cu, b):
                    fail_keys[key] = fail_keys.get(key, 0) + 1
                    ok_b = False
            if ok_b:
                total_ok += 1
            total_states += 1

    print(f"\n== SUMMARY: {total_ok}/{total_states} states match exactly ==")
    if fail_keys:
        print("Per-key failure counts:")
        for k, v in sorted(fail_keys.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
    else:
        print("All checks pass.")


if __name__ == "__main__":
    main()
