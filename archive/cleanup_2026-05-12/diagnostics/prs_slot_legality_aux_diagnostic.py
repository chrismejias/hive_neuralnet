from __future__ import annotations

import argparse
import collections
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")

from prs_legal_mass_diagnostic import load_net, sample_midgame_states
from prs_illegal_slot_reasons import (
    classify_slot,
    articulation_cells,
    DIR_REVERSE,
    LONG_REVERSE,
    PIECE_NAME,
)
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.slot_map import (
    SlotMapper,
    NEIGHBORS,
    NUM_CELLS,
    DIR_OFFSET,
    THROW_OFFSET,
    LONG_OFFSET,
    HAND_OFFSET,
    PASS_SLOT,
    PT_GRASSHOPPER,
    PT_MOSQUITO,
    PT_PILLBUG,
    PT_QUEEN,
    PT_BEETLE,
)


def cycle_cells(mapper: SlotMapper) -> set[int]:
    occ = [cell for cell in range(NUM_CELLS) if int(mapper.heights[cell]) > 0]
    occ_set = set(occ)
    degree = {c: 0 for c in occ}
    adj = {}
    for c in occ:
        nbrs = [int(nb) for nb in NEIGHBORS[c] if nb >= 0 and int(nb) in occ_set]
        adj[c] = nbrs
        degree[c] = len(nbrs)
    queue = collections.deque([c for c in occ if degree[c] < 2])
    alive = {c: True for c in occ}
    while queue:
        u = queue.popleft()
        if not alive[u]:
            continue
        alive[u] = False
        for v in adj[u]:
            if alive[v]:
                degree[v] -= 1
                if degree[v] == 1:
                    queue.append(v)
    return {c for c in occ if alive[c]}


def source_has_legal_move(legal_np: np.ndarray, nlegal: int, fc: int) -> bool:
    for j in range(nlegal):
        mv = legal_np[j]
        if int(mv[0]) != 1:
            continue
        mfc = int(mv[2]) | (int(mv[3]) << 8)
        if mfc == fc:
            return True
    return False


def gate_blocked_adjacent_slot(slot: int, mapper: SlotMapper) -> bool:
    if not (DIR_OFFSET <= slot < THROW_OFFSET):
        return False
    rel = slot - DIR_OFFSET
    pidx = rel // 6
    d = rel % 6
    pt, rank = DIR_REVERSE[pidx]
    if pt == PT_GRASSHOPPER:
        return False
    cells = mapper.by_type.get(pt, [])
    if rank >= len(cells):
        return False
    fc = int(cells[rank])
    tc = int(NEIGHBORS[fc, d])
    if tc < 0:
        return False
    # Ground-level adjacent slide gate heuristic.
    if int(mapper.heights[fc]) != 1:
        return False
    if int(mapper.heights[tc]) != 0:
        return False
    shoulders = []
    for dd in range(6):
        nb = int(NEIGHBORS[fc, dd])
        if nb >= 0 and any(int(NEIGHBORS[tc, td]) == nb for td in range(6)):
            shoulders.append(nb)
    shoulders = list(dict.fromkeys(shoulders))
    if len(shoulders) != 2:
        return False
    return int(mapper.heights[shoulders[0]]) > 0 and int(mapper.heights[shoulders[1]]) > 0


def legal_move_piece_type(move: np.ndarray) -> int:
    return int(move[1] & 0x0F)


def summarize_scores(name: str, vals: list[float], positives: list[bool]) -> str:
    if not vals:
        return f"{name}: n=0"
    arr = np.asarray(vals, dtype=np.float32)
    pos = np.asarray(positives, dtype=np.bool_)
    return (
        f"{name}: n={arr.size}  mean_score={arr.mean():.3f}  "
        f"median={np.median(arr):.3f}  recall@0.5={pos.mean():.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/workspace/hive_neuralnet/checkpoints_prs_v2/prs_v2_iter_1450.pt")
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--min-ply", type=int, default=20)
    ap.add_argument("--max-ply", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expansion-mask", type=int, default=7)
    args = ap.parse_args()

    net = load_net(args.checkpoint)
    states = sample_midgame_states(
        num_samples=args.samples,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        expansion_mask=args.expansion_mask,
        seed=args.seed,
    )
    B = int(states.shape[0])
    orch = PRSMCTSOrchestratorV2(
        net,
        PRSMCTSConfigV2(num_simulations=16, batch_size=B, expansion_mask=args.expansion_mask),
    )

    prs_batch = orch.encoder.encode_batch(states, B)
    legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(states, B)
    kernel_out = orch._classify_kernel(states, legal_t, nlegal_t, B)
    slot_of_legal_t = kernel_out[8].to(torch.long)
    with torch.inference_mode():
        _policy, _value, aux = net.forward_train_from_kernel(prs_batch, kernel_out)
    slot_probs = torch.sigmoid(aux["slot_legality_logits"].float()).detach().cpu().numpy()

    legal_np = legal_t.cpu().numpy()
    nlegal_np = nlegal_t.cpu().numpy()
    state_np = states.cpu().numpy()

    piece_scores: dict[str, list[float]] = collections.defaultdict(list)
    piece_pos: dict[str, list[bool]] = collections.defaultdict(list)
    onehive_scores: list[float] = []
    onehive_pos: list[bool] = []
    gate_illegal_scores: list[float] = []
    gate_illegal_pos: list[bool] = []
    gate_legal_scores: list[float] = []
    gate_legal_pos: list[bool] = []
    ring_legal_scores: list[float] = []
    ring_legal_pos: list[bool] = []
    nonring_legal_scores: list[float] = []
    nonring_legal_pos: list[bool] = []
    legal_scores: list[float] = []
    legal_pos: list[bool] = []
    illegal_scores: list[float] = []
    illegal_pos: list[bool] = []

    for i in range(B):
        n_i = int(nlegal_np[i])
        mapper = SlotMapper(state_np[i])
        ap_cells = articulation_cells(mapper)
        cyc_cells = cycle_cells(mapper)
        legal_slots = set(int(x) for x in slot_of_legal_t[i, :n_i].cpu().numpy() if int(x) >= 0)

        # Actual legal moves by piece type and ring status.
        for j in range(n_i):
            slot = int(slot_of_legal_t[i, j].item())
            if slot < 0:
                continue
            prob = float(slot_probs[i, slot])
            move = legal_np[i, j]
            pt = legal_move_piece_type(move)
            name = PIECE_NAME.get(pt, f"pt{pt}")
            piece_scores[name].append(prob)
            piece_pos[name].append(prob >= 0.5)
            legal_scores.append(prob)
            legal_pos.append(prob >= 0.5)
            if int(move[0]) == 1:
                fc = int(move[2]) | (int(move[3]) << 8)
                if fc in cyc_cells:
                    ring_legal_scores.append(prob)
                    ring_legal_pos.append(prob >= 0.5)
                else:
                    nonring_legal_scores.append(prob)
                    nonring_legal_pos.append(prob >= 0.5)
                # Legal adjacent non-gated slides for contrast.
                tc = int(move[4]) | (int(move[5]) << 8)
                if any(int(NEIGHBORS[fc, d]) == tc for d in range(6)):
                    if not gate_blocked_adjacent_slot(slot, mapper):
                        gate_legal_scores.append(prob)
                        gate_legal_pos.append(prob >= 0.5)

        # Illegal slots by reason.
        for slot in range(PASS_SLOT + 1):
            if slot in legal_slots:
                continue
            prob = float(slot_probs[i, slot])
            reason = classify_slot(slot, mapper, legal_np[i], n_i, ap_cells)
            illegal_scores.append(prob)
            illegal_pos.append(prob < 0.5)
            if reason == "articulation_break":
                onehive_scores.append(prob)
                onehive_pos.append(prob < 0.5)
            if gate_blocked_adjacent_slot(slot, mapper):
                gate_illegal_scores.append(prob)
                gate_illegal_pos.append(prob < 0.5)

    print(f"checkpoint: {args.checkpoint}")
    print(f"positions: {B}  ply_range=[{args.min_ply}, {args.max_ply}]  expansion_mask={args.expansion_mask}")
    print()
    print(summarize_scores("overall legal slots", legal_scores, legal_pos))
    print(summarize_scores("overall illegal slots", illegal_scores, illegal_pos))
    print()
    print("per-piece legal move recognition:")
    piece_order = ["queen", "ant", "grasshopper", "spider", "beetle", "mosquito", "ladybug", "pillbug"]
    for name in piece_order:
        print("  " + summarize_scores(name, piece_scores[name], piece_pos[name]))
    print()
    print(summarize_scores("one-hive / articulation illegal slots", onehive_scores, onehive_pos))
    print(summarize_scores("gate-blocked adjacent illegal slots", gate_illegal_scores, gate_illegal_pos))
    print(summarize_scores("legal adjacent non-gated slides", gate_legal_scores, gate_legal_pos))
    print(summarize_scores("legal moves from ring cells", ring_legal_scores, ring_legal_pos))
    print(summarize_scores("legal moves from non-ring cells", nonring_legal_scores, nonring_legal_pos))


if __name__ == "__main__":
    main()
