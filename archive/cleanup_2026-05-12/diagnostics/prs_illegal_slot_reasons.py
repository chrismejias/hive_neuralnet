from __future__ import annotations

import argparse
import collections
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")

from prs_legal_mass_diagnostic import load_net, sample_midgame_states
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.slot_map import (
    SlotMapper,
    NEIGHBORS,
    HEIGHT_OFFSET,
    MAX_STACK,
    NUM_CELLS,
    DIR_OFFSET,
    THROW_OFFSET,
    LONG_OFFSET,
    HAND_OFFSET,
    PASS_SLOT,
    C_HAND,
    C_MOVE,
    decode_throw_slot,
    PT_QUEEN, PT_ANT, PT_GRASSHOPPER, PT_SPIDER, PT_BEETLE, PT_MOSQUITO, PT_LADYBUG, PT_PILLBUG,
)

PIECE_TOTALS = {
    PT_QUEEN: 1,
    PT_ANT: 3,
    PT_GRASSHOPPER: 3,
    PT_SPIDER: 2,
    PT_BEETLE: 2,
    PT_MOSQUITO: 1,
    PT_LADYBUG: 1,
    PT_PILLBUG: 1,
}

DIR_REVERSE = {
    0: (PT_QUEEN, 0),
    1: (PT_BEETLE, 0),
    2: (PT_BEETLE, 1),
    3: (PT_GRASSHOPPER, 0),
    4: (PT_GRASSHOPPER, 1),
    5: (PT_GRASSHOPPER, 2),
    6: (PT_PILLBUG, 0),
    7: (PT_MOSQUITO, 0),
}

LONG_REVERSE = {
    0: (PT_ANT, 0),
    1: (PT_ANT, 1),
    2: (PT_ANT, 2),
    3: (PT_SPIDER, 0),
    4: (PT_SPIDER, 1),
    5: (PT_LADYBUG, 0),
    6: (PT_MOSQUITO, 0),
}

PIECE_NAME = {
    PT_QUEEN: "queen",
    PT_ANT: "ant",
    PT_GRASSHOPPER: "grasshopper",
    PT_SPIDER: "spider",
    PT_BEETLE: "beetle",
    PT_MOSQUITO: "mosquito",
    PT_LADYBUG: "ladybug",
    PT_PILLBUG: "pillbug",
}


def decode_piece_counts(state_bytes: np.ndarray) -> np.ndarray:
    heights = state_bytes[HEIGHT_OFFSET:HEIGHT_OFFSET + NUM_CELLS]
    pieces = state_bytes[:MAX_STACK * NUM_CELLS].reshape(MAX_STACK, NUM_CELLS)
    counts = np.zeros((2, 9), dtype=np.int32)
    for cell in range(NUM_CELLS):
        h = int(heights[cell])
        for level in range(h):
            packed = int(pieces[level, cell])
            pt = packed & 0x0F
            pc = (packed >> 4) & 0x01
            if 1 <= pt <= 8:
                counts[pc, pt] += 1
    return counts


def articulation_cells(mapper: SlotMapper) -> set[int]:
    occ = [cell for cell in range(NUM_CELLS) if int(mapper.heights[cell]) > 0]
    if len(occ) <= 2:
        return set()
    occ_set = set(occ)
    adj = {c: [int(nb) for nb in NEIGHBORS[c] if nb >= 0 and int(nb) in occ_set] for c in occ}

    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int] = {}
    ap: set[int] = set()
    t = 0

    def dfs(u: int) -> None:
        nonlocal t
        children = 0
        t += 1
        disc[u] = low[u] = t
        for v in adj[u]:
            if v not in disc:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if u not in parent and children > 1:
                    ap.add(u)
                if u in parent and low[v] >= disc[u]:
                    ap.add(u)
            elif parent.get(u) != v:
                low[u] = min(low[u], disc[v])

    start = occ[0]
    dfs(start)
    return ap


def source_has_any_move(legal_np: np.ndarray, nlegal: int, from_cell: int) -> bool:
    for j in range(nlegal):
        mv = legal_np[j]
        if int(mv[0]) != 1:
            continue
        fc = int(mv[2]) | (int(mv[3]) << 8)
        if fc == from_cell:
            return True
    return False


def candidate_move_legal(legal_np: np.ndarray, nlegal: int, pt: int, fc: int, tc: int) -> bool:
    for j in range(nlegal):
        mv = legal_np[j]
        if int(mv[0]) != 1:
            continue
        mpt = int(mv[1] & 0x0F)
        mfc = int(mv[2]) | (int(mv[3]) << 8)
        mtc = int(mv[4]) | (int(mv[5]) << 8)
        if mpt == pt and mfc == fc and mtc == tc:
            return True
    return False


def source_has_any_adjacent_move(legal_np: np.ndarray, nlegal: int, from_cell: int) -> bool:
    for j in range(nlegal):
        mv = legal_np[j]
        if int(mv[0]) != 1:
            continue
        fc = int(mv[2]) | (int(mv[3]) << 8)
        tc = int(mv[4]) | (int(mv[5]) << 8)
        if fc != from_cell:
            continue
        for d in range(6):
            if int(NEIGHBORS[from_cell, d]) == tc:
                return True
    return False


def classify_slot(
    slot: int,
    mapper: SlotMapper,
    legal_np: np.ndarray,
    nlegal: int,
    ap_cells: set[int],
) -> str:
    color = mapper.color
    board_counts = decode_piece_counts(mapper.state)
    if slot == PASS_SLOT:
        return "pass_illegal"

    if HAND_OFFSET <= slot < PASS_SLOT:
        rel = slot - HAND_OFFSET
        pt = rel // C_HAND + 1
        cell_tok = rel % C_HAND
        pieces_on_board = int(board_counts[color, pt])
        in_hand = PIECE_TOTALS[pt] - pieces_on_board
        if in_hand <= 0:
            return "hand_piece_unavailable"
        if cell_tok >= len(mapper.place_cells):
            return "inactive_place_cell"
        return "placement_rule_other"

    if LONG_OFFSET <= slot < HAND_OFFSET:
        rel = slot - LONG_OFFSET
        pidx = rel // C_MOVE
        cell_tok = rel % C_MOVE
        pt, rank = LONG_REVERSE[pidx]
        cells = mapper.by_type.get(pt, [])
        if rank >= len(cells):
            return "piece_unavailable"
        if cell_tok >= len(mapper.move_cells):
            return "inactive_move_cell"
        fc = int(cells[rank])
        tc = int(mapper.move_cells[cell_tok])
        if candidate_move_legal(legal_np, nlegal, pt, fc, tc):
            return "actually_legal"
        if not source_has_any_move(legal_np, nlegal, fc):
            if int(mapper.heights[fc]) == 1 and fc in ap_cells:
                return "articulation_break"
            return "immobile_other"
        return f"movement_rule_long_{PIECE_NAME[pt]}"

    if THROW_OFFSET <= slot < LONG_OFFSET:
        rel = slot - THROW_OFFSET
        thrower_idx = rel // 30
        throw_slot = rel % 30
        thrower_pt = PT_PILLBUG if thrower_idx == 0 else PT_MOSQUITO
        cells = mapper.by_type.get(thrower_pt, [])
        if not cells:
            return "piece_unavailable"
        pb = int(cells[0])
        td, dd = decode_throw_slot(throw_slot)
        fc = int(NEIGHBORS[pb, td])
        tc = int(NEIGHBORS[pb, dd])
        if fc < 0 or tc < 0:
            return "throw_geometry_invalid"
        if int(mapper.heights[fc]) == 0:
            return "throw_target_missing"
        if int(mapper.heights[tc]) > 0:
            return "throw_destination_occupied"
        return "throw_rule_other"

    if DIR_OFFSET <= slot < THROW_OFFSET:
        rel = slot - DIR_OFFSET
        pidx = rel // 6
        d = rel % 6
        pt, rank = DIR_REVERSE[pidx]
        cells = mapper.by_type.get(pt, [])
        if rank >= len(cells):
            return "piece_unavailable"
        fc = int(cells[rank])
        tc = int(NEIGHBORS[fc, d])
        if tc < 0:
            return "offboard_direction"
        if candidate_move_legal(legal_np, nlegal, pt, fc, tc):
            return "actually_legal"
        if not source_has_any_move(legal_np, nlegal, fc):
            if int(mapper.heights[fc]) == 1 and fc in ap_cells:
                return "articulation_break"
            return "immobile_other"
        if pt == PT_MOSQUITO and not source_has_any_adjacent_move(legal_np, nlegal, fc):
            return "movement_rule_dir_mosquito_ability"
        return f"movement_rule_dir_{PIECE_NAME[pt]}"

    return "unknown"


def run_probe(net, exp_mask: int, samples: int, min_ply: int, max_ply: int, seed: int) -> None:
    states = sample_midgame_states(
        num_samples=samples, min_ply=min_ply, max_ply=max_ply, expansion_mask=exp_mask, seed=seed,
    )
    B = int(states.shape[0])
    orch = PRSMCTSOrchestratorV2(net, PRSMCTSConfigV2(num_simulations=16, batch_size=B, expansion_mask=0))
    prs_batch = orch.encoder.encode_batch(states, B)
    legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(states, B)
    kernel_out = orch._classify_kernel(states, legal_t, nlegal_t, B)
    slot_of_legal_t = kernel_out[8].to(torch.long)
    logits_813, _ = orch._net_forward(prs_batch, kernel_out, B)
    probs = torch.softmax(logits_813.float(), dim=1)

    legal_mask = torch.zeros(B, probs.shape[1], dtype=torch.bool, device="cuda")
    valid = slot_of_legal_t >= 0
    legal_mask.scatter_(1, slot_of_legal_t.clamp(min=0), valid)

    topk = torch.topk(probs, 16, dim=1)
    states_np = states.cpu().numpy()
    legal_np = legal_t.cpu().numpy()
    nlegal_np = nlegal_t.cpu().numpy()
    top_idx = topk.indices.cpu().numpy()
    top_prob = topk.values.cpu().numpy()
    legal_mask_np = legal_mask.cpu().numpy()

    top1_counts = collections.Counter()
    top1_prob_sums = collections.Counter()
    mass_sums = collections.Counter()
    illegal_slots_seen = 0

    for i in range(B):
        mapper = SlotMapper(states_np[i])
        ap_cells = articulation_cells(mapper)
        first_illegal_done = False
        for rank in range(16):
            slot = int(top_idx[i, rank])
            prob = float(top_prob[i, rank])
            if legal_mask_np[i, slot]:
                continue
            illegal_slots_seen += 1
            reason = classify_slot(slot, mapper, legal_np[i], int(nlegal_np[i]), ap_cells)
            mass_sums[reason] += prob
            if not first_illegal_done:
                top1_counts[reason] += 1
                top1_prob_sums[reason] += prob
                first_illegal_done = True

    print(f"\n=== expansion_mask={exp_mask} ===")
    print(f"sampled states: {B}")
    print(f"illegal top-16 slots analyzed: {illegal_slots_seen}")
    print("top illegal slot per position:")
    total_top1 = sum(top1_counts.values())
    for reason, count in top1_counts.most_common():
        avg_prob = top1_prob_sums[reason] / max(count, 1)
        print(f"  {reason:24s}  {count:4d}  ({count/max(total_top1,1)*100:5.1f}%)  avg_p={avg_prob*100:5.2f}%")
    print("illegal top-16 mass by category:")
    total_mass = sum(mass_sums.values())
    for reason, mass in sorted(mass_sums.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {reason:24s}  {mass*100:6.2f}%  ({mass/max(total_mass,1e-9)*100:5.1f}% of illegal top16 mass)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--min-ply", type=int, default=20)
    ap.add_argument("--max-ply", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = load_net(args.checkpoint)
    run_probe(net, 0, args.samples, args.min_ply, args.max_ply, args.seed)
    run_probe(net, 7, args.samples, args.min_ply, args.max_ply, args.seed + 7)


if __name__ == "__main__":
    main()
