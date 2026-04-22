"""
Probe the PRS v2 checkpoint on random win-in-one positions.

For each expansion subset:
  1. Randomly sample positions where the side to move has at least one
     immediate winning move.
  2. Evaluate the current PRS v2 checkpoint on the raw GPU HiveState bytes.
  3. Report whether the value head scores the position highly and whether the
     policy ranks the winning move(s) near the top.

Unlike the legacy PRS probe, this script uses the current 813-slot PRS v2
head and the same slot-to-legal prior conversion used by self-play:

  * softmax over legal slots only
  * if multiple legal moves share a slot, that slot mass is divided equally
    across those legal moves
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from hive_engine.game_state import GameState, GameResult
from hive_engine.pieces import ExpansionConfig
from hive_gpu.endgame_generator import gamestate_to_gpu_bytes, positions_to_tensor
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.slot_map import N_SLOTS, THROW_OFFSET, map_legal_moves, SlotMapper

import hive_gpu

_OFF_TURN = 3412

_PIECE_NAME = {
    1: "queen",
    2: "ant",
    3: "grasshopper",
    4: "spider",
    5: "beetle",
    6: "mosquito",
    7: "ladybug",
    8: "pillbug",
}


def _latest_checkpoint() -> str | None:
    paths = sorted(glob.glob("checkpoints_prs_v2/prs_v2_iter_*.pt"))
    return paths[-1] if paths else None


def mask_label(mask: int) -> str:
    parts = []
    if mask & 1:
        parts.append("M")
    if mask & 2:
        parts.append("L")
    if mask & 4:
        parts.append("P")
    return "base+" + "+".join(parts) if parts else "base"


def mask_to_cfg(mask: int) -> ExpansionConfig:
    return ExpansionConfig(
        mosquito=bool(mask & 1),
        ladybug=bool(mask & 2),
        pillbug=bool(mask & 4),
    )


def random_game_until_win1(exp_cfg: ExpansionConfig, max_moves: int):
    """Yield (state, win_count) for random games that contain a win-in-one."""
    state = GameState(expansions=exp_cfg)
    for _ in range(max_moves):
        if state.result != GameResult.IN_PROGRESS:
            return
        moves = list(state.legal_moves())
        if not moves:
            return

        mover = state.current_player
        win_count = 0
        for move in moves:
            state.apply_move(move)
            won = (
                (mover.name == "WHITE" and state.result == GameResult.WHITE_WINS)
                or (mover.name == "BLACK" and state.result == GameResult.BLACK_WINS)
            )
            state.undo_move()
            if won:
                win_count += 1

        if win_count > 0:
            yield state.copy(), win_count
            return

        state.apply_move(random.choice(moves))


def find_gpu_winning_moves(
    ext,
    state_t: torch.Tensor,
    legal_moves_t: torch.Tensor,
    n_legal: int,
) -> np.ndarray:
    """Return legal indices of all immediate wins for the side to move."""
    if n_legal <= 0:
        return np.empty((0,), dtype=np.int64)

    trial = state_t.expand(n_legal, -1).clone()
    ext.apply_moves_batch(trial, legal_moves_t[:n_legal], n_legal)
    results = ext.check_results_batch(trial, n_legal)

    turn_byte = int(state_t[0, _OFF_TURN].item())
    is_white = (turn_byte % 2 == 0)
    win_mask = (results == 1) if is_white else (results == 2)
    return win_mask.nonzero(as_tuple=True)[0].cpu().numpy().astype(np.int64)


def winning_move_label(mapper: SlotMapper, move: np.ndarray) -> str:
    """Return a compact move-type label for a winning legal move."""
    mtype = int(move[0])
    piece_name = _PIECE_NAME.get(int(move[1] & 0x0F), f"piece{int(move[1] & 0x0F)}")
    if mtype == 0:
        return f"place:{piece_name}"
    if mtype == 2:
        return "pass"
    slot = int(mapper.classify(move))
    if THROW_OFFSET <= slot < THROW_OFFSET + 60:
        return f"throw:{piece_name}"
    return f"move:{piece_name}"


def evaluate_position(
    ext,
    encoder: PRSEncoder,
    net: HivePRSTransformerV2,
    state: GameState,
    device: str,
) -> dict[str, float] | None:
    """Evaluate one Python GameState with the current PRS v2 network."""
    state_bytes = np.frombuffer(gamestate_to_gpu_bytes(state), dtype=np.uint8).copy()
    states_t = positions_to_tensor([state_bytes.tobytes()], device=device)
    legal_t, nlegal_t = ext.generate_legal_moves_batch(states_t, 1)

    n_legal = int(nlegal_t[0].item())
    if n_legal <= 0:
        return None

    legal_moves_t = legal_t[0]
    legal_moves_np = legal_moves_t[:n_legal].cpu().numpy()
    win_idx = find_gpu_winning_moves(ext, states_t, legal_moves_t, n_legal)
    if win_idx.size == 0:
        return None

    slot_of_legal, _, _ = map_legal_moves(state_bytes, legal_moves_np, n_legal)
    valid_slots = slot_of_legal >= 0
    if not bool(valid_slots.any()):
        return None
    mapper = SlotMapper(state_bytes)

    prs_batch = encoder.encode_batch(states_t, 1)
    with torch.no_grad():
        logits, value = net(prs_batch, state_bytes[None, :])

    logits_813 = logits[0]
    safe_slots = np.clip(slot_of_legal, 0, None)
    legal_mask = torch.zeros(N_SLOTS, dtype=torch.bool, device=logits_813.device)
    if np.any(valid_slots):
        legal_mask[torch.from_numpy(safe_slots[valid_slots]).to(logits_813.device)] = True

    masked_logits = logits_813.masked_fill(~legal_mask, float("-inf"))
    slot_probs = torch.softmax(masked_logits, dim=0).cpu().numpy()

    counts = np.zeros(N_SLOTS, dtype=np.float32)
    for slot in slot_of_legal:
        if slot >= 0:
            counts[int(slot)] += 1.0

    move_probs = np.zeros(n_legal, dtype=np.float32)
    move_logits = np.full(n_legal, -1e30, dtype=np.float32)
    for i, slot in enumerate(slot_of_legal):
        if slot < 0:
            continue
        move_probs[i] = float(slot_probs[int(slot)] / max(counts[int(slot)], 1.0))
        move_logits[i] = float(logits_813[int(slot)].item())

    # Rank legal moves by the same per-legal probability used in self-play.
    order = np.argsort(-move_probs, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n_legal + 1)

    top_idx = int(order[0])
    win_prob_mass = float(move_probs[win_idx].sum())
    best_win_prob = float(move_probs[win_idx].max())
    best_win_rank = int(ranks[win_idx].min())
    top_move_is_win = bool(np.any(win_idx == top_idx))

    win_labels = [winning_move_label(mapper, legal_moves_np[int(i)]) for i in win_idx]
    per_label_rows: list[dict[str, float | str]] = []
    for i in win_idx:
        i_int = int(i)
        per_label_rows.append({
            "label": winning_move_label(mapper, legal_moves_np[i_int]),
            "prob": float(move_probs[i_int]),
            "rank": float(ranks[i_int]),
            "top1": float(ranks[i_int] <= 1),
            "top3": float(ranks[i_int] <= 3),
            "top10": float(ranks[i_int] <= 10),
        })

    return {
        "value": float(value[0, 0].item()),
        "n_legal": float(n_legal),
        "n_winning": float(win_idx.size),
        "win_mass": win_prob_mass,
        "best_win_prob": best_win_prob,
        "best_win_rank": float(best_win_rank),
        "top_move_is_win": float(top_move_is_win),
        "top3_has_win": float(best_win_rank <= 3),
        "top10_has_win": float(best_win_rank <= 10),
        "all_win_slots_mappable": float(np.all(slot_of_legal[win_idx] >= 0)),
        "top_move_prob": float(move_probs[top_idx]),
        "top_move_logit": float(move_logits[top_idx]),
        "win_labels": win_labels,
        "per_label_rows": per_label_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num", type=int, default=10, help="Positions per expansion subset.")
    parser.add_argument("--max-games", type=int, default=3000, help="Random games to try per subset.")
    parser.add_argument("--max-moves", type=int, default=120, help="Random plies per game.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or _latest_checkpoint()
    if checkpoint_path is None:
        raise SystemExit("No PRS v2 checkpoint found in checkpoints_prs_v2/")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = hive_gpu.load_extension()
    encoder = PRSEncoder()

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = HivePRSTransformerV2(ckpt["net_config"]).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    print(f"  Iteration : {ckpt.get('iteration')}")
    print(f"  Device    : {device}")
    print(f"  Params    : {sum(p.numel() for p in net.parameters()):,}")

    print(
        f"\n{'Mask':<6} {'Label':<12} {'n':>4} {'Val':>8} {'WinMass':>9} "
        f"{'BestWinP':>9} {'Top1':>7} {'Top3':>7} {'Top10':>7} {'MeanRank':>10}"
    )
    print("-" * 92)

    all_rows: list[dict[str, float]] = []
    label_rows: list[dict[str, float | str]] = []

    for mask in range(8):
        exp_cfg = mask_to_cfg(mask)
        rows: list[dict[str, float]] = []
        games_tried = 0

        while len(rows) < args.num and games_tried < args.max_games:
            games_tried += 1
            hit = False
            for state, _win_count_py in random_game_until_win1(exp_cfg, args.max_moves):
                hit = True
                row = evaluate_position(ext, encoder, net, state, device)
                if row is not None:
                    rows.append(row)
                break
            if not hit:
                continue

        if not rows:
            print(f"{mask:<6} {mask_label(mask):<12} {0:>4} {'--':>8} {'--':>9} {'--':>9} {'--':>7} {'--':>7} {'--':>7} {'--':>10}")
            continue

        arr = rows
        n = len(arr)
        mean_val = float(np.mean([r["value"] for r in arr]))
        mean_mass = float(np.mean([r["win_mass"] for r in arr]))
        mean_best = float(np.mean([r["best_win_prob"] for r in arr]))
        top1 = int(sum(r["top_move_is_win"] > 0.5 for r in arr))
        top3 = int(sum(r["top3_has_win"] > 0.5 for r in arr))
        top10 = int(sum(r["top10_has_win"] > 0.5 for r in arr))
        mean_rank = float(np.mean([r["best_win_rank"] for r in arr]))

        print(
            f"{mask:<6} {mask_label(mask):<12} {n:>4} "
            f"{mean_val:>+8.3f} {mean_mass:>9.4f} {mean_best:>9.4f} "
            f"{top1:>3}/{n:<3} {top3:>3}/{n:<3} {top10:>3}/{n:<3} {mean_rank:>10.2f}"
        )
        all_rows.extend(arr)
        for row in arr:
            label_rows.extend(row["per_label_rows"])

    print("-" * 92)
    n_all = len(all_rows)
    if n_all == 0:
        print("No evaluable win-in-one positions found.")
        return

    mean_val = float(np.mean([r["value"] for r in all_rows]))
    mean_mass = float(np.mean([r["win_mass"] for r in all_rows]))
    mean_best = float(np.mean([r["best_win_prob"] for r in all_rows]))
    mean_rank = float(np.mean([r["best_win_rank"] for r in all_rows]))
    top1 = int(sum(r["top_move_is_win"] > 0.5 for r in all_rows))
    top3 = int(sum(r["top3_has_win"] > 0.5 for r in all_rows))
    top10 = int(sum(r["top10_has_win"] > 0.5 for r in all_rows))
    full_map = int(sum(r["all_win_slots_mappable"] > 0.5 for r in all_rows))

    print(f"\n=== Aggregate across all subsets ({n_all} positions) ===")
    print(f"  Mean value output        : {mean_val:+.4f}")
    print(f"  Mean winning move mass   : {mean_mass:.4f}")
    print(f"  Mean best winning move p : {mean_best:.4f}")
    print(f"  Winning move is top-1    : {top1}/{n_all}")
    print(f"  Winning move in top-3    : {top3}/{n_all}")
    print(f"  Winning move in top-10   : {top10}/{n_all}")
    print(f"  Mean best winning rank   : {mean_rank:.2f}")
    print(f"  All winning slots mapped : {full_map}/{n_all}")

    by_label: dict[str, list[dict[str, float | str]]] = collections.defaultdict(list)
    for row in label_rows:
        by_label[str(row["label"])].append(row)

    print("\n=== Winning Move Breakdown ===")
    print(
        f"{'Label':<20} {'n':>5} {'MeanP':>9} {'Top1':>9} "
        f"{'Top3':>9} {'Top10':>9} {'MeanRank':>10}"
    )
    print("-" * 76)
    for label, rows in sorted(by_label.items(), key=lambda item: (-len(item[1]), item[0])):
        n = len(rows)
        mean_prob = float(np.mean([float(r["prob"]) for r in rows]))
        top1_n = int(sum(float(r["top1"]) > 0.5 for r in rows))
        top3_n = int(sum(float(r["top3"]) > 0.5 for r in rows))
        top10_n = int(sum(float(r["top10"]) > 0.5 for r in rows))
        mean_rank = float(np.mean([float(r["rank"]) for r in rows]))
        print(
            f"{label:<20} {n:>5} {mean_prob:>9.4f} "
            f"{top1_n:>4}/{n:<4} {top3_n:>4}/{n:<4} "
            f"{top10_n:>4}/{n:<4} {mean_rank:>10.2f}"
        )


if __name__ == "__main__":
    main()
