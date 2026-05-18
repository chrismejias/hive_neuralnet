from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import torch

import hive_gpu
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.slot_map import NEIGHBORS, decode_top_colors_and_types

TURN_OFFSET = 3412
QUEEN_CELL_OFFSET = 3392


def load_prs_checkpoint(path: str) -> HivePRSTransformerV2:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = HivePRSTransformerV2(ckpt["net_config"]).cuda().eval()
    model_state = ckpt["model_state"]
    current = net.state_dict()
    filtered = {}
    for key, tensor in model_state.items():
        if key not in current:
            continue
        if current[key].shape != tensor.shape:
            continue
        filtered[key] = tensor
    net.load_state_dict(filtered, strict=False)
    return net


def parse_turn(state_bytes: np.ndarray) -> int:
    return int(state_bytes[TURN_OFFSET]) | (int(state_bytes[TURN_OFFSET + 1]) << 8)


def current_queen_surround_counts(state_bytes: np.ndarray) -> np.ndarray:
    target = np.zeros(32, dtype=np.float32)
    top_color, top_type, heights = decode_top_colors_and_types(state_bytes)
    queen_cells = state_bytes[QUEEN_CELL_OFFSET:QUEEN_CELL_OFFSET + 4].view(np.uint16)
    for queen_color in range(2):
        qc = int(queen_cells[queen_color])
        if qc == 0xFFFF:
            continue
        for d in range(6):
            nb = int(NEIGHBORS[qc, d])
            if nb >= 0 and int(heights[nb]) > 0:
                pt = int(top_type[nb])
                pc = int(top_color[nb])
                if pt > 0 and pc >= 0:
                    target[queen_color * 16 + pc * 8 + (pt - 1)] += 1.0
    return target


def mobility_count_vectors(orch: PRSMCTSOrchestratorV2, step_states: list[np.ndarray]) -> list[np.ndarray]:
    return orch._compute_mobility_count_vectors(step_states)


def infer_aux(
    net: HivePRSTransformerV2,
    states_np: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    ext = hive_gpu.load_extension()
    encoder = PRSEncoder()
    qs_out: list[np.ndarray] = []
    fm_out: list[np.ndarray] = []
    for start in range(0, len(states_np), batch_size):
        end = min(start + batch_size, len(states_np))
        sb = states_np[start:end]
        states_gpu = torch.from_numpy(sb).cuda()
        prs_batch = encoder.encode_batch(states_gpu, end - start)
        legal_t, nlegal_t = ext.generate_legal_moves_batch(states_gpu, end - start)
        kernel_out = ext.prs_v2_classify_batch(states_gpu, legal_t, nlegal_t, end - start, ext.MAX_LEGAL_MOVES)
        with torch.inference_mode():
            _, _, aux = net.forward_train_from_kernel(prs_batch, kernel_out)
        qs_out.append(aux["queen_surround_logits"].detach().cpu().numpy())
        fm_out.append(aux["final_mobility_logits"].detach().cpu().numpy())
    return np.concatenate(qs_out, axis=0), np.concatenate(fm_out, axis=0)


def bucket_label(turn: int) -> str | None:
    if turn < 40:
        return None
    lo = 40 + ((turn - 40) // 40) * 40
    hi = lo + 40
    return f"{lo}-{hi}"


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).mean())


def summarize_bucket(rows: Iterable[dict]) -> dict[str, float]:
    rows = list(rows)
    out: dict[str, float] = {"n": float(len(rows))}
    if not rows:
        return out
    for key in (
        "qs_0950", "qs_1000", "qs_heur",
        "fm_0950", "fm_1000", "fm_heur",
    ):
        out[key] = float(np.mean([r[key] for r in rows]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-950", required=True)
    ap.add_argument("--checkpoint-1000", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--simulations", type=int, default=256)
    ap.add_argument("--max-considered", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    net_1000 = load_prs_checkpoint(args.checkpoint_1000)
    net_950 = load_prs_checkpoint(args.checkpoint_950)

    cfg = PRSMCTSConfigV2(
        num_simulations=args.simulations,
        max_num_considered_actions=args.max_considered,
        batch_size=args.games,
        wave_parallel=True,
        temperature=1.0,
        temperature_drop_move=0,
        dirichlet_epsilon=0.0,
    )
    orch = PRSMCTSOrchestratorV2(net_1000, cfg)
    games = orch.self_play_batch()

    examples = []
    completed_games = 0
    for game in games:
        if not game:
            continue
        if not any(ex.queen_surround_mask > 0.0 for ex in game):
            continue
        completed_games += 1
        prev_state = None
        for ex in game:
            if ex.queen_surround_mask <= 0.0:
                prev_state = ex.state_bytes
                continue
            turn = parse_turn(ex.state_bytes)
            label = bucket_label(turn)
            if label is None:
                prev_state = ex.state_bytes
                continue
            examples.append(
                {
                    "bucket": label,
                    "turn": turn,
                    "state": ex.state_bytes,
                    "prev_state": prev_state if prev_state is not None else ex.state_bytes,
                    "qs_target": ex.queen_surround_target,
                    "fm_target": ex.final_mobility_target,
                }
            )
            prev_state = ex.state_bytes

    if not examples:
        raise SystemExit("No completed non-capped positions >= turn 40 found in sample.")

    states_np = np.stack([r["state"] for r in examples], axis=0)
    qs_950, fm_950 = infer_aux(net_950, states_np)
    qs_1000, fm_1000 = infer_aux(net_1000, states_np)

    heur_qs = np.stack([current_queen_surround_counts(r["state"]) for r in examples], axis=0)
    heur_prev = mobility_count_vectors(orch, [r["prev_state"] for r in examples])
    heur_curr = mobility_count_vectors(orch, [r["state"] for r in examples])
    heur_fm = np.stack(
        [np.concatenate([heur_prev[i], heur_curr[i]], axis=0) for i in range(len(examples))],
        axis=0,
    )

    rows = []
    for i, r in enumerate(examples):
        rows.append(
            {
                "bucket": r["bucket"],
                "turn": r["turn"],
                "qs_0950": mae(qs_950[i], r["qs_target"]),
                "qs_1000": mae(qs_1000[i], r["qs_target"]),
                "qs_heur": mae(heur_qs[i], r["qs_target"]),
                "fm_0950": mae(fm_950[i], r["fm_target"]),
                "fm_1000": mae(fm_1000[i], r["fm_target"]),
                "fm_heur": mae(heur_fm[i], r["fm_target"]),
            }
        )

    print(f"Completed sampled games: {completed_games}/{args.games}")
    print(f"Scored positions >= turn 40: {len(rows)}")
    print()
    print("bucket\tn\tqs_0950\tqs_1000\tqs_heur\tfm_0950\tfm_1000\tfm_heur")
    bucket_order = sorted({r["bucket"] for r in rows}, key=lambda s: int(s.split("-")[0]))
    for bucket in bucket_order:
        s = summarize_bucket(r for r in rows if r["bucket"] == bucket)
        print(
            f"{bucket}\t{int(s['n'])}\t"
            f"{s.get('qs_0950', float('nan')):.4f}\t"
            f"{s.get('qs_1000', float('nan')):.4f}\t"
            f"{s.get('qs_heur', float('nan')):.4f}\t"
            f"{s.get('fm_0950', float('nan')):.4f}\t"
            f"{s.get('fm_1000', float('nan')):.4f}\t"
            f"{s.get('fm_heur', float('nan')):.4f}"
        )

    overall = summarize_bucket(rows)
    print()
    print(
        "overall\t{n}\t{qs_0950:.4f}\t{qs_1000:.4f}\t{qs_heur:.4f}\t{fm_0950:.4f}\t{fm_1000:.4f}\t{fm_heur:.4f}".format(
            n=int(overall["n"]),
            qs_0950=overall["qs_0950"],
            qs_1000=overall["qs_1000"],
            qs_heur=overall["qs_heur"],
            fm_0950=overall["fm_0950"],
            fm_1000=overall["fm_1000"],
            fm_heur=overall["fm_heur"],
        )
    )


if __name__ == "__main__":
    main()
