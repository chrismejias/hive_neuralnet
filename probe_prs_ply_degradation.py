"""Probe whether PRS policy/value quality degrades as game length increases.

Two sampling modes:
  * random: fast random legal playouts, good for checking generic late-game
    collapse in value magnitude or legal-policy sharpness.
  * selfplay: on-distribution PRS self-play, slower but can also compare the
    raw network policy to the MCTS target distribution.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import hive_gpu
from hive_engine.game_state import GameResult, GameState
from hive_engine.pieces import ExpansionConfig
from hive_gpu.endgame_generator import gamestate_to_gpu_bytes, positions_to_tensor
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
from hive_prs.prs_replay_buffer_v2 import PRSTrainingExampleV2, _collate_noaug
from hive_prs.prs_encoder import PRSEncoder
from hive_prs.slot_map import N_SLOTS, map_legal_moves
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2


def latest_checkpoint() -> str | None:
    paths = sorted(glob.glob("checkpoints_prs_v2/prs_v2_iter_*.pt"))
    return paths[-1] if paths else None


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def summarize_bucket(bucket_stats: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not bucket_stats:
        return out
    keys = bucket_stats[0].keys()
    for key in keys:
        out[key] = float(np.mean([row[key] for row in bucket_stats]))
    out["n"] = float(len(bucket_stats))
    return out


def compute_policy_metrics(
    logits: torch.Tensor,         # (B, 813)
    legal_masks: torch.Tensor,    # (B, 813) bool
    targets: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    masked_logits = logits.masked_fill(~legal_masks, float("-inf"))
    any_legal = legal_masks.any(dim=1, keepdim=True)
    safe_logits = torch.where(any_legal, masked_logits, torch.zeros_like(masked_logits))
    probs = torch.softmax(safe_logits, dim=-1)
    probs = probs * any_legal.float()

    rows: list[dict[str, float]] = []
    probs_np = probs.detach().cpu().numpy()
    masks_np = legal_masks.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy() if targets is not None else None

    for i in range(probs_np.shape[0]):
        mask = masks_np[i].astype(bool)
        p = probs_np[i][mask]
        t = targets_np[i][mask] if targets_np is not None else None
        nlegal = int(mask.sum())
        if nlegal <= 0:
            rows.append({
                "nlegal": 0.0,
                "policy_entropy": 0.0,
                "policy_entropy_norm": 0.0,
                "policy_top1": 0.0,
                "policy_gap12": 0.0,
            })
            continue

        p = np.clip(p, 1e-12, 1.0)
        p_sorted = np.sort(p)[::-1]
        log_n = math.log(max(nlegal, 2))

        p_h = float(-(p * np.log(p)).sum())

        row = {
            "nlegal": float(nlegal),
            "policy_entropy": p_h,
            "policy_entropy_norm": p_h / log_n if log_n > 0 else 0.0,
            "policy_top1": float(p_sorted[0]),
            "policy_gap12": float(p_sorted[0] - p_sorted[1]) if p_sorted.size > 1 else float(p_sorted[0]),
        }
        if t is not None:
            t = np.clip(t, 0.0, 1.0)
            t_sorted = np.sort(t)[::-1]
            t_pos = t[t > 0]
            t_h = float(-(t_pos * np.log(t_pos)).sum()) if t_pos.size else 0.0
            row["target_entropy"] = t_h
            row["target_entropy_norm"] = t_h / log_n if log_n > 0 else 0.0
            row["target_top1"] = float(t_sorted[0]) if t_sorted.size else 0.0
        rows.append(row)
    return rows


def mask_to_cfg(mask: int) -> ExpansionConfig:
    return ExpansionConfig(
        mosquito=bool(mask & 1),
        ladybug=bool(mask & 2),
        pillbug=bool(mask & 4),
    )


def collect_random_positions(
    games: int,
    max_game_len: int,
    expansion_mask: int,
    seed: int,
) -> list[tuple[int, np.ndarray]]:
    rng = random.Random(seed)
    out: list[tuple[int, np.ndarray]] = []
    cfg = mask_to_cfg(expansion_mask)
    for _ in range(games):
        state = GameState(expansions=cfg)
        ply = 0
        while state.result == GameResult.IN_PROGRESS and ply < max_game_len:
            out.append((ply, np.frombuffer(gamestate_to_gpu_bytes(state), dtype=np.uint8).copy()))
            moves = list(state.legal_moves())
            if not moves:
                break
            state.apply_move(rng.choice(moves))
            ply += 1
    return out


def evaluate_random_positions(
    net: HivePRSTransformerV2,
    items: list[tuple[int, np.ndarray]],
    eval_batch_size: int,
) -> tuple[dict[int, list[dict[str, float]]], list[dict[str, float]]]:
    ext = hive_gpu.load_extension()
    encoder = PRSEncoder()
    bucket_rows: dict[int, list[dict[str, float]]] = defaultdict(list)
    all_rows: list[dict[str, float]] = []

    for chunk in batched(items, eval_batch_size):
        plys = [ply for ply, _ in chunk]
        state_bytes = [sb for _, sb in chunk]
        states_t = positions_to_tensor([sb.tobytes() for sb in state_bytes], device="cuda")
        B = len(chunk)
        legal_t, nlegal_t = ext.generate_legal_moves_batch(states_t, B)
        prs_batch = encoder.encode_batch(states_t, B)
        with torch.no_grad():
            logits, values = net(prs_batch, np.stack(state_bytes, axis=0))

        legal_masks = np.zeros((B, N_SLOTS), dtype=bool)
        nlegal_np = nlegal_t.cpu().numpy()
        legal_np = legal_t.cpu().numpy()
        for i in range(B):
            n_i = int(nlegal_np[i])
            if n_i <= 0:
                continue
            slot_of_legal, _, _ = map_legal_moves(state_bytes[i], legal_np[i, :n_i], n_i)
            for s in slot_of_legal:
                if s >= 0:
                    legal_masks[i, int(s)] = True
        policy_rows = compute_policy_metrics(
            logits.float(),
            torch.from_numpy(legal_masks).to(logits.device),
            None,
        )
        values_np = values.squeeze(-1).detach().cpu().numpy()
        for ply, v, prow in zip(plys, values_np, policy_rows):
            row = {
                "value_mean": float(v),
                "value_abs": float(abs(v)),
                "value_near_zero": float(abs(v) < 0.10),
                **prow,
            }
            all_rows.append((ply, row))
    return bucket_rows, all_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe PRS policy/value by ply.")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--mode", choices=["random", "selfplay"], default="random")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--max-considered", type=int, default=16)
    ap.add_argument("--expansion-mask", type=int, default=7)
    ap.add_argument("--bucket-size", type=int, default=20)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--max-game-len", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = args.checkpoint or latest_checkpoint()
    if not ckpt_path:
        raise SystemExit("No checkpoint found")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_config = ckpt["net_config"]
    net = HivePRSTransformerV2(net_config).cuda()
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    bucket_rows: dict[int, list[dict[str, float]]] = defaultdict(list)
    all_rows_only: list[dict[str, float]] = []

    print(f"Checkpoint: {ckpt_path}")

    if args.mode == "selfplay":
        mcts_cfg = PRSMCTSConfigV2(
            num_simulations=args.simulations,
            max_num_considered_actions=args.max_considered,
            batch_size=args.games,
            expansion_mask=args.expansion_mask,
            max_game_length=args.max_game_len,
        )
        orch = PRSMCTSOrchestratorV2(net, mcts_cfg)
        with torch.no_grad():
            games = orch.self_play_batch()

        flat: list[tuple[int, PRSTrainingExampleV2]] = []
        game_lengths = []
        for game_exs in games:
            game_lengths.append(len(game_exs))
            for ply, ex in enumerate(game_exs):
                flat.append((ply, ex))

        for chunk in batched(flat, args.eval_batch_size):
            plys = [ply for ply, _ in chunk]
            samples = [ex for _, ex in chunk]
            batch = _collate_noaug(samples).to("cuda", non_blocking=False)
            with torch.no_grad():
                logits, values = net(batch.prs_batch, batch.state_bytes)
            policy_rows = compute_policy_metrics(logits.float(), batch.legal_masks, batch.slot_targets)
            values_np = values.squeeze(-1).detach().cpu().numpy()

            for ply, v, prow in zip(plys, values_np, policy_rows):
                row = {
                    "value_mean": float(v),
                    "value_abs": float(abs(v)),
                    "value_near_zero": float(abs(v) < 0.10),
                    **prow,
                }
                bucket = (ply // args.bucket_size) * args.bucket_size
                bucket_rows[bucket].append(row)
                all_rows_only.append(row)

        print(
            f"Self-play sample: {args.games} games, {args.simulations} sims, "
            f"k={args.max_considered}, expansions={args.expansion_mask}"
        )
        print(
            f"Recorded positions: {len(flat)} "
            f"(mean game length {np.mean(game_lengths):.1f}, median {np.median(game_lengths):.1f}, "
            f"max {max(game_lengths) if game_lengths else 0})"
        )
        header = (
            "Bucket   N     |v|   |v|<.1  Hpol/logN  top1   gap12  "
            "Htgt/logN  tgt_top1  nlegal"
        )
    else:
        items = collect_random_positions(
            games=args.games,
            max_game_len=args.max_game_len,
            expansion_mask=args.expansion_mask,
            seed=args.seed,
        )
        encoder = PRSEncoder()
        ext = hive_gpu.load_extension()
        del encoder, ext
        for chunk in batched(items, args.eval_batch_size):
            plys = [ply for ply, _ in chunk]
            state_bytes = [sb for _, sb in chunk]
            states_t = positions_to_tensor([sb.tobytes() for sb in state_bytes], device="cuda")
            B = len(chunk)
            legal_t, nlegal_t = hive_gpu.load_extension().generate_legal_moves_batch(states_t, B)
            prs_batch = PRSEncoder().encode_batch(states_t, B)
            with torch.no_grad():
                logits, values = net(prs_batch, np.stack(state_bytes, axis=0))
            legal_masks = np.zeros((B, N_SLOTS), dtype=bool)
            nlegal_np = nlegal_t.cpu().numpy()
            legal_np = legal_t.cpu().numpy()
            for i in range(B):
                n_i = int(nlegal_np[i])
                if n_i <= 0:
                    continue
                slot_of_legal, _, _ = map_legal_moves(state_bytes[i], legal_np[i, :n_i], n_i)
                for s in slot_of_legal:
                    if s >= 0:
                        legal_masks[i, int(s)] = True
            policy_rows = compute_policy_metrics(
                logits.float(),
                torch.from_numpy(legal_masks).to(logits.device),
            )
            values_np = values.squeeze(-1).detach().cpu().numpy()
            for ply, v, prow in zip(plys, values_np, policy_rows):
                row = {
                    "value_mean": float(v),
                    "value_abs": float(abs(v)),
                    "value_near_zero": float(abs(v) < 0.10),
                    **prow,
                }
                bucket = (ply // args.bucket_size) * args.bucket_size
                bucket_rows[bucket].append(row)
                all_rows_only.append(row)
        game_lengths = []
        for _ in range(args.games):
            pass
        print(
            f"Random playout sample: {args.games} games, expansions={args.expansion_mask}, "
            f"positions={len(items)}"
        )
        header = "Bucket   N     |v|   |v|<.1  Hpol/logN  top1   gap12  nlegal"

    print()
    print(header)
    print("-" * 86)

    for bucket in sorted(bucket_rows):
        s = summarize_bucket(bucket_rows[bucket])
        lo = bucket
        hi = bucket + args.bucket_size - 1
        if args.mode == "selfplay":
            print(
                f"{lo:3d}-{hi:<3d}  "
                f"{int(s['n']):4d}  "
                f"{s['value_abs']:.3f}   "
                f"{s['value_near_zero']:.3f}    "
                f"{s['policy_entropy_norm']:.3f}      "
                f"{s['policy_top1']:.3f}  "
                f"{s['policy_gap12']:.3f}   "
                f"{s['target_entropy_norm']:.3f}       "
                f"{s['target_top1']:.3f}    "
                f"{s['nlegal']:.1f}"
            )
        else:
            print(
                f"{lo:3d}-{hi:<3d}  "
                f"{int(s['n']):4d}  "
                f"{s['value_abs']:.3f}   "
                f"{s['value_near_zero']:.3f}    "
                f"{s['policy_entropy_norm']:.3f}      "
                f"{s['policy_top1']:.3f}  "
                f"{s['policy_gap12']:.3f}   "
                f"{s['nlegal']:.1f}"
            )

    print()
    overall = summarize_bucket(all_rows_only)
    if args.mode == "selfplay":
        print(
            "Overall: "
            f"|v|={overall['value_abs']:.3f}, "
            f"|v|<.1={overall['value_near_zero']:.3f}, "
            f"Hpol/logN={overall['policy_entropy_norm']:.3f}, "
            f"top1={overall['policy_top1']:.3f}, "
            f"gap12={overall['policy_gap12']:.3f}, "
            f"Htgt/logN={overall['target_entropy_norm']:.3f}, "
            f"tgt_top1={overall['target_top1']:.3f}, "
            f"nlegal={overall['nlegal']:.1f}"
        )
    else:
        print(
            "Overall: "
            f"|v|={overall['value_abs']:.3f}, "
            f"|v|<.1={overall['value_near_zero']:.3f}, "
            f"Hpol/logN={overall['policy_entropy_norm']:.3f}, "
            f"top1={overall['policy_top1']:.3f}, "
            f"gap12={overall['policy_gap12']:.3f}, "
            f"nlegal={overall['nlegal']:.1f}"
        )


if __name__ == "__main__":
    main()
