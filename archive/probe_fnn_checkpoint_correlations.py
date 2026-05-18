"""Compare FNN checkpoint evaluations on shared midgame positions.

This probe:
  - samples a batch of midgame Hive positions by playing random legal moves
  - evaluates each position with several FNN checkpoints
  - compares every checkpoint against a reference checkpoint

By default it compares:
  - checkpoints_fnn_small/hive_fnn_checkpoint_0050.pt
  - checkpoints_fnn_medium/hive_fnn_checkpoint_0050.pt
  - checkpoints_fnn_large/hive_fnn_checkpoint_0050.pt
  against
  - checkpoints_fnn_large/hive_fnn_checkpoint_0100.pt

The comparison uses:
  - value head output per position
  - policy logits over all legal successor moves

Policy is reported two ways:
  - pooled Pearson correlation over all legal moves
  - mean per-position Pearson correlation over legal-move vectors

Usage:
  python3.11 probe_fnn_checkpoint_correlations.py
  python3.11 probe_fnn_checkpoint_correlations.py --samples 256 --min-ply 8 --max-ply 24
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import hive_gpu
from hive_fnn.fnn_network import HiveFNN


REPO_ROOT = os.path.dirname(__file__)


@dataclass
class FNNEvalBatch:
    value: torch.Tensor
    policy_logits: torch.Tensor
    legal_mask: torch.Tensor


def load_fnn(path: str) -> HiveFNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = HiveFNN(ckpt["net_config"]).cuda().eval()
    net.load_state_dict(ckpt["model_state_dict"])
    return net


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def sample_midgame_states(
    *,
    num_samples: int,
    min_ply: int,
    max_ply: int,
    expansion_mask: int,
    seed: int,
) -> torch.Tensor:
    """Sample random midgame states by playing random legal moves."""
    rng = np.random.default_rng(seed)
    ext = hive_gpu.load_extension()
    collected: list[torch.Tensor] = []
    batch_games = max(64, num_samples * 2)

    while len(collected) < num_samples:
        states = ext.create_initial_states(batch_games, expansion_mask)
        move_numbers = np.zeros(batch_games, dtype=np.int32)
        active = np.ones(batch_games, dtype=bool)
        targets = rng.integers(min_ply, max_ply + 1, size=batch_games, dtype=np.int32)
        done = np.zeros(batch_games, dtype=bool)

        while bool(active.any()):
            idx = np.flatnonzero(active)
            sub_states = states[idx].clone()
            legal_moves, num_legal = ext.generate_legal_moves_batch(sub_states, idx.size)
            legal_np = legal_moves.cpu().numpy()
            num_legal_np = num_legal.cpu().numpy()

            move_bytes = np.zeros((idx.size, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
            for j, n_i in enumerate(num_legal_np):
                if n_i <= 0:
                    active[idx[j]] = False
                    continue
                choice = int(rng.integers(int(n_i)))
                move_bytes[j] = legal_np[j, choice]

            move_t = torch.from_numpy(move_bytes).cuda()
            ext.apply_moves_batch(sub_states, move_t, idx.size)
            states[idx] = sub_states
            move_numbers[idx] += 1

            results = ext.check_results_batch(states, batch_games).cpu().numpy()
            active = active & (results == 0) & (move_numbers < max_ply + 2)

            collect_mask = (~done) & (move_numbers == targets) & (results == 0)
            for i in np.flatnonzero(collect_mask):
                collected.append(states[i].clone())
                done[i] = True
                if len(collected) >= num_samples:
                    break

            if len(collected) >= num_samples:
                break

        if len(collected) >= num_samples:
            break

    if not collected:
        raise RuntimeError("Failed to sample any midgame positions")
    return torch.stack(collected[:num_samples], dim=0)


def eval_fnn_batch(
    net: HiveFNN,
    states: torch.Tensor,
    ext,
) -> FNNEvalBatch:
    """Evaluate root values and per-legal successor logits for a batch."""
    B = int(states.shape[0])
    legal_moves, num_legal, root_features = ext.generate_legal_moves_and_fnn_features_batch(states, B)

    max_legal = int(legal_moves.shape[1])
    slot_idx = torch.arange(max_legal, device=states.device, dtype=torch.int64).unsqueeze(0)
    valid = slot_idx < num_legal.to(torch.int64).unsqueeze(1)

    if B == 0:
        return FNNEvalBatch(
            value=torch.zeros((0,), dtype=torch.float32, device=states.device),
            policy_logits=torch.zeros((0, 0), dtype=torch.float32, device=states.device),
            legal_mask=torch.zeros((0, 0), dtype=torch.bool, device=states.device),
        )

    action_to_root = torch.arange(B, device=states.device, dtype=torch.int64).unsqueeze(1).expand_as(valid)[valid]
    move_indices = slot_idx.expand_as(valid)[valid]
    total_actions = int(action_to_root.shape[0])

    with torch.inference_mode():
        root_emb = net.encode(root_features)
        root_values = net.value_head(root_emb).squeeze(-1).float()

        if total_actions == 0:
            policy_logits = torch.full(
                (B, max_legal), float("-inf"), dtype=torch.float32, device=states.device,
            )
            policy_logits = policy_logits.masked_fill(~valid, float("-inf"))
            return FNNEvalBatch(root_values, policy_logits, valid)

        succ_features = ext.fnn_successor_features_batch(
            states, legal_moves, action_to_root, move_indices, total_actions,
        )
        succ_emb = net.encode(succ_features)
        logits = net.score_actions(root_emb[action_to_root], succ_emb).float()

    policy_logits = torch.full(
        (B, max_legal), float("-inf"), dtype=torch.float32, device=states.device,
    )
    policy_logits[valid] = logits
    return FNNEvalBatch(root_values, policy_logits, valid)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.size != y.size:
        raise ValueError("pearson inputs must have matching sizes")
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    xm = x.mean()
    ym = y.mean()
    xc = x - xm
    yc = y - ym
    denom = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def mean_position_policy_corr(
    pred_logits: np.ndarray,
    ref_logits: np.ndarray,
    legal_mask: np.ndarray,
) -> tuple[float, int]:
    corrs: list[float] = []
    skipped = 0
    for i in range(pred_logits.shape[0]):
        mask = legal_mask[i]
        n = int(mask.sum())
        if n < 2:
            skipped += 1
            continue
        x = pred_logits[i, mask]
        y = ref_logits[i, mask]
        c = pearson(x, y)
        if np.isfinite(c):
            corrs.append(c)
        else:
            skipped += 1
    if not corrs:
        return float("nan"), skipped
    return float(np.mean(corrs)), skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare FNN checkpoint evaluations on shared midgame positions.",
    )
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--max-ply", type=int, default=24)
    ap.add_argument("--expansion-mask", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--small-checkpoint",
        type=str,
        default="checkpoints_fnn_small/hive_fnn_checkpoint_0050.pt",
    )
    ap.add_argument(
        "--medium-checkpoint",
        type=str,
        default="checkpoints_fnn_medium/hive_fnn_checkpoint_0050.pt",
    )
    ap.add_argument(
        "--large50-checkpoint",
        type=str,
        default="checkpoints_fnn_large/hive_fnn_checkpoint_0050.pt",
    )
    ap.add_argument(
        "--reference-checkpoint",
        type=str,
        default="checkpoints_fnn_large/hive_fnn_checkpoint_0100.pt",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ext = hive_gpu.load_extension()
    small_ckpt = resolve_path(args.small_checkpoint)
    medium_ckpt = resolve_path(args.medium_checkpoint)
    large50_ckpt = resolve_path(args.large50_checkpoint)
    reference_ckpt = resolve_path(args.reference_checkpoint)

    states = sample_midgame_states(
        num_samples=args.samples,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        expansion_mask=args.expansion_mask,
        seed=args.seed,
    )

    nets = {
        "small_50": load_fnn(small_ckpt),
        "medium_50": load_fnn(medium_ckpt),
        "large_50": load_fnn(large50_ckpt),
        "large_100": load_fnn(reference_ckpt),
    }

    batch = eval_fnn_batch(nets["large_100"], states, ext)
    ref_value = batch.value.cpu().numpy()
    ref_policy = batch.policy_logits.cpu().numpy()
    legal_mask = batch.legal_mask.cpu().numpy()
    ref_policy_flat = ref_policy[legal_mask]

    print("\n=== Midgame sampling ===")
    print(
        f"Samples: {states.shape[0]}  "
        f"ply range: [{args.min_ply}, {args.max_ply}]  "
        f"expansion_mask={args.expansion_mask}"
    )
    print("\n=== Reference ===")
    print(f"Reference checkpoint: {reference_ckpt}")

    rows = []
    for name, net in ("small_50", nets["small_50"]), ("medium_50", nets["medium_50"]), ("large_50", nets["large_50"]):
        out = eval_fnn_batch(net, states, ext)
        value = out.value.cpu().numpy()
        policy = out.policy_logits.cpu().numpy()
        value_corr = pearson(value, ref_value)
        policy_flat_corr = pearson(policy[legal_mask], ref_policy_flat)
        policy_pos_corr, skipped = mean_position_policy_corr(policy, ref_policy, legal_mask)
        rows.append((name, value_corr, policy_flat_corr, policy_pos_corr, skipped))

    print("\n=== Correlations vs large_100 ===")
    print("model       value_pearson  policy_flat_pearson  policy_mean_pos_pearson  skipped_pos")
    for name, vcorr, pcorr, ppos, skipped in rows:
        print(f"{name:<11} {vcorr:>13.4f}  {pcorr:>20.4f}  {ppos:>24.4f}  {skipped:>11d}")


if __name__ == "__main__":
    main()
