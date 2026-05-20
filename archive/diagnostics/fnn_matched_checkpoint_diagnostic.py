"""Compare FNN checkpoints on the same sampled positions.

This is intended for regression triage: generate one fixed set of positions,
then evaluate multiple checkpoints on exactly those states and legal moves.
It reports value calibration against the sampled game outcomes, policy
sharpness, target cross-entropy, and top-move agreement.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, "/workspace/hive_neuralnet")

import hive_gpu
from hive_fnn.fnn_mcts_orchestrator import FNNMCTSConfig, FNNMCTSOrchestrator
from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_fnn.fnn_replay_buffer import FNNTrainingExample


@dataclass
class EvalResult:
    name: str
    values: np.ndarray
    policies: np.ndarray
    top1: np.ndarray
    entropy: np.ndarray
    top1_mass: np.ndarray
    top3_mass: np.ndarray
    ce_to_target: np.ndarray


def load_fnn(path: str) -> HiveFNN:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_raw = ckpt.get("net_config")
    cfg = cfg_raw if isinstance(cfg_raw, FNNConfig) else FNNConfig.large()
    net = HiveFNN(cfg)
    net.load_state_dict(ckpt["model_state_dict"])
    return net.cuda().eval()


def collect_examples(
    checkpoint: str,
    *,
    games: int,
    sims: int,
    max_examples: int,
    seed: int,
) -> list[FNNTrainingExample]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = load_fnn(checkpoint)
    cfg = FNNMCTSConfig(
        num_simulations=sims,
        max_num_considered_actions=16,
        batch_size=games,
        expansion_mask=7,
        wave_parallel=True,
        wave_size=4,
        max_game_length=300,
        short_forced_win_probe=True,
        probe_win_in_one=True,
        probe_check_opponent_wins=True,
        probe_win_in_two=False,
    )
    orch = FNNMCTSOrchestrator(net, cfg)
    by_game = orch.self_play_batch()
    flat = [ex for game in by_game for ex in game if ex.use_for_value]
    if len(flat) > max_examples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flat), size=max_examples, replace=False)
        flat = [flat[int(i)] for i in idx]
    return flat


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def evaluate_checkpoint(
    name: str,
    checkpoint: str,
    states: torch.Tensor,
    legal_moves: torch.Tensor,
    num_legal: torch.Tensor,
    root_features: torch.Tensor,
    targets: np.ndarray,
) -> EvalResult:
    ext = hive_gpu.load_extension()
    net = load_fnn(checkpoint)
    B = int(states.shape[0])
    max_legal = int(ext.MAX_LEGAL_MOVES)
    slot_idx = torch.arange(max_legal, device="cuda", dtype=torch.int64).unsqueeze(0)
    valid = slot_idx < num_legal.to(torch.int64).unsqueeze(1)
    action_to_root = torch.arange(B, device="cuda", dtype=torch.int64).unsqueeze(1).expand_as(valid)[valid]
    move_indices = slot_idx.expand_as(valid)[valid]
    total_actions = int(action_to_root.shape[0])

    with torch.inference_mode():
        succ_features = ext.fnn_successor_features_batch(
            states, legal_moves, action_to_root, move_indices, total_actions,
        )
        combined = torch.cat([root_features, succ_features], dim=0)
        emb = net.encode(combined)
        root_emb = emb[:B]
        succ_emb = emb[B:]
        values = net.value_head(root_emb).squeeze(-1).float()
        logits_flat = net.score_actions(root_emb[action_to_root], succ_emb).float()

        logits = torch.full((B, max_legal), -1e30, dtype=torch.float32, device="cuda")
        logits[valid] = logits_flat
        policies = torch.softmax(logits, dim=1).masked_fill(~valid, 0.0)

    pol_np = policies.cpu().numpy()
    val_np = values.cpu().numpy()
    top1 = pol_np.argmax(axis=1)
    entropy = _entropy(pol_np)
    top1_mass = pol_np.max(axis=1)
    top3_mass = np.sort(pol_np, axis=1)[:, -3:].sum(axis=1)
    ce = -(targets * np.log(np.clip(pol_np, 1e-12, 1.0))).sum(axis=1)
    return EvalResult(
        name=name,
        values=val_np,
        policies=pol_np,
        top1=top1,
        entropy=entropy,
        top1_mass=top1_mass,
        top3_mass=top3_mass,
        ce_to_target=ce,
    )


def summarize_result(result: EvalResult, value_targets: np.ndarray, policy_mask: np.ndarray) -> None:
    mask = policy_mask.astype(bool)
    mse = float(np.mean((result.values - value_targets) ** 2))
    mae = float(np.mean(np.abs(result.values - value_targets)))
    sign_mask = value_targets != 0
    sign_acc = float(np.mean(np.sign(result.values[sign_mask]) == np.sign(value_targets[sign_mask]))) if sign_mask.any() else float("nan")
    std = float(np.std(result.values))
    mean = float(np.mean(result.values))
    ce = float(np.mean(result.ce_to_target[mask])) if mask.any() else float("nan")
    print(f"\n[{result.name}]")
    print(f"  value mean/std:       {mean:+.4f} / {std:.4f}")
    print(f"  value MSE/MAE:        {mse:.4f} / {mae:.4f}")
    print(f"  value sign accuracy:  {sign_acc:.3f}")
    print(f"  policy entropy:       {float(np.mean(result.entropy[mask])):.3f} nats")
    print(f"  policy effective N:   {float(np.mean(np.exp(result.entropy[mask]))):.2f}")
    print(f"  policy top1 mass:     {float(np.mean(result.top1_mass[mask])):.3f}")
    print(f"  policy top3 mass:     {float(np.mean(result.top3_mass[mask])):.3f}")
    print(f"  CE to search target:  {ce:.3f}")


def pairwise(results: list[EvalResult], policy_mask: np.ndarray) -> None:
    mask = policy_mask.astype(bool)
    print("\nPairwise top-1 agreement:")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            agree = float(np.mean(results[i].top1[mask] == results[j].top1[mask]))
            print(f"  {results[i].name} vs {results[j].name}: {agree:.3f}")

    print("\nPairwise value correlation:")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i].values
            b = results[j].values
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else float("nan")
            print(f"  {results[i].name} vs {results[j].name}: {corr:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare FNN checkpoints on matched positions")
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--sims", type=int, default=512)
    p.add_argument("--max-examples", type=int, default=1024)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    examples = collect_examples(
        args.source_checkpoint,
        games=args.games,
        sims=args.sims,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("No examples collected")

    ext = hive_gpu.load_extension()
    states_np = np.stack([ex.state_bytes for ex in examples], axis=0).astype(np.uint8, copy=False)
    states = torch.from_numpy(states_np).cuda()
    B = int(states.shape[0])
    legal_moves, num_legal, root_features = ext.generate_legal_moves_and_fnn_features_batch(states, B)

    max_legal = int(ext.MAX_LEGAL_MOVES)
    targets = np.zeros((B, max_legal), dtype=np.float32)
    policy_mask = np.zeros((B,), dtype=np.float32)
    value_targets = np.asarray([ex.value_target for ex in examples], dtype=np.float32)
    for i, ex in enumerate(examples):
        n = min(len(ex.policy_target), max_legal)
        targets[i, :n] = ex.policy_target[:n]
        policy_mask[i] = float(ex.use_for_policy)

    target_entropy = _entropy(targets)
    target_top1 = targets.max(axis=1)
    target_nonzero = (targets > 0).sum(axis=1)
    mask = policy_mask.astype(bool)
    avg_legal = float(num_legal.float().mean().item())
    print(f"Collected positions: {B}")
    print(f"Average legal moves: {avg_legal:.2f}")
    print(f"Value target mean/std: {float(value_targets.mean()):+.4f} / {float(value_targets.std()):.4f}")
    print(f"Policy-mask fraction: {float(policy_mask.mean()):.3f}")
    print(f"Search target entropy: {float(target_entropy[mask].mean()):.3f} nats")
    print(f"Search target effective N: {float(np.exp(target_entropy[mask]).mean()):.2f}")
    print(f"Search target top1 mass: {float(target_top1[mask].mean()):.3f}")
    print(f"Search target nonzero moves: {float(target_nonzero[mask].mean()):.2f}")

    results: list[EvalResult] = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH checkpoint spec, got {spec!r}")
        name, path = spec.split("=", 1)
        result = evaluate_checkpoint(
            name, path, states, legal_moves, num_legal, root_features, targets,
        )
        summarize_result(result, value_targets, policy_mask)
        results.append(result)

    pairwise(results, policy_mask)


if __name__ == "__main__":
    main()
