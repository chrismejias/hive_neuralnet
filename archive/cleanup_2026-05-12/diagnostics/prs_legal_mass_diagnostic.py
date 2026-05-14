from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = "/workspace/hive_neuralnet"
sys.path.insert(0, REPO_ROOT)

import hive_gpu
from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
from hive_prs.prs_transformer_v3 import HivePRSTransformerV3
from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2


def load_net(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_version = ckpt.get("model_version", "v2")
    net_cls = HivePRSTransformerV3 if model_version == "v3" else HivePRSTransformerV2
    net = net_cls(ckpt["net_config"]).cuda().eval()
    current = net.state_dict()
    filtered = {}
    for key, tensor in ckpt["model_state"].items():
        if key not in current:
            continue
        if current[key].shape != tensor.shape:
            continue
        filtered[key] = tensor
    net.load_state_dict(filtered, strict=False)
    return net


def sample_midgame_states(
    *,
    num_samples: int,
    min_ply: int,
    max_ply: int,
    expansion_mask: int,
    seed: int,
) -> torch.Tensor:
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
                move_bytes[j] = legal_np[j, int(rng.integers(int(n_i)))]

            ext.apply_moves_batch(sub_states, torch.from_numpy(move_bytes).cuda(), idx.size)
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

    return torch.stack(collected[:num_samples], dim=0)


def summarize(vals: np.ndarray) -> str:
    if vals.size == 0:
        return "n=0"
    return (
        f"n={vals.size}  mean={vals.mean()*100:.1f}%  median={np.median(vals)*100:.1f}%  "
        f"p10={np.percentile(vals, 10)*100:.1f}%  p90={np.percentile(vals, 90)*100:.1f}%"
    )


def run_mass_probe(
    net: HivePRSTransformerV2,
    states: torch.Tensor,
) -> dict[str, np.ndarray]:
    B = int(states.shape[0])
    orch = PRSMCTSOrchestratorV2(
        net,
        PRSMCTSConfigV2(num_simulations=16, batch_size=B, expansion_mask=0),
    )

    prs_batch = orch.encoder.encode_batch(states, B)
    legal_t, nlegal_t = orch.ext.generate_legal_moves_batch(states, B)
    kernel_out = orch._classify_kernel(states, legal_t, nlegal_t, B)
    slot_of_legal_t = kernel_out[8].to(torch.long)
    logits_813, _ = orch._net_forward(prs_batch, kernel_out, B)
    probs_813 = torch.softmax(logits_813.float(), dim=1)

    legal_mask = torch.zeros(B, probs_813.shape[1], dtype=torch.bool, device="cuda")
    valid_legal = slot_of_legal_t >= 0
    legal_mask.scatter_(1, slot_of_legal_t.clamp(min=0), valid_legal)
    legal_mass = (probs_813 * legal_mask.float()).sum(dim=1).detach().cpu().numpy()

    # Slot ranges for expansion-movement diagnostics.
    throw_mask = torch.zeros_like(legal_mask)
    throw_mask[:, 48:108] = True
    pillbug_move_mask = torch.zeros_like(legal_mask)
    pillbug_move_mask[:, 36:42] = True
    pillbug_move_mask[:, 48:78] = True
    mosquito_move_mask = torch.zeros_like(legal_mask)
    mosquito_move_mask[:, 42:48] = True
    mosquito_move_mask[:, 78:108] = True
    mosquito_move_mask[:, 492:556] = True
    expansion_move_mask = torch.zeros_like(legal_mask)
    expansion_move_mask[:, 36:42] = True      # pillbug dir
    expansion_move_mask[:, 42:48] = True      # mosquito dir
    expansion_move_mask[:, 48:108] = True     # throws
    expansion_move_mask[:, 428:492] = True    # ladybug long
    expansion_move_mask[:, 492:556] = True    # mosquito long

    throw_legal = (legal_mask & throw_mask).any(dim=1).cpu().numpy()
    pillbug_legal = (legal_mask & pillbug_move_mask).any(dim=1).cpu().numpy()
    mosquito_legal = (legal_mask & mosquito_move_mask).any(dim=1).cpu().numpy()
    expansion_legal = (legal_mask & expansion_move_mask).any(dim=1).cpu().numpy()

    return {
        "legal_mass": legal_mass,
        "throw_legal_mass": legal_mass[throw_legal],
        "pillbug_legal_mass": legal_mass[pillbug_legal],
        "mosquito_legal_mass": legal_mass[mosquito_legal],
        "expansion_legal_mass": legal_mass[expansion_legal],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--min-ply", type=int, default=20)
    ap.add_argument("--max-ply", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = load_net(args.checkpoint)

    for label, exp_mask in (("base", 0), ("full", 7)):
        states = sample_midgame_states(
            num_samples=args.samples,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            expansion_mask=exp_mask,
            seed=args.seed + exp_mask,
        )
        res = run_mass_probe(net, states)
        print(f"\n=== {label} game (expansion_mask={exp_mask}) ===")
        print(f"overall legal-slot mass:     {summarize(res['legal_mass'])}")
        print(f"positions with throw moves:  {summarize(res['throw_legal_mass'])}")
        print(f"positions with pillbug move: {summarize(res['pillbug_legal_mass'])}")
        print(f"positions with mosquito move:{summarize(res['mosquito_legal_mass'])}")
        print(f"positions with any expansion move: {summarize(res['expansion_legal_mass'])}")


if __name__ == "__main__":
    main()
