from __future__ import annotations

import argparse
import time

import torch

from fnn_forced_mate_q_diagnostic import (
    apply_line,
    build_artificial_base,
    forced_black_win,
    gamestate_to_gpu_bytes,
    load_net,
    positions_to_tensor,
)
from hive_fnn.fnn_mcts_orchestrator import FNNMCTSConfig, FNNMCTSOrchestrator
from hive_gpu.endgame_generator import generate_endgame_positions


def find_surround5_threat_state(orch: FNNMCTSOrchestrator, sample_count: int) -> bytes | None:
    ext = orch.ext
    samples = generate_endgame_positions(
        sample_count,
        expansion_mask=7,
        min_surround=5,
        max_surround=5,
        gpu_batch=min(sample_count, 256),
    )
    states = positions_to_tensor(samples, device="cuda")
    legal_moves, num_legal = ext.generate_legal_moves_batch(states, len(samples))
    root_turn = (
        states[:, 3412].to(torch.int32)
        | (states[:, 3413].to(torch.int32) << 8)
    )
    root_player = (root_turn & 1).to(torch.int64)
    opp_color = ((root_player + 1) & 1).to(torch.int64)
    surround, _ = orch._queen_surround_count(states, opp_color)
    immediate = orch._find_immediate_wins(states, legal_moves, num_legal, len(samples))
    mask = (surround == 5) & (immediate < 0)
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return None
    return samples[int(idx[0, 0].item())]


def run_batch(
    net,
    state_bytes: bytes,
    *,
    simulations: int,
    batch_size: int,
    short_probe: bool,
) -> float:
    orch = FNNMCTSOrchestrator(
        net,
        FNNMCTSConfig(
            num_simulations=simulations,
            max_num_considered_actions=16,
            batch_size=batch_size,
            max_game_length=1,
            queen_surround_reserve_slots=10,
            queen_surround_reserve_immobile_only=True,
            short_forced_win_probe=short_probe,
        ),
    )
    states = positions_to_tensor([state_bytes] * batch_size, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    orch.self_play_batch(start_states=states.clone())
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_root_probe(
    net,
    state_bytes: bytes,
    *,
    batch_size: int,
    short_probe: bool,
) -> float:
    orch = FNNMCTSOrchestrator(
        net,
        FNNMCTSConfig(
            num_simulations=1,
            max_num_considered_actions=16,
            batch_size=batch_size,
            queen_surround_reserve_slots=10,
            queen_surround_reserve_immobile_only=True,
            short_forced_win_probe=short_probe,
        ),
    )
    states = positions_to_tensor([state_bytes] * batch_size, device="cuda")
    legal_moves, num_legal, root_features = orch.ext.generate_legal_moves_and_fnn_features_batch(states, batch_size)
    priors_per_legal, _, _ = orch._eval_states(states, legal_moves, num_legal, batch_size, root_features)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if short_probe:
        orch._find_short_forced_wins(states, legal_moves, num_legal, batch_size, priors_per_legal)
    else:
        orch._find_immediate_wins(states, legal_moves, num_legal, batch_size)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_case(
    name: str,
    net,
    state_bytes: bytes,
    simulations: int,
    batch_size: int,
    repeats: int,
    *,
    probe_only: bool,
) -> None:
    # Warmup both paths once.
    runner = run_root_probe if probe_only else run_batch
    runner(
        net,
        state_bytes,
        batch_size=batch_size,
        short_probe=False,
        **({} if probe_only else {"simulations": simulations}),
    )
    runner(
        net,
        state_bytes,
        batch_size=batch_size,
        short_probe=True,
        **({} if probe_only else {"simulations": simulations}),
    )

    baseline = []
    probe = []
    for _ in range(repeats):
        baseline.append(
            runner(
                net,
                state_bytes,
                batch_size=batch_size,
                short_probe=False,
                **({} if probe_only else {"simulations": simulations}),
            )
        )
        probe.append(
            runner(
                net,
                state_bytes,
                batch_size=batch_size,
                short_probe=True,
                **({} if probe_only else {"simulations": simulations}),
            )
        )

    base_mean = sum(baseline) / len(baseline)
    probe_mean = sum(probe) / len(probe)
    ratio = probe_mean / max(base_mean, 1e-9)
    print(
        f"{name}: baseline={base_mean:.3f}s probe={probe_mean:.3f}s "
        f"delta={probe_mean - base_mean:+.3f}s ratio={ratio:.3f}x"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--simulations", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threat-samples", type=int, default=512)
    ap.add_argument("--skip-threat", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    net = load_net(args.checkpoint)

    base = build_artificial_base()
    forced0 = forced_black_win(base, 4)
    surround4 = base.copy()
    surround4.turn += 1
    surround4._legal_moves_cache = None
    surround4_bytes = gamestate_to_gpu_bytes(surround4)

    mate_in_one = apply_line(surround4, forced_black_win(surround4, 3).line, 2)
    mate_in_one._legal_moves_cache = None
    mate_in_one_bytes = gamestate_to_gpu_bytes(mate_in_one)

    probe_orch = FNNMCTSOrchestrator(
        net,
        FNNMCTSConfig(
            num_simulations=args.simulations,
            max_num_considered_actions=16,
            batch_size=1,
        ),
    )
    threat_bytes = None if args.skip_threat else find_surround5_threat_state(probe_orch, args.threat_samples)

    print(f"checkpoint={args.checkpoint}")
    print(
        f"simulations={args.simulations} batch_size={args.batch_size} "
        f"repeats={args.repeats} probe_only={args.probe_only}"
    )
    bench_case(
        "surround4_shell", net, surround4_bytes, args.simulations, args.batch_size, args.repeats,
        probe_only=args.probe_only,
    )
    bench_case(
        "surround5_mate_in_one", net, mate_in_one_bytes, args.simulations, args.batch_size, args.repeats,
        probe_only=args.probe_only,
    )
    if threat_bytes is not None:
        bench_case(
            "surround5_threat", net, threat_bytes, args.simulations, args.batch_size, args.repeats,
            probe_only=args.probe_only,
        )
    else:
        print("surround5_threat: no sample found")


if __name__ == "__main__":
    main()
