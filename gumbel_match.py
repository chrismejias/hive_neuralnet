"""
Head-to-head match between two checkpoints using Gumbel AlphaZero search.

Runs all games in parallel: each move step, active games are split by whose
turn it is and searched in two sub-batches (one per net), then moves are
scattered back into the shared state tensor.

Usage:
    python gumbel_match.py <checkpoint_a> <checkpoint_b> [--games 50] [--sims 512] [--gumbel-k 16]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import hive_gpu
from hive_gpu.gumbel_mcts import GumbelAlphaZeroOrchestrator, GumbelConfig
from hive_transformer.transformer_net import TransformerConfig, HiveTransformer


def load_net(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net_config = ckpt["net_config"]
    net = HiveTransformer(net_config)
    net.load_state_dict(ckpt["model_state_dict"])
    net = net.to(device)
    net.eval()
    iteration = ckpt.get("iteration", "?")
    arch = f"Transformer(d={net_config.d_model}, layers={net_config.num_layers})"
    params = sum(p.numel() for p in net.parameters())
    return net, iteration, arch, params


def _apply_sub_batch(
    orch: GumbelAlphaZeroOrchestrator,
    states: torch.Tensor,
    full_idxs: np.ndarray,
    move_numbers: list[int],
    ext,
) -> None:
    """Run Gumbel search on a sub-batch and apply moves back to `states` in-place."""
    B_sub = len(full_idxs)
    if B_sub == 0:
        return

    # Extract sub-states (advanced indexing returns a copy — safe to modify)
    idxs_t = torch.tensor(full_idxs, device="cuda", dtype=torch.long)
    sub_states = states[idxs_t].clone()
    sub_active = [True] * B_sub
    sub_move_nums = [move_numbers[i] for i in full_idxs]

    policies, _graphs, _seqs, _nn_priors, legal_moves, legal_action_indices, num_legal_moves = \
        orch._gumbel_search(sub_states, B_sub, sub_active, sub_move_nums)

    # Greedy action selection (temperature=0)
    selected = np.array([
        int(act_idx[int(np.argmax(probs))]) if len(act_idx) > 0 else -1
        for act_idx, probs in policies
    ], dtype=np.int64)

    actions_t = torch.from_numpy(selected).to("cuda").unsqueeze(1)  # [B_sub, 1]
    matched = orch._match_actions_to_legal_moves(
        actions_t, legal_action_indices, num_legal_moves,
    ).squeeze(1)  # [B_sub]

    safe_idx = matched.clamp(min=0).long()
    move_bytes = legal_moves[torch.arange(B_sub, device="cuda"), safe_idx]
    move_bytes[matched < 0] = 0  # no-op bytes for unmatched (shouldn't happen)

    ext.apply_moves_batch(sub_states, move_bytes, B_sub)

    # Scatter modified sub-states back into the full state tensor
    states[idxs_t] = sub_states


def run_match(
    ckpt_a: str,
    ckpt_b: str,
    num_games: int,
    num_sims: int,
    gumbel_k: int,
    max_moves: int = 300,
) -> None:
    device = torch.device("cuda")

    print("Loading checkpoints...")
    net_a, iter_a, arch_a, params_a = load_net(ckpt_a, device)
    net_b, iter_b, arch_b, params_b = load_net(ckpt_b, device)
    print(f"  A (latest): iter={iter_a}  {arch_a}  params={params_a:,}")
    print(f"  B (first):  iter={iter_b}  {arch_b}  params={params_b:,}")
    print(f"\nPlaying {num_games} games in parallel  sims={num_sims}  gumbel_k={gumbel_k}\n")

    cfg = GumbelConfig(
        num_simulations=num_sims,
        max_num_considered_actions=gumbel_k,
        temperature=0.0,
        temperature_drop_move=0,  # always greedy in a match
        batch_size=num_games,
        encoder_type="transformer",
        expansion_mask=0,
    )

    orch_a = GumbelAlphaZeroOrchestrator(net_a, cfg)
    orch_b = GumbelAlphaZeroOrchestrator(net_b, cfg)
    ext = orch_a.ext

    # games 0..half-1 → A is white; games half..num_games-1 → A is black
    half = num_games // 2
    a_is_white = np.array([i < half for i in range(num_games)])

    states = ext.create_initial_states(num_games, 0)
    active = [True] * num_games
    move_numbers = [0] * num_games
    game_start_times = [time.perf_counter()] * num_games

    t_total = time.perf_counter()

    while any(active):
        # Determine current player for each active game
        turns = np.array(orch_a._get_turns(states, num_games))
        white_to_move = (turns % 2 == 0)

        # net_a moves when A is white and it's white's turn, OR A is black and it's black's turn
        net_a_turn = np.array([
            active[i] and (a_is_white[i] == white_to_move[i])
            for i in range(num_games)
        ])
        net_b_turn = np.array([
            active[i] and not (a_is_white[i] == white_to_move[i])
            for i in range(num_games)
        ])

        a_idxs = np.where(net_a_turn)[0]
        b_idxs = np.where(net_b_turn)[0]

        _apply_sub_batch(orch_a, states, a_idxs, move_numbers, ext)
        _apply_sub_batch(orch_b, states, b_idxs, move_numbers, ext)

        for i in np.where(np.array(active))[0]:
            move_numbers[i] += 1

        # Check for finished games
        results = ext.check_results_batch(states, num_games).cpu().numpy()
        result_names = {0: "IN_PROGRESS", 1: "WHITE_WINS", 2: "BLACK_WINS", 3: "DRAW"}

        for i in range(num_games):
            if not active[i]:
                continue
            finished = results[i] != 0 or move_numbers[i] >= max_moves
            if finished:
                active[i] = False
                elapsed = time.perf_counter() - game_start_times[i]
                result = int(results[i])

                if result == 1:
                    a_score = 1.0 if a_is_white[i] else 0.0
                elif result == 2:
                    a_score = 1.0 if not a_is_white[i] else 0.0
                else:
                    a_score = 0.5  # draw or timeout

                a_color = "W" if a_is_white[i] else "B"
                outcome = {1.0: "A wins", 0.5: "draw", 0.0: "B wins"}[a_score]
                print(
                    f"  Game {i+1:2d}: A={a_color}  {move_numbers[i]:3d} moves  "
                    f"{result_names[result]:<15}  {outcome}  ({elapsed:.0f}s)",
                    flush=True,
                )

    total_elapsed = time.perf_counter() - t_total

    # Tally scores
    final_results = ext.check_results_batch(states, num_games).cpu().numpy()
    a_wins = 0.0
    for i in range(num_games):
        r = int(final_results[i])
        if r == 1:
            a_score = 1.0 if a_is_white[i] else 0.0
        elif r == 2:
            a_score = 1.0 if not a_is_white[i] else 0.0
        else:
            a_score = 0.5
        a_wins += a_score

    win_pct = 100 * a_wins / num_games
    print(f"\n{'='*52}")
    print(f"A (iter {iter_a}) score: {a_wins}/{num_games}  ({win_pct:.0f}%)")
    print(f"B (iter {iter_b}) score: {num_games - a_wins}/{num_games}  ({100 - win_pct:.0f}%)")
    print(f"Total wall time: {total_elapsed:.0f}s")
    print(f"{'='*52}")


def main():
    parser = argparse.ArgumentParser(description="Gumbel head-to-head match.")
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--sims", type=int, default=512)
    parser.add_argument("--gumbel-k", type=int, default=16)
    parser.add_argument("--max-moves", type=int, default=300)
    args = parser.parse_args()

    run_match(
        args.checkpoint_a,
        args.checkpoint_b,
        num_games=args.games,
        num_sims=args.sims,
        gumbel_k=args.gumbel_k,
        max_moves=args.max_moves,
    )


if __name__ == "__main__":
    main()
