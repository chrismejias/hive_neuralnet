"""
Probe: find positions with a forced win-in-one and check network evaluation.

Runs across all 8 expansion subsets (0-7) and reports per-subset results.

For each position:
  - Value output  (should be close to +1 for the active player)
  - Probability assigned to the winning move
  - Rank of the winning move among all legal moves (1 = top)
"""

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import argparse
import glob

def _latest_checkpoint():
    paths = sorted(glob.glob("checkpoints_gpu/hive_gpu_checkpoint_*.pt"))
    return paths[-1] if paths else None

_parser = argparse.ArgumentParser()
_parser.add_argument("--checkpoint", default=None)
_args, _ = _parser.parse_known_args()
CHECKPOINT = _args.checkpoint or _latest_checkpoint()
NUM_WANTED  = 30        # positions per expansion subset
MAX_GAMES   = 10_000    # random games to try per subset
MAX_MOVES   = 120       # cap game length
SEED        = 42

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from hive_engine.game_state import GameState, GameResult, MoveType
from hive_engine.pieces import ExpansionConfig
from hive_engine.encoder import HiveEncoder
from hive_transformer.transformer_encoder import TransformerEncoder
from hive_transformer.transformer_net import HiveTransformer

# ---------------------------------------------------------------------------
# Load network
# ---------------------------------------------------------------------------
print(f"Loading checkpoint: {CHECKPOINT}")
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
net_config = checkpoint["net_config"]
net = HiveTransformer(net_config)
net.load_state_dict(checkpoint["model_state_dict"])
net.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
net = net.to(device)
print(f"  Iteration : {checkpoint.get('iteration')}")
print(f"  Device    : {device}")
print(f"  Params    : {sum(p.numel() for p in net.parameters()):,}")

tx_encoder = TransformerEncoder()
enc        = HiveEncoder()

# ---------------------------------------------------------------------------
# Expansion subset labels
# ---------------------------------------------------------------------------
def mask_label(mask: int) -> str:
    parts = []
    if mask & 1: parts.append("M")
    if mask & 2: parts.append("L")
    if mask & 4: parts.append("P")
    return "base+" + "+".join(parts) if parts else "base"

def mask_to_cfg(mask: int) -> ExpansionConfig:
    return ExpansionConfig(
        mosquito=bool(mask & 1),
        ladybug =bool(mask & 2),
        pillbug =bool(mask & 4),
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_game_until_win1(exp_cfg, max_moves=MAX_MOVES):
    """Play a random game, yielding (state, winning_moves, all_moves) at each
    step where a win-in-1 exists."""
    state = GameState(expansions=exp_cfg)
    for _ in range(max_moves):
        if state.result != GameResult.IN_PROGRESS:
            return
        moves = list(state.legal_moves())
        if not moves:
            return

        mover = state.current_player
        win_moves = []
        for m in moves:
            state.apply_move(m)
            won = (mover.name == "WHITE" and state.result == GameResult.WHITE_WINS) or \
                  (mover.name == "BLACK" and state.result == GameResult.BLACK_WINS)
            state.undo_move()
            if won:
                win_moves.append(m)

        if win_moves:
            yield state, win_moves, moves

        state.apply_move(random.choice(moves))


def evaluate_position(state, winning_move, all_moves):
    """Run the network; return (value, win_prob, win_rank, n_legal) or None if
    the position is outside the Python encoder's grid."""
    try:
        legal_mask = enc.get_legal_action_mask(state)
        seq        = tx_encoder.encode_state(state)
    except (ValueError, KeyError):
        return None

    with torch.no_grad():
        probs, value = net.predict(seq, legal_mask)

    try:
        win_idx = enc.encode_action(winning_move, state)
    except (ValueError, KeyError):
        return None

    win_prob = float(probs[win_idx]) if win_idx >= 0 else 0.0
    legal_probs = probs[legal_mask > 0]
    win_rank    = int((legal_probs > win_prob).sum()) + 1

    return float(value), win_prob, win_rank, len(all_moves)


# ---------------------------------------------------------------------------
# Collect and evaluate per expansion subset
# ---------------------------------------------------------------------------
all_values    = []
all_win_probs = []
all_win_ranks = []

print(f"\n{'Mask':<10} {'Label':<12} {'n':>4}  {'MeanVal':>8}  "
      f"{'MeanWinP':>9}  {'#1':>5}  {'Top3':>5}  {'MeanRank':>9}")
print("-" * 75)

for mask in range(8):
    exp_cfg = mask_to_cfg(mask)
    label   = mask_label(mask)

    positions = []
    games_tried = 0

    while len(positions) < NUM_WANTED and games_tried < MAX_GAMES:
        games_tried += 1
        for state, win_moves, all_moves in random_game_until_win1(exp_cfg):
            if len(positions) >= NUM_WANTED:
                break
            positions.append((state.copy(), win_moves[0], all_moves[:]))
            break   # one position per game

    if not positions:
        print(f"{mask:<10} {label:<12} {'0':>4}  -- no positions found --")
        continue

    values, win_probs, win_ranks = [], [], []
    skipped = 0
    for state, win_move, all_moves in positions:
        result = evaluate_position(state, win_move, all_moves)
        if result is None:
            skipped += 1
            continue
        v, wp, wr, _ = result
        values.append(v)
        win_probs.append(wp)
        win_ranks.append(wr)

    n = len(values)
    if n == 0:
        print(f"{mask:<10} {label:<12} {'0':>4}  -- all skipped (out of encoder grid) --")
        continue
    mv  = np.mean(values)
    mwp = np.mean(win_probs)
    n1  = sum(r == 1 for r in win_ranks)
    n3  = sum(r <= 3 for r in win_ranks)
    mr  = np.mean(win_ranks)
    skip_str = f" ({skipped} skipped)" if skipped else ""

    print(f"{mask:<10} {label:<12} {n:>4}  {mv:>+8.3f}  {mwp:>9.4f}  "
          f"{n1:>3}/{n:<3}  {n3:>3}/{n:<3}  {mr:>9.1f}{skip_str}")

    all_values.extend(values)
    all_win_probs.extend(win_probs)
    all_win_ranks.extend(win_ranks)

# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------
print("-" * 75)
n = len(all_values)
if n > 0:
    print(f"\n=== Aggregate across all 8 subsets ({n} positions) ===")
    print(f"  Mean value output : {np.mean(all_values):+.4f}  (ideal: +1.00)")
    print(f"  Mean win-move prob: {np.mean(all_win_probs):.4f}  (ideal: 1.00)")
    print(f"  Win move is #1    : {sum(r==1 for r in all_win_ranks)}/{n}")
    print(f"  Win move top-3    : {sum(r<=3 for r in all_win_ranks)}/{n}")
    print(f"  Win move top-10   : {sum(r<=10 for r in all_win_ranks)}/{n}")
    print(f"  Mean rank of win  : {np.mean(all_win_ranks):.1f}  (ideal: 1.0)")
