"""
Diagnostic: does the value head correctly evaluate lopsided queen-pressure positions?

Each test position has:
  - Black queen pinned: surrounded by 3-4 pieces (White pieces closing in)
  - White queen safe: touching only 1 piece

Expected: value > 0 (White winning) from White's perspective.

Usage:
    python eval_value_head.py [checkpoint1] [checkpoint2] ...

Defaults to checkpoints_gpu_transformer/hive_gpu_checkpoint_0050.pt
         and checkpoints_gpu/hive_gpu_checkpoint_0157.pt
"""

from __future__ import annotations
import sys
import torch
import numpy as np

from hive_engine.game_state import GameState
from hive_engine.board import Board
from hive_engine.pieces import Piece, PieceType, Color, ExpansionConfig
from hive_engine.hex_coord import HexCoord
from hive_transformer.transformer_encoder import TransformerEncoder
from hive_transformer.transformer_net import TransformerConfig, HiveTransformer


# ── Helpers ──────────────────────────────────────────────────────────

W, B = Color.WHITE, Color.BLACK
Q  = PieceType.QUEEN
A  = PieceType.ANT
G  = PieceType.GRASSHOPPER
S  = PieceType.SPIDER
Be = PieceType.BEETLE

def piece(pt, color, n=0):
    return Piece(pt, color, n)

def coord(q, r):
    return HexCoord(q, r)


def build_state(placements: list[tuple[Piece, HexCoord]]) -> GameState:
    """
    Build a GameState by directly placing pieces onto the board,
    bypassing the legal-move system. Pieces are removed from each
    player's hand to keep hand counts consistent.
    """
    gs = GameState()
    # We need an even turn count so it's White's turn; set turn manually after.
    for pc, pos in placements:
        # Remove from hand
        hand = gs._hands[pc.color]
        # Find matching piece in hand
        match = next((p for p in hand if p.piece_type == pc.piece_type and p.piece_id == pc.piece_id), None)
        if match is None:
            raise ValueError(f"Piece {pc} not found in hand (already placed or wrong id?)")
        hand.remove(match)
        gs.board.place_piece(match, pos)
        # Track queen placement flag
        if pc.piece_type == PieceType.QUEEN:
            gs._queen_placed[pc.color] = True

    # Set turn so it's White's move (even turn)
    placed = len(placements)
    gs.turn = placed if placed % 2 == 0 else placed + 1
    # Invalidate caches
    gs._legal_moves_cache = None
    return gs


def evaluate(gs: GameState, net: HiveTransformer, encoder: TransformerEncoder) -> float:
    """Return value from current player's perspective (+1=winning, -1=losing)."""
    seq = encoder.encode_state(gs)
    mask = encoder.get_legal_action_mask(gs)
    with torch.no_grad():
        _, val = net.predict(seq, mask)
    return float(val)


def load_net(path: str) -> HiveTransformer:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["net_config"]
    net = HiveTransformer(cfg)
    sd_key = "model_state_dict" if "model_state_dict" in ckpt else "net_state_dict"
    net.load_state_dict(ckpt[sd_key], strict=False)
    net.eval()
    print(f"  Loaded: {path}  (ELO={ckpt.get('elo_rating', 0):.0f})")
    return net


# ── Test positions ────────────────────────────────────────────────────
#
# Flat-top hex grid.  HexCoord(q, r) neighbors are at offsets:
#   E=(+1,0)  W=(-1,0)  NE=(+1,-1)  NW=(0,-1)  SE=(0,+1)  SW=(-1,+1)
#
# Convention: it's WHITE's turn in every position.
#   Positive value → White is winning (correct prediction).
#   Negative value → model thinks White is losing (wrong).

POSITIONS = []

# ── Position 1: Black queen surrounded on 3 sides, White queen free ──
# Black queen at (0,0).  White ants at E, NE, NW → 3 neighbours.
# White queen at (4,0), touching only one White ant at (3,0).
# White has 1 ant left to place; it's White's turn.
def pos1():
    return "Black queen surrounded 3 sides; White queen has 1 contact", build_state([
        # Centre of battle: Black queen surrounded by White ants
        (piece(Q,  B),    coord(0,  0)),
        (piece(A,  W, 0), coord(1,  0)),   # E  of bQ
        (piece(A,  W, 1), coord(1, -1)),   # NE of bQ
        (piece(A,  W, 2), coord(0, -1)),   # NW of bQ
        # White queen far away, connected via one ant
        (piece(A,  B, 0), coord(-1, 0)),   # W of bQ (bridges hive)
        (piece(Q,  W),    coord(4,  0)),   # White queen, safe
        (piece(A,  B, 1), coord(3,  0)),   # only neighbour of wQ
    ])

POSITIONS.append(pos1)

# ── Position 2: Black queen surrounded on 4 sides ────────────────────
# Black queen at (0,0) with 4 White pieces (ants+grasshopper) surrounding it.
# White queen at (5,0) touching one piece.
def pos2():
    return "Black queen surrounded 4 sides; White queen has 1 contact", build_state([
        (piece(Q,  B),    coord(0,  0)),
        (piece(A,  W, 0), coord(1,  0)),   # E
        (piece(A,  W, 1), coord(1, -1)),   # NE
        (piece(A,  W, 2), coord(0, -1)),   # NW
        (piece(G,  W, 0), coord(-1, 1)),   # SW
        # Black pieces connecting hive
        (piece(A,  B, 0), coord(-1, 0)),   # W of bQ
        (piece(A,  B, 1), coord(0,  1)),   # SE of bQ (also connects)
        # White queen safe
        (piece(Q,  W),    coord(5,  0)),
        (piece(A,  B, 2), coord(4,  0)),   # only neighbour of wQ
    ])

POSITIONS.append(pos2)

# ── Position 3: Black queen surrounded 3 sides with Spider+Ants ──────
# Different pieces to test piece-type generalization.
def pos3():
    return "Black queen 3-surrounded (spider+ants); White queen safe", build_state([
        (piece(Q,  B),    coord(0,  0)),
        (piece(S,  W, 0), coord(1,  0)),   # E
        (piece(A,  W, 0), coord(0, -1)),   # NW
        (piece(A,  W, 1), coord(1, -1)),   # NE
        # Bridge pieces (hive connectivity)
        (piece(A,  B, 0), coord(-1, 0)),
        (piece(S,  B, 0), coord(0,  1)),
        # White queen
        (piece(Q,  W),    coord(4, -1)),
        (piece(A,  B, 1), coord(3, -1)),
    ])

POSITIONS.append(pos3)

# ── Position 4: Black queen surrounded 4 sides, one move from losing ─
# Beetles on top add to effective surround count (beetle covers adjacency).
def pos4():
    return "Black queen 4-surrounded including beetle stack; White queen safe", build_state([
        (piece(Q,  B),    coord(0,  0)),
        (piece(A,  W, 0), coord(1,  0)),   # E
        (piece(A,  W, 1), coord(0, -1)),   # NW
        (piece(A,  W, 2), coord(1, -1)),   # NE
        (piece(Be, W, 0), coord(-1, 1)),   # SW (beetle on ground)
        # Hive bridge
        (piece(A,  B, 0), coord(-1, 0)),
        (piece(G,  B, 0), coord(0,  1)),
        # White queen
        (piece(Q,  W),    coord(4,  0)),
        (piece(G,  W, 0), coord(3,  0)),
    ])

POSITIONS.append(pos4)

# ── Position 5: Symmetric but with positions swapped (White pinned) ──
# Now White queen is surrounded 3 sides → value should be NEGATIVE (Black winning).
# Model should output a value < 0 from White's perspective.
def pos5():
    return "REVERSED: White queen 3-surrounded; Black queen safe (expect value < 0)", build_state([
        (piece(Q,  W),    coord(0,  0)),
        (piece(A,  B, 0), coord(1,  0)),
        (piece(A,  B, 1), coord(1, -1)),
        (piece(A,  B, 2), coord(0, -1)),
        # Hive bridge
        (piece(A,  W, 0), coord(-1, 0)),
        (piece(G,  W, 0), coord(0,  1)),
        # Black queen safe
        (piece(Q,  B),    coord(4,  0)),
        (piece(A,  W, 1), coord(3,  0)),
    ])

POSITIONS.append(pos5)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    checkpoints = sys.argv[1:] if len(sys.argv) > 1 else [
        "checkpoints_gpu_transformer/hive_gpu_checkpoint_0050.pt",
        "checkpoints_gpu/hive_gpu_checkpoint_0157.pt",
    ]

    encoder = TransformerEncoder()

    nets = {}
    print("\n=== Loading checkpoints ===")
    for path in checkpoints:
        label = path.split("/")[-1].replace("hive_gpu_checkpoint_", "iter").replace(".pt", "")
        nets[label] = load_net(path)

    print("\n=== Value head evaluation ===")
    print(f"  Value is from White's (current player's) perspective.")
    print(f"  Positions 1-4: White should be WINNING  -> expect value > 0")
    print(f"  Position  5  : White is LOSING         -> expect value < 0")
    print()

    header = f"{'Position':<55}" + "".join(f"{k:>12}" for k in nets)
    print(header)
    print("-" * len(header))

    for i, pos_fn in enumerate(POSITIONS, 1):
        label, gs = pos_fn()
        vals = {}
        for name, net in nets.items():
            try:
                v = evaluate(gs, net, encoder)
                vals[name] = v
            except Exception as e:
                vals[name] = f"ERR:{e}"

        row = f"  {i}. {label:<51}"
        for name in nets:
            v = vals[name]
            if isinstance(v, float):
                marker = "OK" if (i < 5 and v > 0) or (i == 5 and v < 0) else "XX"
                row += f"  {v:+.3f} {marker}  "
            else:
                row += f"  {v:>10}  "
        print(row)

    print()


if __name__ == "__main__":
    main()
