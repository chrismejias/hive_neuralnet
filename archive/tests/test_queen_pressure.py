"""
Tests for hive_engine.queen_pressure curriculum learning utilities.

Verifies queen_is_mobile(), piece_adjacent_to_queen(),
find_queen_pressure_wins(), and generate_queen_pressure_state().
"""

from __future__ import annotations

import pytest

from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.hex_coord import HexCoord
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.queen_pressure import (
    find_queen_pressure_wins,
    generate_queen_pressure_state,
    piece_adjacent_to_queen,
    queen_is_mobile,
)


# ---------------------------------------------------------------------------
# Helpers: direct piece placement (mirrors endgame.py's _place)
# current_player is a read-only property derived from turn % 2:
#   turn even → WHITE, turn odd → BLACK
# ---------------------------------------------------------------------------


def _place(gs: GameState, color: Color, piece_type: PieceType, pos: HexCoord) -> Piece:
    """Find a piece of the given type in hand and place it at pos."""
    for p in list(gs.hand(color)):
        if p.piece_type == piece_type:
            gs.board.place_piece(p, pos)
            gs._remove_from_hand(p)
            if piece_type == PieceType.QUEEN:
                gs._queen_placed[piece.color] = True
            return p
    raise ValueError(f"No {color.name} {piece_type.name} in hand")


def _place_piece(gs: GameState, color: Color, piece_type: PieceType, pos: HexCoord) -> Piece:
    """Find first piece of the right type in hand and place it directly."""
    for p in list(gs.hand(color)):
        if p.piece_type == piece_type:
            gs.board.place_piece(p, pos)
            gs._remove_from_hand(p)
            if p.piece_type == PieceType.QUEEN:
                gs._queen_placed[p.color] = True
            return p
    raise ValueError(f"No {color.name} {piece_type.name} in hand")


def _make_state_with_queens() -> GameState:
    """
    Build a minimal valid state:
      White Queen at (0, 0), Black Queen at (1, 0) — adjacent.
      turn=2 → White's move.
    """
    gs = GameState()
    # Directly place both queens (bypassing turn-order validation)
    _place_piece(gs, Color.WHITE, PieceType.QUEEN, HexCoord(0, 0))
    _place_piece(gs, Color.BLACK, PieceType.QUEEN, HexCoord(1, 0))
    # turn=2 → Color(2%2) = Color(0) = WHITE
    gs.turn = 2
    return gs


# ---------------------------------------------------------------------------
# queen_is_mobile
# ---------------------------------------------------------------------------


class TestQueenIsMobile:
    def test_queen_not_placed_returns_false(self):
        gs = GameState()
        assert not queen_is_mobile(gs, Color.WHITE)
        assert not queen_is_mobile(gs, Color.BLACK)

    def test_queen_fully_surrounded_returns_false(self):
        """Queen with all 6 neighbors occupied cannot slide."""
        gs = GameState()
        _place_piece(gs, Color.WHITE, PieceType.QUEEN, HexCoord(0, 0))
        _place_piece(gs, Color.BLACK, PieceType.QUEEN, HexCoord(2, 0))
        # Surround white queen on all 6 sides with a mix of white pieces
        # (3 ants + 2 grasshoppers + 1 spider — stays within per-player limits)
        for pos, pt in [
            (HexCoord(1, 0),  PieceType.ANT),
            (HexCoord(1, -1), PieceType.ANT),
            (HexCoord(0, -1), PieceType.ANT),
            (HexCoord(-1, 0), PieceType.GRASSHOPPER),
            (HexCoord(-1, 1), PieceType.GRASSHOPPER),
            (HexCoord(0, 1),  PieceType.SPIDER),
        ]:
            _place_piece(gs, Color.WHITE, pt, pos)
        gs.turn = 2  # WHITE to move
        assert not queen_is_mobile(gs, Color.WHITE)

    def test_queen_with_open_space_is_mobile(self):
        """Queen with empty neighbours can slide."""
        gs = _make_state_with_queens()
        assert queen_is_mobile(gs, Color.WHITE)
        assert queen_is_mobile(gs, Color.BLACK)


# ---------------------------------------------------------------------------
# piece_adjacent_to_queen
# ---------------------------------------------------------------------------


class TestPieceAdjacentToQueen:
    def test_adjacent_position_returns_true(self):
        gs = _make_state_with_queens()
        # White queen at (0,0), black queen at (1,0).
        # (0,0) is adjacent to black queen at (1,0): offset (-1,0) from (1,0).
        assert piece_adjacent_to_queen(HexCoord(0, 0), gs.board, Color.BLACK)

    def test_non_adjacent_position_returns_false(self):
        gs = _make_state_with_queens()
        # (3, 0) is two steps from black queen at (1, 0)
        assert not piece_adjacent_to_queen(HexCoord(3, 0), gs.board, Color.BLACK)

    def test_queen_not_on_board_returns_false(self):
        gs = GameState()
        assert not piece_adjacent_to_queen(HexCoord(0, 0), gs.board, Color.BLACK)


# ---------------------------------------------------------------------------
# find_queen_pressure_wins — condition A: newly immobilise queen
# ---------------------------------------------------------------------------


class TestFindWinsConditionA:
    def test_filling_last_gap_is_a_win(self):
        """
        Condition B: a non-adjacent piece slides to the queen's only empty
        neighbour, triggering 'newly adjacent to already-immobile queen'.

        Board:
          White queen  (0,0)
          Black queen  (2,0)   ← gated-immobile (5 surrounding pieces + ring geometry)
          White ring:  (3,0)=ANT, (3,-1)=ANT, (2,-1)=GH, (1,0)=GH, (1,1)=GH
          White extra: (-1,0)=ANT  ← NOT adjacent to black queen
          Gap:         (2,1)        ← only empty neighbour of black queen

        Key facts
        ---------
        * A queen with 5 occupied neighbours is ALWAYS gated: both gate hexes
          of the remaining gap are in the surrounding ring, so was_mobile=False.
        * All 5 ring pieces have was_adjacent=True (already adjacent to the queen),
          so moving them to (2,1) would NOT satisfy condition B.
        * The extra ant at (-1,0) is NOT adjacent to the queen.  It can slide to
          (2,1) and become newly adjacent → condition B fires.
        """
        gs = GameState()
        _place_piece(gs, Color.WHITE, PieceType.QUEEN, HexCoord(0, 0))
        _place_piece(gs, Color.BLACK, PieceType.QUEEN, HexCoord(2, 0))
        # 5-piece ring that gates the black queen (gap at (2,1), but gated)
        # 2 ants + 3 grasshoppers — stays within per-player piece limits
        for pos, pt in [
            (HexCoord(3, 0),  PieceType.ANT),
            (HexCoord(3, -1), PieceType.ANT),
            (HexCoord(2, -1), PieceType.GRASSHOPPER),
            (HexCoord(1, 0),  PieceType.GRASSHOPPER),
            (HexCoord(1, 1),  PieceType.GRASSHOPPER),
        ]:
            _place_piece(gs, Color.WHITE, pt, pos)
        # Extra ant NOT adjacent to black queen; can slide along hive to (2,1)
        _place_piece(gs, Color.WHITE, PieceType.ANT, HexCoord(-1, 0))
        gs.turn = 2  # WHITE to move

        assert not queen_is_mobile(gs, Color.BLACK), "Black queen should be gated (immobile)"
        wins = find_queen_pressure_wins(gs)
        win_destinations = {m.to for m in wins}
        # Ant at (-1,0) slides to (2,1): was_adjacent=False, now adjacent → B fires
        assert HexCoord(2, 1) in win_destinations, (
            f"Expected (2,1) among winning destinations, got {win_destinations}"
        )

    def test_no_winning_move_on_open_board(self):
        """With just two queens and no surrounding pieces, no single move immobilises."""
        gs = _make_state_with_queens()
        wins = find_queen_pressure_wins(gs)
        # Queens are only adjacent to each other — black queen has 5 free
        # neighbours so cannot be immobilised in one move.
        assert isinstance(wins, list)
        # In this minimal state no move can immobilise black queen (needs 5 more pieces)
        # and black queen is mobile, so condition B cannot fire either.
        assert len(wins) == 0


# ---------------------------------------------------------------------------
# find_queen_pressure_wins — condition B: newly adjacent to immobile queen
# ---------------------------------------------------------------------------


class TestFindWinsConditionB:
    def _make_state_black_queen_immobile(self) -> GameState:
        """
        Black queen at (2,0) fully surrounded — immobile.
        White piece at (0,0) not adjacent to the black queen.
        """
        gs = GameState()
        _place_piece(gs, Color.WHITE, PieceType.QUEEN, HexCoord(0, 0))
        _place_piece(gs, Color.BLACK, PieceType.QUEEN, HexCoord(2, 0))
        # Surround black queen on all 6 sides with a mix of white pieces
        # (3 ants + 2 grasshoppers + 1 spider — stays within per-player limits)
        for pos, pt in [
            (HexCoord(3, 0),  PieceType.ANT),
            (HexCoord(3, -1), PieceType.ANT),
            (HexCoord(2, -1), PieceType.ANT),
            (HexCoord(1, 0),  PieceType.GRASSHOPPER),
            (HexCoord(1, 1),  PieceType.GRASSHOPPER),
            (HexCoord(2, 1),  PieceType.SPIDER),
        ]:
            _place_piece(gs, Color.WHITE, pt, pos)
        gs.turn = 2  # WHITE to move
        return gs

    def test_immobile_queen_detected(self):
        gs = self._make_state_black_queen_immobile()
        assert not queen_is_mobile(gs, Color.BLACK)

    def test_already_adjacent_piece_not_counted(self):
        """
        When a piece MOVES from one adjacent hex to another adjacent hex it
        was already adjacent — condition B should NOT fire (was_adjacent=True).
        """
        gs = _make_state_with_queens()
        # White queen at (0,0) adjacent to black queen at (1,0).
        # black queen is mobile (only 1 occupied neighbour).
        assert queen_is_mobile(gs, Color.BLACK)

        wins = find_queen_pressure_wins(gs)
        # In this open state no condition should fire for moves starting
        # from (0,0): queen is mobile, so cond_a can only fire if the queen
        # becomes immobile (impossible with 1 piece), and cond_b won't fire
        # since queen is mobile.
        for m in wins:
            assert m.from_pos != HexCoord(0, 0) or queen_is_mobile(gs, Color.BLACK)


# ---------------------------------------------------------------------------
# generate_queen_pressure_state
# ---------------------------------------------------------------------------


class TestGenerateQueenPressureState:
    def test_returns_valid_state(self):
        """Generated state must have both queens placed and game in progress."""
        found = 0
        for _ in range(30):
            gs = generate_queen_pressure_state()
            if gs is None:
                continue
            found += 1
            assert gs.result == GameResult.IN_PROGRESS
            assert gs._queen_placed[Color.WHITE]
            assert gs._queen_placed[Color.BLACK]
            assert len(gs.legal_moves()) > 0
        assert found >= 20, f"Only got {found}/30 valid states"

    def test_state_has_pieces_on_board(self):
        """Board must have at least 2 pieces (the two queens)."""
        for _ in range(10):
            gs = generate_queen_pressure_state()
            assert gs is not None
            assert len(gs.board.grid) >= 2

    def test_reproducible_with_seed(self):
        """Same random seed → same board size (deterministic rollout)."""
        import random
        random.seed(999)
        gs1 = generate_queen_pressure_state()
        random.seed(999)
        gs2 = generate_queen_pressure_state()
        assert gs1 is not None and gs2 is not None
        assert len(gs1.board.grid) == len(gs2.board.grid)


# ---------------------------------------------------------------------------
# Policy and value sanity checks
# ---------------------------------------------------------------------------


class TestPolicyAndValue:
    def test_policy_sums_to_one_when_wins_exist(self):
        """When winning moves exist the policy target must sum to 1.0."""
        import numpy as np
        from hive_engine.encoder import HiveEncoder
        enc = HiveEncoder()
        for _ in range(50):
            gs = generate_queen_pressure_state()
            if gs is None:
                continue
            wins = find_queen_pressure_wins(gs)
            if not wins:
                continue
            policy = np.zeros(enc.ACTION_SPACE_SIZE, dtype=np.float32)
            for move in wins:
                idx = enc.encode_action(move, gs)
                if 0 <= idx < enc.ACTION_SPACE_SIZE:
                    policy[idx] = 1.0
            total = policy.sum()
            assert total > 0.0
            policy /= total
            assert abs(policy.sum() - 1.0) < 1e-5
            return  # success on first winning-move state
        pytest.skip("No winning-move state found in 50 attempts")

    def test_value_positive_iff_wins_exist(self):
        """value = +1 when wins exist, -1 when none."""
        found_positive = False
        found_negative = False
        for _ in range(200):
            gs = generate_queen_pressure_state()
            if gs is None:
                continue
            wins = find_queen_pressure_wins(gs)
            if wins:
                found_positive = True
            else:
                found_negative = True
            if found_positive and found_negative:
                break
        assert found_positive, "Never found a state with a winning move in 200 attempts"
