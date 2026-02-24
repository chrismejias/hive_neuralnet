"""Comprehensive tests for HiveEncoder — state encoding, action encoding, and legal masking."""

import random

import numpy as np
import pytest

from hive_engine.hex_coord import HexCoord, Direction, ORIGIN
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.board import Board
from hive_engine.game_state import GameState, Move, MoveType, GameResult
from hive_engine.encoder import HiveEncoder


# ── Helpers ──────────────────────────────────────────────────────

def _piece(pt: PieceType, color: Color = Color.WHITE, pid: int = 0) -> Piece:
    return Piece(pt, color, pid)


def _setup_two_queens() -> GameState:
    """Set up a game with both queens placed at origin and (1, 0)."""
    gs = GameState()
    wq = gs.pieces_in_hand(Color.WHITE, PieceType.QUEEN)[0]
    gs.apply_move(Move(MoveType.PLACE, wq, ORIGIN))
    bq = gs.pieces_in_hand(Color.BLACK, PieceType.QUEEN)[0]
    gs.apply_move(Move(MoveType.PLACE, bq, HexCoord(1, 0)))
    return gs


def _play_n_turns(gs: GameState, n: int, seed: int = 42) -> GameState:
    """Play n random turns from the current state."""
    rng = random.Random(seed)
    for _ in range(n):
        if gs.result != GameResult.IN_PROGRESS:
            break
        moves = gs.legal_moves()
        gs.apply_move(rng.choice(moves))
    return gs


# ── Encoder Constants ────────────────────────────────────────────

class TestEncoderConstants:

    def test_board_size(self):
        assert HiveEncoder.BOARD_SIZE == 13
        assert HiveEncoder.HALF_BOARD == 6

    def test_channel_count(self):
        assert HiveEncoder.NUM_PIECE_CHANNELS == 20
        assert HiveEncoder.NUM_META_CHANNELS == 6
        assert HiveEncoder.NUM_CHANNELS == 26

    def test_action_space_size(self):
        assert HiveEncoder.NUM_PLACEMENT_ACTIONS == 5 * 169  # 845
        assert HiveEncoder.NUM_MOVEMENT_ACTIONS == 169 * 169  # 28561
        assert HiveEncoder.ACTION_SPACE_SIZE == 845 + 28561 + 1  # 29407
        assert HiveEncoder.PASS_ACTION_INDEX == 29406


# ── State Encoding Basics ───────────────────────────────────────

class TestEncodeStateBasics:

    def test_output_shape(self):
        enc = HiveEncoder()
        gs = GameState()
        tensor = enc.encode_state(gs)
        assert tensor.shape == (26, 13, 13)

    def test_dtype_float32(self):
        enc = HiveEncoder()
        gs = GameState()
        tensor = enc.encode_state(gs)
        assert tensor.dtype == np.float32

    def test_empty_board_piece_planes_zero(self):
        """On an empty board, all piece planes should be zero."""
        enc = HiveEncoder()
        gs = GameState()
        tensor = enc.encode_state(gs)
        # Channels 0-19 (piece planes) should be all zeros
        assert np.all(tensor[:20] == 0.0)

    def test_empty_board_meta_planes(self):
        """Check meta plane values on an empty board."""
        enc = HiveEncoder()
        gs = GameState()
        tensor = enc.encode_state(gs)
        # Current player = WHITE = 1.0
        assert np.all(tensor[20] == 1.0)
        # Turn 0 -> normalized = 0.0
        assert np.all(tensor[21] == 0.0)
        # No queens placed
        assert np.all(tensor[22] == 0.0)
        assert np.all(tensor[23] == 0.0)
        # Full hands (11/11 = 1.0)
        assert np.allclose(tensor[24], 1.0)
        assert np.allclose(tensor[25], 1.0)

    def test_single_piece_at_origin(self):
        """Place white queen at origin. Should appear at center of grid."""
        enc = HiveEncoder()
        gs = GameState()
        wq = gs.pieces_in_hand(Color.WHITE, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, wq, ORIGIN))

        tensor = enc.encode_state(gs)

        # White Queen (ground) = channel 0, at center (6, 6)
        assert tensor[0, 6, 6] == 1.0
        # Only one piece on board — all other piece plane cells should be 0
        piece_sum = tensor[:20].sum()
        assert piece_sum == 1.0

    def test_two_pieces_adjacent(self):
        """Place white queen at origin, black queen at (1, 0)."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        tensor = enc.encode_state(gs)

        # Centroid of (0,0) and (1,0) is (0.5, 0) -> rounds to (0, 0) or (1, 0)
        center_q, center_r = HiveEncoder._compute_center(gs.board)

        # White queen (channel 0)
        wq_col = 0 - center_q + 6
        wq_row = 0 - center_r + 6
        assert tensor[0, wq_row, wq_col] == 1.0

        # Black queen (channel 5)
        bq_col = 1 - center_q + 6
        bq_row = 0 - center_r + 6
        assert tensor[5, bq_row, bq_col] == 1.0

        # Exactly 2 piece indicators total
        assert tensor[:20].sum() == 2.0


# ── Centering ────────────────────────────────────────────────────

class TestCentering:

    def test_centroid_single_piece(self):
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN), HexCoord(3, -2))
        center_q, center_r = HiveEncoder._compute_center(board)
        assert center_q == 3
        assert center_r == -2

    def test_centroid_symmetric_pair(self):
        board = Board()
        board.place_piece(_piece(PieceType.QUEEN, Color.WHITE), HexCoord(0, 0))
        board.place_piece(_piece(PieceType.QUEEN, Color.BLACK), HexCoord(2, 0))
        center_q, center_r = HiveEncoder._compute_center(board)
        assert center_q == 1
        assert center_r == 0

    def test_centroid_empty_board(self):
        board = Board()
        center_q, center_r = HiveEncoder._compute_center(board)
        assert center_q == 0
        assert center_r == 0

    def test_translation_invariance(self):
        """Same board shape at different absolute positions → identical tensors."""
        enc = HiveEncoder()

        wq = _piece(PieceType.QUEEN, Color.WHITE)
        bq = _piece(PieceType.QUEEN, Color.BLACK)
        wa = _piece(PieceType.ANT, Color.WHITE)

        # Board A: 3 pieces centered around (0,0) — centroid is exactly (0, 0)
        # Use an odd-count set so centroid is unambiguous
        gs_a = GameState()
        gs_a.board.place_piece(wq, HexCoord(-1, 0))
        gs_a.board.place_piece(bq, HexCoord(0, 0))
        gs_a.board.place_piece(wa, HexCoord(1, 0))
        gs_a.turn = 2

        # Board B: same shape translated by (10, 5)
        gs_b = GameState()
        gs_b.board.place_piece(wq, HexCoord(9, 5))
        gs_b.board.place_piece(bq, HexCoord(10, 5))
        gs_b.board.place_piece(wa, HexCoord(11, 5))
        gs_b.turn = 2

        # Verify centroids are correct
        ca = HiveEncoder._compute_center(gs_a.board)
        cb = HiveEncoder._compute_center(gs_b.board)
        assert ca == (0, 0)
        assert cb == (10, 5)

        tensor_a = enc.encode_state(gs_a)
        tensor_b = enc.encode_state(gs_b)

        # Piece planes should be identical
        np.testing.assert_array_equal(tensor_a[:20], tensor_b[:20])

    def test_pieces_centered_in_grid(self):
        """After encoding, occupied cells should be near the center."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 10)

        tensor = enc.encode_state(gs)
        # Find all cells with any piece data
        piece_data = tensor[:20].sum(axis=0)  # (13, 13)
        occupied_rows, occupied_cols = np.where(piece_data > 0)

        if len(occupied_rows) > 0:
            # Average position should be near center (6, 6)
            avg_row = occupied_rows.mean()
            avg_col = occupied_cols.mean()
            assert abs(avg_row - 6) < 4, f"avg_row={avg_row} too far from center"
            assert abs(avg_col - 6) < 4, f"avg_col={avg_col} too far from center"


# ── Piece Planes ─────────────────────────────────────────────────

class TestPiecePlanes:

    def test_all_piece_types_channel_mapping(self):
        """Verify the channel formula: color * 5 + piece_type."""
        enc = HiveEncoder()
        gs = GameState()

        # Place one of each type for white (using direct board manipulation)
        pieces_and_positions = [
            (_piece(PieceType.QUEEN, Color.WHITE), HexCoord(0, 0)),
            (_piece(PieceType.ANT, Color.WHITE), HexCoord(1, 0)),
            (_piece(PieceType.GRASSHOPPER, Color.WHITE), HexCoord(2, 0)),
            (_piece(PieceType.SPIDER, Color.WHITE), HexCoord(3, 0)),
            (_piece(PieceType.BEETLE, Color.WHITE), HexCoord(4, 0)),
            (_piece(PieceType.QUEEN, Color.BLACK), HexCoord(0, 1)),
            (_piece(PieceType.ANT, Color.BLACK), HexCoord(1, 1)),
            (_piece(PieceType.GRASSHOPPER, Color.BLACK), HexCoord(2, 1)),
            (_piece(PieceType.SPIDER, Color.BLACK), HexCoord(3, 1)),
            (_piece(PieceType.BEETLE, Color.BLACK), HexCoord(4, 1)),
        ]

        for p, pos in pieces_and_positions:
            gs.board.place_piece(p, pos)

        tensor = enc.encode_state(gs)

        center_q, center_r = HiveEncoder._compute_center(gs.board)

        # Check each piece appears in the correct channel
        for p, pos in pieces_and_positions:
            expected_ch = p.color.value * 5 + p.piece_type.value
            grid = HiveEncoder._hex_to_grid(pos.q, pos.r, center_q, center_r)
            assert grid is not None
            row, col = grid
            assert tensor[expected_ch, row, col] == 1.0, (
                f"{p} at ({pos.q},{pos.r}) should be in channel {expected_ch}"
            )

    def test_stacking_beetle_on_queen(self):
        """Beetle on top of queen: ground=queen, stacked=beetle."""
        enc = HiveEncoder()
        gs = GameState()

        wq = _piece(PieceType.QUEEN, Color.WHITE)
        bb = _piece(PieceType.BEETLE, Color.BLACK)

        gs.board.place_piece(wq, ORIGIN)
        gs.board.place_piece(bb, ORIGIN)  # Beetle on top

        tensor = enc.encode_state(gs)

        # Ground: white queen → channel 0
        assert tensor[0, 6, 6] == 1.0
        # Stacked: black beetle → channel 10 + 1*5 + 4 = 19
        assert tensor[19, 6, 6] == 1.0
        # No other piece data at (6,6)
        total_at_center = sum(tensor[ch, 6, 6] for ch in range(20))
        assert total_at_center == 2.0

    def test_stack_height_3(self):
        """Three pieces stacked: ground=bottom, stacked=top, middle ignored."""
        enc = HiveEncoder()
        gs = GameState()

        bottom = _piece(PieceType.QUEEN, Color.WHITE)
        middle = _piece(PieceType.BEETLE, Color.BLACK)
        top = _piece(PieceType.BEETLE, Color.WHITE)

        gs.board.place_piece(bottom, ORIGIN)
        gs.board.place_piece(middle, ORIGIN)
        gs.board.place_piece(top, ORIGIN)

        tensor = enc.encode_state(gs)

        # Ground: white queen → channel 0
        assert tensor[0, 6, 6] == 1.0
        # Stacked: white beetle (top) → channel 10 + 0*5 + 4 = 14
        assert tensor[14, 6, 6] == 1.0
        # Middle piece (black beetle, ch 19) should NOT appear in stacked
        assert tensor[19, 6, 6] == 0.0
        # Total piece data at center = 2 (ground + top)
        total_at_center = sum(tensor[ch, 6, 6] for ch in range(20))
        assert total_at_center == 2.0


# ── Meta Planes ──────────────────────────────────────────────────

class TestMetaPlanes:

    def test_current_player_white(self):
        enc = HiveEncoder()
        gs = GameState()  # Turn 0, WHITE's turn
        tensor = enc.encode_state(gs)
        assert np.all(tensor[20] == 1.0)

    def test_current_player_black(self):
        enc = HiveEncoder()
        gs = GameState()
        # Play white's first move
        moves = gs.legal_moves()
        gs.apply_move(moves[0])
        tensor = enc.encode_state(gs)
        assert np.all(tensor[20] == 0.0)

    def test_turn_normalization(self):
        enc = HiveEncoder()
        gs = GameState()

        # Turn 0
        tensor = enc.encode_state(gs)
        assert np.all(tensor[21] == 0.0)

        # Play 10 turns
        gs = _play_n_turns(gs, 10)
        tensor = enc.encode_state(gs)
        expected = min(gs.turn / 100.0, 1.0)
        assert np.allclose(tensor[21], expected)

    def test_turn_normalization_clamps_at_1(self):
        enc = HiveEncoder()
        gs = GameState()
        gs.turn = 200  # Force high turn
        tensor = enc.encode_state(gs)
        assert np.all(tensor[21] == 1.0)

    def test_queen_placed_channels(self):
        enc = HiveEncoder()

        # No queens placed
        gs = GameState()
        tensor = enc.encode_state(gs)
        assert np.all(tensor[22] == 0.0)  # White queen not placed
        assert np.all(tensor[23] == 0.0)  # Black queen not placed

        # Place white queen
        wq = gs.pieces_in_hand(Color.WHITE, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, wq, ORIGIN))
        tensor = enc.encode_state(gs)
        assert np.all(tensor[22] == 1.0)  # White queen placed
        assert np.all(tensor[23] == 0.0)  # Black queen not placed

        # Place black queen
        bq = gs.pieces_in_hand(Color.BLACK, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, bq, HexCoord(1, 0)))
        tensor = enc.encode_state(gs)
        assert np.all(tensor[22] == 1.0)
        assert np.all(tensor[23] == 1.0)

    def test_hand_count_channels(self):
        enc = HiveEncoder()
        gs = GameState()
        tensor = enc.encode_state(gs)

        # Full hands: 11/11 = 1.0
        assert np.allclose(tensor[24], 1.0)
        assert np.allclose(tensor[25], 1.0)

        # After white places, white hand = 10/11
        moves = gs.legal_moves()
        gs.apply_move(moves[0])
        tensor = enc.encode_state(gs)
        assert np.allclose(tensor[24], 10.0 / 11.0)
        assert np.allclose(tensor[25], 1.0)


# ── Hex ↔ Grid Conversion ───────────────────────────────────────

class TestGridConversion:

    def test_hex_to_grid_center(self):
        """Origin with center at origin → grid center (6, 6)."""
        result = HiveEncoder._hex_to_grid(0, 0, 0, 0)
        assert result == (6, 6)

    def test_hex_to_grid_offset(self):
        """Hex (1, 0) with center at origin → grid (6, 7)."""
        result = HiveEncoder._hex_to_grid(1, 0, 0, 0)
        assert result == (6, 7)  # row=0-0+6=6, col=1-0+6=7

    def test_hex_to_grid_out_of_bounds(self):
        """Position far from center should return None."""
        result = HiveEncoder._hex_to_grid(100, 0, 0, 0)
        assert result is None

    def test_grid_to_hex_roundtrip(self):
        """hex_to_grid → grid_to_hex should be identity."""
        for q in range(-5, 6):
            for r in range(-5, 6):
                grid = HiveEncoder._hex_to_grid(q, r, 0, 0)
                if grid is not None:
                    row, col = grid
                    q2, r2 = HiveEncoder._grid_to_hex(row, col, 0, 0)
                    assert q2 == q and r2 == r, f"Roundtrip failed for ({q},{r})"

    def test_grid_boundaries(self):
        """Positions at exactly the grid edge should be valid."""
        # With center at (0,0), valid q range is [-6, 6]
        assert HiveEncoder._hex_to_grid(-6, -6, 0, 0) == (0, 0)
        assert HiveEncoder._hex_to_grid(6, 6, 0, 0) == (12, 12)
        assert HiveEncoder._hex_to_grid(-7, 0, 0, 0) is None
        assert HiveEncoder._hex_to_grid(7, 0, 0, 0) is None


# ── Action Encoding ──────────────────────────────────────────────

class TestEncodeAction:

    def test_pass_action(self):
        enc = HiveEncoder()
        gs = GameState()
        move = Move(MoveType.PASS)
        idx = enc.encode_action(move, gs)
        assert idx == HiveEncoder.PASS_ACTION_INDEX

    def test_placement_action_range(self):
        """All placement actions should be in [0, 845)."""
        enc = HiveEncoder()
        gs = GameState()
        moves = gs.legal_moves()
        for move in moves:
            if move.move_type == MoveType.PLACE:
                idx = enc.encode_action(move, gs)
                assert 0 <= idx < HiveEncoder.NUM_PLACEMENT_ACTIONS

    def test_movement_action_range(self):
        """All movement actions should be in [845, 29406)."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 8)
        moves = gs.legal_moves()
        for move in moves:
            if move.move_type == MoveType.MOVE:
                idx = enc.encode_action(move, gs)
                assert HiveEncoder._MOVEMENT_OFFSET <= idx < HiveEncoder.PASS_ACTION_INDEX

    def test_placement_roundtrip(self):
        """encode → decode should preserve piece type and position for placements."""
        enc = HiveEncoder()
        gs = GameState()
        moves = gs.legal_moves()
        for move in moves:
            if move.move_type == MoveType.PLACE:
                idx = enc.encode_action(move, gs)
                decoded = enc.decode_action(idx, gs)
                assert decoded.move_type == MoveType.PLACE
                assert decoded.piece.piece_type == move.piece.piece_type
                assert decoded.to == move.to

    def test_movement_roundtrip(self):
        """encode → decode should preserve source, destination, and piece for movements."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 8)
        moves = gs.legal_moves()
        movement_moves = [m for m in moves if m.move_type == MoveType.MOVE]
        for move in movement_moves:
            idx = enc.encode_action(move, gs)
            decoded = enc.decode_action(idx, gs)
            assert decoded.move_type == MoveType.MOVE
            assert decoded.from_pos == move.from_pos
            assert decoded.to == move.to
            assert decoded.piece == move.piece

    def test_pass_roundtrip(self):
        enc = HiveEncoder()
        gs = GameState()
        move = Move(MoveType.PASS)
        idx = enc.encode_action(move, gs)
        decoded = enc.decode_action(idx, gs)
        assert decoded.move_type == MoveType.PASS

    def test_action_indices_unique(self):
        """All legal moves should map to distinct action indices."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 8)
        moves = gs.legal_moves()
        indices = [enc.encode_action(m, gs) for m in moves]
        assert len(indices) == len(set(indices)), "Duplicate action indices found"

    def test_all_indices_in_range(self):
        """All encoded action indices should be valid."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 10)
        moves = gs.legal_moves()
        for move in moves:
            idx = enc.encode_action(move, gs)
            assert 0 <= idx < HiveEncoder.ACTION_SPACE_SIZE


# ── Decode Action ────────────────────────────────────────────────

class TestDecodeAction:

    def test_decode_invalid_index(self):
        enc = HiveEncoder()
        gs = GameState()
        with pytest.raises(ValueError):
            enc.decode_action(-1, gs)
        with pytest.raises(ValueError):
            enc.decode_action(HiveEncoder.ACTION_SPACE_SIZE, gs)

    def test_decode_placement_first_turn(self):
        """Decode a placement action on the first turn."""
        enc = HiveEncoder()
        gs = GameState()
        # Manually construct action for placing QUEEN at grid center
        # PieceType.QUEEN = 0, center = (6,6) -> pos_idx = 6*13+6 = 84
        action_idx = 0 * 169 + 6 * 13 + 6  # = 84
        move = enc.decode_action(action_idx, gs)
        assert move.move_type == MoveType.PLACE
        assert move.piece.piece_type == PieceType.QUEEN
        assert move.to == ORIGIN  # Center maps to origin when board is empty

    def test_decode_pass(self):
        enc = HiveEncoder()
        gs = GameState()
        move = enc.decode_action(HiveEncoder.PASS_ACTION_INDEX, gs)
        assert move.move_type == MoveType.PASS


# ── Legal Action Mask ────────────────────────────────────────────

class TestLegalActionMask:

    def test_mask_shape(self):
        enc = HiveEncoder()
        gs = GameState()
        mask = enc.get_legal_action_mask(gs)
        assert mask.shape == (HiveEncoder.ACTION_SPACE_SIZE,)

    def test_mask_dtype(self):
        enc = HiveEncoder()
        gs = GameState()
        mask = enc.get_legal_action_mask(gs)
        assert mask.dtype == np.float32

    def test_mask_first_turn(self):
        """First turn: 5 piece types at 1 position (ORIGIN) = 5 legal moves."""
        enc = HiveEncoder()
        gs = GameState()
        mask = enc.get_legal_action_mask(gs)
        assert mask.sum() == 5

    def test_mask_count_matches_legal_moves(self):
        """The number of 1s in the mask should equal len(legal_moves())."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 10)

        moves = gs.legal_moves()
        mask = enc.get_legal_action_mask(gs, legal_moves=moves)
        assert int(mask.sum()) == len(moves)

    def test_mask_with_precomputed_moves(self):
        """Passing pre-computed moves should give same result."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 6)

        moves = gs.legal_moves()
        mask_auto = enc.get_legal_action_mask(gs)
        mask_pre = enc.get_legal_action_mask(gs, legal_moves=moves)
        np.testing.assert_array_equal(mask_auto, mask_pre)

    def test_pass_only_state(self):
        """When only pass is legal, mask should have exactly one 1."""
        enc = HiveEncoder()
        gs = GameState()
        # Create a state where only pass is available
        gs._hands[Color.WHITE] = []
        gs._queen_placed[Color.WHITE] = True

        moves = gs.legal_moves()
        assert len(moves) == 1
        assert moves[0].move_type == MoveType.PASS

        mask = enc.get_legal_action_mask(gs)
        assert mask.sum() == 1
        assert mask[HiveEncoder.PASS_ACTION_INDEX] == 1.0

    def test_all_masked_actions_decodable(self):
        """Every action index with mask=1 should decode without error."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 8)

        mask = enc.get_legal_action_mask(gs)
        legal_indices = np.where(mask > 0)[0]

        for idx in legal_indices:
            move = enc.decode_action(int(idx), gs)
            assert move is not None


# ── Roundtrip Consistency ────────────────────────────────────────

class TestRoundtripConsistency:

    def test_encode_decode_all_legal_moves(self):
        """For a mid-game state, every legal move roundtrips correctly."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 12)

        moves = gs.legal_moves()
        for move in moves:
            if move.move_type == MoveType.PASS:
                idx = enc.encode_action(move, gs)
                decoded = enc.decode_action(idx, gs)
                assert decoded.move_type == MoveType.PASS
            elif move.move_type == MoveType.PLACE:
                idx = enc.encode_action(move, gs)
                decoded = enc.decode_action(idx, gs)
                assert decoded.move_type == MoveType.PLACE
                assert decoded.piece.piece_type == move.piece.piece_type
                assert decoded.to == move.to
            elif move.move_type == MoveType.MOVE:
                idx = enc.encode_action(move, gs)
                decoded = enc.decode_action(idx, gs)
                assert decoded.move_type == MoveType.MOVE
                assert decoded.from_pos == move.from_pos
                assert decoded.to == move.to
                assert decoded.piece == move.piece

    def test_consistent_centering(self):
        """encode_state and encode_action should use the same center."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 10)

        # Encode state — internally computes center
        tensor = enc.encode_state(gs)

        # Encode all legal moves — should use same center
        moves = gs.legal_moves()
        for move in moves:
            # This should not raise ValueError (would indicate center mismatch)
            idx = enc.encode_action(move, gs)
            assert 0 <= idx < HiveEncoder.ACTION_SPACE_SIZE


# ── Full Game Encoding ───────────────────────────────────────────

class TestFullGameEncoding:

    def test_random_game_no_crashes(self):
        """Play a full random game, encoding every state and move."""
        enc = HiveEncoder()
        rng = random.Random(42)
        gs = GameState()

        for turn in range(100):
            if gs.result != GameResult.IN_PROGRESS:
                break

            # Encode state
            tensor = enc.encode_state(gs)
            assert tensor.shape == (26, 13, 13)
            assert not np.any(np.isnan(tensor))
            assert not np.any(np.isinf(tensor))

            # Encode legal mask — may be less than len(moves) if some
            # moves target positions outside the 13×13 grid (clipped)
            moves = gs.legal_moves()
            mask = enc.get_legal_action_mask(gs, legal_moves=moves)
            assert mask.sum() <= len(moves)
            assert mask.sum() >= 1  # At least one move should be encodable

            # Pick a move that's encodable (in the mask)
            encodable_moves = []
            for m in moves:
                try:
                    idx = enc.encode_action(m, gs)
                    encodable_moves.append((m, idx))
                except ValueError:
                    pass  # Clipped move

            assert len(encodable_moves) > 0
            move, idx = rng.choice(encodable_moves)
            assert 0 <= idx < HiveEncoder.ACTION_SPACE_SIZE
            assert mask[idx] == 1.0  # Chosen move should be in the mask

            gs.apply_move(move)

    def test_multiple_random_games(self):
        """Run several random games with encoding to check for crashes."""
        enc = HiveEncoder()
        for seed in range(5):
            rng = random.Random(seed)
            gs = GameState()
            for _ in range(80):
                if gs.result != GameResult.IN_PROGRESS:
                    break
                tensor = enc.encode_state(gs)
                moves = gs.legal_moves()
                mask = enc.get_legal_action_mask(gs, legal_moves=moves)
                assert mask.sum() >= 1

                # Pick an encodable move
                move = rng.choice(moves)
                try:
                    idx = enc.encode_action(move, gs)
                    assert mask[idx] == 1.0
                except ValueError:
                    pass  # Clipped — acceptable
                gs.apply_move(move)


# ── Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:

    def test_board_exceeds_grid_no_crash(self):
        """Board extending beyond 13×13 should not crash, just clip pieces."""
        enc = HiveEncoder()
        gs = GameState()

        # Place pieces far apart (synthetic, not legal play)
        gs.board.place_piece(
            _piece(PieceType.QUEEN, Color.WHITE), HexCoord(0, 0)
        )
        gs.board.place_piece(
            _piece(PieceType.QUEEN, Color.BLACK), HexCoord(20, 0)
        )

        # Should not crash
        tensor = enc.encode_state(gs)
        assert tensor.shape == (26, 13, 13)
        # At least one piece should be visible (within grid)
        # The centroid is at (10, 0), so one piece is at (-10, 0) from center
        # which is col = -10 + 6 = -4, out of bounds → clipped
        # Other piece at (10, 0) from center → col = 10 + 6 = 16, also out of bounds
        # Actually centroid = (10, 0), piece at (0,0) → col = 0-10+6 = -4 (clipped)
        # piece at (20,0) → col = 20-10+6 = 16 (clipped)
        # Both clipped is expected for extreme cases

    def test_first_move_encoding(self):
        """The very first placement on an empty board should encode correctly."""
        enc = HiveEncoder()
        gs = GameState()
        moves = gs.legal_moves()

        # All moves should be PLACE at ORIGIN
        for move in moves:
            assert move.move_type == MoveType.PLACE
            assert move.to == ORIGIN
            idx = enc.encode_action(move, gs)
            decoded = enc.decode_action(idx, gs)
            assert decoded.to == ORIGIN
            assert decoded.piece.piece_type == move.piece.piece_type

    def test_encode_state_values_bounded(self):
        """All tensor values should be in [0, 1]."""
        enc = HiveEncoder()
        gs = _setup_two_queens()
        gs = _play_n_turns(gs, 15)
        tensor = enc.encode_state(gs)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
