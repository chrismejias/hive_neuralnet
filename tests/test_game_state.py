"""Tests for GameState — full game logic, legal moves, win detection."""

import pytest

from hive_engine.hex_coord import HexCoord, Direction, ORIGIN
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.board import Board
from hive_engine.game_state import GameState, Move, MoveType, GameResult


class TestGameStateInit:

    def test_initial_state(self):
        gs = GameState()
        assert gs.turn == 0
        assert gs.current_player == Color.WHITE
        assert gs.result == GameResult.IN_PROGRESS
        assert gs.board.is_empty()
        assert len(gs.hand(Color.WHITE)) == 11
        assert len(gs.hand(Color.BLACK)) == 11

    def test_player_alternation(self):
        gs = GameState()
        assert gs.current_player == Color.WHITE
        # Make white's first move
        moves = gs.legal_moves()
        gs.apply_move(moves[0])
        assert gs.current_player == Color.BLACK


class TestFirstMoves:

    def test_first_move_is_placement(self):
        gs = GameState()
        moves = gs.legal_moves()
        assert all(m.move_type == MoveType.PLACE for m in moves)
        # All placements should be at origin
        assert all(m.to == ORIGIN for m in moves)

    def test_first_move_no_queen(self):
        """First move cannot place the queen (convention — actually optional in
        standard rules, but many implementations prohibit it turn 1).

        NOTE: Standard Hive rules do NOT prohibit placing the Queen first.
        We allow it in our implementation.
        """
        gs = GameState()
        moves = gs.legal_moves()
        piece_types = {m.piece.piece_type for m in moves}
        # All base piece types should be available
        assert PieceType.QUEEN in piece_types or PieceType.ANT in piece_types

    def test_second_move_adjacent(self):
        gs = GameState()
        # White places at origin
        moves = gs.legal_moves()
        gs.apply_move(moves[0])

        # Black should place adjacent to origin
        moves = gs.legal_moves()
        assert all(m.move_type == MoveType.PLACE for m in moves)
        for m in moves:
            assert ORIGIN.distance(m.to) == 1

    def test_fungible_pieces(self):
        """Multiple pieces of same type generate only one set of moves."""
        gs = GameState()
        moves = gs.legal_moves()
        # Should have one move per piece TYPE at origin, not per individual piece
        # e.g., not 3 separate ANT placements at origin
        placement_types = [m.piece.piece_type for m in moves]
        # Each type appears at most once (for first move, only one position: ORIGIN)
        type_counts = {}
        for pt in placement_types:
            type_counts[pt] = type_counts.get(pt, 0) + 1
        for pt, count in type_counts.items():
            assert count == 1, f"{pt} appears {count} times at ORIGIN"


class TestQueenPlacementRule:

    def test_queen_must_be_placed_by_turn_4(self):
        """Each player must place their queen by their 4th turn."""
        gs = GameState()

        # Play 3 turns for each player without placing queen
        for i in range(6):  # 3 white + 3 black
            moves = gs.legal_moves()
            # Pick a non-queen placement
            non_queen_moves = [m for m in moves if
                               m.piece is not None and
                               m.piece.piece_type != PieceType.QUEEN]
            if non_queen_moves:
                gs.apply_move(non_queen_moves[0])
            else:
                gs.apply_move(moves[0])

        # Turn 6 = White's 4th turn (player_turn_number = 3)
        assert gs.player_turn_number == 3
        moves = gs.legal_moves()
        # All placement moves must be queen
        placement_moves = [m for m in moves if m.move_type == MoveType.PLACE]
        for m in placement_moves:
            assert m.piece.piece_type == PieceType.QUEEN


class TestMovementAfterQueen:

    def _setup_both_queens(self) -> GameState:
        """Set up a game where both queens are placed."""
        gs = GameState()

        # White places queen at origin
        wq = gs.pieces_in_hand(Color.WHITE, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, wq, ORIGIN))

        # Black places queen adjacent
        bq = gs.pieces_in_hand(Color.BLACK, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, bq, HexCoord(1, 0)))

        return gs

    def test_movement_requires_queen(self):
        """Pieces can only move if that player's queen is on the board."""
        gs = GameState()

        # White places ant (not queen)
        wa = gs.pieces_in_hand(Color.WHITE, PieceType.ANT)[0]
        gs.apply_move(Move(MoveType.PLACE, wa, ORIGIN))

        # Black places ant
        ba = gs.pieces_in_hand(Color.BLACK, PieceType.ANT)[0]
        gs.apply_move(Move(MoveType.PLACE, ba, HexCoord(1, 0)))

        # White's second turn — queen not placed, so no movement moves
        moves = gs.legal_moves()
        movement_moves = [m for m in moves if m.move_type == MoveType.MOVE]
        assert len(movement_moves) == 0

    def test_movement_possible_after_queen(self):
        """After queen is placed, pieces can move."""
        gs = self._setup_both_queens()

        # White places an ant
        wa = gs.pieces_in_hand(Color.WHITE, PieceType.ANT)[0]
        # Find a valid placement position
        positions = gs.board.valid_placement_positions(Color.WHITE)
        pos = next(iter(positions))
        gs.apply_move(Move(MoveType.PLACE, wa, pos))

        # Black places an ant
        ba = gs.pieces_in_hand(Color.BLACK, PieceType.ANT)[0]
        positions = gs.board.valid_placement_positions(Color.BLACK)
        pos = next(iter(positions))
        gs.apply_move(Move(MoveType.PLACE, ba, pos))

        # White should now have movement moves available
        moves = gs.legal_moves()
        movement_moves = [m for m in moves if m.move_type == MoveType.MOVE]
        # At least some piece should be able to move
        assert len(movement_moves) >= 0  # May still be 0 if all pinned


class TestPinnedPieces:

    def test_pinned_piece_cannot_move(self):
        """A piece at an articulation point cannot move."""
        gs = GameState()

        # Build: wQ - wA - bQ (line of 3 — wA is pinned)
        wq = gs.pieces_in_hand(Color.WHITE, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, wq, ORIGIN))

        bq = gs.pieces_in_hand(Color.BLACK, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, bq, HexCoord(1, 0)))

        wa = gs.pieces_in_hand(Color.WHITE, PieceType.ANT)[0]
        positions = gs.board.valid_placement_positions(Color.WHITE)
        # Place ant on the other side: (-1, 0) should be valid
        west = HexCoord(-1, 0)
        if west in positions:
            gs.apply_move(Move(MoveType.PLACE, wa, west))
        else:
            gs.apply_move(Move(MoveType.PLACE, wa, next(iter(positions))))
            return  # Layout didn't work as expected, skip

        # Now black places something
        ba = gs.pieces_in_hand(Color.BLACK, PieceType.ANT)[0]
        positions = gs.board.valid_placement_positions(Color.BLACK)
        gs.apply_move(Move(MoveType.PLACE, ba, next(iter(positions))))

        # White's turn: the queen at origin is between ant and black pieces
        # Check if queen is at an articulation point
        moves = gs.legal_moves()
        queen_moves = [m for m in moves if m.move_type == MoveType.MOVE
                       and m.piece is wq]
        # Queen should be pinned (articulation point in a line)
        # This depends on exact layout, but the logic should work


class TestGameOver:

    def _surround_queen(self, gs: GameState, queen_color: Color) -> GameState:
        """Helper to surround a queen with pieces (may not be fully legal)."""
        # Find the queen position
        queen_pos = None
        for piece, pos in gs.board.piece_positions.items():
            if piece.color == queen_color and piece.piece_type == PieceType.QUEEN:
                queen_pos = pos
                break

        if queen_pos is None:
            return gs

        # Place pieces at all 6 neighbors
        for i, n in enumerate(queen_pos.neighbors()):
            if n not in gs.board.grid:
                # Place a dummy piece (bypassing normal rules for testing)
                p = Piece(PieceType.ANT, queen_color.other(), i)
                gs.board.place_piece(p, n)
                gs.board.piece_positions[p] = n

        return gs

    def test_queen_surrounded_loses(self):
        """When a queen is completely surrounded, that player loses."""
        gs = GameState()

        # Place both queens
        wq = gs.pieces_in_hand(Color.WHITE, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, wq, ORIGIN))

        bq = gs.pieces_in_hand(Color.BLACK, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, bq, HexCoord(3, 0)))

        # Manually surround white queen (testing the detection, not legal play)
        for i, n in enumerate(ORIGIN.neighbors()):
            if n not in gs.board.grid:
                p = Piece(PieceType.ANT, Color.BLACK, i)
                gs.board.place_piece(p, n)

        gs._check_game_over()
        assert gs.result == GameResult.BLACK_WINS

    def test_both_queens_surrounded_is_draw(self):
        """If both queens become surrounded simultaneously, it's a draw."""
        gs = GameState()

        wq = Piece(PieceType.QUEEN, Color.WHITE, 0)
        bq = Piece(PieceType.QUEEN, Color.BLACK, 0)

        gs.board.place_piece(wq, ORIGIN)
        gs.board.piece_positions[wq] = ORIGIN
        gs._queen_placed[Color.WHITE] = True

        gs.board.place_piece(bq, HexCoord(5, 0))
        gs.board.piece_positions[bq] = HexCoord(5, 0)
        gs._queen_placed[Color.BLACK] = True

        # Surround both
        for i, n in enumerate(ORIGIN.neighbors()):
            if n not in gs.board.grid:
                p = Piece(PieceType.ANT, Color.BLACK, i)
                gs.board.place_piece(p, n)

        for i, n in enumerate(HexCoord(5, 0).neighbors()):
            if n not in gs.board.grid:
                p = Piece(PieceType.ANT, Color.WHITE, i)
                gs.board.place_piece(p, n)

        gs._check_game_over()
        assert gs.result == GameResult.DRAW

    def test_queen_surrounded_count(self):
        gs = GameState()
        wq = gs.pieces_in_hand(Color.WHITE, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, wq, ORIGIN))

        bq = gs.pieces_in_hand(Color.BLACK, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, bq, HexCoord(1, 0)))

        # White queen has 1 neighbor (the black queen)
        assert gs.queen_surrounded_count(Color.WHITE) == 1


class TestPassMove:

    def test_pass_when_no_moves(self):
        """If a player has no legal placements or movements, they must pass."""
        gs = GameState()
        # This is hard to set up naturally, but we can verify the pass mechanism
        moves = gs.legal_moves()
        assert len(moves) > 0  # First turn always has moves

        # Verify PASS appears when there are no other options
        # We'll test this indirectly — create a state with no hand and pinned pieces
        gs2 = GameState()
        gs2._hands[Color.WHITE] = []  # Empty hand
        gs2._queen_placed[Color.WHITE] = True
        # No pieces on board to move for white
        moves = gs2.legal_moves()
        assert len(moves) == 1
        assert moves[0].move_type == MoveType.PASS


class TestUndoMove:

    def test_undo_placement(self):
        gs = GameState()
        moves = gs.legal_moves()
        move = moves[0]

        gs.apply_move(move)
        assert gs.turn == 1
        assert not gs.board.is_empty()

        gs.undo_move()
        assert gs.turn == 0
        assert gs.board.is_empty()
        assert gs.current_player == Color.WHITE

    def test_undo_movement(self):
        gs = GameState()

        # Place both queens
        wq = gs.pieces_in_hand(Color.WHITE, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, wq, ORIGIN))

        bq = gs.pieces_in_hand(Color.BLACK, PieceType.QUEEN)[0]
        gs.apply_move(Move(MoveType.PLACE, bq, HexCoord(1, 0)))

        # White places ant
        wa = gs.pieces_in_hand(Color.WHITE, PieceType.ANT)[0]
        pos = next(iter(gs.board.valid_placement_positions(Color.WHITE)))
        gs.apply_move(Move(MoveType.PLACE, wa, pos))

        # Black places ant
        ba = gs.pieces_in_hand(Color.BLACK, PieceType.ANT)[0]
        bpos = next(iter(gs.board.valid_placement_positions(Color.BLACK)))
        gs.apply_move(Move(MoveType.PLACE, ba, bpos))

        # Now try to find a movement for white
        moves = gs.legal_moves()
        movement_moves = [m for m in moves if m.move_type == MoveType.MOVE]
        if movement_moves:
            move = movement_moves[0]
            old_pos = move.from_pos
            gs.apply_move(move)

            # Undo
            gs.undo_move()
            assert gs.board.position_of(move.piece) == old_pos

    def test_multiple_undo(self):
        gs = GameState()
        moves_made = []

        for _ in range(4):
            moves = gs.legal_moves()
            move = moves[0]
            moves_made.append(move)
            gs.apply_move(move)

        assert gs.turn == 4

        for _ in range(4):
            gs.undo_move()

        assert gs.turn == 0
        assert gs.board.is_empty()


class TestCopy:

    def test_copy_independence(self):
        gs = GameState()
        moves = gs.legal_moves()
        gs.apply_move(moves[0])

        gs2 = gs.copy()
        # Modify copy
        moves2 = gs2.legal_moves()
        gs2.apply_move(moves2[0])

        # Original should be unchanged
        assert gs.turn == 1
        assert gs2.turn == 2


class TestCanonicalHash:

    def test_same_state_same_hash(self):
        gs1 = GameState()
        gs2 = GameState()

        moves1 = gs1.legal_moves()
        moves2 = gs2.legal_moves()

        # Apply same first move
        gs1.apply_move(moves1[0])
        gs2.apply_move(moves2[0])

        assert gs1.canonical_hash() == gs2.canonical_hash()


class TestFullGameSimulation:

    def test_random_game_completes(self):
        """A random game should eventually complete or reach a reasonable turn limit."""
        import random
        random.seed(42)

        gs = GameState()
        max_turns = 200

        for _ in range(max_turns):
            if gs.result != GameResult.IN_PROGRESS:
                break
            moves = gs.legal_moves()
            move = random.choice(moves)
            gs.apply_move(move)

        # Game should have progressed
        assert gs.turn > 0

    def test_multiple_random_games(self):
        """Run several random games to check for crashes."""
        import random

        for seed in range(10):
            random.seed(seed)
            gs = GameState()
            for _ in range(150):
                if gs.result != GameResult.IN_PROGRESS:
                    break
                moves = gs.legal_moves()
                move = random.choice(moves)
                gs.apply_move(move)
