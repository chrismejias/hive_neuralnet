"""
Fuzz tests for MoveGenCache correctness.

Plays many random games and at every state verifies:
1. Cached placement positions match board.valid_placement_positions()
2. Cached ant moves match board._ant_bfs() for every movable ant
3. Cached spider moves match board.generate_slides(piece, max_distance=3)

Also tests edge cases: undo, copy, beetle stacking, ring formation.
"""

import random

import pytest

from hive_engine.board import Board
from hive_engine.game_state import GameState, GameResult, MoveType
from hive_engine.hex_coord import HexCoord, ORIGIN
from hive_engine.move_gen import (
    MoveGenCache,
    _classify_placement,
    _boundary_is_connected,
    _adjusted_boundary,
    _has_obstruction,
    _spider_dfs_tuples,
)
from hive_engine.pieces import Color, PieceType


def _verify_placements(gs: GameState) -> None:
    """Verify cached placement matches fresh computation."""
    if gs.turn < 2:
        return  # Cache only used for turn >= 2

    cache = gs._move_gen_cache
    board = gs.board

    for color in (Color.WHITE, Color.BLACK):
        cached = cache.get_placement_positions(board, color)
        expected = board.valid_placement_positions(color)
        assert cached == expected, (
            f"Turn {gs.turn}, {color.name}: "
            f"cached={sorted((h.q, h.r) for h in cached)}, "
            f"expected={sorted((h.q, h.r) for h in expected)}"
        )


def _verify_ant_moves(gs: GameState) -> None:
    """Verify cached ant moves match fresh BFS for every movable ant."""
    board = gs.board
    color = gs.current_player

    if not gs._queen_placed[color]:
        return

    ap = board.find_articulation_points()
    cache = gs._move_gen_cache

    for piece in board.pieces_of_color(color):
        if piece.piece_type != PieceType.ANT:
            continue
        pos = board.position_of(piece)
        if pos is None:
            continue
        if not board.is_on_top(piece):
            continue
        if pos in ap and board.stack_height(pos) == 1:
            continue

        cached = cache.get_ant_moves(board, pos)
        expected = board._ant_bfs(pos)
        assert cached == expected, (
            f"Turn {gs.turn}, ant at ({pos.q}, {pos.r}): "
            f"cached {len(cached)} moves, expected {len(expected)} moves. "
            f"diff: cached-expected={sorted((h.q, h.r) for h in cached - expected)}, "
            f"expected-cached={sorted((h.q, h.r) for h in expected - cached)}"
        )


def _verify_spider_moves(gs: GameState) -> None:
    """Verify cached spider moves match board.generate_slides for every movable spider."""
    board = gs.board
    color = gs.current_player

    if not gs._queen_placed[color]:
        return

    ap = board.find_articulation_points()
    cache = gs._move_gen_cache

    for piece in board.pieces_of_color(color):
        if piece.piece_type != PieceType.SPIDER:
            continue
        pos = board.position_of(piece)
        if pos is None:
            continue
        if not board.is_on_top(piece):
            continue
        if pos in ap and board.stack_height(pos) == 1:
            continue

        cached = cache.get_spider_moves(board, pos)
        expected = board.generate_slides(piece, max_distance=3)
        assert cached == expected, (
            f"Turn {gs.turn}, spider at ({pos.q}, {pos.r}): "
            f"cached {len(cached)} moves, expected {len(expected)} moves. "
            f"diff: cached-expected={sorted((h.q, h.r) for h in cached - expected)}, "
            f"expected-cached={sorted((h.q, h.r) for h in expected - cached)}"
        )


def play_random_game_with_verification(seed: int, max_turns: int = 200) -> int:
    """Play a random game, verifying cache at every state. Returns turn count."""
    rng = random.Random(seed)
    gs = GameState()

    for turn in range(max_turns):
        if gs.result != GameResult.IN_PROGRESS:
            break

        # Verify cache correctness BEFORE generating moves
        _verify_placements(gs)
        _verify_ant_moves(gs)
        _verify_spider_moves(gs)

        moves = gs.legal_moves()
        move = rng.choice(moves)
        gs.apply_move(move)

    return gs.turn


class TestMoveGenCacheFuzz:
    """Fuzz testing: random games with cache verification at every step."""

    @pytest.mark.parametrize("seed", range(50))
    def test_random_game(self, seed: int) -> None:
        """Play a random game and verify cache at every state."""
        play_random_game_with_verification(seed)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_game_long(self, seed: int) -> None:
        """Longer random games to reach complex board states."""
        play_random_game_with_verification(seed + 1000, max_turns=400)


class TestMoveGenCacheUndo:
    """Test cache correctness after undo operations."""

    @pytest.mark.parametrize("seed", range(10))
    def test_undo_and_verify(self, seed: int) -> None:
        """Play moves, undo some, verify cache is correct."""
        rng = random.Random(seed + 2000)
        gs = GameState()

        for _ in range(60):
            if gs.result != GameResult.IN_PROGRESS:
                break

            moves = gs.legal_moves()
            move = rng.choice(moves)
            gs.apply_move(move)

            # Occasionally undo and verify
            if rng.random() < 0.3 and gs._history:
                gs.undo_move()
                _verify_placements(gs)
                _verify_ant_moves(gs)


class TestMoveGenCacheCopy:
    """Test cache correctness across copy() operations."""

    @pytest.mark.parametrize("seed", range(10))
    def test_copy_diverge(self, seed: int) -> None:
        """Copy a state, play different moves on each, verify both."""
        rng = random.Random(seed + 3000)
        gs = GameState()

        # Play some moves to get a non-trivial state
        for _ in range(20):
            if gs.result != GameResult.IN_PROGRESS:
                break
            moves = gs.legal_moves()
            gs.apply_move(rng.choice(moves))

        if gs.result != GameResult.IN_PROGRESS:
            return

        # Copy the state
        gs_copy = gs.copy()

        # Play different moves on each
        for _ in range(20):
            if gs.result != GameResult.IN_PROGRESS:
                break
            _verify_placements(gs)
            _verify_ant_moves(gs)
            moves = gs.legal_moves()
            gs.apply_move(rng.choice(moves))

        rng2 = random.Random(seed + 4000)
        for _ in range(20):
            if gs_copy.result != GameResult.IN_PROGRESS:
                break
            _verify_placements(gs_copy)
            _verify_ant_moves(gs_copy)
            moves = gs_copy.legal_moves()
            gs_copy.apply_move(rng2.choice(moves))


class TestMoveGenCacheClassify:
    """Unit tests for _classify_placement."""

    def test_empty_board(self) -> None:
        board = Board()
        assert _classify_placement(board.grid, ORIGIN) == -1

    def test_white_only_neighbor(self) -> None:
        from hive_engine.pieces import Piece

        board = Board()
        p = Piece(PieceType.QUEEN, Color.WHITE, 0)
        board.place_piece(p, ORIGIN)
        # All neighbors should be valid for white
        for n in ORIGIN.neighbors():
            assert _classify_placement(board.grid, n) == 0  # WHITE

    def test_black_only_neighbor(self) -> None:
        from hive_engine.pieces import Piece

        board = Board()
        p = Piece(PieceType.QUEEN, Color.BLACK, 0)
        board.place_piece(p, ORIGIN)
        for n in ORIGIN.neighbors():
            assert _classify_placement(board.grid, n) == 1  # BLACK

    def test_both_colors_neighbor(self) -> None:
        from hive_engine.pieces import Piece

        board = Board()
        pw = Piece(PieceType.QUEEN, Color.WHITE, 0)
        pb = Piece(PieceType.QUEEN, Color.BLACK, 0)
        board.place_piece(pw, ORIGIN)
        board.place_piece(pb, HexCoord(1, 0))
        # Position at (1, -1) is adjacent to both => invalid
        assert _classify_placement(board.grid, HexCoord(1, -1)) == -1

    def test_occupied_position(self) -> None:
        from hive_engine.pieces import Piece

        board = Board()
        p = Piece(PieceType.QUEEN, Color.WHITE, 0)
        board.place_piece(p, ORIGIN)
        # ORIGIN is occupied but _classify_placement doesn't check occupancy
        # (caller should check). With a piece there, neighbors are white,
        # so ORIGIN's neighbors check passes.


class TestBoundaryConnectivity:
    """Unit tests for boundary connectivity check."""

    def test_empty_boundary(self) -> None:
        assert _boundary_is_connected(set()) is True

    def test_single_position(self) -> None:
        assert _boundary_is_connected({ORIGIN}) is True

    def test_connected_ring(self) -> None:
        # All 6 neighbors of origin form a connected ring
        boundary = set(ORIGIN.neighbors())
        assert _boundary_is_connected(boundary) is True

    def test_disconnected_boundary(self) -> None:
        # Two separate positions far apart
        boundary = {HexCoord(0, 0), HexCoord(10, 10)}
        assert _boundary_is_connected(boundary) is False


class TestAntFastPath:
    """Test ant fast path specific scenarios."""

    def test_simple_line_no_gates(self) -> None:
        """Ant on a simple line of pieces: fast path should work."""
        from hive_engine.pieces import Piece

        board = Board()
        cache = MoveGenCache()

        # Place 3 pieces in a line: (0,0), (1,0), (2,0)
        p1 = Piece(PieceType.QUEEN, Color.WHITE, 0)
        p2 = Piece(PieceType.ANT, Color.BLACK, 0)
        p3 = Piece(PieceType.QUEEN, Color.BLACK, 0)

        for p, pos in [(p1, ORIGIN), (p2, HexCoord(1, 0)), (p3, HexCoord(2, 0))]:
            board.place_piece(p, pos)
            cache.notify_place(board, p, pos)

        # Ant at (1,0) - get its moves
        ant_pos = HexCoord(1, 0)
        cached_moves = cache.get_ant_moves(board, ant_pos)
        expected_moves = board._ant_bfs(ant_pos)
        assert cached_moves == expected_moves

    def test_fallback_on_gate(self) -> None:
        """When a gate exists, fast path should fall back to BFS."""
        from hive_engine.pieces import Piece

        board = Board()
        cache = MoveGenCache()

        # Build a shape that creates a gate:
        #   (0,-1)  (1,-1)
        # (0,0)  X  (1,0)
        #   (0,1)
        # Where X at (0,0)-(1,0) gap has both flanking occupied
        positions = [
            (ORIGIN, PieceType.QUEEN, Color.WHITE, 0),
            (HexCoord(1, 0), PieceType.QUEEN, Color.BLACK, 0),
            (HexCoord(0, -1), PieceType.ANT, Color.WHITE, 0),
            (HexCoord(1, -1), PieceType.ANT, Color.BLACK, 0),
            (HexCoord(0, 1), PieceType.ANT, Color.WHITE, 1),
        ]

        for pos, pt, color, pid in positions:
            p = Piece(pt, color, pid)
            board.place_piece(p, pos)
            cache.notify_place(board, p, pos)

        # The ant at (0,-1) should still get correct moves (via fallback)
        ant_pos = HexCoord(0, -1)
        cached_moves = cache.get_ant_moves(board, ant_pos)
        expected_moves = board._ant_bfs(ant_pos)
        assert cached_moves == expected_moves


class TestPlacementCache:
    """Direct tests for placement cache operations."""

    def test_init_matches_board(self) -> None:
        """Fresh cache init should match board.valid_placement_positions()."""
        rng = random.Random(7777)
        gs = GameState()
        for _ in range(10):
            if gs.result != GameResult.IN_PROGRESS:
                break
            moves = gs.legal_moves()
            gs.apply_move(rng.choice(moves))

        if gs.turn >= 2:
            _verify_placements(gs)

    def test_incremental_matches_full(self) -> None:
        """After incremental updates, cache should match full recompute."""
        rng = random.Random(8888)
        gs = GameState()
        for _ in range(30):
            if gs.result != GameResult.IN_PROGRESS:
                break
            _verify_placements(gs)
            moves = gs.legal_moves()
            gs.apply_move(rng.choice(moves))

    def test_beetle_stack_changes_color(self) -> None:
        """
        When a beetle stacks on an enemy piece, the top color changes.
        Placement cache should update correctly.
        """
        rng = random.Random(9999)
        gs = GameState()

        # Play until we get a beetle move that stacks
        for _ in range(100):
            if gs.result != GameResult.IN_PROGRESS:
                break
            moves = gs.legal_moves()
            # Prefer beetle moves onto occupied positions
            beetle_stack_moves = [
                m for m in moves
                if m.move_type == MoveType.MOVE
                and m.piece is not None
                and m.piece.piece_type == PieceType.BEETLE
                and m.to is not None
                and m.to in gs.board.grid
            ]
            if beetle_stack_moves:
                gs.apply_move(rng.choice(beetle_stack_moves))
                _verify_placements(gs)
            else:
                gs.apply_move(rng.choice(moves))


class TestSpiderMoves:
    """Tests for the tuple-based spider DFS (_spider_dfs_tuples / get_spider_moves)."""

    def _reference(self, board: Board, piece) -> set[HexCoord]:
        """Reference: board's recursive spider walk."""
        return board.generate_slides(piece, max_distance=3)

    def test_fuzz_random_games(self) -> None:
        """50 random games verifying spider cache at every state."""
        for seed in range(50):
            play_random_game_with_verification(seed)

    def test_fuzz_long_games(self) -> None:
        """10 longer games to reach complex configurations."""
        for seed in range(10):
            play_random_game_with_verification(seed + 2000, max_turns=400)

    def test_simple_line_spider(self) -> None:
        """Spider on a simple line: verifies basic 3-step walk."""
        from hive_engine.pieces import Piece

        board = Board()
        cache = MoveGenCache()

        # Line: (0,0) Q_W -- (1,0) S_B -- (2,0) Q_B
        pieces_and_pos = [
            (Piece(PieceType.QUEEN, Color.WHITE, 0), ORIGIN),
            (Piece(PieceType.SPIDER, Color.BLACK, 0), HexCoord(1, 0)),
            (Piece(PieceType.QUEEN, Color.BLACK, 0), HexCoord(2, 0)),
        ]
        for p, pos in pieces_and_pos:
            board.place_piece(p, pos)
            cache.notify_place(board, p, pos)

        spider_piece = pieces_and_pos[1][0]
        spider_pos = pieces_and_pos[1][1]
        cached = cache.get_spider_moves(board, spider_pos)
        expected = self._reference(board, spider_piece)
        assert cached == expected

    def test_spider_blocked_by_gate(self) -> None:
        """Spider can't pass through a gate — both paths should agree."""
        from hive_engine.pieces import Piece

        board = Board()
        cache = MoveGenCache()

        # Build a shape with a gate between two regions:
        # (0,-1) and (1,-1) form the gate across the (0,0)-(1,0) slide
        positions = [
            (Piece(PieceType.QUEEN,  Color.WHITE, 0), ORIGIN),
            (Piece(PieceType.SPIDER, Color.BLACK, 0), HexCoord(1, 0)),
            (Piece(PieceType.QUEEN,  Color.BLACK, 0), HexCoord(2, 0)),
            (Piece(PieceType.ANT,    Color.WHITE, 0), HexCoord(0, -1)),
            (Piece(PieceType.ANT,    Color.BLACK, 1), HexCoord(1, -1)),
        ]
        spider_piece = positions[1][0]
        spider_pos = positions[1][1]

        for p, pos in positions:
            board.place_piece(p, pos)
            cache.notify_place(board, p, pos)

        cached = cache.get_spider_moves(board, spider_pos)
        expected = self._reference(board, spider_piece)
        assert cached == expected

    def test_spider_returns_empty_when_stuck(self) -> None:
        """If all exits are gated, spider has no moves."""
        from hive_engine.pieces import Piece

        board = Board()
        cache = MoveGenCache()

        # Surround spider fully so every slide is gated:
        # Place 6 pieces around the spider (hexagonal ring)
        spider_piece = Piece(PieceType.SPIDER, Color.WHITE, 0)
        board.place_piece(spider_piece, ORIGIN)
        cache.notify_place(board, spider_piece, ORIGIN)

        for i, pos in enumerate(ORIGIN.neighbors()):
            color = Color.WHITE if i % 2 == 0 else Color.BLACK
            p = Piece(PieceType.QUEEN, color, i)
            board.place_piece(p, pos)
            cache.notify_place(board, p, pos)

        cached = cache.get_spider_moves(board, ORIGIN)
        expected = self._reference(board, spider_piece)
        assert cached == expected  # both should be empty set

    def test_spider_no_revisit_within_path(self) -> None:
        """Spider cannot revisit its own start or intermediate positions."""
        from hive_engine.pieces import Piece
        import random

        rng = random.Random(99)
        gs = GameState()
        # Advance to a mid-game state
        for _ in range(30):
            if gs.result != GameResult.IN_PROGRESS:
                break
            moves = gs.legal_moves()
            gs.apply_move(rng.choice(moves))

        board = gs.board
        cache = gs._move_gen_cache
        ap = board.find_articulation_points()
        color = gs.current_player

        for piece in board.pieces_of_color(color):
            if piece.piece_type != PieceType.SPIDER:
                continue
            pos = board.position_of(piece)
            if pos is None or not board.is_on_top(piece):
                continue
            if pos in ap and board.stack_height(pos) == 1:
                continue

            cached = cache.get_spider_moves(board, pos)
            # The spider's start position must never appear in results
            assert pos not in cached, (
                f"Spider at {pos} returned itself as a destination"
            )

    @pytest.mark.parametrize("seed", range(20))
    def test_spider_vs_reference_per_seed(self, seed: int) -> None:
        """Per-seed parametrized verification against the reference implementation."""
        rng = random.Random(seed + 5000)
        gs = GameState()
        for _ in range(60):
            if gs.result != GameResult.IN_PROGRESS:
                break
            _verify_spider_moves(gs)
            moves = gs.legal_moves()
            gs.apply_move(rng.choice(moves))
