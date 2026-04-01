"""
Incremental legal move generation optimizations.

Provides MoveGenCache which caches and incrementally updates:
1. Placement positions per color (mutual-exclusion aware)
2. Occupied-tuples set for fast ant BFS (~4x faster than HexCoord BFS)

The cache is notified of board mutations and updates incrementally,
avoiding full recomputation.

Used by GameState to accelerate legal_moves() generation.
"""

from __future__ import annotations

from hive_engine.board import Board, _SLIDE_OFFSETS
from hive_engine.hex_coord import HexCoord, _OFFSET_LIST
from hive_engine.pieces import Color, Piece


class MoveGenCache:
    """
    Cached incremental move generation data.

    Maintains placement positions and a tuple-based occupied set across
    moves, updating only the local neighborhood affected by each mutation.

    Thread safety: NOT thread-safe. Each GameState copy gets its own cache.
    """

    __slots__ = (
        "_placement_white",
        "_placement_black",
        "_placement_initialized",
        "_occupied_tuples",
        "_occupied_tuples_valid",
    )

    def __init__(self) -> None:
        self._placement_white: set[HexCoord] = set()
        self._placement_black: set[HexCoord] = set()
        self._placement_initialized: bool = False
        # Tuple-based occupied set for fast ant BFS
        self._occupied_tuples: set[tuple[int, int]] = set()
        self._occupied_tuples_valid: bool = False

    def copy(self) -> MoveGenCache:
        """Deep copy for MCTS rollouts."""
        c = MoveGenCache.__new__(MoveGenCache)
        c._placement_white = set(self._placement_white)
        c._placement_black = set(self._placement_black)
        c._placement_initialized = self._placement_initialized
        c._occupied_tuples = set(self._occupied_tuples)
        c._occupied_tuples_valid = self._occupied_tuples_valid
        return c

    def invalidate(self) -> None:
        """Force full recomputation on next query (used by undo_move)."""
        self._placement_initialized = False
        self._occupied_tuples_valid = False

    # == Placement Cache ================================================

    def get_placement_positions(self, board: Board, color: Color) -> set[HexCoord]:
        """
        Get valid placement positions for a color.

        Returns cached result, initializing lazily on first call.
        Positions are mutually exclusive: valid for WHITE, BLACK, or neither.
        """
        if not self._placement_initialized:
            self._init_placement_cache(board)
        if color == Color.WHITE:
            return self._placement_white
        return self._placement_black

    def _init_placement_cache(self, board: Board) -> None:
        """Full computation of placement positions for both colors."""
        pw: set[HexCoord] = set()
        pb: set[HexCoord] = set()
        checked: set[HexCoord] = set()
        grid = board.grid
        for pos in grid:
            pq, pr = pos.q, pos.r
            for dq, dr in _OFFSET_LIST:
                n = HexCoord(pq + dq, pr + dr)
                if n not in grid and n not in checked:
                    checked.add(n)
                    c = _classify_placement(grid, n)
                    if c == 0:  # WHITE
                        pw.add(n)
                    elif c == 1:  # BLACK
                        pb.add(n)
        self._placement_white = pw
        self._placement_black = pb
        self._placement_initialized = True

    def _update_placement_around(self, board: Board, positions: set[HexCoord]) -> None:
        """
        Incrementally update placement positions near affected hexes.

        Re-classifies all empty positions within distance 1 of the
        given positions, updating both color sets in one pass.
        """
        grid = board.grid
        pw = self._placement_white
        pb = self._placement_black

        # Collect all empty positions to re-evaluate
        recheck: set[HexCoord] = set()
        for p in positions:
            # If p is now empty, it might be a valid placement
            if p not in grid:
                recheck.add(p)
            # Check all neighbors of p
            pq, pr = p.q, p.r
            for dq, dr in _OFFSET_LIST:
                n = HexCoord(pq + dq, pr + dr)
                if n not in grid:
                    recheck.add(n)

        # Re-classify each
        for p in recheck:
            c = _classify_placement(grid, p)
            if c == 0:  # WHITE
                pw.add(p)
                pb.discard(p)
            elif c == 1:  # BLACK
                pb.add(p)
                pw.discard(p)
            else:
                pw.discard(p)
                pb.discard(p)

        # Explicitly remove occupied positions from placement sets
        for p in positions:
            if p in grid:
                pw.discard(p)
                pb.discard(p)

    # == Notification ===================================================

    def notify_place(self, board: Board, piece: Piece, pos: HexCoord) -> None:
        """
        Notify the cache that a piece was placed at pos.

        Called AFTER board.place_piece() has executed.
        """
        if self._placement_initialized:
            self._update_placement_around(board, {pos})
        if self._occupied_tuples_valid:
            self._occupied_tuples.add((pos.q, pos.r))

    def notify_move(
        self, board: Board, piece: Piece, from_pos: HexCoord, to_pos: HexCoord
    ) -> None:
        """
        Notify the cache that a piece moved from from_pos to to_pos.

        Called AFTER board.move_piece() has executed.
        """
        if self._placement_initialized:
            self._update_placement_around(board, {from_pos, to_pos})
        if self._occupied_tuples_valid:
            if from_pos not in board.grid:
                self._occupied_tuples.discard((from_pos.q, from_pos.r))
            self._occupied_tuples.add((to_pos.q, to_pos.r))

    # == Ant Movement ===================================================

    def get_ant_moves(self, board: Board, start: HexCoord) -> set[HexCoord]:
        """
        Get ant move destinations using tuple-based BFS.

        Uses raw (q, r) tuples instead of HexCoord objects in the inner
        BFS loop, avoiding expensive HexCoord allocation (~500ns/object
        vs ~100ns/tuple). Results are converted to HexCoord at the end.

        This is ~4x faster than the HexCoord-based board._ant_bfs().
        """
        if not self._occupied_tuples_valid:
            self._init_occupied_tuples(board)
        return _ant_bfs_tuples(self._occupied_tuples, (start.q, start.r))

    # == Spider Movement ================================================

    def get_spider_moves(self, board: Board, start: HexCoord) -> set[HexCoord]:
        """
        Get spider 3-step destinations using tuple-based unrolled DFS.

        Replaces the recursive _spider_walk + can_slide() calls with 3
        nested loops that inline the slide checks using _SLIDE_OFFSETS
        tuple arithmetic. Avoids HexCoord allocation, recursive function
        calls, and Direction enum operations in the hot path.

        This is ~3-4x faster than the HexCoord-based board.generate_slides().
        """
        if not self._occupied_tuples_valid:
            self._init_occupied_tuples(board)
        return _spider_dfs_tuples(self._occupied_tuples, (start.q, start.r))

    def _init_occupied_tuples(self, board: Board) -> None:
        """Build occupied set from board grid."""
        self._occupied_tuples = {(pos.q, pos.r) for pos in board.grid}
        self._occupied_tuples_valid = True


# == Module-level helpers (avoid method dispatch overhead) ==============


def _classify_placement(grid: dict, pos: HexCoord) -> int:
    """
    Classify a position for placement validity.

    Returns:
        0 = valid for WHITE
        1 = valid for BLACK
        -1 = invalid (occupied, adjacent to both, or adjacent to none)

    A position is valid for exactly one color or neither (mutual exclusion).
    Adjacent to only white pieces => WHITE can place.
    Adjacent to only black pieces => BLACK can place.
    Adjacent to both or no pieces => neither can place.
    """
    has_white = False
    has_black = False
    pq, pr = pos.q, pos.r
    for dq, dr in _OFFSET_LIST:
        stack = grid.get(HexCoord(pq + dq, pr + dr))
        if stack:
            if stack[-1].color == Color.WHITE:
                has_white = True
            else:
                has_black = True
            if has_white and has_black:
                return -1  # Adjacent to both => invalid
    if has_white:
        return 0  # WHITE
    if has_black:
        return 1  # BLACK
    return -1  # No adjacent pieces


def _ant_bfs_tuples(
    occ: set[tuple[int, int]], start: tuple[int, int]
) -> set[HexCoord]:
    """
    Ant BFS using raw tuples instead of HexCoord objects.

    Avoids expensive HexCoord allocation in the inner loop
    (~500ns per object vs ~100ns per tuple). Results are converted
    to HexCoord only at the end.

    Returns set[HexCoord] for API compatibility.
    """
    results_t: set[tuple[int, int]] = set()
    visited = {start}
    frontier = [start]

    while frontier:
        cq, cr = frontier.pop()
        for dq, dr, cw_dq, cw_dr, ccw_dq, ccw_dr in _SLIDE_OFFSETS:
            dest = (cq + dq, cr + dr)
            if dest in occ and dest != start:
                continue
            cw = (cq + cw_dq, cr + cw_dr)
            ccw = (cq + ccw_dq, cr + ccw_dr)
            cw_occ = cw in occ and cw != start
            ccw_occ = ccw in occ and ccw != start
            if cw_occ and ccw_occ:
                continue  # Gate
            if not cw_occ and not ccw_occ:
                continue  # No contact
            if dest not in visited:
                visited.add(dest)
                results_t.add(dest)
                frontier.append(dest)

    return {HexCoord(q, r) for q, r in results_t}


def _spider_dfs_tuples(
    occ: set[tuple[int, int]], start: tuple[int, int]
) -> set[HexCoord]:
    """
    Spider 3-step DFS using raw tuples with unrolled loops.

    Since the spider always walks exactly 3 steps, the recursive DFS is
    replaced by 3 nested loops — one per step — with inlined slide checks
    from _SLIDE_OFFSETS. This avoids:
      - HexCoord allocation in the inner loops
      - Recursive function call overhead
      - Direction enum operations (clockwise/counter_clockwise)
      - can_slide() / is_gate_blocked() call overhead

    Path constraint (no revisiting within the current walk) is enforced
    by simple equality comparisons: at each step, skip positions already
    in {start, pos1} or {start, pos1, pos2}.

    The spider's starting position is treated as empty throughout
    (exclude_pos=start semantics), matching board._spider_walk behaviour.

    Returns set[HexCoord] for API compatibility.
    """
    results_t: set[tuple[int, int]] = set()
    sq, sr = start

    # ── Step 1: slide from start → pos1 ─────────────────────────────────
    for dq1, dr1, cw_dq1, cw_dr1, ccw_dq1, ccw_dr1 in _SLIDE_OFFSETS:
        # pos1 is always a distinct hex from start (dq1/dr1 are non-zero)
        p1q = sq + dq1
        p1r = sr + dr1
        pos1 = (p1q, p1r)
        if pos1 in occ:
            continue  # occupied (pos1 ≠ start always, so no exception needed)
        cw1 = (sq + cw_dq1, sr + cw_dr1)
        ccw1 = (sq + ccw_dq1, sr + ccw_dr1)
        cw1_occ = cw1 in occ and cw1 != start
        ccw1_occ = ccw1 in occ and ccw1 != start
        if cw1_occ and ccw1_occ:
            continue  # gate
        if not cw1_occ and not ccw1_occ:
            continue  # no contact

        # ── Step 2: slide from pos1 → pos2 ──────────────────────────────
        for dq2, dr2, cw_dq2, cw_dr2, ccw_dq2, ccw_dr2 in _SLIDE_OFFSETS:
            p2q = p1q + dq2
            p2r = p1r + dr2
            pos2 = (p2q, p2r)
            if pos2 in occ and pos2 != start:
                continue  # occupied (treat start as empty)
            if pos2 == start or pos2 == pos1:
                continue  # path constraint: {start, pos1}
            cw2 = (p1q + cw_dq2, p1r + cw_dr2)
            ccw2 = (p1q + ccw_dq2, p1r + ccw_dr2)
            cw2_occ = cw2 in occ and cw2 != start
            ccw2_occ = ccw2 in occ and ccw2 != start
            if cw2_occ and ccw2_occ:
                continue  # gate
            if not cw2_occ and not ccw2_occ:
                continue  # no contact

            # ── Step 3: slide from pos2 → pos3 (destination) ────────────
            for dq3, dr3, cw_dq3, cw_dr3, ccw_dq3, ccw_dr3 in _SLIDE_OFFSETS:
                p3q = p2q + dq3
                p3r = p2r + dr3
                pos3 = (p3q, p3r)
                if pos3 in occ and pos3 != start:
                    continue  # occupied (treat start as empty)
                if pos3 == start or pos3 == pos1 or pos3 == pos2:
                    continue  # path constraint: {start, pos1, pos2}
                cw3 = (p2q + cw_dq3, p2r + cw_dr3)
                ccw3 = (p2q + ccw_dq3, p2r + ccw_dr3)
                cw3_occ = cw3 in occ and cw3 != start
                ccw3_occ = ccw3 in occ and ccw3 != start
                if cw3_occ and ccw3_occ:
                    continue  # gate
                if not cw3_occ and not ccw3_occ:
                    continue  # no contact
                results_t.add(pos3)

    return {HexCoord(q, r) for q, r in results_t}


# == Legacy helpers (used by tests) ====================================


def _boundary_is_connected(boundary: set[HexCoord]) -> bool:
    """
    Check if boundary set forms a single connected component
    under hex adjacency.
    """
    if len(boundary) <= 1:
        return True
    start = next(iter(boundary))
    visited = {start}
    queue = [start]
    while queue:
        cq, cr = (current := queue.pop()).q, current.r
        for dq, dr in _OFFSET_LIST:
            n = HexCoord(cq + dq, cr + dr)
            if n in boundary and n not in visited:
                visited.add(n)
                queue.append(n)
    return len(visited) == len(boundary)


def _adjusted_boundary(
    board: Board, boundary: set[HexCoord], start: HexCoord
) -> set[HexCoord]:
    """Compute the boundary adjusted for treating start as empty."""
    adjusted = set(boundary)
    grid = board.grid
    sq, sr = start.q, start.r

    add_start = False
    for dq, dr in _OFFSET_LIST:
        n = HexCoord(sq + dq, sr + dr)
        if n in grid and n != start:
            add_start = True
            break
    if add_start:
        adjusted.add(start)

    for dq, dr in _OFFSET_LIST:
        n = HexCoord(sq + dq, sr + dr)
        if n not in grid:
            has_other = False
            nq, nr = n.q, n.r
            for dq2, dr2 in _OFFSET_LIST:
                nn = HexCoord(nq + dq2, nr + dr2)
                if nn != start and nn in grid:
                    has_other = True
                    break
            if not has_other:
                adjusted.discard(n)

    return adjusted


def _has_obstruction(
    board: Board, boundary: set[HexCoord], start: HexCoord
) -> bool:
    """Check if any slide between adjacent boundary positions is obstructed."""
    grid = board.grid
    for pos in boundary:
        pq, pr = pos.q, pos.r
        for dq, dr, cw_dq, cw_dr, ccw_dq, ccw_dr in _SLIDE_OFFSETS:
            dest = HexCoord(pq + dq, pr + dr)
            if dest not in boundary:
                continue
            cw = HexCoord(pq + cw_dq, pr + cw_dr)
            ccw = HexCoord(pq + ccw_dq, pr + ccw_dr)
            cw_occ = cw in grid and cw != start
            ccw_occ = ccw in grid and ccw != start
            if cw_occ and ccw_occ:
                return True  # Gate
            if not cw_occ and not ccw_occ:
                return True  # No contact
    return False
