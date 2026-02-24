"""
Hive board representation and movement mechanics.

The board is represented as a sparse mapping from HexCoord to stacks of pieces.
This naturally handles the dynamic, unbounded nature of the Hive board.

Key algorithms:
  - Articulation point detection (Tarjan's) for One Hive rule
  - Gate detection for Freedom of Movement
  - Per-piece move generation (Queen, Ant, Grasshopper, Spider, Beetle)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

from hive_engine.hex_coord import (
    HexCoord,
    Direction,
    ALL_DIRECTIONS,
    ORIGIN,
    _OFFSET_LIST,
)
from hive_engine.pieces import Color, PieceType, Piece


class Board:
    """
    Sparse board representation for Hive.

    The board stores stacks of pieces at hex coordinates.
    Position (q, r) maps to a list of Piece objects (bottom to top).

    Attributes:
        grid: Mapping from HexCoord to list of pieces (stack, bottom-up).
        piece_positions: Mapping from Piece to its current HexCoord (or None if in hand).
    """

    __slots__ = ("grid", "piece_positions", "_ap_cache", "_ap_dirty")

    def __init__(self) -> None:
        # HexCoord -> [Piece, ...] (bottom of stack first)
        self.grid: dict[HexCoord, list[Piece]] = {}
        # Piece -> HexCoord (only for placed pieces)
        self.piece_positions: dict[Piece, HexCoord] = {}
        # Articulation point cache (invalidated on mutation)
        self._ap_cache: set[HexCoord] | None = None
        self._ap_dirty: bool = True

    def copy(self) -> Board:
        """Create a deep copy for MCTS rollouts."""
        b = Board()
        b.grid = {pos: stack[:] for pos, stack in self.grid.items()}
        b.piece_positions = dict(self.piece_positions)
        # Don't copy AP cache — copies diverge from original in MCTS
        b._ap_cache = None
        b._ap_dirty = True
        return b

    # ── Query ───────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return len(self.grid) == 0

    def occupied_positions(self) -> set[HexCoord]:
        """All positions with at least one piece."""
        return set(self.grid.keys())

    def top_piece_at(self, pos: HexCoord) -> Piece | None:
        """Return the top piece at a position, or None."""
        stack = self.grid.get(pos)
        if stack:
            return stack[-1]
        return None

    def stack_at(self, pos: HexCoord) -> list[Piece]:
        """Return the full stack at a position (empty list if unoccupied)."""
        return self.grid.get(pos, [])

    def stack_height(self, pos: HexCoord) -> int:
        """Height of the stack at pos (0 if empty)."""
        stack = self.grid.get(pos)
        return len(stack) if stack else 0

    def piece_height(self, piece: Piece) -> int:
        """Return the height (0-indexed) of a placed piece in its stack."""
        pos = self.piece_positions.get(piece)
        if pos is None:
            return -1
        stack = self.grid[pos]
        return stack.index(piece)

    def is_on_top(self, piece: Piece) -> bool:
        """Check if the piece is on top of its stack."""
        pos = self.piece_positions.get(piece)
        if pos is None:
            return False
        return self.grid[pos][-1] is piece

    def position_of(self, piece: Piece) -> HexCoord | None:
        """Return the position of a piece, or None if in hand."""
        return self.piece_positions.get(piece)

    def all_pieces_on_board(self) -> list[Piece]:
        """Return all pieces currently on the board."""
        return list(self.piece_positions.keys())

    def pieces_of_color(self, color: Color) -> list[Piece]:
        """Return all placed pieces belonging to a color."""
        return [p for p in self.piece_positions if p.color == color]

    def num_pieces_on_board(self) -> int:
        return len(self.piece_positions)

    # ── Mutate ──────────────────────────────────────────────────

    def place_piece(self, piece: Piece, pos: HexCoord) -> None:
        """Place a piece from hand onto the board at pos (on top of any stack)."""
        if pos not in self.grid:
            self.grid[pos] = []
        self.grid[pos].append(piece)
        self.piece_positions[piece] = pos
        self._ap_dirty = True

    def remove_piece(self, piece: Piece) -> HexCoord:
        """
        Remove a piece from the board. Returns its former position.
        The piece must be on top of its stack.
        """
        pos = self.piece_positions[piece]
        stack = self.grid[pos]
        assert stack[-1] is piece, "Can only remove the top piece"
        stack.pop()
        if not stack:
            del self.grid[pos]
        del self.piece_positions[piece]
        self._ap_dirty = True
        return pos

    def move_piece(self, piece: Piece, to: HexCoord) -> HexCoord:
        """
        Move a piece from its current position to a new position.
        Returns the old position.
        """
        old_pos = self.remove_piece(piece)
        self.place_piece(piece, to)
        return old_pos

    # ── Neighbor Queries ────────────────────────────────────────

    def occupied_neighbors(self, pos: HexCoord) -> list[tuple[Direction, HexCoord]]:
        """Return (direction, coord) pairs for all occupied neighbors."""
        result = []
        for d in ALL_DIRECTIONS:
            n = pos.neighbor(d)
            if n in self.grid:
                result.append((d, n))
        return result

    def num_occupied_neighbors(self, pos: HexCoord) -> int:
        """Count how many of the 6 neighbors are occupied."""
        grid = self.grid
        q, r = pos.q, pos.r
        count = 0
        for dq, dr in _OFFSET_LIST:
            if HexCoord(q + dq, r + dr) in grid:
                count += 1
        return count

    def empty_neighbors(self, pos: HexCoord) -> list[HexCoord]:
        """Return all unoccupied neighbors of pos."""
        return [pos.neighbor(d) for d in ALL_DIRECTIONS if pos.neighbor(d) not in self.grid]

    def adjacent_positions_with_pieces(self, pos: HexCoord) -> list[HexCoord]:
        """Return occupied neighbor positions."""
        return [pos.neighbor(d) for d in ALL_DIRECTIONS if pos.neighbor(d) in self.grid]

    # ── One Hive Rule ───────────────────────────────────────────

    def find_articulation_points(self, exclude: Piece | None = None) -> set[HexCoord]:
        """
        Find all articulation points in the hive graph using iterative Tarjan's.

        An articulation point is a position whose removal would disconnect the hive.
        Pieces at articulation points are "pinned" and cannot move (One Hive rule).

        Uses an iterative DFS to avoid Python recursion overhead.
        Results are cached and reused until the board is mutated.
        """
        # Return cached result for the common case (no exclude, cache valid)
        if exclude is None and not self._ap_dirty and self._ap_cache is not None:
            return self._ap_cache

        # Build adjacency from current board state
        occupied = set(self.grid.keys())
        if exclude is not None:
            exclude_pos = self.piece_positions.get(exclude)
            if exclude_pos and len(self.grid.get(exclude_pos, [])) == 1:
                occupied = occupied - {exclude_pos}

        if len(occupied) <= 2:
            return set()

        start = next(iter(occupied))

        disc: dict[HexCoord, int] = {}
        low: dict[HexCoord, int] = {}
        parent: dict[HexCoord, HexCoord | None] = {start: None}
        child_count: dict[HexCoord, int] = {}
        ap: set[HexCoord] = set()
        timer = 0

        # Iterative DFS using explicit stack
        # Stack entries: (node, neighbor_iterator, is_entering)
        # We pre-compute neighbor lists to avoid repeated neighbor() calls
        _neighbor_cache: dict[HexCoord, list[HexCoord]] = {}
        for pos in occupied:
            neighbors = []
            for d in ALL_DIRECTIONS:
                n = pos.neighbor(d)
                if n in occupied:
                    neighbors.append(n)
            _neighbor_cache[pos] = neighbors

        disc[start] = low[start] = timer
        timer += 1
        child_count[start] = 0
        stack: list[tuple[HexCoord, int]] = [(start, 0)]  # (node, neighbor_index)

        while stack:
            u, ni = stack[-1]
            neighbors = _neighbor_cache[u]

            if ni < len(neighbors):
                # Advance to next neighbor
                stack[-1] = (u, ni + 1)
                v = neighbors[ni]

                if v not in disc:
                    # Tree edge: v is unvisited
                    parent[v] = u
                    child_count[u] = child_count.get(u, 0) + 1
                    disc[v] = low[v] = timer
                    timer += 1
                    child_count[v] = 0
                    stack.append((v, 0))
                elif v != parent.get(u):
                    # Back edge
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            else:
                # Done with u, backtrack
                stack.pop()
                if stack:
                    p = stack[-1][0]  # parent of u
                    if low[u] < low[p]:
                        low[p] = low[u]
                    # Check if p is articulation point
                    if parent[p] is not None and low[u] >= disc[p]:
                        ap.add(p)
                    if parent[p] is None and child_count.get(p, 0) > 1:
                        ap.add(p)

        # Cache result for the common case (no exclude)
        if exclude is None:
            self._ap_cache = ap
            self._ap_dirty = False

        return ap

    def is_connected_without(self, piece: Piece) -> bool:
        """
        Check if the hive remains connected after removing `piece`.
        Used to validate the One Hive rule for movement.
        """
        pos = self.piece_positions.get(piece)
        if pos is None:
            return True

        # If piece is not on top, it can't move anyway
        if self.grid[pos][-1] is not piece:
            return True

        # If there's a piece underneath, removing top doesn't disconnect
        if len(self.grid[pos]) > 1:
            return True

        # Get all occupied positions excluding this one
        grid_keys = self.grid
        target_count = len(grid_keys) - 1
        if target_count <= 0:
            return True  # Only piece on the board

        # BFS from any remaining position (inlined for speed)
        start = None
        for k in grid_keys:
            if k is not pos and k != pos:
                start = k
                break
        if start is None:
            return True

        visited = {start}
        queue = [start]
        while queue:
            current = queue.pop()
            cq, cr = current.q, current.r
            for dq, dr in _OFFSET_LIST:
                n = HexCoord(cq + dq, cr + dr)
                if n != pos and n in grid_keys and n not in visited:
                    visited.add(n)
                    queue.append(n)

        return len(visited) == target_count

    # ── Gate Detection (Freedom of Movement) ────────────────────

    def is_gate_blocked(self, from_pos: HexCoord, direction: Direction,
                        exclude_pos: HexCoord | None = None) -> bool:
        """
        Check if movement from `from_pos` in `direction` is blocked by a gate.

        A gate exists when both neighbors flanking the movement direction
        are occupied (at ground level). The piece physically cannot slide through.

        Args:
            from_pos: Starting position.
            direction: Direction of movement.
            exclude_pos: A position to treat as empty (the moving piece's original pos).
        """
        cw = direction.clockwise()
        ccw = direction.counter_clockwise()

        cw_pos = from_pos.neighbor(cw)
        ccw_pos = from_pos.neighbor(ccw)

        def is_occupied(p: HexCoord) -> bool:
            if p == exclude_pos:
                return False
            return p in self.grid

        return is_occupied(cw_pos) and is_occupied(ccw_pos)

    def can_slide(self, from_pos: HexCoord, direction: Direction,
                  exclude_pos: HexCoord | None = None) -> bool:
        """
        Check if a ground-level piece can slide from `from_pos` in `direction`.

        Requirements:
        1. Destination must be empty
        2. Not gate-blocked
        3. Must maintain contact with hive (at least one of the two flanking
           neighbors must be occupied — ensures piece slides along the hive)
        """
        to_pos = from_pos.neighbor(direction)

        # Destination must be empty
        if to_pos in self.grid and to_pos != exclude_pos:
            return False

        # Gate check
        if self.is_gate_blocked(from_pos, direction, exclude_pos):
            return False

        # Must maintain contact: the piece must remain adjacent to the hive
        # while sliding. At least one of the two flanking positions must be occupied.
        cw = direction.clockwise()
        ccw = direction.counter_clockwise()
        cw_pos = from_pos.neighbor(cw)
        ccw_pos = from_pos.neighbor(ccw)

        def is_occupied(p: HexCoord) -> bool:
            if p == exclude_pos:
                return False
            return p in self.grid

        return is_occupied(cw_pos) or is_occupied(ccw_pos)

    # ── Movement Generation ─────────────────────────────────────

    def generate_slides(self, piece: Piece, max_distance: int = -1) -> set[HexCoord]:
        """
        Generate all valid slide destinations for a ground-level piece.

        Args:
            piece: The piece to move.
            max_distance: Maximum slide distance (-1 = unlimited, used for Ant).
                         For Queen, max_distance=1. For Spider, max_distance=3.

        Returns:
            Set of valid destination positions.
        """
        start = self.piece_positions[piece]
        results: set[HexCoord] = set()

        if max_distance == 0:
            return results

        # Temporarily remove the piece so it doesn't block its own slides
        self.remove_piece(piece)

        if max_distance == 3:
            # Spider: exactly 3 steps, no revisiting
            self._spider_walk(start, start, 0, 3, [start], results)
        elif max_distance == 1:
            # Queen: 1 step
            for d in ALL_DIRECTIONS:
                if self.can_slide(start, d, exclude_pos=None):
                    dest = start.neighbor(d)
                    results.add(dest)
        else:
            # Ant: unlimited, BFS
            visited = {start}
            frontier = [start]
            while frontier:
                current = frontier.pop()
                for d in ALL_DIRECTIONS:
                    if self.can_slide(current, d, exclude_pos=None):
                        dest = current.neighbor(d)
                        if dest not in visited:
                            visited.add(dest)
                            results.add(dest)
                            frontier.append(dest)

        # Put the piece back
        self.place_piece(piece, start)
        return results

    def _spider_walk(
        self,
        current: HexCoord,
        start: HexCoord,
        depth: int,
        target_depth: int,
        path: list[HexCoord],
        results: set[HexCoord],
    ) -> None:
        """Recursive DFS for spider's exactly-3-step walk."""
        if depth == target_depth:
            results.add(current)
            return

        for d in ALL_DIRECTIONS:
            if self.can_slide(current, d, exclude_pos=None):
                dest = current.neighbor(d)
                if dest not in path:
                    path.append(dest)
                    self._spider_walk(dest, start, depth + 1, target_depth, path, results)
                    path.pop()

    def generate_grasshopper_moves(self, piece: Piece) -> set[HexCoord]:
        """
        Generate grasshopper destinations.

        The grasshopper jumps in a straight line over one or more pieces,
        landing in the first empty space.
        """
        start = self.piece_positions[piece]
        results: set[HexCoord] = set()

        for d in ALL_DIRECTIONS:
            # Must start by jumping over at least one piece
            pos = start.neighbor(d)
            if pos not in self.grid:
                continue
            # Keep going in the same direction until we hit an empty space
            while pos in self.grid:
                pos = pos.neighbor(d)
            results.add(pos)

        return results

    def generate_beetle_moves(self, piece: Piece) -> set[HexCoord]:
        """
        Generate beetle destinations.

        The beetle can:
        1. Slide one space like the queen (if at ground level and no piece on top)
        2. Climb onto an adjacent occupied hex (going up)
        3. Move along the top of the hive (if already on top of pieces)
        4. Climb down from the hive
        """
        start = self.piece_positions[piece]
        results: set[HexCoord] = set()

        my_height = self.piece_height(piece)
        is_elevated = my_height > 0

        # Temporarily remove the piece
        self.remove_piece(piece)

        for d in ALL_DIRECTIONS:
            dest = start.neighbor(d)
            dest_height = self.stack_height(dest)
            start_height = self.stack_height(start)  # After removal

            # Moving to elevated position or from elevated position
            move_height = max(dest_height, start_height)

            if move_height > 0:
                # Elevated move (climbing up, across top, or down)
                # Gate check for elevated moves: check flanking positions
                # At height, gate is blocked if BOTH flanking neighbors have
                # stacks at least as high as the movement height
                cw_pos = start.neighbor(d.clockwise())
                ccw_pos = start.neighbor(d.counter_clockwise())
                cw_height = self.stack_height(cw_pos)
                ccw_height = self.stack_height(ccw_pos)

                if cw_height >= move_height and ccw_height >= move_height:
                    continue  # Gate blocked at this height

                results.add(dest)
            else:
                # Ground-level slide (both start and dest are ground level)
                if self.can_slide(start, d, exclude_pos=None):
                    results.add(dest)

        # Put the piece back
        self.place_piece(piece, start)
        return results

    def generate_piece_moves(self, piece: Piece) -> set[HexCoord]:
        """
        Generate all valid move destinations for a placed piece.

        This is the main dispatch that applies piece-specific movement rules.
        Assumes the piece is already confirmed as movable (not pinned, on top, etc.).
        """
        pt = piece.piece_type

        if pt == PieceType.QUEEN:
            return self.generate_slides(piece, max_distance=1)
        elif pt == PieceType.ANT:
            return self.generate_slides(piece, max_distance=-1)
        elif pt == PieceType.SPIDER:
            return self.generate_slides(piece, max_distance=3)
        elif pt == PieceType.GRASSHOPPER:
            return self.generate_grasshopper_moves(piece)
        elif pt == PieceType.BEETLE:
            return self.generate_beetle_moves(piece)
        else:
            return set()

    # ── Placement Positions ─────────────────────────────────────

    def valid_placement_positions(self, color: Color) -> set[HexCoord]:
        """
        Find all positions where a player can place a new piece from hand.

        Rules:
        - Must be adjacent to at least one friendly piece.
        - Must NOT be adjacent to any enemy piece.
        - Exception: On the very first move for each player (move 0 and move 1),
          the adjacency-to-enemy rule is relaxed. This is handled in GameState.
        """
        if self.is_empty():
            return {ORIGIN}

        friendly_adjacent: set[HexCoord] = set()
        for piece, pos in self.piece_positions.items():
            if piece.color == color:
                for n in pos.neighbors():
                    if n not in self.grid:
                        friendly_adjacent.add(n)

        # Remove positions adjacent to enemy pieces
        enemy_color = color.other()
        valid = set()
        for pos in friendly_adjacent:
            adjacent_to_enemy = False
            for n in pos.neighbors():
                stack = self.grid.get(n)
                if stack and stack[-1].color == enemy_color:
                    adjacent_to_enemy = True
                    break
            if not adjacent_to_enemy:
                valid.add(pos)

        return valid

    # ── Serialization ───────────────────────────────────────────

    def canonical_hash(self) -> int:
        """
        Compute a translation-invariant hash of the board state.

        Normalizes the board so the minimum position is at the origin,
        then hashes the resulting piece configuration.
        """
        if not self.grid:
            return hash(())

        # Find minimum position for translation normalization
        min_q = min(p.q for p in self.grid)
        min_r = min(p.r for p in self.grid)

        # Build a canonical representation
        canonical = []
        for pos in sorted(self.grid.keys()):
            nq = pos.q - min_q
            nr = pos.r - min_r
            stack = self.grid[pos]
            for i, piece in enumerate(stack):
                canonical.append((nq, nr, i, piece.piece_type.value,
                                  piece.color.value, piece.piece_id))

        return hash(tuple(canonical))

    def to_dict(self) -> dict:
        """Serialize board state to a dictionary."""
        result = {}
        for pos, stack in self.grid.items():
            key = (pos.q, pos.r)
            result[key] = [(p.piece_type.value, p.color.value, p.piece_id) for p in stack]
        return result

    def __repr__(self) -> str:
        lines = []
        for pos in sorted(self.grid.keys()):
            stack = self.grid[pos]
            stack_str = ", ".join(repr(p) for p in stack)
            lines.append(f"  ({pos.q:+d}, {pos.r:+d}): [{stack_str}]")
        return "Board{\n" + "\n".join(lines) + "\n}"
