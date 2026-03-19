"""
Hive board representation and movement mechanics.

The board is represented as a sparse mapping from HexCoord to stacks of pieces.
This naturally handles the dynamic, unbounded nature of the Hive board.

Key algorithms:
  - Articulation point detection (Tarjan's) for One Hive rule plus dynamic block cut tree for gradually updating articulation points over time, rather than recomputing
  - Gate detection for Freedom of Movement
  - Per-piece move generation (Queen, Ant, Grasshopper, Spider, Beetle,
    Mosquito, Ladybug, Pillbug)
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

# Pre-computed slide offsets: (dq, dr, cw_dq, cw_dr, ccw_dq, ccw_dr)
# For each direction, includes the direction itself plus its clockwise and
# counter-clockwise flanking directions (used for gate + contact checks).
_SLIDE_OFFSETS: tuple[tuple[int, int, int, int, int, int], ...] = (
    (+1, 0, 0, +1, +1, -1),   # E:  cw=SE, ccw=NE
    (+1, -1, +1, 0, 0, -1),   # NE: cw=E,  ccw=NW
    (0, -1, +1, -1, -1, 0),   # NW: cw=NE, ccw=W
    (-1, 0, 0, -1, -1, +1),   # W:  cw=NW, ccw=SW
    (-1, +1, -1, 0, 0, +1),   # SW: cw=W,  ccw=SE
    (0, +1, -1, +1, +1, 0),   # SE: cw=SW, ccw=E
)


class Board:
    """
    Sparse board representation for Hive.

    The board stores stacks of pieces at hex coordinates.
    Position (q, r) maps to a list of Piece objects (bottom to top).

    Attributes:
        grid: Mapping from HexCoord to list of pieces (stack, bottom-up).
        piece_positions: Mapping from Piece to its current HexCoord (or None if in hand).
    """

    __slots__ = (
        "grid", "piece_positions", "_ap_cache", "_ap_dirty",
        "_bct_blocks", "_bct_vertex_blocks", "_bct_cut_vertices", "_bct_valid",
    )

    def __init__(self) -> None:
        # HexCoord -> [Piece, ...] (bottom of stack first)
        self.grid: dict[HexCoord, list[Piece]] = {}
        # Piece -> HexCoord (only for placed pieces)
        self.piece_positions: dict[Piece, HexCoord] = {}
        # Articulation point cache (invalidated on mutation)
        self._ap_cache: set[HexCoord] | None = None
        self._ap_dirty: bool = True
        # Block-cut tree for incremental AP maintenance
        self._bct_blocks: list[set[HexCoord]] = []
        self._bct_vertex_blocks: dict[HexCoord, list[int]] = {}
        self._bct_cut_vertices: set[HexCoord] = set()
        self._bct_valid: bool = False

    def copy(self) -> Board:
        """Create a deep copy for MCTS rollouts."""
        b = Board()
        b.grid = {pos: stack[:] for pos, stack in self.grid.items()}
        b.piece_positions = dict(self.piece_positions)
        # Copy AP cache so copies start with valid AP data
        b._ap_cache = set(self._ap_cache) if self._ap_cache is not None else None
        b._ap_dirty = self._ap_dirty
        # Copy BCT data
        b._bct_blocks = [s.copy() for s in self._bct_blocks]
        b._bct_vertex_blocks = {k: list(v) for k, v in self._bct_vertex_blocks.items()}
        b._bct_cut_vertices = set(self._bct_cut_vertices)
        b._bct_valid = self._bct_valid
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
        topology_changed = pos not in self.grid  # New vertex in hex graph
        if topology_changed:
            self.grid[pos] = []
        self.grid[pos].append(piece)
        self.piece_positions[piece] = pos
        if topology_changed:
            # Try incremental BCT update for single-neighbor leaf insertion
            neighbors = [pos.neighbor(d) for d in ALL_DIRECTIONS
                         if pos.neighbor(d) in self.grid]
            if self._bct_valid and len(neighbors) == 1:
                self._bct_add_leaf(pos, neighbors[0])
                # AP cache updated in-place by _bct_add_leaf
            else:
                self._bct_valid = False
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
            # Count neighbors BEFORE removing from grid (for BCT update)
            degree = sum(1 for d in ALL_DIRECTIONS if pos.neighbor(d) in self.grid)
            del self.grid[pos]
            # Try incremental BCT update for leaf removal
            if self._bct_valid and degree <= 1:
                self._bct_remove_leaf(pos)
            else:
                self._bct_valid = False
                self._ap_dirty = True
        del self.piece_positions[piece]
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
        Find all articulation points in the hive graph.

        An articulation point is a position whose removal would disconnect the hive.
        Pieces at articulation points are "pinned" and cannot move (One Hive rule).

        Uses a block-cut tree for incremental maintenance when possible,
        falling back to Tarjan's DFS when the BCT is invalidated.
        """
        if exclude is not None:
            # The 'exclude' case is rare — always use Tarjan's directly
            return self._tarjan_ap(exclude)

        # Check AP cache first
        if not self._ap_dirty and self._ap_cache is not None:
            return self._ap_cache

        # Rebuild BCT if invalid, which also sets AP cache
        if not self._bct_valid:
            self._build_bct()

        return self._ap_cache  # type: ignore[return-value]

    def _tarjan_ap(self, exclude: Piece | None = None) -> set[HexCoord]:
        """
        Compute articulation points using iterative Tarjan's DFS.
        Used as fallback and for the exclude-piece case.
        """
        occupied = set(self.grid.keys())
        if exclude is not None:
            exclude_pos = self.piece_positions.get(exclude)
            if exclude_pos and len(self.grid.get(exclude_pos, [])) == 1:
                occupied = occupied - {exclude_pos}

        if len(occupied) <= 2:
            if exclude is None:
                self._ap_cache = set()
                self._ap_dirty = False
            return set()

        start = next(iter(occupied))

        disc: dict[HexCoord, int] = {}
        low: dict[HexCoord, int] = {}
        parent: dict[HexCoord, HexCoord | None] = {start: None}
        child_count: dict[HexCoord, int] = {}
        ap: set[HexCoord] = set()
        timer = 0

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
        stack: list[tuple[HexCoord, int]] = [(start, 0)]

        while stack:
            u, ni = stack[-1]
            neighbors = _neighbor_cache[u]

            if ni < len(neighbors):
                stack[-1] = (u, ni + 1)
                v = neighbors[ni]

                if v not in disc:
                    parent[v] = u
                    child_count[u] = child_count.get(u, 0) + 1
                    disc[v] = low[v] = timer
                    timer += 1
                    child_count[v] = 0
                    stack.append((v, 0))
                elif v != parent.get(u):
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    if low[u] < low[p]:
                        low[p] = low[u]
                    if parent[p] is not None and low[u] >= disc[p]:
                        ap.add(p)
                    if parent[p] is None and child_count.get(p, 0) > 1:
                        ap.add(p)

        if exclude is None:
            self._ap_cache = ap
            self._ap_dirty = False

        return ap

    # ── Block-Cut Tree ─────────────────────────────────────────

    def _build_bct(self) -> None:
        """
        Build the block-cut tree from scratch using Tarjan's algorithm
        to find biconnected components.

        A block-cut tree represents the biconnected component structure:
        - Each maximal biconnected component (block) is tracked
        - A vertex is an articulation point iff it belongs to 2+ blocks
        """
        self._bct_blocks = []
        self._bct_vertex_blocks = {}
        self._bct_cut_vertices = set()

        occupied = set(self.grid.keys())
        if len(occupied) <= 1:
            # 0 or 1 vertices: no blocks, no APs
            for v in occupied:
                self._bct_vertex_blocks[v] = []
            self._ap_cache = set()
            self._ap_dirty = False
            self._bct_valid = True
            return

        if len(occupied) == 2:
            # Exactly 2 vertices: one block, no APs
            block = set(occupied)
            self._bct_blocks.append(block)
            for v in occupied:
                self._bct_vertex_blocks[v] = [0]
            self._ap_cache = set()
            self._ap_dirty = False
            self._bct_valid = True
            return

        # Full Tarjan's with edge stack to identify biconnected components
        start = next(iter(occupied))

        disc: dict[HexCoord, int] = {}
        low: dict[HexCoord, int] = {}
        parent: dict[HexCoord, HexCoord | None] = {start: None}
        child_count: dict[HexCoord, int] = {}
        timer = 0

        # Pre-compute adjacency
        adj: dict[HexCoord, list[HexCoord]] = {}
        for pos in occupied:
            adj[pos] = [pos.neighbor(d) for d in ALL_DIRECTIONS
                        if pos.neighbor(d) in occupied]

        # Edge stack for biconnected component detection
        edge_stack: list[tuple[HexCoord, HexCoord]] = []
        blocks: list[set[HexCoord]] = []

        disc[start] = low[start] = timer
        timer += 1
        child_count[start] = 0
        dfs_stack: list[tuple[HexCoord, int]] = [(start, 0)]

        while dfs_stack:
            u, ni = dfs_stack[-1]
            neighbors = adj[u]

            if ni < len(neighbors):
                dfs_stack[-1] = (u, ni + 1)
                v = neighbors[ni]

                if v not in disc:
                    parent[v] = u
                    child_count[u] = child_count.get(u, 0) + 1
                    disc[v] = low[v] = timer
                    timer += 1
                    child_count[v] = 0
                    edge_stack.append((u, v))
                    dfs_stack.append((v, 0))
                elif v != parent.get(u) and disc[v] < disc[u]:
                    # Back edge (only push once per direction)
                    edge_stack.append((u, v))
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            else:
                dfs_stack.pop()
                if dfs_stack:
                    p = dfs_stack[-1][0]
                    if low[u] < low[p]:
                        low[p] = low[u]

                    # Check if p is an AP and extract biconnected component
                    is_root = parent[p] is None
                    is_ap = False
                    if is_root and child_count.get(p, 0) > 1:
                        is_ap = True
                    if not is_root and low[u] >= disc[p]:
                        is_ap = True

                    if (not is_root and low[u] >= disc[p]) or \
                       (is_root and child_count.get(p, 0) > 1):
                        # Pop edges from edge_stack to form a biconnected component
                        block: set[HexCoord] = set()
                        while edge_stack and edge_stack[-1] != (p, u):
                            e = edge_stack.pop()
                            block.add(e[0])
                            block.add(e[1])
                        if edge_stack:
                            e = edge_stack.pop()  # pop (p, u)
                            block.add(e[0])
                            block.add(e[1])
                        if block:
                            blocks.append(block)

        # Remaining edges on the stack form one last biconnected component
        if edge_stack:
            block = set()
            while edge_stack:
                e = edge_stack.pop()
                block.add(e[0])
                block.add(e[1])
            blocks.append(block)

        # Build vertex → blocks mapping
        vertex_blocks: dict[HexCoord, list[int]] = {v: [] for v in occupied}
        for i, blk in enumerate(blocks):
            for v in blk:
                vertex_blocks[v].append(i)

        # Cut vertices = vertices in 2+ blocks
        cut_vertices = {v for v, blks in vertex_blocks.items() if len(blks) >= 2}

        self._bct_blocks = blocks
        self._bct_vertex_blocks = vertex_blocks
        self._bct_cut_vertices = cut_vertices
        self._ap_cache = cut_vertices
        self._ap_dirty = False
        self._bct_valid = True

    def _bct_add_leaf(self, v: HexCoord, u: HexCoord) -> None:
        """
        Incrementally update BCT when a leaf vertex v is added,
        connected to exactly one existing neighbor u.

        Adding a leaf:
        - Creates a new block {u, v}
        - v is never an AP (leaf vertex)
        - u becomes an AP if it now belongs to 2+ blocks
        - No other vertex changes AP status
        """
        block_idx = len(self._bct_blocks)
        self._bct_blocks.append({u, v})

        # Update vertex → blocks mapping
        self._bct_vertex_blocks.setdefault(v, []).append(block_idx)
        self._bct_vertex_blocks.setdefault(u, []).append(block_idx)

        # u becomes AP if now in 2+ blocks (and graph has 3+ vertices)
        if len(self._bct_vertex_blocks[u]) >= 2 and len(self.grid) >= 3:
            self._bct_cut_vertices.add(u)

        # Update AP cache in-place
        self._ap_cache = self._bct_cut_vertices
        self._ap_dirty = False

    def _bct_remove_leaf(self, v: HexCoord) -> None:
        """
        Incrementally update BCT when a leaf vertex v is removed.

        v was in exactly 1 block with its single neighbor u.
        Removing v:
        - Removes v from its block
        - If u now has only 1 block, u is no longer an AP
        """
        block_indices = self._bct_vertex_blocks.pop(v, [])
        if not block_indices:
            return

        block_idx = block_indices[0]
        block = self._bct_blocks[block_idx]
        block.discard(v)

        # Find the neighbor u (the other vertex in this 2-vertex block)
        # If block is now singleton {u}, remove the block entry from u
        if len(block) <= 1:
            for u in block:
                u_blocks = self._bct_vertex_blocks.get(u, [])
                if block_idx in u_blocks:
                    u_blocks.remove(block_idx)
                # u is no longer AP if it has < 2 blocks
                if len(u_blocks) < 2:
                    self._bct_cut_vertices.discard(u)

        # Also check: if only 2 vertices left total, no APs possible
        if len(self.grid) <= 2:
            self._bct_cut_vertices.clear()

        self._ap_cache = self._bct_cut_vertices
        self._ap_dirty = False

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

        Uses exclude_pos instead of physically removing/replacing the piece,
        avoiding board mutation and AP cache invalidation.

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

        # Use exclude_pos=start so can_slide treats the piece's position as empty
        # (semantically identical to removing and re-placing the piece)
        if max_distance == 3:
            # Spider: exactly 3 steps, no revisiting
            self._spider_walk(start, start, 0, 3, [start], results, start)
        elif max_distance == 1:
            # Queen: 1 step
            for d in ALL_DIRECTIONS:
                if self.can_slide(start, d, exclude_pos=start):
                    dest = start.neighbor(d)
                    results.add(dest)
        else:
            # Ant: unlimited, inlined BFS
            return self._ant_bfs(start)

        return results

    def _ant_bfs(self, start: HexCoord) -> set[HexCoord]:
        """
        Optimized BFS for ant movement with inlined slide checks.

        Avoids per-step function call overhead of can_slide() + is_gate_blocked()
        by pre-computing all direction/flanking offsets and inlining the three
        slide conditions (destination empty, not gate-blocked, maintains contact).
        """
        grid = self.grid
        results: set[HexCoord] = set()
        visited = {start}
        frontier = [start]

        while frontier:
            cq, cr = (current := frontier.pop()).q, current.r
            for dq, dr, cw_dq, cw_dr, ccw_dq, ccw_dr in _SLIDE_OFFSETS:
                dest = HexCoord(cq + dq, cr + dr)
                # 1. Destination must be empty (treat start as empty)
                if dest in grid and dest != start:
                    continue
                # 2+3. Flanking positions for gate + contact check
                cw = HexCoord(cq + cw_dq, cr + cw_dr)
                ccw = HexCoord(cq + ccw_dq, cr + ccw_dr)
                cw_occ = cw in grid and cw != start
                ccw_occ = ccw in grid and ccw != start
                # Gate: both flanking occupied → blocked
                if cw_occ and ccw_occ:
                    continue
                # Contact: at least one flanking must be occupied
                if not cw_occ and not ccw_occ:
                    continue
                # Valid slide destination
                if dest not in visited:
                    visited.add(dest)
                    results.add(dest)
                    frontier.append(dest)

        return results

    def _spider_walk(
        self,
        current: HexCoord,
        start: HexCoord,
        depth: int,
        target_depth: int,
        path: list[HexCoord],
        results: set[HexCoord],
        exclude_pos: HexCoord | None = None,
    ) -> None:
        """Recursive DFS for spider's exactly-3-step walk."""
        if depth == target_depth:
            results.add(current)
            return

        for d in ALL_DIRECTIONS:
            if self.can_slide(current, d, exclude_pos=exclude_pos):
                dest = current.neighbor(d)
                if dest not in path:
                    path.append(dest)
                    self._spider_walk(dest, start, depth + 1, target_depth, path, results, exclude_pos)
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

        Uses arithmetic instead of physically removing/replacing the piece.
        """
        start = self.piece_positions[piece]
        results: set[HexCoord] = set()

        # Height after conceptual removal of beetle from top of stack
        start_height = self.stack_height(start) - 1

        for d in ALL_DIRECTIONS:
            dest = start.neighbor(d)
            dest_height = self.stack_height(dest)

            # Moving to elevated position or from elevated position
            move_height = max(dest_height, start_height)

            if move_height > 0:
                # Elevated move (climbing up, across top, or down)
                # Gate check for elevated moves: check flanking positions.
                # A gate blocks only when BOTH flanking neighbors are STRICTLY
                # TALLER than the movement height. Pieces at the same height
                # as the movement do not form a gate for elevated pieces —
                # the beetle travels above them.
                cw_pos = start.neighbor(d.clockwise())
                ccw_pos = start.neighbor(d.counter_clockwise())
                cw_height = self.stack_height(cw_pos)
                ccw_height = self.stack_height(ccw_pos)

                if cw_height > move_height and ccw_height > move_height:
                    continue  # Gate blocked at this height

                results.add(dest)
            else:
                # Ground-level slide (both start and dest are ground level)
                if self.can_slide(start, d, exclude_pos=start):
                    results.add(dest)

        return results

    # ── Expansion Piece Moves ────────────────────────────────────

    def _elevated_gate_blocked(
        self, from_pos: HexCoord, d: Direction, move_height: int,
    ) -> bool:
        """Check if elevated movement in direction d is blocked by flanking stacks.

        A gate blocks only when BOTH flanking neighbors have stack height
        STRICTLY GREATER than the movement height.
        """
        cw_pos = from_pos.neighbor(d.clockwise())
        ccw_pos = from_pos.neighbor(d.counter_clockwise())
        cw_h = self.stack_height(cw_pos)
        ccw_h = self.stack_height(ccw_pos)
        return cw_h > move_height and ccw_h > move_height

    def generate_ladybug_moves(self, piece: Piece) -> set[HexCoord]:
        """
        Generate ladybug destinations.

        The ladybug moves exactly 3 steps:
        1. Ascend onto an adjacent occupied hex
        2. Traverse to another adjacent occupied hex (on top)
        3. Descend to an adjacent empty hex

        Elevated gate checks apply at each step.
        """
        start = self.piece_positions[piece]
        results: set[HexCoord] = set()
        start_height = self.stack_height(start) - 1  # after removing ladybug

        for d1 in ALL_DIRECTIONS:
            n1 = start.neighbor(d1)
            if n1 not in self.grid:
                continue  # Step 1: must ascend onto occupied

            # Gate check: ascending from start to top of n1
            n1_height = self.stack_height(n1)
            move_h1 = max(n1_height, start_height)
            if self._elevated_gate_blocked(start, d1, move_h1):
                continue

            for d2 in ALL_DIRECTIONS:
                n2 = n1.neighbor(d2)
                if n2 == start or n2 not in self.grid:
                    continue  # Step 2: must traverse onto another occupied (not start)

                # Gate check: traversing from top of n1 to top of n2
                n2_height = self.stack_height(n2)
                move_h2 = max(n2_height, n1_height)
                if self._elevated_gate_blocked(n1, d2, move_h2):
                    continue

                for d3 in ALL_DIRECTIONS:
                    n3 = n2.neighbor(d3)
                    if n3 == start or n3 == n1:
                        continue  # No revisiting
                    if n3 in self.grid:
                        continue  # Step 3: must descend to empty

                    # Gate check: descending from top of n2 to empty n3
                    move_h3 = n2_height  # max(0, n2_height)
                    if self._elevated_gate_blocked(n2, d3, move_h3):
                        continue

                    results.add(n3)

        return results

    def generate_pillbug_moves(self, piece: Piece) -> set[HexCoord]:
        """
        Generate pillbug standard movement destinations.

        Standard movement: 1-step slide, identical to queen.
        The pillbug special ability (throwing adjacent pieces) is handled
        separately by generate_pillbug_throws().
        """
        return self.generate_slides(piece, max_distance=1)

    def generate_pillbug_throws(
        self, pillbug_pos: HexCoord, articulation_points: set[HexCoord],
    ) -> list[tuple[Piece, HexCoord]]:
        """
        Generate pillbug special ability moves (throws).

        The pillbug can pick up any adjacent top piece and place it in another
        adjacent empty space. The piece passes over the pillbug's height.

        Rules:
        - Target piece must be on top of its stack
        - Target position must NOT be an articulation point (unless stack > 1)
        - Gate checks apply for both lifting and placing
        - The pillbug itself can be an articulation point and still throw

        Args:
            pillbug_pos: Position of the pillbug (or mosquito acting as pillbug).
            articulation_points: Current articulation point set.

        Returns:
            List of (target_piece, destination) tuples.
        """
        results: list[tuple[Piece, HexCoord]] = []
        pillbug_height = self.stack_height(pillbug_pos)

        # Collect valid targets (adjacent top pieces that can be lifted)
        targets: list[tuple[Piece, HexCoord, Direction]] = []
        for d in ALL_DIRECTIONS:
            adj = pillbug_pos.neighbor(d)
            stack = self.grid.get(adj)
            if not stack:
                continue
            target_piece = stack[-1]  # top piece

            # Target position must not be AP (unless stack > 1)
            if adj in articulation_points and len(stack) == 1:
                continue

            # Gate check: lifting piece from adj over the pillbug
            # Move height = max(height after removing target, pillbug height)
            target_h_after = len(stack) - 1
            move_h_lift = max(target_h_after, pillbug_height)
            if self._elevated_gate_blocked(adj, d.opposite(), move_h_lift):
                continue

            targets.append((target_piece, adj, d))

        # For each valid target, find valid drop destinations
        for target_piece, target_pos, _target_dir in targets:
            for d_drop in ALL_DIRECTIONS:
                dest = pillbug_pos.neighbor(d_drop)
                if dest == target_pos:
                    continue  # Can't drop back where it was
                if dest in self.grid:
                    continue  # Must be empty

                # Gate check: lowering piece from pillbug height to empty dest
                move_h_drop = pillbug_height  # max(0, pillbug_height)
                if self._elevated_gate_blocked(pillbug_pos, d_drop, move_h_drop):
                    continue

                results.append((target_piece, dest))

        return results

    def generate_mosquito_moves(self, piece: Piece) -> set[HexCoord]:
        """
        Generate mosquito movement destinations.

        The mosquito copies the movement abilities of adjacent top pieces:
        - If elevated (on top of hive): acts as beetle only
        - If at ground level: union of all adjacent top piece movement types
        - Adjacent to only other mosquitos: cannot move
        - Does NOT copy pillbug special ability here (handled in game_state)
        """
        start = self.piece_positions[piece]

        # If elevated (on top of a stack), act as beetle
        if self.stack_height(start) > 1:
            return self.generate_beetle_moves(piece)

        # Collect piece types of adjacent top pieces (skip Mosquito)
        adj_types: set[PieceType] = set()
        for d in ALL_DIRECTIONS:
            n = start.neighbor(d)
            top = self.top_piece_at(n)
            if top is not None and top.piece_type != PieceType.MOSQUITO:
                adj_types.add(top.piece_type)

        if not adj_types:
            return set()  # Adjacent only to mosquitos or nothing

        results: set[HexCoord] = set()
        for pt in adj_types:
            if pt == PieceType.QUEEN or pt == PieceType.PILLBUG:
                results |= self.generate_slides(piece, max_distance=1)
            elif pt == PieceType.ANT:
                results |= self.generate_slides(piece, max_distance=-1)
            elif pt == PieceType.SPIDER:
                results |= self.generate_slides(piece, max_distance=3)
            elif pt == PieceType.GRASSHOPPER:
                results |= self.generate_grasshopper_moves(piece)
            elif pt == PieceType.BEETLE:
                results |= self.generate_beetle_moves(piece)
            elif pt == PieceType.LADYBUG:
                results |= self.generate_ladybug_moves(piece)

        return results

    # ── Piece Move Dispatch ──────────────────────────────────────

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
        elif pt == PieceType.MOSQUITO:
            return self.generate_mosquito_moves(piece)
        elif pt == PieceType.LADYBUG:
            return self.generate_ladybug_moves(piece)
        elif pt == PieceType.PILLBUG:
            return self.generate_pillbug_moves(piece)
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
