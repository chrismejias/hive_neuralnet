"""
pygame GUI for playing Hive against a trained AI.

Run from hive_neuralnet/:
    python gui.py
    python gui.py --checkpoint checkpoints_gpu/hive_gpu_checkpoint_0068.pt
    python gui.py --color black --simulations 200

Controls:
    Click hand panel       → select piece type to place
    Click board piece      → select piece to move
    Click green hex        → confirm placement / movement
    Click elsewhere        → cancel selection
    Right-click + drag     → pan the board
    Middle-click + drag    → pan the board
    Scroll wheel           → zoom in/out (future)
    R                      → restart game
    C                      → re-center camera
    Escape                 → quit
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import threading
from enum import Enum, auto

import numpy as np
import pygame
import torch

from hive_engine.game_state import GameState, GameResult, Move, MoveType
from hive_engine.mcts import MCTS, MCTSConfig
from hive_engine.pieces import Color, PieceType, ExpansionConfig
from hive_engine.hex_coord import HexCoord

try:
    from hive_gnn.gnn_encoder import GNNEncoder
    from hive_gnn.gnn_net import GNNNetConfig, HiveGNN
except ImportError:
    GNNEncoder = HiveGNN = GNNNetConfig = None  # type: ignore

try:
    from hive_nnue.nnue_encoder import NNUEEncoder
    from hive_nnue.nnue_net import NNUEConfig, HiveNNUE
except ImportError:
    NNUEEncoder = HiveNNUE = NNUEConfig = None  # type: ignore

from hive_transformer.transformer_encoder import TransformerEncoder
from hive_transformer.transformer_net import TransformerConfig, HiveTransformer

try:
    from hive_prs.prs_transformer import PRSConfig
    from hive_prs.prs_transformer_v2 import HivePRSTransformerV2
    from hive_prs.prs_mcts_orchestrator_v2 import PRSMCTSConfigV2, PRSMCTSOrchestratorV2
    _HAS_PRS = True
except ImportError:
    PRSConfig = HivePRSTransformerV2 = PRSMCTSConfigV2 = PRSMCTSOrchestratorV2 = None  # type: ignore
    _HAS_PRS = False

try:
    from hive_fnn.fnn_network import FNNConfig, HiveFNN
    from hive_fnn.fnn_mcts_orchestrator import FNNMCTSConfig, FNNMCTSOrchestrator
    _HAS_FNN = True
except ImportError:
    FNNConfig = HiveFNN = FNNMCTSConfig = FNNMCTSOrchestrator = None  # type: ignore
    _HAS_FNN = False


# ── Layout constants ────────────────────────────────────────────────

WIN_W, WIN_H = 1120, 820
STATUS_H = 58      # top status bar
PANEL_W  = 190     # each side hand panel
BOARD_X  = PANEL_W
BOARD_Y  = STATUS_H
BOARD_W  = WIN_W - 2 * PANEL_W
BOARD_H  = WIN_H - STATUS_H
BOARD_CX = BOARD_X + BOARD_W // 2
BOARD_CY = BOARD_Y + BOARD_H // 2

HEX_SIZE = 40  # flat-top hex, pixels from center to corner

# ── Color palette ───────────────────────────────────────────────────

C_BG         = ( 26,  28,  36)
C_STATUS_BG  = ( 18,  20,  26)
C_PANEL_BG   = ( 33,  36,  46)
C_BOARD_BG   = ( 40,  44,  56)
C_HEX_EMPTY  = ( 50,  56,  70)
C_HEX_BORDER = ( 68,  78,  96)
C_HEX_HOVER  = ( 62,  70,  88)
C_W_PIECE    = (228, 222, 206)   # white piece fill
C_B_PIECE    = ( 44,  48,  60)   # black piece fill
C_W_TEXT     = ( 24,  24,  30)   # text on white piece
C_B_TEXT     = (210, 215, 228)   # text on black piece
C_SELECTED   = (255, 195,  40)   # selected piece highlight
C_LEGAL      = ( 76, 188,  96)   # legal destination
C_LAST_MOVE  = ( 96, 150, 255)   # last move highlight
C_THINKING   = (205, 135,  55)
C_WIN_W      = (255, 240, 160)
C_WIN_B      = ( 90, 100, 215)
C_DRAW       = (150, 158, 168)
C_QUEEN_WARN = (215,  55,  55)
C_PANEL_TEXT = (165, 175, 198)
C_TITLE      = (218, 224, 242)
C_DIM        = ( 95, 108, 130)
C_ROW_NORMAL = ( 40,  44,  58)
C_ROW_HOVER  = ( 48,  54,  70)
C_ROW_SEL    = ( 48, 118,  64)

# ── Piece metadata ──────────────────────────────────────────────────

PIECE_SYM = {
    PieceType.QUEEN:       "Q",
    PieceType.ANT:         "A",
    PieceType.GRASSHOPPER: "G",
    PieceType.SPIDER:      "S",
    PieceType.BEETLE:      "B",
    PieceType.MOSQUITO:    "M",
    PieceType.LADYBUG:     "L",
    PieceType.PILLBUG:     "P",
}

PIECE_NAME = {
    PieceType.QUEEN:       "Queen Bee",
    PieceType.ANT:         "Soldier Ant",
    PieceType.GRASSHOPPER: "Grasshopper",
    PieceType.SPIDER:      "Spider",
    PieceType.BEETLE:      "Beetle",
    PieceType.MOSQUITO:    "Mosquito",
    PieceType.LADYBUG:     "Ladybug",
    PieceType.PILLBUG:     "Pillbug",
}

PIECE_ORDER = [
    PieceType.QUEEN,
    PieceType.ANT,
    PieceType.GRASSHOPPER,
    PieceType.SPIDER,
    PieceType.BEETLE,
    PieceType.MOSQUITO,
    PieceType.LADYBUG,
    PieceType.PILLBUG,
]


# ── Hex geometry ────────────────────────────────────────────────────

def axial_to_pixel(q: int, r: int, cam_x: float = 0, cam_y: float = 0) -> tuple[int, int]:
    """Flat-top axial → pixel center, with camera offset."""
    x = BOARD_CX + cam_x + HEX_SIZE * 1.5 * q
    y = BOARD_CY + cam_y + HEX_SIZE * math.sqrt(3) * (r + q * 0.5)
    return int(x), int(y)


def pixel_to_axial(px: int, py: int, cam_x: float = 0, cam_y: float = 0) -> tuple[float, float]:
    """Pixel → fractional flat-top axial coords, with camera offset."""
    dx = px - BOARD_CX - cam_x
    dy = py - BOARD_CY - cam_y
    q = (2 / 3 * dx) / HEX_SIZE
    r = (-1 / 3 * dx + math.sqrt(3) / 3 * dy) / HEX_SIZE
    return q, r


def hex_round(q: float, r: float) -> tuple[int, int]:
    """Round fractional axial to nearest hex using cube rounding."""
    s = -q - r
    rq, rr, rs = round(q), round(r), round(s)
    dq = abs(rq - q)
    dr = abs(rr - r)
    ds = abs(rs - s)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return rq, rr


def flat_hex_corners(
    cx: float, cy: float, size: float
) -> list[tuple[float, float]]:
    """Six corners of a flat-top hexagon."""
    return [
        (
            cx + size * math.cos(math.radians(60 * i)),
            cy + size * math.sin(math.radians(60 * i)),
        )
        for i in range(6)
    ]


def in_board_viewport(x: int, y: int) -> bool:
    """Return True if the pixel position is within the board viewport."""
    margin = HEX_SIZE
    return (
        BOARD_X - margin < x < BOARD_X + BOARD_W + margin
        and BOARD_Y - margin < y < BOARD_Y + BOARD_H + margin
    )


# ── Win-in-one detection ────────────────────────────────────────────

def find_immediate_win(game: GameState) -> Move | None:
    """Return an immediately winning move if one exists, else None.

    Only considers moves that could plausibly complete a queen surround:
      - Any move whose destination is adjacent to the enemy queen
      - Any pillbug move that relocates the enemy queen itself
    Handles enemy queens buried under beetles (queen position still matters).
    """
    current = game.current_player
    enemy = Color.BLACK if current == Color.WHITE else Color.WHITE

    if not game._queen_placed[enemy]:
        return None  # Enemy queen not on board yet

    # Find enemy queen position — it may be buried under a beetle stack
    enemy_queen_pos: HexCoord | None = None
    for pos, stack in game.board.grid.items():
        for piece in stack:
            if piece.piece_type == PieceType.QUEEN and piece.color == enemy:
                enemy_queen_pos = pos
                break
        if enemy_queen_pos is not None:
            break

    if enemy_queen_pos is None:
        return None

    enemy_queen_neighbors = set(enemy_queen_pos.neighbors())
    current_surrounding = game.board.num_occupied_neighbors(enemy_queen_pos)

    for move in game.legal_moves():
        if move.move_type == MoveType.PASS:
            continue

        should_check = False

        if move.to in enemy_queen_neighbors:
            # Moving FROM another queen-neighbor to a queen-neighbor: net change = 0,
            # can never complete the surround in one step → skip.
            from_is_queen_nbr = (
                move.move_type == MoveType.MOVE
                and move.from_pos in enemy_queen_neighbors
            )
            if from_is_queen_nbr:
                continue
            # Otherwise this adds one more piece adjacent to enemy queen.
            # We need exactly 5 already surrounding to win with this one move.
            if current_surrounding >= 5:
                should_check = True

        elif (
            move.move_type == MoveType.MOVE
            and move.piece is not None
            and move.piece.piece_type == PieceType.QUEEN
            and move.piece.color == enemy
        ):
            # Pillbug relocating the enemy queen — check if new position is surrounded
            should_check = True

        if should_check:
            test = game.copy()
            test.apply_move(move)
            r = test.result
            if r in (GameResult.WHITE_WINS, GameResult.BLACK_WINS):
                return move

    return None


# ── GPU AI move conversion ──────────────────────────────────────────

# GPU board uses 23×23 cells: cell = (r+11)*23 + (q+11)
_GPU_BOARD_SIZE = 23
_GPU_HALF       = 11

# CPU PieceType → GPU PieceType (GPU PT_EMPTY=0, so CPU+1)
_CPU_TO_GPU_PT: dict[PieceType, int] = {
    PieceType.QUEEN:       1,
    PieceType.ANT:         2,
    PieceType.GRASSHOPPER: 3,
    PieceType.SPIDER:      4,
    PieceType.BEETLE:      5,
    PieceType.MOSQUITO:    6,
    PieceType.LADYBUG:     7,
    PieceType.PILLBUG:     8,
}
_GPU_TO_CPU_PT: dict[int, PieceType] = {v: k for k, v in _CPU_TO_GPU_PT.items()}


def _cpu_move_to_gpu_bytes(move: Move) -> bytes:
    """Convert a CPU Move to the 6-byte GPU move format."""
    if move.move_type == MoveType.PASS:
        return struct.pack("<BBHH", 2, 0, 0, 0)
    gpu_pt = _CPU_TO_GPU_PT[move.piece.piece_type]
    to_cell = (move.to.r + _GPU_HALF) * _GPU_BOARD_SIZE + (move.to.q + _GPU_HALF)
    if move.move_type == MoveType.PLACE:
        return struct.pack("<BBHH", 0, gpu_pt, 0, to_cell)
    from_cell = (move.from_pos.r + _GPU_HALF) * _GPU_BOARD_SIZE + (move.from_pos.q + _GPU_HALF)
    return struct.pack("<BBHH", 1, gpu_pt, from_cell, to_cell)


def _gpu_bytes_to_cpu_move(move_bytes: bytes | np.ndarray, game: GameState) -> Move | None:
    """Convert 6-byte GPU move to a CPU Move by matching legal moves.

    Returns None if no matching legal move is found.
    """
    if isinstance(move_bytes, np.ndarray):
        m = move_bytes
        m_type   = int(m[0])
        gpu_pt   = int(m[1])
        m_from   = int(m[2]) | (int(m[3]) << 8)
        m_to     = int(m[4]) | (int(m[5]) << 8)
    else:
        m_type, gpu_pt = struct.unpack_from("BB", move_bytes, 0)
        m_from, m_to   = struct.unpack_from("<HH", move_bytes, 2)

    if m_type == 2:  # PASS
        for mv in game.legal_moves():
            if mv.move_type == MoveType.PASS:
                return mv
        return None

    to_q = m_to % _GPU_BOARD_SIZE - _GPU_HALF
    to_r = m_to // _GPU_BOARD_SIZE - _GPU_HALF
    to_pos = HexCoord(to_q, to_r)

    cpu_pt = _GPU_TO_CPU_PT.get(gpu_pt)

    if m_type == 0:  # PLACE
        # Match by destination and piece type
        for mv in game.legal_moves():
            if (mv.move_type == MoveType.PLACE
                    and mv.to == to_pos
                    and mv.piece is not None
                    and mv.piece.piece_type == cpu_pt):
                return mv
    else:  # MOVE
        from_q = m_from % _GPU_BOARD_SIZE - _GPU_HALF
        from_r = m_from // _GPU_BOARD_SIZE - _GPU_HALF
        from_pos = HexCoord(from_q, from_r)
        for mv in game.legal_moves():
            if (mv.move_type == MoveType.MOVE
                    and mv.from_pos == from_pos
                    and mv.to == to_pos):
                return mv

    # Fallback: any legal move (should not happen in practice)
    legal = game.legal_moves()
    if legal:
        return legal[0]
    return None


# ── GPU AI wrapper ──────────────────────────────────────────────────

class GpuAI:
    """Wraps a GPU MCTS orchestrator (FNN or PRS v2) for single-move selection.

    Maintains a persistent GPU HiveState that mirrors the CPU GameState.
    """

    def __init__(self, orchestrator, expansion_mask: int = 0) -> None:
        import hive_gpu
        self.orch           = orchestrator
        self.ext            = hive_gpu.load_extension()
        self.expansion_mask = expansion_mask
        self.gpu_state      = self.ext.create_initial_states(1, expansion_mask)

    def apply_cpu_move(self, move: Move) -> None:
        """Mirror a CPU move onto the GPU state."""
        mb = _cpu_move_to_gpu_bytes(move)
        t  = torch.tensor(list(mb), dtype=torch.uint8, device="cuda").unsqueeze(0)
        self.ext.apply_moves_batch(self.gpu_state, t, 1)

    def select_move(self, game: GameState) -> Move:
        """Run GPU Gumbel MCTS and return the chosen CPU Move."""
        move_bytes = self.orch.select_move_gpu(self.gpu_state)   # (6,) uint8 CPU
        move = _gpu_bytes_to_cpu_move(move_bytes.numpy(), game)
        if move is None:
            legal = game.legal_moves()
            move  = legal[0] if legal else Move(MoveType.PASS, None, HexCoord(0, 0))
        return move

    def reset(self) -> None:
        """Reinitialize the GPU state to the starting position."""
        self.gpu_state = self.ext.create_initial_states(1, self.expansion_mask)


# ── Selection state ─────────────────────────────────────────────────

class SelectMode(Enum):
    IDLE        = auto()
    HAND_PIECE  = auto()   # piece type chosen from hand → show placements
    BOARD_PIECE = auto()   # board piece chosen → show movements


# ── GPU validator ──────────────────────────────────────────────────

class GPUValidator:
    """Keeps a GPU HiveState in sync with the CPU GameState and compares legal moves."""

    def __init__(self):
        import hive_gpu
        self.ext = hive_gpu.load_extension()
        self.gpu_states = self.ext.create_initial_states(1)
        self.gpu_move_count: int | None = None
        self.cpu_move_count: int | None = None
        self.mismatch = False
        self.error: str | None = None

    def _cpu_move_to_gpu_bytes(self, move: Move) -> bytes:
        """Convert a CPU Move to the 6-byte GPUMove struct."""
        return _cpu_move_to_gpu_bytes(move)

    def apply_move(self, move: Move) -> None:
        """Apply a move to the GPU state and update validation status."""
        try:
            move_bytes = self._cpu_move_to_gpu_bytes(move)
            move_tensor = torch.tensor(
                list(move_bytes), dtype=torch.uint8, device="cuda"
            ).unsqueeze(0)
            self.ext.apply_moves_batch(self.gpu_states, move_tensor, 1)
            self.error = None
        except Exception as e:
            self.error = str(e)

    def validate(self, cpu_game: GameState) -> None:
        """Compare GPU vs CPU legal move counts for the current position."""
        try:
            cpu_moves = cpu_game.legal_moves()
            self.cpu_move_count = len(cpu_moves)

            _, num_legal = self.ext.generate_legal_moves_batch(self.gpu_states, 1)
            self.gpu_move_count = num_legal[0].item()

            self.mismatch = self.cpu_move_count != self.gpu_move_count
            self.error = None
        except Exception as e:
            self.error = str(e)
            self.mismatch = True

    def reset(self) -> None:
        """Reset GPU state to initial position."""
        self.gpu_states = self.ext.create_initial_states(1)
        self.gpu_move_count = None
        self.cpu_move_count = None
        self.mismatch = False
        self.error = None

    @property
    def status_text(self) -> str:
        if self.error:
            return f"GPU ERROR: {self.error}"
        if self.gpu_move_count is None:
            return "GPU: ready"
        if self.mismatch:
            return f"GPU: {self.gpu_move_count} moves  CPU: {self.cpu_move_count}  MISMATCH!"
        return f"GPU: {self.gpu_move_count} moves  CPU: {self.cpu_move_count}  OK"


# ── GUI class ───────────────────────────────────────────────────────

class HiveGUI:
    """pygame-based Hive board for human vs AI."""

    # ── Init ───────────────────────────────────────────────────────

    def __init__(
        self,
        net,
        encoder,
        mcts_config: MCTSConfig,
        human_color: Color,
        cpu_player=None,
        self_play: bool = False,
        gpu_validate: bool = False,
        expansions: ExpansionConfig | None = None,
        gpu_ai: GpuAI | None = None,
    ) -> None:
        # GPU AI path (FNN / PRS v2): net/encoder/mcts_config unused
        self.gpu_ai = gpu_ai
        self.net = net
        self.encoder = encoder
        self.mcts_config = mcts_config
        self.human_color = human_color
        self.cpu_player = cpu_player
        self.self_play = self_play  # True = no AI, human controls both sides
        self.expansions = expansions
        self.gpu_validator: GPUValidator | None = None
        if gpu_validate:
            self.gpu_validator = GPUValidator()
            self.gpu_validator.validate(GameState(expansions=self.expansions))

        # Game
        self.game = GameState(expansions=self.expansions)
        self.move_number = 0
        self.last_move: Move | None = None

        # Selection
        self.mode = SelectMode.IDLE
        self.selected_piece_type: PieceType | None = None
        self.selected_board_hex: HexCoord | None = None
        self.legal_destinations: set[HexCoord] = set()

        # AI thread
        self._game_gen = 0          # incremented on restart to discard stale results
        self.ai_thinking = False
        self._ai_result: tuple[Move, int] | None = None   # (move, game_gen)
        self._ai_lock = threading.Lock()

        # Camera (pan offset in pixels)
        self.cam_x: float = 0.0
        self.cam_y: float = 0.0
        self._panning = False
        self._pan_start: tuple[int, int] | None = None

        # Hover
        self.hover_hex: HexCoord | None = None

        # Fonts (set in run())
        self.f_large:  pygame.font.Font
        self.f_med:    pygame.font.Font
        self.f_small:  pygame.font.Font
        self.f_status: pygame.font.Font

    @property
    def _human(self) -> Color:
        """Active human color: current player in self-play, fixed color otherwise."""
        return self.game.current_player if self.self_play else self.human_color

    # ── AI thread ──────────────────────────────────────────────────

    def _start_ai_think(self) -> None:
        self.ai_thinking = True
        gen = self._game_gen
        game_snapshot = self.game.copy()

        if self.gpu_ai is not None:
            # ── GPU Gumbel MCTS path (FNN / PRS v2) ──────────────────
            game_snapshot = self.game   # safe to read from thread (no mutation)
            gpu_ai        = self.gpu_ai

            def gpu_worker() -> None:
                # CPU immediate-win check (fast) before paying full MCTS cost
                immediate = find_immediate_win(game_snapshot)
                if immediate is not None:
                    with self._ai_lock:
                        self._ai_result = (immediate, gen)
                    self.ai_thinking = False
                    return

                move = gpu_ai.select_move(game_snapshot)
                with self._ai_lock:
                    self._ai_result = (move, gen)
                self.ai_thinking = False

            threading.Thread(target=gpu_worker, daemon=True).start()
        else:
            # ── CPU path (FNN fallback or legacy CPU MCTS) ────────────
            def worker() -> None:
                immediate = find_immediate_win(game_snapshot)
                if immediate is not None:
                    with self._ai_lock:
                        self._ai_result = (immediate, gen)
                    self.ai_thinking = False
                    return

                if self.cpu_player is not None:
                    move = self.cpu_player.choose_move(game_snapshot)
                else:
                    mcts = MCTS(self.net, self.encoder, self.mcts_config)
                    policy = mcts.search(game_snapshot, move_number=self.move_number)
                    action = int(np.argmax(policy))
                    mask = self.encoder.get_legal_action_mask(game_snapshot)
                    if mask[action] > 0:
                        move = self.encoder.decode_action(action, game_snapshot)
                    else:
                        legal = np.where(mask > 0)[0]
                        best = legal[np.argmax(policy[legal])]
                        move = self.encoder.decode_action(int(best), game_snapshot)
                with self._ai_lock:
                    self._ai_result = (move, gen)
                self.ai_thinking = False

            threading.Thread(target=worker, daemon=True).start()

    # ── Legal move helpers ─────────────────────────────────────────

    def _placements_for(self, pt: PieceType) -> set[HexCoord]:
        return {
            m.to
            for m in self.game.legal_moves()
            if m.move_type == MoveType.PLACE
            and m.piece is not None
            and m.piece.piece_type == pt
        }

    def _destinations_for(self, src: HexCoord) -> set[HexCoord]:
        return {
            m.to
            for m in self.game.legal_moves()
            if m.move_type == MoveType.MOVE and m.from_pos == src
        }

    # ── Drawing helpers ────────────────────────────────────────────

    def _poly_hex(
        self,
        surf: pygame.Surface,
        q: int, r: int,
        fill: tuple,
        border: tuple,
        border_w: int = 1,
    ) -> None:
        cx, cy = axial_to_pixel(q, r, self.cam_x, self.cam_y)
        if not in_board_viewport(cx, cy):
            return
        pts = flat_hex_corners(cx, cy, HEX_SIZE - 1)
        pygame.draw.polygon(surf, fill, pts)
        if border_w > 0:
            pygame.draw.polygon(surf, border, pts, border_w)

    def _draw_piece(
        self,
        surf: pygame.Surface,
        q: int, r: int,
        color: Color,
        symbol: str,
        height: int = 1,
        fill_override: tuple | None = None,
    ) -> None:
        cx, cy = axial_to_pixel(q, r, self.cam_x, self.cam_y)
        if not in_board_viewport(cx, cy):
            return

        # Draw stack shadow layers
        for layer in range(height - 1, 0, -1):
            off = layer * 3
            pts = flat_hex_corners(cx + off, cy - off, HEX_SIZE - 2)
            pygame.draw.polygon(surf, (28, 30, 38), pts)
            pygame.draw.polygon(surf, C_HEX_BORDER, pts, 1)

        # Top face
        fill = fill_override or (C_W_PIECE if color == Color.WHITE else C_B_PIECE)
        pts = flat_hex_corners(cx, cy, HEX_SIZE - 1)
        pygame.draw.polygon(surf, fill, pts)
        pygame.draw.polygon(surf, C_HEX_BORDER, pts, 2)

        # Symbol
        text_col = C_W_TEXT if color == Color.WHITE else C_B_TEXT
        lbl = self.f_large.render(symbol, True, text_col)
        surf.blit(lbl, lbl.get_rect(center=(cx, cy)))

    # ── Board rendering ────────────────────────────────────────────

    def _draw_board(self, surf: pygame.Surface) -> None:
        # Draw board background and set clip rect to prevent drawing over panels
        pygame.draw.rect(surf, C_BOARD_BG, (BOARD_X, BOARD_Y, BOARD_W, BOARD_H))
        surf.set_clip(pygame.Rect(BOARD_X, BOARD_Y, BOARD_W, BOARD_H))

        grid = self.game.board.grid

        # All hexes to consider: occupied + their neighbors + legal destinations
        context: set[HexCoord] = set(grid.keys())
        for pos in list(grid.keys()):
            for n in pos.neighbors():
                context.add(n)
        context |= self.legal_destinations
        if not context:
            context.add(HexCoord(0, 0))

        # Empty context hexes (background grid)
        for pos in context:
            if pos in grid:
                continue
            if pos in self.legal_destinations:
                fill = C_LEGAL
            elif pos == self.hover_hex:
                fill = C_HEX_HOVER
            else:
                fill = C_HEX_EMPTY
            self._poly_hex(surf, pos.q, pos.r, fill, C_HEX_BORDER)

        # Pieces
        for pos, stack in grid.items():
            if not stack:
                continue
            top = stack[-1]
            height = len(stack)

            selected_src = (
                self.mode == SelectMode.BOARD_PIECE
                and self.selected_board_hex == pos
            )
            legal_dest = pos in self.legal_destinations
            last_move_dest = (
                self.last_move is not None
                and self.last_move.move_type in (MoveType.PLACE, MoveType.MOVE)
                and self.last_move.to == pos
            )
            queen_warn = (
                top.piece_type == PieceType.QUEEN
                and self.game.board.num_occupied_neighbors(pos) >= 5
            )

            fill = None
            if selected_src:
                fill = C_SELECTED
            elif legal_dest:
                fill = C_LEGAL
            elif last_move_dest:
                fill = C_LAST_MOVE
            elif queen_warn:
                fill = C_QUEEN_WARN

            self._draw_piece(
                surf, pos.q, pos.r, top.color,
                PIECE_SYM[top.piece_type], height, fill
            )

        # Remove clip rect so panels/status bar draw normally
        surf.set_clip(None)

    # ── Hand panel ─────────────────────────────────────────────────

    def _panel_x(self, color: Color) -> int:
        return 0 if color == Color.WHITE else WIN_W - PANEL_W

    def _hand_row_rects(self, color: Color) -> list[tuple[pygame.Rect, PieceType]]:
        x = self._panel_x(color)
        rows = []
        row_y = STATUS_H + 52
        for pt in PIECE_ORDER:
            rect = pygame.Rect(x + 8, row_y, PANEL_W - 16, 62)
            rows.append((rect, pt))
            row_y += 68
        return rows

    def _draw_hand_panel(self, surf: pygame.Surface, color: Color) -> None:
        x = self._panel_x(color)
        pygame.draw.rect(surf, C_PANEL_BG, (x, STATUS_H, PANEL_W, BOARD_H))
        # Divider line
        lx = x + PANEL_W if color == Color.WHITE else x
        pygame.draw.line(surf, C_HEX_BORDER, (lx, STATUS_H), (lx, WIN_H), 1)

        # Header
        name = "WHITE" if color == Color.WHITE else "BLACK"
        header_col = (200, 195, 175) if color == Color.WHITE else (130, 145, 200)
        you_lbl = " (You)" if (self.self_play or color == self.human_color) else " (AI)"
        h_surf = self.f_med.render(name + you_lbl, True, header_col)
        surf.blit(h_surf, (x + (PANEL_W - h_surf.get_width()) // 2, STATUS_H + 14))

        # Count pieces in hand
        hand = self.game.hand(color)
        counts: dict[PieceType, int] = {}
        for p in hand:
            counts[p.piece_type] = counts.get(p.piece_type, 0) + 1

        is_human = self.self_play or color == self.human_color
        is_active = self.game.current_player == color

        for rect, pt in self._hand_row_rects(color):
            count = counts.get(pt, 0)
            is_sel = (
                is_human
                and self.mode == SelectMode.HAND_PIECE
                and self.selected_piece_type == pt
            )
            can_select = (
                is_human and is_active
                and count > 0
                and self.mode in (SelectMode.IDLE, SelectMode.HAND_PIECE)
                and not self.ai_thinking
            )

            # Row background
            if count == 0:
                row_fill = (28, 30, 40)
            elif is_sel:
                row_fill = C_ROW_SEL
            elif can_select:
                row_fill = C_ROW_NORMAL
            else:
                row_fill = (34, 37, 50)

            pygame.draw.rect(surf, row_fill, rect, border_radius=6)
            if count > 0 and can_select:
                border_c = C_LEGAL if is_sel else C_HEX_BORDER
                pygame.draw.rect(surf, border_c, rect, 1, border_radius=6)

            # Mini hex icon
            ix = x + 28
            iy = rect.centery
            icon_pts = flat_hex_corners(ix, iy, 17)
            icon_fill = (
                C_W_PIECE if color == Color.WHITE else C_B_PIECE
            ) if count > 0 else (36, 38, 50)
            pygame.draw.polygon(surf, icon_fill, icon_pts)
            pygame.draw.polygon(surf, C_HEX_BORDER, icon_pts, 1)

            if count > 0:
                sym_c = C_W_TEXT if color == Color.WHITE else C_B_TEXT
                sym = self.f_small.render(PIECE_SYM[pt], True, sym_c)
                surf.blit(sym, sym.get_rect(center=(ix, iy)))

            # Name and count text
            name_c = C_DIM if count == 0 else C_PANEL_TEXT
            count_c = C_DIM if count == 0 else C_TITLE
            name_surf = self.f_small.render(PIECE_NAME[pt], True, name_c)
            cnt_surf  = self.f_med.render(f"×{count}", True, count_c)
            surf.blit(name_surf, (x + 52, rect.top + 8))
            surf.blit(cnt_surf,  (x + 52, rect.top + 30))

        # "Thinking..." indicator on AI panel
        if not is_human and self.ai_thinking and is_active:
            think = self.f_small.render("Thinking...", True, C_THINKING)
            surf.blit(think, (x + (PANEL_W - think.get_width()) // 2, STATUS_H + 430))

    # ── Status bar ─────────────────────────────────────────────────

    def _draw_status(self, surf: pygame.Surface) -> None:
        pygame.draw.rect(surf, C_STATUS_BG, (0, 0, WIN_W, STATUS_H))
        pygame.draw.line(surf, C_HEX_BORDER, (0, STATUS_H - 1), (WIN_W, STATUS_H - 1), 1)

        result = self.game.result
        if result == GameResult.IN_PROGRESS:
            player = self.game.current_player
            pname = "WHITE" if player == Color.WHITE else "BLACK"
            you = " (You)" if player == self.human_color else " (AI)"
            if self.ai_thinking:
                msg = f"Turn {self.game.turn + 1}  ·  {pname} is thinking..."
                col = C_THINKING
            else:
                msg = f"Turn {self.game.turn + 1}  ·  {pname}'s move{you}"
                col = C_TITLE
        elif result == GameResult.WHITE_WINS:
            msg = "WHITE WINS!   Press R to restart"
            col = C_WIN_W
        elif result == GameResult.BLACK_WINS:
            msg = "BLACK WINS!   Press R to restart"
            col = C_WIN_B
        else:
            msg = "DRAW!   Press R to restart"
            col = C_DRAW

        ms = self.f_status.render(msg, True, col)
        surf.blit(ms, (WIN_W // 2 - ms.get_width() // 2, 12))

        # GPU validation status (right side of status bar)
        if self.gpu_validator:
            gpu_text = self.gpu_validator.status_text
            gpu_col = C_QUEEN_WARN if self.gpu_validator.mismatch or self.gpu_validator.error else C_LEGAL
            gs = self.f_small.render(gpu_text, True, gpu_col)
            surf.blit(gs, (WIN_W - gs.get_width() - 12, 6))

        # Hint line
        if result == GameResult.IN_PROGRESS and not self.ai_thinking:
            if self.game.current_player == self.human_color:
                if self.mode == SelectMode.IDLE:
                    hint = "Select a piece from your hand or click a piece on the board"
                else:
                    hint = "Click a green hex to move  ·  click elsewhere to cancel  ·  R = restart"
            else:
                hint = "AI is computing its move..."
            hs = self.f_small.render(hint, True, C_DIM)
            surf.blit(hs, (WIN_W // 2 - hs.get_width() // 2, 38))

    # ── Full draw ──────────────────────────────────────────────────

    def draw(self, surf: pygame.Surface) -> None:
        surf.fill(C_BG)
        self._draw_status(surf)
        self._draw_board(surf)
        self._draw_hand_panel(surf, Color.WHITE)
        self._draw_hand_panel(surf, Color.BLACK)

    # ── Input handling ─────────────────────────────────────────────

    def handle_click(self, px: int, py: int) -> None:
        if self.game.result != GameResult.IN_PROGRESS:
            return
        if self.ai_thinking:
            return
        if not self.self_play and self.game.current_player != self.human_color:
            return

        # Hand panel click (active player's panel)
        for rect, pt in self._hand_row_rects(self._human):
            if rect.collidepoint(px, py):
                self._handle_hand_click(pt)
                return

        # Board area click
        if BOARD_X <= px <= BOARD_X + BOARD_W and BOARD_Y <= py <= BOARD_Y + BOARD_H:
            fq, fr = pixel_to_axial(px, py, self.cam_x, self.cam_y)
            q, r = hex_round(fq, fr)
            self._handle_board_click(HexCoord(q, r))

    def _handle_hand_click(self, pt: PieceType) -> None:
        if not self.game.has_piece_in_hand(self._human, pt):
            return
        dests = self._placements_for(pt)
        if not dests:
            return
        # Toggle deselect
        if self.mode == SelectMode.HAND_PIECE and self.selected_piece_type == pt:
            self._clear_selection()
        else:
            self.mode = SelectMode.HAND_PIECE
            self.selected_piece_type = pt
            self.legal_destinations = dests

    def _handle_board_click(self, pos: HexCoord) -> None:
        if self.mode == SelectMode.HAND_PIECE:
            if pos in self.legal_destinations:
                hand = self.game.pieces_in_hand(self._human, self.selected_piece_type)
                if hand:
                    self._apply_human(Move(MoveType.PLACE, hand[0], pos))
                    return
            self._clear_selection()

        elif self.mode == SelectMode.BOARD_PIECE:
            if pos in self.legal_destinations:
                src = self.selected_board_hex
                stack = self.game.board.grid.get(src, [])
                if stack:
                    piece = stack[-1]
                    self._apply_human(Move(MoveType.MOVE, piece, pos, from_pos=src))
                    return
            # Try re-selecting a different piece
            self._try_select_board(pos)

        else:  # IDLE
            self._try_select_board(pos)

    def _try_select_board(self, pos: HexCoord) -> None:
        stack = self.game.board.grid.get(pos)
        if not stack or stack[-1].color != self._human:
            self._clear_selection()
            return
        # Must have queen placed to move any piece
        if not self.game._queen_placed[self.human_color]:
            self._clear_selection()
            return
        dests = self._destinations_for(pos)
        if not dests:
            self._clear_selection()
            return
        self.mode = SelectMode.BOARD_PIECE
        self.selected_board_hex = pos
        self.legal_destinations = dests

    def _clear_selection(self) -> None:
        self.mode = SelectMode.IDLE
        self.selected_piece_type = None
        self.selected_board_hex = None
        self.legal_destinations = set()

    def _apply_human(self, move: Move) -> None:
        if self.gpu_validator:
            self.gpu_validator.apply_move(move)
        if self.gpu_ai:
            self.gpu_ai.apply_cpu_move(move)
        self.game.apply_move(move)
        self.last_move = move
        self.move_number += 1
        self._clear_selection()
        if self.gpu_validator:
            self.gpu_validator.validate(self.game)

    def _apply_ai(self, move: Move) -> None:
        if self.gpu_validator:
            self.gpu_validator.apply_move(move)
        # For GPU AI path the move was already chosen from the GPU state;
        # we still apply to GPU so it stays in sync for the next turn.
        if self.gpu_ai:
            self.gpu_ai.apply_cpu_move(move)
        self.game.apply_move(move)
        self.last_move = move
        self.move_number += 1
        if self.gpu_validator:
            self.gpu_validator.validate(self.game)

    def _auto_pass(self) -> None:
        """Apply a PASS move automatically (no choice available)."""
        moves = self.game.legal_moves()
        if len(moves) == 1 and moves[0].move_type == MoveType.PASS:
            self._apply_human(moves[0])

    def handle_mouse_motion(self, px: int, py: int) -> None:
        if BOARD_X <= px <= BOARD_X + BOARD_W and BOARD_Y <= py <= BOARD_Y + BOARD_H:
            fq, fr = pixel_to_axial(px, py, self.cam_x, self.cam_y)
            self.hover_hex = HexCoord(*hex_round(fq, fr))
        else:
            self.hover_hex = None

    def handle_pan_start(self, px: int, py: int) -> None:
        """Begin panning (right or middle mouse button)."""
        self._panning = True
        self._pan_start = (px, py)

    def handle_pan_motion(self, px: int, py: int) -> None:
        """Update camera offset during pan drag."""
        if self._panning and self._pan_start is not None:
            dx = px - self._pan_start[0]
            dy = py - self._pan_start[1]
            self.cam_x += dx
            self.cam_y += dy
            self._pan_start = (px, py)

    def handle_pan_end(self) -> None:
        """Stop panning."""
        self._panning = False
        self._pan_start = None

    def center_camera(self) -> None:
        """Reset camera to center on the current pieces."""
        self.cam_x = 0.0
        self.cam_y = 0.0

    # ── Restart ────────────────────────────────────────────────────

    def _restart(self) -> None:
        self._game_gen += 1
        self.game = GameState(expansions=self.expansions)
        self.move_number = 0
        self.last_move = None
        self._clear_selection()
        self.hover_hex = None
        self.cam_x = 0.0
        self.cam_y = 0.0
        with self._ai_lock:
            self._ai_result = None
        if self.gpu_validator:
            self.gpu_validator.reset()
            self.gpu_validator.validate(self.game)
        if self.gpu_ai:
            self.gpu_ai.reset()
        # ai_thinking may remain True briefly; the thread will finish and its
        # result will be discarded because game_gen won't match.

    # ── Main loop ──────────────────────────────────────────────────

    def run(self, title: str = "Hive — vs AI") -> None:
        pygame.init()
        screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(title)
        clock = pygame.time.Clock()
        icon = pygame.Surface((32, 32), pygame.SRCALPHA)
        pts = flat_hex_corners(16, 16, 14)
        pygame.draw.polygon(icon, (80, 190, 100), pts)
        pygame.draw.polygon(icon, (40, 100, 55), pts, 2)
        pygame.display.set_icon(icon)

        self.f_large  = pygame.font.SysFont("segoeui", 20, bold=True)
        self.f_med    = pygame.font.SysFont("segoeui", 15, bold=True)
        self.f_small  = pygame.font.SysFont("segoeui", 13)
        self.f_status = pygame.font.SysFont("segoeui", 22, bold=True)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._restart()
                    elif event.key == pygame.K_c:
                        self.center_camera()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(*event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (2, 3):
                    self.handle_pan_start(*event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
                    self.handle_pan_end()
                elif event.type == pygame.MOUSEMOTION:
                    if self._panning:
                        self.handle_pan_motion(*event.pos)
                    else:
                        self.handle_mouse_motion(*event.pos)

            # Collect AI result (if ready and from current game generation)
            with self._ai_lock:
                pending = self._ai_result
                if pending is not None:
                    self._ai_result = None

            if pending is not None:
                move, gen = pending
                if gen == self._game_gen:
                    self._apply_ai(move)

            # Auto-pass if current player has no choice
            if (
                self.game.result == GameResult.IN_PROGRESS
                and (self.self_play or self.game.current_player == self.human_color)
                and not self.ai_thinking
            ):
                self._auto_pass()

            # Start AI thinking when it's the AI's turn (not in self-play)
            if (
                not self.self_play
                and self.game.result == GameResult.IN_PROGRESS
                and self.game.current_player != self.human_color
                and not self.ai_thinking
            ):
                self._start_ai_think()

            self.draw(screen)
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


# ── Checkpoint loading ──────────────────────────────────────────────

def load_checkpoint(
    path: str, device: torch.device
) -> tuple[object, object, str]:
    """Auto-detect checkpoint type and return (net_or_gpu_ai, encoder, model_name).

    For FNN / PRS v2 checkpoints the first element is a GpuAI instance and
    encoder is None — the GUI will use the GPU Gumbel MCTS path.
    For GNN / NNUE / Transformer the first element is the network and encoder
    is an encoder object — the GUI uses the CPU MCTS path.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net_config = ckpt["net_config"]
    elo = ckpt.get("elo_rating", 0)
    # GPU trainer saves as "model_state_dict"; CPU trainers use "net_state_dict"
    state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "net_state_dict"

    # ── PRS v2 ────────────────────────────────────────────────────────
    if _HAS_PRS and isinstance(net_config, PRSConfig):
        net = HivePRSTransformerV2(net_config).cuda().eval()
        net.load_state_dict(ckpt["model_state"])
        prs_cfg = PRSMCTSConfigV2(
            num_simulations=256,
            batch_size=1,
            wave_parallel=True,
            expansion_mask=7,
        )
        orch = PRSMCTSOrchestratorV2(net, prs_cfg)
        gpu_ai = GpuAI(orch, expansion_mask=7)
        model_name = (
            f"PRS v2 (d={net_config.d_model}, "
            f"h={net_config.num_heads}, L={net_config.num_layers})"
        )
        print(f"Loaded PRS v2: {path}")
        print(
            f"  Architecture: d_model={net_config.d_model}, "
            f"num_heads={net_config.num_heads}, num_layers={net_config.num_layers}"
        )
        print(f"  MCTS: {prs_cfg.num_simulations} sims, Gumbel, wave_parallel=True, exp_mask=7")
        return gpu_ai, None, model_name

    # ── FNN ───────────────────────────────────────────────────────────
    if _HAS_FNN and isinstance(net_config, FNNConfig):
        net = HiveFNN(net_config).cuda().eval()
        net.load_state_dict(ckpt["model_state_dict"])
        train_cfg = ckpt.get("train_config", None)
        exp_mask = int(train_cfg.expansion_mask) if train_cfg is not None else 0
        fnn_cfg = FNNMCTSConfig(
            num_simulations=1024,
            batch_size=1,
            wave_parallel=True,
            expansion_mask=exp_mask,
        )
        orch = FNNMCTSOrchestrator(net, fnn_cfg)
        gpu_ai = GpuAI(orch, expansion_mask=exp_mask)
        dim = net_config.hidden_dim
        size_tag = "large" if dim >= 64 else ("medium" if dim >= 32 else "small")
        model_name = f"FNN {size_tag} (dim={dim})"
        print(f"Loaded FNN: {path}")
        print(f"  Architecture: hidden_dim={dim}")
        print(
            f"  MCTS: {fnn_cfg.num_simulations} sims, Gumbel, "
            f"wave_parallel=True, exp_mask={exp_mask}"
        )
        return gpu_ai, None, model_name

    # ── NNUE ──────────────────────────────────────────────────────────
    if NNUEConfig is not None and isinstance(net_config, NNUEConfig):
        net = HiveNNUE(net_config)
        net.load_state_dict(ckpt[state_dict_key])
        net = net.to(device)
        net.eval()
        encoder = NNUEEncoder()
        model_name = f"NNUE ({net_config.hidden_dims})"
        print(f"Loaded NNUE: {path}")
        print(f"  Architecture: hidden_dims={net_config.hidden_dims}")

    # ── Transformer ───────────────────────────────────────────────────
    elif isinstance(net_config, TransformerConfig):
        net = HiveTransformer(net_config)
        net.load_state_dict(ckpt[state_dict_key], strict=False)
        net = net.to(device)
        net.eval()
        encoder = TransformerEncoder()
        model_name = (
            f"Transformer (d={net_config.d_model}, "
            f"h={net_config.num_heads}, layers={net_config.num_layers})"
        )
        print(f"Loaded Transformer: {path}")
        print(
            f"  Architecture: d_model={net_config.d_model}, "
            f"num_heads={net_config.num_heads}, num_layers={net_config.num_layers}"
        )

    # ── GNN ───────────────────────────────────────────────────────────
    elif HiveGNN is not None:
        net = HiveGNN(net_config)
        net.load_state_dict(ckpt[state_dict_key])
        net = net.to(device)
        net.eval()
        encoder = GNNEncoder()
        model_name = (
            f"GNN (dim={net_config.hidden_dim}, layers={net_config.num_mp_layers})"
        )
        print(f"Loaded GNN: {path}")
        print(
            f"  Architecture: hidden_dim={net_config.hidden_dim}, "
            f"mp_layers={net_config.num_mp_layers}"
        )

    else:
        raise ValueError(f"Unsupported checkpoint type: {type(net_config)}")

    print(f"  ELO rating:   {elo:.0f}")
    return net, encoder, model_name


# ── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Hive against a trained AI"
    )
    parser.add_argument(
        "--engine",
        choices=["fnn-cpu", "legacy"],
        default="fnn-cpu",
        help="AI engine to use. 'fnn-cpu' is the recommended CPU fallback for systems without an NVIDIA GPU.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path. For --engine fnn-cpu the default is the latest "
            "recommended FNN checkpoint. For --engine legacy it follows the "
            "selected --model path or auto-detect behavior."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["auto", "prs_v2", "fnn_large", "fnn_medium", "fnn_small", "gnn"],
        default="auto",
        help=(
            "Legacy engine only. Which model to use "
            "(default: auto = infer from checkpoint)."
        ),
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help=(
            "Override MCTS simulations per move. "
            "Defaults: FNN=1024, PRS v2=256, GNN/NNUE/Transformer=200."
        ),
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default="white",
        help="Your color (default: white, moves first)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Legacy engine only: compute device: cuda, cpu, or auto (default: auto)",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Play both sides yourself — no AI (useful for testing rules)",
    )
    parser.add_argument(
        "--gpu-validate",
        action="store_true",
        help="Legacy engine only: run GPU move gen in parallel and compare with CPU (requires CUDA)",
    )
    parser.add_argument(
        "--root-workers",
        type=int,
        default=2,
        help="FNN CPU engine: process count for parallel root-candidate Gumbel search (default: 2).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="FNN CPU engine: torch CPU thread count per process (default: 1).",
    )
    parser.add_argument(
        "--expansion",
        default="",
        help="Expansion pieces to enable: M=Mosquito, L=Ladybug, P=Pillbug (e.g., MLP for all)",
    )
    parser.add_argument(
        "--base-game",
        action="store_true",
        help="Force base game only (no expansion pieces), overriding model defaults",
    )
    args = parser.parse_args()

    human_color = Color.WHITE if args.color == "white" else Color.BLACK

    # Default checkpoint paths per model
    _DEFAULT_CHECKPOINTS = {
        "prs_v2":     "checkpoints_prs_v2/prs_v2_iter_0600.pt",
        "fnn_large":  "checkpoints_fnn_large/hive_fnn_checkpoint_0100.pt",
        "fnn_medium": "checkpoints_fnn_medium/hive_fnn_checkpoint_0100.pt",
        "fnn_small":  "checkpoints_fnn_small/hive_fnn_checkpoint_0100.pt",
        "gnn":        "checkpoints_gnn/hive_gnn_checkpoint_0020.pt",
    }

    if args.self_play:
        net, encoder, gpu_ai, cpu_player, title = None, None, None, None, "Hive — Self-Play"
        mcts_config = MCTSConfig(num_simulations=200, temperature=0.0)
        print("Self-play mode: you control both WHITE and BLACK.")
        print("Controls: click to play · R = restart · Escape = quit\n")
    elif args.engine == "fnn-cpu":
        from hive_fnn.fnn_cpu_player import FNNCPUPlayer, FNNCPUMCTSConfig

        checkpoint = (
            args.checkpoint
            or "checkpoints_fnn/resume0500_24576/hive_fnn_checkpoint_0500.pt"
        )
        net, encoder, gpu_ai = None, None, None
        n_sims = args.simulations if args.simulations is not None else 256
        mcts_config = MCTSConfig(num_simulations=n_sims, temperature=0.0)
        cpu_player = FNNCPUPlayer.from_checkpoint(
            checkpoint,
            config=FNNCPUMCTSConfig(
                num_simulations=n_sims,
                use_gumbel_root=True,
                gumbel_considered=16,
                gumbel_noise_scale=0.0,
                temperature=0.0,
                root_parallel_workers=args.root_workers,
                torch_threads=args.threads,
            ),
        )
        title = "Hive — vs FNN CPU"
        print(f"\nYou are playing as {human_color.name}.")
        print("AI engine: FNN CPU fallback")
        print(f"Checkpoint: {checkpoint}")
        print(f"AI uses {n_sims} simulations per move.")
        print(f"CPU root workers: {args.root_workers}")
        print("Controls: click to play · R = restart · Escape = quit\n")
    else:
        cpu_player = None
        # Resolve checkpoint path
        if args.checkpoint:
            checkpoint = args.checkpoint
        elif args.model != "auto":
            checkpoint = _DEFAULT_CHECKPOINTS[args.model]
        else:
            checkpoint = _DEFAULT_CHECKPOINTS["prs_v2"]

        # Device selection
        if args.device and args.device not in ("auto", ""):
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Device: {device}")

        loaded, encoder, model_name = load_checkpoint(checkpoint, device)

        if isinstance(loaded, GpuAI):
            # GPU AI path (FNN / PRS v2)
            gpu_ai = loaded
            net    = None
            # Override simulations if requested
            if args.simulations is not None:
                gpu_ai.orch.config.num_simulations = args.simulations
            n_sims = gpu_ai.orch.config.num_simulations
            mcts_config = MCTSConfig(num_simulations=n_sims, temperature=0.0)
        else:
            # CPU MCTS path (GNN / NNUE / Transformer)
            gpu_ai = None
            net    = loaded
            n_sims = args.simulations if args.simulations is not None else 200
            mcts_config = MCTSConfig(num_simulations=n_sims, temperature=0.0)

        title = f"Hive — vs {model_name}"
        print(f"\nYou are playing as {human_color.name}.")
        print(f"AI model: {model_name}")
        print(f"AI uses {n_sims} MCTS simulations per move.")
        print("Controls: click to play · R = restart · Escape = quit\n")

    # Parse expansion config from CLI flag
    exp_str = args.expansion.upper()
    expansion_config = ExpansionConfig(
        mosquito="M" in exp_str,
        ladybug="L" in exp_str,
        pillbug="P" in exp_str,
    ) if exp_str else None

    # For PRS v2 (exp_mask=7), default expansions to MLP unless --base-game or explicit --expansion
    if gpu_ai is not None and not exp_str and not args.base_game:
        exp_mask = gpu_ai.expansion_mask
        if exp_mask & 1:  # mosquito bit set → model trained with expansions
            expansion_config = ExpansionConfig(
                mosquito=bool(exp_mask & 1),
                ladybug=bool(exp_mask & 2),
                pillbug=bool(exp_mask & 4),
            )
    elif args.engine == "fnn-cpu" and not exp_str and not args.base_game:
        expansion_config = ExpansionConfig(mosquito=True, ladybug=True, pillbug=True)

    gui = HiveGUI(
        net, encoder, mcts_config, human_color,
        cpu_player=cpu_player,
        self_play=args.self_play,
        gpu_validate=args.gpu_validate,
        expansions=expansion_config,
        gpu_ai=gpu_ai,
    )
    try:
        gui.run(title=title)
    finally:
        if cpu_player is not None:
            cpu_player.close()


if __name__ == "__main__":
    main()
