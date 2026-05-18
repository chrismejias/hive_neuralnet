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
import time
import traceback
from dataclasses import dataclass
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
    from hive_fnn.fnn_mcts_orchestrator import (
        FNNMCTSConfig,
        FNNMCTSOrchestrator,
        _GUMBEL_ROUNDS,
        _GUMBEL_WAVE_SCHEDULE,
    )
    _HAS_FNN = True
except ImportError:
    FNNConfig = HiveFNN = FNNMCTSConfig = FNNMCTSOrchestrator = None  # type: ignore
    _GUMBEL_ROUNDS = 0  # type: ignore
    _GUMBEL_WAVE_SCHEDULE = []  # type: ignore
    _HAS_FNN = False


# ── Layout constants ────────────────────────────────────────────────

WIN_W, WIN_H = 1460, 820
STATUS_H = 58      # top status bar
PANEL_W  = 190     # each side hand panel
ANALYSIS_W = 300
BOARD_X  = PANEL_W
BOARD_Y  = STATUS_H
BOARD_W  = WIN_W - 2 * PANEL_W - ANALYSIS_W
BOARD_H  = WIN_H - STATUS_H
BOARD_CX = BOARD_X + BOARD_W // 2
BOARD_CY = BOARD_Y + BOARD_H // 2
ANALYSIS_X = BOARD_X + BOARD_W
ANALYSIS_Y = STATUS_H
ANALYSIS_H = BOARD_H

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
C_ANALYSIS_BG = (24, 27, 35)
C_ANALYSIS_CARD = (32, 36, 46)
C_GRAPH_GRID = (58, 65, 80)
C_GRAPH_WHITE = (228, 222, 206)
C_GRAPH_BLACK = (74, 82, 108)
C_GRAPH_AXIS = (118, 130, 160)
C_PENDING = (145, 152, 170)

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

    def set_expansion_mask(self, expansion_mask: int) -> None:
        """Retarget the GPU state to the requested ruleset and reset search state."""
        self.expansion_mask = expansion_mask
        if hasattr(self.orch, "config"):
            self.orch.config.expansion_mask = expansion_mask
        self.reset()

    def apply_cpu_move(self, move: Move) -> None:
        """Mirror a CPU move onto the GPU state."""
        mb = _cpu_move_to_gpu_bytes(move)
        t  = torch.tensor(list(mb), dtype=torch.uint8, device="cuda").unsqueeze(0)
        self.ext.apply_moves_batch(self.gpu_state, t, 1)

    def _normalized_root_confidence(
        self,
        visits: torch.Tensor,
        n_legal: int,
    ) -> float:
        """Convert root visit concentration into a [0, 1] confidence score."""
        if n_legal <= 0:
            return 0.0
        root_visits = visits[0, :n_legal].to(torch.float32)
        total = float(root_visits.sum().item())
        if total <= 0.0:
            return 0.0
        top = float(root_visits.max().item())
        visited_children = int((root_visits > 0).sum().item())
        if visited_children <= 1:
            return 1.0
        share = top / total
        baseline = 1.0 / float(visited_children)
        conf = (share - baseline) / max(1e-6, 1.0 - baseline)
        return max(0.0, min(1.0, conf))

    def _gather_plain_tree_args(
        self,
        tree: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            tree["visit_count"],
            tree["total_value"],
            tree["prior"],
            tree["parent_idx"],
            tree["move_bytes"],
            tree["action_idx"],
            tree["first_child"],
            tree["num_children"],
            tree["is_terminal"],
            tree["terminal_value"],
            tree["node_count"],
        ]

    def _evaluate_fnn_state_puct(self, state: torch.Tensor) -> dict[str, float]:
        """Run a separate plain-PUCT root search for GUI evaluation only."""
        orch = self.orch
        cfg = orch.config
        tree = orch._alloc_tree(1)
        orch._reset_tree(tree)
        tree_args = self._gather_plain_tree_args(tree)

        states = state.clone()
        legal_moves, num_legal, root_features = (
            self.ext.generate_legal_moves_and_fnn_features_batch(states, 1)
        )
        n = int(num_legal[0].item())
        if n <= 0:
            return {
                "eval_value": 0.0,
                "eval_confidence": 0.0,
                "eval_top_q": 0.0,
            }

        priors, root_values, _ = orch._eval_states(
            states, legal_moves, num_legal, 1, root_features,
        )
        results = self.ext.check_results_batch(states, 1)
        leaf_indices = tree["root_node"].to(torch.int32).clone()
        self.ext.mcts_expand_dense_priors_batch(
            *tree_args,
            leaf_indices,
            states,
            legal_moves,
            num_legal,
            priors,
            results,
            1,
            1,
            orch._max_nodes,
        )

        alive_mask = (
            torch.arange(orch._max_legal, device="cuda").unsqueeze(0)
            < num_legal.to(torch.int64).unsqueeze(1)
        ).to(torch.int8)
        game_active_t = torch.ones((1,), dtype=torch.int8, device="cuda")

        sims = max(1, int(cfg.num_simulations))
        wave_size = max(1, int(getattr(cfg, "wave_size", 1)))
        num_waves = math.ceil(sims / wave_size)
        state_size = states.shape[1]
        leaf_states = torch.zeros(
            wave_size, state_size, dtype=torch.uint8, device="cuda",
        )

        for wave in range(num_waves):
            actual_w = min(wave_size, sims - wave * wave_size)
            total = actual_w
            (
                leaf_idx,
                move_paths,
                path_lens,
                vl_paths,
                vl_lens,
            ) = self.ext.mcts_select_with_root_mask_batch(
                *tree_args,
                game_active_t,
                tree["root_node"],
                alive_mask,
                orch._max_legal,
                cfg.c_puct,
                1,
                actual_w,
                orch._max_nodes,
            )
            self.ext.mcts_replay_batch(
                states,
                leaf_states[:total],
                move_paths[:total],
                path_lens[:total],
                leaf_idx[:total],
                1,
                total,
            )
            leaf_results = self.ext.check_results_batch(leaf_states[:total], total)
            legal_moves_leaf, num_legal_leaf, leaf_features = (
                self.ext.generate_legal_moves_and_fnn_features_batch(
                    leaf_states[:total], total,
                )
            )
            priors_leaf, leaf_vals, _ = orch._eval_states(
                leaf_states[:total], legal_moves_leaf, num_legal_leaf, total, leaf_features,
            )
            self.ext.mcts_expand_and_backprop_dense_priors_batch(
                *tree_args,
                leaf_idx[:total],
                leaf_states[:total],
                legal_moves_leaf,
                num_legal_leaf,
                priors_leaf,
                leaf_results,
                leaf_vals,
                vl_paths[:total],
                vl_lens[:total],
                1,
                total,
                orch._max_nodes,
            )

        slot_visits, slot_q = orch._gather_root_child_stats(tree, 1)
        root_visits = slot_visits[0, :n]
        root_q = slot_q[0, :n]
        top_idx = int(root_visits.argmax().item()) if n > 0 else 0
        top_q = float(root_q[top_idx].item()) if n > 0 else 0.0
        confidence = self._normalized_root_confidence(slot_visits, n)
        eval_value = (1.0 - confidence) * float(root_values[0].item()) + confidence * top_q
        return {
            "eval_value": float(eval_value),
            "eval_confidence": confidence,
            "eval_top_q": top_q,
        }

    def _analyze_fnn_state(self, state: torch.Tensor) -> dict[str, object]:
        """Run one FNN root search and return move bytes plus GUI eval signals."""
        orch = self.orch
        cfg = orch.config
        tree = orch._alloc_tree(1)
        orch._reset_tree(tree)

        states = state.clone()
        legal_moves, num_legal, root_features = (
            self.ext.generate_legal_moves_and_fnn_features_batch(states, 1)
        )
        n = int(num_legal[0].item())
        if n <= 0:
            return {
                "move_bytes": torch.zeros((6,), dtype=torch.uint8, device="cpu"),
                "top_child_q": 0.0,
                "root_value": 0.0,
                "blend_value": 0.0,
                "confidence": 0.0,
            }

        priors, root_values, _child_q_init = orch._eval_states(
            states, legal_moves, num_legal, 1, root_features,
        )
        n_per_game = num_legal.to(torch.int64)
        slot_idx = torch.arange(orch._max_legal, device="cuda").unsqueeze(0)
        valid_slot = slot_idx < n_per_game.unsqueeze(1)
        allowed_slot_mask = valid_slot

        if getattr(cfg, "short_forced_win_probe", False):
            tactical = orch._analyze_short_tactics(
                states, legal_moves, num_legal, 1, priors,
            )
            allowed_slot_mask = tactical.allowed_slots & valid_slot

        safe_prior = priors.clamp(min=1e-20)
        legal_logits = torch.where(
            allowed_slot_mask,
            safe_prior.log(),
            torch.full_like(priors, -1e30),
        )
        u = torch.rand(1, orch._max_legal, device="cuda").clamp(1e-4, 1 - 1e-4)
        gumbel = -torch.log(-torch.log(u))
        gumbel = torch.where(
            allowed_slot_mask,
            gumbel,
            torch.full_like(gumbel, -1e30),
        )
        if hasattr(self.ext, "queen_escape_flags_batch"):
            candidate_slots = orch._root_candidate_slots(
                states,
                legal_moves,
                num_legal,
                priors,
                legal_logits,
                1,
                allowed_slot_mask=allowed_slot_mask,
            ).to(torch.int32)
        else:
            max_considered = max(
                1,
                min(
                    int(getattr(cfg, "max_num_considered_actions", 16)),
                    n,
                    orch._max_legal,
                ),
            )
            perturbed = torch.where(
                allowed_slot_mask,
                gumbel + legal_logits,
                torch.full_like(legal_logits, -1e30),
            )
            _, candidate_slots = torch.topk(perturbed, max_considered, dim=1)
            candidate_slots = candidate_slots.to(torch.int32)
        candidate_valid = torch.gather(
            valid_slot, 1, candidate_slots.long().clamp(min=0),
        ) & (candidate_slots >= 0)
        candidate_slots = torch.where(
            candidate_valid,
            candidate_slots,
            torch.full_like(candidate_slots, -1),
        )

        game_active_t = torch.ones((1,), dtype=torch.int8, device="cuda")
        leaf_indices = tree["root_node"].to(torch.int32).clone()
        results = self.ext.check_results_batch(states, 1)
        tree_args = self._gather_plain_tree_args(tree)
        self.ext.mcts_expand_dense_priors_batch(
            *tree_args,
            leaf_indices,
            states,
            legal_moves,
            num_legal,
            priors,
            results,
            1,
            1,
            orch._max_nodes,
        )
        sims_per_round = max(1, int(cfg.num_simulations) // _GUMBEL_ROUNDS)
        for round_i in range(_GUMBEL_ROUNDS):
            num_candidates = int(candidate_slots.shape[1])
            if num_candidates <= 0:
                break
            sims_per_candidate = max(1, sims_per_round // num_candidates)
            wave_size = (
                _GUMBEL_WAVE_SCHEDULE[min(round_i, len(_GUMBEL_WAVE_SCHEDULE) - 1)]
                if getattr(cfg, "wave_parallel", False)
                else 1
            )
            C = int(candidate_slots.shape[1])
            W = max(1, int(wave_size))
            num_waves = math.ceil(sims_per_candidate / W)
            use_fused_prefix = hasattr(
                self.ext, "mcts_select_replay_legal_fnn_root_slots_batch",
            )
            leaf_states = None
            if not use_fused_prefix:
                state_size = states.shape[1]
                leaf_states = torch.zeros(
                    W * C, state_size, dtype=torch.uint8, device="cuda",
                )

            for wave in range(num_waves):
                actual_w = min(W, sims_per_candidate - wave * W)
                total = actual_w * C

                if use_fused_prefix:
                    (
                        leaf_idx,
                        leaf_states_wave,
                        legal_moves_leaf,
                        num_legal_leaf,
                        leaf_features,
                        results,
                        vl_paths,
                        vl_lens,
                    ) = self.ext.mcts_select_replay_legal_fnn_root_slots_batch(
                        *tree_args,
                        states,
                        game_active_t,
                        tree["root_node"],
                        candidate_slots,
                        C,
                        orch._max_legal,
                        cfg.c_puct,
                        1,
                        actual_w,
                        orch._max_nodes,
                    )
                else:
                    (
                        leaf_idx,
                        move_paths,
                        path_lens,
                        vl_paths,
                        vl_lens,
                    ) = self.ext.mcts_select_with_root_slots_batch(
                        *tree_args,
                        game_active_t,
                        tree["root_node"],
                        candidate_slots,
                        C,
                        orch._max_legal,
                        cfg.c_puct,
                        1,
                        actual_w,
                        orch._max_nodes,
                    )
                    self.ext.mcts_replay_batch(
                        states,
                        leaf_states[:total],
                        move_paths[:total],
                        path_lens[:total],
                        leaf_idx[:total],
                        1,
                        total,
                    )
                    leaf_states_wave = leaf_states[:total]
                    results = self.ext.check_results_batch(leaf_states_wave, total)
                    legal_moves_leaf, num_legal_leaf, leaf_features = (
                        self.ext.generate_legal_moves_and_fnn_features_batch(
                            leaf_states_wave,
                            total,
                        )
                    )

                priors_leaf, leaf_vals, _child_q_leaf = orch._eval_states(
                    leaf_states_wave,
                    legal_moves_leaf,
                    num_legal_leaf,
                    total,
                    leaf_features,
                )
                self.ext.mcts_expand_and_backprop_dense_priors_batch(
                    *tree_args,
                    leaf_idx[:total],
                    leaf_states_wave,
                    legal_moves_leaf,
                    num_legal_leaf,
                    priors_leaf,
                    results,
                    leaf_vals,
                    vl_paths[:total],
                    vl_lens[:total],
                    1,
                    total,
                    orch._max_nodes,
                )
            if num_candidates <= 1:
                continue

            candidate_valid = candidate_slots >= 0
            per_game_keep = (candidate_valid.sum(dim=1) // 2).clamp(min=1)
            max_keep = num_candidates // 2
            cand_visits, cand_q = orch._gather_root_candidate_stats(
                tree, 1, candidate_slots,
            )
            sigma_norm = (cfg.c_visit + cand_visits.max()) * cfg.c_scale
            cand_idx = candidate_slots.long().clamp(min=0)
            cand_score = (
                torch.gather(gumbel + legal_logits, 1, cand_idx)
                + sigma_norm * cand_q
            ).float()
            cand_score = torch.where(
                candidate_valid,
                cand_score,
                torch.full_like(cand_score, -1e30),
            )
            _, keep_pos = torch.topk(cand_score, max_keep, dim=1)
            keep_rank = orch._keep_rank(max_keep)
            keep_valid = keep_rank < per_game_keep.unsqueeze(1)
            new_slots = torch.gather(candidate_slots, 1, keep_pos)
            candidate_slots = torch.where(
                keep_valid,
                new_slots,
                torch.full_like(new_slots, -1),
            )

        candidate_valid = candidate_slots >= 0
        cand_visits, cand_q = orch._gather_root_candidate_stats(
            tree, 1, candidate_slots,
        )
        sigma_norm = (cfg.c_visit + cand_visits.max()) * cfg.c_scale
        cand_idx = candidate_slots.long().clamp(min=0)
        final_score = (
            torch.gather(gumbel + legal_logits, 1, cand_idx)
            + sigma_norm * cand_q
        ).float()
        final_score = torch.where(
            candidate_valid,
            final_score,
            torch.full_like(final_score, -1e30),
        )
        chosen_pos = torch.argmax(final_score, dim=1)
        chosen_slot = torch.gather(
            candidate_slots, 1, chosen_pos.unsqueeze(1),
        ).squeeze(1).clamp(min=0).to(torch.long)
        move_bytes = legal_moves[0, chosen_slot[0]].detach().cpu()

        slot_visits, slot_q = orch._gather_root_child_stats(tree, 1)
        top_visits, top_indices = slot_visits[0, :n].max(dim=0)
        top_child_q = float(slot_q[0, int(top_indices.item())].item()) if n > 0 else 0.0
        root_value = float(root_values[0].item())
        confidence = self._normalized_root_confidence(slot_visits, n)
        blend_value = (1.0 - confidence) * root_value + confidence * top_child_q

        return {
            "move_bytes": move_bytes,
            "top_child_q": top_child_q,
            "root_value": root_value,
            "blend_value": float(blend_value),
            "confidence": confidence,
            "top_visits": int(top_visits.item()) if n > 0 else 0,
        }

    def _analyze_state_tensor(self, state: torch.Tensor) -> MoveEval:
        if _HAS_FNN and isinstance(self.orch, FNNMCTSOrchestrator):
            info = self._analyze_fnn_state(state)
            eval_info = self._evaluate_fnn_state_puct(state)
            return MoveEval(
                q_value=float(info.get("top_child_q", 0.0)),
                root_value=float(info.get("root_value", 0.0)),
                blend_value=float(info.get("blend_value", 0.0)),
                eval_value=float(eval_info.get("eval_value", 0.0)),
                source="search",
                simulations=int(getattr(self.orch.config, "num_simulations", 0)),
                confidence=float(eval_info.get("eval_confidence", info.get("confidence", 0.0))),
            )
        if hasattr(self.orch, "analyze_state_gpu"):
            info = self.orch.analyze_state_gpu(state)
            return MoveEval(
                q_value=float(info.get("chosen_q", 0.0)),
                root_value=float(info.get("root_value", 0.0)),
                source="search",
                simulations=int(getattr(self.orch.config, "num_simulations", 0)),
            )
        return MoveEval()

    def analyze_current_state(self) -> MoveEval:
        """Analyze the current GPU state without mutating it."""
        return self._analyze_state_tensor(self.gpu_state)

    def analyze_prefix(self, moves: list[Move]) -> MoveEval:
        """Analyze a historical prefix by replaying moves on a temporary GPU state."""
        temp_state = self.ext.create_initial_states(1, self.expansion_mask)
        for move in moves:
            mb = _cpu_move_to_gpu_bytes(move)
            t = torch.tensor(list(mb), dtype=torch.uint8, device="cuda").unsqueeze(0)
            self.ext.apply_moves_batch(temp_state, t, 1)
        return self._analyze_state_tensor(temp_state)

    def select_move(self, game: GameState) -> Move:
        """Run GPU Gumbel MCTS and return the chosen CPU Move."""
        legal = game.legal_moves()
        if not legal:
            return Move(MoveType.PASS, None, HexCoord(0, 0))

        legal_by_bytes = {_cpu_move_to_gpu_bytes(mv): mv for mv in legal}
        move_bytes = self.orch.select_move_gpu(self.gpu_state)   # (6,) uint8 CPU
        raw_bytes = bytes(move_bytes.cpu().tolist())

        move = legal_by_bytes.get(raw_bytes)
        if move is not None:
            return move

        # Defensive fallback: decode by structure, then verify it is truly legal
        # under the CPU ruleset before accepting it.
        move = _gpu_bytes_to_cpu_move(move_bytes.numpy(), game)
        if move is not None:
            legal_move = legal_by_bytes.get(_cpu_move_to_gpu_bytes(move))
            if legal_move is not None:
                return legal_move

        print("WARNING: GPU search returned a move illegal under the CPU ruleset; falling back to first legal move.")
        return legal[0]

    def select_move_with_eval(self, game: GameState) -> tuple[Move, MoveEval]:
        """Run GPU search and return both the move and root evaluation."""
        legal = game.legal_moves()
        if not legal:
            return Move(MoveType.PASS, None, HexCoord(0, 0)), MoveEval(q_value=0.0, source="empty")

        if _HAS_FNN and isinstance(self.orch, FNNMCTSOrchestrator):
            info = self._analyze_fnn_state(self.gpu_state)
            eval_info = self._evaluate_fnn_state_puct(self.gpu_state)
            move_bytes = info["move_bytes"]
            raw_bytes = bytes(move_bytes.tolist())
            legal_by_bytes = {_cpu_move_to_gpu_bytes(mv): mv for mv in legal}
            move = legal_by_bytes.get(raw_bytes)
            if move is None:
                move = _gpu_bytes_to_cpu_move(move_bytes.numpy(), game)
                if move is not None:
                    move = legal_by_bytes.get(_cpu_move_to_gpu_bytes(move))
            if move is None:
                print("WARNING: GPU FNN analysis returned a move illegal under the CPU ruleset; falling back to first legal move.")
                move = legal[0]
            move_eval = MoveEval(
                q_value=float(info.get("top_child_q", 0.0)),
                root_value=float(info.get("root_value", 0.0)),
                blend_value=float(info.get("blend_value", 0.0)),
                eval_value=float(eval_info.get("eval_value", 0.0)),
                source="search",
                simulations=int(getattr(self.orch.config, "num_simulations", 0)),
                confidence=float(eval_info.get("eval_confidence", info.get("confidence", 0.0))),
            )
            return move, move_eval

        if hasattr(self.orch, "analyze_state_gpu"):
            info = self.orch.analyze_state_gpu(self.gpu_state)
            move_bytes = info["move_bytes"]
            raw_bytes = bytes(move_bytes.cpu().tolist())
            legal_by_bytes = {_cpu_move_to_gpu_bytes(mv): mv for mv in legal}
            move = legal_by_bytes.get(raw_bytes)
            if move is None:
                move = _gpu_bytes_to_cpu_move(move_bytes.numpy(), game)
                if move is not None:
                    move = legal_by_bytes.get(_cpu_move_to_gpu_bytes(move))
            if move is None:
                print("WARNING: GPU analysis returned a move illegal under the CPU ruleset; falling back to first legal move.")
                move = legal[0]
            move_eval = MoveEval(
                q_value=float(info.get("chosen_q", 0.0)),
                root_value=float(info.get("root_value", 0.0)),
                source="search",
                simulations=int(getattr(self.orch.config, "num_simulations", 0)),
            )
            return move, move_eval

        return self.select_move(game), MoveEval()

    def reset(self) -> None:
        """Reinitialize the GPU state to the starting position."""
        self.gpu_state = self.ext.create_initial_states(1, self.expansion_mask)


# ── Selection state ─────────────────────────────────────────────────

class SelectMode(Enum):
    IDLE        = auto()
    HAND_PIECE  = auto()   # piece type chosen from hand → show placements
    BOARD_PIECE = auto()   # board piece chosen → show movements


@dataclass
class MoveEval:
    q_value: float | None = None
    root_value: float | None = None
    blend_value: float | None = None
    eval_value: float | None = None
    source: str = ""
    simulations: int = 0
    confidence: float | None = None


@dataclass
class MoveRecord:
    move: Move
    mover: Color
    eval: MoveEval


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
        self.move_records: list[MoveRecord] = []
        self.history_states: list[GameState] = [self.game.copy()]
        self.view_ply = 0
        self.analysis_enabled = False
        self.analysis_running = False
        self._analysis_lock = threading.Lock()
        self._analysis_pending: list[tuple[int, MoveEval]] = []

        # Selection
        self.mode = SelectMode.IDLE
        self.selected_piece_type: PieceType | None = None
        self.selected_board_hex: HexCoord | None = None
        self.legal_destinations: set[HexCoord] = set()

        # AI thread
        self._game_gen = 0          # incremented on restart to discard stale results
        self.ai_thinking = False
        self._ai_result: tuple[Move, MoveEval, int] | None = None   # (move, eval, game_gen)
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

    def _shown_game(self) -> GameState:
        idx = max(0, min(self.view_ply, len(self.history_states) - 1))
        return self.history_states[idx]

    def _latest_ply(self) -> int:
        return len(self.move_records)

    def _is_live_view(self) -> bool:
        return self.view_ply == self._latest_ply()

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
                try:
                    t0 = time.perf_counter()
                    immediate = find_immediate_win(game_snapshot)
                    t1 = time.perf_counter()
                    print(f"[GUI] immediate-win check: {t1 - t0:.3f}s")
                    if immediate is not None:
                        move_eval = gpu_ai.analyze_current_state()
                        move_eval.source = "forced"
                        with self._ai_lock:
                            self._ai_result = (immediate, move_eval, gen)
                        return

                    move, move_eval = gpu_ai.select_move_with_eval(game_snapshot)
                    t2 = time.perf_counter()
                    print(
                        f"[GUI] GPU search: {t2 - t1:.3f}s "
                        f"(total {t2 - t0:.3f}s, sims={getattr(gpu_ai.orch.config, 'num_simulations', 0)})"
                    )
                    with self._ai_lock:
                        self._ai_result = (move, move_eval, gen)
                except Exception:
                    print("[GUI] GPU worker failed:")
                    traceback.print_exc()
                finally:
                    self.ai_thinking = False

            threading.Thread(target=gpu_worker, daemon=True).start()
        else:
            # ── CPU path (FNN fallback or legacy CPU MCTS) ────────────
            def worker() -> None:
                try:
                    immediate = find_immediate_win(game_snapshot)
                    if immediate is not None:
                        with self._ai_lock:
                            self._ai_result = (immediate, MoveEval(source="forced"), gen)
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
                        self._ai_result = (move, MoveEval(), gen)
                except Exception:
                    print("[GUI] CPU worker failed:")
                    traceback.print_exc()
                finally:
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
        game = self._shown_game()
        # Draw board background and set clip rect to prevent drawing over panels
        pygame.draw.rect(surf, C_BOARD_BG, (BOARD_X, BOARD_Y, BOARD_W, BOARD_H))
        surf.set_clip(pygame.Rect(BOARD_X, BOARD_Y, BOARD_W, BOARD_H))

        grid = game.board.grid

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
        shown_last_move = (
            self.move_records[self.view_ply - 1].move
            if self.view_ply > 0 and self.view_ply <= len(self.move_records)
            else None
        )
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
                shown_last_move is not None
                and shown_last_move.move_type in (MoveType.PLACE, MoveType.MOVE)
                and shown_last_move.to == pos
            )
            queen_warn = (
                top.piece_type == PieceType.QUEEN
                and game.board.num_occupied_neighbors(pos) >= 5
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
        game = self._shown_game()
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
        hand = game.hand(color)
        counts: dict[PieceType, int] = {}
        for p in hand:
            counts[p.piece_type] = counts.get(p.piece_type, 0) + 1

        is_human = self.self_play or color == self.human_color
        is_active = game.current_player == color

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
        game = self._shown_game()
        pygame.draw.rect(surf, C_STATUS_BG, (0, 0, WIN_W, STATUS_H))
        pygame.draw.line(surf, C_HEX_BORDER, (0, STATUS_H - 1), (WIN_W, STATUS_H - 1), 1)

        result = game.result
        if result == GameResult.IN_PROGRESS:
            player = game.current_player
            pname = "WHITE" if player == Color.WHITE else "BLACK"
            you = ""
            if not self.self_play:
                you = " (You)" if player == self.human_color else " (AI)"
            if self.ai_thinking:
                msg = f"Turn {game.turn + 1}  ·  {pname} is thinking..."
                col = C_THINKING
            else:
                msg = f"Turn {game.turn + 1}  ·  {pname}'s move{you}"
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
        if not self._is_live_view():
            hint = (
                f"Viewing move {self.view_ply}/{self._latest_ply()}  ·  "
                "Left/Right navigate  ·  End returns live"
            )
            hs = self.f_small.render(hint, True, C_DIM)
            surf.blit(hs, (WIN_W // 2 - hs.get_width() // 2, 38))
        elif result == GameResult.IN_PROGRESS and not self.ai_thinking:
            if self.game.current_player == self.human_color:
                if self.mode == SelectMode.IDLE:
                    hint = "Select a piece from your hand or click a piece on the board"
                else:
                    hint = "Click a green hex to move  ·  click elsewhere to cancel  ·  R = restart"
            else:
                hint = "AI is computing its move..."
            hs = self.f_small.render(hint, True, C_DIM)
            surf.blit(hs, (WIN_W // 2 - hs.get_width() // 2, 38))

    def _move_summary(self, move: Move) -> str:
        if move.move_type == MoveType.PASS:
            return "PASS"
        piece = PIECE_SYM.get(move.piece.piece_type, "?") if move.piece else "?"
        if move.move_type == MoveType.PLACE:
            return f"{piece} -> ({move.to.q},{move.to.r})"
        return f"{piece} ({move.from_pos.q},{move.from_pos.r}) -> ({move.to.q},{move.to.r})"

    def _analysis_button_rects(self) -> dict[str, pygame.Rect]:
        y = ANALYSIS_Y + ANALYSIS_H - 48
        w = 64
        gap = 8
        x0 = ANALYSIS_X + 18
        return {
            "first": pygame.Rect(x0, y, w, 30),
            "prev": pygame.Rect(x0 + (w + gap), y, w, 30),
            "next": pygame.Rect(x0 + 2 * (w + gap), y, w, 30),
            "last": pygame.Rect(x0 + 3 * (w + gap), y, w, 30),
        }

    def _draw_analysis_panel(self, surf: pygame.Surface) -> None:
        pygame.draw.rect(surf, C_ANALYSIS_BG, (ANALYSIS_X, ANALYSIS_Y, ANALYSIS_W, ANALYSIS_H))
        pygame.draw.line(surf, C_HEX_BORDER, (ANALYSIS_X, ANALYSIS_Y), (ANALYSIS_X, WIN_H), 1)

        title = "Analyzer On" if self.analysis_enabled else "Analyzer Off"
        title_s = self.f_med.render(title, True, C_TITLE)
        surf.blit(title_s, (ANALYSIS_X + 18, ANALYSIS_Y + 12))
        hint_s = self.f_small.render("A toggles analyzer", True, C_DIM)
        surf.blit(hint_s, (ANALYSIS_X + 18, ANALYSIS_Y + 34))

        if not self.analysis_enabled:
            body = self.f_small.render(
                "Toggle on to track move values, browse plies, and show the game graph.",
                True,
                C_PANEL_TEXT,
            )
            surf.blit(body, (ANALYSIS_X + 18, ANALYSIS_Y + 74))
            return

        latest = self._latest_ply()
        selected_idx = self.view_ply - 1
        card = pygame.Rect(ANALYSIS_X + 14, ANALYSIS_Y + 64, ANALYSIS_W - 28, 108)
        pygame.draw.rect(surf, C_ANALYSIS_CARD, card, border_radius=8)

        if 0 <= selected_idx < len(self.move_records):
            rec = self.move_records[selected_idx]
            mover = "White" if rec.mover == Color.WHITE else "Black"
            move_s = self.f_med.render(
                f"Move {selected_idx + 1}  ·  {mover}",
                True,
                C_TITLE,
            )
            surf.blit(move_s, (card.x + 12, card.y + 10))
            desc_s = self.f_small.render(self._move_summary(rec.move), True, C_PANEL_TEXT)
            surf.blit(desc_s, (card.x + 12, card.y + 38))
            if rec.eval.q_value is None:
                eval_text = "Top: pending"
                eval_col = C_PENDING
            else:
                eval_text = f"Top: {rec.eval.q_value:+.3f}"
                eval_col = C_WIN_W if rec.eval.q_value > 0 else (C_WIN_B if rec.eval.q_value < 0 else C_TITLE)
            eval_s = self.f_large.render(eval_text, True, eval_col)
            surf.blit(eval_s, (card.x + 12, card.y + 58))
            if rec.eval.root_value is None:
                root_text = "V: pending"
                root_col = C_PENDING
            else:
                root_text = f"V: {rec.eval.root_value:+.3f}"
                root_col = C_PANEL_TEXT
            root_s = self.f_small.render(root_text, True, root_col)
            surf.blit(root_s, (card.x + 14, card.y + 86))
            if rec.eval.blend_value is None:
                blend_text = "Blend: pending"
                blend_col = C_PENDING
            else:
                blend_text = f"Blend: {rec.eval.blend_value:+.3f}"
                blend_col = C_PANEL_TEXT
            blend_s = self.f_small.render(blend_text, True, blend_col)
            surf.blit(blend_s, (card.x + 14, card.y + 104))
            if rec.eval.eval_value is None:
                eval2_text = "Eval: pending"
                eval2_col = C_PENDING
            else:
                eval2_text = f"Eval: {rec.eval.eval_value:+.3f}"
                eval2_col = C_PANEL_TEXT
            eval2_s = self.f_small.render(eval2_text, True, eval2_col)
            surf.blit(eval2_s, (card.x + 14, card.y + 122))
            meta = rec.eval.source or "unavailable"
            if rec.eval.confidence is not None:
                meta = f"{meta}  ·  conf {rec.eval.confidence:.2f}"
            meta_s = self.f_small.render(
                f"{meta}  ·  sims {rec.eval.simulations}",
                True,
                C_DIM,
            )
            surf.blit(meta_s, (card.x + 132, card.y + 106))
        else:
            move_s = self.f_med.render("Initial position", True, C_TITLE)
            surf.blit(move_s, (card.x + 12, card.y + 10))
            desc_s = self.f_small.render("No move selected yet.", True, C_PANEL_TEXT)
            surf.blit(desc_s, (card.x + 12, card.y + 38))

        graph = pygame.Rect(ANALYSIS_X + 14, ANALYSIS_Y + 190, ANALYSIS_W - 28, 180)
        pygame.draw.rect(surf, C_ANALYSIS_CARD, graph, border_radius=8)
        gtitle = self.f_small.render("Game graph (white advantage)", True, C_TITLE)
        surf.blit(gtitle, (graph.x + 12, graph.y + 8))

        plot = pygame.Rect(graph.x + 12, graph.y + 30, graph.w - 24, graph.h - 54)
        pygame.draw.rect(surf, (26, 28, 36), plot, border_radius=4)
        mid_y = plot.y + plot.h // 2
        pygame.draw.line(surf, C_GRAPH_AXIS, (plot.x, mid_y), (plot.right, mid_y), 1)
        for i in range(1, 4):
            y = plot.y + i * plot.h // 4
            pygame.draw.line(surf, C_GRAPH_GRID, (plot.x, y), (plot.right, y), 1)

        values: list[float | None] = []
        for rec in self.move_records:
            graph_value = rec.eval.eval_value
            if graph_value is None:
                graph_value = rec.eval.blend_value
            if graph_value is None:
                values.append(None)
            elif rec.mover == Color.WHITE:
                values.append(graph_value)
            else:
                values.append(-graph_value)

        if values:
            pts: list[tuple[int, int]] = []
            denom = max(1, len(values) - 1)
            for idx, val in enumerate(values):
                if val is None:
                    continue
                x = plot.x + int(idx * plot.w / denom)
                clipped = max(-1.0, min(1.0, float(val)))
                y = plot.y + int((1.0 - (clipped + 1.0) * 0.5) * plot.h)
                pts.append((x, y))
            if len(pts) >= 2:
                pygame.draw.lines(surf, C_TITLE, False, pts, 2)
            for idx, val in enumerate(values):
                if val is None:
                    continue
                x = plot.x + int(idx * plot.w / denom)
                clipped = max(-1.0, min(1.0, float(val)))
                y = plot.y + int((1.0 - (clipped + 1.0) * 0.5) * plot.h)
                radius = 4 if idx == selected_idx else 3
                color = C_W_PIECE if clipped >= 0 else C_B_PIECE
                pygame.draw.circle(surf, color, (x, y), radius)
                pygame.draw.circle(surf, C_HEX_BORDER, (x, y), radius, 1)

        status = "Backfilling opponent moves..." if self.analysis_running else "Live"
        status_s = self.f_small.render(status, True, C_DIM)
        surf.blit(status_s, (graph.x + 12, graph.bottom - 20))

        for name, rect in self._analysis_button_rects().items():
            pygame.draw.rect(surf, C_ROW_NORMAL, rect, border_radius=6)
            pygame.draw.rect(surf, C_HEX_BORDER, rect, 1, border_radius=6)
            label = {"first": "|<", "prev": "<", "next": ">", "last": ">|"}[name]
            txt = self.f_med.render(label, True, C_TITLE)
            surf.blit(txt, txt.get_rect(center=rect.center))

    # ── Full draw ──────────────────────────────────────────────────

    def draw(self, surf: pygame.Surface) -> None:
        surf.fill(C_BG)
        self._draw_status(surf)
        self._draw_board(surf)
        self._draw_analysis_panel(surf)
        self._draw_hand_panel(surf, Color.WHITE)
        self._draw_hand_panel(surf, Color.BLACK)

    # ── Input handling ─────────────────────────────────────────────

    def handle_click(self, px: int, py: int) -> None:
        if self._handle_analysis_click(px, py):
            return
        if not self._is_live_view():
            return
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
        self._record_move(move, MoveEval())
        if self.gpu_validator:
            self.gpu_validator.apply_move(move)
        if self.gpu_ai:
            self.gpu_ai.apply_cpu_move(move)
        self.game.apply_move(move)
        self.last_move = move
        self.move_number += 1
        self.history_states.append(self.game.copy())
        self.view_ply = self._latest_ply()
        self._clear_selection()
        if self.gpu_validator:
            self.gpu_validator.validate(self.game)

    def _apply_ai(self, move: Move, move_eval: MoveEval) -> None:
        self._record_move(move, move_eval)
        if self.gpu_validator:
            self.gpu_validator.apply_move(move)
        # For GPU AI path the move was already chosen from the GPU state;
        # we still apply to GPU so it stays in sync for the next turn.
        if self.gpu_ai:
            self.gpu_ai.apply_cpu_move(move)
        self.game.apply_move(move)
        self.last_move = move
        self.move_number += 1
        self.history_states.append(self.game.copy())
        self.view_ply = self._latest_ply()
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

    def _set_view_ply(self, ply: int) -> None:
        self.view_ply = max(0, min(ply, self._latest_ply()))
        if not self._is_live_view():
            self._clear_selection()

    def _handle_analysis_click(self, px: int, py: int) -> bool:
        if not self.analysis_enabled:
            return False
        for name, rect in self._analysis_button_rects().items():
            if not rect.collidepoint(px, py):
                continue
            if name == "first":
                self._set_view_ply(0)
            elif name == "prev":
                self._set_view_ply(self.view_ply - 1)
            elif name == "next":
                self._set_view_ply(self.view_ply + 1)
            else:
                self._set_view_ply(self._latest_ply())
            return True
        return False

    def _record_move(self, move: Move, move_eval: MoveEval) -> None:
        mover = self.game.current_player
        self.move_records.append(MoveRecord(move=move, mover=mover, eval=move_eval))

    def _queue_postgame_analysis(self) -> None:
        if not self.analysis_enabled or self.gpu_ai is None or self.analysis_running:
            return
        pending = [i for i, rec in enumerate(self.move_records) if rec.eval.q_value is None]
        if not pending:
            return
        self.analysis_running = True
        moves_prefix = [rec.move for rec in self.move_records]
        gpu_ai = self.gpu_ai

        def worker() -> None:
            updates: list[tuple[int, MoveEval]] = []
            for idx in pending:
                move_eval = gpu_ai.analyze_prefix(moves_prefix[:idx])
                updates.append((idx, move_eval))
            with self._analysis_lock:
                self._analysis_pending.extend(updates)
            self.analysis_running = False

        threading.Thread(target=worker, daemon=True).start()

    # ── Restart ────────────────────────────────────────────────────

    def _restart(self) -> None:
        self._game_gen += 1
        self.game = GameState(expansions=self.expansions)
        self.move_number = 0
        self.last_move = None
        self.move_records = []
        self.history_states = [self.game.copy()]
        self.view_ply = 0
        self._clear_selection()
        self.hover_hex = None
        self.cam_x = 0.0
        self.cam_y = 0.0
        with self._ai_lock:
            self._ai_result = None
        with self._analysis_lock:
            self._analysis_pending = []
        self.analysis_running = False
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
                    elif event.key == pygame.K_a:
                        self.analysis_enabled = not self.analysis_enabled
                    elif event.key == pygame.K_c:
                        self.center_camera()
                    elif event.key == pygame.K_LEFT:
                        self._set_view_ply(self.view_ply - 1)
                    elif event.key == pygame.K_RIGHT:
                        self._set_view_ply(self.view_ply + 1)
                    elif event.key == pygame.K_HOME:
                        self._set_view_ply(0)
                    elif event.key == pygame.K_END:
                        self._set_view_ply(self._latest_ply())
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
                move, move_eval, gen = pending
                if gen == self._game_gen:
                    self._apply_ai(move, move_eval)

            with self._analysis_lock:
                analysis_updates = self._analysis_pending
                self._analysis_pending = []
            for idx, move_eval in analysis_updates:
                if 0 <= idx < len(self.move_records):
                    self.move_records[idx].eval = move_eval

            # Auto-pass if current player has no choice
            if (
                self._is_live_view()
                and
                self.game.result == GameResult.IN_PROGRESS
                and (self.self_play or self.game.current_player == self.human_color)
                and not self.ai_thinking
            ):
                self._auto_pass()

            # Start AI thinking when it's the AI's turn (not in self-play)
            if (
                self._is_live_view()
                and
                not self.self_play
                and self.game.result == GameResult.IN_PROGRESS
                and self.game.current_player != self.human_color
                and not self.ai_thinking
            ):
                self._start_ai_think()

            if self.game.result != GameResult.IN_PROGRESS:
                self._queue_postgame_analysis()

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
    For GNN / NNUE the first element is the network and encoder
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


def _expansion_mask_from_config(expansions: ExpansionConfig | None) -> int:
    """Convert CPU expansion config to the GPU expansion bitmask."""
    if expansions is None:
        return 0
    return (
        (1 if expansions.mosquito else 0)
        | (2 if expansions.ladybug else 0)
        | (4 if expansions.pillbug else 0)
    )


# ── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Hive against a trained AI"
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "fnn-cpu", "legacy"],
        default="auto",
        help=(
            "AI engine to use. 'auto' prefers the GPU/legacy path when CUDA is "
            "available and falls back to 'fnn-cpu' otherwise."
        ),
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
            "Defaults: FNN=1024, PRS v2=256, GNN/NNUE=200."
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
    selected_engine = args.engine
    if selected_engine == "auto":
        selected_engine = "legacy" if torch.cuda.is_available() else "fnn-cpu"

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
    elif selected_engine == "fnn-cpu":
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
    elif selected_engine == "fnn-cpu" and not exp_str and not args.base_game:
        expansion_config = ExpansionConfig(mosquito=True, ladybug=True, pillbug=True)

    if gpu_ai is not None:
        gpu_ai.set_expansion_mask(_expansion_mask_from_config(expansion_config))

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
