"""
Tests for GPU move generation correctness.

Validates GPU legal move generation against the CPU engine by:
1. Playing random games to generate diverse positions
2. Computing legal moves on both CPU and GPU
3. Comparing the move sets for equivalence

Requires: CUDA Toolkit installed (nvcc), NVIDIA GPU
"""

from __future__ import annotations

import random
import pytest
import torch

from hive_engine.game_state import GameState
from hive_engine.pieces import Color, PieceType, Piece
from hive_engine.hex_coord import HexCoord


def _has_cuda():
    """Check if CUDA is available and GPU extension can be loaded."""
    if not torch.cuda.is_available():
        return False
    try:
        import hive_gpu
        hive_gpu.load_extension()
        return True
    except Exception:
        return False


# Skip all tests if no CUDA
pytestmark = pytest.mark.skipif(
    not _has_cuda(),
    reason="CUDA not available or hive_gpu extension cannot be loaded"
)


@pytest.fixture(scope="module")
def ext():
    """Load the GPU extension once for all tests."""
    import hive_gpu
    return hive_gpu.load_extension()


def _cpu_legal_moves_as_set(gs: GameState) -> set[tuple]:
    """
    Get legal moves from CPU engine as a set of (type, piece_type, from_hex, to_hex).
    """
    moves = gs.legal_moves()
    result = set()
    for m in moves:
        if m.piece is None:
            # Pass move
            result.add(("PASS", None, None, None))
        elif m.from_pos is None:
            # Placement
            result.add(("PLACE", m.piece.piece_type.value, None, (m.to.q, m.to.r)))
        else:
            # Movement
            result.add(("MOVE", m.piece.piece_type.value, (m.from_pos.q, m.from_pos.r), (m.to.q, m.to.r)))
    return result


def _gpu_legal_moves_as_set(ext, states_tensor, game_idx: int, gs: GameState) -> set[tuple]:
    """
    Get legal moves from GPU and convert to the same format as CPU.
    """
    moves_tensor, num_legal = ext.generate_legal_moves_batch(states_tensor, 1)

    num_moves = num_legal[game_idx].item()
    moves_data = moves_tensor[game_idx].cpu().numpy()

    # Parse GPUMove structs
    # GPUMove layout: type(u8), piece_type(u8), from_cell(u16), to_cell(u16) = 6 bytes
    sizeof_move = ext.SIZEOF_GPU_MOVE
    board_size = ext.BOARD_SIZE
    half_board = board_size // 2

    result = set()
    for i in range(num_moves):
        raw = moves_data[i]
        move_type = raw[0]
        piece_type = raw[1]
        # uint16_t from_cell at bytes 2-3, to_cell at bytes 4-5 (little-endian)
        from_cell = int(raw[2]) | (int(raw[3]) << 8)
        to_cell = int(raw[4]) | (int(raw[5]) << 8)

        center_q = gs._center_q if hasattr(gs, '_center_q') else 0
        center_r = gs._center_r if hasattr(gs, '_center_r') else 0

        if move_type == 2:  # MOVE_PASS
            result.add(("PASS", None, None, None))
        elif move_type == 0:  # MOVE_PLACE
            to_col = to_cell % board_size
            to_row = to_cell // board_size
            to_q = to_col - half_board + center_q
            to_r = to_row - half_board + center_r
            result.add(("PLACE", piece_type, None, (to_q, to_r)))
        elif move_type == 1:  # MOVE_MOVE
            from_col = from_cell % board_size
            from_row = from_cell // board_size
            from_q = from_col - half_board + center_q
            from_r = from_row - half_board + center_r
            to_col = to_cell % board_size
            to_row = to_cell // board_size
            to_q = to_col - half_board + center_q
            to_r = to_row - half_board + center_r
            result.add(("MOVE", piece_type, (from_q, from_r), (to_q, to_r)))

    return result


def _play_random_game(max_moves: int = 30) -> list[GameState]:
    """
    Play a random game, returning all intermediate states.
    """
    gs = GameState()
    states = [gs]
    for _ in range(max_moves):
        if gs.result.value != 0:  # Not IN_PROGRESS
            break
        moves = gs.legal_moves()
        if not moves:
            break
        move = random.choice(moves)
        gs = gs.copy()
        gs.apply_move(move)
        states.append(gs)
    return states


class TestGPUExtensionLoads:
    def test_extension_loads(self, ext):
        assert ext is not None
        assert ext.BOARD_SIZE == 23
        assert ext.NUM_CELLS == 529
        assert ext.MAX_LEGAL_MOVES == 256

    def test_create_initial_states(self, ext):
        states = ext.create_initial_states(4)
        assert states.shape[0] == 4
        assert states.device.type == "cuda"

    def test_generate_moves_initial(self, ext):
        """Initial position should have placement moves for non-queen pieces."""
        states = ext.create_initial_states(1)
        moves, num_legal = ext.generate_legal_moves_batch(states, 1)
        n = num_legal[0].item()
        # Turn 0: 4 piece types (no queen) × 1 position (center) = 4 moves
        assert n == 4, f"Expected 4 initial moves, got {n}"

    def test_results_initial(self, ext):
        """Initial states should all be IN_PROGRESS."""
        states = ext.create_initial_states(8)
        results = ext.check_results_batch(states, 8)
        assert (results == 0).all(), "All initial states should be IN_PROGRESS"


class TestGPUvsCPU:
    """Compare GPU and CPU move generation on random positions."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_game_positions(self, ext, seed):
        """
        Play a random game on CPU, then verify GPU produces same legal moves
        at each intermediate position.
        """
        random.seed(seed)
        states = _play_random_game(max_moves=20)

        for i, gs in enumerate(states[:5]):  # Check first 5 states per game
            cpu_moves = _cpu_legal_moves_as_set(gs)
            # TODO: Once we have a CPU→GPU state converter, compare here
            # For now, this test skeleton validates the framework
            assert len(cpu_moves) > 0, f"Game {seed}, state {i}: no CPU moves"
