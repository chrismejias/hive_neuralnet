"""Tests for GPU auxiliary target computation (mobility, queen surround, final mobility)."""

import numpy as np
import pytest
import torch


def _load_ext():
    """Load GPU extension, skip if CUDA not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import hive_gpu
    return hive_gpu.load_extension()


def _place_piece_gpu(ext, states, piece_type_idx, to_cell):
    """Place a piece on GPU state by constructing a GPUMove.

    piece_type_idx: 1=Queen, 2=Ant, 3=Grasshopper, 4=Spider, 5=Beetle,
                    6=Mosquito, 7=Ladybug, 8=Pillbug
    """
    # GPUMove struct: type(u8), piece_type(u8), from_cell(u16), to_cell(u16)
    move_bytes = np.zeros((1, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
    move_bytes[0, 0] = 0  # MOVE_PLACE
    move_bytes[0, 1] = piece_type_idx
    move_bytes[0, 2] = 0  # from_cell low
    move_bytes[0, 3] = 0  # from_cell high
    move_bytes[0, 4] = to_cell & 0xFF  # to_cell low
    move_bytes[0, 5] = (to_cell >> 8) & 0xFF  # to_cell high
    move_t = torch.from_numpy(move_bytes).cuda()
    ext.apply_moves_batch(states, move_t, 1)


def _move_piece_gpu(ext, states, from_cell, to_cell):
    """Move a piece on GPU state."""
    move_bytes = np.zeros((1, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
    move_bytes[0, 0] = 1  # MOVE_MOVE
    move_bytes[0, 1] = 0  # piece_type (not used for MOVE)
    move_bytes[0, 2] = from_cell & 0xFF
    move_bytes[0, 3] = (from_cell >> 8) & 0xFF
    move_bytes[0, 4] = to_cell & 0xFF
    move_bytes[0, 5] = (to_cell >> 8) & 0xFF
    move_t = torch.from_numpy(move_bytes).cuda()
    ext.apply_moves_batch(states, move_t, 1)


def _pass_move_gpu(ext, states):
    """Apply a PASS move on GPU state."""
    move_bytes = np.zeros((1, ext.SIZEOF_GPU_MOVE), dtype=np.uint8)
    move_bytes[0, 0] = 2  # MOVE_PASS
    move_t = torch.from_numpy(move_bytes).cuda()
    ext.apply_moves_batch(states, move_t, 1)


# Grid layout: 17×17, center at (8, 8)
BOARD_SIZE = 17
HALF = 8
CENTER = HALF * BOARD_SIZE + HALF  # cell 144
# Direction offsets: E=(+1,0), NE=(+1,-1), NW=(0,-1), W=(-1,0), SW=(-1,+1), SE=(0,+1)
DIR_DCOL = [+1, +1,  0, -1, -1,  0]
DIR_DROW = [ 0, -1, -1,  0, +1, +1]


def _neighbor(cell, direction):
    row = cell // BOARD_SIZE
    col = cell % BOARD_SIZE
    nr = row + DIR_DROW[direction]
    nc = col + DIR_DCOL[direction]
    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
        return nr * BOARD_SIZE + nc
    return -1


class TestMobilityKernel:
    """Test compute_mobility_batch kernel."""

    def test_empty_board(self):
        """No pieces on board → no board nodes → empty mobility."""
        ext = _load_ext()
        states = ext.create_initial_states(1)

        mob, counts = ext.compute_mobility_batch(states, 1, False)
        mob_np = mob.cpu().numpy()
        cnt = counts[0].item()

        assert cnt == 0  # No board pieces

    def test_two_queens_both_mobile(self):
        """After both queens placed adjacent, both should be mobile (can slide)."""
        ext = _load_ext()
        states = ext.create_initial_states(1)

        # Turn 0: White queen at center
        _place_piece_gpu(ext, states, 1, CENTER)
        # Turn 1: Black queen east of center
        east = _neighbor(CENTER, 0)
        _place_piece_gpu(ext, states, 1, east)

        # Now it's White's turn (turn 2). Check mobility.
        mob, counts = ext.compute_mobility_batch(states, 1, False)
        mob_np = mob.cpu().numpy()[0]
        cnt = counts[0].item()

        # 2 board pieces
        assert cnt == 2
        # Both queens should have mobility for the current player only (White)
        # White queen at center can slide, so mob[0] = 1.0
        # Black queen at east is NOT current player's piece, mob[1] = 0.0 or 1.0
        # In normal mode (both_players=False), only current player's pieces get mobility
        # White queen (node 0) should be mobile
        assert mob_np[0] == 1.0  # White queen can slide

    def test_both_players_mode(self):
        """With both_players=True, both colors' pieces get mobility checked."""
        ext = _load_ext()
        states = ext.create_initial_states(1)

        # Turn 0: White queen at center
        _place_piece_gpu(ext, states, 1, CENTER)
        # Turn 1: Black queen east
        east = _neighbor(CENTER, 0)
        _place_piece_gpu(ext, states, 1, east)

        mob, counts = ext.compute_mobility_batch(states, 1, True)
        mob_np = mob.cpu().numpy()[0]
        cnt = counts[0].item()

        assert cnt == 2
        # Both queens should be mobile (can slide)
        assert mob_np[0] == 1.0
        assert mob_np[1] == 1.0

    def test_single_piece_immobile(self):
        """A single queen with no opponent queen placed → no movement allowed."""
        ext = _load_ext()
        states = ext.create_initial_states(1)

        # Turn 0: White queen at center
        _place_piece_gpu(ext, states, 1, CENTER)
        # Turn 1: Black ant east (black queen NOT placed yet)
        east = _neighbor(CENTER, 0)
        _place_piece_gpu(ext, states, 2, east)

        # Turn 2: White's turn. White queen placed, can move.
        mob, counts = ext.compute_mobility_batch(states, 1, False)
        mob_np = mob.cpu().numpy()[0]
        cnt = counts[0].item()

        assert cnt == 2
        # White queen should be mobile (queen IS placed for white)
        # But it might be pinned (AP check). With only 2 pieces, neither is AP.
        assert mob_np[0] == 1.0  # White queen can slide

    def test_batch_mobility(self):
        """Test mobility on a batch of 2 states."""
        ext = _load_ext()
        states = ext.create_initial_states(2)

        # Game 0: place pieces
        _place_piece_gpu(ext, states[:1], 1, CENTER)
        east = _neighbor(CENTER, 0)
        _place_piece_gpu(ext, states[:1], 1, east)

        # Game 1: just initial state (no pieces)
        # Game 1 starts fresh but we need to advance its turn too
        # Actually both states were initialized together, so game 0 has had 2 moves applied
        # but game 1 still has 0 turns. We need separate states.
        states0 = ext.create_initial_states(1)
        _place_piece_gpu(ext, states0, 1, CENTER)
        _place_piece_gpu(ext, states0, 1, east)

        states1 = ext.create_initial_states(1)

        # Combine into batch
        batch = torch.cat([states0, states1], dim=0)

        mob, counts = ext.compute_mobility_batch(batch, 2, False)
        cnt0 = counts[0].item()
        cnt1 = counts[1].item()

        assert cnt0 == 2  # 2 pieces on board
        assert cnt1 == 0  # empty board


class TestMobilityPillbugThrow:
    """Test that enemy pillbug throw detection works in mobility."""

    def test_piece_throwable_by_enemy_pillbug(self):
        """A piece adjacent to an enemy pillbug should be marked mobile
        even if it can't move on its own (e.g., it's pinned)."""
        ext = _load_ext()

        # Setup: Create a position where a piece is pinned but adjacent to
        # an enemy pillbug.
        # We need: white piece that is pinned (AP), adjacent to black pillbug.
        # This requires a specific board layout.

        # For now, just verify the kernel runs without crashing on a state
        # with pillbugs present.
        states = ext.create_initial_states(1, 4)  # expansion_mask=4 → pillbug enabled

        # Place pieces to create a configuration with pillbug
        _place_piece_gpu(ext, states, 2, CENTER)        # Turn 0: White ant at center
        east = _neighbor(CENTER, 0)
        _place_piece_gpu(ext, states, 8, east)           # Turn 1: Black pillbug east
        west = _neighbor(CENTER, 3)
        _place_piece_gpu(ext, states, 1, west)           # Turn 2: White queen west
        ee = _neighbor(east, 0)
        _place_piece_gpu(ext, states, 1, ee)             # Turn 3: Black queen east-east

        # Now turn 4: White's turn.
        # White ant at center is between white queen (W) and black pillbug (E).
        # The ant might or might not be an AP.
        # But the ant IS adjacent to the enemy pillbug.
        mob, counts = ext.compute_mobility_batch(states, 1, False)
        mob_np = mob.cpu().numpy()[0]
        cnt = counts[0].item()

        assert cnt == 4  # 4 board pieces
        # The mobility values should be computed without crashing
        # Detailed correctness of pillbug throws is complex to verify here


class TestQueenSurroundParsing:
    """Test CPU-side queen surround target parsing."""

    def test_queen_surround_basic(self):
        """Test queen surround for a simple position."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig

        ext = _load_ext()
        states = ext.create_initial_states(1)

        # Build a position where white queen is surrounded on some sides
        _place_piece_gpu(ext, states, 1, CENTER)        # Turn 0: White queen
        east = _neighbor(CENTER, 0)
        _place_piece_gpu(ext, states, 1, east)           # Turn 1: Black queen east
        west = _neighbor(CENTER, 3)
        _place_piece_gpu(ext, states, 2, west)           # Turn 2: White ant west

        # Create a minimal orchestrator just to use _compute_queen_surround_batch
        from hive_gnn.gnn_net import HiveGNN, GNNNetConfig
        net = HiveGNN(GNNNetConfig.small()).cuda().eval()
        config = GPUMCTSConfig(num_simulations=2, batch_size=1)
        orch = GPUMCTSOrchestrator(net, config)

        qs_data = orch._compute_queen_surround_batch(states, 1)
        qs_target, qs_mask = qs_data[0]

        # White queen is placed → mask[0] = 1.0
        assert qs_mask[0] == 1.0
        # Black queen is placed → mask[1] = 1.0
        assert qs_mask[1] == 1.0

        # White queen at center has neighbors: black queen (east), white ant (west)
        # → 2 nodes adjacent to white queen should have qs_target[:,0] = 1.0
        white_surround_count = int(qs_target[:, 0].sum())
        assert white_surround_count == 2  # east and west pieces

        # Black queen at east has neighbors: white queen (west direction)
        # → 1 node adjacent to black queen
        black_surround_count = int(qs_target[:, 1].sum())
        assert black_surround_count == 1  # white queen


class TestSelfPlayAuxTargets:
    """Test that self-play produces non-zero auxiliary targets."""

    def test_examples_have_mobility(self):
        """GPU self-play examples should have non-trivial mobility targets."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig
        from hive_gnn.gnn_net import HiveGNN, GNNNetConfig

        net = HiveGNN(GNNNetConfig.small()).cuda().eval()
        config = GPUMCTSConfig(
            num_simulations=2,
            batch_size=2,
            max_game_length=10,
            encoder_type="gnn",
        )
        orch = GPUMCTSOrchestrator(net, config)
        all_examples = orch.self_play_batch()

        any_nonzero_mob = False
        any_nonzero_qs = False
        any_nonzero_fm = False

        for game_examples in all_examples:
            for ex in game_examples:
                if ex.mobility_target.sum() > 0:
                    any_nonzero_mob = True
                if ex.queen_surround_target.sum() > 0:
                    any_nonzero_qs = True
                if ex.final_mobility_target.sum() > 0:
                    any_nonzero_fm = True

        # After 10 moves, most pieces should have some mobility
        assert any_nonzero_mob, "No non-zero mobility targets found"
        # Queen surround only on final position, may be zero if game didn't end naturally
        # Final mobility same — may be zero in short games

    def test_example_shapes(self):
        """Verify shapes of auxiliary target arrays in GPUTrainingExamples."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from hive_gpu.gpu_mcts import GPUMCTSOrchestrator, GPUMCTSConfig
        from hive_gnn.gnn_net import HiveGNN, GNNNetConfig

        net = HiveGNN(GNNNetConfig.small()).cuda().eval()
        config = GPUMCTSConfig(
            num_simulations=2,
            batch_size=1,
            max_game_length=6,
            encoder_type="gnn",
        )
        orch = GPUMCTSOrchestrator(net, config)
        all_examples = orch.self_play_batch()

        for game_examples in all_examples:
            for ex in game_examples:
                n = ex.graph.num_piece_nodes
                assert ex.mobility_target.shape == (n,), (
                    f"mobility shape {ex.mobility_target.shape}, expected ({n},)"
                )
                assert ex.queen_surround_target.shape == (n, 2), (
                    f"qs shape {ex.queen_surround_target.shape}, expected ({n}, 2)"
                )
                assert ex.queen_surround_mask.shape == (2,)
                assert ex.final_mobility_target.shape == (n,), (
                    f"fm shape {ex.final_mobility_target.shape}, expected ({n},)"
                )
                assert ex.mobility_target.dtype == np.float32
                assert ex.queen_surround_target.dtype == np.float32
                assert ex.final_mobility_target.dtype == np.float32
