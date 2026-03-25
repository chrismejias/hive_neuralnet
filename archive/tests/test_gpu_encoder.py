"""
Tests for GPU state encoder — validates GPU encoder output matches CPU encoder.

Compares:
  1. Node features (22-dim) between GPU and CPU encoders
  2. Edge structure and features
  3. Global features
  4. GNN batch format (HiveGraphBatch) compatibility
  5. Transformer batch format (HiveTokenBatch) compatibility
"""

import random

import numpy as np
import pytest
import torch

from hive_engine.game_state import GameState, GameResult, MoveType
from hive_engine.pieces import Color, PieceType
from hive_gnn.graph_encoder import GraphEncoder
from hive_transformer.token_encoder import TokenEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _have_gpu_ext():
    """Check if the GPU extension is available."""
    try:
        import hive_gpu
        ext = hive_gpu.load_extension()
        ext.initialize_tables()
        return True
    except Exception:
        return False


def _play_random_moves_parallel(ext, gs, gpu_states, n, rng):
    """Play n random moves on both CPU and GPU simultaneously.

    Returns (gs, gpu_states) after the moves.
    """
    BOARD_SIZE = ext.BOARD_SIZE
    HALF = BOARD_SIZE // 2

    for _ in range(n):
        if gs.result != GameResult.IN_PROGRESS:
            break
        cpu_moves = gs.legal_moves()
        if not cpu_moves:
            break

        # Pick a random CPU move
        move = rng.choice(cpu_moves)
        gs.apply_move(move)

        # Find and apply the matching move on GPU
        moves_tensor, num_legal = ext.generate_legal_moves_batch(gpu_states, 1)
        n_legal = num_legal[0].item()

        # Convert CPU move to GPU coordinates
        if move.move_type == MoveType.PLACE:
            cpu_type = 0  # MOVE_PLACE
            cpu_to = (move.to.r + HALF) * BOARD_SIZE + (move.to.q + HALF)
            cpu_pt = move.piece.piece_type.value + 1  # GPU is 1-indexed
        elif move.move_type == MoveType.MOVE:
            cpu_type = 1  # MOVE_MOVE
            cpu_from = (move.from_pos.r + HALF) * BOARD_SIZE + (move.from_pos.q + HALF)
            cpu_to = (move.to.r + HALF) * BOARD_SIZE + (move.to.q + HALF)
        else:
            # PASS move — find pass in GPU moves
            cpu_type = 2
            cpu_to = 0  # unused

        found = False
        for i in range(n_legal):
            raw = moves_tensor[0, i].cpu().numpy()
            gpu_move_type = raw[0]
            piece_type = raw[1]
            from_cell = int.from_bytes(raw[2:4].tobytes(), 'little')
            to_cell = int.from_bytes(raw[4:6].tobytes(), 'little')

            if move.move_type == MoveType.PASS:
                if gpu_move_type == 2:
                    single_move = moves_tensor[0, i:i+1].reshape(1, -1)
                    ext.apply_moves_batch(gpu_states, single_move, 1)
                    found = True
                    break
            elif gpu_move_type == cpu_type and to_cell == cpu_to:
                if move.move_type == MoveType.PLACE:
                    if piece_type == cpu_pt:
                        single_move = moves_tensor[0, i:i+1].reshape(1, -1)
                        ext.apply_moves_batch(gpu_states, single_move, 1)
                        found = True
                        break
                else:
                    if from_cell == cpu_from:
                        single_move = moves_tensor[0, i:i+1].reshape(1, -1)
                        ext.apply_moves_batch(gpu_states, single_move, 1)
                        found = True
                        break

        if not found:
            raise RuntimeError(f"Could not find matching GPU move for: {move}")

    return gs, gpu_states


def _sort_key(row):
    """Sort key for matching nodes between GPU and CPU (order-independent)."""
    return tuple(row[:7].tolist()) + (row[21],)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

requires_gpu = pytest.mark.skipif(
    not _have_gpu_ext(), reason="GPU extension not available"
)


@requires_gpu
class TestGPUEncoderFeatures:
    """Compare GPU-encoded features against CPU encoder."""

    def setup_method(self):
        import hive_gpu
        self.ext = hive_gpu.load_extension()
        self.ext.initialize_tables()
        self.cpu_graph_enc = GraphEncoder()

    def _compare_features(self, gs, gpu_states, label=""):
        """Compare GPU and CPU encoder outputs for matched game states."""
        # CPU encode
        cpu_graph = self.cpu_graph_enc.encode(gs)
        cpu_nf = cpu_graph.node_features  # (N, 22)
        cpu_gf = cpu_graph.global_features  # (6,)

        # GPU encode
        (
            node_features, node_grid_pos, node_piece_types,
            global_features, num_nodes, num_board_nodes,
            edge_index, edge_features, num_edges,
        ) = self.ext.encode_states_batch(gpu_states, 1)

        n_gpu = num_nodes[0].item()
        nb_gpu = num_board_nodes[0].item()

        gpu_nf = node_features[0, :n_gpu].cpu().numpy()
        gpu_gf = global_features[0].cpu().numpy()
        gpu_pos = node_grid_pos[0, :nb_gpu].cpu().numpy()  # (nb, 2)
        gpu_pt = node_piece_types[0, :nb_gpu].cpu().numpy()

        # Check node counts match
        n_cpu = cpu_nf.shape[0]
        assert n_gpu == n_cpu, (
            f"{label}: node count mismatch: GPU={n_gpu}, CPU={n_cpu}"
        )

        # Check board node count
        assert nb_gpu == cpu_graph.num_piece_nodes, (
            f"{label}: board node count mismatch: GPU={nb_gpu}, CPU={cpu_graph.num_piece_nodes}"
        )

        # Match board nodes by (grid_row, grid_col, piece_type, stack_position)
        cpu_board = cpu_nf[:cpu_graph.num_piece_nodes]
        gpu_board = gpu_nf[:nb_gpu]
        cpu_pos = cpu_graph.node_positions  # (nb, 2)
        cpu_pt_arr = cpu_graph.node_piece_types

        def board_key_cpu(i):
            return (int(cpu_pos[i, 0]), int(cpu_pos[i, 1]),
                    int(cpu_pt_arr[i]), float(cpu_board[i, 21]))

        def board_key_gpu(i):
            return (int(gpu_pos[i, 0]), int(gpu_pos[i, 1]),
                    int(gpu_pt[i]), float(gpu_board[i, 21]))

        cpu_sorted = sorted(range(len(cpu_board)), key=board_key_cpu)
        gpu_sorted = sorted(range(len(gpu_board)), key=board_key_gpu)

        for ci, gi in zip(cpu_sorted, gpu_sorted):
            ck = board_key_cpu(ci)
            gk = board_key_gpu(gi)
            assert ck == gk, (
                f"{label}: board node key mismatch: CPU={ck}, GPU={gk}"
            )
            np.testing.assert_allclose(
                gpu_board[gi], cpu_board[ci],
                atol=1e-5,
                err_msg=(
                    f"{label}: board node feature mismatch at pos=({ck[0]},{ck[1]}), "
                    f"pt={ck[2]}, stack_pos={ck[3]}"
                )
            )

        # Compare hand nodes by (piece_type, color) key
        n_hand_cpu = n_cpu - cpu_graph.num_piece_nodes
        n_hand_gpu = n_gpu - nb_gpu
        assert n_hand_gpu == n_hand_cpu, (
            f"{label}: hand node count mismatch: GPU={n_hand_gpu}, CPU={n_hand_cpu}"
        )

        if n_hand_cpu > 0:
            cpu_hand = cpu_nf[cpu_graph.num_piece_nodes:]
            gpu_hand = gpu_nf[nb_gpu:]

            def hand_key(row):
                return tuple(row[:7].tolist())

            cpu_hand_sorted = sorted(range(n_hand_cpu), key=lambda i: hand_key(cpu_hand[i]))
            gpu_hand_sorted = sorted(range(n_hand_gpu), key=lambda i: hand_key(gpu_hand[i]))

            for ci, gi in zip(cpu_hand_sorted, gpu_hand_sorted):
                np.testing.assert_allclose(
                    gpu_hand[gi], cpu_hand[ci],
                    atol=1e-5,
                    err_msg=f"{label}: hand node feature mismatch"
                )

        # Compare global features
        np.testing.assert_allclose(
            gpu_gf, cpu_gf, atol=1e-5,
            err_msg=f"{label}: global feature mismatch"
        )

    def test_initial_position(self):
        """Initial position: only hand nodes, no board pieces."""
        gs = GameState()
        gpu_states = self.ext.create_initial_states(1)
        self._compare_features(gs, gpu_states, "initial")

    def test_after_few_moves(self):
        """After 6 moves, compare board + hand features."""
        gs = GameState()
        gpu_states = self.ext.create_initial_states(1)
        rng = random.Random(123)
        gs, gpu_states = _play_random_moves_parallel(
            self.ext, gs, gpu_states, 6, rng
        )
        self._compare_features(gs, gpu_states, "6 moves")

    def test_after_many_moves(self):
        """After 20 moves with stacks possible."""
        gs = GameState()
        gpu_states = self.ext.create_initial_states(1)
        rng = random.Random(456)
        gs, gpu_states = _play_random_moves_parallel(
            self.ext, gs, gpu_states, 20, rng
        )
        self._compare_features(gs, gpu_states, "20 moves")

    def test_multiple_random_games(self):
        """Run 5 random games, compare at turn 10."""
        for game_idx in range(5):
            gs = GameState()
            gpu_states = self.ext.create_initial_states(1)
            rng = random.Random(1000 + game_idx)
            gs, gpu_states = _play_random_moves_parallel(
                self.ext, gs, gpu_states, 10, rng
            )
            self._compare_features(gs, gpu_states, f"game{game_idx}")


@requires_gpu
class TestGPUEncoderEdges:
    """Validate GPU edge construction matches CPU."""

    def setup_method(self):
        import hive_gpu
        self.ext = hive_gpu.load_extension()
        self.ext.initialize_tables()
        self.cpu_enc = GraphEncoder()

    def test_edge_counts_match(self):
        """Edge counts should match between GPU and CPU after several moves."""
        gs = GameState()
        gpu_states = self.ext.create_initial_states(1)
        rng = random.Random(789)
        gs, gpu_states = _play_random_moves_parallel(
            self.ext, gs, gpu_states, 10, rng
        )

        cpu_graph = self.cpu_enc.encode(gs)
        cpu_n_edges = cpu_graph.edge_index.shape[1]

        results = self.ext.encode_states_batch(gpu_states, 1)
        gpu_n_edges = results[8][0].item()

        assert gpu_n_edges == cpu_n_edges, (
            f"Edge count mismatch: GPU={gpu_n_edges}, CPU={cpu_n_edges}"
        )

    def test_edge_feature_consistency(self):
        """Verify edge features have valid structure."""
        gs = GameState()
        gpu_states = self.ext.create_initial_states(1)
        rng = random.Random(321)
        gs, gpu_states = _play_random_moves_parallel(
            self.ext, gs, gpu_states, 10, rng
        )

        results = self.ext.encode_states_batch(gpu_states, 1)
        edge_features = results[7]
        num_edges = results[8][0].item()

        ef = edge_features[0, :num_edges].cpu().numpy()

        for e in range(num_edges):
            is_stacked = ef[e, 8]
            if is_stacked > 0.5:
                # Vertical edge: dq=0, dr=0, no direction one-hot
                assert ef[e, 0] == 0.0 and ef[e, 1] == 0.0, (
                    f"Stacked edge {e} has nonzero dq/dr"
                )
                assert ef[e, 2:8].sum() == 0.0, (
                    f"Stacked edge {e} has direction one-hot"
                )
            else:
                # Spatial edge: exactly one direction one-hot should be set
                dir_sum = ef[e, 2:8].sum()
                assert dir_sum == 1.0, f"Spatial edge {e}: dir one-hot sum = {dir_sum}"
                dq, dr = ef[e, 0], ef[e, 1]
                valid_offsets = {(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)}
                assert (dq, dr) in valid_offsets, f"Invalid offset ({dq}, {dr})"


@requires_gpu
class TestGPUGNNEncoder:
    """Test the GPUGNNEncoder Python wrapper."""

    def setup_method(self):
        import hive_gpu
        self.ext = hive_gpu.load_extension()
        self.ext.initialize_tables()

    def test_gnn_batch_format(self):
        """GPUGNNEncoder produces valid HiveGraphBatch."""
        from hive_gpu.gpu_encoder import GPUGNNEncoder

        encoder = GPUGNNEncoder()
        states = self.ext.create_initial_states(4)
        batch = encoder.encode_batch(states, 4)

        assert batch.node_features.dim() == 2
        assert batch.node_features.shape[1] == 25
        assert batch.edge_index.dim() == 2
        assert batch.edge_index.shape[0] == 2
        assert batch.global_features.shape == (4, 6)
        assert batch.batch.shape[0] == batch.node_features.shape[0]

    def test_gnn_batch_after_moves(self):
        """GPUGNNEncoder works after playing moves."""
        from hive_gpu.gpu_encoder import GPUGNNEncoder

        encoder = GPUGNNEncoder()
        batch_size = 2
        states = self.ext.create_initial_states(batch_size)

        for _ in range(4):
            moves_tensor, num_legal = self.ext.generate_legal_moves_batch(states, batch_size)
            selected = torch.stack([moves_tensor[i, 0] for i in range(batch_size)])
            self.ext.apply_moves_batch(states, selected, batch_size)

        batch = encoder.encode_batch(states, batch_size)
        assert batch.node_features.shape[0] > 0
        assert batch.node_positions.shape[0] > 0


@requires_gpu
class TestGPUTransformerEncoder:
    """Test the GPUTransformerEncoder Python wrapper."""

    def setup_method(self):
        import hive_gpu
        self.ext = hive_gpu.load_extension()
        self.ext.initialize_tables()

    def test_transformer_batch_format(self):
        """GPUTransformerEncoder produces valid HiveTokenBatch."""
        from hive_gpu.gpu_encoder import GPUTransformerEncoder

        encoder = GPUTransformerEncoder()
        states = self.ext.create_initial_states(4)
        batch = encoder.encode_batch(states, 4)

        assert batch.token_features.dim() == 3
        assert batch.token_features.shape[0] == 4
        assert batch.token_features.shape[2] == 25
        assert batch.token_positions.shape == batch.token_types.shape
        assert batch.attention_mask.shape == batch.token_types.shape
        assert batch.global_features.shape == (4, 6)

        # CLS token should be first
        assert (batch.token_types[:, 0] == 0).all()

        # At initial state, all active tokens after CLS should be hand type
        for i in range(4):
            slen = batch.seq_lengths[i].item()
            active_types = batch.token_types[i, 1:slen]
            assert (active_types == 2).all()

    def test_transformer_batch_after_moves(self):
        """GPUTransformerEncoder works after playing moves."""
        from hive_gpu.gpu_encoder import GPUTransformerEncoder

        encoder = GPUTransformerEncoder()
        batch_size = 2
        states = self.ext.create_initial_states(batch_size)

        for _ in range(6):
            moves_tensor, num_legal = self.ext.generate_legal_moves_batch(states, batch_size)
            selected = torch.stack([moves_tensor[i, 0] for i in range(batch_size)])
            self.ext.apply_moves_batch(states, selected, batch_size)

        batch = encoder.encode_batch(states, batch_size)

        has_board = (batch.token_types == 1).any()
        assert has_board, "Expected board tokens after placing pieces"

        for i in range(batch_size):
            slen = batch.seq_lengths[i].item()
            assert batch.attention_mask[i, :slen].all()
            if slen < batch.attention_mask.shape[1]:
                assert not batch.attention_mask[i, slen:].any()


@requires_gpu
class TestGPUEncoderGlobalFeatures:
    """Validate global features specifically."""

    def setup_method(self):
        import hive_gpu
        self.ext = hive_gpu.load_extension()
        self.ext.initialize_tables()

    def test_initial_global_features(self):
        """Initial state: WHITE to move, turn 0, no queens, full hands."""
        states = self.ext.create_initial_states(1)
        results = self.ext.encode_states_batch(states, 1)
        gf = results[3][0].cpu().numpy()

        assert gf[0] == 1.0, "current_player should be WHITE (1.0)"
        assert gf[1] == 0.0, "turn/100 should be 0"
        assert gf[2] == 0.0, "white queen not placed"
        assert gf[3] == 0.0, "black queen not placed"
        assert abs(gf[4] - 11.0 / 14.0) < 1e-5, "white hand should be full"
        assert abs(gf[5] - 11.0 / 14.0) < 1e-5, "black hand should be full"

    def test_global_features_after_moves(self):
        """After moves, turn count and hand sizes should update."""
        states = self.ext.create_initial_states(1)

        for _ in range(2):
            moves_tensor, num_legal = self.ext.generate_legal_moves_batch(states, 1)
            single_move = moves_tensor[0, 0:1].reshape(1, -1)
            self.ext.apply_moves_batch(states, single_move, 1)

        results = self.ext.encode_states_batch(states, 1)
        gf = results[3][0].cpu().numpy()

        assert gf[1] == pytest.approx(2.0 / 100.0, abs=1e-5), "turn should be 2"
        assert gf[4] == pytest.approx(10.0 / 14.0, abs=1e-5)
        assert gf[5] == pytest.approx(10.0 / 14.0, abs=1e-5)
