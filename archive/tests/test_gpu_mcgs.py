"""Tests for Monte Carlo Graph Search (MCGS) data structures and orchestrator."""

import numpy as np
import pytest

from hive_gpu.gpu_mcgs import MCGSEdge, MCGSNode, MCGSDAG


# ── MCGSEdge tests ─────────────────────────────────────────────────────


class TestMCGSEdge:
    def test_creation(self):
        mb = np.zeros(6, dtype=np.uint8)
        edge = MCGSEdge(action=42, move_bytes=mb, prior=0.5)
        assert edge.action == 42
        assert edge.prior == 0.5
        assert edge.visit_count == 0
        assert edge.total_value == 0.0
        assert edge.child_hash is None

    def test_mean_value_zero_visits(self):
        mb = np.zeros(6, dtype=np.uint8)
        edge = MCGSEdge(action=0, move_bytes=mb, prior=0.1)
        assert edge.mean_value == 0.0

    def test_mean_value_with_visits(self):
        mb = np.zeros(6, dtype=np.uint8)
        edge = MCGSEdge(action=0, move_bytes=mb, prior=0.1)
        edge.visit_count = 10
        edge.total_value = 5.0
        assert edge.mean_value == pytest.approx(0.5)

    def test_virtual_loss_decreases_q(self):
        """Virtual loss should make the edge look worse (lower Q)."""
        mb = np.zeros(6, dtype=np.uint8)
        edge = MCGSEdge(action=0, move_bytes=mb, prior=0.5)
        edge.visit_count = 10
        edge.total_value = 3.0  # Q = 0.3 (good for parent)
        q_before = edge.mean_value

        # Apply virtual loss
        edge.visit_count += 1
        edge.total_value -= 1.0
        q_after = edge.mean_value

        assert q_after < q_before


# ── MCGSNode tests ─────────────────────────────────────────────────────


class TestMCGSNode:
    def test_creation(self):
        node = MCGSNode(state_hash=12345)
        assert node.state_hash == 12345
        assert node.edges == {}
        assert node.visit_count == 0
        assert not node.is_expanded
        assert not node.is_terminal

    def test_mean_value(self):
        node = MCGSNode(state_hash=0)
        assert node.mean_value == 0.0
        node.visit_count = 4
        node.total_value = 2.0
        assert node.mean_value == pytest.approx(0.5)


# ── MCGSDAG tests ──────────────────────────────────────────────────────


class TestMCGSDAG:
    def test_get_or_create_new(self):
        dag = MCGSDAG()
        node, is_new = dag.get_or_create(42)
        assert is_new
        assert node.state_hash == 42
        assert 42 in dag
        assert len(dag) == 1

    def test_get_or_create_existing(self):
        dag = MCGSDAG()
        node1, is_new1 = dag.get_or_create(42)
        node2, is_new2 = dag.get_or_create(42)
        assert is_new1
        assert not is_new2
        assert node1 is node2
        assert len(dag) == 1

    def test_transposition_counter(self):
        dag = MCGSDAG()
        dag.get_or_create(1)
        dag.get_or_create(2)
        dag.get_or_create(1)  # transposition
        dag.get_or_create(2)  # transposition
        dag.get_or_create(3)
        assert dag.transposition_hits == 0  # get_or_create doesn't count hits
        # (hits are counted by the orchestrator during search)


# ── MCGS integration tests (require GPU) ───────────────────────────────

try:
    import torch
    import hive_gpu
    _ext = hive_gpu.load_extension()
    HAS_GPU = torch.cuda.is_available()
except Exception:
    HAS_GPU = False


@pytest.mark.skipif(not HAS_GPU, reason="Requires CUDA GPU and hive_gpu extension")
class TestMCGSOrchestrator:
    """Integration tests for the MCGS orchestrator."""

    def _make_orchestrator(self, num_sims=10, batch_size=2, wave_size=2):
        from hive_gpu.gpu_mcts import GPUMCTSConfig
        from hive_gpu.gpu_mcgs import MCGSOrchestrator
        from hive_gnn.gnn_net import GNNNetConfig, HiveGNN

        config = GPUMCTSConfig(
            num_simulations=num_sims,
            batch_size=batch_size,
            wave_size=wave_size,
            max_game_length=20,
            temperature_drop_move=5,
            playout_cap_randomize=False,
            policy_target_pruning=0.0,
            root_policy_temp=1.0,
            shaped_dirichlet=False,
        )
        net_config = GNNNetConfig.small()
        net = HiveGNN(net_config).to("cuda")
        net.eval()
        return MCGSOrchestrator(net, config)

    def test_self_play_produces_examples(self):
        """MCGS self-play should produce training examples."""
        orch = self._make_orchestrator(num_sims=10, batch_size=2)
        examples = orch.self_play_batch()
        assert len(examples) == 2  # one list per game
        for game_examples in examples:
            assert len(game_examples) > 0
            for ex in game_examples:
                assert ex.graph is not None
                assert ex.policy_target.shape == (orch._action_space_size,)
                assert ex.value_target in (-1.0, 0.0, 1.0)

    def test_dag_stats_accumulated(self):
        """MCGS should report DAG statistics."""
        orch = self._make_orchestrator(num_sims=20, batch_size=2)
        orch.self_play_batch()
        # After self-play, DAG stats should be populated
        assert orch.total_nn_evals > 0
        assert orch.total_dag_nodes > 0
        # transposition_hits may be 0 for very short games

    def test_policies_sum_to_one(self):
        """MCGS policies should be valid probability distributions."""
        orch = self._make_orchestrator(num_sims=10, batch_size=2)
        examples = orch.self_play_batch()
        for game_examples in examples:
            for ex in game_examples:
                total = ex.policy_target.sum()
                if total > 0:
                    assert abs(total - 1.0) < 1e-5

    def test_wave_size_1_works(self):
        """MCGS with wave_size=1 should work (sequential mode)."""
        orch = self._make_orchestrator(num_sims=5, batch_size=1, wave_size=1)
        examples = orch.self_play_batch()
        assert len(examples) == 1
        assert len(examples[0]) > 0

    def test_select_returns_valid_path(self):
        """Selection should return a valid path through the DAG."""
        orch = self._make_orchestrator(num_sims=5, batch_size=1)

        # Create a simple DAG with root + one child
        root = MCGSNode(state_hash=0)
        root.is_expanded = True
        mb = np.zeros(6, dtype=np.uint8)
        edge = MCGSEdge(action=0, move_bytes=mb, prior=1.0, child_hash=1)
        root.edges[0] = edge

        child = MCGSNode(state_hash=1)
        # child is NOT expanded → should be returned as leaf

        dag = MCGSDAG()
        dag.nodes[0] = root
        dag.nodes[1] = child

        leaf, path, vl_edges, move_path = orch._select_mcgs(root, dag)

        assert leaf is child
        assert len(path) == 1
        assert path[0] == (root, edge)
        assert len(vl_edges) == 1
        assert vl_edges[0] is edge

    def test_select_unresolved_edge(self):
        """Selection should return None for edges with no child_hash."""
        orch = self._make_orchestrator(num_sims=5, batch_size=1)

        root = MCGSNode(state_hash=0)
        root.is_expanded = True
        mb = np.zeros(6, dtype=np.uint8)
        edge = MCGSEdge(action=0, move_bytes=mb, prior=1.0)
        # child_hash is None → unresolved
        root.edges[0] = edge

        dag = MCGSDAG()
        dag.nodes[0] = root

        leaf, path, vl_edges, move_path = orch._select_mcgs(root, dag)

        assert leaf is None  # unresolved
        assert len(path) == 1
        assert path[0] == (root, edge)

    def test_backprop_updates_edges_and_nodes(self):
        """Backpropagation should update edge and node statistics."""
        orch = self._make_orchestrator()

        root = MCGSNode(state_hash=0)
        child = MCGSNode(state_hash=1)
        leaf = MCGSNode(state_hash=2)

        mb = np.zeros(6, dtype=np.uint8)
        e1 = MCGSEdge(action=0, move_bytes=mb, prior=0.5, child_hash=1)
        e2 = MCGSEdge(action=1, move_bytes=mb, prior=0.5, child_hash=2)
        root.edges[0] = e1
        child.edges[1] = e2

        path = [(root, e1), (child, e2)]
        vl_edges = [e1, e2]
        # Apply virtual loss
        for e in vl_edges:
            e.visit_count += 1
            e.total_value -= 1.0

        # Backprop with value=0.8 from leaf's perspective
        orch._backpropagate_mcgs(path, vl_edges, leaf, value=0.8)

        # Virtual loss should be undone
        # Leaf updated with its perspective
        assert leaf.visit_count == 1
        assert leaf.total_value == pytest.approx(0.8)

        # child→leaf edge: value from child's perspective = -0.8
        assert e2.visit_count == 1
        assert e2.total_value == pytest.approx(-0.8)
        # child node: from child's perspective = -0.8
        assert child.visit_count == 1
        assert child.total_value == pytest.approx(-0.8)

        # root→child edge: value from root's perspective = 0.8
        assert e1.visit_count == 1
        assert e1.total_value == pytest.approx(0.8)
        # root node: from root's perspective = 0.8
        assert root.visit_count == 1
        assert root.total_value == pytest.approx(0.8)

    def test_puct_prefers_high_prior_unvisited(self):
        """PUCT should prefer high-prior edges when all unvisited."""
        orch = self._make_orchestrator()
        mb = np.zeros(6, dtype=np.uint8)

        e_high = MCGSEdge(action=0, move_bytes=mb, prior=0.9)
        e_low = MCGSEdge(action=1, move_bytes=mb, prior=0.1)

        s_high = orch._puct_score_edge(10, e_high)
        s_low = orch._puct_score_edge(10, e_low)

        assert s_high > s_low

    def test_policy_extraction(self):
        """Policy should be proportional to edge visit counts."""
        orch = self._make_orchestrator()

        root = MCGSNode(state_hash=0)
        mb = np.zeros(6, dtype=np.uint8)
        e1 = MCGSEdge(action=5, move_bytes=mb, prior=0.5)
        e2 = MCGSEdge(action=10, move_bytes=mb, prior=0.5)
        e1.visit_count = 80
        e2.visit_count = 20
        root.edges[5] = e1
        root.edges[10] = e2

        policy = orch._get_policy_mcgs(root, move_number=0)
        assert policy[5] == pytest.approx(0.8)
        assert policy[10] == pytest.approx(0.2)

    def test_dirichlet_noise_changes_priors(self):
        """Dirichlet noise should modify edge priors."""
        orch = self._make_orchestrator()

        root = MCGSNode(state_hash=0)
        mb = np.zeros(6, dtype=np.uint8)
        for a in range(5):
            root.edges[a] = MCGSEdge(action=a, move_bytes=mb, prior=0.2)

        original_priors = [root.edges[a].prior for a in range(5)]
        orch._add_dirichlet_noise_mcgs(root)
        new_priors = [root.edges[a].prior for a in range(5)]

        # Priors should be modified
        assert original_priors != new_priors
        # Should still sum to ~1
        assert abs(sum(new_priors) - 1.0) < 1e-5
