import torch

from hive_gpu.gpu_trainer import GPUTrainer
from hive_gpu.gumbel_mcts import GumbelAlphaZeroOrchestrator, GumbelConfig


def test_random_expansion_run_list_preserves_requested_game_count():
    run_list = GPUTrainer._build_run_list(
        games_per_batch=10,
        batches_per_iteration=1,
        expansion_mask=-1,
    )

    assert sum(size for _, size in run_list) == 10
    assert [mask for mask, _ in run_list] == list(range(8))
    assert max(size for _, size in run_list) - min(size for _, size in run_list) <= 1


def test_gumbel_improved_policy_stays_on_considered_actions():
    orchestrator = GumbelAlphaZeroOrchestrator.__new__(GumbelAlphaZeroOrchestrator)
    orchestrator.config = GumbelConfig(
        c_visit=50.0,
        c_scale=1.0,
        policy_target_pruning=0.0,
    )
    orchestrator._action_space_size = 4

    logits = torch.tensor([[1.0, 10.0, 2.0, -5.0]], dtype=torch.float32)
    topk_actions = torch.tensor([[0, 2]], dtype=torch.int64)
    q_sums = torch.tensor([[3.0, 1.0]], dtype=torch.float32)
    visit_counts = torch.tensor([[3, 1]], dtype=torch.int32)
    legal_mask = torch.tensor([[True, True, True, False]])
    root_values = torch.tensor([0.25], dtype=torch.float32)
    nn_prior_probs = torch.softmax(logits, dim=-1)

    policies, _ = orchestrator._compute_improved_policy(
        logits=logits,
        topk_actions=topk_actions,
        q_sums=q_sums,
        visit_counts=visit_counts,
        legal_mask=legal_mask,
        root_values=root_values,
        nn_prior_probs=nn_prior_probs,
        B=1,
        max_k=2,
        active=[True],
    )

    # policies[0] is now (action_indices [max_k], probs [max_k]) — sparse format.
    act_indices, probs = policies[0]
    # topk_actions were [0, 2], so only those two actions can have non-zero prob.
    assert 1 not in act_indices
    assert 3 not in act_indices
    assert abs(probs.sum() - 1.0) < 1e-6


def test_gumbel_matches_candidate_actions_to_legal_move_indices():
    orchestrator = GumbelAlphaZeroOrchestrator.__new__(GumbelAlphaZeroOrchestrator)

    actions = torch.tensor([[7, 3, 9], [4, 1, 2]], dtype=torch.int64)
    legal_action_indices = torch.tensor(
        [[5, 7, 9, -1], [4, 8, -1, -1]],
        dtype=torch.int32,
    )
    num_legal = torch.tensor([3, 2], dtype=torch.int32)

    move_indices = orchestrator._match_actions_to_legal_moves(
        actions, legal_action_indices, num_legal,
    )

    assert move_indices.tolist() == [[1, -1, 2], [0, -1, -1]]


def test_gumbel_gathers_scores_for_legal_moves_only():
    orchestrator = GumbelAlphaZeroOrchestrator.__new__(GumbelAlphaZeroOrchestrator)

    scores = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.8, 0.7, 0.6, 0.5]],
        dtype=torch.float32,
    )
    legal_action_indices = torch.tensor(
        [[3, 1, -1], [0, 2, -1]],
        dtype=torch.int32,
    )

    gathered = orchestrator._gather_legal_action_scores(scores, legal_action_indices)

    assert gathered[0, 0].item() == scores[0, 3].item()
    assert gathered[0, 1].item() == scores[0, 1].item()
    assert gathered[1, 0].item() == scores[1, 0].item()
    assert gathered[1, 1].item() == scores[1, 2].item()
    assert torch.isneginf(gathered[:, 2]).all()
