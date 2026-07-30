import random
from pathlib import Path

import torch
import pytest

import hive_fnn.fnn_alphabeta_training as alpha_beta_training
from hive_fnn.fnn_alphabeta_training import (
    AlphaBetaGenerationConfig,
    AlphaBetaLossConfig,
    AlphaBetaRecord,
    AlphaBetaReplayPriorityConfig,
    AlphaBetaReplayBuffer,
    AlphaBetaTrainingBatch,
    alpha_beta_expansion_runs,
    compute_alpha_beta_loss,
    alpha_beta_value_targets,
    opening_diversity_candidates,
    sample_opening_move_index,
)
from hive_fnn.fnn_native_alphabeta import AlphaBetaBound
from hive_fnn.fnn_network import FNNConfig, HiveFNN
from hive_fnn.train_fnn_alphabeta import (
    _load_resume_state,
    _save_resume_state,
)


def test_all_expansion_combinations_preserve_game_count() -> None:
    runs = alpha_beta_expansion_runs(10, -1)

    assert [mask for mask, _count in runs] == list(range(8))
    assert sum(count for _mask, count in runs) == 10
    assert max(count for _mask, count in runs) - min(
        count for _mask, count in runs
    ) <= 1


def test_fixed_expansion_mask_uses_one_batch() -> None:
    assert alpha_beta_expansion_runs(512, 7) == [(7, 512)]


def test_alpha_beta_loss_trains_only_scalar_value(monkeypatch) -> None:
    batch = AlphaBetaTrainingBatch(
        states=torch.zeros((2, 1), dtype=torch.uint8),
        legal_moves=torch.zeros((2, 1, 1), dtype=torch.uint8),
        num_legal=torch.ones(2, dtype=torch.int64),
        search_values=torch.tensor([0.8, -0.4]),
        final_results=torch.tensor([-1.0, 1.0]),
    )
    monkeypatch.setattr(
        alpha_beta_training, "_root_values",
        lambda _net, _batch: torch.tensor([0.1, -0.2], requires_grad=True),
    )

    loss, components = compute_alpha_beta_loss(
        None, batch, AlphaBetaLossConfig(),
    )

    expected_targets = torch.tensor([-0.55, 0.65])
    expected = torch.nn.functional.mse_loss(
        torch.tensor([0.1, -0.2]), expected_targets,
    )
    assert loss.item() == pytest.approx(expected.item())
    assert set(components) == {"value_loss", "value_target_mean"}


def test_value_target_uses_player_to_move_convex_blend() -> None:
    search = torch.tensor([0.8, -0.4, 10.0])
    outcomes = torch.tensor([-1.0, 1.0, 1.0])

    targets = alpha_beta_value_targets(search, outcomes, 0.25)

    assert torch.allclose(targets, torch.tensor([-0.55, 0.65, 1.0]))


def test_unknown_outcome_uses_search_target_without_fake_draw() -> None:
    search = torch.tensor([0.8, -0.4])
    outcomes = torch.tensor([float("nan"), 1.0])

    targets = alpha_beta_value_targets(search, outcomes, 0.25)

    assert targets[0].item() == pytest.approx(0.8)
    assert targets[1].item() == pytest.approx(0.65)


def test_opening_diversity_uses_only_exact_near_best_moves() -> None:
    scores = torch.tensor([0.55, 0.50, 0.44, 0.53, float("nan"), 0.54])
    bounds = torch.tensor([
        AlphaBetaBound.EXACT,
        AlphaBetaBound.EXACT,
        AlphaBetaBound.EXACT,
        AlphaBetaBound.UPPER,
        AlphaBetaBound.EXACT,
        AlphaBetaBound.LOWER,
    ], dtype=torch.uint8)

    candidates = opening_diversity_candidates(
        scores,
        bounds,
        selected_index=0,
        max_candidates=4,
        value_window=0.06,
    )

    assert candidates == [0, 1]


def test_opening_diversity_caps_candidates_after_score_ordering() -> None:
    scores = torch.tensor([0.30, 0.50, 0.40, 0.45])
    bounds = torch.full((4,), AlphaBetaBound.EXACT, dtype=torch.uint8)

    candidates = opening_diversity_candidates(
        scores,
        bounds,
        selected_index=1,
        max_candidates=2,
        value_window=1.0,
    )

    assert candidates == [1, 3]


def test_opening_diversity_requires_exact_selected_move() -> None:
    scores = torch.tensor([0.5, 0.4])
    bounds = torch.tensor([
        AlphaBetaBound.UPPER,
        AlphaBetaBound.EXACT,
    ], dtype=torch.uint8)

    candidates = opening_diversity_candidates(
        scores,
        bounds,
        selected_index=0,
    )

    assert candidates == []


def test_opening_sampling_is_seed_reproducible() -> None:
    scores = torch.tensor([0.3, 0.2, 0.1])
    first_rng = random.Random(17)
    second_rng = random.Random(17)

    first = [
        sample_opening_move_index([0, 1, 2], scores, first_rng)
        for _ in range(12)
    ]
    second = [
        sample_opening_move_index([0, 1, 2], scores, second_rng)
        for _ in range(12)
    ]

    assert first == second
    assert len(set(first)) > 1


def test_positive_temperature_favors_better_exact_move() -> None:
    scores = torch.tensor([0.5, 0.0])
    rng = random.Random(3)

    samples = [
        sample_opening_move_index(
            [0, 1], scores, rng, temperature=0.1,
        )
        for _ in range(200)
    ]

    assert samples.count(0) > 190


def test_alpha_beta_replay_state_round_trip() -> None:
    record = AlphaBetaRecord(
        state=torch.tensor([1, 2, 3], dtype=torch.uint8),
        legal_moves=torch.tensor([[4, 5]], dtype=torch.uint8),
        raw_value=0.1,
        search_value=0.2,
        selected_index=0,
        pv_moves=torch.tensor([[4, 5]], dtype=torch.uint8),
        root_scores=torch.tensor([0.2]),
        root_bounds=torch.tensor([AlphaBetaBound.EXACT], dtype=torch.uint8),
        completed_depth=2,
        nodes=17,
        final_result=1.0,
    )
    original = AlphaBetaReplayBuffer(capacity=7)
    original.add([record])

    restored = AlphaBetaReplayBuffer(capacity=1)
    restored.load_state_dict(original.state_dict())

    assert restored.capacity == 7
    assert len(restored) == 1
    sampled = restored.sample(1, seed=3)[0]
    assert sampled.nodes == 17
    assert torch.equal(sampled.state, record.state)


def _priority_record(
    marker: int,
    *,
    raw_value: float = 0.0,
    search_value: float = 0.0,
    final_result: float = 0.0,
    depth_value_delta: float = 0.0,
    depth_move_changed: bool = False,
) -> AlphaBetaRecord:
    return AlphaBetaRecord(
        state=torch.tensor([marker], dtype=torch.uint8),
        legal_moves=torch.tensor([[marker, 0]], dtype=torch.uint8),
        raw_value=raw_value,
        search_value=search_value,
        selected_index=0,
        pv_moves=torch.tensor([[marker, 0]], dtype=torch.uint8),
        root_scores=torch.tensor([search_value]),
        root_bounds=torch.tensor([AlphaBetaBound.EXACT], dtype=torch.uint8),
        completed_depth=2,
        nodes=17,
        final_result=final_result,
        depth_value_delta=depth_value_delta,
        depth_move_changed=depth_move_changed,
    )


def test_prioritized_replay_favors_surprise_and_depth_change() -> None:
    replay = AlphaBetaReplayBuffer(
        capacity=2,
        priority_config=AlphaBetaReplayPriorityConfig(
            alpha=1.0,
            uniform_fraction=0.0,
            complexity_weight=0.0,
        ),
    )
    replay.add([
        _priority_record(1),
        _priority_record(
            2,
            raw_value=-1.0,
            search_value=1.0,
            final_result=-1.0,
            depth_value_delta=1.0,
            depth_move_changed=True,
        ),
    ])

    sampled = replay.sample(2_000, seed=7)
    high_priority = sum(int(record.state[0]) == 2 for record in sampled)

    assert high_priority > 1_700


def test_uniform_replay_flag_preserves_uniform_sampling() -> None:
    replay = AlphaBetaReplayBuffer(
        capacity=2,
        priority_config=AlphaBetaReplayPriorityConfig(enabled=False),
    )
    replay.add([
        _priority_record(1),
        _priority_record(2, raw_value=-1.0, search_value=1.0),
    ])

    sampled = replay.sample(2_000, seed=11)
    first_count = sum(int(record.state[0]) == 1 for record in sampled)

    assert 900 < first_count < 1_100


def test_old_replay_state_reconstructs_priorities() -> None:
    record = _priority_record(3, raw_value=-0.5, search_value=0.5)
    replay = AlphaBetaReplayBuffer(capacity=4)
    replay.load_state_dict({
        "capacity": 4,
        "next": 1,
        "records": [record],
    })

    assert len(replay._priorities) == 1
    assert replay._priorities[0] > 0.05


def test_full_resume_restores_models_optimizer_replay_and_rng() -> None:
    random.seed(41)
    torch.manual_seed(41)
    net = HiveFNN(FNNConfig.small())
    ema = HiveFNN(FNNConfig.small())
    ema.load_state_dict(net.state_dict())
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in net.parameters()).backward()
    optimizer.step()

    replay = AlphaBetaReplayBuffer(capacity=7)
    replay.add([
        AlphaBetaRecord(
            state=torch.tensor([1, 2, 3], dtype=torch.uint8),
            legal_moves=torch.tensor([[4, 5]], dtype=torch.uint8),
            raw_value=0.1,
            search_value=0.2,
            selected_index=0,
            pv_moves=torch.tensor([[4, 5]], dtype=torch.uint8),
            root_scores=torch.tensor([0.2]),
            root_bounds=torch.tensor(
                [AlphaBetaBound.EXACT], dtype=torch.uint8,
            ),
            completed_depth=2,
            nodes=17,
            final_result=1.0,
        ),
    ])
    expected_model = {
        name: value.detach().clone()
        for name, value in net.state_dict().items()
    }
    state_path = Path(__file__).parent.parent / ".alpha_beta_resume_test.pt"
    try:
        _save_resume_state(
            state_path,
            net=net,
            ema=ema,
            optimizer=optimizer,
            replay=replay,
            iteration=12,
            generation_config=AlphaBetaGenerationConfig(games=2),
            loss_config=AlphaBetaLossConfig(),
        )
        expected_python_random = random.random()
        expected_torch_random = torch.rand(1)

        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
        replay = AlphaBetaReplayBuffer(capacity=1)
        random.seed(99)
        torch.manual_seed(99)

        start_iteration, saved_generation_config = _load_resume_state(
            state_path,
            net=net,
            ema=ema,
            optimizer=optimizer,
            replay=replay,
        )

        assert start_iteration == 13
        assert saved_generation_config["games"] == 2
        assert len(replay) == 1
        assert optimizer.state
        for name, value in net.state_dict().items():
            assert torch.equal(value, expected_model[name])
        assert random.random() == expected_python_random
        assert torch.equal(torch.rand(1), expected_torch_random)
    finally:
        state_path.unlink(missing_ok=True)
