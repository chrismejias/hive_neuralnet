from pathlib import Path

import numpy as np
import pytest

from hive_fnn.fnn_alphabeta_training import AlphaBetaReplayBuffer
from hive_fnn.fnn_native_alphabeta import AlphaBetaSearchConfig
from hive_fnn.fnn_alphabeta_tuning import (
    AlphaBetaSPSATuner,
    PairedArenaResult,
    decode_search_config,
    default_normalized_parameters,
    load_search_config,
)

def test_default_parameter_vector_decodes_to_current_search_defaults() -> None:
    config = decode_search_config(default_normalized_parameters())

    assert config.aspiration_window == 0.0
    assert config.lmr_min_depth == 4
    assert config.lmr_min_move == 4
    assert config.lmr_reduction == 1
    assert config.quiescence_plies == 1
    assert config.quiescence_budget_fraction == pytest.approx(0.2)
    assert config.force_win_probes
    assert config.tactical_immobilization
    assert config.tactical_opponent_surround
    assert config.tactical_own_relief
    assert config.tactical_queen_threat
    assert config.branching_allocation == 0.0
    assert config.early_stop_score == 9.0
    assert config.early_stop_min_depth == 1

def test_named_profiles_are_independent_and_value_only_alias_is_legacy() -> None:
    baseline = AlphaBetaSearchConfig.from_profile("baseline")
    value_only = AlphaBetaSearchConfig.from_profile("value-only")
    threat = AlphaBetaSearchConfig.from_profile("threat")
    proof = AlphaBetaSearchConfig.from_profile("proof")
    full = AlphaBetaSearchConfig.from_profile("full")

    assert baseline == AlphaBetaSearchConfig()
    assert value_only == baseline
    assert not baseline.recursive_threat_qsearch
    assert threat.recursive_threat_qsearch and threat.quiescence_plies == 4
    assert threat.forced_extensions and not threat.proof_search
    assert proof.proof_search and not proof.forced_extensions
    assert full.proof_search and full.persistent_tt
    assert full.countermove_ordering and full.continuation_history

def test_legacy_policy_metadata_loads_without_reenabling_policy() -> None:
    config = AlphaBetaSearchConfig.from_metadata({
        "policy_ordering_weight": 3.0,
        "tactical_ordering_weight": 2.0,
        "internal_policy_ordering": True,
    })

    assert config.internal_heuristic_ordering
    assert not hasattr(config, "policy_ordering_weight")

def test_default_alpha_beta_replay_capacity_is_75000() -> None:
    assert AlphaBetaReplayBuffer().capacity == 75_000

def test_spsa_update_is_bounded_and_resumable() -> None:
    output = Path(__file__).parent / ".alpha_beta_spsa_test"
    output.mkdir(exist_ok=True)
    try:
        tuner = AlphaBetaSPSATuner(output, seed=17)
        plus, minus, delta, a_k, c_k = tuner.ask()

        assert np.all((0.0 <= plus) & (plus <= 1.0))
        assert np.all((0.0 <= minus) & (minus <= 1.0))
        record = tuner.tell(
            plus,
            minus,
            delta,
            a_k,
            c_k,
            PairedArenaResult(
                plus_score=0.625,
                games=8,
                plus_wins=4,
                minus_wins=2,
                draws=2,
                plus_nodes=800,
                minus_nodes=720,
                plus_moves=20,
                minus_moves=20,
            ),
        )

        assert tuner.iteration == 1
        assert np.all((0.0 <= tuner.theta) & (tuner.theta <= 1.0))
        assert record["objective_difference"] < 0.25
        restored = AlphaBetaSPSATuner(output, seed=999)
        restored.load()
        assert restored.iteration == tuner.iteration
        assert np.array_equal(restored.theta, tuner.theta)
        next_original = tuner.ask()
        next_restored = restored.ask()
        for original, loaded in zip(next_original[:3], next_restored[:3]):
            assert np.array_equal(original, loaded)
        loaded_config = load_search_config(restored.state_path)
        assert loaded_config == decode_search_config(restored.theta)
    finally:
        for path in output.glob("*"):
            path.unlink()
        output.rmdir()
