import pytest

from hive_fnn.fnn_alphabeta_player import AlphaBetaConfig, normalize_alpha_beta_score


def test_normal_alpha_beta_scores_preserve_outcome_scale() -> None:
    assert normalize_alpha_beta_score(0.42) == pytest.approx((0.42, None))
    assert normalize_alpha_beta_score(-0.75) == pytest.approx((-0.75, None))
    assert normalize_alpha_beta_score(1.5) == (1.0, None)


def test_mate_band_is_separate_and_decodes_distance() -> None:
    assert normalize_alpha_beta_score(10.0) == (1.0, 0)
    assert normalize_alpha_beta_score(9.97) == (1.0, 3)
    assert normalize_alpha_beta_score(-9.91) == (-1.0, 9)


def test_cpu_search_profiles_are_toggleable() -> None:
    plain = AlphaBetaConfig.from_profile("plain")
    assert plain.quiescence_plies == 0
    assert not plain.forced_extensions

    baseline = AlphaBetaConfig.from_profile("baseline")
    assert baseline.quiescence_plies == 0
    assert not baseline.recursive_threat_qsearch

    quiescence = AlphaBetaConfig.from_profile("quiescence")
    assert quiescence.quiescence_plies == 1

    threat = AlphaBetaConfig.from_profile("threat")
    assert threat.quiescence_plies == 4
    assert threat.recursive_threat_qsearch
    assert threat.forced_extensions
