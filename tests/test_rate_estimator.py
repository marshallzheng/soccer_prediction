import math

from corner_predictor.features.models import FeatureSnapshot
from corner_predictor.model.rate_estimator import RateEstimator


def _snapshot(**overrides) -> FeatureSnapshot:
    defaults = dict(
        minutes_elapsed=45.0,
        minutes_remaining=45.0,
        corners_so_far_total=5,
        corner_rate_so_far=5.0 / 45.0,
        attacks_last_window_home=50,
        attacks_last_window_away=50,
        possession_home_pct=50.0,
        score_diff=0,
        shots_last_window_home=6,
        shots_last_window_away=6,
        urgency_multiplier=1.0,
    )
    defaults.update(overrides)
    return FeatureSnapshot(**defaults)


def test_early_match_estimate_leans_toward_prior() -> None:
    estimator = RateEstimator()
    prior = 10.0 / 90.0
    # 1 minute elapsed with an unrepresentative early observed rate should barely move the estimate.
    features = _snapshot(minutes_elapsed=1.0, minutes_remaining=89.0, corner_rate_so_far=1.0)
    rate = estimator.estimate_remaining_rate(features, prior_rate_per_min=prior)
    assert abs(rate - prior) < abs(1.0 - prior)


def test_late_match_estimate_leans_toward_observed_pace() -> None:
    estimator = RateEstimator()
    prior = 10.0 / 90.0
    observed_rate = 0.2
    features = _snapshot(minutes_elapsed=80.0, minutes_remaining=10.0, corner_rate_so_far=observed_rate)
    rate = estimator.estimate_remaining_rate(features, prior_rate_per_min=prior)
    assert abs(rate - observed_rate) < abs(rate - prior)


def test_rate_is_clipped_to_configured_bounds() -> None:
    estimator = RateEstimator(rate_multiplier_min=0.3, rate_multiplier_max=3.0)
    prior = 10.0 / 90.0
    features = _snapshot(
        minutes_elapsed=90.0,
        minutes_remaining=0.0,
        corner_rate_so_far=5.0,
        urgency_multiplier=2.0,
    )
    rate = estimator.estimate_remaining_rate(features, prior_rate_per_min=prior)
    assert math.isclose(rate, prior * 3.0)
