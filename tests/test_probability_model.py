import math

import pytest
from scipy.stats import poisson

from corner_predictor.model.probability import (
    NegBinomCornerModel,
    PoissonCornerModel,
    get_model,
)


def test_poisson_prob_over_matches_hand_computed_value() -> None:
    model = PoissonCornerModel()
    # observed=5, rate=0.1/min, minutes_remaining=20 -> lambda_remaining=2.0
    result = model.predict(observed_corners=5, minutes_remaining=20.0, rate_per_min=0.1, threshold=6.5)
    assert math.isclose(result.lambda_remaining, 2.0)
    expected = poisson.sf(6.5 - 5, mu=2.0)  # P(remaining > 1.5) = P(remaining >= 2)
    assert math.isclose(result.prob_over, expected, rel_tol=1e-9)
    assert math.isclose(result.prob_over + result.prob_under, 1.0)


def test_pmf_sums_to_approximately_one() -> None:
    model = PoissonCornerModel()
    result = model.predict(observed_corners=3, minutes_remaining=45.0, rate_per_min=0.12, threshold=9.5)
    total_prob = sum(p for _, p in result.pmf)
    assert math.isclose(total_prob, 1.0, abs_tol=1e-6)


def test_pmf_totals_are_offset_by_observed_corners() -> None:
    model = PoissonCornerModel()
    result = model.predict(observed_corners=4, minutes_remaining=30.0, rate_per_min=0.1, threshold=9.5)
    totals = [total for total, _ in result.pmf]
    assert totals[0] == 4
    assert totals == list(range(4, 4 + len(totals)))


def test_prob_over_is_one_when_observed_already_exceeds_threshold() -> None:
    model = PoissonCornerModel()
    result = model.predict(observed_corners=11, minutes_remaining=20.0, rate_per_min=0.1, threshold=9.5)
    assert result.prob_over == pytest.approx(1.0)
    assert result.prob_under == pytest.approx(0.0)


def test_zero_minutes_remaining_is_deterministic() -> None:
    model = PoissonCornerModel()
    result = model.predict(observed_corners=8, minutes_remaining=0.0, rate_per_min=0.15, threshold=9.5)
    assert result.lambda_remaining == 0.0
    assert result.expected_total_corners == 8.0
    # total is deterministically 8, so P(total > 9.5) must be 0
    assert result.prob_over == pytest.approx(0.0)

    result_below_threshold = model.predict(observed_corners=10, minutes_remaining=0.0, rate_per_min=0.15, threshold=9.5)
    assert result_below_threshold.prob_over == pytest.approx(1.0)


def test_negative_binomial_model_produces_valid_probabilities() -> None:
    model = NegBinomCornerModel(overdispersion_k=8.0)
    result = model.predict(observed_corners=4, minutes_remaining=40.0, rate_per_min=0.1, threshold=9.5)
    assert 0.0 <= result.prob_over <= 1.0
    total_prob = sum(p for _, p in result.pmf)
    assert math.isclose(total_prob, 1.0, abs_tol=1e-6)


def test_get_model_factory_returns_expected_types() -> None:
    assert isinstance(get_model("poisson"), PoissonCornerModel)
    assert isinstance(get_model("negative_binomial"), NegBinomCornerModel)
    with pytest.raises(ValueError):
        get_model("not_a_real_model")
