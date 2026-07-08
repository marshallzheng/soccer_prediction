from corner_predictor.config import settings
from corner_predictor.features.models import FeatureSnapshot


class RateEstimator:
    """Estimates the corner rate (corners/minute) expected for the rest of the match.

    Blends a league-wide prior baseline rate with the pace observed so far this
    match (Bayesian shrinkage toward the prior when little of the match has
    elapsed), then adjusts for match-state urgency and recent attacking
    intensity relative to a baseline expectation.
    """

    def __init__(
        self,
        prior_weight_minutes: float | None = None,
        rate_multiplier_min: float | None = None,
        rate_multiplier_max: float | None = None,
    ) -> None:
        self.prior_weight_minutes = (
            prior_weight_minutes if prior_weight_minutes is not None else settings.prior_weight_minutes
        )
        self.rate_multiplier_min = (
            rate_multiplier_min if rate_multiplier_min is not None else settings.rate_multiplier_min
        )
        self.rate_multiplier_max = (
            rate_multiplier_max if rate_multiplier_max is not None else settings.rate_multiplier_max
        )

    def estimate_remaining_rate(self, features: FeatureSnapshot, prior_rate_per_min: float) -> float:
        k = self.prior_weight_minutes
        w = features.minutes_elapsed / (features.minutes_elapsed + k) if features.minutes_elapsed > 0 else 0.0
        blended_rate = w * features.corner_rate_so_far + (1.0 - w) * prior_rate_per_min

        expected_window_attacks = 2 * settings.baseline_attacks_per_min_per_team * settings.rolling_window_minutes
        observed_window_attacks = features.attacks_last_window_home + features.attacks_last_window_away
        intensity_factor = observed_window_attacks / expected_window_attacks if expected_window_attacks > 0 else 1.0
        intensity_factor = min(max(intensity_factor, settings.intensity_factor_min), settings.intensity_factor_max)

        adjusted_rate = blended_rate * features.urgency_multiplier * intensity_factor

        lower = prior_rate_per_min * self.rate_multiplier_min
        upper = prior_rate_per_min * self.rate_multiplier_max
        return min(max(adjusted_rate, lower), upper)
