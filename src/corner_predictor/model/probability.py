from abc import ABC, abstractmethod

from scipy.stats import nbinom, poisson
from scipy.stats._distn_infrastructure import rv_frozen

from corner_predictor.config import settings
from corner_predictor.model.schemas import ProbabilityResult


class CornerProbabilityModel(ABC):
    """Converts an estimated remaining-corner rate into an over/under probability + PMF.

    Observed corners so far are deterministic; only the *remaining* corners
    are modeled as a random variable. The final-total distribution is just
    that remaining-corner distribution shifted by the observed count.
    """

    @abstractmethod
    def _remaining_distribution(self, lambda_remaining: float) -> rv_frozen: ...

    def predict(
        self,
        observed_corners: int,
        minutes_remaining: float,
        rate_per_min: float,
        threshold: float,
    ) -> ProbabilityResult:
        minutes_remaining = max(minutes_remaining, 0.0)
        lambda_remaining = max(rate_per_min, 0.0) * minutes_remaining
        dist = self._remaining_distribution(lambda_remaining)

        prob_over = float(dist.sf(threshold - observed_corners))
        prob_over = min(max(prob_over, 0.0), 1.0)
        prob_under = 1.0 - prob_over

        k_max = settings.pmf_max_remaining_corners
        pmf: list[tuple[int, float]] = [
            (observed_corners + k, float(dist.pmf(k))) for k in range(k_max)
        ]
        tail_prob = float(dist.sf(k_max - 1))
        pmf.append((observed_corners + k_max, tail_prob))

        return ProbabilityResult(
            threshold=threshold,
            prob_over=prob_over,
            prob_under=prob_under,
            expected_total_corners=observed_corners + lambda_remaining,
            pmf=pmf,
            lambda_remaining=lambda_remaining,
            observed_corners=observed_corners,
            minutes_remaining=minutes_remaining,
        )


class PoissonCornerModel(CornerProbabilityModel):
    """MVP default: remaining corners ~ Poisson(lambda_remaining)."""

    def _remaining_distribution(self, lambda_remaining: float) -> rv_frozen:
        return poisson(mu=lambda_remaining)


class NegBinomCornerModel(CornerProbabilityModel):
    """Allows overdispersion (variance > mean), a documented real phenomenon in corner counts.

    Dispersion is a placeholder config constant pending calibration against
    real historical data (Phase 2).
    """

    def __init__(self, overdispersion_k: float | None = None) -> None:
        self.overdispersion_k = overdispersion_k if overdispersion_k is not None else settings.overdispersion_k

    def _remaining_distribution(self, lambda_remaining: float) -> rv_frozen:
        k = self.overdispersion_k
        if lambda_remaining <= 0:
            p = 1.0
        else:
            p = k / (k + lambda_remaining)
        return nbinom(n=k, p=p)


_MODEL_REGISTRY: dict[str, type[CornerProbabilityModel]] = {
    "poisson": PoissonCornerModel,
    "negative_binomial": NegBinomCornerModel,
}


def get_model(name: str | None = None) -> CornerProbabilityModel:
    """Factory so swapping Poisson/NegBinom/future-ML models is a one-line config change."""
    key = name or settings.probability_model
    try:
        model_cls = _MODEL_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(f"Unknown probability model {key!r}; available: {list(_MODEL_REGISTRY)}") from exc
    return model_cls()
