from pydantic import BaseModel


class ProbabilityResult(BaseModel):
    threshold: float
    prob_over: float
    prob_under: float
    expected_total_corners: float
    pmf: list[tuple[int, float]]
    """(total_corners, probability) pairs; the last bucket folds in the tail beyond pmf_max_remaining_corners."""
    lambda_remaining: float
    observed_corners: int
    minutes_remaining: float
