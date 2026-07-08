from pydantic import BaseModel


class FeatureSnapshot(BaseModel):
    """Rolling-window features computed from a match's state + recent event history."""

    minutes_elapsed: float
    minutes_remaining: float
    corners_so_far_total: int
    corner_rate_so_far: float
    """Corners per minute observed so far this match (empirical pace)."""
    attacks_last_window_home: int
    attacks_last_window_away: int
    shots_last_window_home: int
    shots_last_window_away: int
    possession_home_pct: float
    score_diff: int
    """Home minus away."""
    urgency_multiplier: float
    """Heuristic multiplier (score-differential + late-game pressure), applied on top of the blended rate."""
