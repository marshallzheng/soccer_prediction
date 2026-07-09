from corner_predictor.config import settings
from corner_predictor.data_sources.models import MatchState
from corner_predictor.features.models import FeatureSnapshot


class FeatureEngine:
    """Pure, I/O-free computation of rolling-window features from match state + state history.

    Sportmonks models corners/shots/dangerous-attacks as cumulative statistics
    (not discrete timestamped events), so "activity in the last N minutes" is
    computed as a delta against the nearest prior snapshot at least N minutes
    back, rather than by filtering a discrete event list.
    """

    def __init__(self, rolling_window_minutes: float | None = None) -> None:
        self.rolling_window_minutes = (
            rolling_window_minutes if rolling_window_minutes is not None else settings.rolling_window_minutes
        )

    def compute(self, state: MatchState, state_history: list[MatchState]) -> FeatureSnapshot:
        minutes_elapsed = state.minute
        minutes_remaining = max(90.0 - minutes_elapsed, 0.0)
        corners_so_far = state.corners_total
        corner_rate_so_far = corners_so_far / minutes_elapsed if minutes_elapsed > 0 else 0.0

        prior = self._nearest_prior_snapshot(state, state_history)
        prior_attacks_home = prior.dangerous_attacks_home if prior else 0
        prior_attacks_away = prior.dangerous_attacks_away if prior else 0
        prior_shots_home = prior.shots_home if prior else 0
        prior_shots_away = prior.shots_away if prior else 0

        urgency_multiplier = 1.0 + settings.urgency_score_diff_weight * abs(state.score_diff)
        if minutes_remaining <= settings.late_game_threshold_minutes:
            urgency_multiplier *= settings.late_game_multiplier

        return FeatureSnapshot(
            minutes_elapsed=minutes_elapsed,
            minutes_remaining=minutes_remaining,
            corners_so_far_total=corners_so_far,
            corner_rate_so_far=corner_rate_so_far,
            attacks_last_window_home=max(0, state.dangerous_attacks_home - prior_attacks_home),
            attacks_last_window_away=max(0, state.dangerous_attacks_away - prior_attacks_away),
            shots_last_window_home=max(0, state.shots_home - prior_shots_home),
            shots_last_window_away=max(0, state.shots_away - prior_shots_away),
            possession_home_pct=state.possession_home_pct,
            score_diff=state.score_diff,
            urgency_multiplier=urgency_multiplier,
        )

    def _nearest_prior_snapshot(self, state: MatchState, state_history: list[MatchState]) -> MatchState | None:
        """Most recent snapshot at or before `state.minute - rolling_window_minutes`."""
        window_cutoff = state.minute - self.rolling_window_minutes
        candidates = [s for s in state_history if s.minute <= window_cutoff]
        return max(candidates, key=lambda s: s.minute) if candidates else None
