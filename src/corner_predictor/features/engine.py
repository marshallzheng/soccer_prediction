from corner_predictor.config import settings
from corner_predictor.data_sources.models import EventType, MatchEvent, MatchState
from corner_predictor.features.models import FeatureSnapshot


class FeatureEngine:
    """Pure, I/O-free computation of rolling-window features from match state + event history."""

    def __init__(self, rolling_window_minutes: float | None = None) -> None:
        self.rolling_window_minutes = (
            rolling_window_minutes if rolling_window_minutes is not None else settings.rolling_window_minutes
        )

    def compute(self, state: MatchState, event_history: list[MatchEvent]) -> FeatureSnapshot:
        minutes_elapsed = state.minute
        minutes_remaining = max(90.0 - minutes_elapsed, 0.0)
        corners_so_far = state.corners_total
        corner_rate_so_far = corners_so_far / minutes_elapsed if minutes_elapsed > 0 else 0.0

        window_start = minutes_elapsed - self.rolling_window_minutes
        recent_events = [e for e in event_history if e.minute > window_start]

        attacks_home = self._count(recent_events, EventType.DANGEROUS_ATTACK, "home")
        attacks_away = self._count(recent_events, EventType.DANGEROUS_ATTACK, "away")
        shots_home = self._count(recent_events, EventType.SHOT, "home")
        shots_away = self._count(recent_events, EventType.SHOT, "away")

        urgency_multiplier = 1.0 + settings.urgency_score_diff_weight * abs(state.score_diff)
        if minutes_remaining <= settings.late_game_threshold_minutes:
            urgency_multiplier *= settings.late_game_multiplier

        return FeatureSnapshot(
            minutes_elapsed=minutes_elapsed,
            minutes_remaining=minutes_remaining,
            corners_so_far_total=corners_so_far,
            corner_rate_so_far=corner_rate_so_far,
            attacks_last_window_home=attacks_home,
            attacks_last_window_away=attacks_away,
            shots_last_window_home=shots_home,
            shots_last_window_away=shots_away,
            possession_home_pct=state.possession_home_pct,
            score_diff=state.score_diff,
            urgency_multiplier=urgency_multiplier,
        )

    @staticmethod
    def _count(events: list[MatchEvent], event_type: EventType, team: str) -> int:
        return sum(1 for e in events if e.event_type == event_type and e.team == team)
