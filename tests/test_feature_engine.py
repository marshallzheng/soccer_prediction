import math

from corner_predictor.data_sources.models import EventType, MatchEvent, MatchState
from corner_predictor.features.engine import FeatureEngine


def _state(**overrides) -> MatchState:
    defaults = dict(
        match_id="m1",
        minute=50.0,
        period=2,
        home_team="Home FC",
        away_team="Away United",
        corners_home=4,
        corners_away=2,
        score_home=1,
        score_away=1,
        possession_home_pct=55.0,
    )
    defaults.update(overrides)
    return MatchState(**defaults)


def _event(minute: float, event_type: EventType, team: str) -> MatchEvent:
    return MatchEvent(match_id="m1", minute=minute, event_type=event_type, team=team)


def test_corner_rate_and_totals() -> None:
    engine = FeatureEngine(rolling_window_minutes=10.0)
    state = _state(minute=40.0, corners_home=3, corners_away=1)
    snapshot = engine.compute(state, event_history=[])
    assert snapshot.corners_so_far_total == 4
    assert math.isclose(snapshot.corner_rate_so_far, 4.0 / 40.0)
    assert math.isclose(snapshot.minutes_remaining, 50.0)


def test_rolling_window_only_counts_recent_events() -> None:
    engine = FeatureEngine(rolling_window_minutes=10.0)
    state = _state(minute=50.0)
    history = [
        _event(35.0, EventType.DANGEROUS_ATTACK, "home"),  # outside window (50-10=40 cutoff)
        _event(42.0, EventType.DANGEROUS_ATTACK, "home"),  # inside window
        _event(45.0, EventType.DANGEROUS_ATTACK, "away"),  # inside window
        _event(48.0, EventType.SHOT, "home"),  # inside window
        _event(20.0, EventType.SHOT, "away"),  # outside window
    ]
    snapshot = engine.compute(state, history)
    assert snapshot.attacks_last_window_home == 1
    assert snapshot.attacks_last_window_away == 1
    assert snapshot.shots_last_window_home == 1
    assert snapshot.shots_last_window_away == 0


def test_urgency_multiplier_scales_with_score_diff_and_late_game() -> None:
    engine = FeatureEngine()
    even_early = engine.compute(_state(minute=30.0, score_home=1, score_away=1), [])
    lopsided_early = engine.compute(_state(minute=30.0, score_home=3, score_away=0), [])
    even_late = engine.compute(_state(minute=80.0, score_home=1, score_away=1), [])

    assert lopsided_early.urgency_multiplier > even_early.urgency_multiplier
    assert even_late.urgency_multiplier > even_early.urgency_multiplier


def test_zero_minutes_elapsed_gives_zero_rate() -> None:
    engine = FeatureEngine()
    snapshot = engine.compute(_state(minute=0.0, corners_home=0, corners_away=0), [])
    assert snapshot.corner_rate_so_far == 0.0
