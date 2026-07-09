import math

from corner_predictor.data_sources.models import MatchState, StatisticEntry, StatisticTypeId
from corner_predictor.features.engine import FeatureEngine


def _state(
    minute: int,
    corners_home: float = 0,
    corners_away: float = 0,
    dangerous_attacks_home: float = 0,
    dangerous_attacks_away: float = 0,
    shots_home: float = 0,
    shots_away: float = 0,
    possession_home_pct: float = 50.0,
    score_home: int = 0,
    score_away: int = 0,
) -> MatchState:
    statistics = [
        StatisticEntry(type_id=StatisticTypeId.CORNERS, participant_id=1, location="home", value=corners_home),
        StatisticEntry(type_id=StatisticTypeId.CORNERS, participant_id=2, location="away", value=corners_away),
        StatisticEntry(
            type_id=StatisticTypeId.DANGEROUS_ATTACKS, participant_id=1, location="home", value=dangerous_attacks_home
        ),
        StatisticEntry(
            type_id=StatisticTypeId.DANGEROUS_ATTACKS, participant_id=2, location="away", value=dangerous_attacks_away
        ),
        StatisticEntry(type_id=StatisticTypeId.SHOTS_TOTAL, participant_id=1, location="home", value=shots_home),
        StatisticEntry(type_id=StatisticTypeId.SHOTS_TOTAL, participant_id=2, location="away", value=shots_away),
        StatisticEntry(
            type_id=StatisticTypeId.BALL_POSSESSION, participant_id=1, location="home", value=possession_home_pct
        ),
    ]
    return MatchState(
        fixture_id="m1",
        minute=minute,
        home_participant_id=1,
        away_participant_id=2,
        home_team="Home FC",
        away_team="Away United",
        score_home=score_home,
        score_away=score_away,
        statistics=statistics,
    )


def test_corner_rate_and_totals() -> None:
    engine = FeatureEngine(rolling_window_minutes=10.0)
    state = _state(minute=40, corners_home=3, corners_away=1)
    snapshot = engine.compute(state, state_history=[])
    assert snapshot.corners_so_far_total == 4
    assert math.isclose(snapshot.corner_rate_so_far, 4.0 / 40.0)
    assert math.isclose(snapshot.minutes_remaining, 50.0)


def test_rolling_window_uses_delta_against_nearest_prior_snapshot() -> None:
    engine = FeatureEngine(rolling_window_minutes=10.0)
    state_history = [
        _state(minute=30, dangerous_attacks_home=1, shots_home=0),
        _state(minute=40, dangerous_attacks_home=2, shots_home=1),  # nearest snapshot at/before cutoff (50-10=40)
        _state(minute=45, dangerous_attacks_home=4, shots_home=2),  # after cutoff, must be ignored as baseline
    ]
    current = _state(minute=50, dangerous_attacks_home=6, dangerous_attacks_away=1, shots_home=3, shots_away=0)

    snapshot = engine.compute(current, state_history)

    assert snapshot.attacks_last_window_home == 4  # 6 - 2 (the minute=40 snapshot)
    assert snapshot.attacks_last_window_away == 1  # 1 - 0 (no away activity in any snapshot)
    assert snapshot.shots_last_window_home == 2  # 3 - 1


def test_no_prior_snapshot_counts_full_cumulative_value() -> None:
    engine = FeatureEngine(rolling_window_minutes=10.0)
    current = _state(minute=5, dangerous_attacks_home=3, shots_home=2)
    snapshot = engine.compute(current, state_history=[])
    assert snapshot.attacks_last_window_home == 3
    assert snapshot.shots_last_window_home == 2


def test_urgency_multiplier_scales_with_score_diff_and_late_game() -> None:
    engine = FeatureEngine()
    even_early = engine.compute(_state(minute=30, score_home=1, score_away=1), [])
    lopsided_early = engine.compute(_state(minute=30, score_home=3, score_away=0), [])
    even_late = engine.compute(_state(minute=80, score_home=1, score_away=1), [])

    assert lopsided_early.urgency_multiplier > even_early.urgency_multiplier
    assert even_late.urgency_multiplier > even_early.urgency_multiplier


def test_zero_minutes_elapsed_gives_zero_rate() -> None:
    engine = FeatureEngine()
    snapshot = engine.compute(_state(minute=0, corners_home=0, corners_away=0), [])
    assert snapshot.corner_rate_so_far == 0.0
