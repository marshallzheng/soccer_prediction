import pytest

from corner_predictor.data_sources.models import EventTypeId, MatchEvent, MatchState, StatisticEntry, StatisticTypeId
from corner_predictor.features.models import FeatureSnapshot
from corner_predictor.model.schemas import ProbabilityResult
from corner_predictor.persistence.repository import MatchRepository


def _state() -> MatchState:
    return MatchState(
        fixture_id="m1",
        minute=42,
        home_participant_id=1,
        away_participant_id=2,
        home_team="Home FC",
        away_team="Away United",
        statistics=[
            StatisticEntry(type_id=StatisticTypeId.CORNERS, participant_id=1, location="home", value=3),
            StatisticEntry(type_id=StatisticTypeId.CORNERS, participant_id=2, location="away", value=2),
        ],
    )


def _features() -> FeatureSnapshot:
    return FeatureSnapshot(
        minutes_elapsed=42.0,
        minutes_remaining=48.0,
        corners_so_far_total=5,
        corner_rate_so_far=5 / 42,
        attacks_last_window_home=4,
        attacks_last_window_away=3,
        shots_last_window_home=2,
        shots_last_window_away=1,
        possession_home_pct=55.0,
        score_diff=0,
        urgency_multiplier=1.0,
    )


def _result() -> ProbabilityResult:
    return ProbabilityResult(
        threshold=9.5,
        prob_over=0.4,
        prob_under=0.6,
        expected_total_corners=8.5,
        pmf=[(5, 0.1), (6, 0.2)],
        lambda_remaining=3.5,
        observed_corners=5,
        minutes_remaining=48.0,
    )


def test_create_match_and_list_matches() -> None:
    repo = MatchRepository()
    repo.create_match("m1", "Home FC", "Away United", source="mock")
    matches = repo.list_matches()
    assert len(matches) == 1
    assert matches[0].fixture_id == "m1"
    assert matches[0].status == "live"


def test_save_tick_and_get_history() -> None:
    repo = MatchRepository()
    repo.create_match("m1", "Home FC", "Away United", source="mock")
    repo.save_tick("m1", _state(), _features(), _result())

    history = repo.get_tick_history("m1")
    assert len(history) == 1
    tick = history[0]
    assert tick.minute == 42
    assert tick.corners_home == 3
    assert tick.prob_over == pytest.approx(0.4)
    assert tick.pmf == [[5, 0.1], [6, 0.2]]
    assert tick.statistics == [s.model_dump() for s in _state().statistics]


def test_save_events() -> None:
    repo = MatchRepository()
    repo.create_match("m1", "Home FC", "Away United", source="mock")
    events = [
        MatchEvent(id=1, fixture_id="m1", type_id=EventTypeId.GOAL, participant_id=1, location="home", minute=10),
        MatchEvent(
            id=2, fixture_id="m1", type_id=EventTypeId.SUBSTITUTION, participant_id=2, location="away", minute=15
        ),
    ]
    repo.save_events(events)
    repo.save_events([])  # no-op, should not raise


def test_finalize_match_sets_status_and_final_score() -> None:
    repo = MatchRepository()
    repo.create_match("m1", "Home FC", "Away United", source="mock")
    repo.finalize_match("m1", final_corners_home=6, final_corners_away=4)

    match = repo.get_match("m1")
    assert match is not None
    assert match.status == "finished"
    assert match.final_corners_home == 6
    assert match.final_corners_away == 4
    assert match.ended_at is not None


def test_get_match_returns_none_when_missing() -> None:
    repo = MatchRepository()
    assert repo.get_match("does-not-exist") is None
