import datetime as dt

from sqlalchemy import select

from corner_predictor.data_sources.models import MatchEvent, MatchState
from corner_predictor.features.models import FeatureSnapshot
from corner_predictor.model.schemas import ProbabilityResult
from corner_predictor.persistence.db import get_session
from corner_predictor.persistence.schema import Match, MatchEventRecord, MatchTick


class MatchRepository:
    """Thin CRUD wrapper so callers never touch SQLAlchemy sessions directly.

    Keeps persistence swappable (SQLite -> DuckDB/Postgres later) without
    touching business logic in MatchRunner or the API routes.
    """

    def create_match(self, match_id: str, home_team: str, away_team: str, source: str) -> None:
        with get_session() as session:
            session.add(Match(id=match_id, home_team=home_team, away_team=away_team, source=source))
            session.commit()

    def save_tick(
        self,
        match_id: str,
        state: MatchState,
        features: FeatureSnapshot,
        result: ProbabilityResult,
    ) -> None:
        with get_session() as session:
            session.add(
                MatchTick(
                    match_id=match_id,
                    minute=state.minute,
                    score_home=state.score_home,
                    score_away=state.score_away,
                    corners_home=state.corners_home,
                    corners_away=state.corners_away,
                    shots_home=state.shots_home,
                    shots_away=state.shots_away,
                    shots_on_target_home=state.shots_on_target_home,
                    shots_on_target_away=state.shots_on_target_away,
                    dangerous_attacks_home=state.dangerous_attacks_home,
                    dangerous_attacks_away=state.dangerous_attacks_away,
                    possession_home_pct=state.possession_home_pct,
                    corner_rate_so_far=features.corner_rate_so_far,
                    urgency_multiplier=features.urgency_multiplier,
                    threshold=result.threshold,
                    prob_over=result.prob_over,
                    lambda_remaining=result.lambda_remaining,
                    pmf=[list(pair) for pair in result.pmf],
                )
            )
            session.commit()

    def save_event(self, event: MatchEvent) -> None:
        with get_session() as session:
            session.add(
                MatchEventRecord(
                    match_id=event.match_id,
                    minute=event.minute,
                    event_type=event.event_type.value,
                    team=event.team,
                    payload=event.payload,
                )
            )
            session.commit()

    def save_events(self, events: list[MatchEvent]) -> None:
        if not events:
            return
        with get_session() as session:
            session.add_all(
                MatchEventRecord(
                    match_id=e.match_id,
                    minute=e.minute,
                    event_type=e.event_type.value,
                    team=e.team,
                    payload=e.payload,
                )
                for e in events
            )
            session.commit()

    def finalize_match(self, match_id: str, final_corners_home: int, final_corners_away: int) -> None:
        with get_session() as session:
            match = session.get(Match, match_id)
            if match is None:
                return
            match.status = "finished"
            match.ended_at = dt.datetime.now(dt.UTC)
            match.final_corners_home = final_corners_home
            match.final_corners_away = final_corners_away
            session.commit()

    def list_matches(self) -> list[Match]:
        with get_session() as session:
            return list(session.scalars(select(Match).order_by(Match.started_at.desc())))

    def get_match(self, match_id: str) -> Match | None:
        with get_session() as session:
            return session.get(Match, match_id)

    def get_tick_history(self, match_id: str) -> list[MatchTick]:
        with get_session() as session:
            stmt = select(MatchTick).where(MatchTick.match_id == match_id).order_by(MatchTick.minute)
            return list(session.scalars(stmt))
