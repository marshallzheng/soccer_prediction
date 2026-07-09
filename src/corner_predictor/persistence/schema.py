import datetime as dt

from sqlalchemy import JSON, ForeignKey, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Match(Base):
    __tablename__ = "matches"

    fixture_id: Mapped[str] = mapped_column(String, primary_key=True)
    home_team: Mapped[str]
    away_team: Mapped[str]
    source: Mapped[str]
    """'mock' or a future vendor name."""
    started_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    ended_at: Mapped[dt.datetime | None] = mapped_column(default=None)
    final_corners_home: Mapped[int | None] = mapped_column(default=None)
    final_corners_away: Mapped[int | None] = mapped_column(default=None)
    status: Mapped[str] = mapped_column(default="live")
    """'live' | 'finished'."""


class MatchTick(Base):
    """One row per tick per match; the training row for a future ML model.

    Stores raw match-state snapshot fields plus computed features and the
    live model's output, so a later Phase 2 job can join "features at time T"
    against "actual final corner count" (Match.final_corners_*) as a label.
    """

    __tablename__ = "match_ticks"
    __table_args__ = (Index("ix_match_ticks_fixture_id_minute", "fixture_id", "minute"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fixture_id: Mapped[str] = mapped_column(ForeignKey("matches.fixture_id"))
    minute: Mapped[int]
    timestamp: Mapped[dt.datetime] = mapped_column(default=_utcnow)

    # Raw statistics payload, mirroring Sportmonks' `statistics` include verbatim
    # (list of {type_id, participant_id, location, value}), for full-fidelity ML training later.
    statistics: Mapped[list] = mapped_column(JSON)

    # Flattened convenience columns (derived from `statistics` at write time) for querying/indexing.
    score_home: Mapped[int]
    score_away: Mapped[int]
    corners_home: Mapped[int]
    corners_away: Mapped[int]
    shots_home: Mapped[int]
    shots_away: Mapped[int]
    shots_on_target_home: Mapped[int]
    shots_on_target_away: Mapped[int]
    dangerous_attacks_home: Mapped[int]
    dangerous_attacks_away: Mapped[int]
    possession_home_pct: Mapped[float]

    # Computed features
    corner_rate_so_far: Mapped[float]
    urgency_multiplier: Mapped[float]

    # Model output
    threshold: Mapped[float]
    prob_over: Mapped[float]
    lambda_remaining: Mapped[float]
    pmf: Mapped[list] = mapped_column(JSON)


class MatchEventRecord(Base):
    """Raw event log (goals/cards/substitutions), independent of tick snapshots."""

    __tablename__ = "match_events"
    __table_args__ = (Index("ix_match_events_fixture_id_minute", "fixture_id", "minute"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    fixture_id: Mapped[str] = mapped_column(ForeignKey("matches.fixture_id"))
    minute: Mapped[int]
    type_id: Mapped[int]
    location: Mapped[str | None]
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    """Extra fields not worth their own column, e.g. extra_minute, result."""
