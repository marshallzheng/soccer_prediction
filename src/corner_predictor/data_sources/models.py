from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

Location = Literal["home", "away"]


class EventTypeId(IntEnum):
    """Mirrors Sportmonks' event `type_id` taxonomy (subset relevant to this project).

    Only covers goal/card/substitution events -- Sportmonks models corners,
    shots, and dangerous attacks as cumulative `statistics` values instead of
    discrete timestamped events (see StatisticEntry / StatisticTypeId).
    """

    GOAL = 14
    OWNGOAL = 15
    PENALTY = 16
    MISSED_PENALTY = 17
    SUBSTITUTION = 18
    YELLOWCARD = 19
    REDCARD = 20


class StatisticTypeId(IntEnum):
    """Mirrors Sportmonks' statistic `type_id` taxonomy (subset relevant to this project)."""

    CORNERS = 34
    SHOTS_TOTAL = 42
    ATTACKS = 43
    DANGEROUS_ATTACKS = 44
    BALL_POSSESSION = 45
    SHOTS_ON_TARGET = 86


class FixtureStateId(IntEnum):
    """Mirrors Sportmonks' fixture `state_id` taxonomy (subset relevant to this project)."""

    NS = 1
    INPLAY_1ST_HALF = 2
    HT = 3
    BREAK = 4
    FT = 5
    INPLAY_ET = 6
    AET = 7
    FT_PEN = 8
    INPLAY_PENALTIES = 9
    POSTPONED = 10
    SUSPENDED = 11
    CANCELLED = 12
    INPLAY_2ND_HALF = 22


FINISHED_STATE_IDS = frozenset({FixtureStateId.FT, FixtureStateId.AET, FixtureStateId.FT_PEN})
LIVE_STATE_IDS = frozenset(
    {
        FixtureStateId.INPLAY_1ST_HALF,
        FixtureStateId.INPLAY_2ND_HALF,
        FixtureStateId.HT,
        FixtureStateId.BREAK,
        FixtureStateId.INPLAY_ET,
        FixtureStateId.INPLAY_PENALTIES,
    }
)


class MatchEvent(BaseModel):
    """Mirrors a Sportmonks fixture `events` include entry."""

    id: int
    fixture_id: str
    type_id: EventTypeId
    participant_id: int
    location: Location
    minute: int
    extra_minute: int | None = None
    result: str | None = None
    """Score at the time of the event, e.g. "1-0"."""


class StatisticEntry(BaseModel):
    """Mirrors a single Sportmonks fixture `statistics` include entry (`data.value` flattened here)."""

    type_id: StatisticTypeId
    participant_id: int
    location: Location
    value: float


class MatchState(BaseModel):
    """Snapshot mirroring the subset of a Sportmonks Fixture (+ periods/participants/statistics
    includes) this system needs.

    Simplifications vs. the real API: `minute` is flattened here (the real API
    derives it from the ticking entry in the `periods` include's `minutes`/
    `seconds` fields); `home_team`/`away_team` are flattened strings (the real
    API returns full team objects via the `participants` include). A real
    adapter is responsible for this flattening -- the internal model shape
    doesn't need to change when one is added.
    """

    fixture_id: str
    state_id: FixtureStateId = FixtureStateId.NS
    minute: int = 0
    home_participant_id: int
    away_participant_id: int
    home_team: str
    away_team: str
    score_home: int = 0
    score_away: int = 0
    statistics: list[StatisticEntry] = Field(default_factory=list)

    def _stat(self, type_id: StatisticTypeId, location: Location) -> float:
        return next(
            (s.value for s in self.statistics if s.type_id == type_id and s.location == location),
            0.0,
        )

    @computed_field
    @property
    def corners_home(self) -> int:
        return int(self._stat(StatisticTypeId.CORNERS, "home"))

    @computed_field
    @property
    def corners_away(self) -> int:
        return int(self._stat(StatisticTypeId.CORNERS, "away"))

    @computed_field
    @property
    def corners_total(self) -> int:
        return self.corners_home + self.corners_away

    @computed_field
    @property
    def shots_home(self) -> int:
        return int(self._stat(StatisticTypeId.SHOTS_TOTAL, "home"))

    @computed_field
    @property
    def shots_away(self) -> int:
        return int(self._stat(StatisticTypeId.SHOTS_TOTAL, "away"))

    @computed_field
    @property
    def shots_on_target_home(self) -> int:
        return int(self._stat(StatisticTypeId.SHOTS_ON_TARGET, "home"))

    @computed_field
    @property
    def shots_on_target_away(self) -> int:
        return int(self._stat(StatisticTypeId.SHOTS_ON_TARGET, "away"))

    @computed_field
    @property
    def dangerous_attacks_home(self) -> int:
        return int(self._stat(StatisticTypeId.DANGEROUS_ATTACKS, "home"))

    @computed_field
    @property
    def dangerous_attacks_away(self) -> int:
        return int(self._stat(StatisticTypeId.DANGEROUS_ATTACKS, "away"))

    @computed_field
    @property
    def possession_home_pct(self) -> float:
        value = self._stat(StatisticTypeId.BALL_POSSESSION, "home")
        return value if value > 0 else 50.0

    @computed_field
    @property
    def is_live(self) -> bool:
        return self.state_id not in FINISHED_STATE_IDS

    @computed_field
    @property
    def score_diff(self) -> int:
        """Home minus away."""
        return self.score_home - self.score_away
