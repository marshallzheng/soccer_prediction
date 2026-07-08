from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

Team = Literal["home", "away"]


class EventType(StrEnum):
    CORNER = "corner"
    SHOT = "shot"
    SHOT_ON_TARGET = "shot_on_target"
    GOAL = "goal"
    DANGEROUS_ATTACK = "dangerous_attack"
    POSSESSION_UPDATE = "possession_update"
    CARD = "card"
    SUBSTITUTION = "substitution"
    HALF_START = "half_start"
    HALF_END = "half_end"
    MATCH_END = "match_end"


class MatchEvent(BaseModel):
    match_id: str
    minute: float
    event_type: EventType
    team: Team | None = None
    payload: dict[str, Any] = {}


class MatchState(BaseModel):
    """Vendor-neutral snapshot of a live match at a point in time."""

    match_id: str
    minute: float
    period: Literal[1, 2] = 1
    home_team: str
    away_team: str
    score_home: int = 0
    score_away: int = 0
    corners_home: int = 0
    corners_away: int = 0
    shots_home: int = 0
    shots_away: int = 0
    shots_on_target_home: int = 0
    shots_on_target_away: int = 0
    dangerous_attacks_home: int = 0
    dangerous_attacks_away: int = 0
    possession_home_pct: float = 50.0
    is_live: bool = True

    @property
    def corners_total(self) -> int:
        return self.corners_home + self.corners_away

    @property
    def score_diff(self) -> int:
        """Home minus away."""
        return self.score_home - self.score_away
