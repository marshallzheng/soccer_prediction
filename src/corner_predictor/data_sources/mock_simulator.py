import zlib

import numpy as np

from corner_predictor.data_sources.models import (
    FINISHED_STATE_IDS,
    EventTypeId,
    FixtureStateId,
    Location,
    MatchEvent,
    MatchState,
    StatisticEntry,
    StatisticTypeId,
)

# Calibrated so a full 90-minute match averages ~10 total corners, ~24 shots,
# ~2.6 goals, and ~100 dangerous attacks (split across both teams), roughly in
# line with typical professional match statistics.
_BASE_CORNER_RATE_PER_TEAM_PER_MIN = 5.0 / 90.0
_BASE_SHOT_RATE_PER_TEAM_PER_MIN = 12.0 / 90.0
_SHOT_ON_TARGET_FRACTION = 0.35
_BASE_DANGEROUS_ATTACK_RATE_PER_TEAM_PER_MIN = 50.0 / 90.0
_BASE_GOAL_RATE_PER_TEAM_PER_MIN = 1.3 / 90.0

_INTENSITY_MEAN_REVERSION = 0.08
_INTENSITY_NOISE_STD = 0.12
_INTENSITY_MIN = 0.2
_INTENSITY_MAX = 2.5
_BURST_PROBABILITY_PER_MIN = 0.03
_BURST_SIZE = 0.8
_TRAILING_URGENCY_BUMP = 0.35


def _stable_participant_id(team_name: str) -> int:
    """Deterministic id from a team name, so re-simulating the same team yields the same id
    (mirrors Sportmonks' persistent per-team participant_id)."""
    return zlib.crc32(team_name.encode()) % 1_000_000


class MockMatchSimulator:
    """Generates a plausible live event stream without needing a real data-provider API.

    Corners, shots, shots-on-target, and dangerous attacks are modeled as
    cumulative `statistics` values (mirroring Sportmonks' shape) rather than
    discrete events -- only goals are discrete `MatchEvent`s. Attack intensity
    per team follows a mean-reverting random walk (with occasional momentum
    "bursts") that drives the Poisson rate of every stat each tick, so
    downstream features actually correlate with the corner rate, the way they
    would against a real feed.
    """

    def __init__(
        self,
        home_team: str = "Home FC",
        away_team: str = "Away United",
        minutes_per_tick: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.home_team = home_team
        self.away_team = away_team
        self.minutes_per_tick = minutes_per_tick
        self._rng = np.random.default_rng(seed)

        self._fixture_id: str | None = None
        self._state: MatchState | None = None
        self._minute_float = 0.0
        self._intensity_home = 1.0
        self._intensity_away = 1.0
        self._stats: dict[tuple[StatisticTypeId, Location], float] = {}
        self._next_event_id = 1

    async def start_match(self, fixture_id: str) -> MatchState:
        self._fixture_id = fixture_id
        self._minute_float = 0.0
        self._stats = {
            (type_id, location): 0.0
            for type_id in StatisticTypeId
            for location in ("home", "away")
        }
        self._stats[(StatisticTypeId.BALL_POSSESSION, "home")] = 50.0
        self._stats[(StatisticTypeId.BALL_POSSESSION, "away")] = 50.0
        self._state = MatchState(
            fixture_id=fixture_id,
            state_id=FixtureStateId.NS,
            minute=0,
            home_participant_id=_stable_participant_id(self.home_team),
            away_participant_id=_stable_participant_id(self.away_team),
            home_team=self.home_team,
            away_team=self.away_team,
            statistics=self._statistics_snapshot(),
        )
        return self._state

    def is_finished(self) -> bool:
        return self._state is not None and self._state.state_id in FINISHED_STATE_IDS

    async def next_tick(self) -> tuple[MatchState, list[MatchEvent]]:
        if self._state is None or self._fixture_id is None:
            raise RuntimeError("start_match() must be called before next_tick()")
        if self.is_finished():
            return self._state, []

        state = self._state
        dt = self.minutes_per_tick
        events: list[MatchEvent] = []

        if state.state_id == FixtureStateId.NS:
            state.state_id = FixtureStateId.INPLAY_1ST_HALF

        prev_minute = self._minute_float
        self._minute_float = min(prev_minute + dt, 90.0)
        state.minute = int(round(self._minute_float))

        if prev_minute < 45.0 <= self._minute_float and state.state_id == FixtureStateId.INPLAY_1ST_HALF:
            state.state_id = FixtureStateId.INPLAY_2ND_HALF

        self._step_intensity(dt)

        self._accumulate(StatisticTypeId.CORNERS, _BASE_CORNER_RATE_PER_TEAM_PER_MIN, dt)
        self._accumulate(StatisticTypeId.SHOTS_TOTAL, _BASE_SHOT_RATE_PER_TEAM_PER_MIN, dt)
        self._accumulate(
            StatisticTypeId.SHOTS_ON_TARGET, _BASE_SHOT_RATE_PER_TEAM_PER_MIN * _SHOT_ON_TARGET_FRACTION, dt
        )
        self._accumulate(StatisticTypeId.DANGEROUS_ATTACKS, _BASE_DANGEROUS_ATTACK_RATE_PER_TEAM_PER_MIN, dt)
        events.extend(self._sample_goal_events(dt))

        total_intensity = self._intensity_home + self._intensity_away
        possession_home = round(100.0 * self._intensity_home / total_intensity, 1)
        self._stats[(StatisticTypeId.BALL_POSSESSION, "home")] = possession_home
        self._stats[(StatisticTypeId.BALL_POSSESSION, "away")] = round(100.0 - possession_home, 1)

        state.statistics = self._statistics_snapshot()

        if self._minute_float >= 90.0:
            state.state_id = FixtureStateId.FT

        return state, events

    def _statistics_snapshot(self) -> list[StatisticEntry]:
        home_id = _stable_participant_id(self.home_team)
        away_id = _stable_participant_id(self.away_team)
        participant_id = {"home": home_id, "away": away_id}
        return [
            StatisticEntry(type_id=type_id, participant_id=participant_id[location], location=location, value=value)
            for (type_id, location), value in self._stats.items()
        ]

    def _step_intensity(self, dt: float) -> None:
        for attr in ("_intensity_home", "_intensity_away"):
            value = getattr(self, attr)
            drift = _INTENSITY_MEAN_REVERSION * (1.0 - value) * dt
            noise = self._rng.normal(0.0, _INTENSITY_NOISE_STD * np.sqrt(dt))
            value += drift + noise
            if self._rng.random() < _BURST_PROBABILITY_PER_MIN * dt:
                value += self._rng.normal(_BURST_SIZE, 0.2)
            value = float(np.clip(value, _INTENSITY_MIN, _INTENSITY_MAX))
            setattr(self, attr, value)

    def _accumulate(self, type_id: StatisticTypeId, base_rate_per_team_per_min: float, dt: float) -> None:
        for location, intensity in (("home", self._intensity_home), ("away", self._intensity_away)):
            lam = base_rate_per_team_per_min * intensity * dt
            count = int(self._rng.poisson(lam))
            if count > 0:
                self._stats[(type_id, location)] += count

    def _sample_goal_events(self, dt: float) -> list[MatchEvent]:
        assert self._state is not None and self._fixture_id is not None
        state = self._state
        events: list[MatchEvent] = []
        for location, intensity in (("home", self._intensity_home), ("away", self._intensity_away)):
            lam = _BASE_GOAL_RATE_PER_TEAM_PER_MIN * intensity * dt
            count = int(self._rng.poisson(lam))
            if count <= 0:
                continue
            for _ in range(count):
                if location == "home":
                    state.score_home += 1
                    self._intensity_away = float(
                        np.clip(self._intensity_away + _TRAILING_URGENCY_BUMP, _INTENSITY_MIN, _INTENSITY_MAX)
                    )
                else:
                    state.score_away += 1
                    self._intensity_home = float(
                        np.clip(self._intensity_home + _TRAILING_URGENCY_BUMP, _INTENSITY_MIN, _INTENSITY_MAX)
                    )
                participant_id = state.home_participant_id if location == "home" else state.away_participant_id
                events.append(
                    MatchEvent(
                        id=self._next_event_id,
                        fixture_id=self._fixture_id,
                        type_id=EventTypeId.GOAL,
                        participant_id=participant_id,
                        location=location,
                        minute=state.minute,
                        result=f"{state.score_home}-{state.score_away}",
                    )
                )
                self._next_event_id += 1
        return events
