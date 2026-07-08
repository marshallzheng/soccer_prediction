import numpy as np

from corner_predictor.data_sources.models import EventType, MatchEvent, MatchState

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


class MockMatchSimulator:
    """Generates a plausible live event stream without needing a real data-provider API.

    Attack intensity per team follows a mean-reverting random walk (with
    occasional momentum "bursts") that drives the Poisson rate of corners,
    shots, and dangerous attacks each tick -- so downstream features actually
    correlate with the corner rate, the way they would against a real feed.
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

        self._match_id: str | None = None
        self._state: MatchState | None = None
        self._intensity_home = 1.0
        self._intensity_away = 1.0
        self._finished = False

    async def start_match(self, match_id: str) -> MatchState:
        self._match_id = match_id
        self._state = MatchState(
            match_id=match_id,
            minute=0.0,
            period=1,
            home_team=self.home_team,
            away_team=self.away_team,
        )
        self._finished = False
        return self._state

    def is_finished(self) -> bool:
        return self._finished

    async def next_tick(self) -> tuple[MatchState, list[MatchEvent]]:
        if self._state is None or self._match_id is None:
            raise RuntimeError("start_match() must be called before next_tick()")
        if self._finished:
            return self._state, []

        state = self._state
        dt = self.minutes_per_tick
        events: list[MatchEvent] = []

        prev_minute = state.minute
        new_minute = min(prev_minute + dt, 90.0)
        state.minute = new_minute

        if prev_minute < 45.0 <= new_minute and state.period == 1:
            events.append(self._event(EventType.HALF_END, prev_minute))
            state.period = 2
            events.append(self._event(EventType.HALF_START, new_minute))

        self._step_intensity(dt)

        events.extend(self._sample_events(EventType.CORNER, dt, "corners"))
        events.extend(self._sample_events(EventType.SHOT, dt, "shots"))
        events.extend(self._sample_events(EventType.SHOT_ON_TARGET, dt, "shots_on_target"))
        events.extend(self._sample_events(EventType.DANGEROUS_ATTACK, dt, "dangerous_attacks"))
        events.extend(self._sample_goal_events(dt))

        total_intensity = self._intensity_home + self._intensity_away
        state.possession_home_pct = round(100.0 * self._intensity_home / total_intensity, 1)

        if new_minute >= 90.0:
            state.is_live = False
            self._finished = True
            events.append(self._event(EventType.MATCH_END, new_minute))

        return state, events

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

    def _sample_events(self, event_type: EventType, dt: float, field_prefix: str) -> list[MatchEvent]:
        assert self._state is not None
        state = self._state
        events: list[MatchEvent] = []
        base_rate = {
            EventType.CORNER: _BASE_CORNER_RATE_PER_TEAM_PER_MIN,
            EventType.SHOT: _BASE_SHOT_RATE_PER_TEAM_PER_MIN,
            EventType.SHOT_ON_TARGET: _BASE_SHOT_RATE_PER_TEAM_PER_MIN * _SHOT_ON_TARGET_FRACTION,
            EventType.DANGEROUS_ATTACK: _BASE_DANGEROUS_ATTACK_RATE_PER_TEAM_PER_MIN,
        }[event_type]

        for team, intensity in (("home", self._intensity_home), ("away", self._intensity_away)):
            lam = base_rate * intensity * dt
            count = int(self._rng.poisson(lam))
            if count <= 0:
                continue
            field = f"{field_prefix}_{team}"
            setattr(state, field, getattr(state, field) + count)
            events.extend(
                MatchEvent(match_id=state.match_id, minute=state.minute, event_type=event_type, team=team)
                for _ in range(count)
            )
        return events

    def _sample_goal_events(self, dt: float) -> list[MatchEvent]:
        assert self._state is not None
        state = self._state
        events: list[MatchEvent] = []
        for team, intensity in (("home", self._intensity_home), ("away", self._intensity_away)):
            lam = _BASE_GOAL_RATE_PER_TEAM_PER_MIN * intensity * dt
            count = int(self._rng.poisson(lam))
            if count <= 0:
                continue
            for _ in range(count):
                if team == "home":
                    state.score_home += 1
                    self._intensity_away = float(
                        np.clip(self._intensity_away + _TRAILING_URGENCY_BUMP, _INTENSITY_MIN, _INTENSITY_MAX)
                    )
                else:
                    state.score_away += 1
                    self._intensity_home = float(
                        np.clip(self._intensity_home + _TRAILING_URGENCY_BUMP, _INTENSITY_MIN, _INTENSITY_MAX)
                    )
                events.append(
                    MatchEvent(match_id=state.match_id, minute=state.minute, event_type=EventType.GOAL, team=team)
                )
        return events

    def _event(self, event_type: EventType, minute: float) -> MatchEvent:
        assert self._match_id is not None
        return MatchEvent(match_id=self._match_id, minute=minute, event_type=event_type)
