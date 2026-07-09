import asyncio
from typing import Protocol

from pydantic import BaseModel

from corner_predictor.config import settings
from corner_predictor.data_sources.base import MatchDataSource
from corner_predictor.data_sources.models import MatchEvent, MatchState
from corner_predictor.features.engine import FeatureEngine
from corner_predictor.features.models import FeatureSnapshot
from corner_predictor.model.probability import CornerProbabilityModel, get_model
from corner_predictor.model.rate_estimator import RateEstimator
from corner_predictor.model.schemas import ProbabilityResult
from corner_predictor.persistence.repository import MatchRepository


class Broadcaster(Protocol):
    async def publish(self, fixture_id: str, message: dict) -> None: ...


class LiveUpdate(BaseModel):
    """Single serialization point for what gets pushed to WebSocket subscribers each tick."""

    fixture_id: str
    minute: int
    state_id: int
    home_team: str
    away_team: str
    score_home: int
    score_away: int
    corners_home: int
    corners_away: int
    possession_home_pct: float
    is_live: bool
    threshold: float
    prob_over: float
    prob_under: float
    expected_total_corners: float
    pmf: list[tuple[int, float]]
    lambda_remaining: float


class MatchRunner:
    """Owns one live match's tick loop: data source -> features -> model -> persistence -> broadcast."""

    def __init__(
        self,
        fixture_id: str,
        data_source: MatchDataSource,
        repository: MatchRepository,
        source_name: str = "mock",
        threshold: float | None = None,
        probability_model: CornerProbabilityModel | None = None,
        broadcaster: Broadcaster | None = None,
        tick_interval_seconds: float | None = None,
    ) -> None:
        self.fixture_id = fixture_id
        self.data_source = data_source
        self.repository = repository
        self.source_name = source_name
        self.threshold = threshold if threshold is not None else settings.default_threshold
        self.model = probability_model or get_model()
        self.broadcaster = broadcaster
        self.tick_interval_seconds = (
            tick_interval_seconds if tick_interval_seconds is not None else settings.tick_interval_seconds
        )

        self.feature_engine = FeatureEngine()
        self.rate_estimator = RateEstimator()
        self.prior_rate_per_min = settings.prior_corners_per_90min / 90.0

        self._events: list[MatchEvent] = []
        """Discrete goal/card/substitution events only -- see StatisticEntry for cumulative stats."""
        self._state_history: list[MatchState] = []
        """Past ticks' snapshots, used by FeatureEngine to compute rolling-window deltas."""
        self._stopped = False

        self.latest_state: MatchState | None = None
        self.latest_features: FeatureSnapshot | None = None
        self.latest_rate: float | None = None
        self.latest_result: ProbabilityResult | None = None

    def stop(self) -> None:
        self._stopped = True

    async def run(self) -> None:
        state = await self.data_source.start_match(self.fixture_id)
        self.repository.create_match(self.fixture_id, state.home_team, state.away_team, source=self.source_name)

        while not self.data_source.is_finished() and not self._stopped:
            state, events = await self.data_source.next_tick()
            self._events.extend(events)

            features = self.feature_engine.compute(state, self._state_history)
            self._state_history.append(state)
            rate = self.rate_estimator.estimate_remaining_rate(features, self.prior_rate_per_min)
            result = self.model.predict(
                observed_corners=state.corners_total,
                minutes_remaining=features.minutes_remaining,
                rate_per_min=rate,
                threshold=self.threshold,
            )

            self.latest_state = state
            self.latest_features = features
            self.latest_rate = rate
            self.latest_result = result

            self.repository.save_tick(self.fixture_id, state, features, result)
            self.repository.save_events(events)

            if self.broadcaster is not None:
                update = self._build_live_update(state, result)
                await self.broadcaster.publish(self.fixture_id, update.model_dump())

            await asyncio.sleep(self.tick_interval_seconds)

        if self.latest_state is not None:
            self.repository.finalize_match(
                self.fixture_id,
                final_corners_home=self.latest_state.corners_home,
                final_corners_away=self.latest_state.corners_away,
            )

    def predict_for_threshold(self, threshold: float) -> ProbabilityResult | None:
        """Recompute a probability result for an arbitrary threshold using the latest known rate."""
        if self.latest_state is None or self.latest_features is None or self.latest_rate is None:
            return None
        return self.model.predict(
            observed_corners=self.latest_state.corners_total,
            minutes_remaining=self.latest_features.minutes_remaining,
            rate_per_min=self.latest_rate,
            threshold=threshold,
        )

    def _build_live_update(self, state: MatchState, result: ProbabilityResult) -> LiveUpdate:
        return LiveUpdate(
            fixture_id=state.fixture_id,
            minute=state.minute,
            state_id=int(state.state_id),
            home_team=state.home_team,
            away_team=state.away_team,
            score_home=state.score_home,
            score_away=state.score_away,
            corners_home=state.corners_home,
            corners_away=state.corners_away,
            possession_home_pct=state.possession_home_pct,
            is_live=state.is_live,
            threshold=result.threshold,
            prob_over=result.prob_over,
            prob_under=result.prob_under,
            expected_total_corners=result.expected_total_corners,
            pmf=result.pmf,
            lambda_remaining=result.lambda_remaining,
        )


class MatchRegistry:
    """In-memory registry of running MatchRunners + their asyncio tasks."""

    def __init__(self) -> None:
        self._runners: dict[str, MatchRunner] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def register(self, runner: MatchRunner) -> None:
        self._runners[runner.fixture_id] = runner
        task = asyncio.create_task(runner.run())
        task.add_done_callback(lambda _: self._forget(runner.fixture_id))
        self._tasks[runner.fixture_id] = task

    def _forget(self, fixture_id: str) -> None:
        """Drop the runner once its tick loop finishes, so long-running servers
        don't accumulate MatchRunner/state-history state for matches that ended."""
        self._runners.pop(fixture_id, None)
        self._tasks.pop(fixture_id, None)

    def get(self, fixture_id: str) -> MatchRunner | None:
        return self._runners.get(fixture_id)

    def list_ids(self) -> list[str]:
        return list(self._runners.keys())


registry = MatchRegistry()
