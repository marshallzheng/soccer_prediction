import asyncio

from corner_predictor.data_sources.mock_simulator import MockMatchSimulator
from corner_predictor.engine.match_runner import MatchRunner
from corner_predictor.persistence.repository import MatchRepository


class RecordingBroadcaster:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def publish(self, fixture_id: str, message: dict) -> None:
        self.messages.append(message)


async def test_full_match_runs_headless_and_persists_ticks() -> None:
    simulator = MockMatchSimulator(minutes_per_tick=1.0, seed=7)
    repository = MatchRepository()
    broadcaster = RecordingBroadcaster()
    runner = MatchRunner(
        fixture_id="test-match",
        data_source=simulator,
        repository=repository,
        threshold=9.5,
        broadcaster=broadcaster,
        tick_interval_seconds=0.0,
    )

    await runner.run()

    assert runner.latest_state is not None
    assert runner.latest_state.minute == 90
    assert runner.latest_result is not None
    assert 0.0 <= runner.latest_result.prob_over <= 1.0

    history = repository.get_tick_history("test-match")
    assert len(history) == 90
    assert history[-1].minute == 90

    match = repository.get_match("test-match")
    assert match is not None
    assert match.status == "finished"
    assert match.final_corners_home == runner.latest_state.corners_home
    assert match.final_corners_away == runner.latest_state.corners_away

    assert len(broadcaster.messages) == 90
    last_message = broadcaster.messages[-1]
    assert last_message["fixture_id"] == "test-match"
    assert 0.0 <= last_message["prob_over"] <= 1.0


async def test_predict_for_threshold_uses_latest_rate() -> None:
    simulator = MockMatchSimulator(minutes_per_tick=1.0, seed=3)
    repository = MatchRepository()
    runner = MatchRunner(
        fixture_id="test-match-2",
        data_source=simulator,
        repository=repository,
        threshold=9.5,
        tick_interval_seconds=0.0,
    )

    assert runner.predict_for_threshold(9.5) is None  # before any ticks

    await runner.run()

    result_low = runner.predict_for_threshold(0.5)
    result_high = runner.predict_for_threshold(50.5)
    assert result_low.prob_over >= result_high.prob_over


async def test_stop_halts_the_loop_early() -> None:
    simulator = MockMatchSimulator(minutes_per_tick=1.0, seed=9)
    repository = MatchRepository()
    runner = MatchRunner(
        fixture_id="test-match-3",
        data_source=simulator,
        repository=repository,
        tick_interval_seconds=0.0,
    )

    async def stop_after_a_few_ticks():
        # Let the loop run briefly then request a stop.
        await asyncio.sleep(0.01)
        runner.stop()

    await asyncio.gather(runner.run(), stop_after_a_few_ticks())

    assert runner.latest_state is not None
    assert runner.latest_state.minute < 90
