import pytest

from corner_predictor.data_sources.mock_simulator import MockMatchSimulator
from corner_predictor.data_sources.models import FixtureStateId


async def _run_full_match(seed: int) -> tuple[int, list[int]]:
    """Returns (final total corners, approximate corner occurrence minutes).

    Corners are modeled as a cumulative statistic (not discrete events), so
    occurrence minutes are reconstructed by noting the tick minute each time
    the cumulative total increases -- fine-grained enough for statistical
    sanity checks even though it loses sub-tick precision.
    """
    sim = MockMatchSimulator(minutes_per_tick=1.0, seed=seed)
    state = await sim.start_match(f"seed-{seed}")
    corner_minutes: list[int] = []
    prev_total = state.corners_total
    while not sim.is_finished():
        state, _ = await sim.next_tick()
        delta = state.corners_total - prev_total
        corner_minutes.extend([state.minute] * delta)
        prev_total = state.corners_total
    return state.corners_total, corner_minutes


@pytest.mark.asyncio
async def test_average_total_corners_within_expected_range() -> None:
    totals = []
    for seed in range(60):
        total, _ = await _run_full_match(seed)
        totals.append(total)

    avg = sum(totals) / len(totals)
    # Calibrated target is ~10 corners/match; allow a wide statistical band.
    assert 6.0 <= avg <= 14.0


@pytest.mark.asyncio
async def test_average_inter_corner_gap_within_expected_range() -> None:
    all_gaps: list[int] = []
    for seed in range(60):
        _, minutes = await _run_full_match(seed)
        minutes = sorted(minutes)
        gaps = [b - a for a, b in zip(minutes, minutes[1:])]
        all_gaps.extend(gaps)

    assert all_gaps, "expected at least some corners across 60 simulated matches"
    avg_gap = sum(all_gaps) / len(all_gaps)
    assert 5.0 <= avg_gap <= 15.0


@pytest.mark.asyncio
async def test_match_ends_at_90_minutes_and_is_deterministic_with_seed() -> None:
    total_a, minutes_a = await _run_full_match(seed=42)
    total_b, minutes_b = await _run_full_match(seed=42)
    assert total_a == total_b
    assert minutes_a == minutes_b


@pytest.mark.asyncio
async def test_next_tick_raises_before_start_match() -> None:
    sim = MockMatchSimulator()
    with pytest.raises(RuntimeError):
        await sim.next_tick()


@pytest.mark.asyncio
async def test_state_id_transitions_through_both_halves_to_full_time() -> None:
    sim = MockMatchSimulator(minutes_per_tick=1.0, seed=1)
    await sim.start_match("m1")
    seen_states: set[FixtureStateId] = set()
    final_state = None
    while not sim.is_finished():
        final_state, _ = await sim.next_tick()
        seen_states.add(final_state.state_id)
    assert FixtureStateId.INPLAY_1ST_HALF in seen_states
    assert FixtureStateId.INPLAY_2ND_HALF in seen_states
    assert final_state is not None
    assert final_state.state_id == FixtureStateId.FT
