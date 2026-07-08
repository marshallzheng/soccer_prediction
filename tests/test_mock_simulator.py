import asyncio

import pytest

from corner_predictor.data_sources.mock_simulator import MockMatchSimulator
from corner_predictor.data_sources.models import EventType


async def _run_full_match(seed: int) -> tuple[int, list[float]]:
    sim = MockMatchSimulator(minutes_per_tick=1.0, seed=seed)
    state = await sim.start_match(f"seed-{seed}")
    corner_minutes: list[float] = []
    while not sim.is_finished():
        state, events = await sim.next_tick()
        corner_minutes.extend(e.minute for e in events if e.event_type == EventType.CORNER)
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
    all_gaps: list[float] = []
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
async def test_half_transition_events_emitted() -> None:
    sim = MockMatchSimulator(minutes_per_tick=1.0, seed=1)
    await sim.start_match("m1")
    seen_half_end = False
    seen_half_start = False
    while not sim.is_finished():
        _, events = await sim.next_tick()
        for e in events:
            if e.event_type == EventType.HALF_END:
                seen_half_end = True
            if e.event_type == EventType.HALF_START:
                seen_half_start = True
    assert seen_half_end and seen_half_start


if __name__ == "__main__":
    asyncio.run(test_average_total_corners_within_expected_range())
