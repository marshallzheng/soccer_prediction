from typing import Protocol

from corner_predictor.data_sources.models import MatchEvent, MatchState


class MatchDataSource(Protocol):
    """Vendor-neutral interface for anything that can drive a live match tick loop.

    Both the mock simulator and any future real-API adapter implement this
    same shape, so MatchRunner never needs to know which one it's talking to.
    """

    async def start_match(self, match_id: str) -> MatchState: ...

    async def next_tick(self) -> tuple[MatchState, list[MatchEvent]]:
        """Advance one tick and return the new state plus events since the last tick."""
        ...

    def is_finished(self) -> bool: ...
