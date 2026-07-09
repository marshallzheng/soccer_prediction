from typing import Protocol

from corner_predictor.data_sources.models import MatchEvent, MatchState


class MatchDataSource(Protocol):
    """Interface for anything that can drive a live match tick loop, speaking Sportmonks'
    data shape (MatchState/MatchEvent). Both the mock simulator and a future Sportmonks API
    adapter implement this same shape, so MatchRunner never needs to know which one it's
    talking to.
    """

    async def start_match(self, fixture_id: str) -> MatchState: ...

    async def next_tick(self) -> tuple[MatchState, list[MatchEvent]]:
        """Advance one tick and return the new state plus events since the last tick."""
        ...

    def is_finished(self) -> bool: ...
