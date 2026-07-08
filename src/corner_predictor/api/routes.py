import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from corner_predictor.api.deps import get_registry, get_repository
from corner_predictor.api.websocket import manager
from corner_predictor.config import settings
from corner_predictor.data_sources.mock_simulator import MockMatchSimulator
from corner_predictor.engine.match_runner import MatchRegistry, MatchRunner
from corner_predictor.persistence.repository import MatchRepository

router = APIRouter(prefix="/api")


class SimulateMatchRequest(BaseModel):
    home_team: str = "Home FC"
    away_team: str = "Away United"
    threshold: float = Field(default_factory=lambda: settings.default_threshold)
    seed: int | None = None


class SimulateMatchResponse(BaseModel):
    match_id: str


class MatchSummary(BaseModel):
    id: str
    home_team: str
    away_team: str
    source: str
    status: str
    final_corners_home: int | None
    final_corners_away: int | None
    is_running: bool


class MatchDetail(MatchSummary):
    minute: float | None = None
    corners_home: int | None = None
    corners_away: int | None = None
    threshold: float | None = None
    prob_over: float | None = None
    prob_under: float | None = None
    expected_total_corners: float | None = None
    pmf: list[tuple[int, float]] | None = None


@router.get("/matches", response_model=list[MatchSummary])
def list_matches(
    repository: MatchRepository = Depends(get_repository),
) -> list[MatchSummary]:
    return [
        MatchSummary(
            id=m.id,
            home_team=m.home_team,
            away_team=m.away_team,
            source=m.source,
            status=m.status,
            final_corners_home=m.final_corners_home,
            final_corners_away=m.final_corners_away,
            is_running=m.status == "live",
        )
        for m in repository.list_matches()
    ]


@router.post("/matches/simulate", response_model=SimulateMatchResponse)
async def simulate_match(
    body: SimulateMatchRequest,
    repository: MatchRepository = Depends(get_repository),
    reg: MatchRegistry = Depends(get_registry),
) -> SimulateMatchResponse:
    match_id = str(uuid.uuid4())
    simulator = MockMatchSimulator(
        home_team=body.home_team,
        away_team=body.away_team,
        minutes_per_tick=settings.minutes_per_tick,
        seed=body.seed,
    )
    runner = MatchRunner(
        match_id=match_id,
        data_source=simulator,
        repository=repository,
        source_name="mock",
        threshold=body.threshold,
        broadcaster=manager,
    )
    reg.register(runner)
    return SimulateMatchResponse(match_id=match_id)


@router.get("/matches/{match_id}", response_model=MatchDetail)
def get_match(
    match_id: str,
    repository: MatchRepository = Depends(get_repository),
    reg: MatchRegistry = Depends(get_registry),
) -> MatchDetail:
    match = repository.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    runner = reg.get(match_id)
    detail = MatchDetail(
        id=match.id,
        home_team=match.home_team,
        away_team=match.away_team,
        source=match.source,
        status=match.status,
        final_corners_home=match.final_corners_home,
        final_corners_away=match.final_corners_away,
        is_running=match.status == "live",
    )

    if runner is not None and runner.latest_state is not None and runner.latest_result is not None:
        state = runner.latest_state
        result = runner.latest_result
        detail.minute = state.minute
        detail.corners_home = state.corners_home
        detail.corners_away = state.corners_away
        detail.threshold = result.threshold
        detail.prob_over = result.prob_over
        detail.prob_under = result.prob_under
        detail.expected_total_corners = result.expected_total_corners
        detail.pmf = result.pmf

    return detail


@router.get("/matches/{match_id}/history")
def get_match_history(
    match_id: str,
    repository: MatchRepository = Depends(get_repository),
) -> list[dict]:
    if repository.get_match(match_id) is None:
        raise HTTPException(status_code=404, detail="Match not found")
    ticks = repository.get_tick_history(match_id)
    return [
        {
            "minute": t.minute,
            "corners_home": t.corners_home,
            "corners_away": t.corners_away,
            "threshold": t.threshold,
            "prob_over": t.prob_over,
            "lambda_remaining": t.lambda_remaining,
        }
        for t in ticks
    ]
