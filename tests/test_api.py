import pytest
from fastapi.testclient import TestClient

from corner_predictor.config import settings
from corner_predictor.main import app


@pytest.fixture(autouse=True)
def fast_ticks(monkeypatch):
    # Real tick loop defaults to 1s/tick; speed it up so tests don't sit waiting on sleeps.
    monkeypatch.setattr(settings, "tick_interval_seconds", 0.01)


def test_simulate_match_and_stream_websocket_update() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/matches/simulate",
            json={"home_team": "Red FC", "away_team": "Blue FC", "threshold": 9.5, "seed": 1},
        )
        assert response.status_code == 200
        match_id = response.json()["match_id"]

        with client.websocket_connect(f"/ws/matches/{match_id}") as ws:
            message = ws.receive_json()

        assert message["match_id"] == match_id
        assert message["home_team"] == "Red FC"
        assert message["away_team"] == "Blue FC"
        assert 0.0 <= message["prob_over"] <= 1.0
        assert isinstance(message["pmf"], list)

        # Give the background tick loop a moment then check REST reflects the running match.
        list_response = client.get("/api/matches")
        assert list_response.status_code == 200
        ids = [m["id"] for m in list_response.json()]
        assert match_id in ids

        detail_response = client.get(f"/api/matches/{match_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["is_running"] is True
        assert detail["threshold"] == 9.5


def test_get_unknown_match_returns_404() -> None:
    with TestClient(app) as client:
        response = client.get("/api/matches/does-not-exist")
        assert response.status_code == 404


def test_match_history_returns_ticks_after_running_briefly() -> None:
    with TestClient(app) as client:
        response = client.post("/api/matches/simulate", json={"seed": 2})
        match_id = response.json()["match_id"]

        with client.websocket_connect(f"/ws/matches/{match_id}") as ws:
            ws.receive_json()
            ws.receive_json()

        history = client.get(f"/api/matches/{match_id}/history")
        assert history.status_code == 200
        ticks = history.json()
        assert len(ticks) >= 1
        assert "prob_over" in ticks[0]
