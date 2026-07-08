from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Implements the MatchRunner Broadcaster protocol; fans out ticks to all subscribers of a match."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, match_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(match_id, []).append(websocket)

    def disconnect(self, match_id: str, websocket: WebSocket) -> None:
        conns = self._connections.get(match_id)
        if conns and websocket in conns:
            conns.remove(websocket)
            if not conns:
                del self._connections[match_id]

    async def publish(self, match_id: str, message: dict) -> None:
        stale: list[WebSocket] = []
        for ws in self._connections.get(match_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(match_id, ws)


manager = ConnectionManager()


@router.websocket("/ws/matches/{match_id}")
async def match_updates(websocket: WebSocket, match_id: str) -> None:
    await manager.connect(match_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(match_id, websocket)
