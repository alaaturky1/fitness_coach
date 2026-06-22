from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.auth import verify_ws_api_key
from app.core.models import AnalyzeFrameResponse, FrameInput
from app.storage.hybrid_storage import get_hybrid_sessions

router = APIRouter()


@router.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    if not verify_ws_api_key(websocket):
        await websocket.accept()
        await websocket.send_text(json.dumps({"error": "invalid_api_key"}))
        await websocket.close(code=1008)
        return

    await websocket.accept()
    hybrid_sessions = get_hybrid_sessions()
    session = hybrid_sessions.get(session_id)
    if session is None:
        await websocket.send_text(json.dumps({"error": "session_not_found"}))
        await websocket.close(code=1008)
        return

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                frame = FrameInput.model_validate(payload)
            except Exception:
                await websocket.send_text(json.dumps({"error": "invalid_frame"}))
                continue

            resp: AnalyzeFrameResponse = session.engine.analyze(frame)
            hybrid_sessions.update_session(session)
            await websocket.send_text(resp.model_dump_json())
    except WebSocketDisconnect:
        return

