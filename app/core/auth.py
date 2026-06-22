from __future__ import annotations

from fastapi import Header, HTTPException, WebSocket

from app.core.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()
    if not settings.require_api_key:
        return
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="server_api_key_not_configured")
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid_api_key")


def verify_ws_api_key(websocket: WebSocket) -> bool:
    """
    WebSocket auth: clients must send header X-API-Key.
    """

    settings = get_settings()
    if not settings.require_api_key:
        return True
    if not settings.api_key:
        return False
    # Browsers cannot set arbitrary headers for native WebSocket connections.
    # Support both header and query param to keep browser-based testers compatible.
    provided = websocket.headers.get("x-api-key") or websocket.query_params.get("x_api_key")
    return provided == settings.api_key

