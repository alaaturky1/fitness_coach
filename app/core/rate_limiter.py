from __future__ import annotations

import time
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
from starlette.responses import JSONResponse

from app.core.config import get_settings


def get_session_id(request: Request) -> str:
    """Get session identifier for rate limiting"""
    # Try API key first, then IP address
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    return f"ip:{get_remote_address(request)}"


# Initialize rate limiter
limiter = Limiter(key_func=get_session_id)


class RateLimitConfig:
    """Rate limit configuration for different endpoints"""
    
    # API endpoints (requests per minute)
    START_SESSION = "10/minute"
    ANALYZE_FRAME = "60/minute"
    END_SESSION = "10/minute"
    SESSION_SUMMARY = "30/minute"
    STATS = "20/minute"
    HEALTH = "100/minute"
    
    # WebSocket (connections per minute)
    WEBSOCKET_CONNECT = "10/minute"


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom handler for rate limit exceeded"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": str(exc.retry_after) if hasattr(exc, 'retry_after') else "60"
        },
        headers={"Retry-After": str(getattr(exc, 'retry_after', 60))}
    )


class SessionRateLimiter:
    """Session-specific rate limiting for frame analysis"""
    
    def __init__(self) -> None:
        self._session_limits: dict[str, dict] = {}
        self._settings = get_settings()
    
    def check_frame_rate(self, session_id: str, timestamp: float) -> bool:
        """Check if session exceeds frame analysis rate limit"""
        current_time = time.time()
        
        if session_id not in self._session_limits:
            self._session_limits[session_id] = {
                "frames": [],
                "last_cleanup": current_time
            }
        
        session_data = self._session_limits[session_id]
        
        # Clean up old frames (older than 1 minute)
        session_data["frames"] = [
            t for t in session_data["frames"] 
            if current_time - t < 60
        ]
        
        # Check rate limit (30 frames per minute per session)
        if len(session_data["frames"]) >= 30:
            return False
        
        # Add current frame
        session_data["frames"].append(timestamp)
        return True
    
    def cleanup_expired_sessions(self, max_age: float = 3600) -> None:
        """Clean up expired session data"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, data in self._session_limits.items():
            if current_time - data.get("last_cleanup", 0) > max_age:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._session_limits[session_id]


# Global session rate limiter
session_rate_limiter = SessionRateLimiter()


def check_session_rate_limit(session_id: str, timestamp: float) -> bool:
    """Check session-specific rate limit"""
    return session_rate_limiter.check_frame_rate(session_id, timestamp)
