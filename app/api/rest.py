from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import require_api_key
from app.core.models import (
    AnalyzeFrameRequest,
    AnalyzeFrameResponse,
    EndSessionRequest,
    SessionSummaryResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from app.storage.hybrid_storage import get_hybrid_sessions

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/start-session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    started_at = perf_counter()
    hybrid_sessions = get_hybrid_sessions()
    session = hybrid_sessions.create_session(language=req.language, level=req.level)
    
    # Log session start
    from app.core.enhanced_logging import get_logger
    logger = get_logger()
    logger.session_started(
        session_id=session.session_id,
        language=req.language.value,
        level=req.level.value
    )
    
    # Register session in metrics
    from app.core.metrics import get_metrics
    metrics = get_metrics()
    metrics.register_session(session.session_id)
    metrics.record_request(
        method="POST",
        path="/start-session",
        status_code=200,
        duration_ms=(perf_counter() - started_at) * 1000.0,
        session_id=session.session_id,
    )
    
    return StartSessionResponse(session_id=session.session_id, ws_url=f"/ws/session/{session.session_id}")


@router.post("/analyze-frame", response_model=AnalyzeFrameResponse)
def analyze_frame(req: AnalyzeFrameRequest) -> AnalyzeFrameResponse:
    started_at = perf_counter()
    hybrid_sessions = get_hybrid_sessions()
    session = hybrid_sessions.get(req.session_id)
    from app.core.metrics import get_metrics
    metrics = get_metrics()
    if session is None:
        metrics.record_request(
            method="POST",
            path="/analyze-frame",
            status_code=404,
            duration_ms=(perf_counter() - started_at) * 1000.0,
            session_id=req.session_id,
        )
        raise HTTPException(status_code=404, detail="session_not_found")
    
    # Check session-specific rate limit
    from app.core.rate_limiter import check_session_rate_limit
    if not check_session_rate_limit(req.session_id, req.frame.timestamp):
        duration_ms = (perf_counter() - started_at) * 1000.0
        metrics.record_session_frame(req.session_id, duration_ms, is_error=True)
        metrics.record_request(
            method="POST",
            path="/analyze-frame",
            status_code=429,
            duration_ms=duration_ms,
            session_id=req.session_id,
        )
        raise HTTPException(status_code=429, detail="session_rate_limit_exceeded")
    
    try:
        result = session.engine.analyze(req.frame)
    except Exception:
        duration_ms = (perf_counter() - started_at) * 1000.0
        metrics.record_session_frame(req.session_id, duration_ms, is_error=True)
        metrics.record_request(
            method="POST",
            path="/analyze-frame",
            status_code=500,
            duration_ms=duration_ms,
            session_id=req.session_id,
        )
        raise
    hybrid_sessions.update_session(session)
    duration_ms = (perf_counter() - started_at) * 1000.0
    metrics.record_session_frame(req.session_id, duration_ms)
    metrics.record_request(
        method="POST",
        path="/analyze-frame",
        status_code=200,
        duration_ms=duration_ms,
        session_id=req.session_id,
    )
    return result


@router.post("/end-session", response_model=SessionSummaryResponse)
def end_session(req: EndSessionRequest) -> SessionSummaryResponse:
    started_at = perf_counter()
    hybrid_sessions = get_hybrid_sessions()
    session = hybrid_sessions.get(req.session_id)
    from app.core.metrics import get_metrics
    metrics = get_metrics()
    if session is None:
        metrics.record_request(
            method="POST",
            path="/end-session",
            status_code=404,
            duration_ms=(perf_counter() - started_at) * 1000.0,
            session_id=req.session_id,
        )
        raise HTTPException(status_code=404, detail="session_not_found")
    session.ended = True
    hybrid_sessions.update_session(session)
    summary = session.engine.summary()
    metrics.remove_session(req.session_id)
    metrics.record_request(
        method="POST",
        path="/end-session",
        status_code=200,
        duration_ms=(perf_counter() - started_at) * 1000.0,
        session_id=req.session_id,
    )
    return summary


@router.get("/session-summary/{session_id}", response_model=SessionSummaryResponse)
def session_summary(session_id: str) -> SessionSummaryResponse:
    started_at = perf_counter()
    hybrid_sessions = get_hybrid_sessions()
    session = hybrid_sessions.get(session_id)
    from app.core.metrics import get_metrics
    metrics = get_metrics()
    if session is None:
        metrics.record_request(
            method="GET",
            path="/session-summary/{session_id}",
            status_code=404,
            duration_ms=(perf_counter() - started_at) * 1000.0,
            session_id=session_id,
        )
        raise HTTPException(status_code=404, detail="session_not_found")
    summary = session.engine.summary()
    metrics.record_request(
        method="GET",
        path="/session-summary/{session_id}",
        status_code=200,
        duration_ms=(perf_counter() - started_at) * 1000.0,
        session_id=session_id,
    )
    return summary


@router.get("/stats")
def get_stats() -> dict:
    hybrid_sessions = get_hybrid_sessions()
    storage_stats = hybrid_sessions.get_stats()
    
    # Get metrics
    from app.core.metrics import get_metrics
    metrics = get_metrics()
    system_metrics = metrics.collect_system_metrics()
    endpoint_stats = metrics.get_endpoint_stats()
    session_stats = metrics.get_session_stats()
    
    return {
        "storage": storage_stats,
        "system": {
            "cpu_percent": system_metrics.cpu_percent,
            "memory_percent": system_metrics.memory_percent,
            "memory_used_mb": system_metrics.memory_used_mb,
            "active_sessions": system_metrics.active_sessions,
            "requests_per_second": system_metrics.requests_per_second,
            "error_rate": system_metrics.error_rate
        },
        "endpoints": endpoint_stats,
        "sessions": session_stats
    }
