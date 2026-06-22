from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil
from fastapi import Request


@dataclass
class RequestMetrics:
    """Individual request metrics"""
    method: str
    path: str
    status_code: int
    duration_ms: float
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class SessionMetrics:
    """Session-specific metrics"""
    session_id: str
    created_at: float
    frames_processed: int = 0
    total_duration_ms: float = 0.0
    avg_frame_time_ms: float = 0.0
    errors: int = 0
    last_activity: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    active_sessions: int
    requests_per_second: float
    error_rate: float


class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self, max_requests: int = 10000, max_sessions: int = 1000) -> None:
        self.max_requests = max_requests
        self.max_sessions = max_sessions
        
        # Request metrics
        self.requests: deque[RequestMetrics] = deque(maxlen=max_requests)
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.request_durations: Dict[str, List[float]] = defaultdict(list)
        
        # Session metrics
        self.sessions: Dict[str, SessionMetrics] = {}
        
        # System metrics history
        self.system_history: deque[SystemMetrics] = deque(maxlen=1000)
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recent_errors: deque[RequestMetrics] = deque(maxlen=100)
    
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Record a request metric"""
        timestamp = time.time()
        metric = RequestMetrics(
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            timestamp=timestamp,
            user_id=user_id,
            session_id=session_id
        )
        
        self.requests.append(metric)
        self.request_counts[f"{method} {path}"] += 1
        self.request_durations[f"{method} {path}"].append(duration_ms)
        
        # Track errors
        if status_code >= 400:
            self.error_counts[f"{method} {path}"] += 1
            self.recent_errors.append(metric)
        
        # Update session metrics
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.frames_processed += 1
            session.total_duration_ms += duration_ms
            session.avg_frame_time_ms = session.total_duration_ms / session.frames_processed
            session.last_activity = timestamp
            if status_code >= 400:
                session.errors += 1
    
    def register_session(self, session_id: str) -> None:
        """Register a new session"""
        if session_id not in self.sessions and len(self.sessions) < self.max_sessions:
            self.sessions[session_id] = SessionMetrics(
                session_id=session_id,
                created_at=time.time()
            )
    
    def remove_session(self, session_id: str) -> None:
        """Remove a session"""
        self.sessions.pop(session_id, None)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Calculate requests per second (last minute)
        now = time.time()
        recent_requests = [
            r for r in self.requests 
            if now - r.timestamp < 60
        ]
        requests_per_second = len(recent_requests) / 60.0
        
        # Calculate error rate (last minute)
        recent_errors = len([r for r in recent_requests if r.status_code >= 400])
        error_rate = recent_errors / max(len(recent_requests), 1) * 100
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            active_sessions=len(self.sessions),
            requests_per_second=requests_per_second,
            error_rate=error_rate
        )
        
        self.system_history.append(metrics)
        return metrics
    
    def get_endpoint_stats(self) -> Dict[str, Dict]:
        """Get statistics for each endpoint"""
        stats = {}
        for endpoint, durations in self.request_durations.items():
            if durations:
                stats[endpoint] = {
                    "count": self.request_counts[endpoint],
                    "avg_duration_ms": sum(durations) / len(durations),
                    "min_duration_ms": min(durations),
                    "max_duration_ms": max(durations),
                    "error_count": self.error_counts[endpoint],
                    "error_rate": self.error_counts[endpoint] / self.request_counts[endpoint] * 100
                }
        return stats
    
    def get_session_stats(self) -> Dict[str, Dict]:
        """Get statistics for active sessions"""
        stats = {}
        for session_id, session in self.sessions.items():
            stats[session_id] = {
                "created_at": session.created_at,
                "frames_processed": session.frames_processed,
                "total_duration_ms": session.total_duration_ms,
                "avg_frame_time_ms": session.avg_frame_time_ms,
                "errors": session.errors,
                "last_activity": session.last_activity,
                "session_age_s": time.time() - session.created_at
            }
        return stats
    
    def cleanup_old_sessions(self, max_age_hours: float = 24.0) -> int:
        """Clean up old inactive sessions"""
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_sessions = [
            session_id for session_id, session in self.sessions.items()
            if now - session.last_activity > max_age_seconds
        ]
        
        for session_id in old_sessions:
            self.remove_session(session_id)
        
        return len(old_sessions)


# Global metrics collector
metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return metrics


async def metrics_middleware(request: Request, call_next):
    """FastAPI middleware for collecting request metrics"""
    start_time = time.time()
    
    # Extract session ID from request
    session_id = None
    if request.method == "POST" and "session" in request.url.path:
        try:
            body = await request.json()
            session_id = body.get("session_id")
        except Exception:
            pass
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    
    # Record metrics
    metrics.record_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
        session_id=session_id
    )
    
    return response
