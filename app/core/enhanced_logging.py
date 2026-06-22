from __future__ import annotations

import logging
import sys
import json
from pathlib import Path
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from app.core.config import get_settings


def configure_logging() -> None:
    """Configure structured logging with structlog"""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_level == "INFO" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "fitness_coach.log"),
        ],
    )
    
    # Configure specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("fitness").setLevel(log_level)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("slowapi").setLevel(logging.WARNING)


class StructuredLogger:
    """Structured logger for fitness coaching events"""
    
    def __init__(self, name: str = "fitness") -> None:
        self.logger = structlog.get_logger(name)
    
    def session_started(self, session_id: str, language: str, level: str, **kwargs) -> None:
        """Log session start"""
        self.logger.info(
            "session_started",
            session_id=session_id,
            language=language,
            level=level,
            **kwargs
        )
    
    def session_ended(self, session_id: str, duration: float, reps: int, **kwargs) -> None:
        """Log session end"""
        self.logger.info(
            "session_ended",
            session_id=session_id,
            duration_s=duration,
            reps=reps,
            **kwargs
        )
    
    def frame_analyzed(self, session_id: str, exercise: str, score: float, issues: list, **kwargs) -> None:
        """Log frame analysis"""
        self.logger.info(
            "frame_analyzed",
            session_id=session_id,
            exercise=exercise,
            score=score,
            issues=issues,
            issues_count=len(issues),
            **kwargs
        )
    
    def error(self, event: str, error: Exception, **kwargs) -> None:
        """Log error with context"""
        self.logger.error(
            event,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs
        )
    
    def performance(self, operation: str, duration_ms: float, **kwargs) -> None:
        """Log performance metrics"""
        self.logger.info(
            "performance",
            operation=operation,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def rate_limit_hit(self, identifier: str, endpoint: str, **kwargs) -> None:
        """Log rate limiting events"""
        self.logger.warning(
            "rate_limit_hit",
            identifier=identifier,
            endpoint=endpoint,
            **kwargs
        )
    
    def system_metrics(self, cpu_percent: float, memory_percent: float, active_sessions: int, **kwargs) -> None:
        """Log system metrics"""
        self.logger.info(
            "system_metrics",
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_sessions=active_sessions,
            **kwargs
        )


# Global structured logger instance
structured_logger = StructuredLogger()


def get_logger(name: str = "fitness") -> StructuredLogger:
    """Get structured logger instance"""
    return StructuredLogger(name) if name != "fitness" else structured_logger
