from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog
from structlog.stdlib import LoggerFactory

from app.core.config import get_settings


def configure_logging() -> None:
    """Configure basic logging for now"""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Configure specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("fitness").setLevel(log_level)

