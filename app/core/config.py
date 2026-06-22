from __future__ import annotations

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    require_api_key: bool
    log_level: str
    redis_url: str | None
    use_redis: bool
    llm_enabled: bool
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    llm_timeout_s: float
    llm_temperature: float
    llm_max_tokens: int


def get_settings() -> Settings:
    api_key = os.getenv("FITCOACH_API_KEY")
    require_api_key = _get_bool("FITCOACH_REQUIRE_API_KEY", default=True)
    log_level = os.getenv("FITCOACH_LOG_LEVEL", "INFO").upper()
    redis_url = os.getenv("FITCOACH_REDIS_URL")
    use_redis = _get_bool("FITCOACH_USE_REDIS", default=bool(redis_url))
    llm_api_key = os.getenv("FITCOACH_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_enabled = _get_bool("FITCOACH_LLM_ENABLED", default=bool(llm_api_key))
    llm_base_url = os.getenv("FITCOACH_LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.getenv("FITCOACH_LLM_MODEL", "gpt-4o-mini")
    return Settings(
        api_key=api_key, 
        require_api_key=require_api_key, 
        log_level=log_level,
        redis_url=redis_url,
        use_redis=use_redis,
        llm_enabled=llm_enabled,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_timeout_s=_get_float("FITCOACH_LLM_TIMEOUT_S", 1.5),
        llm_temperature=_get_float("FITCOACH_LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_get_int("FITCOACH_LLM_MAX_TOKENS", 60),
    )
