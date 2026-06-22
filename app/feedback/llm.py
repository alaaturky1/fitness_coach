from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import Settings, get_settings
from app.core.models import Language


@dataclass(frozen=True)
class FeedbackContext:
    language: Language
    level: str
    exercise: str | None
    issues: tuple[str, ...]
    fallback: str
    score: float | None = None
    priority: str = "low"
    paused: bool = False


class LLMFeedbackClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[tuple[object, ...], str] = {}
        self._lock = Lock()

    @property
    def available(self) -> bool:
        return bool(self.settings.llm_enabled and self.settings.llm_model)

    def generate(self, context: FeedbackContext) -> str:
        if not self.available:
            return context.fallback

        cache_key = self._cache_key(context)
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            text = self._call_llm(context)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            logging.getLogger("fitness").warning("llm_feedback_failed", extra={"error": str(exc)})
            return context.fallback

        cleaned = self._clean_text(text)
        if not cleaned:
            return context.fallback

        with self._lock:
            self._cache[cache_key] = cleaned
        return cleaned

    def _cache_key(self, context: FeedbackContext) -> tuple[object, ...]:
        score_bucket = None if context.score is None else int(context.score // 10) * 10
        return (
            context.language.value,
            context.level,
            context.exercise,
            context.issues,
            score_bucket,
            context.priority,
            context.paused,
        )

    def _call_llm(self, context: FeedbackContext) -> str:
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": self._system_prompt(context.language)},
                {"role": "user", "content": self._user_prompt(context)},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.settings.llm_timeout_s) as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]

    def _system_prompt(self, language: Language) -> str:
        dialect = "natural Egyptian Arabic that sounds good when read aloud by text-to-speech" if language == Language.ar else "plain English"
        arabic_rules = ""
        if language == Language.ar:
            arabic_rules = (
                "Use Arabic script only, except unavoidable product names. "
                "Avoid English words, issue-code wording, tashkeel, emoji, markdown, abbreviations, and long punctuation. "
                "Prefer short spoken coaching phrases with clear Egyptian wording. "
            )
        return (
            "You are a real-time fitness coach. "
            f"Reply in {dialect}. "
            "Return exactly one short sentence, no markdown, no emojis. "
            "Keep it direct, encouraging, and actionable. "
            f"{arabic_rules}"
            "Do not mention issue codes, scores, or that you are an AI. "
            "Do not give medical advice."
        )

    def _user_prompt(self, context: FeedbackContext) -> str:
        return json.dumps(
            {
                "exercise": context.exercise or "unknown",
                "athlete_level": context.level,
                "issues": list(context.issues) or ["good_form"],
                "score": context.score,
                "priority": context.priority,
                "paused": context.paused,
                "base_cue": context.fallback,
                "instruction": "Rewrite the base cue as one concise coaching line.",
            },
            ensure_ascii=False,
        )

    def _clean_text(self, text: str) -> str:
        cleaned = " ".join(text.strip().strip("\"'`").split())
        if not cleaned:
            return ""
        if len(cleaned) > 160:
            cleaned = cleaned[:157].rstrip() + "..."
        return cleaned


@lru_cache(maxsize=1)
def get_llm_feedback_client() -> LLMFeedbackClient:
    return LLMFeedbackClient(get_settings())


def reset_llm_feedback_client() -> None:
    get_llm_feedback_client.cache_clear()
