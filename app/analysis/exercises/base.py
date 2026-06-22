from __future__ import annotations

from dataclasses import dataclass

from app.core.models import FrameInput


@dataclass
class ExerciseFrameResult:
    issues: list[str]
    score: float  # 0..100 per-frame
    rep_increment: int  # 0 or 1
    rep_score: float | None  # score captured when rep ends
    rep_issues: list[str] | None
    debug: dict[str, float | str | int] | None = None


class ExerciseAnalyzer:
    name: str

    def analyze(self, frame: FrameInput) -> ExerciseFrameResult:
        raise NotImplementedError

    def summary(self) -> dict:
        return {}

