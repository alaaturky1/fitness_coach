from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return sqrt(self.variance)

    def to_state(self) -> dict[str, float | int]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> RunningStats:
        return cls(
            count=int(state.get("count", 0)),
            mean=float(state.get("mean", 0.0)),
            m2=float(state.get("m2", 0.0)),
        )


@dataclass
class LearningSignal:
    frames_seen: int
    calibrated: bool
    score_adjustment: float
    issue_confidence: dict[str, float]
    dominant_issue: str | None
    tracking_confidence: float | None

    def to_debug(self) -> dict[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "calibrated": self.calibrated,
            "score_adjustment": round(self.score_adjustment, 3),
            "issue_confidence": {k: round(v, 3) for k, v in self.issue_confidence.items()},
            "dominant_issue": self.dominant_issue,
            "tracking_confidence": None
            if self.tracking_confidence is None
            else round(self.tracking_confidence, 3),
        }


@dataclass
class OnlineFrameLearner:
    """Small per-session online model updated once for every analyzed frame."""

    calibration_frames: int = 20
    issue_alpha: float = 0.08
    score_alpha: float = 0.08
    confidence_alpha: float = 0.08
    frames_seen: int = 0
    issue_ema: dict[str, float] = field(default_factory=dict)
    angle_stats: dict[str, RunningStats] = field(default_factory=dict)
    score_ema: float | None = None
    confidence_ema: float | None = None

    def learn(
        self,
        *,
        angles: dict[str, float],
        issues: list[str],
        score: float,
        avg_confidence: float | None,
    ) -> LearningSignal:
        self.frames_seen += 1

        for key, value in angles.items():
            self.angle_stats.setdefault(key, RunningStats()).update(float(value))

        self.score_ema = self._ema(self.score_ema, float(score), self.score_alpha)
        if avg_confidence is not None:
            self.confidence_ema = self._ema(
                self.confidence_ema,
                float(avg_confidence),
                self.confidence_alpha,
            )

        current_issues = set(issues)
        for issue in set(self.issue_ema) | current_issues:
            target = 1.0 if issue in current_issues else 0.0
            self.issue_ema[issue] = self._ema(self.issue_ema.get(issue), target, self.issue_alpha)

        self._trim_issue_memory()
        calibrated = self.frames_seen >= self.calibration_frames
        issue_confidence = {issue: self.issue_ema.get(issue, 0.0) for issue in issues}
        dominant_issue = self._dominant_issue()
        score_adjustment = self._score_adjustment(issues, calibrated)

        return LearningSignal(
            frames_seen=self.frames_seen,
            calibrated=calibrated,
            score_adjustment=score_adjustment,
            issue_confidence=issue_confidence,
            dominant_issue=dominant_issue,
            tracking_confidence=self.confidence_ema,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "calibration_frames": self.calibration_frames,
            "issue_alpha": self.issue_alpha,
            "score_alpha": self.score_alpha,
            "confidence_alpha": self.confidence_alpha,
            "frames_seen": self.frames_seen,
            "issue_ema": dict(self.issue_ema),
            "angle_stats": {key: stats.to_state() for key, stats in self.angle_stats.items()},
            "score_ema": self.score_ema,
            "confidence_ema": self.confidence_ema,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> OnlineFrameLearner:
        if not state:
            return cls()
        learner = cls(
            calibration_frames=int(state.get("calibration_frames", 20)),
            issue_alpha=float(state.get("issue_alpha", 0.08)),
            score_alpha=float(state.get("score_alpha", 0.08)),
            confidence_alpha=float(state.get("confidence_alpha", 0.08)),
            frames_seen=int(state.get("frames_seen", 0)),
            issue_ema={str(k): float(v) for k, v in state.get("issue_ema", {}).items()},
            score_ema=state.get("score_ema"),
            confidence_ema=state.get("confidence_ema"),
        )
        learner.angle_stats = {
            str(key): RunningStats.from_state(value)
            for key, value in state.get("angle_stats", {}).items()
        }
        return learner

    def _score_adjustment(self, issues: list[str], calibrated: bool) -> float:
        if not calibrated:
            return 0.0

        if not issues:
            if self.score_ema is not None and self.score_ema >= 85.0:
                return 1.5
            return 0.0

        strongest_issue = max(self.issue_ema.get(issue, 0.0) for issue in issues)
        if strongest_issue < 0.18 and (self.confidence_ema is None or self.confidence_ema >= 0.55):
            return 3.0
        if strongest_issue > 0.55:
            return -_clamp((strongest_issue - 0.55) * 18.0, 1.0, 8.0)
        return 0.0

    def _dominant_issue(self) -> str | None:
        if not self.issue_ema:
            return None
        issue, confidence = max(self.issue_ema.items(), key=lambda item: item[1])
        if confidence < 0.35:
            return None
        return issue

    def _trim_issue_memory(self) -> None:
        stale = [issue for issue, value in self.issue_ema.items() if value < 0.01]
        for issue in stale:
            del self.issue_ema[issue]

    @staticmethod
    def _ema(previous: float | None, value: float, alpha: float) -> float:
        if previous is None:
            return value
        return previous + alpha * (value - previous)

