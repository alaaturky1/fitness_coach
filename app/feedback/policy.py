from __future__ import annotations

from dataclasses import dataclass, field


CRITICAL_ISSUES: set[str] = {
    "visibility_low",
    "hips_sagging",
    "pose_detection_failed",
    "pose_detection_error",
}

HIGH_ISSUES: set[str] = {
    "excessive_forward_lean",
    "knee_valgus_left",
    "knee_valgus_right",
}

MEDIUM_ISSUES: set[str] = {
    "shallow_depth",
    "hips_off_line",
}


def issue_priority(issue: str) -> str:
    if issue in CRITICAL_ISSUES:
        return "high"
    if issue in HIGH_ISSUES:
        return "high"
    if issue in MEDIUM_ISSUES:
        return "medium"
    if issue == "unknown_exercise":
        return "high"
    return "low"


def sort_issues(issues: list[str]) -> list[str]:
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda i: (order.get(issue_priority(i), 2), i))


@dataclass
class FeedbackGate:
    """
    Prevents spamming the same spoken feedback.
    """

    last_spoken_at: dict[str, float] = field(default_factory=dict)  # key -> timestamp

    def allow(self, key: str, now_s: float, cooldown_s: float) -> bool:
        last = self.last_spoken_at.get(key)
        if last is None:
            self.last_spoken_at[key] = now_s
            return True
        if now_s - last >= cooldown_s:
            self.last_spoken_at[key] = now_s
            return True
        return False
