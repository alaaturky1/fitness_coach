from __future__ import annotations

from dataclasses import dataclass

from app.analysis.pose import Pose


REQUIRED_JOINTS: dict[str, set[str]] = {
    # For knee angle + torso inclination + basic alignment checks.
    "squat": {"hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "shoulder_l", "shoulder_r"},
    # For elbow angle + hip-line check.
    "pushup": {"shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r", "hip_l", "hip_r", "ankle_l", "ankle_r"},
    # For hip-line check.
    "plank": {"shoulder_l", "shoulder_r", "hip_l", "hip_r", "ankle_l", "ankle_r"},
}


@dataclass(frozen=True)
class VisibilityResult:
    ok: bool
    missing: list[str]
    low_confidence: list[str]
    avg_confidence: float | None


def check_visibility(
    pose: Pose,
    exercise: str,
    *,
    min_required_present_ratio: float = 0.85,
    min_confidence: float = 0.5,
) -> VisibilityResult:
    required = REQUIRED_JOINTS.get(exercise, set())
    if not required:
        return VisibilityResult(ok=True, missing=[], low_confidence=[], avg_confidence=None)

    missing: list[str] = []
    low: list[str] = []
    confs: list[float] = []

    for name in sorted(required):
        j = pose.joints.get(name)
        if j is None:
            missing.append(name)
            continue
        if j.confidence is not None:
            confs.append(float(j.confidence))
            if j.confidence < min_confidence:
                low.append(name)

    present = len(required) - len(missing)
    ok_present = (present / max(1, len(required))) >= min_required_present_ratio
    ok_conf = len(low) == 0
    avg = (sum(confs) / len(confs)) if confs else None

    return VisibilityResult(ok=bool(ok_present and ok_conf), missing=missing, low_confidence=low, avg_confidence=avg)

