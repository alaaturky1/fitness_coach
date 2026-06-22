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

SIDE_CHAINS: dict[str, dict[str, set[str]]] = {
    "squat": {
        "left": {"shoulder_l", "hip_l", "knee_l", "ankle_l"},
        "right": {"shoulder_r", "hip_r", "knee_r", "ankle_r"},
    },
    "pushup": {
        "left": {"shoulder_l", "elbow_l", "wrist_l", "hip_l", "ankle_l"},
        "right": {"shoulder_r", "elbow_r", "wrist_r", "hip_r", "ankle_r"},
    },
    "plank": {
        "left": {"shoulder_l", "hip_l", "ankle_l"},
        "right": {"shoulder_r", "hip_r", "ankle_r"},
    },
}


@dataclass(frozen=True)
class VisibilityResult:
    ok: bool
    missing: list[str]
    low_confidence: list[str]
    avg_confidence: float | None
    present: list[str]
    usable_sides: list[str]
    required_present_ratio: float


def check_visibility(
    pose: Pose,
    exercise: str,
    *,
    min_required_present_ratio: float = 0.85,
    min_confidence: float = 0.5,
) -> VisibilityResult:
    required = REQUIRED_JOINTS.get(exercise, set())
    if not required:
        return VisibilityResult(
            ok=True,
            missing=[],
            low_confidence=[],
            avg_confidence=None,
            present=[],
            usable_sides=[],
            required_present_ratio=1.0,
        )

    missing: list[str] = []
    low: list[str] = []
    present: list[str] = []
    confs: list[float] = []

    for name in sorted(required):
        j = pose.joints.get(name)
        if j is None:
            missing.append(name)
            continue
        present.append(name)
        if j.confidence is not None:
            confs.append(float(j.confidence))
            if j.confidence < min_confidence:
                low.append(name)

    present_count = len(required) - len(missing)
    present_ratio = present_count / max(1, len(required))
    ok_present = present_ratio >= min_required_present_ratio
    ok_conf = len(low) == 0
    avg = (sum(confs) / len(confs)) if confs else None

    usable_sides = _usable_sides(pose, exercise, min_confidence=min_confidence)
    ok = bool(usable_sides) if exercise in SIDE_CHAINS else bool(ok_present and ok_conf)

    return VisibilityResult(
        ok=ok,
        missing=missing,
        low_confidence=low,
        avg_confidence=avg,
        present=sorted(set(required) - set(missing)),
        usable_sides=usable_sides,
        required_present_ratio=present_ratio,
    )


def _usable_sides(pose: Pose, exercise: str, *, min_confidence: float) -> list[str]:
    chains = SIDE_CHAINS.get(exercise, {})
    usable: list[str] = []

    for side, joints in chains.items():
        if all(_joint_usable(pose, name, min_confidence=min_confidence) for name in joints):
            usable.append(side)

    return usable


def _joint_usable(pose: Pose, name: str, *, min_confidence: float) -> bool:
    joint = pose.joints.get(name)
    if joint is None:
        return False
    if joint.confidence is None:
        return True
    return float(joint.confidence) >= min_confidence
