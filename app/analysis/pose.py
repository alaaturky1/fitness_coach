from __future__ import annotations

import math
from dataclasses import dataclass

from app.core.geometry import Point, angle_degrees, vector_angle_to_vertical_degrees
from app.core.models import Joint


@dataclass(frozen=True)
class Pose:
    joints: dict[str, Joint]

    def get_point(self, name: str, *, min_confidence: float | None = None) -> Point | None:
        j = self.joints.get(name)
        if j is None:
            return None
        if min_confidence is not None:
            c = j.confidence
            if c is not None and c < min_confidence:
                return None
        return Point(x=j.x, y=j.y)

    def get_confidence(self, name: str) -> float | None:
        j = self.joints.get(name)
        return None if j is None else j.confidence


def normalize_joints(joints: list[Joint] | dict[str, Joint] | None) -> dict[str, Joint]:
    if joints is None:
        return {}
    if isinstance(joints, dict):
        return joints
    out: dict[str, Joint] = {}
    for j in joints:
        out[j.name] = j
    return out


def _safe_angle(a: Point | None, b: Point | None, c: Point | None) -> float | None:
    if a is None or b is None or c is None:
        return None
    v = angle_degrees(a, b, c)
    if math.isnan(v):
        return None
    return v


def _safe_vertical_angle(p_from: Point | None, p_to: Point | None) -> float | None:
    if p_from is None or p_to is None:
        return None
    v = vector_angle_to_vertical_degrees(p_from, p_to)
    if math.isnan(v):
        return None
    return v


def compute_common_angles(pose: Pose, *, min_confidence: float | None = None) -> dict[str, float]:
    """
    Computes commonly-used angles in degrees.
    Convention: 'knee_l' is angle at left knee (hip_l - knee_l - ankle_l).
    """

    p = lambda name: pose.get_point(name, min_confidence=min_confidence)
    angles: dict[str, float] = {}

    knee_l = _safe_angle(p("hip_l"), p("knee_l"), p("ankle_l"))
    knee_r = _safe_angle(p("hip_r"), p("knee_r"), p("ankle_r"))
    hip_l = _safe_angle(p("shoulder_l"), p("hip_l"), p("knee_l"))
    hip_r = _safe_angle(p("shoulder_r"), p("hip_r"), p("knee_r"))
    elbow_l = _safe_angle(p("shoulder_l"), p("elbow_l"), p("wrist_l"))
    elbow_r = _safe_angle(p("shoulder_r"), p("elbow_r"), p("wrist_r"))

    if knee_l is not None:
        angles["knee_l"] = knee_l
    if knee_r is not None:
        angles["knee_r"] = knee_r
    if hip_l is not None:
        angles["hip_l"] = hip_l
    if hip_r is not None:
        angles["hip_r"] = hip_r
    if elbow_l is not None:
        angles["elbow_l"] = elbow_l
    if elbow_r is not None:
        angles["elbow_r"] = elbow_r

    torso_l = _safe_vertical_angle(p("hip_l"), p("shoulder_l"))
    torso_r = _safe_vertical_angle(p("hip_r"), p("shoulder_r"))
    if torso_l is not None:
        angles["torso_l_vs_vertical"] = torso_l
    if torso_r is not None:
        angles["torso_r_vs_vertical"] = torso_r

    return angles

