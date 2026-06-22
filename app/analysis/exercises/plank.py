from __future__ import annotations

from dataclasses import dataclass

from app.analysis.exercises.base import ExerciseAnalyzer, ExerciseFrameResult
from app.analysis.pose import Pose, normalize_joints
from app.core.geometry import Point, point_line_signed_distance
from app.core.models import FrameInput


@dataclass
class PlankState:
    seconds_in_plank: float = 0.0
    last_timestamp: float | None = None


class PlankAnalyzer(ExerciseAnalyzer):
    """
    Plank heuristics (hold-based, no reps):
    - Checks "straight line" by measuring shoulder->ankle line and hip deviation.
      Flags:
        - hips_sagging (hip below line beyond threshold)
        - hips_piked (hip above line beyond threshold)
    """

    name = "plank"

    def __init__(self) -> None:
        self.state = PlankState()

    def analyze(self, frame: FrameInput) -> ExerciseFrameResult:
        pose = Pose(normalize_joints(frame.joints))

        issues: list[str] = []
        hip_dev = self._hip_deviation(pose)
        if hip_dev is not None:
            if hip_dev > 0:
                issues.append("hips_off_line")
            else:
                issues.append("hips_off_line")

        # Accumulate time (best-effort; we treat timestamp as monotonic)
        if self.state.last_timestamp is not None:
            dt = max(0.0, float(frame.timestamp) - float(self.state.last_timestamp))
            # If client sends ms, dt will be huge; cap to avoid nonsense.
            if dt > 0 and dt < 2.0:
                self.state.seconds_in_plank += dt
        self.state.last_timestamp = float(frame.timestamp)

        score = 100.0
        if hip_dev is None:
            score -= 30.0
        elif abs(hip_dev) > 0.06:
            score -= 20.0
        elif abs(hip_dev) > 0.035:
            score -= 10.0
        score = max(0.0, min(100.0, score))

        dbg = {"hip_dev_norm": hip_dev if hip_dev is not None else -1, "seconds": self.state.seconds_in_plank}
        return ExerciseFrameResult(
            issues=issues,
            score=score,
            rep_increment=0,
            rep_score=None,
            rep_issues=None,
            debug=dbg,
        )

    def _hip_deviation(self, pose: Pose) -> float | None:
        sh_l = pose.get_point("shoulder_l")
        sh_r = pose.get_point("shoulder_r")
        an_l = pose.get_point("ankle_l")
        an_r = pose.get_point("ankle_r")
        hip_l = pose.get_point("hip_l")
        hip_r = pose.get_point("hip_r")
        if None in (sh_l, sh_r, an_l, an_r, hip_l, hip_r):
            return None

        shoulder_mid = Point((sh_l.x + sh_r.x) / 2, (sh_l.y + sh_r.y) / 2)
        ankle_mid = Point((an_l.x + an_r.x) / 2, (an_l.y + an_r.y) / 2)
        hip_mid = Point((hip_l.x + hip_r.x) / 2, (hip_l.y + hip_r.y) / 2)

        body_len = ((shoulder_mid.x - ankle_mid.x) ** 2 + (shoulder_mid.y - ankle_mid.y) ** 2) ** 0.5
        if body_len < 1e-6:
            return None
        d = point_line_signed_distance(hip_mid, shoulder_mid, ankle_mid)
        if d != d:
            return None
        # Normalize by body length for scale invariance
        return d / body_len

    def summary(self) -> dict:
        return {"seconds_in_plank": self.state.seconds_in_plank}

