from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from app.analysis.exercises.base import ExerciseAnalyzer, ExerciseFrameResult
from app.analysis.pose import Pose, compute_common_angles, normalize_joints
from app.core.geometry import Point, distance
from app.core.models import FrameInput


def _avg(a: float | None, b: float | None) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


@dataclass
class SquatState:
    phase: str = "up"  # up -> down -> up
    rep_count: int = 0
    current_rep_min_knee: float | None = None
    current_rep_issues: set[str] = field(default_factory=set)
    rep_summaries: list[tuple[float, list[str]]] = field(default_factory=list)  # (score, issues)
    knee_window: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    bottom_reached: bool = False


class SquatAnalyzer(ExerciseAnalyzer):
    """
    Conservative squat heuristics (2D camera view):
    - Depth proxy via knee flexion angle (hip-knee-ankle). Standing is ~180°.
      A "parallel-ish" squat often reaches ~90°–110° depending on definition/model.
      We count a clear down->up cycle even if it is shallow, then score/flag depth separately.
      This keeps the rep counter responsive with mobile camera sampling.
    - Excessive forward torso lean is flagged when torso-to-vertical angle > 55°.
      (Deep squats can lean; we keep this threshold conservative to avoid false positives.)
    - Knee valgus proxy (2D): knee drifting medially relative to ankle by > ~10% of hip width.
      This is view-dependent; we only flag when geometry is confident.
    """

    name = "squat"
    down_enter_threshold = 150.0
    up_lockout_threshold = 165.0
    minimum_bottom_threshold = 135.0

    def __init__(self) -> None:
        self.state = SquatState()

    def analyze(self, frame: FrameInput) -> ExerciseFrameResult:
        pose = Pose(normalize_joints(frame.joints))
        computed = compute_common_angles(pose)
        angles = dict(computed)
        if frame.angles:
            angles.update(frame.angles)

        raw_knee = _avg(angles.get("knee_l"), angles.get("knee_r"))
        torso = _avg(angles.get("torso_l_vs_vertical"), angles.get("torso_r_vs_vertical"))
        knee = self._stable_knee(raw_knee)
        phase_knee = raw_knee if raw_knee is not None else knee

        issues: list[str] = []

        # Form checks (frame-level)
        if torso is not None and torso > 55:
            issues.append("excessive_forward_lean")
            self.state.current_rep_issues.add("excessive_forward_lean")

        valgus = self._knee_valgus_issue(pose)
        if valgus is not None:
            issues.append(valgus)
            self.state.current_rep_issues.add(valgus)

        shallow = False
        if phase_knee is not None:
            if self.state.current_rep_min_knee is None:
                self.state.current_rep_min_knee = phase_knee
            else:
                self.state.current_rep_min_knee = min(self.state.current_rep_min_knee, phase_knee)
            if phase_knee <= self.minimum_bottom_threshold:
                self.state.bottom_reached = True

        # Depth is evaluated against the minimum knee angle reached during the descent.
        # Only warn while "down" if depth is still shallow so far.
        if self.state.phase == "down" and self.state.current_rep_min_knee is not None:
            if self.state.current_rep_min_knee > 125:
                shallow = True
                issues.append("shallow_depth")
                self.state.current_rep_issues.add("shallow_depth")

        # Rep state machine (angle-based)
        rep_inc = 0
        rep_score: float | None = None
        rep_issues: list[str] | None = None

        # "Down" when descent is clearly started (allows shallow reps; avoids counting fidgets)
        if self.state.phase == "up":
            if phase_knee is not None and phase_knee <= self.down_enter_threshold:
                self.state.phase = "down"
                self.state.current_rep_min_knee = phase_knee
                self.state.current_rep_issues = set(issues)
                self.state.bottom_reached = phase_knee <= self.minimum_bottom_threshold
        elif self.state.phase == "down":
            # Consider rep completed when returning close to standing.
            # Shallow reps still count, but they are scored and reported as shallow.
            if phase_knee is not None and phase_knee >= self.up_lockout_threshold:
                self.state.phase = "up"
                self.state.rep_count += 1
                rep_inc = 1
                if self.state.current_rep_min_knee is None or self.state.current_rep_min_knee > 125:
                    self.state.current_rep_issues.add("shallow_depth")
                rep_score = self._score_rep(self.state.current_rep_min_knee, list(self.state.current_rep_issues))
                rep_issues = sorted(self.state.current_rep_issues)
                self.state.rep_summaries.append((rep_score, rep_issues))
                self.state.current_rep_min_knee = None
                self.state.current_rep_issues = set()
                self.state.bottom_reached = False

        # Frame score is primarily a smooth proxy, not the rep score.
        score = 100.0
        if shallow:
            score -= 20.0
        if "excessive_forward_lean" in issues:
            score -= 15.0
        if valgus is not None:
            score -= 15.0
        score = max(0.0, min(100.0, score))

        dbg = {
            "knee_avg": knee if knee is not None else -1,
            "knee_raw": raw_knee if raw_knee is not None else -1,
            "torso_vs_vertical_avg": torso if torso is not None else -1,
            "phase": self.state.phase,
            "rep_count": self.state.rep_count,
            "bottom_reached": int(self.state.bottom_reached),
        }

        return ExerciseFrameResult(
            issues=issues,
            score=score,
            rep_increment=rep_inc,
            rep_score=rep_score,
            rep_issues=rep_issues,
            debug=dbg,
        )

    def _score_rep(self, min_knee: float | None, issues: list[str]) -> float:
        # Depth mapping: <=95 excellent; 95-110 good; 110-125 partial; >125 shallow.
        score = 100.0
        if min_knee is None:
            score -= 40.0
        else:
            if min_knee <= 95:
                score -= 0.0
            elif min_knee <= 110:
                score -= 8.0
            elif min_knee <= 125:
                score -= 20.0
            else:
                score -= 35.0

        # Penalize issues (conservative)
        for issue in issues:
            if issue == "excessive_forward_lean":
                score -= 10.0
            elif issue in ("knee_valgus_left", "knee_valgus_right"):
                score -= 10.0
        return max(0.0, min(100.0, score))

    def _knee_valgus_issue(self, pose: Pose) -> str | None:
        # Requires hip/knee/ankle on both sides. Uses hip width as scale.
        hip_l = pose.get_point("hip_l")
        hip_r = pose.get_point("hip_r")
        knee_l = pose.get_point("knee_l")
        knee_r = pose.get_point("knee_r")
        ankle_l = pose.get_point("ankle_l")
        ankle_r = pose.get_point("ankle_r")
        if None in (hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r):
            return None

        hip_width = distance(hip_l, hip_r)
        if hip_width <= 1e-6:
            return None

        # Determine medial direction in image by comparing hip x positions.
        # We assume left hip has smaller x than right hip (common). If not, still works by sign.
        left_is_left = hip_l.x < hip_r.x

        # For left leg, valgus: knee x is shifted towards midline relative to ankle.
        # Use threshold 10% of hip width to avoid flagging noise.
        th = 0.10 * hip_width

        if left_is_left:
            # midline is between hips; for left knee, moving right is medial.
            if (knee_l.x - ankle_l.x) > th:
                return "knee_valgus_left"
            # for right knee, moving left is medial.
            if (ankle_r.x - knee_r.x) > th:
                return "knee_valgus_right"
        else:
            # swapped
            if (ankle_l.x - knee_l.x) > th:
                return "knee_valgus_left"
            if (knee_r.x - ankle_r.x) > th:
                return "knee_valgus_right"

        return None

    def _stable_knee(self, knee: float | None) -> float | None:
        if knee is None:
            return None
        self.state.knee_window.append(float(knee))
        return float(median(self.state.knee_window))

    def summary(self) -> dict:
        return {
            "rep_count": self.state.rep_count,
            "rep_summaries": self.state.rep_summaries,
        }
