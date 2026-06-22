from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from app.analysis.exercises.base import ExerciseAnalyzer, ExerciseFrameResult
from app.analysis.pose import Pose, compute_common_angles, normalize_joints
from app.core.geometry import Point, point_line_signed_distance
from app.core.models import FrameInput


def _avg(a: float | None, b: float | None) -> float | None:
    vals = [v for v in (a, b) if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


@dataclass
class PushupState:
    phase: str = "up"  # up -> down -> up
    rep_count: int = 0
    current_rep_min_elbow: float | None = None
    current_rep_issues: set[str] = field(default_factory=set)
    rep_summaries: list[tuple[float, list[str]]] = field(default_factory=list)
    elbow_window: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    bottom_reached: bool = False


class PushupAnalyzer(ExerciseAnalyzer):
    """
    Conservative push-up heuristics:
    - Depth proxy via elbow flexion angle (shoulder-elbow-wrist). Straight is ~180°.
      A solid bottom position often yields elbow angle ~70°–110° depending on form/model.
      We count "bottom reached" when <= 120°, and "good depth" when <= 105°.
    - Sagging hips (loss of plank line) proxy using signed distance of hip midpoint to
      line through shoulder midpoint -> ankle midpoint; threshold scaled to body length.
    """

    name = "pushup"
    down_enter_threshold = 160.0
    up_lockout_threshold = 165.0
    minimum_bottom_threshold = 150.0

    def __init__(self) -> None:
        self.state = PushupState()

    def analyze(self, frame: FrameInput) -> ExerciseFrameResult:
        pose = Pose(normalize_joints(frame.joints))
        computed = compute_common_angles(pose)
        angles = dict(computed)
        if frame.angles:
            angles.update(frame.angles)

        raw_elbow = _avg(angles.get("elbow_l"), angles.get("elbow_r"))
        elbow = self._stable_elbow(raw_elbow)
        phase_elbow = raw_elbow if raw_elbow is not None else elbow
        issues: list[str] = []

        hip_sag = self._hip_sag(pose)
        if hip_sag:
            issues.append("hips_sagging")
            self.state.current_rep_issues.add("hips_sagging")

        shallow = False
        if phase_elbow is not None:
            if self.state.current_rep_min_elbow is None:
                self.state.current_rep_min_elbow = phase_elbow
            else:
                self.state.current_rep_min_elbow = min(self.state.current_rep_min_elbow, phase_elbow)
            if phase_elbow <= self.minimum_bottom_threshold:
                self.state.bottom_reached = True

        if self.state.phase == "down" and self.state.current_rep_min_elbow is not None:
            if self.state.current_rep_min_elbow > 135:
                shallow = True
                issues.append("shallow_depth")
                self.state.current_rep_issues.add("shallow_depth")

        rep_inc = 0
        rep_score: float | None = None
        rep_issues: list[str] | None = None

        if self.state.phase == "up":
            if phase_elbow is not None and phase_elbow <= self.down_enter_threshold:
                self.state.phase = "down"
                self.state.current_rep_min_elbow = phase_elbow
                self.state.current_rep_issues = set(issues)
                self.state.bottom_reached = phase_elbow <= self.minimum_bottom_threshold
        elif self.state.phase == "down":
            if phase_elbow is not None and phase_elbow >= self.up_lockout_threshold and self.state.bottom_reached:
                self.state.phase = "up"
                self.state.rep_count += 1
                rep_inc = 1
                if self.state.current_rep_min_elbow is not None and self.state.current_rep_min_elbow > 135:
                    self.state.current_rep_issues.add("shallow_depth")
                rep_score = self._score_rep(self.state.current_rep_min_elbow, list(self.state.current_rep_issues))
                rep_issues = sorted(self.state.current_rep_issues)
                self.state.rep_summaries.append((rep_score, rep_issues))
                self.state.current_rep_min_elbow = None
                self.state.current_rep_issues = set()
                self.state.bottom_reached = False
            elif phase_elbow is not None and phase_elbow >= self.up_lockout_threshold:
                # Reset descent if athlete returned to top too early.
                self.state.phase = "up"
                self.state.current_rep_min_elbow = None
                self.state.current_rep_issues = set()
                self.state.bottom_reached = False

        score = 100.0
        if shallow:
            score -= 20.0
        if hip_sag:
            score -= 15.0
        score = max(0.0, min(100.0, score))

        dbg = {
            "elbow_avg": elbow if elbow is not None else -1,
            "elbow_raw": raw_elbow if raw_elbow is not None else -1,
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

    def _hip_sag(self, pose: Pose) -> bool:
        sh_l = pose.get_point("shoulder_l")
        sh_r = pose.get_point("shoulder_r")
        an_l = pose.get_point("ankle_l")
        an_r = pose.get_point("ankle_r")
        hip_l = pose.get_point("hip_l")
        hip_r = pose.get_point("hip_r")
        if None in (sh_l, sh_r, an_l, an_r, hip_l, hip_r):
            return False

        shoulder_mid = Point((sh_l.x + sh_r.x) / 2, (sh_l.y + sh_r.y) / 2)
        ankle_mid = Point((an_l.x + an_r.x) / 2, (an_l.y + an_r.y) / 2)
        hip_mid = Point((hip_l.x + hip_r.x) / 2, (hip_l.y + hip_r.y) / 2)

        # Scale by shoulder-ankle distance to be resolution/scale invariant.
        body_len = ((shoulder_mid.x - ankle_mid.x) ** 2 + (shoulder_mid.y - ankle_mid.y) ** 2) ** 0.5
        if body_len < 1e-6:
            return False

        d = point_line_signed_distance(hip_mid, shoulder_mid, ankle_mid)
        if d != d:  # NaN
            return False

        # Threshold: 6% of body length. Conservative; flags only clear deviations.
        return abs(d) > 0.06 * body_len

    def _score_rep(self, min_elbow: float | None, issues: list[str]) -> float:
        score = 100.0
        if min_elbow is None:
            score -= 40.0
        else:
            if min_elbow <= 105:
                score -= 0.0
            elif min_elbow <= 120:
                score -= 10.0
            elif min_elbow <= 135:
                score -= 22.0
            else:
                score -= 35.0

        for issue in issues:
            if issue == "hips_sagging":
                score -= 12.0
        return max(0.0, min(100.0, score))

    def _stable_elbow(self, elbow: float | None) -> float | None:
        if elbow is None:
            return None
        self.state.elbow_window.append(float(elbow))
        return float(median(self.state.elbow_window))

    def summary(self) -> dict:
        return {
            "rep_count": self.state.rep_count,
            "rep_summaries": self.state.rep_summaries,
        }

