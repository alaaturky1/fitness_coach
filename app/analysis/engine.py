from __future__ import annotations

from dataclasses import dataclass, field
import logging
from time import perf_counter
from uuid import uuid4

from app.analysis.exercises.base import ExerciseAnalyzer
from app.analysis.exercises.plank import PlankAnalyzer
from app.analysis.exercises.pushup import PushupAnalyzer
from app.analysis.exercises.squat import SquatAnalyzer
from app.analysis.pose import Pose, compute_common_angles, normalize_joints
from app.analysis.smoothing import EmaSmoother
from app.analysis.visibility import check_visibility
from app.core.models import AnalyzeFrameResponse, FrameInput, Language, RepSummary, SessionSummaryResponse
from app.feedback.generator import pick_feedback
from app.feedback.policy import FeedbackGate, issue_priority, sort_issues
from app.ml.online import LearningSignal, OnlineFrameLearner


EXERCISE_ALIASES = {
    "push-up": "pushup",
    "pushup": "pushup",
    "push_up": "pushup",
    "squat": "squat",
    "plank": "plank",
}


def _normalize_exercise(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    return EXERCISE_ALIASES.get(key, key)


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))


@dataclass
class RepRecord:
    rep_index: int
    score: float
    issues: list[str]


@dataclass
class CoachingEngine:
    session_id: str
    language: Language
    level: str

    analyzer: ExerciseAnalyzer | None = None
    exercise: str | None = None

    rep_records: list[RepRecord] = field(default_factory=list)
    issues_tally: dict[str, int] = field(default_factory=dict)

    # Robustness controls
    # We smooth only angles used for form feedback (e.g., torso lean).
    # Rep phase transitions rely on knee/elbow peaks and should not be delayed by filtering.
    angle_smoother: EmaSmoother = field(default_factory=lambda: EmaSmoother(alpha=0.35))
    min_joint_confidence: float = 0.5
    rep_cooldown_s: float = 0.75
    feedback_gate: FeedbackGate = field(default_factory=FeedbackGate)
    online_learner: OnlineFrameLearner = field(default_factory=OnlineFrameLearner)

    # Timing/analytics
    last_timestamp: float | None = None
    active_time_s: float = 0.0
    idle_time_s: float = 0.0
    last_rep_timestamp: float | None = None

    def analyze(self, frame: FrameInput) -> AnalyzeFrameResponse:
        start = perf_counter()
        log = logging.getLogger("fitness")

        ex = _normalize_exercise(frame.exercise) or self.exercise or self._detect_exercise(frame)
        if ex is None:
            learning = self._learn_from_frame(
                angles={},
                issues=["unknown_exercise"],
                score=0.0,
                avg_confidence=None,
            )
            resp = AnalyzeFrameResponse(
                feedback=pick_feedback(
                    self.language,
                    ["unknown_exercise"],
                    level=self.level,
                    exercise="unknown",
                    priority="high",
                    paused=True,
                ),
                score=0.0,
                issues=["unknown_exercise"],
                rep_count=len(self.rep_records),
                exercise="unknown",
                paused=True,
                speak=True,
                priority="high",
                lang=self.language,
                debug={"reason": "exercise_missing", "ml": learning.to_debug()},
            )
            return resp

        # Initialize or switch analyzer if needed.
        if self.analyzer is None or self.exercise != ex:
            self._set_exercise(ex)

        # Handle image-only frames by extracting pose
        if frame.joints is None and frame.image_b64 is not None:
            try:
                from app.analysis.pose_detector import detect_pose_from_image
                pose_result = detect_pose_from_image(frame.image_b64)
                
                if pose_result.error and pose_result.confidence < 0.3:
                    # Pose detection failed completely
                    issues = ["pose_detection_failed"]
                    self._accumulate_time(frame.timestamp, active=False)
                    elapsed_ms = (perf_counter() - start) * 1000.0
                    learning = self._learn_from_frame(
                        angles={},
                        issues=issues,
                        score=0.0,
                        avg_confidence=pose_result.confidence,
                    )
                    resp = AnalyzeFrameResponse(
                        feedback=pick_feedback(
                            self.language,
                            issues,
                            level=self.level,
                            exercise=ex,
                            score=0.0,
                            priority="high",
                            paused=True,
                        ),
                        score=0.0,
                        issues=issues,
                        rep_count=len(self.rep_records),
                        exercise=ex,
                        paused=True,
                        speak=self.feedback_gate.allow("pose_detection_failed", float(frame.timestamp), cooldown_s=5.0),
                        priority="high",
                        lang=self.language,
                        debug={
                            "pose_error": pose_result.error,
                            "pose_confidence": pose_result.confidence,
                            "elapsed_ms": elapsed_ms,
                            "ml": learning.to_debug(),
                        },
                    )
                    return resp
                
                # Use detected joints
                frame.joints = pose_result.joints
                
            except Exception as e:
                # Pose detection failed, use fallback
                issues = ["pose_detection_error"]
                self._accumulate_time(frame.timestamp, active=False)
                elapsed_ms = (perf_counter() - start) * 1000.0
                learning = self._learn_from_frame(
                    angles={},
                    issues=issues,
                    score=0.0,
                    avg_confidence=None,
                )
                resp = AnalyzeFrameResponse(
                    feedback=pick_feedback(
                        self.language,
                        issues,
                        level=self.level,
                        exercise=ex,
                        score=0.0,
                        priority="high",
                        paused=True,
                    ),
                    score=0.0,
                    issues=issues,
                    rep_count=len(self.rep_records),
                    exercise=ex,
                    paused=True,
                    speak=self.feedback_gate.allow("pose_detection_error", float(frame.timestamp), cooldown_s=5.0),
                    priority="high",
                    lang=self.language,
                    debug={
                        "pose_error": str(e),
                        "elapsed_ms": elapsed_ms,
                        "ml": learning.to_debug(),
                    },
                )
                return resp

        # Visibility/confidence checks
        joints_norm = normalize_joints(frame.joints)
        pose = Pose(joints_norm)
        vis = check_visibility(pose, ex, min_confidence=self.min_joint_confidence)
        if not vis.ok and (frame.angles is None or len(frame.angles) == 0):
            # If client did not provide angles, and pose tracking is unstable, pause analysis to avoid bad feedback.
            issues = ["visibility_low"]
            self._accumulate_time(frame.timestamp, active=False)
            elapsed_ms = (perf_counter() - start) * 1000.0
            feedback_issues = sort_issues(issues)
            learning = self._learn_from_frame(
                angles={},
                issues=feedback_issues,
                score=0.0,
                avg_confidence=vis.avg_confidence,
            )
            resp = AnalyzeFrameResponse(
                feedback=pick_feedback(
                    self.language,
                    feedback_issues,
                    level=self.level,
                    exercise=ex,
                    score=0.0,
                    priority="high",
                    paused=True,
                ),
                score=0.0,
                issues=issues,
                rep_count=len(self.rep_records),
                exercise=ex,
                paused=True,
                speak=self.feedback_gate.allow("visibility_low", float(frame.timestamp), cooldown_s=5.0),
                priority="high",
                lang=self.language,
                debug={
                    "missing_joints": vis.missing,
                    "low_confidence_joints": vis.low_confidence,
                    "avg_confidence": vis.avg_confidence,
                    "elapsed_ms": elapsed_ms,
                    "pose_extracted": frame.joints is not None,
                    "ml": learning.to_debug(),
                },
            )
            log.warning(
                "paused_visibility_low",
                extra={
                    "session_id": self.session_id,
                    "exercise": ex,
                    "rep_count": len(self.rep_records),
                    "issues": issues,
                    "elapsed_ms": elapsed_ms,
                    "speak": resp.speak,
                    "priority": resp.priority,
                },
            )
            return resp

        # Compute angles (confidence-aware), merge with client-provided angles, then smooth.
        computed = compute_common_angles(pose, min_confidence=self.min_joint_confidence)
        merged_angles: dict[str, float] = dict(computed)
        if frame.angles:
            merged_angles.update({k: float(v) for k, v in frame.angles.items()})

        smoothed_angles = dict(merged_angles)
        for k in ("torso_l_vs_vertical", "torso_r_vs_vertical"):
            if k in merged_angles:
                smoothed_angles[k] = self.angle_smoother.update(k, merged_angles[k])
        smooth_frame = FrameInput(
            exercise=frame.exercise,
            joints=frame.joints,
            angles=smoothed_angles,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            image_b64=frame.image_b64,
        )

        result = self.analyzer.analyze(smooth_frame)

        # Rep cooldown (prevents double counting due to jitter near lockout)
        rep_increment = result.rep_increment
        rep_score = result.rep_score
        rep_issues = result.rep_issues
        if rep_increment:
            if self.last_rep_timestamp is not None:
                dt = float(frame.timestamp) - float(self.last_rep_timestamp)
                if 0 <= dt < self.rep_cooldown_s:
                    rep_increment = 0
                    rep_score = None
                    rep_issues = None

        # Tally issues (frame-level)
        for issue in result.issues:
            self.issues_tally[issue] = self.issues_tally.get(issue, 0) + 1

        # Record rep if allowed
        if rep_increment and rep_score is not None and rep_issues is not None:
            self.last_rep_timestamp = float(frame.timestamp)
            idx = len(self.rep_records) + 1
            self.rep_records.append(RepRecord(rep_index=idx, score=float(rep_score), issues=list(rep_issues)))
            for issue in rep_issues:
                self.issues_tally[issue] = self.issues_tally.get(issue, 0) + 1

        # Analytics: treat "down" phase as active; otherwise idle.
        active = bool((result.debug or {}).get("phase") == "down")
        self._accumulate_time(frame.timestamp, active=active)

        sorted_issues = sort_issues(list(result.issues))
        elapsed_ms = (perf_counter() - start) * 1000.0
        learning = self._learn_from_frame(
            angles=smoothed_angles,
            issues=sorted_issues,
            score=float(result.score),
            avg_confidence=vis.avg_confidence,
        )
        learned_score = _clamp_score(float(result.score) + learning.score_adjustment)

        debug = dict(result.debug or {})
        debug.update(
            {
                "elapsed_ms": elapsed_ms,
                "visibility_ok": vis.ok,
                "avg_confidence": vis.avg_confidence,
                "ml": learning.to_debug(),
            }
        )

        # Speak trigger policy:
        # - on rep completion (once per rep)
        # - or on high-priority issues with cooldown (avoid spam)
        speak = False
        priority = "low"
        if rep_increment and rep_issues:
            speak = True
            priority = "medium" if not sorted_issues else issue_priority(sorted_issues[0])
        elif sorted_issues:
            priority = issue_priority(sorted_issues[0])
            if priority == "high":
                speak = self.feedback_gate.allow(sorted_issues[0], float(frame.timestamp), cooldown_s=3.0)
        if learning.calibrated and sorted_issues and not rep_increment:
            strongest_issue_confidence = max(learning.issue_confidence.values(), default=0.0)
            if strongest_issue_confidence < 0.18 and priority == "high":
                priority = "medium"
                speak = False
            elif learning.dominant_issue in sorted_issues and priority == "low":
                priority = "medium"
        feedback = pick_feedback(
            self.language,
            sorted_issues,
            level=self.level,
            exercise=ex,
            score=learned_score,
            priority=priority,
            paused=False,
        )
        resp = AnalyzeFrameResponse(
            feedback=feedback,
            score=learned_score,
            issues=sorted_issues,
            rep_count=len(self.rep_records),
            exercise=ex,
            paused=False,
            speak=speak,
            priority=priority,
            lang=self.language,
            debug=debug,
        )
        log.info(
            "analyze_frame",
            extra={
                "session_id": self.session_id,
                "exercise": ex,
                "rep_count": len(self.rep_records),
                "issues": sorted_issues,
                "elapsed_ms": elapsed_ms,
                "speak": speak,
                "priority": priority,
                "ml_frames_seen": learning.frames_seen,
                "ml_score_adjustment": learning.score_adjustment,
            },
        )
        return resp

    def _learn_from_frame(
        self,
        *,
        angles: dict[str, float],
        issues: list[str],
        score: float,
        avg_confidence: float | None,
    ) -> LearningSignal:
        return self.online_learner.learn(
            angles=angles,
            issues=issues,
            score=score,
            avg_confidence=avg_confidence,
        )

    def _accumulate_time(self, timestamp: float, *, active: bool) -> None:
        if self.last_timestamp is None:
            self.last_timestamp = float(timestamp)
            return
        dt = max(0.0, float(timestamp) - float(self.last_timestamp))
        # Prevent nonsense if client timestamps jump (ms vs s, pauses, etc.)
        if 0.0 < dt < 2.0:
            if active:
                self.active_time_s += dt
            else:
                self.idle_time_s += dt
        self.last_timestamp = float(timestamp)

    def summary(self) -> SessionSummaryResponse:
        reps = len(self.rep_records)
        scores = [r.score for r in self.rep_records]
        avg = (sum(scores) / reps) if reps else None
        best = max(scores) if scores else None
        worst = min(scores) if scores else None

        most_freq = None
        if self.issues_tally:
            # Exclude purely operational issues from "mistake" summary if present.
            filtered = {k: v for k, v in self.issues_tally.items() if k not in ("visibility_low", "unknown_exercise")}
            pool = filtered if filtered else self.issues_tally
            most_freq = max(pool.items(), key=lambda kv: kv[1])[0]

        return SessionSummaryResponse(
            session_id=self.session_id,
            exercise=self.exercise,
            reps=reps,
            avg_rep_score=avg,
            best_rep_score=best,
            worst_rep_score=worst,
            most_frequent_mistake=most_freq,
            active_time_s=float(self.active_time_s),
            idle_time_s=float(self.idle_time_s),
            rep_summaries=[RepSummary(rep_index=r.rep_index, score=r.score, issues=r.issues) for r in self.rep_records],
            issues_tally=dict(self.issues_tally),
        )

    def _make_analyzer(self, ex: str) -> ExerciseAnalyzer:
        if ex == "squat":
            return SquatAnalyzer()
        if ex == "pushup":
            return PushupAnalyzer()
        if ex == "plank":
            return PlankAnalyzer()
        return PlankAnalyzer()

    def _set_exercise(self, ex: str) -> None:
        initial = self.exercise is None
        changed = self.exercise is not None and self.exercise != ex
        self.exercise = ex
        self.analyzer = self._make_analyzer(ex)

        if initial or changed:
            self.rep_records.clear()
            self.issues_tally.clear()
            self.angle_smoother = EmaSmoother(alpha=0.35)
            self.feedback_gate = FeedbackGate()
            self.online_learner = OnlineFrameLearner()
            self.last_timestamp = None
            self.last_rep_timestamp = None
            self.active_time_s = 0.0
            self.idle_time_s = 0.0

    def _detect_exercise(self, frame: FrameInput) -> str | None:
        joints = frame.joints
        if joints is None:
            return None
        names = set(joints.keys()) if isinstance(joints, dict) else {j.name for j in joints}
        has_wrists = ("wrist_l" in names) or ("wrist_r" in names)
        has_knees = ("knee_l" in names) or ("knee_r" in names)
        if has_wrists:
            return "pushup"
        if has_knees:
            return "squat"
        return None


def new_session_engine(language: Language, level: str) -> CoachingEngine:
    return CoachingEngine(session_id=str(uuid4()), language=language, level=level)
