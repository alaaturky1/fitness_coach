from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from typing import Any

from app.analysis.engine import CoachingEngine, RepRecord
from app.analysis.smoothing import EmaSmoother
from app.feedback.policy import FeedbackGate
from app.core.models import Language
from app.ml.online import OnlineFrameLearner


@dataclass
class EngineState:
    """Serializable state for CoachingEngine"""
    session_id: str
    language: str
    level: str
    exercise: str | None = None
    
    # Rep tracking
    rep_records: list[dict[str, Any]] = field(default_factory=list)
    issues_tally: dict[str, int] = field(default_factory=dict)
    
    # Timing
    last_timestamp: float | None = None
    active_time_s: float = 0.0
    idle_time_s: float = 0.0
    last_rep_timestamp: float | None = None
    
    # Configuration
    min_joint_confidence: float = 0.5
    rep_cooldown_s: float = 0.75
    
    # Smoother state (simplified)
    angle_smoother_state: dict[str, float] = field(default_factory=dict)
    online_learner_state: dict[str, Any] = field(default_factory=dict)
    engine_blob: bytes | None = None


def serialize_engine(engine: CoachingEngine) -> EngineState:
    """Serialize CoachingEngine to EngineState"""
    try:
        engine_blob = pickle.dumps(engine)
    except (pickle.PickleError, TypeError, AttributeError):
        engine_blob = None

    return EngineState(
        session_id=engine.session_id,
        language=engine.language.value,
        level=engine.level,
        exercise=engine.exercise,
        rep_records=[
            {
                'rep_index': r.rep_index,
                'score': r.score,
                'issues': r.issues
            } for r in engine.rep_records
        ],
        issues_tally=engine.issues_tally,
        last_timestamp=engine.last_timestamp,
        active_time_s=engine.active_time_s,
        idle_time_s=engine.idle_time_s,
        last_rep_timestamp=engine.last_rep_timestamp,
        min_joint_confidence=engine.min_joint_confidence,
        rep_cooldown_s=engine.rep_cooldown_s,
        angle_smoother_state=engine.angle_smoother.values.copy(),
        online_learner_state=engine.online_learner.to_state(),
        engine_blob=engine_blob,
    )


def deserialize_engine(state: EngineState) -> CoachingEngine:
    """Deserialize EngineState back to CoachingEngine"""
    from app.analysis.engine import new_session_engine

    if state.engine_blob:
        try:
            engine = pickle.loads(state.engine_blob)
            if isinstance(engine, CoachingEngine):
                return engine
        except (pickle.PickleError, AttributeError, TypeError, EOFError):
            pass
    
    # Create new engine
    engine = new_session_engine(
        language=Language(state.language),
        level=state.level
    )
    
    # Restore state
    engine.session_id = state.session_id
    engine.exercise = state.exercise
    engine.rep_records = [
        RepRecord(
            rep_index=r['rep_index'],
            score=r['score'],
            issues=r['issues']
        ) for r in state.rep_records
    ]
    engine.issues_tally = state.issues_tally
    engine.last_timestamp = state.last_timestamp
    engine.active_time_s = state.active_time_s
    engine.idle_time_s = state.idle_time_s
    engine.last_rep_timestamp = state.last_rep_timestamp
    engine.min_joint_confidence = state.min_joint_confidence
    engine.rep_cooldown_s = state.rep_cooldown_s
    
    # Restore smoother state
    if state.angle_smoother_state:
        for key, value in state.angle_smoother_state.items():
            engine.angle_smoother.values[key] = value
    engine.online_learner = OnlineFrameLearner.from_state(state.online_learner_state)
    
    # Restore analyzer if exercise is set
    if state.exercise:
        engine.analyzer = engine._make_analyzer(state.exercise)
    
    return engine
