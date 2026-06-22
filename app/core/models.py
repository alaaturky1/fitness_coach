from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Language(str, Enum):
    en = "en"
    ar = "ar"


class Level(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class Joint(BaseModel):
    """
    Pose joint in a consistent coordinate system per stream.
    x,y can be pixels or normalized; analysis uses scale-invariant ratios where possible.
    """

    name: str
    x: float
    y: float
    z: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


AnglesDict = dict[str, float]


class FrameInput(BaseModel):
    exercise: str | None = None
    joints: list[Joint] | dict[str, Joint] | None = None
    angles: AnglesDict | None = None
    timestamp: float = Field(..., description="Client timestamp; treated as monotonic within session")
    frame_id: int | None = None
    image_b64: str | None = Field(
        default=None,
        description="Optional fallback. Not analyzed server-side in this version.",
    )


class StartSessionRequest(BaseModel):
    language: Language = Language.en
    level: Level = Level.beginner


class StartSessionResponse(BaseModel):
    session_id: str
    ws_url: str


class AnalyzeFrameRequest(BaseModel):
    session_id: str
    frame: FrameInput


class AnalyzeFrameResponse(BaseModel):
    feedback: str
    score: float = Field(..., ge=0.0, le=100.0)
    issues: list[str]
    rep_count: int
    exercise: str
    paused: bool = False
    # Voice-ready contract
    speak: bool = False
    priority: str = Field(default="low", description="low|medium|high")
    lang: Language = Language.en
    debug: dict[str, Any] | None = None


class EndSessionRequest(BaseModel):
    session_id: str


class RepSummary(BaseModel):
    rep_index: int
    score: float
    issues: list[str]


class SessionSummaryResponse(BaseModel):
    session_id: str
    exercise: str | None
    reps: int
    avg_rep_score: float | None
    best_rep_score: float | None
    worst_rep_score: float | None
    most_frequent_mistake: str | None
    active_time_s: float
    idle_time_s: float
    rep_summaries: list[RepSummary]
    issues_tally: dict[str, int]

