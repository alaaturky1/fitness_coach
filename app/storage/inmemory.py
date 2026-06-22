from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from app.analysis.engine import CoachingEngine, new_session_engine
from app.core.models import Language, Level


@dataclass
class Session:
    session_id: str
    engine: CoachingEngine
    ended: bool = False


class InMemorySessions:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, Session] = {}

    def create_session(self, language: Language, level: Level) -> Session:
        engine = new_session_engine(language=language, level=level.value)
        session = Session(session_id=engine.session_id, engine=engine)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)


sessions = InMemorySessions()

