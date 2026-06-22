from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from app.analysis.engine import CoachingEngine
from app.core.config import get_settings
from app.core.models import Language, Level
from app.storage.inmemory import InMemorySessions, Session
from app.storage.simple_redis_storage import SimpleRedisSessions, RedisSession, get_simple_redis_sessions
from app.storage.engine_serializer import serialize_engine, deserialize_engine


logger = logging.getLogger("fitness")


@dataclass
class HybridSession:
    """Hybrid session that works with both Redis and in-memory"""
    session_id: str
    engine: CoachingEngine
    ended: bool = False
    use_redis: bool = True


class HybridSessions:
    """Hybrid storage system that uses Redis as primary and in-memory as fallback"""
    
    def __init__(self, use_redis: bool = True) -> None:
        self.use_redis = use_redis
        self._memory_sessions = InMemorySessions()
        self._redis_sessions = None
    
    def _get_redis(self) -> Optional[SimpleRedisSessions]:
        if not self.use_redis:
            return None
        if self._redis_sessions is None:
            self._redis_sessions = get_simple_redis_sessions()
        return self._redis_sessions
    
    def create_session(self, language: Language, level: Level) -> HybridSession:
        if self.use_redis:
            try:
                redis = self._get_redis()
                if redis:
                    redis_session = redis.create_session(language, level)
                    # Reconstruct engine from stored data
                    from app.storage.engine_serializer import deserialize_engine
                    
                    # Create engine from stored state
                    engine_state = serialize_engine(CoachingEngine(
                        session_id=redis_session.session_id,
                        language=language,
                        level=level.value
                    ))
                    engine_state.session_id = redis_session.session_id
                    engine_state.language = language.value
                    engine_state.level = level.value
                    engine_state.engine_data = redis_session.engine_data
                    
                    engine = deserialize_engine(engine_state)
                    
                    return HybridSession(
                        session_id=redis_session.session_id,
                        engine=engine,
                        ended=redis_session.ended,
                        use_redis=True
                    )
            except Exception as e:
                logger.warning("redis_create_failed_fallback_to_memory", extra={"error": str(e)})
                self.use_redis = False
        
        # Fallback to in-memory
        memory_session = self._memory_sessions.create_session(language, level)
        return HybridSession(
            session_id=memory_session.session_id,
            engine=memory_session.engine,
            ended=memory_session.ended,
            use_redis=False
        )
    
    def get(self, session_id: str) -> Optional[HybridSession]:
        if self.use_redis:
            try:
                redis = self._get_redis()
                if redis:
                    redis_session = redis.get(session_id)
                    if redis_session:
                        # Reconstruct engine
                        from app.storage.engine_serializer import deserialize_engine
                        
                        engine_state = serialize_engine(CoachingEngine(
                            session_id=redis_session.session_id,
                            language=Language(redis_session.engine_data.get('language', 'en')),
                            level=redis_session.engine_data.get('level', 'beginner')
                        ))
                        engine_state.session_id = redis_session.session_id
                        engine_state.language = redis_session.engine_data.get('language', 'en')
                        engine_state.level = redis_session.engine_data.get('level', 'beginner')
                        engine_state.exercise = redis_session.engine_data.get('exercise')
                        engine_state.rep_records = redis_session.engine_data.get('rep_records', [])
                        engine_state.issues_tally = redis_session.engine_data.get('issues_tally', {})
                        engine_state.last_timestamp = redis_session.engine_data.get('last_timestamp')
                        engine_state.active_time_s = redis_session.engine_data.get('active_time_s', 0.0)
                        engine_state.idle_time_s = redis_session.engine_data.get('idle_time_s', 0.0)
                        engine_state.last_rep_timestamp = redis_session.engine_data.get('last_rep_timestamp')
                        engine_state.min_joint_confidence = redis_session.engine_data.get('min_joint_confidence', 0.5)
                        engine_state.rep_cooldown_s = redis_session.engine_data.get('rep_cooldown_s', 0.75)
                        engine_state.angle_smoother_state = redis_session.engine_data.get('angle_smoother_state', {})
                        engine_state.online_learner_state = redis_session.engine_data.get('online_learner_state', {})
                        engine_state.engine_blob = redis_session.engine_data.get('engine_blob')
                        
                        engine = deserialize_engine(engine_state)
                        
                        return HybridSession(
                            session_id=redis_session.session_id,
                            engine=engine,
                            ended=redis_session.ended,
                            use_redis=True
                        )
            except Exception as e:
                logger.warning("redis_get_failed_trying_memory", extra={"error": str(e)})
        
        # Fallback to in-memory
        memory_session = self._memory_sessions.get(session_id)
        if memory_session:
            return HybridSession(
                session_id=memory_session.session_id,
                engine=memory_session.engine,
                ended=memory_session.ended,
                use_redis=False
            )
        
        return None
    
    def update_session(self, session: HybridSession) -> None:
        if session.use_redis:
            try:
                redis = self._get_redis()
                if redis:
                    # Serialize engine state
                    engine_state = serialize_engine(session.engine)
                    redis_session = RedisSession(
                        session_id=session.session_id,
                        engine_data={
                            'session_id': engine_state.session_id,
                            'language': engine_state.language,
                            'level': engine_state.level,
                            'exercise': engine_state.exercise,
                            'rep_records': engine_state.rep_records,
                            'issues_tally': engine_state.issues_tally,
                            'last_timestamp': engine_state.last_timestamp,
                            'active_time_s': engine_state.active_time_s,
                            'idle_time_s': engine_state.idle_time_s,
                            'last_rep_timestamp': engine_state.last_rep_timestamp,
                            'min_joint_confidence': engine_state.min_joint_confidence,
                            'rep_cooldown_s': engine_state.rep_cooldown_s,
                            'angle_smoother_state': engine_state.angle_smoother_state,
                            'online_learner_state': engine_state.online_learner_state,
                            'engine_blob': engine_state.engine_blob,
                        },
                        ended=session.ended
                    )
                    redis.update_session(redis_session)
                    return
            except Exception as e:
                logger.warning("redis_update_failed", extra={"error": str(e)})
                session.use_redis = False
        
        # Update in-memory
        memory_session = self._memory_sessions.get(session.session_id)
        if memory_session:
            memory_session.ended = session.ended
            memory_session.engine = session.engine
        else:
            with self._memory_sessions._lock:
                self._memory_sessions._sessions[session.session_id] = Session(
                    session_id=session.session_id,
                    engine=session.engine,
                    ended=session.ended,
                )
    
    def delete_session(self, session_id: str) -> bool:
        if self.use_redis:
            try:
                redis = self._get_redis()
                if redis:
                    deleted = redis.delete_session(session_id)
                    if deleted:
                        return True
            except Exception as e:
                logger.warning("redis_delete_failed", extra={"error": str(e)})
        
        # Fallback to in-memory
        memory_session = self._memory_sessions.get(session_id)
        if memory_session:
            del self._memory_sessions._sessions[session_id]
            return True
        
        return False
    
    def get_stats(self) -> dict[str, int]:
        """Get storage statistics"""
        stats = {
            'memory_sessions': len(self._memory_sessions._sessions),
            'redis_sessions': 0,
            'total_sessions': 0
        }
        
        if self.use_redis:
            try:
                redis = self._get_redis()
                if redis:
                    stats['redis_sessions'] = redis.get_active_sessions_count()
            except Exception:
                pass
        
        stats['total_sessions'] = max(stats['memory_sessions'], stats['redis_sessions'])
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self._redis_sessions:
            self._redis_sessions.close()


# Global hybrid instance
hybrid_sessions = HybridSessions(use_redis=get_settings().use_redis)


def get_hybrid_sessions() -> HybridSessions:
    return hybrid_sessions
