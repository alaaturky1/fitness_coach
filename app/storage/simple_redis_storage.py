from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Optional

import redis
from redis import Redis

from app.analysis.engine import new_session_engine
from app.core.config import get_settings
from app.core.models import Language, Level
from app.storage.engine_serializer import serialize_engine


@dataclass
class RedisSession:
    session_id: str
    engine_data: dict[str, Any]
    ended: bool = False
    created_at: float = field(default_factory=lambda: __import__('time').time())
    last_accessed: float = field(default_factory=lambda: __import__('time').time())


class SimpleRedisSessions:
    def __init__(self) -> None:
        self._redis: Optional[Redis] = None
        self._settings = get_settings()
        self._session_ttl = 3600  # 1 hour
        self._key_prefix = "fitness_session:"

    def get_redis(self) -> Redis:
        if self._redis is None:
            redis_url = self._settings.redis_url
            if not redis_url:
                raise RuntimeError("FITCOACH_REDIS_URL is required when Redis is enabled")
            self._redis = Redis.from_url(redis_url, decode_responses=False)
        return self._redis

    def _get_session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def create_session(self, language: Language, level: Level) -> RedisSession:
        engine = new_session_engine(language=language, level=level.value)
        engine_state = serialize_engine(engine)
        
        # Serialize engine state
        engine_data = {
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
        }
        
        session = RedisSession(
            session_id=engine.session_id,
            engine_data=engine_data
        )
        
        redis = self.get_redis()
        session_key = self._get_session_key(session.session_id)
        redis.setex(
            session_key, 
            self._session_ttl, 
            pickle.dumps(session)
        )
        
        return session

    def get(self, session_id: str) -> Optional[RedisSession]:
        redis = self.get_redis()
        session_key = self._get_session_key(session_id)
        
        data = redis.get(session_key)
        if data is None:
            return None
            
        try:
            session = pickle.loads(data)
            session.last_accessed = __import__('time').time()
            
            # Update TTL on access
            redis.expire(session_key, self._session_ttl)
            
            return session
        except (pickle.PickleError, AttributeError, TypeError):
            # Corrupted data, remove it
            redis.delete(session_key)
            return None

    def update_session(self, session: RedisSession) -> None:
        redis = self.get_redis()
        session_key = self._get_session_key(session.session_id)
        
        redis.setex(
            session_key,
            self._session_ttl,
            pickle.dumps(session)
        )

    def delete_session(self, session_id: str) -> bool:
        redis = self.get_redis()
        session_key = self._get_session_key(session_id)
        
        result = redis.delete(session_key)
        return result > 0

    def get_active_sessions_count(self) -> int:
        redis = self.get_redis()
        pattern = f"{self._key_prefix}*"
        keys = redis.keys(pattern)
        return len(keys)

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions (handled automatically by Redis TTL)"""
        # Redis handles TTL automatically, but we can force cleanup if needed
        return 0

    def close(self) -> None:
        if self._redis:
            self._redis.close()
            self._redis = None


# Global instance
simple_redis_sessions = SimpleRedisSessions()


def get_simple_redis_sessions() -> SimpleRedisSessions:
    return simple_redis_sessions
