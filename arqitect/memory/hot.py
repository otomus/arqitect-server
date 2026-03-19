"""
Hot Memory — Redis-backed session context and conversation buffer.
Stores ephemeral data: current session info and recent conversation history.
"""

import json

from arqitect.types import RedisKey

SESSION_KEY = RedisKey.SESSION
CONVO_KEY = RedisKey.CONVERSATION
MAX_CONVO = 20


class HotMemory:
    def __init__(self, redis_client):
        self.r = redis_client

    # ── Key helpers ───────────────────────────────────────────────────────

    def _session_key(self, user_id: str = "") -> str:
        return f"synapse:session:{user_id}" if user_id else "synapse:session"

    def _convo_key(self, user_id: str = "") -> str:
        return f"synapse:conversation:{user_id}" if user_id else "synapse:conversation"

    # ── Session ──────────────────────────────────────────────────────────

    def set_session(self, data: dict, user_id: str = ""):
        """Store session context (location, timezone, country, etc.)."""
        key = self._session_key(user_id)
        self.r.hset(key, mapping={k: str(v) for k, v in data.items()})

    def get_session(self, user_id: str = "") -> dict:
        """Retrieve current session context."""
        key = self._session_key(user_id)
        raw = self.r.hgetall(key)
        return dict(raw) if raw else {}

    def update_session(self, updates: dict, user_id: str = ""):
        """Merge updates into current session."""
        key = self._session_key(user_id)
        self.r.hset(key, mapping={k: str(v) for k, v in updates.items()})

    # ── Conversation buffer ──────────────────────────────────────────────

    def add_message(self, role: str, content: str, user_id: str = ""):
        """Append a message to the conversation buffer."""
        key = self._convo_key(user_id)
        msg = json.dumps({"role": role, "content": content})
        self.r.rpush(key, msg)
        self.r.ltrim(key, -MAX_CONVO, -1)

    def clear_conversation(self, user_id: str = ""):
        """Clear the conversation buffer (e.g. after waking from dream state)."""
        key = self._convo_key(user_id)
        self.r.delete(key)

    def clear_all_conversations(self):
        """Clear all conversation buffers (global + per-user). Called on startup."""
        for key in self.r.scan_iter("synapse:conversation*"):
            self.r.delete(key)

    def get_conversation(self, limit: int = MAX_CONVO, user_id: str = "") -> list[dict]:
        """Retrieve the last N messages."""
        key = self._convo_key(user_id)
        raw = self.r.lrange(key, -limit, -1)
        msgs = []
        for item in raw:
            try:
                msgs.append(json.loads(item))
            except (json.JSONDecodeError, TypeError):
                pass
        return msgs
