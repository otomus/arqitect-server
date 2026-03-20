"""Tests for arqitect.memory.hot — HotMemory with Redis backend."""

import json

import fakeredis
import pytest

from arqitect.memory.hot import HotMemory, MAX_CONVO


@pytest.fixture
def hot():
    """Create a HotMemory backed by a fake Redis instance."""
    client = fakeredis.FakeRedis(decode_responses=True)
    return HotMemory(client)


# ── Session ──────────────────────────────────────────────────────────────────


class TestSession:
    def test_set_and_get_session(self, hot):
        hot.set_session({"timezone": "UTC", "city": "TLV"})
        session = hot.get_session()
        assert session["timezone"] == "UTC"
        assert session["city"] == "TLV"

    def test_get_session_empty(self, hot):
        assert hot.get_session() == {}

    def test_update_session_merges_fields(self, hot):
        hot.set_session({"timezone": "UTC", "city": "TLV"})
        hot.update_session({"city": "NYC", "country": "US"})
        session = hot.get_session()
        assert session["timezone"] == "UTC"
        assert session["city"] == "NYC"
        assert session["country"] == "US"

    def test_session_user_scoped(self, hot):
        hot.set_session({"timezone": "UTC"}, user_id="u1")
        hot.set_session({"timezone": "PST"}, user_id="u2")
        assert hot.get_session(user_id="u1")["timezone"] == "UTC"
        assert hot.get_session(user_id="u2")["timezone"] == "PST"

    def test_session_global_vs_user_isolated(self, hot):
        hot.set_session({"scope": "global"})
        hot.set_session({"scope": "user"}, user_id="u1")
        assert hot.get_session()["scope"] == "global"
        assert hot.get_session(user_id="u1")["scope"] == "user"

    def test_update_session_user_scoped(self, hot):
        hot.set_session({"a": "1"}, user_id="u1")
        hot.update_session({"b": "2"}, user_id="u1")
        session = hot.get_session(user_id="u1")
        assert session["a"] == "1"
        assert session["b"] == "2"

    def test_session_values_stored_as_strings(self, hot):
        hot.set_session({"count": 42, "flag": True})
        session = hot.get_session()
        assert session["count"] == "42"
        assert session["flag"] == "True"


# ── Conversation ─────────────────────────────────────────────────────────────


class TestConversation:
    def test_add_and_get_message(self, hot):
        hot.add_message("user", "hello")
        hot.add_message("assistant", "hi there")
        convo = hot.get_conversation()
        assert len(convo) == 2
        assert convo[0] == {"role": "user", "content": "hello"}
        assert convo[1] == {"role": "assistant", "content": "hi there"}

    def test_get_conversation_empty(self, hot):
        assert hot.get_conversation() == []

    def test_get_conversation_with_limit(self, hot):
        for i in range(10):
            hot.add_message("user", f"msg {i}")
        convo = hot.get_conversation(limit=3)
        assert len(convo) == 3
        # Should be the last 3 messages
        assert convo[0]["content"] == "msg 7"
        assert convo[2]["content"] == "msg 9"

    def test_conversation_buffer_trims_at_max(self, hot):
        for i in range(MAX_CONVO + 10):
            hot.add_message("user", f"msg {i}")
        convo = hot.get_conversation()
        assert len(convo) == MAX_CONVO
        # Oldest messages should have been trimmed
        assert convo[0]["content"] == "msg 10"

    def test_clear_conversation(self, hot):
        hot.add_message("user", "hello")
        hot.clear_conversation()
        assert hot.get_conversation() == []

    def test_clear_conversation_user_scoped(self, hot):
        hot.add_message("user", "global msg")
        hot.add_message("user", "user msg", user_id="u1")
        hot.clear_conversation(user_id="u1")
        assert hot.get_conversation(user_id="u1") == []
        assert len(hot.get_conversation()) == 1

    def test_clear_all_conversations(self, hot):
        hot.add_message("user", "global")
        hot.add_message("user", "u1 msg", user_id="u1")
        hot.add_message("user", "u2 msg", user_id="u2")
        hot.clear_all_conversations()
        assert hot.get_conversation() == []
        assert hot.get_conversation(user_id="u1") == []
        assert hot.get_conversation(user_id="u2") == []

    def test_conversation_user_isolation(self, hot):
        hot.add_message("user", "msg for u1", user_id="u1")
        hot.add_message("user", "msg for u2", user_id="u2")
        u1_convo = hot.get_conversation(user_id="u1")
        u2_convo = hot.get_conversation(user_id="u2")
        assert len(u1_convo) == 1
        assert u1_convo[0]["content"] == "msg for u1"
        assert len(u2_convo) == 1
        assert u2_convo[0]["content"] == "msg for u2"

    def test_conversation_global_vs_user_isolated(self, hot):
        hot.add_message("user", "global msg")
        hot.add_message("user", "user msg", user_id="u1")
        assert len(hot.get_conversation()) == 1
        assert len(hot.get_conversation(user_id="u1")) == 1
        assert hot.get_conversation()[0]["content"] == "global msg"
        assert hot.get_conversation(user_id="u1")[0]["content"] == "user msg"

    def test_malformed_json_in_conversation_skipped(self, hot):
        """Manually push invalid JSON; get_conversation should skip it."""
        key = hot._convo_key()
        hot.r.rpush(key, "not-valid-json")
        hot.add_message("user", "valid")
        convo = hot.get_conversation()
        assert len(convo) == 1
        assert convo[0]["content"] == "valid"

    def test_add_message_preserves_order(self, hot):
        hot.add_message("user", "first")
        hot.add_message("assistant", "second")
        hot.add_message("user", "third")
        convo = hot.get_conversation()
        contents = [m["content"] for m in convo]
        assert contents == ["first", "second", "third"]
