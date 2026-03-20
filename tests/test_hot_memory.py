"""Tests for arqitect.memory.hot — HotMemory with Redis backend."""

import json

import pytest
from dirty_equals import IsInstance
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from arqitect.memory.hot import HotMemory, MAX_CONVO


@pytest.fixture
def hot(test_redis):
    """Create a HotMemory backed by the shared fake Redis instance."""
    return HotMemory(test_redis)


# ── Session ──────────────────────────────────────────────────────────────────


@pytest.mark.timeout(10)
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


@pytest.mark.timeout(10)
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


# ── Property-based tests ────────────────────────────────────────────────────


@pytest.mark.timeout(10)
class TestConversationProperties:
    """Hypothesis-driven property tests for conversation trimming and retrieval."""

    @given(num_messages=st.integers(min_value=0, max_value=MAX_CONVO * 3))
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_conversation_never_exceeds_max(self, test_redis, num_messages):
        """No matter how many messages are added, buffer never exceeds MAX_CONVO."""
        # flushall resets state between hypothesis iterations
        test_redis.flushall()
        hot = HotMemory(test_redis)

        for i in range(num_messages):
            hot.add_message("user", f"msg {i}")

        convo = hot.get_conversation()
        assert len(convo) <= MAX_CONVO
        for msg in convo:
            assert msg == IsInstance[dict]

    @given(num_messages=st.integers(min_value=1, max_value=MAX_CONVO * 3))
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_trimming_preserves_most_recent(self, test_redis, num_messages):
        """After trimming, the most recent message is always the last one added."""
        test_redis.flushall()
        hot = HotMemory(test_redis)

        for i in range(num_messages):
            hot.add_message("user", f"msg {i}")

        convo = hot.get_conversation()
        assert len(convo) >= 1
        assert convo[-1]["content"] == f"msg {num_messages - 1}"

    @given(num_messages=st.integers(min_value=0, max_value=MAX_CONVO * 3))
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_stored_count_matches_expectation(self, test_redis, num_messages):
        """Stored message count equals min(num_messages, MAX_CONVO)."""
        test_redis.flushall()
        hot = HotMemory(test_redis)

        for i in range(num_messages):
            hot.add_message("user", f"msg {i}")

        convo = hot.get_conversation()
        assert len(convo) == min(num_messages, MAX_CONVO)


@pytest.mark.timeout(10)
class TestContextStorageProperties:
    """Hypothesis-driven property tests for session/context storage."""

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(
                whitelist_categories=("L", "N"),
            )),
            values=st.text(min_size=0, max_size=100),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_session_roundtrip_with_random_payloads(self, test_redis, data):
        """Any string-valued dict survives a set/get roundtrip."""
        test_redis.flushall()
        hot = HotMemory(test_redis)

        hot.set_session(data)
        result = hot.get_session()
        assert result == data

    @given(
        initial=st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet=st.characters(
                whitelist_categories=("L", "N"),
            )),
            values=st.text(min_size=0, max_size=50),
            min_size=1,
            max_size=5,
        ),
        updates=st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet=st.characters(
                whitelist_categories=("L", "N"),
            )),
            values=st.text(min_size=0, max_size=50),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_update_session_merges_correctly(self, test_redis, initial, updates):
        """update_session merges keys: all initial keys remain, update keys overwrite."""
        test_redis.flushall()
        hot = HotMemory(test_redis)

        hot.set_session(initial)
        hot.update_session(updates)
        result = hot.get_session()

        # Every key from initial should still exist
        for key in initial:
            assert key in result

        # Every key from updates should reflect the updated value
        for key, value in updates.items():
            assert result[key] == value

    @given(
        content=st.text(min_size=1, max_size=200),
        role=st.sampled_from(["user", "assistant", "system"]),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_message_content_survives_roundtrip(self, test_redis, content, role):
        """Any message content survives JSON serialization roundtrip in Redis."""
        test_redis.flushall()
        hot = HotMemory(test_redis)

        hot.add_message(role, content)
        convo = hot.get_conversation()
        assert len(convo) == 1
        assert convo[0] == {"role": role, "content": content}
