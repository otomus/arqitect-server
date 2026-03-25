"""Tests for arqitect.memory.cold — ColdMemory with SQLite backend."""

import itertools
import json
import time

import pytest
from dirty_equals import IsInstance, IsPositiveInt, IsStr
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from arqitect.memory.cold import ColdMemory

# Counter for generating unique names in hypothesis tests that share a fixture
_counter = itertools.count()

# Apply timeout to every test in this module instead of per-class decorators
pytestmark = pytest.mark.timeout(10)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def cold(tmp_memory_dir):
    """Create a ColdMemory backed by the tmp_memory_dir fixture from conftest."""
    return ColdMemory()


# ── Strategies for hypothesis ───────────────────────────────────────────────

# Keys/values that are safe for SQLite text columns (non-empty, no NUL bytes)
safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters=("\x00",)),
    min_size=1,
    max_size=100,
)

safe_name = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters=("_", "-")),
    min_size=1,
    max_size=50,
)


# ── Facts ────────────────────────────────────────────────────────────────────


class TestFacts:
    def test_set_and_get_fact(self, cold):
        cold.set_fact("env", "timezone", "UTC")
        assert cold.get_fact("env", "timezone") == "UTC"

    def test_get_fact_missing_returns_none(self, cold):
        assert cold.get_fact("env", "nonexistent") is None

    def test_set_fact_updates_existing(self, cold):
        cold.set_fact("env", "tz", "UTC")
        cold.set_fact("env", "tz", "US/Eastern")
        assert cold.get_fact("env", "tz") == "US/Eastern"

    def test_get_facts_returns_all_in_category(self, cold):
        cold.set_fact("prefs", "color", "blue")
        cold.set_fact("prefs", "font", "mono")
        cold.set_fact("other", "key", "val")
        facts = cold.get_facts("prefs")
        assert facts == {"color": "blue", "font": "mono"}

    def test_get_facts_empty_category(self, cold):
        assert cold.get_facts("empty") == {}

    def test_set_fact_with_confidence(self, cold):
        cold.set_fact("env", "city", "TLV", confidence=0.8)
        assert cold.get_fact("env", "city") == "TLV"

    def test_set_user_fact_and_get(self, cold):
        cold.set_user_fact("u1", "language", "Hebrew")
        facts = cold.get_user_facts("u1")
        assert facts == {"language": "Hebrew"}

    def test_user_facts_isolated_between_users(self, cold):
        cold.set_user_fact("u1", "language", "Hebrew")
        cold.set_user_fact("u2", "language", "English")
        assert cold.get_user_facts("u1") == {"language": "Hebrew"}
        assert cold.get_user_facts("u2") == {"language": "English"}

    @given(category=safe_name, key=safe_name, value=safe_text)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_roundtrip_random_facts(self, cold, category, key, value):
        """Any category/key/value triple should survive a set/get roundtrip."""
        cold.set_fact(category, key, value)
        assert cold.get_fact(category, key) == value

    @given(category=safe_name, key=safe_name, v1=safe_text, v2=safe_text)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_upsert_overwrites_with_random_values(self, cold, category, key, v1, v2):
        """The last value written for a (category, key) pair wins."""
        cold.set_fact(category, key, v1)
        cold.set_fact(category, key, v2)
        assert cold.get_fact(category, key) == v2


# ── Nerve Registry ───────────────────────────────────────────────────────────


class TestNerveRegistry:
    def test_register_and_get_nerve(self, cold):
        cold.register_nerve("weather", "fetch weather data")
        info = cold.get_nerve_info("weather")
        assert info is not None
        assert info["name"] == "weather"
        assert info["description"] == "fetch weather data"
        assert info["origin"] == "local"

    def test_get_nerve_info_missing(self, cold):
        assert cold.get_nerve_info("nonexistent") is None

    def test_register_nerve_with_community_origin(self, cold):
        cold.register_nerve("translate", "translate text", origin="community")
        info = cold.get_nerve_info("translate")
        assert info["origin"] == "community"

    def test_record_nerve_invocation_success(self, cold):
        cold.register_nerve("calc", "calculator")
        cold.record_nerve_invocation("calc", success=True)
        cold.record_nerve_invocation("calc", success=True)
        cold.record_nerve_invocation("calc", success=False)
        info = cold.get_nerve_info("calc")
        assert info["total_invocations"] == 3
        assert info["successes"] == 2
        assert info["failures"] == 1

    def test_get_last_invoked_at_none_before_invocation(self, cold):
        cold.register_nerve("calc", "calculator")
        assert cold.get_last_invoked_at("calc") is None

    def test_get_last_invoked_at_set_after_invocation(self, cold):
        cold.register_nerve("calc", "calculator")
        cold.record_nerve_invocation("calc", success=True)
        assert cold.get_last_invoked_at("calc") is not None

    def test_list_nerves(self, cold):
        cold.register_nerve("a", "alpha")
        cold.register_nerve("b", "beta")
        listing = cold.list_nerves()
        assert listing == {"a": "alpha", "b": "beta"}

    def test_delete_nerve(self, cold):
        cold.register_nerve("tmp", "temporary")
        cold.record_qualification("nerve", "tmp", True, 1.0, 1, 5, 5)
        cold.delete_nerve("tmp")
        assert cold.get_nerve_info("tmp") is None
        assert cold.get_qualification("nerve", "tmp") is None

    def test_register_nerve_rich_local(self, cold):
        cold.register_nerve_rich(
            "summarize", "summarize text",
            system_prompt="You are a summarizer.",
            examples_json='[{"input":"hi","output":"hello"}]',
            role="tool",
        )
        meta = cold.get_nerve_metadata("summarize")
        assert meta["system_prompt"] == "You are a summarizer."
        assert meta["examples"] == [{"input": "hi", "output": "hello"}]
        assert meta["role"] == "tool"
        assert meta["origin"] == "local"

    def test_register_nerve_rich_community(self, cold):
        cold.register_nerve_rich(
            "translate", "translate text",
            system_prompt="ignored for community",
            role="tool",
            origin="community",
        )
        meta = cold.get_nerve_metadata("translate")
        assert meta["origin"] == "community"
        # Community nerves don't store system_prompt via register_nerve_rich
        assert meta["system_prompt"] == ""

    def test_get_nerve_metadata_missing_returns_defaults(self, cold):
        meta = cold.get_nerve_metadata("nonexistent")
        assert meta["description"] == ""
        assert meta["examples"] == []
        assert meta["origin"] == "local"

    def test_get_nerve_metadata_bad_examples_json(self, cold):
        """Malformed examples JSON should be handled gracefully."""
        cold.register_nerve("broken", "broken nerve")
        # Manually corrupt the examples column
        cold.conn.execute(
            "UPDATE nerve_registry SET examples='not-json' WHERE name='broken'"
        )
        cold.conn.commit()
        meta = cold.get_nerve_metadata("broken")
        assert meta["examples"] == []

    def test_get_nerve_origin_local(self, cold):
        cold.register_nerve("local_nerve", "desc")
        assert cold.get_nerve_origin("local_nerve") == "local"

    def test_get_nerve_origin_community(self, cold):
        cold.register_nerve("comm_nerve", "desc", origin="community")
        assert cold.get_nerve_origin("comm_nerve") == "community"

    def test_get_nerve_origin_missing(self, cold):
        assert cold.get_nerve_origin("missing") == "local"

    def test_is_community_nerve(self, cold):
        cold.register_nerve("comm", "desc", origin="community")
        cold.register_nerve("loc", "desc", origin="local")
        assert cold.is_community_nerve("comm") is True
        assert cold.is_community_nerve("loc") is False
        assert cold.is_community_nerve("missing") is False

    def test_update_nerve_description(self, cold):
        cold.register_nerve_rich("n", "old desc", system_prompt="keep me")
        cold.update_nerve_description("n", "new desc")
        meta = cold.get_nerve_metadata("n")
        assert meta["description"] == "new desc"
        assert meta["system_prompt"] == "keep me"

    @given(name=safe_name, description=safe_text)
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_register_nerve_roundtrip_random(self, cold, name, description):
        """Any name/description pair should survive registration and retrieval."""
        cold.register_nerve(name, description)
        info = cold.get_nerve_info(name)
        assert info is not None
        assert info["name"] == name
        assert info["description"] == description
        assert info["origin"] == "local"
        assert info["total_invocations"] == 0

    def test_register_nerve_upsert_preserves_invocation_stats(self, cold):
        """Re-registering a nerve should not reset invocation counters."""
        cold.register_nerve("calc", "calculator v1")
        cold.record_nerve_invocation("calc", success=True)
        cold.register_nerve("calc", "calculator v2")
        info = cold.get_nerve_info("calc")
        assert info["description"] == "calculator v2"
        assert info["total_invocations"] == 1


# ── Senses ───────────────────────────────────────────────────────────────────


class TestSenses:
    def test_register_sense_and_check(self, cold):
        cold.register_sense("sight", "visual perception")
        assert cold.is_sense("sight") is True

    def test_is_sense_false_for_nerve(self, cold):
        cold.register_nerve("calc", "calculator")
        assert cold.is_sense("calc") is False

    def test_is_sense_false_for_missing(self, cold):
        assert cold.is_sense("nonexistent") is False

    def test_list_senses(self, cold):
        cold.register_sense("sight", "visual perception")
        cold.register_sense("hearing", "audio perception")
        cold.register_nerve("calc", "calculator")
        senses = cold.list_senses()
        assert "sight" in senses
        assert "hearing" in senses
        assert "calc" not in senses

    def test_sense_always_local_origin(self, cold):
        cold.register_sense("touch", "haptic perception")
        assert cold.get_nerve_origin("touch") == "local"


# ── Test Bank ────────────────────────────────────────────────────────────────


class TestTestBank:
    def test_set_and_get_test_bank(self, cold):
        tests = [{"input": "2+2", "expected": "4"}]
        cold.set_test_bank("calc", tests)
        assert cold.get_test_bank("calc") == tests

    def test_get_test_bank_empty(self, cold):
        assert cold.get_test_bank("nonexistent") == []

    def test_get_test_bank_bad_json(self, cold):
        cold.set_fact("test_bank", "broken", "not-json-array")
        assert cold.get_test_bank("broken") == []


# ── Tool Stats ───────────────────────────────────────────────────────────────


class TestToolStats:
    def test_record_tool_call(self, cold):
        cold.record_tool_call("scrape", success=True)
        cold.record_tool_call("scrape", success=True)
        cold.record_tool_call("scrape", success=False)
        row = cold.conn.execute(
            "SELECT total_calls, successes, failures FROM tool_stats WHERE name='scrape'"
        ).fetchone()
        assert row["total_calls"] == 3
        assert row["successes"] == 2
        assert row["failures"] == 1

    @given(
        successes=st.integers(min_value=0, max_value=50),
        failures=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_tool_stats_counts_are_consistent(self, cold, successes, failures):
        """Total calls should always equal successes + failures."""
        assume(successes + failures > 0)
        # Use a unique name per example to avoid accumulation across hypothesis runs
        name = f"tool_{next(_counter)}"
        for _ in range(successes):
            cold.record_tool_call(name, success=True)
        for _ in range(failures):
            cold.record_tool_call(name, success=False)
        row = cold.conn.execute(
            "SELECT total_calls, successes, failures FROM tool_stats WHERE name=?",
            (name,),
        ).fetchone()
        assert row["total_calls"] == successes + failures
        assert row["successes"] == successes
        assert row["failures"] == failures


# ── Nerve Tools ──────────────────────────────────────────────────────────────


class TestNerveTools:
    def test_add_and_get_nerve_tools(self, cold):
        cold.add_nerve_tool("weather", "http_get")
        cold.add_nerve_tool("weather", "json_parse")
        tools = cold.get_nerve_tools("weather")
        assert set(tools) == {"http_get", "json_parse"}

    def test_add_nerve_tool_increments_use_count(self, cold):
        cold.add_nerve_tool("weather", "http_get")
        cold.add_nerve_tool("weather", "http_get")
        cold.add_nerve_tool("weather", "http_get")
        with_counts = cold.get_nerve_tools_with_counts("weather")
        assert len(with_counts) == 1
        assert with_counts[0]["tool"] == "http_get"
        assert with_counts[0]["use_count"] == 3

    def test_get_nerve_tools_ordered_by_use_count(self, cold):
        cold.add_nerve_tool("n", "rare")
        cold.add_nerve_tool("n", "common")
        cold.add_nerve_tool("n", "common")
        cold.add_nerve_tool("n", "common")
        tools = cold.get_nerve_tools("n")
        assert tools[0] == "common"

    def test_get_nerve_tools_empty(self, cold):
        assert cold.get_nerve_tools("nonexistent") == []

    def test_get_nerve_tools_with_counts_empty(self, cold):
        assert cold.get_nerve_tools_with_counts("nonexistent") == []


# ── Qualifications ───────────────────────────────────────────────────────────


class TestQualifications:
    def test_record_and_get_qualification(self, cold):
        cold.record_qualification("nerve", "calc", True, 0.95, 2, 10, 9, '{"notes":"good"}')
        q = cold.get_qualification("nerve", "calc")
        assert q is not None
        assert q["qualified"] is True
        assert q["score"] == 0.95
        assert q["iterations"] == 2
        assert q["test_count"] == 10
        assert q["pass_count"] == 9
        assert q["details"] == {"notes": "good"}

    def test_get_qualification_missing(self, cold):
        assert cold.get_qualification("nerve", "nope") is None

    def test_is_qualified_true(self, cold):
        cold.record_qualification("nerve", "calc", True, 1.0, 1, 5, 5)
        assert cold.is_qualified("nerve", "calc") is True

    def test_is_qualified_false(self, cold):
        cold.record_qualification("nerve", "calc", False, 0.3, 1, 5, 1)
        assert cold.is_qualified("nerve", "calc") is False

    def test_is_qualified_missing(self, cold):
        assert cold.is_qualified("nerve", "nope") is False

    def test_was_qualified_true_even_if_failed(self, cold):
        cold.record_qualification("nerve", "calc", False, 0.3, 1, 5, 1)
        assert cold.was_qualified("nerve", "calc") is True

    def test_was_qualified_false_when_never_tested(self, cold):
        assert cold.was_qualified("nerve", "nope") is False

    def test_list_qualifications(self, cold):
        cold.record_qualification("nerve", "a", True, 1.0, 1, 5, 5)
        cold.record_qualification("nerve", "b", False, 0.4, 1, 5, 2)
        listing = cold.list_qualifications()
        assert len(listing) == 2
        names = {q["subject_name"] for q in listing}
        assert names == {"a", "b"}

    def test_qualification_details_bad_json(self, cold):
        cold.record_qualification("nerve", "broken", True, 1.0, 1, 5, 5, "not-json")
        q = cold.get_qualification("nerve", "broken")
        assert q["details"] == {}

    def test_record_qualification_upsert(self, cold):
        cold.record_qualification("nerve", "calc", False, 0.3, 1, 5, 1)
        cold.record_qualification("nerve", "calc", True, 0.95, 2, 10, 9)
        q = cold.get_qualification("nerve", "calc")
        assert q["qualified"] is True
        assert q["score"] == 0.95

    def test_qualification_timestamp_is_set(self, cold):
        """Qualification records should have a timestamp."""
        cold.record_qualification("nerve", "ts_check", True, 1.0, 1, 5, 5)
        q = cold.get_qualification("nerve", "ts_check")
        assert q["timestamp"] == IsStr()


# ── Personality ──────────────────────────────────────────────────────────────


class TestPersonality:
    def test_append_and_get_signals(self, cold):
        cold.append_personality_signal({"tone": "friendly"})
        cold.append_personality_signal({"tone": "formal"})
        signals = cold.get_personality_signals()
        assert len(signals) == 2
        assert signals[0]["tone"] == "friendly"
        assert signals[1]["tone"] == "formal"

    def test_flush_personality_signals(self, cold):
        cold.append_personality_signal({"tone": "friendly"})
        cold.flush_personality_signals()
        assert cold.get_personality_signals() == []

    def test_get_signals_skips_bad_json(self, cold):
        cold.append_personality_signal({"valid": True})
        # Manually insert corrupt data
        cold.conn.execute(
            "INSERT INTO personality_signals (timestamp, data) VALUES (?, ?)",
            (time.time(), "not-json"),
        )
        cold.conn.commit()
        signals = cold.get_personality_signals()
        assert len(signals) == 1
        assert signals[0]["valid"] is True

    def test_append_and_get_personality_history(self, cold):
        entry = {
            "timestamp": time.time(),
            "old_traits": {"warmth": 0.5},
            "new_traits": {"warmth": 0.7},
            "changes": ["warmth increased"],
            "observation_summary": "User responded warmly",
            "confidence": 0.8,
        }
        cold.append_personality_history(entry)
        history = cold.get_personality_history()
        assert len(history) == 1
        assert history[0]["old_traits"] == {"warmth": 0.5}
        assert history[0]["new_traits"] == {"warmth": 0.7}
        assert history[0]["changes"] == ["warmth increased"]
        assert history[0]["observation_summary"] == "User responded warmly"
        assert history[0]["confidence"] == 0.8

    def test_get_personality_history_with_limit(self, cold):
        for i in range(5):
            cold.append_personality_history({
                "timestamp": float(i),
                "old_traits": {},
                "new_traits": {},
                "changes": [],
            })
        history = cold.get_personality_history(limit=3)
        assert len(history) == 3

    def test_get_personality_history_bad_json_fields(self, cold):
        cold.conn.execute(
            "INSERT INTO personality_history "
            "(timestamp, old_traits, new_traits, changes, observation_summary, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1.0, "not-json", "not-json", "not-json", "", 0.0),
        )
        cold.conn.commit()
        history = cold.get_personality_history()
        assert len(history) == 1
        assert history[0]["old_traits"] == {}
        assert history[0]["new_traits"] == {}
        assert history[0]["changes"] == {}


# ── Users ────────────────────────────────────────────────────────────────────


class TestUsers:
    def test_create_user_with_email_and_resolve(self, cold):
        user_id = cold.create_user_with_email("alice@example.com", "slack", "S123")
        assert user_id == IsStr()
        resolved = cold.resolve_user("slack", "S123")
        assert resolved == user_id

    def test_resolve_user_not_linked(self, cold):
        assert cold.resolve_user("slack", "unknown") == ""

    def test_create_user_deduplicates_by_email(self, cold):
        uid1 = cold.create_user_with_email("bob@example.com", "slack", "S1")
        uid2 = cold.create_user_with_email("bob@example.com", "discord", "D1")
        assert uid1 == uid2
        # Both connectors resolve to the same user
        assert cold.resolve_user("slack", "S1") == uid1
        assert cold.resolve_user("discord", "D1") == uid1

    def test_create_user_normalizes_email(self, cold):
        uid1 = cold.create_user_with_email("  Alice@Example.COM  ", "slack", "S1")
        uid2 = cold.create_user_with_email("alice@example.com", "discord", "D1")
        assert uid1 == uid2

    def test_get_user(self, cold):
        uid = cold.create_user_with_email("carol@example.com", "slack", "S1")
        user = cold.get_user(uid)
        assert user is not None
        assert user["user_id"] == uid
        assert user["email"] == "carol@example.com"
        assert user["secrets"] == {}

    def test_get_user_missing(self, cold):
        assert cold.get_user("nonexistent") is None

    def test_get_user_role_default(self, cold):
        uid = cold.create_user_with_email("a@b.com", "slack", "S1")
        assert cold.get_user_role(uid) == "user"

    def test_set_user_role(self, cold):
        uid = cold.create_user_with_email("a@b.com", "slack", "S1")
        cold.set_user_role(uid, "admin")
        assert cold.get_user_role(uid) == "admin"

    def test_get_user_role_missing_user(self, cold):
        assert cold.get_user_role("nonexistent") == "anon"

    def test_set_user_display_name(self, cold):
        uid = cold.create_user_with_email("a@b.com", "slack", "S1")
        cold.set_user_display_name(uid, "Alice")
        user = cold.get_user(uid)
        assert user["display_name"] == "Alice"

    def test_link_user_connector(self, cold):
        uid = cold.create_user_with_email("a@b.com", "slack", "S1")
        cold.link_user_connector(uid, "telegram", "T1")
        links = cold.get_user_links(uid)
        connectors = {(l["connector"], l["connector_id"]) for l in links}
        assert ("slack", "S1") in connectors
        assert ("telegram", "T1") in connectors

    def test_get_user_links_empty(self, cold):
        assert cold.get_user_links("nonexistent") == []

    def test_get_user_profile(self, cold):
        uid = cold.create_user_with_email("a@b.com", "slack", "S1")
        cold.set_user_display_name(uid, "Alice")
        cold.set_user_fact(uid, "gender", "female")
        cold.set_user_fact(uid, "language", "English")
        profile = cold.get_user_profile(uid)
        assert profile["name"] == "Alice"
        assert profile["gender"] == "female"
        assert profile["language"] == "English"

    def test_get_user_profile_fact_name_overrides_display_name(self, cold):
        uid = cold.create_user_with_email("a@b.com", "slack", "S1")
        cold.set_user_display_name(uid, "DisplayName")
        cold.set_user_fact(uid, "name", "PreferredName")
        profile = cold.get_user_profile(uid)
        assert profile["name"] == "PreferredName"

    def test_get_user_profile_empty(self, cold):
        uid = cold.create_user_with_email("a@b.com", "slack", "S1")
        profile = cold.get_user_profile(uid)
        assert isinstance(profile, dict)


# ── Verification ─────────────────────────────────────────────────────────────


class TestVerification:
    def test_store_and_verify_code_success(self, cold):
        cold.store_verification_code("slack", "S1", "a@b.com", "123456")
        email = cold.verify_code("slack", "S1", "123456")
        assert email == "a@b.com"

    def test_verify_code_wrong_code(self, cold):
        cold.store_verification_code("slack", "S1", "a@b.com", "123456")
        assert cold.verify_code("slack", "S1", "wrong") is None
        # Code should still exist (not deleted on wrong attempt)
        assert cold.verify_code("slack", "S1", "123456") == "a@b.com"

    def test_verify_code_missing(self, cold):
        assert cold.verify_code("slack", "S1", "123456") is None

    def test_verify_code_expired(self, cold):
        cold.store_verification_code("slack", "S1", "a@b.com", "123456")
        # Backdate the created_at to 11 minutes ago
        import datetime
        expired = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(minutes=11)
        ).strftime("%Y-%m-%d %H:%M:%S")
        cold.conn.execute(
            "UPDATE verification_codes SET created_at=? WHERE connector='slack' AND connector_id='S1'",
            (expired,),
        )
        cold.conn.commit()
        assert cold.verify_code("slack", "S1", "123456") is None

    def test_verify_code_deleted_after_success(self, cold):
        cold.store_verification_code("slack", "S1", "a@b.com", "123456")
        cold.verify_code("slack", "S1", "123456")
        # Second verify should fail because code was deleted
        assert cold.verify_code("slack", "S1", "123456") is None

    def test_delete_verification_code(self, cold):
        cold.store_verification_code("slack", "S1", "a@b.com", "123456")
        cold.delete_verification_code("slack", "S1")
        assert cold.verify_code("slack", "S1", "123456") is None

    def test_store_verification_code_upsert(self, cold):
        cold.store_verification_code("slack", "S1", "a@b.com", "111111")
        cold.store_verification_code("slack", "S1", "new@b.com", "222222")
        assert cold.verify_code("slack", "S1", "111111") is None
        assert cold.verify_code("slack", "S1", "222222") == "new@b.com"


# ── Bulk Data ────────────────────────────────────────────────────────────────


class TestBulkData:
    def test_get_all_nerve_data(self, cold):
        cold.register_nerve_rich("calc", "calculator", role="tool")
        cold.add_nerve_tool("calc", "math_eval")
        cold.record_qualification("nerve", "calc", True, 1.0, 1, 5, 5, '{"ok":true}')
        cold.record_nerve_invocation("calc", success=True)

        data = cold.get_all_nerve_data()
        assert "calc" in data
        nerve = data["calc"]
        assert nerve["description"] == "calculator"
        assert nerve["total_invocations"] == 1
        assert "math_eval" in nerve["tools"]
        assert nerve["qualification"]["qualified"] is True
        assert nerve["origin"] == "local"

    def test_get_all_nerve_data_empty(self, cold):
        assert cold.get_all_nerve_data() == {}

    def test_get_all_nerve_data_snapshot(self, cold, snapshot):
        """Snapshot the shape of get_all_nerve_data() for regression detection."""
        cold.register_nerve_rich(
            "summarize", "summarize text",
            system_prompt="Be concise.",
            examples_json='[{"input":"long text","output":"short"}]',
            role="tool",
        )
        cold.add_nerve_tool("summarize", "text_splitter")
        cold.add_nerve_tool("summarize", "llm_call")
        cold.record_qualification(
            "nerve", "summarize", True, 0.92, 3, 10, 9, '{"notes":"good"}'
        )
        cold.record_nerve_invocation("summarize", success=True)
        cold.record_nerve_invocation("summarize", success=True)
        cold.record_nerve_invocation("summarize", success=False)

        data = cold.get_all_nerve_data()
        # Redact the timestamp since it varies per run
        if data["summarize"]["qualification"]:
            data["summarize"]["qualification"]["timestamp"] = "<REDACTED>"
        assert data == snapshot

    def test_get_all_nerve_data_contains_expected_keys(self, cold):
        """Every nerve in get_all_nerve_data should have the full set of keys."""
        cold.register_nerve("a", "alpha")
        cold.register_nerve_rich("b", "beta", role="creative")
        data = cold.get_all_nerve_data()
        expected_keys = {
            "description", "is_sense", "total_invocations", "successes",
            "failures", "system_prompt", "examples", "role", "origin",
            "tools", "qualification",
        }
        for name, nerve in data.items():
            assert set(nerve.keys()) == expected_keys, (
                f"Nerve '{name}' has unexpected key set: {set(nerve.keys())}"
            )


# ── Hypothesis: Nerve Invocation Counters ────────────────────────────────────


@pytest.mark.timeout(30)  # hypothesis needs more time than the module-level 10s
class TestNerveInvocationProperty:
    @given(
        successes=st.integers(min_value=0, max_value=30),
        failures=st.integers(min_value=0, max_value=30),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invocation_counts_always_consistent(self, cold, successes, failures):
        """Nerve invocation counters must always sum correctly."""
        assume(successes + failures > 0)
        # Use a unique name per example to avoid accumulation across hypothesis runs
        name = f"nerve_{next(_counter)}"
        cold.register_nerve(name, "property test nerve")
        for _ in range(successes):
            cold.record_nerve_invocation(name, success=True)
        for _ in range(failures):
            cold.record_nerve_invocation(name, success=False)
        info = cold.get_nerve_info(name)
        assert info["total_invocations"] == successes + failures
        assert info["successes"] == successes
        assert info["failures"] == failures


# ── Edge Cases ──────────────────────────────────────────────────────────────

# Strings designed to stress SQL text handling, unicode support, and boundary limits.
_LONG_STRING = "x" * 5500
_SQL_INJECTION = "'; DROP TABLE facts; --"
_BACKSLASH_HEAVY = r"back\\slash\\party\\"
_EMOJI_NAME = "weather_\U0001f327\ufe0f_nerve"
_CJK_NAME = "\u5929\u6c14\u9884\u62a5"
_ACCENTED = "\u00e9\u00e0\u00fc\u00f1\u00f6\u00e7\u00df"
_WHITESPACE_ONLY = "   "
_NEWLINES_TABS = "line1\nline2\ttab"

_EDGE_CASE_NAMES = [
    pytest.param("", id="empty_string"),
    pytest.param(_WHITESPACE_ONLY, id="whitespace_only"),
    pytest.param(_EMOJI_NAME, id="emoji"),
    pytest.param(_CJK_NAME, id="cjk_chars"),
    pytest.param(_ACCENTED, id="accented_chars"),
    pytest.param(_SQL_INJECTION, id="sql_injection_attempt"),
    pytest.param(_BACKSLASH_HEAVY, id="backslashes"),
    pytest.param(_NEWLINES_TABS, id="newlines_and_tabs"),
]

_EDGE_CASE_VALUES = _EDGE_CASE_NAMES + [
    pytest.param(_LONG_STRING, id="very_long_string"),
]


class TestEdgeCaseFacts:
    """Edge cases for the facts subsystem — names and values that could break SQL or encoding."""

    @pytest.mark.parametrize("key", _EDGE_CASE_NAMES)
    def test_fact_roundtrip_with_edge_case_keys(self, cold, key):
        """Facts must survive a set/get roundtrip for unusual key strings."""
        cold.set_fact("edge", key, "value")
        assert cold.get_fact("edge", key) == "value"

    @pytest.mark.parametrize("value", _EDGE_CASE_VALUES)
    def test_fact_roundtrip_with_edge_case_values(self, cold, value):
        """Facts must survive a set/get roundtrip for unusual value strings."""
        cold.set_fact("edge", "k", value)
        assert cold.get_fact("edge", "k") == value

    @pytest.mark.parametrize("category", _EDGE_CASE_NAMES)
    def test_fact_roundtrip_with_edge_case_categories(self, cold, category):
        """Facts must survive a set/get roundtrip for unusual category strings."""
        cold.set_fact(category, "k", "v")
        assert cold.get_fact(category, "k") == "v"


class TestEdgeCaseNerves:
    """Edge cases for nerve registration — names, descriptions, and upsert behavior."""

    @pytest.mark.parametrize("name", _EDGE_CASE_NAMES)
    def test_register_nerve_with_edge_case_names(self, cold, name):
        """Nerve names including empty, unicode, and SQL-special chars must persist."""
        cold.register_nerve(name, "desc")
        info = cold.get_nerve_info(name)
        assert info is not None
        assert info["name"] == name

    @pytest.mark.parametrize("desc", _EDGE_CASE_VALUES)
    def test_register_nerve_with_edge_case_descriptions(self, cold, desc):
        """Nerve descriptions must handle long strings, emoji, and special chars."""
        cold.register_nerve("edge_nerve", desc)
        info = cold.get_nerve_info("edge_nerve")
        assert info["description"] == desc

    def test_register_nerve_upsert_does_not_duplicate(self, cold):
        """Re-registering the same nerve name must update, not insert a second row."""
        cold.register_nerve("dup", "first")
        cold.register_nerve("dup", "second")
        listing = cold.list_nerves()
        assert listing["dup"] == "second"
        # Verify exactly one row exists
        row = cold.conn.execute(
            "SELECT count(*) as cnt FROM nerve_registry WHERE name='dup'"
        ).fetchone()
        assert row["cnt"] == 1

    @pytest.mark.parametrize("prompt", [
        pytest.param("", id="empty_prompt"),
        pytest.param(_LONG_STRING, id="very_long_prompt"),
        pytest.param(_SQL_INJECTION, id="sql_in_prompt"),
        pytest.param(_EMOJI_NAME, id="emoji_in_prompt"),
    ])
    def test_register_nerve_rich_with_edge_case_prompts(self, cold, prompt):
        """System prompts with unusual content must persist through register_nerve_rich."""
        cold.register_nerve_rich("prompt_edge", "desc", system_prompt=prompt, role="tool")
        meta = cold.get_nerve_metadata("prompt_edge")
        assert meta["system_prompt"] == prompt


class TestEdgeCaseUsers:
    """Edge cases for user creation and connector linking."""

    @pytest.mark.parametrize("email", [
        pytest.param("  UPPER@CASE.COM  ", id="needs_normalization"),
        pytest.param("user+tag@example.com", id="plus_addressing"),
        pytest.param("\u00fc\u00f1\u00ee\u00e7\u00f6d\u00e9@example.com", id="unicode_local_part"),
    ])
    def test_create_user_with_unusual_emails(self, cold, email):
        """User creation must handle email normalization edge cases."""
        uid = cold.create_user_with_email(email, "slack", f"S_{email.strip()}")
        assert uid != ""
        user = cold.get_user(uid)
        assert user is not None
        assert user["email"] == email.lower().strip()

    def test_link_same_connector_twice_updates(self, cold):
        """Linking the same connector+id to a new user must update, not error."""
        uid1 = cold.create_user_with_email("a@b.com", "slack", "S1")
        uid2 = cold.create_user_with_email("c@d.com", "slack", "S2")
        # Re-link S1 to uid2
        cold.link_user_connector(uid2, "slack", "S1")
        assert cold.resolve_user("slack", "S1") == uid2


class TestEdgeCaseQualifications:
    """Edge cases for qualification recording."""

    @pytest.mark.parametrize("details_json", [
        pytest.param("", id="empty_string"),
        pytest.param("null", id="json_null"),
        pytest.param("{}", id="empty_object"),
        pytest.param('{"k": "' + _SQL_INJECTION + '"}', id="sql_in_details"),
        pytest.param("not valid json at all {{", id="malformed_json"),
    ])
    def test_qualification_details_parsing(self, cold, details_json):
        """Qualification details must not crash on any JSON-like string."""
        cold.record_qualification("nerve", "q_edge", True, 1.0, 1, 5, 5, details_json)
        q = cold.get_qualification("nerve", "q_edge")
        assert q is not None
        # details should be a dict regardless of input validity
        assert isinstance(q["details"], dict)
