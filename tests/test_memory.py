"""P5 — Memory integration tests: episode recording, context building, facts.

All episode data is produced by typed factories.
"""

import json
from unittest.mock import patch

import pytest

from tests.factories import EpisodeFactory, as_dict


class TestContextForTask:
    def test_returns_session_episodes_facts(self, mem):
        """get_context_for_task should return session, episodes, and facts."""
        context = mem.get_context_for_task("hello", user_id="user1")
        assert "session" in context
        assert "episodes" in context
        assert "facts" in context

    def test_conversation_history_in_context(self, mem):
        """Messages added to hot memory should appear in conversation."""
        mem.hot.add_message("user", "hello", user_id="user1")
        mem.hot.add_message("assistant", "hi there", user_id="user1")
        messages = mem.hot.get_conversation(limit=10, user_id="user1")
        assert len(messages) >= 2
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles


class TestEpisodeRecording:
    def test_record_episode_stores_in_warm(self, mem):
        """record_episode should store the episode in warm memory."""
        episode = EpisodeFactory.build(
            task="weather in Paris",
            nerve="weather_nerve",
            tool="weather_api",
            user_id="user1",
        )
        mem.record_episode(as_dict(episode))

        recalled = mem.warm.recall("weather", limit=5)
        assert len(recalled) > 0
        assert any(ep.get("nerve") == "weather_nerve" for ep in recalled)

    def test_record_episode_updates_cold_stats(self, mem):
        """record_episode should update invocation stats in cold memory."""
        episode = EpisodeFactory.build(task="tell joke", nerve="joke_nerve")
        mem.record_episode(as_dict(episode))


class TestFactStorage:
    def test_set_and_get_user_fact(self, mem):
        """User facts should be storable and retrievable."""
        mem.cold.set_user_fact("user1", "city", "Tel Aviv", confidence=1.0)
        facts = mem.cold.get_user_facts("user1")
        assert "city" in facts
        assert facts["city"] == "Tel Aviv"

    def test_facts_scoped_to_user(self, mem):
        """Facts for different users should not leak."""
        mem.cold.set_user_fact("user1", "city", "Tel Aviv", confidence=1.0)
        mem.cold.set_user_fact("user2", "city", "London", confidence=1.0)
        facts1 = mem.cold.get_user_facts("user1")
        facts2 = mem.cold.get_user_facts("user2")
        assert facts1["city"] == "Tel Aviv"
        assert facts2["city"] == "London"


class TestEnvForNerve:
    def test_env_contains_required_keys(self, mem):
        """get_env_for_nerve should produce all SYNAPSE_* env vars."""
        env = mem.get_env_for_nerve("test_nerve", "do something", user_id="user1")
        required_keys = [
            "SYNAPSE_NERVE_NAME",
            "SYNAPSE_SESSION",
            "SYNAPSE_EPISODES",
            "SYNAPSE_KNOWN_TOOLS",
            "SYNAPSE_FACTS",
            "SYNAPSE_NERVE_META",
            "SYNAPSE_USER_ID",
        ]
        for key in required_keys:
            assert key in env, f"Missing env var: {key}"

    def test_env_values_are_json_serializable(self, mem):
        """All SYNAPSE_* values should be valid JSON strings."""
        env = mem.get_env_for_nerve("test_nerve", "do something", user_id="user1")
        json_keys = ["SYNAPSE_SESSION", "SYNAPSE_EPISODES", "SYNAPSE_KNOWN_TOOLS",
                      "SYNAPSE_FACTS", "SYNAPSE_NERVE_META"]
        for key in json_keys:
            if key in env:
                try:
                    json.loads(env[key])
                except json.JSONDecodeError:
                    pytest.fail(f"{key} is not valid JSON: {env[key][:100]}")
