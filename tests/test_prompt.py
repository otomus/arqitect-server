"""Tests for arqitect.brain.prompt — system prompt construction."""

import json
from unittest.mock import patch

import pytest
from dirty_equals import IsStr

from arqitect.types import RedisKey


# ---------------------------------------------------------------------------
# _build_calibration_prompt_section
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestBuildCalibrationPromptSection:
    """Tests for _build_calibration_prompt_section."""

    def _call(self, redis_client, senses):
        """Invoke the function under test with patched Redis and senses."""
        with patch("arqitect.brain.prompt.r", redis_client), \
             patch("arqitect.brain.prompt.CORE_SENSES", senses):
            from arqitect.brain.prompt import _build_calibration_prompt_section
            return _build_calibration_prompt_section()

    def test_all_senses_available(self, test_redis):
        """All senses calibrated and fully available."""
        senses = ["hearing", "sight"]
        for sense in senses:
            cal = {
                "status": "calibrated",
                "capabilities": {
                    "cap_a": {"available": True},
                    "cap_b": {"available": True},
                },
            }
            test_redis.hset(RedisKey.SENSE_CALIBRATION, sense, json.dumps(cal))

        result = self._call(test_redis, senses)

        assert "hearing [calibrated]" in result
        assert "sight [calibrated]" in result
        assert "cap_a" in result
        assert "cap_b" in result
        assert "missing" not in result

    def test_some_capabilities_degraded(self, test_redis):
        """One capability unavailable shows up in the 'missing' parenthetical."""
        cal = {
            "status": "degraded",
            "capabilities": {
                "microphone": {"available": True},
                "speaker": {"available": False},
            },
        }
        test_redis.hset(RedisKey.SENSE_CALIBRATION, "hearing", json.dumps(cal))

        result = self._call(test_redis, ["hearing"])

        assert "hearing [degraded]" in result
        assert "microphone" in result
        assert "(missing: speaker)" in result

    def test_no_calibration_data(self, test_redis):
        """Sense with no Redis entry shows 'not yet calibrated'."""
        result = self._call(test_redis, ["sight"])

        assert "sight [unknown]: not yet calibrated" in result

    def test_json_parse_error(self, test_redis):
        """Corrupt JSON triggers the exception branch."""
        test_redis.hset(RedisKey.SENSE_CALIBRATION, "touch", "NOT-JSON{{")

        result = self._call(test_redis, ["touch"])

        assert "touch [unknown]: calibration data unavailable" in result

    def test_empty_senses_list(self, test_redis):
        """No core senses produces empty string."""
        result = self._call(test_redis, [])

        assert result == ""

    def test_senses_sorted(self, test_redis):
        """Senses appear in sorted order regardless of input order."""
        for name in ["touch", "awareness"]:
            cal = {"status": "ok", "capabilities": {}}
            test_redis.hset(RedisKey.SENSE_CALIBRATION, name, json.dumps(cal))

        result = self._call(test_redis, ["touch", "awareness"])

        awareness_pos = result.index("awareness")
        touch_pos = result.index("touch")
        assert awareness_pos < touch_pos

    def test_calibration_snapshot(self, test_redis, snapshot):
        """Snapshot the calibration section structure for regression detection."""
        cal = {
            "status": "calibrated",
            "capabilities": {
                "camera": {"available": True},
                "lidar": {"available": False},
            },
        }
        test_redis.hset(RedisKey.SENSE_CALIBRATION, "sight", json.dumps(cal))

        result = self._call(test_redis, ["sight"])

        assert result == snapshot


# ---------------------------------------------------------------------------
# _build_session_info
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestBuildSessionInfo:
    """Tests for _build_session_info."""

    def _call(self, memory_manager, session_data):
        """Invoke with a real MemoryManager after seeding session data."""
        if session_data:
            memory_manager.hot.set_session(session_data)

        with patch("arqitect.brain.prompt.mem", memory_manager):
            from arqitect.brain.prompt import _build_session_info
            return _build_session_info()

    def test_full_session(self, mem):
        """City, country, and timezone all present."""
        result = self._call(mem, {
            "city": "Tel Aviv",
            "country": "Israel",
            "timezone": "Asia/Jerusalem",
        })

        assert "Tel Aviv" in result
        assert "Israel" in result
        assert "Asia/Jerusalem" in result
        assert "KNOWN USER CONTEXT" in result

    def test_no_city_returns_empty(self, mem):
        """No city key means no session block."""
        result = self._call(mem, {})

        assert result == ""

    def test_city_empty_string_returns_empty(self, mem):
        """Empty-string city is falsy, so no session block."""
        result = self._call(mem, {"city": ""})

        assert result == ""

    def test_missing_country_and_timezone(self, mem):
        """City present but country/timezone missing fall back to '?'."""
        result = self._call(mem, {"city": "London"})

        assert "London" in result
        assert "?" in result

    def test_session_info_snapshot(self, mem, snapshot):
        """Snapshot the session info structure for regression detection."""
        result = self._call(mem, {
            "city": "Berlin",
            "country": "Germany",
            "timezone": "Europe/Berlin",
        })

        assert result == snapshot


# ---------------------------------------------------------------------------
# _build_few_shot_section
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestBuildFewShotSection:
    """Tests for _build_few_shot_section."""

    def _call(self, examples):
        """Invoke the function under test."""
        from arqitect.brain.prompt import _build_few_shot_section
        return _build_few_shot_section(examples)

    def test_valid_examples(self):
        """Well-formed examples produce the expected output."""
        examples = [
            {"input": "hello", "output": "Hi there!"},
            {"input": "weather?", "output": "Let me check."},
        ]
        result = self._call(examples)

        assert "Examples:" in result
        assert 'User: "hello"' in result
        assert "Hi there!" in result
        assert 'User: "weather?"' in result

    def test_empty_list(self):
        """Empty list returns empty string."""
        assert self._call([]) == ""

    def test_malformed_dicts_skipped(self):
        """Dicts missing required keys are silently skipped."""
        examples = [
            {"input": "good", "output": "ok"},
            {"input": "missing output"},
            {"wrong_key": "value"},
            "not a dict",
        ]
        result = self._call(examples)

        assert "Examples:" in result
        assert '"good"' in result
        # Only the valid example should appear as a User: line
        assert result.count("User:") == 1

    def test_all_invalid(self):
        """If every example is malformed, we still get the 'Examples:' header."""
        result = self._call([{"bad": "entry"}])

        assert "Examples:" in result
        assert "User:" not in result

    def test_few_shot_snapshot(self, snapshot):
        """Snapshot the few-shot section structure for regression detection."""
        examples = [
            {"input": "hi", "output": "hello"},
            {"input": "bye", "output": "goodbye"},
        ]
        result = self._call(examples)

        assert result == snapshot


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------

@pytest.mark.timeout(10)
class TestGetSystemPrompt:
    """Tests for get_system_prompt."""

    def test_adapter_found_with_all_sections(self, test_redis, mem):
        """Full prompt includes calibration, few-shot, and session sections."""
        mem.hot.set_session({"city": "Paris", "country": "France", "timezone": "Europe/Paris"})

        adapter = {
            "system_prompt": "You are the brain.",
            "few_shot_examples": [
                {"input": "hi", "output": "hello"},
            ],
        }
        cal = {
            "status": "ok",
            "capabilities": {"mic": {"available": True}},
        }
        test_redis.hset(RedisKey.SENSE_CALIBRATION, "hearing", json.dumps(cal))

        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", ["hearing"]), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value=adapter):
            from arqitect.brain.prompt import get_system_prompt
            result = get_system_prompt()

        assert result.startswith("You are the brain.")
        assert "hearing [ok]" in result
        assert "Paris" in result
        assert '"hi"' in result

    def test_adapter_missing_raises(self, test_redis, mem):
        """RuntimeError when resolve_prompt returns None."""
        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", []), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value=None):
            from arqitect.brain.prompt import get_system_prompt
            with pytest.raises(RuntimeError, match="Brain adapter not found"):
                get_system_prompt()

    def test_adapter_empty_system_prompt_raises(self, test_redis, mem):
        """RuntimeError when adapter has no system_prompt."""
        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", []), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value={"system_prompt": ""}):
            from arqitect.brain.prompt import get_system_prompt
            with pytest.raises(RuntimeError):
                get_system_prompt()

    def test_no_calibration_no_session(self, test_redis, mem):
        """Prompt with empty calibration and no session still works."""
        adapter = {"system_prompt": "Base prompt."}

        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", []), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value=adapter):
            from arqitect.brain.prompt import get_system_prompt
            result = get_system_prompt()

        assert result == "Base prompt."

    def test_full_prompt_snapshot(self, test_redis, mem, snapshot):
        """Snapshot the full composed prompt for regression detection."""
        mem.hot.set_session({"city": "Tokyo", "country": "Japan", "timezone": "Asia/Tokyo"})

        adapter = {
            "system_prompt": "You are Sentient.",
            "few_shot_examples": [
                {"input": "greet", "output": "Hello!"},
            ],
        }
        cal = {
            "status": "calibrated",
            "capabilities": {"camera": {"available": True}},
        }
        test_redis.hset(RedisKey.SENSE_CALIBRATION, "sight", json.dumps(cal))

        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", ["sight"]), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value=adapter):
            from arqitect.brain.prompt import get_system_prompt
            result = get_system_prompt()

        assert result == snapshot

    def test_prompt_contains_all_sections(self, test_redis, mem):
        """The composed prompt contains all expected structural sections."""
        mem.hot.set_session({"city": "NYC", "country": "US", "timezone": "America/New_York"})

        adapter = {
            "system_prompt": "Core identity.",
            "few_shot_examples": [
                {"input": "test", "output": "response"},
            ],
        }
        cal = {"status": "ok", "capabilities": {"sensor": {"available": True}}}
        test_redis.hset(RedisKey.SENSE_CALIBRATION, "touch", json.dumps(cal))

        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", ["touch"]), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value=adapter):
            from arqitect.brain.prompt import get_system_prompt
            result = get_system_prompt()

        # Verify structural sections are present using dirty_equals
        assert result == IsStr(regex="(?s).*Core identity\\..*")
        assert result == IsStr(regex="(?s).*Core senses.*calibration.*")
        assert result == IsStr(regex="(?s).*Examples:.*")
        assert result == IsStr(regex="(?s).*KNOWN USER CONTEXT.*")
