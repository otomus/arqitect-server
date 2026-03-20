"""Tests for arqitect.brain.prompt — system prompt construction."""

import json
from unittest.mock import patch, MagicMock

import pytest

from arqitect.types import RedisKey


# ---------------------------------------------------------------------------
# _build_calibration_prompt_section
# ---------------------------------------------------------------------------

class TestBuildCalibrationPromptSection:
    """Tests for _build_calibration_prompt_section."""

    def _call(self, redis_client, senses):
        """Invoke the function under test with mocked Redis and senses."""
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


# ---------------------------------------------------------------------------
# _build_session_info
# ---------------------------------------------------------------------------

class TestBuildSessionInfo:
    """Tests for _build_session_info."""

    def _call(self, session_data):
        """Invoke with a mock memory manager returning the given session."""
        mock_mem = MagicMock()
        mock_mem.hot.get_session.return_value = session_data
        with patch("arqitect.brain.prompt.mem", mock_mem):
            from arqitect.brain.prompt import _build_session_info
            return _build_session_info()

    def test_full_session(self):
        """City, country, and timezone all present."""
        result = self._call({
            "city": "Tel Aviv",
            "country": "Israel",
            "timezone": "Asia/Jerusalem",
        })

        assert "Tel Aviv" in result
        assert "Israel" in result
        assert "Asia/Jerusalem" in result
        assert "KNOWN USER CONTEXT" in result

    def test_no_city_returns_empty(self):
        """No city key means no session block."""
        result = self._call({})

        assert result == ""

    def test_city_empty_string_returns_empty(self):
        """Empty-string city is falsy, so no session block."""
        result = self._call({"city": ""})

        assert result == ""

    def test_missing_country_and_timezone(self):
        """City present but country/timezone missing fall back to '?'."""
        result = self._call({"city": "London"})

        assert "London" in result
        assert "?" in result


# ---------------------------------------------------------------------------
# _build_few_shot_section
# ---------------------------------------------------------------------------

class TestBuildFewShotSection:
    """Tests for _build_few_shot_section."""

    def _call(self, examples):
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


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------

class TestGetSystemPrompt:
    """Tests for get_system_prompt."""

    def test_adapter_found_with_all_sections(self, test_redis):
        """Full prompt includes calibration, few-shot, and session sections."""
        mock_mem = MagicMock()
        mock_mem.hot.get_session.return_value = {
            "city": "Paris",
            "country": "France",
            "timezone": "Europe/Paris",
        }
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
             patch("arqitect.brain.prompt.mem", mock_mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", ["hearing"]), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value=adapter):
            from arqitect.brain.prompt import get_system_prompt
            result = get_system_prompt()

        assert result.startswith("You are the brain.")
        assert "hearing [ok]" in result
        assert "Paris" in result
        assert '"hi"' in result

    def test_adapter_missing_raises(self, test_redis):
        """RuntimeError when resolve_prompt returns None."""
        mock_mem = MagicMock()
        mock_mem.hot.get_session.return_value = {}

        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mock_mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", []), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value=None):
            from arqitect.brain.prompt import get_system_prompt
            with pytest.raises(RuntimeError, match="Brain adapter not found"):
                get_system_prompt()

    def test_adapter_empty_system_prompt_raises(self, test_redis):
        """RuntimeError when adapter has no system_prompt."""
        mock_mem = MagicMock()
        mock_mem.hot.get_session.return_value = {}

        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mock_mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", []), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value={"system_prompt": ""}):
            from arqitect.brain.prompt import get_system_prompt
            with pytest.raises(RuntimeError):
                get_system_prompt()

    def test_no_calibration_no_session(self, test_redis):
        """Prompt with empty calibration and no session still works."""
        mock_mem = MagicMock()
        mock_mem.hot.get_session.return_value = {}
        adapter = {"system_prompt": "Base prompt."}

        with patch("arqitect.brain.prompt.r", test_redis), \
             patch("arqitect.brain.prompt.mem", mock_mem), \
             patch("arqitect.brain.prompt.CORE_SENSES", []), \
             patch("arqitect.brain.adapters.resolve_prompt", return_value=adapter):
            from arqitect.brain.prompt import get_system_prompt
            result = get_system_prompt()

        assert result == "Base prompt."
