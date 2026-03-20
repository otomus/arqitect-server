"""Tests for arqitect/brain/events.py — event publishing and response validation."""

import json
from unittest.mock import patch

import pytest
from dirty_equals import IsDict, IsStr, IsInt
from hypothesis import given, settings
from hypothesis import strategies as st

from arqitect.brain.events import (
    _build_nerve_details,
    _text_similarity,
    _validate_response,
    get_task_origin,
    publish_event,
    set_task_origin,
)
from arqitect.types import NerveRole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_task_origin():
    """Reset task origin state before each test."""
    set_task_origin(source="", chat_id="", user_id="")
    yield
    set_task_origin(source="", chat_id="", user_id="")


# ---------------------------------------------------------------------------
# set_task_origin / get_task_origin
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestTaskOrigin:
    """Tests for module-level task origin state."""

    def test_set_and_get_roundtrip(self):
        set_task_origin(source="telegram", chat_id="chat-123", user_id="user-456")
        origin = get_task_origin()
        assert origin == IsDict(source="telegram", chat_id="chat-123", user_id="user-456")

    def test_get_before_set_returns_defaults(self):
        origin = get_task_origin()
        assert origin == IsDict(source="", chat_id="", user_id="")

    def test_all_fields_empty(self):
        set_task_origin(source="", chat_id="", user_id="")
        origin = get_task_origin()
        assert origin == {"source": "", "chat_id": "", "user_id": ""}

    def test_partial_update(self):
        set_task_origin(source="discord")
        origin = get_task_origin()
        assert origin["source"] == "discord"
        assert origin["chat_id"] == ""
        assert origin["user_id"] == ""

    def test_overwrite_previous_values(self):
        set_task_origin(source="telegram", chat_id="c1", user_id="u1")
        set_task_origin(source="web", chat_id="c2", user_id="u2")
        origin = get_task_origin()
        assert origin == IsDict(source="web", chat_id="c2", user_id="u2")


# ---------------------------------------------------------------------------
# publish_event
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestPublishEvent:
    """Tests for publish_event — Redis pub/sub wrapper using fakeredis."""

    def test_calls_redis_publish_with_json(self, test_redis):
        """publish_event serializes data as JSON and publishes to the channel."""
        data = {"action": "invoke_nerve", "name": "joke"}
        with patch("arqitect.brain.events.r", test_redis):
            pubsub = test_redis.pubsub()
            pubsub.subscribe("brain:action")
            # Consume the subscribe confirmation message
            pubsub.get_message()

            publish_event("brain:action", data)

            msg = pubsub.get_message()
            assert msg is not None
            assert msg["channel"] == "brain:action"
            assert json.loads(msg["data"]) == data

    def test_serializes_nested_data(self, test_redis):
        data = {"outer": {"inner": [1, 2, 3]}}
        with patch("arqitect.brain.events.r", test_redis):
            pubsub = test_redis.pubsub()
            pubsub.subscribe("test:channel")
            pubsub.get_message()

            publish_event("test:channel", data)

            msg = pubsub.get_message()
            assert json.loads(msg["data"]) == IsDict(outer=IsDict(inner=[1, 2, 3]))

    def test_empty_dict(self, test_redis):
        with patch("arqitect.brain.events.r", test_redis):
            pubsub = test_redis.pubsub()
            pubsub.subscribe("test:channel")
            pubsub.get_message()

            publish_event("test:channel", {})

            msg = pubsub.get_message()
            assert json.loads(msg["data"]) == {}


# ---------------------------------------------------------------------------
# _validate_response
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestValidateResponse:
    """Tests for _validate_response — response quality gate."""

    def test_empty_string_returns_empty(self):
        assert _validate_response("") == "empty"

    def test_none_returns_empty(self):
        assert _validate_response(None) == "empty"

    def test_single_char_returns_empty(self):
        assert _validate_response("x") == "empty"

    def test_whitespace_only_returns_empty(self):
        assert _validate_response("   ") == "empty"

    def test_starts_with_action_json_double_quotes(self):
        assert _validate_response('{"action": "invoke_nerve"}') == "leaked_json"

    def test_starts_with_action_json_single_quotes(self):
        assert _validate_response("{'action': 'invoke_nerve'}") == "leaked_json"

    def test_json_code_fence_with_action(self):
        msg = '```json\n{"action": "invoke_nerve", "name": "test"}\n```'
        assert _validate_response(msg) == "leaked_json"

    def test_echo_detection_exact_match(self):
        task = "What is the weather?"
        assert _validate_response("What is the weather?", task=task) == "echo"

    def test_echo_detection_high_similarity(self):
        task = "What is the weather today?"
        msg = "What is the weather today"  # missing question mark
        assert _validate_response(msg, task=task) == "echo"

    def test_echo_detection_long_strings_skip_check(self):
        """Strings over 100 chars should skip echo similarity check."""
        long_task = "A" * 101
        long_msg = "A" * 101
        # Exact match still catches it
        assert _validate_response(long_msg, task=long_task) == "echo"

    def test_echo_detection_long_strings_similar_but_not_exact(self):
        """Long similar (but not exact) strings should not trigger echo."""
        long_task = "A" * 101
        long_msg = "A" * 100 + "B"
        result = _validate_response(long_msg, task=long_task)
        assert result is None

    def test_valid_response_returns_none(self):
        assert _validate_response("Here is your answer: 42") is None

    def test_valid_response_with_task(self):
        result = _validate_response(
            "The weather in New York is sunny and 75F.",
            task="What is the weather?",
        )
        assert result is None

    def test_leaked_tool_call(self):
        msg = "call: joke_nerve args: tell me a joke"
        assert _validate_response(msg) == "leaked_tool_call"

    def test_no_echo_when_task_empty(self):
        assert _validate_response("some response", task="") is None


# ---------------------------------------------------------------------------
# _text_similarity
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestTextSimilarity:
    """Tests for _text_similarity — character-level similarity."""

    def test_identical_strings(self):
        assert _text_similarity("hello", "hello") == 1.0

    def test_empty_first_string(self):
        assert _text_similarity("", "hello") == 0.0

    def test_empty_second_string(self):
        assert _text_similarity("hello", "") == 0.0

    def test_both_empty(self):
        assert _text_similarity("", "") == 0.0

    def test_completely_different(self):
        score = _text_similarity("abcdef", "zyxwvu")
        assert score < 0.3

    def test_similar_strings(self):
        score = _text_similarity("hello world", "hello worl")
        assert score > 0.8

    def test_none_input(self):
        assert _text_similarity(None, "hello") == 0.0

    def test_none_both(self):
        assert _text_similarity(None, None) == 0.0

    @given(text=st.text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_identical_strings_always_score_one(self, text):
        """Any non-empty string compared to itself must score 1.0."""
        assert _text_similarity(text, text) == 1.0

    @given(a=st.text(min_size=0, max_size=200), b=st.text(min_size=0, max_size=200))
    @settings(max_examples=50)
    def test_score_is_symmetric(self, a, b):
        """Similarity must be symmetric: sim(a, b) == sim(b, a)."""
        assert _text_similarity(a, b) == _text_similarity(b, a)

    @given(a=st.text(min_size=0, max_size=200), b=st.text(min_size=0, max_size=200))
    @settings(max_examples=50)
    def test_score_bounded_zero_to_one(self, a, b):
        """Similarity score must always be in [0.0, 1.0]."""
        score = _text_similarity(a, b)
        assert 0.0 <= score <= 1.0

    @given(
        a=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=50),
        b=st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_different_strings_score_below_one(self, a, b):
        """Two genuinely different non-empty strings should score below 1.0."""
        from hypothesis import assume
        assume(a != b)
        assert _text_similarity(a, b) < 1.0


# ---------------------------------------------------------------------------
# _build_nerve_details
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
class TestBuildNerveDetails:
    """Tests for _build_nerve_details — dashboard nerve detail construction."""

    def test_with_qualification(self):
        info = {"description": "Tells jokes", "total_invocations": 10, "successes": 8, "failures": 2}
        meta = {"role": NerveRole.CREATIVE, "system_prompt": "Be funny", "examples": ["example1"]}
        tools = ["joke_tool"]
        qual = {
            "score": 0.85,
            "qualified": True,
            "iterations": 2,
            "test_count": 5,
            "pass_count": 4,
            "details": [{"test": "t1", "pass": True}],
            "timestamp": "2026-01-01T00:00:00",
        }

        details = _build_nerve_details("joke_nerve", info, meta, tools, qual)

        assert details == IsDict(
            name="joke_nerve",
            description=IsStr(),
            role=NerveRole.CREATIVE,
            system_prompt="Be funny",
            examples=["example1"],
            tools=["joke_tool"],
            total_invocations=IsInt(gt=0),
            successes=IsInt(gt=0),
            failures=IsInt(ge=0),
            score=85,
            qualified=True,
            iterations=2,
            test_count=5,
            pass_count=4,
            test_results=[{"test": "t1", "pass": True}],
            last_qualified="2026-01-01T00:00:00",
        )

    def test_without_qualification(self):
        info = {"description": "A nerve"}
        meta = {"role": NerveRole.TOOL}
        tools = []

        details = _build_nerve_details("test_nerve", info, meta, tools, None)

        assert details["name"] == "test_nerve"
        assert details["score"] == 0
        assert details["qualified"] is None
        assert details["test_results"] == []
        assert "iterations" not in details
        assert "test_count" not in details

    def test_partial_info(self):
        """Missing keys in info/meta should use defaults."""
        details = _build_nerve_details("partial", {}, {}, [], None)

        assert details["description"] == ""
        assert details["role"] == NerveRole.TOOL
        assert details["system_prompt"] == ""
        assert details["examples"] == []
        assert details["total_invocations"] == 0
        assert details["successes"] == 0
        assert details["failures"] == 0

    def test_qual_with_missing_fields_uses_defaults(self):
        """Qualification dict with missing keys should use get() defaults."""
        qual = {"score": 0.5}
        details = _build_nerve_details("n", {}, {}, [], qual)

        assert details["score"] == 50
        assert details["qualified"] is False
        assert details["iterations"] == 0
        assert details["test_count"] == 0
        assert details["pass_count"] == 0
        assert details["test_results"] == []
        assert details["last_qualified"] == ""
