"""Tests for arqitect.brain.intent — intent classification.

Uses FakeLLM for scripted LLM responses and real extract_json (pure function).
"""

from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arqitect.types import IntentType
from tests.conftest import FakeLLM


@pytest.mark.timeout(10)
class TestClassifyIntent:
    """Contract tests for classify_intent.

    Contract:
    - Returns {"type": IntentType.PLAN, "category": ...} for plan intents.
    - Returns {"type": IntentType.DIRECT} for direct intents.
    - Falls back to DIRECT when LLM output is unparseable or has invalid type.
    - Always returns a dict with a "type" key.
    """

    def _call(self, fake_llm: FakeLLM, task: str = "some user message") -> dict:
        """Invoke classify_intent with a FakeLLM and real extract_json.

        Args:
            fake_llm: Scripted FakeLLM instance.
            task: User message to classify.

        Returns:
            The dict returned by classify_intent.
        """
        with patch("arqitect.brain.intent.llm_generate", side_effect=fake_llm):
            from arqitect.brain.intent import classify_intent
            return classify_intent(task)

    def test_valid_plan_json(self):
        """LLM returns valid plan intent with category."""
        fake = FakeLLM([
            ("Classify", '{"type": "plan", "category": "development"}'),
        ])

        result = self._call(fake)

        assert result["type"] == IntentType.PLAN
        assert result["category"] == "development"

    def test_valid_direct_json(self):
        """LLM returns valid direct intent."""
        fake = FakeLLM([
            ("Classify", '{"type": "direct"}'),
        ])

        result = self._call(fake)

        assert result["type"] == IntentType.DIRECT

    def test_non_json_text_falls_back_to_direct(self):
        """LLM returns plain text — extract_json returns None, fallback to direct."""
        fake = FakeLLM([
            ("Classify", "I think this is a greeting"),
        ])

        result = self._call(fake)

        assert result == {"type": IntentType.DIRECT}

    def test_invalid_type_field_falls_back_to_direct(self):
        """LLM returns JSON with an unrecognized type value."""
        fake = FakeLLM([
            ("Classify", '{"type": "unknown_type"}'),
        ])

        result = self._call(fake)

        assert result == {"type": IntentType.DIRECT}

    def test_empty_string_falls_back_to_direct(self):
        """LLM returns empty string — no JSON to extract."""
        fake = FakeLLM([
            ("Classify", ""),
        ])

        result = self._call(fake)

        assert result == {"type": IntentType.DIRECT}

    def test_plan_with_category_preserved(self):
        """Category field is preserved in the returned dict."""
        fake = FakeLLM([
            ("Classify", '{"type": "plan", "category": "debugging"}'),
        ])

        result = self._call(fake)

        assert result["type"] == IntentType.PLAN
        assert result["category"] == "debugging"

    def test_extract_json_returns_empty_dict(self):
        """LLM returns '{}' (no 'type' key) — falls back to direct."""
        fake = FakeLLM([
            ("Classify", "{}"),
        ])

        result = self._call(fake)

        assert result == {"type": IntentType.DIRECT}

    def test_llm_called_with_task_in_prompt(self):
        """Verify the user task appears in the prompt sent to llm_generate."""
        fake = FakeLLM([
            ("Classify", '{"type": "direct"}'),
        ])

        self._call(fake, task="build me a web app")

        assert fake.call_count == 1
        call = fake.calls[0]
        assert "build me a web app" in call["prompt"]

    def test_llm_called_with_system_prompt(self):
        """Verify llm_generate receives a system prompt about classification."""
        fake = FakeLLM([
            ("Classify", '{"type": "direct"}'),
        ])

        self._call(fake, task="hello there")

        call = fake.calls[0]
        assert "classify" in call["system"].lower() or "plan" in call["system"].lower()

    def test_llm_called_with_brain_model(self):
        """Verify llm_generate is called with the configured BRAIN_MODEL."""
        fake = FakeLLM([
            ("Classify", '{"type": "direct"}'),
        ])

        with patch("arqitect.brain.intent.BRAIN_MODEL", "test-model"):
            with patch("arqitect.brain.intent.llm_generate", side_effect=fake):
                from arqitect.brain.intent import classify_intent
                classify_intent("anything")

        assert fake.calls[0]["model"] == "test-model"

    def test_plan_without_category(self):
        """LLM returns plan without optional category field."""
        fake = FakeLLM([
            ("Classify", '{"type": "plan"}'),
        ])

        result = self._call(fake)

        assert result["type"] == IntentType.PLAN
        assert "category" not in result

    def test_json_embedded_in_prose(self):
        """LLM wraps JSON in surrounding text — extract_json still finds it."""
        fake = FakeLLM([
            ("Classify", 'Sure, here is my answer:\n{"type": "plan", "category": "setup"}\nDone.'),
        ])

        result = self._call(fake)

        assert result["type"] == IntentType.PLAN
        assert result["category"] == "setup"


@pytest.mark.timeout(10)
class TestClassifyIntentPropertyBased:
    """Property-based tests using Hypothesis for classify_intent."""

    @given(task=st.text(min_size=1, max_size=500))
    @settings(max_examples=50, deadline=5000)
    def test_always_returns_dict_with_type(self, task: str):
        """classify_intent always returns a dict with a valid 'type' key,
        regardless of input task string.
        """
        fake = FakeLLM([
            ("Classify", '{"type": "direct"}', True),
        ])

        with patch("arqitect.brain.intent.llm_generate", side_effect=fake):
            from arqitect.brain.intent import classify_intent
            result = classify_intent(task)

        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] in (IntentType.PLAN, IntentType.DIRECT)

    @given(task=st.text(min_size=1, max_size=500))
    @settings(max_examples=50, deadline=5000)
    def test_unparseable_llm_output_always_falls_back_to_direct(self, task: str):
        """When the LLM returns garbage, classify_intent always defaults to DIRECT."""
        fake = FakeLLM([
            ("Classify", "not json at all ~~~", True),
        ])

        with patch("arqitect.brain.intent.llm_generate", side_effect=fake):
            from arqitect.brain.intent import classify_intent
            result = classify_intent(task)

        assert result == {"type": IntentType.DIRECT}

    @given(category=st.sampled_from([
        "development", "debugging", "setup", "planning",
        "migration", "deployment", "refactoring",
    ]))
    @settings(max_examples=20, deadline=5000)
    def test_all_known_categories_pass_through(self, category: str):
        """All documented plan categories are preserved in the result."""
        import json
        response = json.dumps({"type": "plan", "category": category})
        fake = FakeLLM([
            ("Classify", response, True),
        ])

        with patch("arqitect.brain.intent.llm_generate", side_effect=fake):
            from arqitect.brain.intent import classify_intent
            result = classify_intent("do something")

        assert result["type"] == IntentType.PLAN
        assert result["category"] == category
