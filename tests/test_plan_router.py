"""Tests for arqitect.brain.plan_router — plan message classification.

Covers:
- Correct classification of continue/aside/approve/abort messages
- Fallback to 'continue' on unparseable LLM output
- Plan context is included in the LLM prompt
"""

from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arqitect.brain.plan_session import PlanSession
from tests.conftest import FakeLLM


def _make_plan() -> PlanSession:
    """Create a test plan in gathering status."""
    plan = PlanSession.create("user-1", "build a REST API", "development")
    plan.add_requirement("Use FastAPI")
    return plan


@pytest.mark.timeout(10)
class TestClassifyPlanMessage:
    """Contract tests for classify_plan_message."""

    def _call(self, fake_llm: FakeLLM, task: str, plan: PlanSession | None = None) -> str:
        """Invoke classify_plan_message with a FakeLLM.

        Args:
            fake_llm: Scripted FakeLLM instance.
            task: The user's new message.
            plan: Active PlanSession (defaults to a test plan).

        Returns:
            The classification string.
        """
        if plan is None:
            plan = _make_plan()
        with patch("arqitect.brain.plan_router.llm_generate", side_effect=fake_llm):
            from arqitect.brain.plan_router import classify_plan_message
            return classify_plan_message(task, plan)

    def test_continue_action(self):
        """LLM classifies message as continuing the plan."""
        fake = FakeLLM([
            ("Active plan goal", '{"action": "continue"}'),
        ])

        result = self._call(fake, "I want PostgreSQL for the database")

        assert result == "continue"

    def test_aside_action(self):
        """LLM classifies message as unrelated side question."""
        fake = FakeLLM([
            ("Active plan goal", '{"action": "aside"}'),
        ])

        result = self._call(fake, "What's the weather like?")

        assert result == "aside"

    def test_approve_action(self):
        """LLM classifies message as approving the plan."""
        fake = FakeLLM([
            ("Active plan goal", '{"action": "approve"}'),
        ])

        result = self._call(fake, "Looks good, let's go!")

        assert result == "approve"

    def test_abort_action(self):
        """LLM classifies message as aborting the plan."""
        fake = FakeLLM([
            ("Active plan goal", '{"action": "abort"}'),
        ])

        result = self._call(fake, "Nevermind, cancel that")

        assert result == "abort"

    def test_unparseable_falls_back_to_continue(self):
        """When LLM returns garbage, default to 'continue'."""
        fake = FakeLLM([
            ("Active plan goal", "I'm not sure what to do"),
        ])

        result = self._call(fake, "some message")

        assert result == "continue"

    def test_invalid_action_falls_back_to_continue(self):
        """When LLM returns JSON with invalid action, default to 'continue'."""
        fake = FakeLLM([
            ("Active plan goal", '{"action": "invalid_action"}'),
        ])

        result = self._call(fake, "some message")

        assert result == "continue"

    def test_plan_context_in_prompt(self):
        """Verify plan goal and requirements appear in the LLM prompt."""
        fake = FakeLLM([
            ("Active plan goal", '{"action": "continue"}'),
        ])

        plan = _make_plan()
        self._call(fake, "add authentication", plan=plan)

        assert fake.call_count == 1
        prompt = fake.calls[0]["prompt"]
        assert "build a REST API" in prompt
        assert "Use FastAPI" in prompt

    def test_system_prompt_describes_classification(self):
        """Verify the system prompt instructs about plan classification."""
        fake = FakeLLM([
            ("Active plan goal", '{"action": "continue"}'),
        ])

        self._call(fake, "any message")

        system = fake.calls[0]["system"]
        assert "active work plan" in system.lower() or "classify" in system.lower()


@pytest.mark.timeout(10)
class TestClassifyPlanMessagePropertyBased:
    """Property-based tests for classify_plan_message."""

    @given(task=st.text(min_size=1, max_size=500))
    @settings(max_examples=30, deadline=5000)
    def test_always_returns_valid_action(self, task: str):
        """classify_plan_message always returns a valid action string."""
        fake = FakeLLM([
            ("Active plan goal", '{"action": "continue"}', True),
        ])
        plan = _make_plan()

        with patch("arqitect.brain.plan_router.llm_generate", side_effect=fake):
            from arqitect.brain.plan_router import classify_plan_message
            result = classify_plan_message(task, plan)

        assert result in {"continue", "aside", "approve", "abort"}

    @given(task=st.text(min_size=1, max_size=500))
    @settings(max_examples=30, deadline=5000)
    def test_garbage_llm_never_crashes(self, task: str):
        """Even with garbage LLM output, classify_plan_message never raises."""
        fake = FakeLLM([
            ("Active plan goal", "random garbage ~~~!!!", True),
        ])
        plan = _make_plan()

        with patch("arqitect.brain.plan_router.llm_generate", side_effect=fake):
            from arqitect.brain.plan_router import classify_plan_message
            result = classify_plan_message(task, plan)

        assert result == "continue"  # fallback
