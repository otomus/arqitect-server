"""Tests for arqitect.brain.plan_session — PlanSession lifecycle.

Covers:
- Create → add_requirement → propose → approve → to_recipe lifecycle
- Status transition validation (reject invalid transitions)
- Serialization round-trip (to_dict / from_dict)
- Redis persistence (save / get_active / delete)
- Cold memory archival and past plan recall
- Abandon flow
"""

import json
from unittest.mock import patch

import pytest

from arqitect.brain.plan_session import PlanSession
from tests.conftest import FakeLLM


@pytest.mark.timeout(10)
class TestPlanSessionLifecycle:
    """Full lifecycle: create -> gather -> propose -> approve -> to_recipe."""

    def test_create_initializes_gathering(self):
        """New plan starts in 'gathering' status with empty collections."""
        plan = PlanSession.create("user-1", "build an API", "development")

        assert plan.status == "gathering"
        assert plan.user_id == "user-1"
        assert plan.goal == "build an API"
        assert plan.category == "development"
        assert plan.requirements == []
        assert plan.steps is None
        assert plan.conversation_context == []
        assert plan.plan_id  # non-empty UUID

    def test_add_requirement(self):
        """Requirements accumulate from conversation."""
        plan = PlanSession.create("user-1", "build an API", "development")

        plan.add_requirement("Use FastAPI framework")
        plan.add_requirement("Need PostgreSQL database")

        assert len(plan.requirements) == 2
        assert "Use FastAPI framework" in plan.requirements

    def test_add_message(self):
        """Messages track plan-specific conversation."""
        plan = PlanSession.create("user-1", "build an API", "development")

        plan.add_message("user", "I want REST endpoints")
        plan.add_message("assistant", "Got it. What database?")

        assert len(plan.conversation_context) == 2
        assert plan.conversation_context[0]["role"] == "user"
        assert plan.conversation_context[1]["role"] == "assistant"
        assert "timestamp" in plan.conversation_context[0]

    def test_propose_transitions_to_proposed(self):
        """propose() sets steps and transitions status to 'proposed'."""
        plan = PlanSession.create("user-1", "build an API", "development")
        steps = [
            {"step": 1, "description": "Set up project"},
            {"step": 2, "description": "Create endpoints"},
        ]

        plan.propose(steps)

        assert plan.status == "proposed"
        assert plan.steps == steps

    def test_propose_rejects_non_gathering_status(self):
        """Cannot propose when not in 'gathering' status."""
        plan = PlanSession.create("user-1", "build an API", "development")
        plan.propose([{"step": 1, "description": "do thing"}])

        with pytest.raises(ValueError, match="Cannot propose"):
            plan.propose([{"step": 1, "description": "another thing"}])

    def test_approve_transitions_to_approved(self):
        """approve() transitions from 'proposed' to 'approved'."""
        plan = PlanSession.create("user-1", "build an API", "development")
        plan.propose([{"step": 1, "description": "Set up"}])

        plan.approve()

        assert plan.status == "approved"

    def test_approve_rejects_non_proposed_status(self):
        """Cannot approve when not in 'proposed' status."""
        plan = PlanSession.create("user-1", "build an API", "development")

        with pytest.raises(ValueError, match="Cannot approve"):
            plan.approve()

    def test_to_recipe_produces_valid_recipe(self):
        """to_recipe() converts approved plan to a recipe dict."""
        plan = PlanSession.create("user-1", "build an API", "development")
        plan.add_requirement("Use FastAPI")
        steps = [{"step": 1, "description": "Set up project"}]
        plan.propose(steps)
        plan.approve()

        recipe = plan.to_recipe()

        assert recipe["goal"] == "build an API"
        assert recipe["steps"] == steps
        assert recipe["category"] == "development"
        assert "Use FastAPI" in recipe["context"]["requirements"]

    def test_to_recipe_rejects_non_approved(self):
        """Cannot convert to recipe unless approved."""
        plan = PlanSession.create("user-1", "build an API", "development")

        with pytest.raises(ValueError, match="Cannot convert"):
            plan.to_recipe()

    def test_to_recipe_rejects_no_steps(self):
        """Cannot convert to recipe if steps are None (should not happen)."""
        plan = PlanSession.create("user-1", "build an API", "development")
        # Force status without going through propose
        plan.status = "approved"

        with pytest.raises(ValueError, match="no steps"):
            plan.to_recipe()

    def test_complete_marks_done(self):
        """complete(success=True) marks status as 'done'."""
        plan = PlanSession.create("user-1", "build an API", "development")

        plan.complete(success=True)

        assert plan.status == "done"

    def test_complete_failure_marks_abandoned(self):
        """complete(success=False) marks status as 'abandoned'."""
        plan = PlanSession.create("user-1", "build an API", "development")

        plan.complete(success=False)

        assert plan.status == "abandoned"

    def test_abandon(self):
        """abandon() marks status as 'abandoned'."""
        plan = PlanSession.create("user-1", "build an API", "development")

        plan.abandon()

        assert plan.status == "abandoned"


@pytest.mark.timeout(10)
class TestPlanSessionSerialization:
    """Round-trip serialization: to_dict / from_dict."""

    def test_round_trip(self):
        """to_dict -> from_dict produces equivalent plan."""
        plan = PlanSession.create("user-1", "build an API", "development")
        plan.add_requirement("Use FastAPI")
        plan.add_message("user", "I want REST endpoints")
        plan.propose([{"step": 1, "description": "Set up"}])

        restored = PlanSession.from_dict(plan.to_dict())

        assert restored.plan_id == plan.plan_id
        assert restored.user_id == plan.user_id
        assert restored.goal == plan.goal
        assert restored.category == plan.category
        assert restored.requirements == plan.requirements
        assert restored.steps == plan.steps
        assert restored.status == plan.status
        assert restored.conversation_context == plan.conversation_context

    def test_to_dict_is_json_serializable(self):
        """to_dict() output can be serialized to JSON."""
        plan = PlanSession.create("user-1", "build an API", "development")
        plan.project_facts = {"language": "python", "framework": "fastapi"}

        serialized = json.dumps(plan.to_dict())
        restored = json.loads(serialized)

        assert restored["goal"] == "build an API"
        assert restored["project_facts"]["language"] == "python"


@pytest.mark.timeout(10)
class TestPlanSessionRedis:
    """Redis persistence: save / get_active / delete."""

    def test_save_and_get_active(self, test_redis):
        """save() then get_active() returns the plan."""
        plan = PlanSession.create("user-1", "build an API", "development")
        plan.save(test_redis)

        loaded = PlanSession.get_active("user-1", test_redis)

        assert loaded is not None
        assert loaded.plan_id == plan.plan_id
        assert loaded.goal == plan.goal

    def test_get_active_returns_none_when_empty(self, test_redis):
        """get_active() returns None when no plan exists."""
        result = PlanSession.get_active("user-nonexistent", test_redis)

        assert result is None

    def test_delete_removes_plan(self, test_redis):
        """delete() removes the plan from Redis."""
        plan = PlanSession.create("user-1", "build an API", "development")
        plan.save(test_redis)
        plan.delete(test_redis)

        result = PlanSession.get_active("user-1", test_redis)

        assert result is None

    def test_get_active_handles_corrupted_data(self, test_redis):
        """get_active() returns None for corrupted Redis data."""
        test_redis.set("synapse:plan:user-1", "not valid json {{{")

        result = PlanSession.get_active("user-1", test_redis)

        assert result is None


@pytest.mark.timeout(10)
class TestPlanSessionArchival:
    """Cold memory archival and past plan recall."""

    def test_archive_stores_in_cold_memory(self, test_redis, tmp_memory_dir, mem):
        """archive() persists plan to cold memory."""
        plan = PlanSession.create("user-1", "build an API", "development")
        plan.complete(success=True)

        plan.archive(mem)

        stored = mem.cold.get_fact("plan_history", plan.plan_id)
        assert stored is not None
        data = json.loads(stored)
        assert data["goal"] == "build an API"
        assert data["status"] == "done"

    def test_get_past_plans_filters_by_category(self, test_redis, tmp_memory_dir, mem):
        """get_past_plans() returns only plans matching the category."""
        # Create and archive two plans in different categories
        plan_dev = PlanSession.create("user-1", "build API", "development")
        plan_dev.complete(success=True)
        plan_dev.archive(mem)

        plan_debug = PlanSession.create("user-1", "fix bug", "debugging")
        plan_debug.complete(success=True)
        plan_debug.archive(mem)

        results = PlanSession.get_past_plans(mem, "development")

        assert len(results) == 1
        assert results[0]["goal"] == "build API"

    def test_get_past_plans_excludes_non_done(self, test_redis, tmp_memory_dir, mem):
        """get_past_plans() only returns plans with 'done' status."""
        plan = PlanSession.create("user-1", "abandoned project", "development")
        plan.abandon()
        plan.archive(mem)

        results = PlanSession.get_past_plans(mem, "development")

        assert len(results) == 0

    def test_get_past_plans_respects_limit(self, test_redis, tmp_memory_dir, mem):
        """get_past_plans() respects the limit parameter."""
        for i in range(5):
            plan = PlanSession.create("user-1", f"project {i}", "development")
            plan.complete(success=True)
            plan.archive(mem)

        results = PlanSession.get_past_plans(mem, "development", limit=2)

        assert len(results) == 2

    def test_get_past_plans_empty_when_no_history(self, test_redis, tmp_memory_dir, mem):
        """get_past_plans() returns empty list when no plans exist."""
        results = PlanSession.get_past_plans(mem, "development")

        assert results == []
