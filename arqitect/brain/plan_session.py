"""PlanSession — Redis-backed planning state, one active plan per user.

A PlanSession tracks a multi-step task through its lifecycle:
gathering -> proposed -> approved -> executing -> done | abandoned.

Active plans are stored in Redis (fast lookup). Completed plans are archived
to cold memory for future reference.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from arqitect.types import RedisKey

if TYPE_CHECKING:
    from arqitect.memory import MemoryManager
    from redis import Redis


# Valid plan statuses in lifecycle order
PLAN_STATUSES = frozenset({
    "gathering", "proposed", "approved", "executing", "done", "abandoned",
})

_PLAN_HISTORY_CATEGORY = "plan_history"


@dataclass
class PlanSession:
    """A single planning session for a user.

    Fields:
        plan_id: Unique identifier for this plan.
        user_id: Owner of the plan.
        goal: High-level description, evolves as conversation clarifies.
        category: Task category (development, debugging, setup, etc.).
        requirements: Accumulated requirements from conversation.
        steps: None until plan is finalized, then a list of step dicts.
        status: Current lifecycle phase.
        project_facts: Detected project context, if any.
        related_plans: Past plans pulled for context.
        related_episodes: Past episodes pulled for context.
        matched_recipe: Existing recipe if one was matched.
        conversation_context: Plan-related messages only (not side messages).
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last modification.
    """

    plan_id: str
    user_id: str
    goal: str
    category: str
    requirements: list[str] = field(default_factory=list)
    steps: list[dict] | None = None
    status: str = "gathering"
    project_facts: dict | None = None
    related_plans: list[dict] = field(default_factory=list)
    related_episodes: list[dict] = field(default_factory=list)
    matched_recipe: dict | None = None
    conversation_context: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # --- Mutation methods ---

    def add_requirement(self, text: str) -> None:
        """Append a requirement gathered from conversation.

        Args:
            text: The requirement text to add.
        """
        self.requirements.append(text)
        self.updated_at = time.time()

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the plan-specific conversation context.

        Args:
            role: Message role ('user' or 'assistant').
            content: Message content.
        """
        self.conversation_context.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self.updated_at = time.time()

    def propose(self, steps: list[dict]) -> None:
        """Transition plan to 'proposed' with the given steps.

        Args:
            steps: List of step dicts describing the plan.

        Raises:
            ValueError: If plan is not in 'gathering' status.
        """
        if self.status != "gathering":
            raise ValueError(f"Cannot propose plan in '{self.status}' status")
        self.steps = steps
        self.status = "proposed"
        self.updated_at = time.time()

    def approve(self) -> None:
        """Transition plan from 'proposed' to 'approved'.

        Raises:
            ValueError: If plan is not in 'proposed' status.
        """
        if self.status != "proposed":
            raise ValueError(f"Cannot approve plan in '{self.status}' status")
        self.status = "approved"
        self.updated_at = time.time()

    def to_recipe(self) -> dict:
        """Convert an approved plan to a recipe decision for chain execution.

        Returns:
            A recipe dict with 'goal', 'steps', 'category', and 'context'.

        Raises:
            ValueError: If plan is not in 'approved' status or has no steps.
        """
        if self.status != "approved":
            raise ValueError(f"Cannot convert plan to recipe in '{self.status}' status")
        if not self.steps:
            raise ValueError("Plan has no steps to convert to recipe")
        return {
            "goal": self.goal,
            "steps": self.steps,
            "category": self.category,
            "context": {
                "requirements": self.requirements,
                "project_facts": self.project_facts,
                "conversation": self.conversation_context,
            },
        }

    def complete(self, success: bool = True) -> None:
        """Mark plan as done.

        Args:
            success: Whether the plan completed successfully.
        """
        self.status = "done" if success else "abandoned"
        self.updated_at = time.time()

    def abandon(self) -> None:
        """Mark plan as abandoned."""
        self.status = "abandoned"
        self.updated_at = time.time()

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Serialize the plan session to a JSON-safe dict.

        Returns:
            Dict representation of all plan fields.
        """
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "goal": self.goal,
            "category": self.category,
            "requirements": self.requirements,
            "steps": self.steps,
            "status": self.status,
            "project_facts": self.project_facts,
            "related_plans": self.related_plans,
            "related_episodes": self.related_episodes,
            "matched_recipe": self.matched_recipe,
            "conversation_context": self.conversation_context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanSession:
        """Deserialize a plan session from a dict.

        Args:
            data: Dict with plan fields (as produced by to_dict).

        Returns:
            Reconstructed PlanSession instance.
        """
        return cls(
            plan_id=data["plan_id"],
            user_id=data["user_id"],
            goal=data["goal"],
            category=data.get("category", ""),
            requirements=data.get("requirements", []),
            steps=data.get("steps"),
            status=data.get("status", "gathering"),
            project_facts=data.get("project_facts"),
            related_plans=data.get("related_plans", []),
            related_episodes=data.get("related_episodes", []),
            matched_recipe=data.get("matched_recipe"),
            conversation_context=data.get("conversation_context", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

    # --- Redis persistence ---

    def save(self, redis_client: Redis) -> None:
        """Persist the active plan to Redis.

        Args:
            redis_client: Redis client instance.
        """
        key = f"{RedisKey.PLAN}:{self.user_id}"
        redis_client.set(key, json.dumps(self.to_dict()))

    def delete(self, redis_client: Redis) -> None:
        """Remove the active plan from Redis.

        Args:
            redis_client: Redis client instance.
        """
        key = f"{RedisKey.PLAN}:{self.user_id}"
        redis_client.delete(key)

    def archive(self, mem: MemoryManager) -> None:
        """Archive the plan to cold memory for future reference.

        Args:
            mem: MemoryManager instance for cold storage.
        """
        mem.cold.set_fact(
            _PLAN_HISTORY_CATEGORY,
            self.plan_id,
            json.dumps(self.to_dict()),
        )

    # --- Static / class methods ---

    @staticmethod
    def create(user_id: str, goal: str, category: str) -> PlanSession:
        """Create a new plan session in the gathering phase.

        Args:
            user_id: The user who owns this plan.
            goal: Initial goal description from the user's message.
            category: Task category from intent classification.

        Returns:
            A new PlanSession in 'gathering' status.
        """
        now = time.time()
        return PlanSession(
            plan_id=str(uuid.uuid4()),
            user_id=user_id,
            goal=goal,
            category=category,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def get_active(user_id: str, redis_client: Redis) -> PlanSession | None:
        """Load the active plan for a user from Redis.

        Args:
            user_id: The user to look up.
            redis_client: Redis client instance.

        Returns:
            The active PlanSession, or None if no active plan exists.
        """
        key = f"{RedisKey.PLAN}:{user_id}"
        raw = redis_client.get(key)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return PlanSession.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def get_past_plans(
        mem: MemoryManager,
        category: str,
        limit: int = 3,
    ) -> list[dict]:
        """Query cold memory for completed plans in the same category.

        Args:
            mem: MemoryManager instance for cold storage.
            category: Category to filter by.
            limit: Maximum number of past plans to return.

        Returns:
            List of plan dicts, sorted by recency, filtered to 'done' status.
        """
        all_plans = mem.cold.get_facts(_PLAN_HISTORY_CATEGORY)
        candidates = []
        for _plan_id, raw in all_plans.items():
            try:
                plan = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if plan.get("category") == category and plan.get("status") == "done":
                candidates.append(plan)
        candidates.sort(key=lambda p: p.get("updated_at", 0), reverse=True)
        return candidates[:limit]
