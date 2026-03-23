"""Task Registry — tracks all tasks through their lifecycle for visibility and coordination.

Every task flows through: queued → active → (chain_running →) done | failed.
State changes are published to the ``TASK_REGISTRY`` channel so the dashboard
and other components can react in real time. Historical entries are persisted
to cold memory for post-mortem analysis.

Usage::

    from arqitect.brain.task_registry import get_registry

    reg = get_registry()
    reg.register("t-001", "turn on the lights", source="telegram", user_id="u42")
    reg.activate("t-001")
    reg.complete("t-001", result_preview="Lights turned on.")
"""

import json
import logging
import time
from enum import StrEnum
from typing import Any

from arqitect.brain.config import r, mem
from arqitect.brain.events import publish_event
from arqitect.types import Channel

logger = logging.getLogger(__name__)

# Redis hash where all task entries are stored
_TASKS_HASH = "synapse:tasks"

# Cold-memory namespace for historical task records
_HISTORY_NAMESPACE = "task_history"


class TaskStatus(StrEnum):
    """Possible states in a task's lifecycle."""
    QUEUED = "queued"
    ACTIVE = "active"
    CHAIN_RUNNING = "chain_running"
    DONE = "done"
    FAILED = "failed"


class TaskRegistry:
    """Singleton registry that tracks every task from creation to completion.

    Each task is stored as a JSON blob inside the Redis hash ``synapse:tasks``
    (keyed by *task_id*).  Completed / failed tasks are also written to cold
    memory so they survive Redis flushes.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        task_id: str,
        task: str,
        source: str = "",
        user_id: str = "",
    ) -> dict[str, Any]:
        """Register a new task and mark it as queued.

        Args:
            task_id: Unique identifier for the task.
            task: Human-readable task description / prompt text.
            source: Originating connector (e.g. ``"telegram"``, ``"web"``).
            user_id: ID of the user who submitted the task.

        Returns:
            The full task entry dict that was persisted.
        """
        entry = _build_entry(
            task_id=task_id,
            task=task,
            source=source,
            user_id=user_id,
            status=TaskStatus.QUEUED,
        )
        self._persist(task_id, entry)
        self._publish(task_id, entry, event="registered")
        logger.info("Task registered: %s (source=%s, user=%s)", task_id, source, user_id)
        return entry

    def activate(self, task_id: str) -> dict[str, Any] | None:
        """Transition a task to *active* status.

        Args:
            task_id: The task to activate.

        Returns:
            Updated entry, or ``None`` if *task_id* is unknown.
        """
        entry = self._load(task_id)
        if entry is None:
            logger.warning("activate called for unknown task %s", task_id)
            return None
        entry["status"] = TaskStatus.ACTIVE
        entry["activated_at"] = time.time()
        self._persist(task_id, entry)
        self._publish(task_id, entry, event="activated")
        return entry

    def update_chain_progress(
        self,
        task_id: str,
        step_index: int,
        total_steps: int,
    ) -> dict[str, Any] | None:
        """Record progress within a chained-nerve execution.

        Args:
            task_id: The task currently running a chain.
            step_index: Zero-based index of the current step.
            total_steps: Total number of steps in the chain.

        Returns:
            Updated entry, or ``None`` if *task_id* is unknown.
        """
        entry = self._load(task_id)
        if entry is None:
            logger.warning("update_chain_progress called for unknown task %s", task_id)
            return None
        entry["status"] = TaskStatus.CHAIN_RUNNING
        entry["chain_step"] = step_index
        entry["chain_total"] = total_steps
        self._persist(task_id, entry)
        self._publish(task_id, entry, event="chain_progress")
        return entry

    def complete(
        self,
        task_id: str,
        result_preview: str = "",
    ) -> dict[str, Any] | None:
        """Mark a task as successfully completed.

        Args:
            task_id: The task to complete.
            result_preview: Short summary of the result (for dashboard display).

        Returns:
            Updated entry, or ``None`` if *task_id* is unknown.
        """
        entry = self._load(task_id)
        if entry is None:
            logger.warning("complete called for unknown task %s", task_id)
            return None
        entry["status"] = TaskStatus.DONE
        entry["completed_at"] = time.time()
        entry["result_preview"] = result_preview[:200]
        self._persist(task_id, entry)
        self._publish(task_id, entry, event="completed")
        self._archive(task_id, entry)
        logger.info("Task completed: %s", task_id)
        return entry

    def fail(
        self,
        task_id: str,
        reason: str = "",
    ) -> dict[str, Any] | None:
        """Mark a task as failed.

        Args:
            task_id: The task that failed.
            reason: Human-readable failure reason.

        Returns:
            Updated entry, or ``None`` if *task_id* is unknown.
        """
        entry = self._load(task_id)
        if entry is None:
            logger.warning("fail called for unknown task %s", task_id)
            return None
        entry["status"] = TaskStatus.FAILED
        entry["failed_at"] = time.time()
        entry["failure_reason"] = reason[:500]
        self._persist(task_id, entry)
        self._publish(task_id, entry, event="failed")
        self._archive(task_id, entry)
        logger.warning("Task failed: %s — %s", task_id, reason[:120])
        return entry

    def get_active(self) -> list[dict[str, Any]]:
        """Return all tasks with *active* or *chain_running* status.

        Returns:
            List of task entry dicts, sorted by creation time (oldest first).
        """
        active_statuses = {TaskStatus.ACTIVE, TaskStatus.CHAIN_RUNNING}
        return [
            entry
            for entry in self._load_all()
            if entry.get("status") in active_statuses
        ]

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent tasks across all statuses.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of task entry dicts, sorted newest-first.
        """
        entries = self._load_all()
        entries.sort(key=lambda e: e.get("created_at", 0), reverse=True)
        return entries[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self, task_id: str, entry: dict[str, Any]) -> None:
        """Write a task entry to the Redis hash."""
        try:
            r.hset(_TASKS_HASH, task_id, json.dumps(entry))
        except Exception as exc:
            logger.error("Failed to persist task %s: %s", task_id, exc)

    def _load(self, task_id: str) -> dict[str, Any] | None:
        """Load a single task entry from Redis."""
        try:
            raw = r.hget(_TASKS_HASH, task_id)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.error("Failed to load task %s: %s", task_id, exc)
            return None

    def _load_all(self) -> list[dict[str, Any]]:
        """Load all task entries from Redis."""
        try:
            raw_map = r.hgetall(_TASKS_HASH)
            return [json.loads(v) for v in raw_map.values()]
        except Exception as exc:
            logger.error("Failed to load tasks: %s", exc)
            return []

    def _publish(self, task_id: str, entry: dict[str, Any], event: str) -> None:
        """Publish a task state-change event to the synaptic bus."""
        try:
            publish_event(Channel.TASK_REGISTRY, {
                "event": event,
                "task_id": task_id,
                **entry,
            })
        except Exception as exc:
            logger.error("Failed to publish task event %s/%s: %s", event, task_id, exc)

    def _archive(self, task_id: str, entry: dict[str, Any]) -> None:
        """Write a terminal task entry to cold memory for long-term history."""
        try:
            mem.cold.set_fact(
                _HISTORY_NAMESPACE,
                task_id,
                json.dumps(entry),
            )
        except Exception as exc:
            # Non-fatal — the Redis hash is the primary store
            logger.warning("Failed to archive task %s to cold memory: %s", task_id, exc)


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_registry: TaskRegistry | None = None


def get_registry() -> TaskRegistry:
    """Return the singleton TaskRegistry instance.

    Creates it on first call. Thread-safe because Python's GIL protects
    the simple ``is None`` check and assignment for this use case (same
    pattern as ``get_consolidator`` in consolidate.py).
    """
    global _registry
    if _registry is None:
        _registry = TaskRegistry()
    return _registry


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_entry(
    task_id: str,
    task: str,
    source: str,
    user_id: str,
    status: TaskStatus,
) -> dict[str, Any]:
    """Construct a fresh task entry dict with sensible defaults."""
    return {
        "task_id": task_id,
        "task": task,
        "source": source,
        "user_id": user_id,
        "status": status,
        "created_at": time.time(),
        "activated_at": None,
        "completed_at": None,
        "failed_at": None,
        "chain_step": None,
        "chain_total": None,
        "result_preview": "",
        "failure_reason": "",
    }
