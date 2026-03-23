"""Plan message router — classifies messages during an active plan.

When a user has an active plan, each incoming message is classified to determine
whether it continues the plan, is an unrelated aside, approves the plan, or
aborts it. This allows side questions to pass through to normal routing
without polluting the plan context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arqitect.brain.config import BRAIN_MODEL
from arqitect.brain.helpers import llm_generate, extract_json

if TYPE_CHECKING:
    from arqitect.brain.plan_session import PlanSession


_PLAN_CLASSIFY_SYSTEM = (
    "A user has an active work plan. Classify their new message.\n"
    "Reply with ONLY a JSON object.\n\n"
    "Types:\n"
    '- "continue" — message adds info, answers a question, or discusses the plan\n'
    '- "aside" — unrelated side question (weather, greeting, different topic)\n'
    '- "approve" — user is accepting/confirming the proposed plan\n'
    '- "abort" — user wants to cancel or abandon the plan\n\n'
    'Examples:\n'
    '{"action": "continue"}\n'
    '{"action": "approve"}\n'
    '{"action": "aside"}\n'
    '{"action": "abort"}\n'
)

_VALID_PLAN_ACTIONS = frozenset({"continue", "aside", "approve", "abort"})


def classify_plan_message(task: str, plan: PlanSession) -> str:
    """Classify whether a message relates to the active plan.

    Args:
        task: The user's new message.
        plan: The currently active PlanSession.

    Returns:
        One of 'continue', 'aside', 'approve', or 'abort'.
        Defaults to 'continue' if the LLM response is unparseable.
    """
    prompt = (
        f"Active plan goal: {plan.goal}\n"
        f"Current requirements: {plan.requirements}\n"
        f"Plan status: {plan.status}\n"
        f"New message: {task}\n"
    )
    raw = llm_generate(BRAIN_MODEL, prompt, system=_PLAN_CLASSIFY_SYSTEM)
    result = extract_json(raw)
    if result and result.get("action") in _VALID_PLAN_ACTIONS:
        return result["action"]
    # Default to continue — conservative, keeps plan context intact
    return "continue"
