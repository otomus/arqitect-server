"""Intent classifier — determines if a user message is a plan or a direct task.

Isolated from the brain's system prompt to avoid bias. Uses the brain model
with its own clean system prompt focused purely on intent detection.
"""

import json

from arqitect.brain.config import BRAIN_MODEL
from arqitect.brain.helpers import llm_generate, extract_json
from arqitect.types import IntentType


_INTENT_SYSTEM = (
    "You classify user messages into exactly two types:\n"
    "1. \"plan\" — the user wants a multi-step PROCESS: development, debugging, "
    "project setup, planning, migration, deployment, refactoring, investigation. "
    "These imply structured work that needs requirements gathering before execution.\n"
    "2. \"direct\" — everything else: greetings, questions, single requests, "
    "creative tasks, lookups, conversations, simple commands.\n\n"
    "If the type is \"plan\", also include a \"category\" field if obvious "
    "(e.g. \"development\", \"debugging\", \"setup\", \"planning\", \"migration\", "
    "\"deployment\", \"refactoring\").\n\n"
    "Reply with ONLY a JSON object. Examples:\n"
    '{"type": "direct"}\n'
    '{"type": "plan", "category": "development"}\n'
    '{"type": "plan", "category": "debugging"}\n'
    '{"type": "direct"}\n'
)


def classify_intent(task: str) -> dict:
    """Classify a user message as plan or direct.

    Args:
        task: The user's message text.

    Returns:
        {"type": "plan", "category": "..."} or {"type": "direct"}
    """
    prompt = f"Classify this message:\n\n{task}"
    raw = llm_generate(BRAIN_MODEL, prompt, system=_INTENT_SYSTEM)

    result = extract_json(raw)
    if result and result.get("type") in (IntentType.PLAN, IntentType.DIRECT):
        return result

    # Fallback: if LLM didn't produce valid JSON, default to direct
    # (conservative — don't hijack normal routing)
    return {"type": IntentType.DIRECT}
