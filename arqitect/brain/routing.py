"""Brain routing — nerve role classification via LLM."""

from arqitect.brain.helpers import llm_generate, extract_json
from arqitect.brain.config import BRAIN_MODEL
from arqitect.types import NerveRole


_VALID_NERVE_ROLES = frozenset({NerveRole.TOOL, NerveRole.CREATIVE, NerveRole.CODE})


def validate_nerve_role(role: str) -> str:
    """Ensure a role string is one of the three valid NerveRole values.

    Returns the role if valid, otherwise falls back to NerveRole.TOOL.
    """
    return role if role in _VALID_NERVE_ROLES else NerveRole.TOOL


def classify_nerve_role(name: str, description: str) -> str:
    """Classify a nerve into one of the three valid nerve roles: tool, creative, or code.

    Uses the brain LLM to pick the best fit. Invalid or unexpected roles
    are clamped to 'tool' to prevent nerves from being assigned system
    roles (brain, awareness, etc.) that have different tuning configs.

    Returns a NerveRole string.
    """
    prompt = (
        f"Classify this nerve agent into exactly one of these three roles:\n"
        f"  tool — structured I/O, precise outputs, API calls, data processing\n"
        f"  creative — generative content, writing, brainstorming, reflection\n"
        f"  code — programming, syntax generation, code review\n\n"
        f"Nerve: {name}\n"
        f"Description: {description}\n\n"
        f"Reply with ONLY one word: tool, creative, or code."
    )
    try:
        raw = llm_generate(BRAIN_MODEL, prompt).strip().lower()
        role = raw.split()[0].strip('"\'.,:;!') if raw else NerveRole.TOOL
        return validate_nerve_role(role)
    except Exception:
        return NerveRole.TOOL
