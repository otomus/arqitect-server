"""Brain routing — nerve role classification via LLM."""

from arqitect.brain.helpers import llm_generate, extract_json
from arqitect.brain.config import BRAIN_MODEL
from arqitect.brain.types import NerveRole


def classify_nerve_role(name: str, description: str) -> str:
    """Classify a nerve into a model category using the brain LLM.

    The LLM suggests the most appropriate role category (e.g. 'tool',
    'creative', 'code', or any other role it deems fitting).
    Unknown roles are handled gracefully by the nerve runtime, which
    falls back to the default nerve model.

    Returns a single-word role string.
    """
    prompt = (
        f"Classify this nerve agent into a single role category.\n\n"
        f"Nerve: {name}\n"
        f"Description: {description}\n\n"
        f"Common categories include tool, creative, code — but you may suggest "
        f"any role that best fits (e.g. data, research, media, orchestrator, etc.).\n\n"
        f"Reply with ONLY the single-word category name. Nothing else."
    )
    try:
        raw = llm_generate(BRAIN_MODEL, prompt).strip().lower()
        # Extract first word — LLM might add quotes, punctuation, or explanation
        role = raw.split()[0].strip('"\'.,:;!') if raw else NerveRole.TOOL
        return role if role else NerveRole.TOOL
    except Exception:
        return NerveRole.TOOL
