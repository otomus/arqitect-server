"""Brain utilities — pure functions with no side effects (except llm_generate)."""

import difflib
import json
import logging
import re

from arqitect.brain.adapters import resolve_prompt as _resolve_adapter, get_temperature, get_max_tokens

logger = logging.getLogger(__name__)

# Error detection thresholds
SHORT_TEXT_LENGTH = 200
FACT_MATCH_THRESHOLD = 0.6

# Definitive error patterns — if any appears, the nerve output is an error.
_ERROR_PATTERNS = (
    "returned an error", "tool error", "mcp call error",
    "timed out", "could not complete", "no output", "failed to",
    "cannot fulfill", "no relevant tool", "outside my capabilities",
    "none of the available", "traceback (most recent",
)

# Softer error patterns — only checked for short outputs.
_SOFT_ERROR_PATTERNS = ("error:", "exception", "traceback", "not found", "not related", "not applicable")

# JSON keys that indicate a successful nerve result.
_SUCCESS_KEYS = ('"result"', '"answer"', '"output"', '"response"')


def llm_generate(model: str, prompt: str, system: str = "") -> str:
    """Generate text for a logical role via the per-role inference router."""
    from arqitect.inference.router import generate_for_role
    return generate_for_role(model, prompt, system=system)


def extract_json(raw: str) -> dict | None:
    """Extract the first JSON object from a string.

    Looks for a ###JSON: marker first (Chain-of-Thought output format).
    Falls back to scanning for '{' positions, longest-first, so braces
    inside string values don't confuse a naive depth counter.
    """
    marker = "###JSON:"
    marker_result = None
    # Use the LAST marker — earlier markers may come from echoed user input
    marker_idx = raw.rfind(marker)
    if marker_idx >= 0:
        marker_result = _extract_json_after_marker(raw[marker_idx + len(marker):].strip())

    scan_result = _extract_json_by_scanning(raw)

    # Prefer scanning result when it comes from after the marker's JSON,
    # since user-injected markers can appear earlier in echoed content
    if marker_result and scan_result:
        # If they're different objects, the scan result (last JSON) is safer
        if scan_result != marker_result:
            return scan_result
        return marker_result
    return marker_result or scan_result


def _extract_json_after_marker(text: str) -> dict | None:
    """Try to parse JSON from text found after a ###JSON: marker."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    if not text.startswith("{"):
        return None

    return _find_json_object(text, start=0)


def _extract_json_by_scanning(raw: str) -> dict | None:
    """Scan for '{' positions and return the last valid JSON object.

    Returns the last JSON object because the LLM's actual decision comes
    after any echoed user content. This prevents user-injected JSON
    earlier in the output from being picked up.
    """
    # Fast path: try the whole string
    try:
        result = json.loads(raw.strip())
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    last_result = None
    for start in (i for i, c in enumerate(raw) if c == "{"):
        result = _find_json_object(raw, start)
        if result is not None:
            last_result = result

    return last_result


def _find_json_object(text: str, start: int) -> dict | None:
    """Try to parse a JSON object starting at `start`, scanning closing braces backwards."""
    for end in range(len(text) - 1, start, -1):
        if text[end] != "}":
            continue
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def strip_markdown_fences(code: str) -> str:
    """Extract code from LLM output, removing markdown fences and trailing text."""
    text = code.strip()

    match = re.search(r'```[\w]*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def match_tool_name(model_name: str, available_tools: list[str]) -> str:
    """Fuzzy match a tool name from the model to actual available tools.

    Small models often hallucinate similar but wrong tool names.
    """
    if model_name in available_tools:
        return model_name

    model_lower = model_name.lower()
    for tool in available_tools:
        if model_lower in tool.lower() or tool.lower() in model_lower:
            return tool

    model_words = set(model_lower.replace("_", " ").split())
    best, best_score = None, 0
    for tool in available_tools:
        tool_words = set(tool.lower().replace("_", " ").split())
        overlap = len(model_words & tool_words)
        if overlap > best_score:
            best, best_score = tool, overlap

    return best if best_score > 0 else model_name


def _is_nerve_error(text: str) -> bool:
    """Detect if a nerve's response is an error using fast pattern matching."""
    if not text or not text.strip():
        return True

    text_lower = text.strip().lower()

    for pat in _ERROR_PATTERNS:
        if pat in text_lower:
            return True

    if text.strip().startswith("{") and any(k in text_lower for k in _SUCCESS_KEYS):
        return False

    if len(text) < SHORT_TEXT_LENGTH:
        for pat in _SOFT_ERROR_PATTERNS:
            if pat in text_lower:
                return True

    return False


def _graceful_failure_message(task: str, nerve_name: str) -> str:
    """Generate a personality-flavored failure message via the communication LLM."""
    prompt_text = _build_failure_prompt(task, nerve_name)
    system_text = _build_failure_system_prompt()

    if prompt_text is None:
        return "I wasn't able to complete that request. Could you try rephrasing?"

    try:
        from arqitect.inference.router import generate_for_role
        result = generate_for_role(
            "communication",
            prompt_text,
            system=system_text,
            max_tokens=get_max_tokens("communication"),
            temperature=get_temperature("communication"),
        ).strip()
        if result and not result.startswith("Error:") and len(result) > 5:
            return result
    except Exception:
        logger.debug("Failed to generate graceful failure message for nerve '%s'", nerve_name)

    return "I wasn't able to complete that request. Could you try rephrasing?"


def _build_failure_prompt(task: str, nerve_name: str) -> str | None:
    """Build the user-facing prompt for a failure message."""
    try:
        return f"The user asked: \"{task[:80]}\"\nThe nerve '{nerve_name}' failed to handle it."
    except Exception:
        return None


def _build_failure_system_prompt() -> str:
    """Build the system prompt for generating a failure message with personality."""
    try:
        from arqitect.senses.communication.nerve import (
            _load_personality_traits, _build_personality_instruction,
        )
        traits = _load_personality_traits()
        personality = _build_personality_instruction(traits)
    except Exception:
        logger.debug("Failed to load personality traits for failure message")
        personality = ""

    comm_adapter = _resolve_adapter("communication") or {}
    comm_sys = comm_adapter.get("system_prompt", "")

    return (
        f"{personality} {comm_sys} "
        "Generate a SHORT (1-2 sentence) failure message to the user. "
        "Be honest that you couldn't complete this request. "
        "Suggest they try rephrasing or provide more detail. "
        "Do NOT apologize excessively. "
        "Do NOT mention technical details (nerves, models, tools). "
        "Output ONLY the message."
    )


def _substitute_fact_values_brain(args_dict: dict, facts: dict, session: dict) -> dict:
    """Brain-side fuzzy substitution of garbled values in tool args."""
    pool = {**session, **facts}
    pool_values = [v for v in pool.values() if isinstance(v, str) and len(v) > 2]
    if not pool_values:
        return args_dict

    return {key: _find_best_fact_match(val, pool_values) for key, val in args_dict.items()}


def _find_best_fact_match(val: object, pool_values: list[str]) -> object:
    """Find the best fuzzy match for a value from known facts."""
    if not isinstance(val, str) or len(val) < 3:
        return val

    best_match, best_ratio = None, 0.0
    for fact_val in pool_values:
        for candidate in [fact_val] + fact_val.split(","):
            candidate = candidate.strip()
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, val.lower(), candidate.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = fact_val

    if best_ratio >= FACT_MATCH_THRESHOLD and best_match and best_match.lower() != val.lower():
        logger.info("[BRAIN] Fact-sub: '%s' -> '%s' (sim=%.2f)", val, best_match, best_ratio)
        return best_match

    return val
