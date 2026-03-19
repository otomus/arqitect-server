"""Brain system prompt construction.

Loads the brain routing prompt from the community adapter,
injecting runtime calibration and session context.
"""

import json

from arqitect.brain.config import r, mem, CORE_SENSES
from arqitect.brain.types import RedisKey


def _build_calibration_prompt_section() -> str:
    """Build the calibration status section for the system prompt."""
    lines = []
    for name in sorted(CORE_SENSES):
        try:
            raw = r.hget(RedisKey.SENSE_CALIBRATION, name)
            if raw:
                cal = json.loads(raw)
                status = cal.get("status", "unknown")
                caps = cal.get("capabilities", {})
                available = [k for k, v in caps.items() if v.get("available")]
                missing = [k for k, v in caps.items() if not v.get("available")]
                line = f"      - {name} [{status}]: {', '.join(available)}"
                if missing:
                    line += f" (missing: {', '.join(missing)})"
                lines.append(line)
            else:
                lines.append(f"      - {name} [unknown]: not yet calibrated")
        except Exception:
            lines.append(f"      - {name} [unknown]: calibration data unavailable")
    return "\n".join(lines) + "\n" if lines else ""


def _build_session_info() -> str:
    """Build the session context section."""
    session = mem.hot.get_session()
    if session.get("city"):
        return (
            f"\n\nKNOWN USER CONTEXT:\n"
            f"  Location: {session.get('city', '?')}, {session.get('country', '?')}\n"
            f"  Timezone: {session.get('timezone', '?')}\n"
        )
    return ""


def _build_few_shot_section(examples: list[dict]) -> str:
    """Format few-shot examples for injection into the prompt."""
    if not examples:
        return ""
    lines = ["\nExamples:"]
    for ex in examples:
        if isinstance(ex, dict) and "input" in ex and "output" in ex:
            lines.append(f'  User: "{ex["input"]}" → {ex["output"]}')
    return "\n".join(lines) + "\n"


def get_system_prompt() -> str:
    """Build the system prompt with session context.

    Loads from the community brain adapter. Injects runtime calibration
    and session info.
    """
    from arqitect.brain.adapters import resolve_prompt

    calibration_section = _build_calibration_prompt_section()
    session_info = _build_session_info()

    adapter = resolve_prompt("brain")
    if not adapter or not adapter.get("system_prompt"):
        raise RuntimeError(
            "Brain adapter not found. Run sync_all_adapters() at startup "
            "or ensure .community/cache/adapters/brain/core/context.json exists."
        )

    system_prompt = adapter["system_prompt"]
    examples = adapter.get("few_shot_examples", [])

    full_prompt = system_prompt
    if calibration_section:
        full_prompt += "\n\nCore senses (live calibration status):\n" + calibration_section
    full_prompt += _build_few_shot_section(examples)
    full_prompt += session_info
    return full_prompt
