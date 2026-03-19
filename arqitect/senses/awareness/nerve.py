"""Sense: Awareness — self-identity, capabilities, ethical boundaries, moral compass.

Modes:
  - Permission check: {"action": "delete", "context": "path=/etc/hosts"} → allow/deny
  - Self-reflection: {"query": "who are you?"} → identity, capabilities, active nerves/tools
"""

import json
import os
import sqlite3
import sys

_SENSE_DIR = os.path.dirname(os.path.abspath(__file__))
from arqitect.config.loader import get_project_root, get_sandbox_dir as _get_sandbox_dir
_PROJECT_ROOT = str(get_project_root())
_RULES_PATH = os.path.join(_SENSE_DIR, "awareness_rules.json")
_PERSONALITY_PATH = os.path.join(_PROJECT_ROOT, "personality.json")
_COLD_DB_PATH = os.path.join(str(get_project_root()), "memory", "knowledge.db")

SENSE_NAME = "awareness"
def _load_adapter_description() -> str:
    try:
        from arqitect.brain.adapters import get_description
        desc = get_description("awareness")
        if desc:
            return desc
    except Exception:
        pass
    return "Sentient's own identity and persona — who Sentient is, what it can do, its personality."

DESCRIPTION = _load_adapter_description()


def _load_rules() -> dict:
    """Load awareness rules from co-located JSON file."""
    try:
        with open(_RULES_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "identity": {"name": "Arqitect", "type": "Autonomous AI nervous system"},
            "never_delete": [],
            "never_execute": [],
            "require_confirmation": [],
            "boundaries": [],
        }


def _load_personality() -> dict:
    """Load personality seed from co-located JSON file, overlaid with cold memory evolution."""
    seed = {}
    try:
        with open(_PERSONALITY_PATH, "r") as f:
            seed = json.load(f)
    except Exception:
        seed = {
            "core_identity": {"name": "Arqitect", "archetype": "Sharp, warm, and resourceful AI with quiet confidence"},
            "voice": {"default_tone": "warm-direct", "humor_style": "dry wit when natural, never forced"},
            "trait_weights": {"wit": 0.5, "swagger": 0.3, "warmth": 0.7, "formality": 0.3},
        }

    # Overlay evolved traits from cold memory (personality facts)
    if os.path.exists(_COLD_DB_PATH):
        try:
            conn = sqlite3.connect(_COLD_DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT key, value FROM facts WHERE category='personality'"
            ).fetchall()
            conn.close()
            for row in rows:
                key, val = row["key"], row["value"]
                if key == "trait_weights":
                    try:
                        seed["trait_weights"] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif key == "humor_style":
                    seed["voice"]["humor_style"] = val
                elif key == "learned_preferences":
                    try:
                        seed["learned_preferences"] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
        except Exception:
            pass

    return seed


def _load_inventory() -> dict:
    """Read cold memory for live inventory of nerves, senses, and tools."""
    inventory = {"nerves": [], "senses": [], "tools": []}
    if not os.path.exists(_COLD_DB_PATH):
        return inventory
    try:
        conn = sqlite3.connect(_COLD_DB_PATH)
        conn.row_factory = sqlite3.Row
        # Nerves
        rows = conn.execute("SELECT name, description, is_sense FROM nerve_registry").fetchall()
        for r in rows:
            entry = {"name": r["name"], "description": r["description"]}
            if r["is_sense"]:
                inventory["senses"].append(entry)
            else:
                inventory["nerves"].append(entry)
        # Tools
        rows = conn.execute("SELECT name, total_calls, successes FROM tool_stats").fetchall()
        for r in rows:
            inventory["tools"].append({
                "name": r["name"],
                "total_calls": r["total_calls"],
                "successes": r["successes"],
            })
        conn.close()
    except Exception:
        pass
    return inventory


def check_permission(action: str, context: str, rules: dict) -> dict:
    """Check if an action is allowed based on awareness rules."""
    action_lower = action.lower().strip()
    context_lower = context.lower().strip()

    # Check never_execute
    for forbidden in rules.get("never_execute", []):
        if forbidden.lower() in context_lower or forbidden.lower() in action_lower:
            return {
                "allowed": False,
                "denied": True,
                "reason": f"Action matches forbidden pattern: '{forbidden}'",
            }

    # Check never_delete — for delete/remove actions on protected paths
    if action_lower in ("delete", "remove", "rm", "unlink"):
        for protected in rules.get("never_delete", []):
            protected_expanded = os.path.expanduser(protected)
            if protected.lower() in context_lower or protected_expanded in context:
                return {
                    "allowed": False,
                    "denied": True,
                    "reason": f"Path '{context}' is protected (matches '{protected}')",
                }

    # Check require_confirmation
    for keyword in rules.get("require_confirmation", []):
        if keyword.lower() in action_lower or keyword.lower() in context_lower:
            return {
                "allowed": True,
                "requires_confirmation": True,
                "reason": f"Action '{action}' requires user confirmation (matches '{keyword}')",
            }

    return {"allowed": True, "denied": False}


def _personality_voice(personality: dict) -> str:
    """Build a short voice description string from personality data."""
    core = personality.get("core_identity", {})
    voice = personality.get("voice", {})
    # voice can be a plain string (from scaffolder) or a dict (from evolved personality)
    if isinstance(voice, str):
        return voice
    traits = core.get("fixed_traits", [])
    archetype = core.get("archetype", "")
    humor = voice.get("humor_style", "")
    parts = []
    if archetype:
        parts.append(archetype)
    if humor:
        parts.append(f"humor: {humor}")
    if traits:
        parts.append("; ".join(traits[:3]))
    return ". ".join(parts)


def _load_user_profile() -> dict:
    """Load user profile from SYNAPSE env vars and cold memory."""
    profile = {}
    # From env var (set by brain before invocation)
    raw = os.environ.get("SYNAPSE_USER_PROFILE", "")
    if raw:
        try:
            profile = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    # Supplement from cold memory user facts
    if os.path.exists(_COLD_DB_PATH):
        try:
            conn = sqlite3.connect(_COLD_DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT key, value FROM facts WHERE category='user'"
            ).fetchall()
            conn.close()
            for row in rows:
                if row["key"] not in profile:
                    profile[row["key"]] = row["value"]
        except Exception:
            pass
    return profile


def _total_invocations() -> int:
    """Count total task invocations across all nerves."""
    if not os.path.exists(_COLD_DB_PATH):
        return 0
    try:
        conn = sqlite3.connect(_COLD_DB_PATH)
        row = conn.execute("SELECT COALESCE(SUM(total_invocations), 0) FROM nerve_registry").fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0


def _load_awareness_adapter() -> dict | None:
    """Load the awareness adapter prompt from community cache."""
    try:
        from arqitect.brain.adapters import resolve_prompt
        return resolve_prompt("awareness")
    except Exception:
        return None


def _llm_respond(system_prompt: str, user_prompt: str) -> str:
    """Call the brain model with the given prompts. Returns empty string on failure."""
    try:
        from arqitect.inference.router import generate_for_role
        result = generate_for_role("brain", user_prompt, system=system_prompt, max_tokens=256)
        if result and not result.startswith("Error:"):
            return result.strip()
    except Exception:
        pass
    return ""


def self_reflect(query: str, rules: dict) -> dict:
    """Answer questions about identity, capabilities, and boundaries.

    Uses the community awareness adapter prompt with the brain model to generate
    natural, personality-infused responses. Falls back to structured data if
    no LLM is available.
    """
    personality = _load_personality()
    voice_desc = _personality_voice(personality)
    core = personality.get("core_identity", {})
    identity = rules.get("identity", {})
    name = personality.get("name") or core.get("name") or identity.get("name", "Arqitect")

    # Build context for LLM
    user_profile = _load_user_profile()
    inventory = _load_inventory()
    total_tasks = _total_invocations()

    context_parts = [
        f"Your name: {name}",
        f"Your voice: {voice_desc}",
        f"Capabilities: see and analyze images, hear and speak, read/write files, run commands, fetch web data, do math, generate code, build new tools on the fly",
        f"Nerves active: {len(inventory.get('nerves', []))}",
        f"Tasks completed: {total_tasks}",
    ]
    if user_profile:
        profile_str = ", ".join(f"{k}: {v}" for k, v in user_profile.items())
        context_parts.append(f"Known about user: {profile_str}")

    context = "\n".join(context_parts)

    # Load community adapter system prompt, fall back to hardcoded
    adapter = _load_awareness_adapter()
    if adapter and adapter.get("system_prompt"):
        system_prompt = adapter["system_prompt"]
        # Inject runtime identity
        system_prompt += f"\n\nYour name is {name}. Your voice: {voice_desc}."
    else:
        system_prompt = (
            f"You are {name}. "
            f"Your voice: {voice_desc}. "
            f"Respond like a person, not a bot. Be concise (1-2 sentences max). "
            f"Never expose internal details like nerve/tool/sense counts. "
            f"Never say 'I am an AI' or 'as an AI'. Just be natural."
        )

    user_prompt = f"Context:\n{context}\n\nUser said: {query}"

    # Try LLM response
    response = _llm_respond(system_prompt, user_prompt)
    if response:
        return {
            "intent": "awareness",
            "name": name,
            "voice": voice_desc,
            "response": response,
        }

    # Fallback: structured data (no LLM available)
    return {
        "intent": "awareness",
        "name": name,
        "voice": voice_desc,
        "user_message": query,
    }


def calibrate() -> dict:
    """Probe awareness sense capabilities."""
    from arqitect.senses.calibration_protocol import build_result, save_calibration

    rules_exist = os.path.isfile(_RULES_PATH)
    cold_db_exists = os.path.isfile(_COLD_DB_PATH)

    capabilities = {
        "permission_check": {
            "available": True,
            "provider": "awareness_rules",
            "notes": "" if rules_exist else "Using default rules (awareness_rules.json not found)",
        },
        "self_reflection": {
            "available": True,
            "provider": "awareness_rules + cold_db",
            "notes": "",
        },
        "inventory": {
            "available": cold_db_exists,
            "provider": "cold_db",
            "notes": "" if cold_db_exists else "Cold DB not found — inventory unavailable until first boot",
        },
    }

    deps = {
        "awareness_rules.json": {"installed": rules_exist, "path": _RULES_PATH if rules_exist else ""},
        "knowledge.db": {"installed": cold_db_exists, "path": _COLD_DB_PATH if cold_db_exists else ""},
    }

    result = build_result(SENSE_NAME, capabilities, deps)
    save_calibration(_SENSE_DIR, result)
    return result


def main():
    raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "{}"
    try:
        input_data = json.loads(raw)
    except json.JSONDecodeError:
        input_data = {"query": raw}

    # Calibration mode
    if input_data.get("mode") == "calibrate":
        print(json.dumps(calibrate()))
        return

    rules = _load_rules()

    # Permission check mode
    if "action" in input_data and input_data.get("action") not in ("reflect", "query"):
        result = check_permission(
            input_data["action"],
            input_data.get("context", ""),
            rules,
        )
        result["sense"] = SENSE_NAME
        print(json.dumps(result))
        return

    # Self-reflection mode
    query = input_data.get("query", "")
    if not query:
        query = input_data.get("message", "who are you?")
    result = self_reflect(query, rules)
    result["sense"] = SENSE_NAME
    print(json.dumps(result))


if __name__ == "__main__":
    main()
