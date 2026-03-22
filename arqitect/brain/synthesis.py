"""Brain synthesis — nerve creation, tool fabrication, and nerve pruning."""

import json
import logging
import os
import re
import shutil
import textwrap
import threading

from arqitect.brain.config import (
    BRAIN_MODEL, CODE_MODEL, NERVES_DIR, MCP_TOOLS_DIR, CORE_SENSES, mem,
)
from arqitect.brain.helpers import llm_generate, extract_json, strip_markdown_fences
from arqitect.brain.catalog import list_mcp_tools_with_info, discover_nerves
from arqitect.brain.community import find_community_bundle, apply_community_bundle
from arqitect.brain.routing import classify_nerve_role, validate_nerve_role
from arqitect.brain.nerve_template import NERVE_TEMPLATE
from arqitect.brain.events import publish_event, publish_nerve_status
from arqitect.types import Channel, NerveRole

logger = logging.getLogger(__name__)

# Description generalization
DESC_MIN_LENGTH = 10
DESC_MAX_LENGTH = 200
_DESCRIPTION_STOPWORDS = frozenset({
    "handles", "provides", "manages", "performs", "processes",
    "general", "various", "the", "a", "an", "and", "or", "for", "of", "to",
})
_DOMAIN_MARKERS = ("handles", "provides", "manages", "performs", "processes",
                   "domain", "expert", "specialist", "operations", "capabilities")

# Catch-all names — these become black holes that swallow every other nerve during consolidation.
_CATCHALL_NAMES = frozenset({
    "general_knowledge_nerve", "knowledge_nerve", "general_nerve",
    "utility_nerve", "misc_nerve", "everything_nerve", "general_purpose_nerve",
    "catchall_nerve", "default_nerve", "info_nerve",
})

# MCP tools that must never be overwritten by fabrication.
CORE_MCP_TOOLS = frozenset({"image_generator"})

MAX_PRESEED_TOOLS = 3
MAX_FABRICATION_RETRIES = 2

# Phrases that indicate a system prompt is too generic to be useful.
_GENERIC_PROMPT_PHRASES = (
    "provide helpful", "assist the user", "help the user", "answer questions",
    "provide accurate", "ensure quality", "be helpful", "respond appropriately",
    "handle requests", "process requests", "provide information",
)


def _derive_nerve_name(description: str, fallback: str) -> str:
    """Derive a domain nerve name from a description, used when the original name is rejected."""
    for word in description.lower().split():
        if word not in _DESCRIPTION_STOPWORDS and len(word) > 2:
            candidate = f"{word}_nerve"
            return re.sub(r"[^a-z0-9_]", "_", candidate).strip("_")
    return fallback


def _is_generic_prompt(system_prompt: str) -> bool:
    """Check if a system prompt is too generic to be useful for a specific nerve."""
    if not system_prompt:
        return True
    prompt_lower = system_prompt.lower()
    generic_hits = sum(1 for phrase in _GENERIC_PROMPT_PHRASES if phrase in prompt_lower)
    return generic_hits >= 2


def _regenerate_specific_prompt(name: str, description: str, domain: str) -> str:
    """Retry system prompt generation with a stricter prompt focused on specificity."""
    try:
        raw = llm_generate(
            BRAIN_MODEL,
            f"Write a system prompt for a nerve called '{name}' that does: {description}\n\n"
            f"Requirements:\n"
            f"- 3-4 sentences, ONLY about {domain}\n"
            f"- State the exact goal: what input it takes and what output it produces\n"
            f"- Include at least one output format rule (e.g. units, structure, length)\n"
            f"- Include one boundary: what this nerve does NOT handle\n"
            f"- Do NOT use generic phrases like 'provide helpful responses'\n\n"
            f"Return ONLY the system prompt text, no quotes or JSON.",
        )
        result = raw.strip().strip('"').strip("'")
        if result and len(result) > 20:
            return result
    except Exception as e:
        print(f"[BRAIN] Specific prompt regeneration failed: {e}")
    return ""


def delete_nerve(name: str):
    """Fully delete a nerve — filesystem, cold memory, and Redis."""
    nerve_dir = os.path.join(NERVES_DIR, name)
    if os.path.isdir(nerve_dir):
        shutil.rmtree(nerve_dir)
    try:
        mem.cold.delete_nerve(name)
    except Exception:
        logger.warning("Failed to delete nerve '%s' from cold memory", name)
    publish_nerve_status()
    print(f"[BRAIN] Deleted nerve '{name}'")


def _generalize_description(name: str, description: str) -> str:
    """Generalize a task-specific description into a domain-level description.

    The brain's small LLM often passes the raw user query as the nerve description
    (e.g. "Calculate 2+2" instead of "Handles arithmetic and mathematical calculations").
    A narrow description causes the nerve's self-awareness gate to reject valid tasks.
    """
    desc_lower = description.lower()
    if any(m in desc_lower for m in _DOMAIN_MARKERS):
        return description

    domain = name.replace("_nerve", "").replace("_", " ")

    try:
        raw = llm_generate(
            BRAIN_MODEL,
            f"A nerve called '{name}' is being created. The user's original request was:\n"
            f'  "{description}"\n\n'
            f"Write a ONE-SENTENCE description for this nerve.\n"
            f"The description must be SPECIFIC to the '{domain}' domain — state exactly what\n"
            f"this nerve does, what inputs it expects, and what outputs it produces.\n"
            f"Do NOT mention specific values, cities, names, or numbers from the request.\n"
            f"Do NOT be vague or generic — avoid catch-all phrases like 'handles various tasks'\n"
            f"or 'provides general assistance'.\n\n"
            f"Examples:\n"
            f'  "Calculate 2+2" -> "Solves arithmetic and algebraic expressions, returning numeric results with step-by-step breakdowns"\n'
            f'  "What is the capital of France?" -> "Answers geography questions about capitals, countries, borders, and population statistics"\n'
            f'  "Weather in Tel Aviv" -> "Retrieves weather forecasts, current conditions, and climate data for specified locations"\n'
            f'  "Tell me a joke" -> "Generates jokes, puns, and comedic content tailored to requested topics or styles"\n\n'
            f"Return ONLY the one-sentence description, no quotes, no explanation.",
        )
        result = raw.strip().strip('"').strip("'")
        if result and DESC_MIN_LENGTH < len(result) < DESC_MAX_LENGTH and not result.startswith("{"):
            print(f"[BRAIN] Generalized description: '{description[:40]}' -> '{result[:60]}'")
            return result
    except Exception as e:
        print(f"[BRAIN] Description generalization failed: {e}")

    return description


# ---------------------------------------------------------------------------
# Name collision guards
# ---------------------------------------------------------------------------

def _apply_name_guards(name: str, description: str, mcp_tools: list[str] | None,
                       all_mcp_tools: dict) -> tuple[str, list[str] | None]:
    """Reject catch-all, sense-collision, and tool-as-nerve names. Returns (fixed_name, mcp_tools)."""
    if name in _CATCHALL_NAMES:
        name = _derive_nerve_name(description, name)
        print(f"[BRAIN] Rejected catch-all name, renamed to '{name}'")

    sense_names = {s for s in CORE_SENSES} | {f"{s}_nerve" for s in CORE_SENSES}
    if name in sense_names:
        name = _derive_nerve_name(description, f"{name}_domain")
        print(f"[BRAIN] Rejected sense-collision name, renamed to '{name}'")

    if all_mcp_tools and (name in all_mcp_tools or name.removesuffix("_nerve") in all_mcp_tools):
        tool_name = name if name in all_mcp_tools else name.removesuffix("_nerve")
        if mcp_tools is None:
            mcp_tools = [tool_name]
        elif tool_name not in mcp_tools:
            mcp_tools.append(tool_name)
        name = _derive_nerve_name(description, name)
        print(f"[BRAIN] Rejected tool-as-nerve name '{tool_name}', renamed to '{name}'")

    return name, mcp_tools


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

def _generate_rich_metadata(name: str, description: str) -> tuple[str, str]:
    """Generate system prompt and examples via LLM. Returns (system_prompt, examples_json)."""
    domain = name.replace("_nerve", "").replace("_", " ")
    try:
        meta_raw = llm_generate(
            BRAIN_MODEL,
            f"You are designing a nerve agent called '{name}' whose purpose is: {description}\n\n"
            f"Generate a JSON object with:\n"
            f'  "system_prompt": "3-4 sentences of behavioral instructions specific to {domain}",\n'
            f'  "examples": [2-3 objects with "input" and "output" showing concrete examples]\n\n'
            f"SYSTEM PROMPT RULES:\n"
            f"- State the nerve's SPECIFIC goal and what domain it operates in\n"
            f"- Include concrete output format requirements (e.g. units, structure, style)\n"
            f"- Include at least one constraint or boundary (what this nerve does NOT do)\n"
            f"- NEVER use generic phrases like 'provide helpful responses' or 'assist the user'\n"
            f"- The prompt must be useful ONLY for {domain} tasks — if it could apply to any nerve, it's too generic\n\n"
            f"Example output for a weather nerve:\n"
            f'{{"system_prompt": "You are a weather specialist. Given a location, provide current conditions, '
            f"temperature in Celsius, humidity, and a 3-day forecast summary. Always specify the data source's "
            f"freshness. Do not provide travel or clothing advice — only weather data.\", "
            f'"examples": [{{"input": "weather in Paris", "output": "Paris: 18°C, partly cloudy, humidity 65%. '
            f'3-day: Wed 20°C sunny, Thu 17°C rain, Fri 19°C cloudy."}}]}}\n\n'
            f"Return ONLY the JSON object.",
        )
        meta_json = extract_json(meta_raw)
        if meta_json:
            system_prompt = meta_json.get("system_prompt", "")
            if _is_generic_prompt(system_prompt):
                print(f"[BRAIN] Rejected generic system prompt for '{name}', requesting specific one")
                system_prompt = _regenerate_specific_prompt(name, description, domain)
            examples = meta_json.get("examples", [])
            examples_json = json.dumps(examples) if isinstance(examples, list) else "[]"
            print(f"[BRAIN] Generated rich metadata for nerve '{name}'")
            return system_prompt, examples_json
    except Exception as e:
        print(f"[BRAIN] Rich metadata generation failed: {e}")

    return "", "[]"


def _write_meta_json(nerve_dir: str, name: str, role: str) -> None:
    """Write meta.json into the nerve directory for tuning and qualification config."""
    try:
        from arqitect.brain.adapters import get_active_variant, get_model_name_for_role, build_meta_json
        size = get_active_variant(role)
        slug = get_model_name_for_role(role)
        if slug:
            meta = build_meta_json(role, slug, size)
            with open(os.path.join(nerve_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[BRAIN] Wrote meta.json for nerve '{name}' ({size}/{slug})")
    except Exception as e:
        print(f"[BRAIN] meta.json generation skipped: {e}")


# ---------------------------------------------------------------------------
# Tool pre-seeding
# ---------------------------------------------------------------------------

def _preseed_nerve_tools(name: str, description: str, trigger_task: str,
                         mcp_tools: list[str] | None, all_mcp_tools: dict, role: str) -> None:
    """Pre-seed a new nerve with matching MCP tools from explicit, mentioned, and keyword sources."""
    try:
        seeded: set[str] = set()
        available_names = set(all_mcp_tools.keys()) if all_mcp_tools else set()

        _preseed_explicit_tools(name, mcp_tools, available_names, seeded)
        _preseed_mentioned_tools(name, trigger_task, available_names, seeded)
        _preseed_matched_tools(name, description, all_mcp_tools, role, seeded)

        if not seeded:
            print(f"[BRAIN] No MCP tools matched nerve '{name}' (desc: {description[:60]})")
    except Exception as e:
        print(f"[BRAIN] Pre-seed tools failed: {e}")


def _preseed_explicit_tools(name: str, mcp_tools: list[str] | None,
                            available: set[str], seeded: set[str]) -> None:
    """Seed tools explicitly requested by the user."""
    if not mcp_tools or not available:
        return
    for tool_name in mcp_tools:
        if tool_name in available:
            mem.cold.add_nerve_tool(name, tool_name)
            seeded.add(tool_name)
            print(f"[BRAIN] Pre-seeded nerve '{name}' with explicit tool '{tool_name}'")


def _preseed_mentioned_tools(name: str, trigger_task: str,
                             available: set[str], seeded: set[str]) -> None:
    """Seed tools whose names appear literally in the trigger task."""
    if not trigger_task or not available:
        return
    task_lower = trigger_task.lower()
    for tool_name in available:
        if tool_name.lower() in task_lower and tool_name not in seeded:
            mem.cold.add_nerve_tool(name, tool_name)
            seeded.add(tool_name)
            print(f"[BRAIN] Pre-seeded nerve '{name}' with mentioned tool '{tool_name}'")


def _preseed_matched_tools(name: str, description: str,
                           all_mcp_tools: dict, role: str, seeded: set[str]) -> None:
    """Seed tools matched by keyword similarity to the nerve description."""
    if not all_mcp_tools:
        return
    from arqitect.matching import match_tools
    from arqitect.brain.adapters import get_tuning_config
    threshold = get_tuning_config(role)["merge_threshold"]
    top_tools = match_tools(description, all_mcp_tools, threshold=threshold)
    for tool_name, score in top_tools[:MAX_PRESEED_TOOLS]:
        if tool_name not in seeded:
            mem.cold.add_nerve_tool(name, tool_name)
            seeded.add(tool_name)
            print(f"[BRAIN] Pre-seeded nerve '{name}' with matched tool '{tool_name}' (score={score:.1f})")


# ---------------------------------------------------------------------------
# Background qualification
# ---------------------------------------------------------------------------

def _start_background_qualification(name: str, description: str, trigger_task: str) -> None:
    """Launch qualification in a background thread. Nerve is usable immediately."""
    if mem.cold.is_qualified("nerve", name):
        print(f"[BRAIN] Nerve '{name}' already qualified — skipping qualification.")
        publish_nerve_status()
        return

    def _qualify():
        try:
            from arqitect.critic.qualify_nerve import qualify_nerve
            print(f"[BRAIN] Starting qualification for nerve '{name}' (background)...")
            publish_event(Channel.BRAIN_THOUGHT, {
                "stage": "qualifying", "nerve": name,
                "message": f"Training nerve '{name}' — testing and tuning for quality. This may take a few minutes."
            })
            qual_result = qualify_nerve(name, description, trigger_task, mem)
            _report_qualification_result(name, qual_result)
        except Exception as e:
            print(f"[BRAIN] Qualification failed for nerve '{name}': {e}")
            publish_event(Channel.BRAIN_THOUGHT, {
                "stage": "qualification_error", "nerve": name,
                "message": f"Qualification for '{name}' encountered an error but the nerve is available."
            })

    threading.Thread(target=_qualify, daemon=True, name=f"qualify-{name}").start()


def _report_qualification_result(name: str, qual_result: dict) -> None:
    """Publish qualification outcome as an event."""
    score_pct = int(qual_result.get("score", 0) * 100)
    if qual_result.get("qualified"):
        print(f"[BRAIN] Nerve '{name}' QUALIFIED (score={qual_result['score']:.2f}, iterations={qual_result['iterations']})")
        publish_event(Channel.BRAIN_THOUGHT, {
            "stage": "qualified", "nerve": name,
            "message": f"Nerve '{name}' is trained and ready (quality: {score_pct}%)"
        })
    else:
        print(f"[BRAIN] Nerve '{name}' FAILED qualification (score={qual_result.get('score', 0):.2f})")
        publish_event(Channel.BRAIN_THOUGHT, {
            "stage": "qualification_failed", "nerve": name,
            "message": f"Nerve '{name}' needs improvement (quality: {score_pct}%). It will still work but may be less reliable."
        })
    publish_nerve_status()


# ---------------------------------------------------------------------------
# Main synthesis entry point
# ---------------------------------------------------------------------------

def synthesize_nerve(name: str, description: str, mcp_tools: list[str] | None = None,
                     trigger_task: str = "", role: str | None = None) -> tuple[str, str]:
    """Generate an autonomous nerve agent.

    Checks the community manifest first — if a bundle exists for this nerve,
    uses the community's description, system prompt, examples, role, and
    declared tool list instead of guessing via LLM.

    Falls back to LLM-based metadata generation and keyword tool matching
    only when no community bundle is available.

    Returns:
        (actual_name, nerve_path) — name may differ from input due to
        collision guards (catch-all, sense-collision, tool-as-nerve).
    """
    name = name.removesuffix(".py")

    all_mcp_tools = _fetch_mcp_tools()
    name, mcp_tools = _apply_name_guards(name, description, mcp_tools, all_mcp_tools)

    # Community-first: use curated bundle when available
    bundle = find_community_bundle(name)
    if bundle:
        return _synthesize_from_community(name, bundle, trigger_task)

    return _synthesize_from_scratch(name, description, mcp_tools, all_mcp_tools,
                                    trigger_task, role)


def _synthesize_from_community(name: str, bundle: dict,
                               trigger_task: str) -> tuple[str, str]:
    """Synthesize a nerve using a community bundle as the source of truth."""
    description = bundle.get("description", name)
    role = validate_nerve_role(bundle.get("role", NerveRole.TOOL))
    print(f"[BRAIN] Synthesizing nerve from community bundle: {name} [role={role}]")

    nerve_dir, nerve_path = _create_nerve_files(name, role, description)
    apply_community_bundle(name, bundle, mem.cold)
    _write_meta_json(nerve_dir, name, role)

    print(f"[BRAIN] Synthesized nerve (community): {name} -> {nerve_path}")
    _start_background_qualification(name, description, trigger_task)
    return name, nerve_path


def _synthesize_from_scratch(name: str, description: str,
                             mcp_tools: list[str] | None, all_mcp_tools: dict,
                             trigger_task: str, role: str | None) -> tuple[str, str]:
    """Synthesize a nerve with LLM-generated metadata and keyword tool matching."""
    description = _generalize_description(name, description)

    if not role:
        role = classify_nerve_role(name, description)
    print(f"[BRAIN] Synthesizing nerve from scratch: {name} [role={role}]")

    nerve_dir, nerve_path = _create_nerve_files(name, role, description)
    system_prompt, examples_json = _generate_rich_metadata(name, description)
    mem.cold.register_nerve_rich(name, description, system_prompt, examples_json, role=role)
    _write_meta_json(nerve_dir, name, role)
    _preseed_nerve_tools(name, description, trigger_task, mcp_tools, all_mcp_tools, role)

    print(f"[BRAIN] Synthesized nerve (scratch): {name} -> {nerve_path}")
    _start_background_qualification(name, description, trigger_task)
    return name, nerve_path


def _fetch_mcp_tools() -> dict:
    """Fetch MCP tools, returning empty dict on failure."""
    try:
        return list_mcp_tools_with_info()
    except Exception as e:
        print(f"[BRAIN] Could not fetch MCP tools: {e}")
        return {}


def _sanitize_description_for_template(description: str) -> str:
    """Make a description safe for insertion into triple-double-quoted strings.

    Escapes sequences that would break the generated nerve.py syntax:
    triple quotes, lone backslashes, and trailing backslashes.
    """
    description = description.replace("\\", "\\\\")
    description = description.replace('"""', '\\"\\"\\"')
    return description.rstrip("\\")


def _validate_nerve_source(source: str, name: str) -> bool:
    """Validate that generated nerve source compiles without syntax errors."""
    try:
        compile(source, f"{name}/nerve.py", "exec")
        return True
    except SyntaxError as e:
        logger.warning("Generated nerve '%s' has syntax error: %s", name, e)
        return False


def _create_nerve_files(name: str, role: str, description: str) -> tuple[str, str]:
    """Create the nerve directory and write nerve.py from template. Returns (nerve_dir, nerve_path)."""
    nerve_dir = os.path.join(NERVES_DIR, name)
    os.makedirs(nerve_dir, exist_ok=True)
    nerve_path = os.path.join(nerve_dir, "nerve.py")

    safe_desc = _sanitize_description_for_template(description)
    content = (NERVE_TEMPLATE.replace("{{NERVE_NAME}}", name)
                              .replace("{{NERVE_ROLE}}", role)
                              .replace("{{DESCRIPTION}}", safe_desc))

    if not _validate_nerve_source(content, name):
        # Fall back to a safe description derived from the nerve name
        fallback_desc = name.replace("_nerve", "").replace("_", " ")
        content = (NERVE_TEMPLATE.replace("{{NERVE_NAME}}", name)
                                  .replace("{{NERVE_ROLE}}", role)
                                  .replace("{{DESCRIPTION}}", fallback_desc))

    with open(nerve_path, "w") as f:
        f.write(content)
    os.chmod(nerve_path, 0o755)
    return nerve_dir, nerve_path


# ---------------------------------------------------------------------------
# MCP tool fabrication
# ---------------------------------------------------------------------------

def fabricate_mcp_tool(name: str, description: str, params: str) -> str:
    """Create a new MCP tool plugin in /mcp_tools/.

    Returns:
        Path to the created tool file.
    """
    if name in CORE_MCP_TOOLS:
        print(f"[BRAIN] Refusing to overwrite core MCP tool: {name}")
        return os.path.join(MCP_TOOLS_DIR, f"{name}.py")

    os.makedirs(MCP_TOOLS_DIR, exist_ok=True)
    print(f"[BRAIN] Fabricating MCP tool: {name}")

    prompt = textwrap.dedent(f"""\
        Write a Python function called `run` for an MCP tool called '{name}'.
        Description: {description}
        Parameters: {params}

        Requirements:
        - The function signature must be: def run({params}) -> str:
        - Include a docstring that describes what the tool does
        - Return a string result
        - Use only standard library + requests
        - Keep it under 30 lines
        - Handle errors gracefully

        Return ONLY the Python code. No markdown fences.
    """)

    code = _generate_valid_tool_code(name, prompt)

    tool_path = os.path.join(MCP_TOOLS_DIR, f"{name}.py")
    with open(tool_path, "w") as f:
        f.write(code)

    print(f"[BRAIN] Fabricated MCP tool: {name} -> {tool_path}")
    print(f"[BRAIN] NOTE: MCP server restart needed to load new tool")
    return tool_path


def _generate_valid_tool_code(name: str, prompt: str) -> str:
    """Generate tool code, retrying on syntax errors up to MAX_FABRICATION_RETRIES."""
    for attempt in range(MAX_FABRICATION_RETRIES):
        suffix = "" if attempt == 0 else "\nPrevious attempt had syntax error. Be careful."
        code = strip_markdown_fences(llm_generate(CODE_MODEL, prompt + suffix))
        try:
            compile(code, f"{name}.py", "exec")
            return code
        except SyntaxError as e:
            print(f"[BRAIN] Tool syntax error (attempt {attempt + 1}/{MAX_FABRICATION_RETRIES}): {e}")
    return code  # Return last attempt even if invalid


# ---------------------------------------------------------------------------
# Nerve pruning
# ---------------------------------------------------------------------------

def prune_degenerate_nerves():
    """Remove duplicate nerves at startup. Keeps the one with more invocations."""
    from arqitect.matching import find_duplicate_nerves

    catalog = discover_nerves()
    duplicates = find_duplicate_nerves(catalog)
    if not duplicates:
        return

    pruned: set[str] = set()
    for nerve_a, nerve_b, score in duplicates:
        if nerve_a in pruned or nerve_b in pruned:
            continue

        winner = _pick_prune_winner(nerve_a, nerve_b)
        if winner is None:
            continue

        keep, remove = winner
        print(f"[BRAIN] Pruning duplicate nerve '{remove}' (score={score:.1f}, keeping '{keep}')")
        delete_nerve(remove)
        pruned.add(remove)


def _pick_prune_winner(nerve_a: str, nerve_b: str) -> tuple[str, str] | None:
    """Decide which nerve to keep in a duplicate pair. Returns (keep, remove) or None to skip."""
    a_is_sense = nerve_a in CORE_SENSES or mem.cold.is_sense(nerve_a)
    b_is_sense = nerve_b in CORE_SENSES or mem.cold.is_sense(nerve_b)

    if a_is_sense and b_is_sense:
        return None
    if a_is_sense:
        return nerve_a, nerve_b
    if b_is_sense:
        return nerve_b, nerve_a

    qual_a = mem.cold.get_qualification("nerve", nerve_a)
    qual_b = mem.cold.get_qualification("nerve", nerve_b)
    score_a = qual_a.get("score", 0) if qual_a else 0
    score_b = qual_b.get("score", 0) if qual_b else 0

    from arqitect.brain.adapters import get_tuning_config
    qual_thresh = get_tuning_config("nerve")["qualification_threshold"]
    if score_a >= qual_thresh and score_b >= qual_thresh:
        print(f"[BRAIN] Skipping prune: both '{nerve_a}' ({score_a:.0%}) and '{nerve_b}' ({score_b:.0%}) are qualified")
        return None

    if score_a != score_b:
        return (nerve_a, nerve_b) if score_a >= score_b else (nerve_b, nerve_a)

    info_a = mem.cold.get_nerve_info(nerve_a) or {}
    info_b = mem.cold.get_nerve_info(nerve_b) or {}
    inv_a = info_a.get("total_invocations", 0) if isinstance(info_a, dict) else 0
    inv_b = info_b.get("total_invocations", 0) if isinstance(info_b, dict) else 0
    return (nerve_a, nerve_b) if inv_a >= inv_b else (nerve_b, nerve_a)


def prune_low_quality_nerves() -> list[str]:
    """Remove nerves that scored below low_quality_threshold after qualification.

    Only prunes nerves that have been qualified (i.e., have a score recorded)
    and are not core senses.

    Returns:
        List of pruned nerve names.
    """
    from arqitect.brain.adapters import get_tuning_config
    low_quality_threshold = get_tuning_config("nerve")["low_quality_threshold"]

    catalog = discover_nerves()
    nerve_scores = {
        q["subject_name"]: q.get("score", 0.0)
        for q in mem.cold.list_qualifications()
        if q["subject_type"] == "nerve"
    }

    pruned = []
    for name in list(catalog.keys()):
        if name in CORE_SENSES or mem.cold.is_sense(name):
            continue
        score = nerve_scores.get(name)
        if score is None:
            continue
        if score < low_quality_threshold:
            print(f"[BRAIN] Pruning low-quality nerve '{name}' (score={score:.0%} < {low_quality_threshold:.0%})")
            delete_nerve(name)
            pruned.append(name)

    if pruned:
        print(f"[BRAIN] Pruned {len(pruned)} low-quality nerve(s): {pruned}")
    return pruned
