"""On-the-fly Nerve Chain Planner — composes chains dynamically from live catalog.

Instead of matching persisted recipes, the planner scans the live nerve catalog
using hybrid keyword+embedding matching, orders candidates via LLM, and
identifies gaps that need synthesis.

Three-phase pipeline:
  1. Filter candidates — match_nerves() scan, no LLM, O(n)
  2. Sort into sequence — LLM orders candidates into a chain
  3. Find gaps — LLM identifies missing capabilities to synthesize
"""

from arqitect.brain.config import BRAIN_MODEL
from arqitect.brain.helpers import llm_generate, extract_json
from arqitect.matching import match_nerves
from arqitect.types import Action

MAX_CANDIDATES = 15
PAIRWISE_CAP = 5


# ── Main entry point ─────────────────────────────────────────────────────────

def compose_chain(
    task: str,
    nerve_catalog: dict[str, str],
    project_facts: dict | None = None,
    size_class: str | None = None,
) -> dict | None:
    """Compose a nerve chain for a task by scanning the live catalog.

    Pipeline: filter candidates → sort into sequence → find gaps.
    Returns a chain_nerves decision dict or None if no candidates match.

    Args:
        task: The user's task description.
        nerve_catalog: Full {name: description} mapping of available nerves.
        project_facts: Optional detected project context (language, framework, etc.).
        size_class: Model size class ('tinylm', 'small', 'medium', 'large').

    Returns:
        A dict with action=chain_nerves, steps, and goal — or None.
    """
    if not task or not nerve_catalog:
        return None

    candidates = _filter_candidates(task, nerve_catalog)
    if not candidates:
        return None

    chain = _sort_into_sequence(task, candidates, nerve_catalog, project_facts, size_class)
    if not chain:
        return None

    gaps = _find_gaps(task, chain, nerve_catalog, size_class)
    if gaps:
        chain.extend(gaps)

    return {
        "action": Action.CHAIN_NERVES,
        "steps": chain,
        "goal": task,
    }


# ── Phase 1: Filter candidates ───────────────────────────────────────────────

def _filter_candidates(
    task: str,
    nerve_catalog: dict[str, str],
) -> list[tuple[str, float]]:
    """Filter nerve catalog to the top candidates for a task.

    Uses hybrid keyword+embedding matching via match_nerves(). No LLM call.

    Args:
        task: The user's task description.
        nerve_catalog: Full {name: description} mapping.

    Returns:
        Up to MAX_CANDIDATES ranked (name, score) pairs, descending by score.
    """
    ranked = match_nerves(task, nerve_catalog)
    return ranked[:MAX_CANDIDATES]


# ── Phase 2: Sort into sequence ──────────────────────────────────────────────

def _sort_into_sequence(
    task: str,
    candidates: list[tuple[str, float]],
    nerve_catalog: dict[str, str],
    project_facts: dict | None,
    size_class: str | None,
) -> list[dict]:
    """Order candidates into an executable chain via LLM.

    Picks a strategy based on model size:
    - medium/large: single-pass LLM call with all candidates
    - tinylm/small: pairwise comparison, capped at PAIRWISE_CAP candidates

    Args:
        task: The user's task description.
        candidates: Ranked (name, score) pairs from phase 1.
        nerve_catalog: Full catalog for description lookup.
        project_facts: Optional project context.
        size_class: Model size class.

    Returns:
        Ordered list of step dicts: [{"nerve": name, "args": instruction}, ...].
    """
    if size_class in ("tinylm", "small"):
        return _sort_pairwise(task, candidates[:PAIRWISE_CAP], nerve_catalog, project_facts)
    return _sort_single_pass(task, candidates, nerve_catalog, project_facts)


def _sort_single_pass(
    task: str,
    candidates: list[tuple[str, float]],
    nerve_catalog: dict[str, str],
    project_facts: dict | None,
) -> list[dict]:
    """Sort candidates in one LLM call — for medium/large models.

    Args:
        task: The user's task description.
        candidates: Ranked (name, score) pairs.
        nerve_catalog: Full catalog for description lookup.
        project_facts: Optional project context.

    Returns:
        Ordered list of step dicts.
    """
    candidate_lines = _format_candidate_list(candidates, nerve_catalog)
    project_context = _format_project_context(project_facts)

    prompt = (
        f"Task: {task}\n"
        f"{project_context}"
        f"Available nerves:\n{candidate_lines}\n\n"
        f"Select the nerves needed for this task and order them into a step-by-step chain.\n"
        f"Each step should have a specific instruction for what that nerve should do.\n"
        f"Use core senses (touch, sight, hearing, communication, awareness) for basic ops.\n"
        f"Keep the chain SHORT — 3-6 steps max.\n\n"
        f'Output a JSON array: [{{"nerve": "name", "args": "instruction for this step"}}, ...]\n'
        f"Output ONLY the JSON array."
    )

    raw = llm_generate(
        BRAIN_MODEL, prompt,
        "You compose nerve chains for tasks. Output only a JSON array of steps.",
    )
    result = extract_json(raw)

    if isinstance(result, list):
        return _validate_steps(result)

    if isinstance(result, dict):
        if "steps" in result:
            return _validate_steps(result["steps"])
        # Single step dict — wrap in list
        if "nerve" in result:
            return _validate_steps([result])

    return []


def _sort_pairwise(
    task: str,
    candidates: list[tuple[str, float]],
    nerve_catalog: dict[str, str],
    project_facts: dict | None,
) -> list[dict]:
    """Sort candidates via pairwise comparison — for tinylm/small models.

    Uses simpler prompts that small models can handle reliably.

    Args:
        task: The user's task description.
        candidates: Ranked (name, score) pairs (capped at PAIRWISE_CAP).
        nerve_catalog: Full catalog for description lookup.
        project_facts: Optional project context.

    Returns:
        Ordered list of step dicts.
    """
    if len(candidates) <= 1:
        return _candidates_to_steps(candidates, nerve_catalog, task)

    candidate_lines = _format_candidate_list(candidates, nerve_catalog)
    project_context = _format_project_context(project_facts)

    prompt = (
        f"Task: {task}\n"
        f"{project_context}"
        f"Nerves:\n{candidate_lines}\n\n"
        f"Order these into a chain for the task. Keep only the needed ones.\n"
        f'Output JSON array: [{{"nerve": "name", "args": "instruction"}}, ...]\n'
        f"Output ONLY the JSON."
    )

    raw = llm_generate(
        BRAIN_MODEL, prompt,
        "Order nerves into a chain. Output only JSON.",
    )
    result = extract_json(raw)

    if isinstance(result, list):
        return _validate_steps(result)

    if isinstance(result, dict):
        if "steps" in result:
            return _validate_steps(result["steps"])
        if "nerve" in result:
            return _validate_steps([result])

    # Fallback: use candidates in score order
    return _candidates_to_steps(candidates, nerve_catalog, task)


# ── Phase 3: Find gaps ───────────────────────────────────────────────────────

def _find_gaps(
    task: str,
    chain: list[dict],
    nerve_catalog: dict[str, str],
    size_class: str | None,
) -> list[dict]:
    """Ask LLM whether the chain is missing any capabilities.

    Gap nerves are returned for synthesis by dispatch's _handle_chain.
    Skipped for tinylm models — they can't reliably assess completeness.

    Args:
        task: The user's task description.
        chain: Current ordered chain steps.
        nerve_catalog: Full catalog for context.
        size_class: Model size class.

    Returns:
        List of gap step dicts to append, or empty list.
    """
    if size_class == "tinylm":
        return []

    chain_desc = "\n".join(
        f"  {i+1}. {s.get('nerve', '?')}: {s.get('args', '')[:80]}"
        for i, s in enumerate(chain)
    )

    prompt = (
        f"Task: {task}\n"
        f"Current chain:\n{chain_desc}\n\n"
        f"Is this chain complete for the task? If not, what nerves are missing?\n"
        f"Only suggest genuinely missing capabilities — do NOT duplicate existing steps.\n\n"
        f'If complete, output: {{"complete": true}}\n'
        f'If missing steps, output: {{"complete": false, "gaps": '
        f'[{{"nerve": "name", "description": "what it does", "args": "instruction"}}]}}\n'
        f"Output ONLY the JSON."
    )

    raw = llm_generate(
        BRAIN_MODEL, prompt,
        "You assess nerve chain completeness. Output only JSON.",
    )
    result = extract_json(raw)

    if not isinstance(result, dict):
        return []

    if result.get("complete", True):
        return []

    gaps = result.get("gaps", [])
    if not isinstance(gaps, list):
        return []

    return _validate_steps(gaps)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_candidate_list(
    candidates: list[tuple[str, float]],
    nerve_catalog: dict[str, str],
) -> str:
    """Format candidates as numbered lines for LLM prompts.

    Args:
        candidates: Ranked (name, score) pairs.
        nerve_catalog: Full catalog for description lookup.

    Returns:
        Formatted string with one nerve per line.
    """
    lines = []
    for i, (name, _score) in enumerate(candidates):
        desc = nerve_catalog.get(name, "")
        lines.append(f"  {i+1}. {name}: {desc}")
    return "\n".join(lines)


def _format_project_context(project_facts: dict | None) -> str:
    """Format project facts as a prompt line, or empty string.

    Args:
        project_facts: Optional project context dict.

    Returns:
        Formatted context line or empty string.
    """
    if not project_facts:
        return ""
    return (
        f"Project: language={project_facts.get('language', '?')}, "
        f"framework={project_facts.get('framework', '?')}, "
        f"test_framework={project_facts.get('test_framework', '?')}\n"
    )


def _validate_steps(steps: list) -> list[dict]:
    """Filter and normalize a list of step dicts from LLM output.

    Ensures each step has at minimum a 'nerve' key.

    Args:
        steps: Raw list from LLM JSON output.

    Returns:
        Validated list of step dicts with 'nerve' and 'args' keys.
    """
    validated = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        nerve = step.get("nerve", "")
        if not nerve:
            continue
        validated.append({
            "nerve": nerve,
            "args": step.get("args", step.get("args_template", "")),
            "description": step.get("description", ""),
        })
    return validated


def _candidates_to_steps(
    candidates: list[tuple[str, float]],
    nerve_catalog: dict[str, str],
    task: str,
) -> list[dict]:
    """Convert raw candidates to step dicts as a fallback when LLM fails.

    Args:
        candidates: Ranked (name, score) pairs.
        nerve_catalog: Full catalog for description lookup.
        task: The user's task, used as default args.

    Returns:
        List of step dicts in candidate order.
    """
    return [
        {
            "nerve": name,
            "args": task,
            "description": nerve_catalog.get(name, ""),
        }
        for name, _score in candidates
    ]
