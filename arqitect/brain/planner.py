"""Recipe-based Planner — matches, generates, executes, and learns recipes.

A recipe is a learned pattern of nerve chain steps for a category of workflow.
Recipes are stored as JSON files in brain/recipes/ and improve over time through
evaluation after each execution.

Flow:
  1. Match stored recipe by category + task similarity
  2. If no match, generate a new recipe using brain LLM
  3. Return a chain_nerves decision for the existing chain handler
  4. After execution, evaluate results and store/update the recipe
"""

import hashlib
import json
import os
import time

from arqitect.brain.config import BRAIN_MODEL
from arqitect.brain.helpers import llm_generate, extract_json


_RECIPES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")
os.makedirs(_RECIPES_DIR, exist_ok=True)


# ── Main entry point ─────────────────────────────────────────────────────────

def plan_task(task: str, category: str, project_facts: dict | None) -> dict | None:
    """Match or generate a recipe for the given workflow task.

    Returns a chain_nerves decision dict, or None if planning fails.
    The decision has: {"action": "chain_nerves", "steps": [...], "goal": "...",
                       "recipe_id": "..."}
    """
    # 1. Try to match a stored recipe
    recipe = match_recipe(task, category)
    if recipe:
        print(f"[PLANNER] Matched stored recipe: {recipe['name']} "
              f"(success_rate={recipe.get('success_rate', 0):.0%}, runs={recipe.get('runs', 0)})")
        return _recipe_to_chain(recipe, task, project_facts)

    # 2. No match — generate a new recipe
    print(f"[PLANNER] No stored recipe for category='{category}', generating new recipe")
    recipe = generate_recipe(task, category, project_facts)
    if recipe:
        return _recipe_to_chain(recipe, task, project_facts)

    return None


# ── Recipe matching ──────────────────────────────────────────────────────────

def match_recipe(task: str, category: str) -> dict | None:
    """Find a stored recipe that matches this task.

    Matches by category first, then picks the best by success rate.
    """
    recipes = _load_all_recipes()
    if not recipes:
        return None

    # Filter by category
    candidates = [r for r in recipes if r.get("category") == category] if category else []

    # If no category match, try all recipes
    if not candidates:
        candidates = recipes

    # Pick the one with highest success rate (and at least 1 run for confidence)
    best = None
    best_score = -1
    for r in candidates:
        score = r.get("success_rate", 0)
        # Bonus for more runs (more tested = more trusted)
        runs = r.get("runs", 0)
        if runs > 0:
            score += min(runs / 10, 0.1)  # small bonus, capped
        if score > best_score:
            best_score = score
            best = r

    return best


# ── Recipe generation ────────────────────────────────────────────────────────

def generate_recipe(task: str, category: str, project_facts: dict | None) -> dict | None:
    """Use brain LLM to generate a new recipe for this task type.

    The recipe is a list of nerve steps — each step describes WHAT a nerve
    should do, and the nerve name is derived from the description.
    """
    project_context = ""
    if project_facts:
        project_context = (
            f"\nProject context: language={project_facts.get('language', '?')}, "
            f"framework={project_facts.get('framework', '?')}, "
            f"test_framework={project_facts.get('test_framework', '?')}"
        )

    prompt = (
        f"Design a step-by-step workflow recipe for this task.\n"
        f"Task: {task}\n"
        f"Category: {category}\n"
        f"{project_context}\n\n"
        f"Each step is a nerve (autonomous AI agent) that does ONE thing.\n\n"
        f"IMPORTANT — Use core senses whenever possible instead of creating new nerves:\n"
        f"  - touch: file/shell operations — read, write, list, exec shell commands, mkdir, etc.\n"
        f"    For ANY shell command, use nerve='touch' with args_template as a JSON string:\n"
        f'    {{"command": "exec", "cmd": "the shell command here"}}\n'
        f"    For file writes: {{\"command\": \"write\", \"path\": \"file.py\", \"content\": \"code here\"}}\n"
        f"  - sight: image analysis\n"
        f"  - hearing: audio input/output\n"
        f"  - communication: formatting, tone adjustment\n"
        f"  - awareness: identity, capabilities\n\n"
        f"Only create a NEW nerve name for tasks that genuinely need a specialized AI agent "
        f"(e.g. code analysis, planning, research). Simple file/shell ops MUST use 'touch'.\n"
        f"Keep recipes SHORT — 3-5 steps max. Combine related shell commands.\n\n"
        f"Output a JSON object:\n"
        f'{{"name": "recipe_name", "description": "what this recipe does", '
        f'"steps": [{{"nerve": "nerve_name", "description": "what this nerve does", '
        f'"args_template": "instruction or JSON args for the nerve"}}, ...]}}\n\n'
        f"Output ONLY the JSON object."
    )

    raw = llm_generate(BRAIN_MODEL, prompt,
                       "You design workflow recipes as nerve chain steps. Output only JSON.")
    result = extract_json(raw)

    if not result or not result.get("steps"):
        print(f"[PLANNER] Failed to generate recipe")
        return None

    # Build recipe structure
    recipe_id = hashlib.md5(f"{category}:{task}".encode()).hexdigest()[:8]
    recipe = {
        "id": recipe_id,
        "name": result.get("name", f"{category}_recipe"),
        "category": category,
        "description": result.get("description", task),
        "steps": result["steps"],
        "success_rate": 0,
        "runs": 0,
        "created": time.strftime("%Y-%m-%d"),
        "last_used": time.strftime("%Y-%m-%d"),
    }

    print(f"[PLANNER] Generated recipe '{recipe['name']}' with {len(recipe['steps'])} steps")
    return recipe


# ── Recipe to chain_nerves ───────────────────────────────────────────────────

def _recipe_to_chain(recipe: dict, task: str, project_facts: dict | None) -> dict:
    """Convert a recipe into a chain_nerves decision."""
    steps = []
    for step in recipe.get("steps", []):
        nerve_name = step.get("nerve", "")
        args_template = step.get("args_template", "")
        description = step.get("description", "")

        # Fill in the task-specific args
        args = args_template.replace("{task}", task) if args_template else task
        if project_facts and project_facts.get("path"):
            args = args.replace("{project_path}", project_facts["path"])

        steps.append({
            "nerve": nerve_name,
            "args": args,
            "description": description,
        })

    return {
        "action": "chain_nerves",
        "steps": steps,
        "goal": task,
        "recipe_id": recipe.get("id", ""),
        "recipe_name": recipe.get("name", ""),
    }


# ── Recipe evaluation and storage ────────────────────────────────────────────

def evaluate_and_store_recipe(recipe_decision: dict, step_results: list[dict],
                              task: str, success: bool):
    """Evaluate recipe execution results and store/update the recipe.

    Called by _handle_recipe_chain after execution completes.
    """
    recipe_id = recipe_decision.get("recipe_id", "")
    recipe_name = recipe_decision.get("recipe_name", "")

    if not recipe_id:
        return

    # Try to load existing recipe
    existing = _load_recipe(recipe_id)

    if existing:
        # Update stats
        runs = existing.get("runs", 0) + 1
        old_rate = existing.get("success_rate", 0)
        # Running average
        new_rate = ((old_rate * (runs - 1)) + (1.0 if success else 0.0)) / runs
        existing["runs"] = runs
        existing["success_rate"] = round(new_rate, 3)
        existing["last_used"] = time.strftime("%Y-%m-%d")

        # If failed, ask LLM for improvement suggestions
        if not success:
            improvement = _suggest_improvement(existing, step_results, task)
            if improvement:
                notes = existing.get("eval_notes", "")
                existing["eval_notes"] = f"{notes}\n{improvement}".strip()

        _save_recipe(recipe_id, existing)
        print(f"[PLANNER] Updated recipe '{recipe_name}': "
              f"success_rate={existing['success_rate']:.0%}, runs={existing['runs']}")
    else:
        # Store as new recipe
        new_recipe = {
            "id": recipe_id,
            "name": recipe_name,
            "category": recipe_decision.get("category", ""),
            "description": recipe_decision.get("goal", task),
            "steps": recipe_decision.get("steps", []),
            "success_rate": 1.0 if success else 0.0,
            "runs": 1,
            "created": time.strftime("%Y-%m-%d"),
            "last_used": time.strftime("%Y-%m-%d"),
        }

        if not success:
            improvement = _suggest_improvement(new_recipe, step_results, task)
            if improvement:
                new_recipe["eval_notes"] = improvement

        _save_recipe(recipe_id, new_recipe)
        print(f"[PLANNER] Stored new recipe '{recipe_name}'")


def _suggest_improvement(recipe: dict, step_results: list[dict], task: str) -> str:
    """Ask LLM to suggest improvements based on failure."""
    failed_steps = [r for r in step_results if r.get("status") == "error"]
    if not failed_steps:
        return ""

    failed_desc = ", ".join(f"step {r['step']+1} ({r.get('nerve', '?')})" for r in failed_steps)
    prompt = (
        f"This recipe failed. Recipe: {recipe.get('name', '?')}\n"
        f"Task: {task}\n"
        f"Failed steps: {failed_desc}\n"
        f"Steps: {json.dumps(recipe.get('steps', []), indent=2)[:500]}\n\n"
        f"Suggest ONE concise improvement (max 100 chars)."
    )
    raw = llm_generate(BRAIN_MODEL, prompt,
                       "You suggest recipe improvements. Be concise — one sentence max.")
    return raw.strip()[:200]


# ── Recipe persistence ───────────────────────────────────────────────────────

def _load_all_recipes() -> list[dict]:
    """Load all stored recipes from disk."""
    recipes = []
    for fname in os.listdir(_RECIPES_DIR):
        if fname.endswith(".json"):
            path = os.path.join(_RECIPES_DIR, fname)
            try:
                with open(path) as f:
                    recipes.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return recipes


def _load_recipe(recipe_id: str) -> dict | None:
    """Load a specific recipe by ID."""
    path = os.path.join(_RECIPES_DIR, f"{recipe_id}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_recipe(recipe_id: str, recipe: dict):
    """Save a recipe to disk."""
    path = os.path.join(_RECIPES_DIR, f"{recipe_id}.json")
    with open(path, "w") as f:
        json.dump(recipe, f, indent=2)
