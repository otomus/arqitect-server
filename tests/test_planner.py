"""Tests for arqitect.brain.planner — recipe matching, generation, evaluation, and storage."""

import hashlib
import json
import os
from unittest.mock import patch

import pytest

from arqitect.brain.planner import (
    _recipe_to_chain,
    evaluate_and_store_recipe,
    generate_recipe,
    match_recipe,
    plan_task,
)
from arqitect.types import Action


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def recipes_dir(tmp_path):
    """Redirect _RECIPES_DIR to a temporary directory for test isolation."""
    with patch("arqitect.brain.planner._RECIPES_DIR", str(tmp_path)):
        yield tmp_path


def _write_recipe(recipes_dir, recipe: dict) -> None:
    """Write a recipe JSON file into the temporary recipes directory."""
    recipe_id = recipe.get("id", "test")
    path = recipes_dir / f"{recipe_id}.json"
    path.write_text(json.dumps(recipe))


def _make_recipe(
    recipe_id: str = "abc123",
    name: str = "test_recipe",
    category: str = "coding",
    success_rate: float = 0.8,
    runs: int = 5,
    steps: list | None = None,
) -> dict:
    """Build a minimal recipe dict for testing."""
    return {
        "id": recipe_id,
        "name": name,
        "category": category,
        "success_rate": success_rate,
        "runs": runs,
        "steps": steps if steps is not None else [{"nerve": "touch", "args_template": "echo {task}", "description": "run cmd"}],
    }


# ── match_recipe ──────────────────────────────────────────────────────────────


class TestMatchRecipe:
    """Tests for match_recipe — find the best stored recipe by category."""

    def test_returns_none_when_no_recipes(self, recipes_dir):
        assert match_recipe("build the app", "coding") is None

    def test_matches_by_category(self, recipes_dir):
        target = _make_recipe(recipe_id="r1", category="coding", success_rate=0.9)
        other = _make_recipe(recipe_id="r2", category="testing", success_rate=0.5)
        _write_recipe(recipes_dir, target)
        _write_recipe(recipes_dir, other)

        result = match_recipe("build it", "coding")
        assert result is not None
        assert result["id"] == "r1"

    def test_falls_back_to_all_when_no_category_match(self, recipes_dir):
        recipe = _make_recipe(recipe_id="r1", category="deploy", success_rate=0.7)
        _write_recipe(recipes_dir, recipe)

        result = match_recipe("build it", "nonexistent_category")
        assert result is not None
        assert result["id"] == "r1"

    def test_picks_best_by_success_rate(self, recipes_dir):
        low = _make_recipe(recipe_id="r1", category="coding", success_rate=0.3, runs=1)
        high = _make_recipe(recipe_id="r2", category="coding", success_rate=0.9, runs=1)
        _write_recipe(recipes_dir, low)
        _write_recipe(recipes_dir, high)

        result = match_recipe("build it", "coding")
        assert result["id"] == "r2"

    def test_bonus_for_runs(self, recipes_dir):
        """A recipe with many runs gets a small score bonus."""
        few_runs = _make_recipe(recipe_id="r1", category="coding", success_rate=0.85, runs=0)
        many_runs = _make_recipe(recipe_id="r2", category="coding", success_rate=0.80, runs=10)
        _write_recipe(recipes_dir, few_runs)
        _write_recipe(recipes_dir, many_runs)

        # r1 score = 0.85 (no bonus, runs=0)
        # r2 score = 0.80 + min(10/10, 0.1) = 0.90
        result = match_recipe("build it", "coding")
        assert result["id"] == "r2"

    def test_empty_category_falls_back_to_all(self, recipes_dir):
        recipe = _make_recipe(recipe_id="r1", category="coding")
        _write_recipe(recipes_dir, recipe)

        result = match_recipe("task", "")
        assert result is not None
        assert result["id"] == "r1"


# ── generate_recipe ───────────────────────────────────────────────────────────


class TestGenerateRecipe:
    """Tests for generate_recipe — LLM-generated recipe creation."""

    @patch("arqitect.brain.planner.extract_json")
    @patch("arqitect.brain.planner.llm_generate")
    def test_valid_llm_response(self, mock_llm, mock_extract):
        mock_llm.return_value = '{"name": "build_app", "steps": [{"nerve": "touch"}]}'
        mock_extract.return_value = {
            "name": "build_app",
            "description": "Build the application",
            "steps": [{"nerve": "touch", "args_template": "make build", "description": "compile"}],
        }

        result = generate_recipe("build app", "coding", {"language": "python"})

        assert result is not None
        assert result["name"] == "build_app"
        assert len(result["steps"]) == 1
        assert result["runs"] == 0
        assert result["success_rate"] == 0
        assert result["category"] == "coding"
        expected_id = hashlib.md5("coding:build app".encode()).hexdigest()[:8]
        assert result["id"] == expected_id

    @patch("arqitect.brain.planner.extract_json")
    @patch("arqitect.brain.planner.llm_generate")
    def test_returns_none_when_llm_returns_no_steps(self, mock_llm, mock_extract):
        mock_llm.return_value = "{}"
        mock_extract.return_value = {"name": "empty"}

        result = generate_recipe("task", "coding", None)
        assert result is None

    @patch("arqitect.brain.planner.extract_json")
    @patch("arqitect.brain.planner.llm_generate")
    def test_returns_none_when_extract_json_fails(self, mock_llm, mock_extract):
        mock_llm.return_value = "not json at all"
        mock_extract.return_value = None

        result = generate_recipe("task", "coding", None)
        assert result is None

    @patch("arqitect.brain.planner.extract_json")
    @patch("arqitect.brain.planner.llm_generate")
    def test_none_project_facts(self, mock_llm, mock_extract):
        mock_extract.return_value = {
            "name": "simple_recipe",
            "steps": [{"nerve": "touch", "args_template": "ls"}],
        }

        result = generate_recipe("list files", "explore", None)
        assert result is not None
        assert result["name"] == "simple_recipe"

    @patch("arqitect.brain.planner.extract_json")
    @patch("arqitect.brain.planner.llm_generate")
    def test_falls_back_to_category_name(self, mock_llm, mock_extract):
        """When LLM result has no name, falls back to '{category}_recipe'."""
        mock_extract.return_value = {
            "steps": [{"nerve": "touch", "args_template": "ls"}],
        }

        result = generate_recipe("task", "deploy", None)
        assert result is not None
        assert result["name"] == "deploy_recipe"


# ── plan_task ─────────────────────────────────────────────────────────────────


class TestPlanTask:
    """Tests for plan_task — main entry point that matches or generates."""

    @patch("arqitect.brain.planner.match_recipe")
    def test_returns_chain_when_recipe_found(self, mock_match):
        mock_match.return_value = _make_recipe()
        result = plan_task("build app", "coding", None)

        assert result is not None
        assert result["action"] == Action.CHAIN_NERVES
        assert "steps" in result

    @patch("arqitect.brain.planner.generate_recipe")
    @patch("arqitect.brain.planner.match_recipe")
    def test_generates_when_no_stored_recipe(self, mock_match, mock_generate):
        mock_match.return_value = None
        mock_generate.return_value = _make_recipe(recipe_id="gen1")

        result = plan_task("build app", "coding", None)

        assert result is not None
        assert result["action"] == Action.CHAIN_NERVES
        mock_generate.assert_called_once()

    @patch("arqitect.brain.planner.generate_recipe")
    @patch("arqitect.brain.planner.match_recipe")
    def test_returns_none_when_generation_fails(self, mock_match, mock_generate):
        mock_match.return_value = None
        mock_generate.return_value = None

        result = plan_task("impossible task", "unknown", None)
        assert result is None


# ── _recipe_to_chain ──────────────────────────────────────────────────────────


class TestRecipeToChain:
    """Tests for _recipe_to_chain — converts a recipe dict to a chain decision."""

    def test_placeholder_substitution(self):
        recipe = _make_recipe(steps=[
            {"nerve": "touch", "args_template": "build {task}", "description": "build step"},
        ])

        result = _recipe_to_chain(recipe, "the_app", {"path": "/project"})

        assert result["steps"][0]["args"] == "build the_app"

    def test_project_path_substitution(self):
        recipe = _make_recipe(steps=[
            {"nerve": "touch", "args_template": "cd {project_path} && make", "description": "build"},
        ])

        result = _recipe_to_chain(recipe, "build", {"path": "/my/project"})

        assert result["steps"][0]["args"] == "cd /my/project && make"

    def test_empty_steps(self):
        recipe = _make_recipe(steps=[])

        result = _recipe_to_chain(recipe, "task", None)

        assert result["action"] == Action.CHAIN_NERVES
        assert result["steps"] == []
        assert result["goal"] == "task"

    def test_no_project_facts(self):
        recipe = _make_recipe(steps=[
            {"nerve": "touch", "args_template": "{task}", "description": "run"},
        ])

        result = _recipe_to_chain(recipe, "hello", None)

        assert result["steps"][0]["args"] == "hello"

    def test_empty_args_template_defaults_to_task(self):
        recipe = _make_recipe(steps=[
            {"nerve": "touch", "args_template": "", "description": "run"},
        ])

        result = _recipe_to_chain(recipe, "my_task", None)

        assert result["steps"][0]["args"] == "my_task"


# ── evaluate_and_store_recipe ─────────────────────────────────────────────────


class TestEvaluateAndStoreRecipe:
    """Tests for evaluate_and_store_recipe — post-execution evaluation and persistence."""

    def test_noop_when_no_recipe_id(self, recipes_dir):
        decision = {"recipe_id": "", "recipe_name": "x", "steps": []}
        evaluate_and_store_recipe(decision, [], "task", success=True)

        assert list(recipes_dir.iterdir()) == []

    def test_stores_new_recipe_on_success(self, recipes_dir):
        decision = {
            "recipe_id": "new1",
            "recipe_name": "my_recipe",
            "goal": "build app",
            "steps": [{"nerve": "touch", "args": "make"}],
        }

        evaluate_and_store_recipe(decision, [], "build app", success=True)

        stored = json.loads((recipes_dir / "new1.json").read_text())
        assert stored["success_rate"] == 1.0
        assert stored["runs"] == 1
        assert stored["name"] == "my_recipe"

    def test_stores_new_recipe_on_failure(self, recipes_dir):
        decision = {
            "recipe_id": "fail1",
            "recipe_name": "fail_recipe",
            "goal": "deploy",
            "steps": [],
        }

        with patch("arqitect.brain.planner._suggest_improvement", return_value="try harder"):
            evaluate_and_store_recipe(decision, [], "deploy", success=False)

        stored = json.loads((recipes_dir / "fail1.json").read_text())
        assert stored["success_rate"] == 0.0
        assert stored["runs"] == 1
        assert stored["eval_notes"] == "try harder"

    def test_updates_existing_recipe_stats(self, recipes_dir):
        existing = _make_recipe(recipe_id="ex1", success_rate=1.0, runs=1)
        _write_recipe(recipes_dir, existing)

        decision = {"recipe_id": "ex1", "recipe_name": "test_recipe", "steps": []}

        evaluate_and_store_recipe(decision, [], "task", success=False)

        updated = json.loads((recipes_dir / "ex1.json").read_text())
        # Running average: ((1.0 * 1) + 0.0) / 2 = 0.5
        assert updated["success_rate"] == 0.5
        assert updated["runs"] == 2

    def test_existing_recipe_running_average_on_success(self, recipes_dir):
        existing = _make_recipe(recipe_id="ex2", success_rate=0.5, runs=2)
        _write_recipe(recipes_dir, existing)

        decision = {"recipe_id": "ex2", "recipe_name": "test_recipe", "steps": []}

        evaluate_and_store_recipe(decision, [], "task", success=True)

        updated = json.loads((recipes_dir / "ex2.json").read_text())
        # Running average: ((0.5 * 2) + 1.0) / 3 = 2.0/3 ≈ 0.667
        assert updated["success_rate"] == pytest.approx(0.667, abs=0.001)
        assert updated["runs"] == 3

    @patch("arqitect.brain.planner._suggest_improvement")
    def test_improvement_called_on_failure(self, mock_suggest, recipes_dir):
        existing = _make_recipe(recipe_id="imp1", success_rate=0.5, runs=1)
        _write_recipe(recipes_dir, existing)

        mock_suggest.return_value = "add error handling"
        decision = {"recipe_id": "imp1", "recipe_name": "test_recipe", "steps": []}
        step_results = [{"step": 0, "nerve": "touch", "status": "error"}]

        evaluate_and_store_recipe(decision, step_results, "task", success=False)

        mock_suggest.assert_called_once()
        updated = json.loads((recipes_dir / "imp1.json").read_text())
        assert "add error handling" in updated.get("eval_notes", "")

    @patch("arqitect.brain.planner._suggest_improvement")
    def test_no_improvement_on_success(self, mock_suggest, recipes_dir):
        existing = _make_recipe(recipe_id="ok1", success_rate=0.5, runs=1)
        _write_recipe(recipes_dir, existing)

        decision = {"recipe_id": "ok1", "recipe_name": "test_recipe", "steps": []}

        evaluate_and_store_recipe(decision, [], "task", success=True)

        mock_suggest.assert_not_called()
