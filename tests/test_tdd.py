"""Tests for arqitect.brain.tdd — TDD chain builder functions."""

import os
from unittest.mock import patch, MagicMock

import pytest

from arqitect.brain.tdd import (
    build_nerve_description,
    build_tdd_chain,
    compress_chain_output,
    detect_project_path,
    is_coding_task,
    pick_test_command,
    stack_fingerprint,
    ROLE_TEMPLATES,
    _find_python,
)


# ── detect_project_path ──────────────────────────────────────────────────────

class TestDetectProjectPath:
    """Tests for detect_project_path() — extracts project root from task string."""

    def test_no_path_in_task(self):
        assert detect_project_path("build a todo app") is None

    def test_nonexistent_path_returns_none(self):
        assert detect_project_path("fix /nonexistent/fake/path/thing") is None

    def test_existing_directory_with_marker(self, tmp_path):
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "package.json").write_text("{}")
        task = f"add feature to {project}"
        result = detect_project_path(task)
        assert result == str(project)

    def test_file_path_resolves_to_parent(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / ".git").mkdir()
        src = project / "src"
        src.mkdir()
        afile = src / "app.py"
        afile.write_text("pass")
        task = f"edit {afile}"
        result = detect_project_path(task)
        assert result == str(project)

    def test_directory_without_marker_uses_path_itself(self, tmp_path):
        bare = tmp_path / "bare_dir"
        bare.mkdir()
        task = f"work on {bare}"
        result = detect_project_path(task)
        assert result == str(bare)

    def test_walks_up_to_find_root(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "pyproject.toml").write_text("")
        nested = root / "a" / "b" / "c"
        nested.mkdir(parents=True)
        task = f"run tests in {nested}"
        result = detect_project_path(task)
        assert result == str(root)

    def test_various_root_markers(self, tmp_path):
        for marker in ("Cargo.toml", "go.mod", "Gemfile", "setup.py"):
            project = tmp_path / f"proj_{marker.replace('.', '_')}"
            project.mkdir()
            (project / marker).write_text("")
            result = detect_project_path(f"do stuff in {project}")
            assert result == str(project), f"Failed for marker {marker}"


# ── stack_fingerprint ─────────────────────────────────────────────────────────

class TestStackFingerprint:
    """Tests for stack_fingerprint() — builds sorted nerve name from project stack."""

    def test_full_stack(self):
        facts = {"language": "typescript", "framework": "react", "styling": "tailwind"}
        result = stack_fingerprint("implementer", facts)
        assert result == "react_tailwind_typescript_implementer"

    def test_test_writer_uses_subset(self):
        facts = {"language": "typescript", "framework": "react", "test_framework": "vitest"}
        result = stack_fingerprint("test_writer", facts)
        assert result == "typescript_vitest_test_writer"

    def test_empty_facts_fallback_to_language(self):
        facts = {"language": "python"}
        result = stack_fingerprint("scout", facts)
        assert result == "python_scout"

    def test_no_language_fallback_unknown(self):
        result = stack_fingerprint("scout", {})
        assert result == "unknown_scout"

    def test_sorted_alphabetically(self):
        facts = {"language": "python", "framework": "django", "styling": "bootstrap"}
        result = stack_fingerprint("implementer", facts)
        parts = result.replace("_implementer", "").split("_")
        assert parts == sorted(parts)

    def test_empty_values_excluded(self):
        facts = {"language": "python", "framework": "", "styling": ""}
        result = stack_fingerprint("implementer", facts)
        assert result == "python_implementer"


# ── build_nerve_description ───────────────────────────────────────────────────

class TestBuildNerveDescription:
    """Tests for build_nerve_description() — template-based description builder."""

    def test_scout_role(self):
        facts = {"language": "python", "test_framework": "pytest"}
        desc = build_nerve_description("scout", facts)
        assert "python" in desc
        assert "pytest" in desc

    def test_test_writer_role(self):
        facts = {"language": "typescript", "framework": "react", "test_framework": "vitest"}
        desc = build_nerve_description("test_writer", facts)
        assert "vitest" in desc
        assert "typescript" in desc

    def test_unknown_role_falls_back_to_implementer(self):
        facts = {"language": "go", "framework": "gin"}
        desc = build_nerve_description("unknown_role", facts)
        # Falls back to implementer template
        assert "gin" in desc or "go" in desc

    def test_missing_facts_use_generic_fallback(self):
        desc = build_nerve_description("scout", {})
        assert "the project's language" in desc


# ── pick_test_command ─────────────────────────────────────────────────────────

class TestPickTestCommand:
    """Tests for pick_test_command() — determines the test runner command."""

    def test_vitest(self):
        cmd = pick_test_command({"test_framework": "vitest"})
        assert "vitest run" in cmd
        assert "{test_file}" in cmd

    def test_jest(self):
        cmd = pick_test_command({"test_framework": "jest"})
        assert "jest" in cmd

    def test_pytest(self):
        cmd = pick_test_command({"test_framework": "pytest", "path": "/nonexistent"})
        assert "pytest" in cmd
        assert "-v" in cmd

    def test_mocha(self):
        cmd = pick_test_command({"test_framework": "mocha"})
        assert "mocha" in cmd

    def test_bun_prefix(self):
        cmd = pick_test_command({"test_framework": "vitest", "pkg_manager": "bun"})
        assert cmd.startswith("bunx")

    def test_pnpm_prefix(self):
        cmd = pick_test_command({"test_framework": "vitest", "pkg_manager": "pnpm"})
        assert cmd.startswith("pnpx")

    def test_npm_fallback_prefix(self):
        cmd = pick_test_command({"test_framework": "vitest", "pkg_manager": "npm"})
        assert cmd.startswith("npx")

    def test_fallback_with_test_script(self):
        facts = {"test_framework": "", "scripts": "start,test,build"}
        cmd = pick_test_command(facts)
        assert "npm run test" in cmd

    def test_fallback_python_language(self):
        cmd = pick_test_command({"language": "python", "path": "/nonexistent"})
        assert "pytest" in cmd

    def test_last_resort_vitest(self):
        cmd = pick_test_command({})
        assert "vitest" in cmd


# ── compress_chain_output ─────────────────────────────────────────────────────

class TestCompressChainOutput:
    """Tests for compress_chain_output() — TDD-aware truncation."""

    def test_empty_output(self):
        assert compress_chain_output("", "scout") == ""

    def test_short_output_unchanged(self):
        short = "hello world"
        assert compress_chain_output(short, "scout") == short

    def test_scout_truncates_by_lines(self):
        long_output = "\n".join(f"file_{i}.py" for i in range(200))
        result = compress_chain_output(long_output, "scout", max_chars=100)
        assert len(result) <= 150  # some overhead for "... (truncated)"
        assert "... (truncated)" in result

    def test_code_step_truncates_at_max(self):
        long_code = "x" * 2000
        result = compress_chain_output(long_code, "code", max_chars=500)
        assert len(result) == 500

    def test_test_fail_with_llm_failure_fallback(self):
        """When LLM import fails, falls back to simple truncation."""
        long_output = "FAIL " + "x" * 2000
        with patch.dict("sys.modules", {"arqitect.brain.helpers": None}):
            result = compress_chain_output(long_output, "test_fail", max_chars=500)
        assert len(result) <= 500

    def test_unknown_step_type_truncates(self):
        long_output = "a" * 2000
        result = compress_chain_output(long_output, "whatever", max_chars=300)
        assert len(result) == 300


# ── is_coding_task ────────────────────────────────────────────────────────────

class TestIsCodingTask:
    """Tests for is_coding_task() — LLM-based classification."""

    def test_yes_response(self):
        with patch("arqitect.brain.helpers.llm_generate", return_value="yes"):
            assert is_coding_task("add a login form") is True

    def test_no_response(self):
        with patch("arqitect.brain.helpers.llm_generate", return_value="no"):
            assert is_coding_task("what is 2+2") is False

    def test_llm_failure_returns_false(self):
        """When the LLM is unavailable, default to False."""
        with patch("arqitect.brain.helpers.llm_generate", side_effect=Exception("no llm")):
            assert is_coding_task("add a feature") is False

    def test_yes_prefix_accepted(self):
        with patch("arqitect.brain.helpers.llm_generate", return_value="Yes, it is"):
            assert is_coding_task("build an API") is True


# ── build_tdd_chain ──────────────────────────────────────────────────────────

class TestBuildTddChain:
    """Tests for build_tdd_chain() — constructs 5-step TDD chain + checklist."""

    @pytest.fixture()
    def project_facts(self):
        return {
            "language": "typescript",
            "framework": "react",
            "styling": "tailwind",
            "test_framework": "vitest",
            "pkg_manager": "npm",
            "path": "/tmp/myproject",
        }

    def test_returns_decision_and_checklist(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, checklist = build_tdd_chain("add a navbar component", project_facts)
        assert decision["action"] == "chain_nerves"
        assert decision["tdd"] is True
        assert len(decision["steps"]) == 5

    def test_step_types_correct(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain("add a sidebar", project_facts)
        step_types = [s["step_type"] for s in decision["steps"]]
        assert step_types == ["scout", "test_writer", "test_fail", "implementer", "verify"]

    def test_checklist_has_five_steps(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            _, checklist = build_tdd_chain("add a sidebar", project_facts)
        assert len(checklist.steps) == 5
        assert all(s["status"] == "pending" for s in checklist.steps)

    def test_python_project_paths(self):
        facts = {
            "language": "python",
            "framework": "fastapi",
            "test_framework": "pytest",
            "path": "/tmp/pyproj",
        }
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain("add authentication middleware", facts)
        # Python projects should use tests/test_*.py and src/*.py
        writer_step = decision["steps"][1]
        impl_step = decision["steps"][3]
        assert writer_step["target_file"].startswith("/tmp/pyproj/tests/test_")
        assert impl_step["target_file"].startswith("/tmp/pyproj/src/")

    def test_js_project_paths(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain("add a navbar", project_facts)
        writer_step = decision["steps"][1]
        impl_step = decision["steps"][3]
        assert "__tests__" in writer_step["target_file"]
        assert "components" in impl_step["target_file"]

    def test_goal_matches_task(self, project_facts):
        task = "implement dark mode toggle"
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain(task, project_facts)
        assert decision["goal"] == task

    def test_touch_exec_steps_have_cmd(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain("add button", project_facts)
        exec_steps = [s for s in decision["steps"] if s["nerve"] == "_touch_exec"]
        assert len(exec_steps) == 2
        for step in exec_steps:
            assert "cmd" in step

    def test_task_id_deterministic(self, project_facts):
        task = "add a widget"
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            _, cl1 = build_tdd_chain(task, project_facts)
            _, cl2 = build_tdd_chain(task, project_facts)
        assert cl1.task_id == cl2.task_id


# ── _find_python ──────────────────────────────────────────────────────────────

class TestFindPython:
    """Tests for _find_python() — locates the best python interpreter."""

    def test_finds_venv_python(self, tmp_path):
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        py = venv_bin / "python"
        py.write_text("#!/usr/bin/env python3")
        result = _find_python(str(tmp_path))
        assert result == str(py)

    def test_prefers_dot_venv_over_venv(self, tmp_path):
        for name in (".venv", "venv"):
            d = tmp_path / name / "bin"
            d.mkdir(parents=True)
            (d / "python").write_text("#!/usr/bin/env python3")
        result = _find_python(str(tmp_path))
        assert ".venv" in result

    def test_fallback_to_python3(self, tmp_path):
        result = _find_python(str(tmp_path))
        # No venv exists, and arqitect venv likely doesn't either in test => python3
        assert "python" in result
