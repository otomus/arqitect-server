"""Tests for arqitect.brain.tdd — TDD chain builder functions.

Contract-based tests exercising detect_project_path, stack_fingerprint,
build_nerve_description, pick_test_command, compress_chain_output,
is_coding_task, build_tdd_chain, and _find_python.
"""

import hashlib
from unittest.mock import patch

import pytest
from dirty_equals import IsStr, HasLen
from hypothesis import given, settings, assume
from hypothesis import strategies as st

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

from tests.conftest import FakeLLM


# ── detect_project_path ──────────────────────────────────────────────────────

class TestDetectProjectPath:
    """Tests for detect_project_path() — extracts project root from task string."""

    @pytest.mark.timeout(10)
    def test_no_path_in_task(self):
        assert detect_project_path("build a todo app") is None

    @pytest.mark.timeout(10)
    def test_nonexistent_path_returns_none(self):
        assert detect_project_path("fix /nonexistent/fake/path/thing") is None

    @pytest.mark.timeout(10)
    def test_existing_directory_with_marker(self, tmp_path):
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "package.json").write_text("{}")
        task = f"add feature to {project}"
        result = detect_project_path(task)
        assert result == str(project)

    @pytest.mark.timeout(10)
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

    @pytest.mark.timeout(10)
    def test_directory_without_marker_uses_path_itself(self, tmp_path):
        bare = tmp_path / "bare_dir"
        bare.mkdir()
        task = f"work on {bare}"
        result = detect_project_path(task)
        assert result == str(bare)

    @pytest.mark.timeout(10)
    def test_walks_up_to_find_root(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "pyproject.toml").write_text("")
        nested = root / "a" / "b" / "c"
        nested.mkdir(parents=True)
        task = f"run tests in {nested}"
        result = detect_project_path(task)
        assert result == str(root)

    @pytest.mark.timeout(10)
    @pytest.mark.parametrize("marker", [
        "Cargo.toml", "go.mod", "Gemfile", "setup.py",
    ])
    def test_various_root_markers(self, tmp_path, marker):
        project = tmp_path / f"proj_{marker.replace('.', '_')}"
        project.mkdir()
        (project / marker).write_text("")
        result = detect_project_path(f"do stuff in {project}")
        assert result == str(project)


# ── stack_fingerprint ─────────────────────────────────────────────────────────

class TestStackFingerprint:
    """Tests for stack_fingerprint() — builds sorted nerve name from project stack."""

    @pytest.mark.timeout(10)
    def test_full_stack(self):
        facts = {"language": "typescript", "framework": "react", "styling": "tailwind"}
        result = stack_fingerprint("implementer", facts)
        assert result == "react_tailwind_typescript_implementer"

    @pytest.mark.timeout(10)
    def test_test_writer_uses_subset(self):
        facts = {"language": "typescript", "framework": "react", "test_framework": "vitest"}
        result = stack_fingerprint("test_writer", facts)
        assert result == "typescript_vitest_test_writer"

    @pytest.mark.timeout(10)
    def test_empty_facts_fallback_to_language(self):
        facts = {"language": "python"}
        result = stack_fingerprint("scout", facts)
        assert result == "python_scout"

    @pytest.mark.timeout(10)
    def test_no_language_fallback_unknown(self):
        result = stack_fingerprint("scout", {})
        assert result == "unknown_scout"

    @pytest.mark.timeout(10)
    def test_sorted_alphabetically(self):
        facts = {"language": "python", "framework": "django", "styling": "bootstrap"}
        result = stack_fingerprint("implementer", facts)
        parts = result.replace("_implementer", "").split("_")
        assert parts == sorted(parts)

    @pytest.mark.timeout(10)
    def test_empty_values_excluded(self):
        facts = {"language": "python", "framework": "", "styling": ""}
        result = stack_fingerprint("implementer", facts)
        assert result == "python_implementer"

    @pytest.mark.timeout(10)
    @given(role=st.sampled_from(["scout", "test_writer", "implementer", "analyzer"]))
    @settings(max_examples=20)
    def test_fingerprint_always_ends_with_role(self, role):
        """The fingerprint must always end with _<role>."""
        facts = {"language": "python", "framework": "flask"}
        result = stack_fingerprint(role, facts)
        assert result.endswith(f"_{role}")

    @pytest.mark.timeout(10)
    @given(
        lang=st.sampled_from(["python", "typescript", "go", "rust"]),
        framework=st.sampled_from(["react", "django", "flask", "gin", ""]),
    )
    @settings(max_examples=20)
    def test_fingerprint_parts_always_sorted(self, lang, framework):
        """Tech parts (before the role suffix) are always alphabetically sorted."""
        facts = {"language": lang, "framework": framework}
        result = stack_fingerprint("implementer", facts)
        tech_part = result.removesuffix("_implementer")
        parts = tech_part.split("_")
        assert parts == sorted(parts)


# ── build_nerve_description ───────────────────────────────────────────────────

class TestBuildNerveDescription:
    """Tests for build_nerve_description() — template-based description builder."""

    @pytest.mark.timeout(10)
    def test_scout_role(self):
        facts = {"language": "python", "test_framework": "pytest"}
        desc = build_nerve_description("scout", facts)
        assert "python" in desc
        assert "pytest" in desc

    @pytest.mark.timeout(10)
    def test_test_writer_role(self):
        facts = {"language": "typescript", "framework": "react", "test_framework": "vitest"}
        desc = build_nerve_description("test_writer", facts)
        assert "vitest" in desc
        assert "typescript" in desc

    @pytest.mark.timeout(10)
    def test_unknown_role_falls_back_to_implementer(self):
        facts = {"language": "go", "framework": "gin"}
        desc = build_nerve_description("unknown_role", facts)
        # Falls back to implementer template
        assert "gin" in desc or "go" in desc

    @pytest.mark.timeout(10)
    def test_missing_facts_use_generic_fallback(self):
        desc = build_nerve_description("scout", {})
        assert "the project's language" in desc

    @pytest.mark.timeout(10)
    @given(role=st.sampled_from(list(ROLE_TEMPLATES.keys())))
    @settings(max_examples=10)
    def test_description_is_nonempty_string(self, role):
        """Every known role produces a non-empty description."""
        facts = {"language": "python", "framework": "django", "test_framework": "pytest"}
        desc = build_nerve_description(role, facts)
        assert desc == IsStr(min_length=10)


# ── pick_test_command ─────────────────────────────────────────────────────────

class TestPickTestCommand:
    """Tests for pick_test_command() — determines the test runner command."""

    @pytest.mark.timeout(10)
    def test_vitest(self):
        cmd = pick_test_command({"test_framework": "vitest"})
        assert "vitest run" in cmd
        assert "{test_file}" in cmd

    @pytest.mark.timeout(10)
    def test_jest(self):
        cmd = pick_test_command({"test_framework": "jest"})
        assert "jest" in cmd

    @pytest.mark.timeout(10)
    def test_pytest(self):
        cmd = pick_test_command({"test_framework": "pytest", "path": "/nonexistent"})
        assert "pytest" in cmd
        assert "-v" in cmd

    @pytest.mark.timeout(10)
    def test_mocha(self):
        cmd = pick_test_command({"test_framework": "mocha"})
        assert "mocha" in cmd

    @pytest.mark.timeout(10)
    def test_bun_prefix(self):
        cmd = pick_test_command({"test_framework": "vitest", "pkg_manager": "bun"})
        assert cmd.startswith("bunx")

    @pytest.mark.timeout(10)
    def test_pnpm_prefix(self):
        cmd = pick_test_command({"test_framework": "vitest", "pkg_manager": "pnpm"})
        assert cmd.startswith("pnpx")

    @pytest.mark.timeout(10)
    def test_npm_fallback_prefix(self):
        cmd = pick_test_command({"test_framework": "vitest", "pkg_manager": "npm"})
        assert cmd.startswith("npx")

    @pytest.mark.timeout(10)
    def test_fallback_with_test_script(self):
        facts = {"test_framework": "", "scripts": "start,test,build"}
        cmd = pick_test_command(facts)
        assert "npm run test" in cmd

    @pytest.mark.timeout(10)
    def test_fallback_python_language(self):
        cmd = pick_test_command({"language": "python", "path": "/nonexistent"})
        assert "pytest" in cmd

    @pytest.mark.timeout(10)
    def test_last_resort_vitest(self):
        cmd = pick_test_command({})
        assert "vitest" in cmd

    @pytest.mark.timeout(10)
    @given(
        tf=st.sampled_from(["vitest", "jest", "pytest", "mocha", ""]),
        pkg=st.sampled_from(["npm", "bun", "pnpm", "yarn"]),
    )
    @settings(max_examples=20)
    def test_command_always_contains_test_file_placeholder(self, tf, pkg):
        """Every generated test command must contain {test_file}."""
        cmd = pick_test_command({"test_framework": tf, "pkg_manager": pkg, "path": "/tmp"})
        assert "{test_file}" in cmd


# ── compress_chain_output ─────────────────────────────────────────────────────

class TestCompressChainOutput:
    """Tests for compress_chain_output() — TDD-aware truncation."""

    @pytest.mark.timeout(10)
    def test_empty_output(self):
        assert compress_chain_output("", "scout") == ""

    @pytest.mark.timeout(10)
    def test_short_output_unchanged(self):
        short = "hello world"
        assert compress_chain_output(short, "scout") == short

    @pytest.mark.timeout(10)
    def test_scout_truncates_by_lines(self):
        long_output = "\n".join(f"file_{i}.py" for i in range(200))
        result = compress_chain_output(long_output, "scout", max_chars=100)
        assert len(result) <= 150  # some overhead for "... (truncated)"
        assert "... (truncated)" in result

    @pytest.mark.timeout(10)
    def test_code_step_truncates_at_max(self):
        long_code = "x" * 2000
        result = compress_chain_output(long_code, "code", max_chars=500)
        assert len(result) == 500

    @pytest.mark.timeout(10)
    def test_test_fail_with_llm_failure_fallback(self):
        """When LLM import fails, falls back to simple truncation."""
        long_output = "FAIL " + "x" * 2000
        with patch.dict("sys.modules", {"arqitect.brain.helpers": None}):
            result = compress_chain_output(long_output, "test_fail", max_chars=500)
        assert len(result) <= 500

    @pytest.mark.timeout(10)
    def test_unknown_step_type_truncates(self):
        long_output = "a" * 2000
        result = compress_chain_output(long_output, "whatever", max_chars=300)
        assert len(result) == 300

    @pytest.mark.timeout(10)
    @given(
        length=st.integers(min_value=1000, max_value=5000),
        max_chars=st.integers(min_value=50, max_value=999),
    )
    @settings(max_examples=20)
    def test_output_never_exceeds_max_for_code_step(self, length, max_chars):
        """Code step truncation always respects max_chars exactly."""
        output = "x" * length
        result = compress_chain_output(output, "code", max_chars=max_chars)
        assert len(result) == max_chars


# ── is_coding_task ────────────────────────────────────────────────────────────

class TestIsCodingTask:
    """Tests for is_coding_task() — LLM-based classification."""

    @pytest.mark.timeout(10)
    def test_yes_response(self):
        fake = FakeLLM([("coding", "yes")])
        with patch("arqitect.brain.helpers.llm_generate", side_effect=fake):
            assert is_coding_task("add a login form") is True

    @pytest.mark.timeout(10)
    def test_no_response(self):
        fake = FakeLLM([("coding", "no")])
        with patch("arqitect.brain.helpers.llm_generate", side_effect=fake):
            assert is_coding_task("what is 2+2") is False

    @pytest.mark.timeout(10)
    def test_llm_failure_returns_false(self):
        """When the LLM is unavailable, default to False."""
        with patch("arqitect.brain.helpers.llm_generate", side_effect=Exception("no llm")):
            assert is_coding_task("add a feature") is False

    @pytest.mark.timeout(10)
    def test_yes_prefix_accepted(self):
        fake = FakeLLM([("coding", "Yes, it is")])
        with patch("arqitect.brain.helpers.llm_generate", side_effect=fake):
            assert is_coding_task("build an API") is True


# ── build_tdd_chain ──────────────────────────────────────────────────────────

class TestBuildTddChain:
    """Tests for build_tdd_chain() — constructs 5-step TDD chain + checklist."""

    EXPECTED_STEP_TYPES = ["scout", "test_writer", "test_fail", "implementer", "verify"]

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

    @pytest.mark.timeout(10)
    def test_returns_decision_and_checklist(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, checklist = build_tdd_chain("add a navbar component", project_facts)
        assert decision["action"] == "chain_nerves"
        assert decision["tdd"] is True
        assert decision["steps"] == HasLen(5)

    @pytest.mark.timeout(10)
    def test_step_types_correct(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain("add a sidebar", project_facts)
        step_types = [s["step_type"] for s in decision["steps"]]
        assert step_types == self.EXPECTED_STEP_TYPES

    @pytest.mark.timeout(10)
    def test_checklist_has_five_steps(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            _, checklist = build_tdd_chain("add a sidebar", project_facts)
        assert checklist.steps == HasLen(5)
        assert all(s["status"] == "pending" for s in checklist.steps)

    @pytest.mark.timeout(10)
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
        assert writer_step["target_file"] == IsStr(regex=r".*/tests/test_.*\.py$")
        assert impl_step["target_file"] == IsStr(regex=r".*/src/.*\.py$")

    @pytest.mark.timeout(10)
    def test_js_project_paths(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain("add a navbar", project_facts)
        writer_step = decision["steps"][1]
        impl_step = decision["steps"][3]
        assert "__tests__" in writer_step["target_file"]
        assert "components" in impl_step["target_file"]

    @pytest.mark.timeout(10)
    def test_goal_matches_task(self, project_facts):
        task = "implement dark mode toggle"
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain(task, project_facts)
        assert decision["goal"] == task

    @pytest.mark.timeout(10)
    def test_touch_exec_steps_have_cmd(self, project_facts):
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, _ = build_tdd_chain("add button", project_facts)
        exec_steps = [s for s in decision["steps"] if s["nerve"] == "_touch_exec"]
        assert exec_steps == HasLen(2)
        for step in exec_steps:
            assert "cmd" in step

    @pytest.mark.timeout(10)
    def test_task_id_deterministic(self, project_facts):
        task = "add a widget"
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            _, cl1 = build_tdd_chain(task, project_facts)
            _, cl2 = build_tdd_chain(task, project_facts)
        assert cl1.task_id == cl2.task_id

    @pytest.mark.timeout(10)
    def test_task_id_matches_md5_prefix(self, project_facts):
        """The task_id is the first 8 hex chars of the MD5 of the task string."""
        task = "add a widget"
        expected_id = hashlib.md5(task.encode()).hexdigest()[:8]
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            _, checklist = build_tdd_chain(task, project_facts)
        assert checklist.task_id == expected_id

    @pytest.mark.timeout(10)
    @given(task=st.text(min_size=5, max_size=80, alphabet=st.characters(whitelist_categories=("L", "Zs"))))
    @settings(max_examples=15)
    def test_chain_always_has_five_steps(self, task):
        """No matter the task text, the chain always produces exactly 5 steps."""
        assume(len(task.strip()) >= 3)
        facts = {
            "language": "typescript",
            "framework": "react",
            "styling": "tailwind",
            "test_framework": "vitest",
            "pkg_manager": "npm",
            "path": "/tmp/myproject",
        }
        with patch("arqitect.knowledge.project_profiler.format_profile_for_prompt", return_value="ctx"):
            decision, checklist = build_tdd_chain(task.strip(), facts)
        assert len(decision["steps"]) == 5
        assert len(checklist.steps) == 5


# ── _find_python ──────────────────────────────────────────────────────────────

class TestFindPython:
    """Tests for _find_python() — locates the best python interpreter."""

    @pytest.mark.timeout(10)
    def test_finds_venv_python(self, tmp_path):
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        py = venv_bin / "python"
        py.write_text("#!/usr/bin/env python3")
        result = _find_python(str(tmp_path))
        assert result == str(py)

    @pytest.mark.timeout(10)
    def test_prefers_dot_venv_over_venv(self, tmp_path):
        for name in (".venv", "venv"):
            d = tmp_path / name / "bin"
            d.mkdir(parents=True)
            (d / "python").write_text("#!/usr/bin/env python3")
        result = _find_python(str(tmp_path))
        assert ".venv" in result

    @pytest.mark.timeout(10)
    def test_fallback_to_python3(self, tmp_path):
        result = _find_python(str(tmp_path))
        # No venv exists => python3 or arqitect venv
        assert "python" in result
