"""TDD chain builder — constructs multi-step test-driven development chains.

Detects project paths in tasks, profiles the project, builds a 5-step
TDD chain (scout, test_writer, touch_exec, implementer, touch_exec),
and creates a TaskChecklist for progress tracking.
"""

import os
import re

from arqitect.brain.checklist import TaskChecklist


# ── Project path detection ─────────────────────────────────────────────────

def detect_project_path(task: str) -> str | None:
    """Extract an absolute project path from the task string.

    Walks up from the detected path to find the project root
    (directory containing package.json, pyproject.toml, Cargo.toml, go.mod, or .git).
    """
    # Match absolute paths like /Users/me/project or ~/project
    match = re.search(r'((?:/[\w.@-]+)+)', task)
    if not match:
        return None

    path = match.group(1)
    if not os.path.exists(path):
        return None

    # If it's a file, use its directory
    if os.path.isfile(path):
        path = os.path.dirname(path)

    # Walk up to find project root
    root_markers = (
        "package.json", "pyproject.toml", "Cargo.toml", "go.mod", ".git",
        "mix.exs", "build.gradle", "pom.xml", "CMakeLists.txt", "Makefile",
        "setup.py", "composer.json", "Gemfile", "stack.yaml", "dune-project",
        ".project",
    )
    current = os.path.abspath(path)
    while current != os.path.dirname(current):  # stop at filesystem root
        if any(os.path.exists(os.path.join(current, m)) for m in root_markers):
            return current
        current = os.path.dirname(current)

    # If the original path itself is a directory, use it even without markers
    if os.path.isdir(path):
        return os.path.abspath(path)

    return None


# ── Stack fingerprint ──────────────────────────────────────────────────────

def stack_fingerprint(role: str, project_facts: dict) -> str:
    """Build a unique, sorted nerve name from the project's stack.

    Examples:
        ("implementer", {language: "typescript", framework: "react", styling: "tailwind"})
        -> "react_tailwind_typescript_implementer"

        ("test_writer", {language: "typescript", test_framework: "vitest"})
        -> "typescript_vitest_test_writer"
    """
    tech_keys = ["language", "framework", "styling", "bundler", "test_framework"]
    # For test_writer, only include language + test_framework
    if role == "test_writer":
        tech_keys = ["language", "test_framework"]
    # For scout, include all stack info
    parts = sorted(v for k, v in project_facts.items() if k in tech_keys and v)
    if not parts:
        parts = [project_facts.get("language", "unknown")]
    return "_".join(parts + [role])


# ── Role templates for nerve descriptions ──────────────────────────────────

ROLE_TEMPLATES = {
    "scout": (
        "Read {language} project files to discover structure, "
        "{test_framework} test patterns, import conventions, and existing code style. "
        "Use touch sense to read files."
    ),
    "test_writer": (
        "Write failing {test_framework} test files for {framework} projects. "
        "Use {language} conventions. Write test file to disk using touch sense."
    ),
    "implementer": (
        "Write {framework}+{language} implementation code that passes given tests. "
        "Use {styling} for styling. Write file to disk using touch sense."
    ),
    "analyzer": (
        "Analyze {test_framework} test failures in {language} code. "
        "Extract root cause and suggest fixes."
    ),
}


def build_nerve_description(role: str, project_facts: dict) -> str:
    """Build a stack-scoped description for nerve synthesis."""
    template = ROLE_TEMPLATES.get(role, ROLE_TEMPLATES["implementer"])
    # Fill in available facts, use generic fallback for missing ones
    return template.format(
        language=project_facts.get("language", "the project's language"),
        framework=project_facts.get("framework", "the project's framework"),
        test_framework=project_facts.get("test_framework", "the project's test framework"),
        styling=project_facts.get("styling", "the project's styling approach"),
    )


# ── Test command picker ────────────────────────────────────────────────────

def _find_python(project_path: str) -> str:
    """Find the best python interpreter for a project."""
    # Check project's own venv
    for venv_dir in (".venv", "venv"):
        py = os.path.join(project_path, venv_dir, "bin", "python")
        if os.path.exists(py):
            return py
    # Fall back to arqitect's venv (has pytest installed)
    arqitect_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".venv", "bin", "python")
    if os.path.exists(arqitect_py):
        return os.path.abspath(arqitect_py)
    return "python3"


def pick_test_command(project_facts: dict) -> str:
    """Determine the right test command from project profile."""
    tf = project_facts.get("test_framework", "")
    pkg = project_facts.get("pkg_manager", "npm")
    project_path = project_facts.get("path", ".")

    # Build the runner prefix
    if pkg == "bun":
        prefix = "bunx"
    elif pkg == "pnpm":
        prefix = "pnpx"
    else:
        prefix = "npx"

    if tf == "vitest":
        return f"{prefix} vitest run {{test_file}}"
    if tf == "jest":
        return f"{prefix} jest {{test_file}}"
    if tf == "pytest":
        py = _find_python(project_path)
        return f"{py} -m pytest {{test_file}} -v"
    if tf == "mocha":
        return f"{prefix} mocha {{test_file}}"

    # Fallback: check scripts for a "test" script
    scripts = project_facts.get("scripts", "")
    if "test" in scripts.split(","):
        run_cmd = "bun run" if pkg == "bun" else f"{pkg} run"
        return f"{run_cmd} test -- {{test_file}}"

    # Last resort
    lang = project_facts.get("language", "")
    if lang == "python":
        py = _find_python(project_path)
        return f"{py} -m pytest {{test_file}} -v"
    return f"{prefix} vitest run {{test_file}}"


# ── Chain output compression ──────────────────────────────────────────────

def compress_chain_output(output: str, step_type: str, max_chars: int = 800) -> str:
    """TDD-aware truncation of chain step output."""
    if not output:
        return ""

    if len(output) <= max_chars:
        return output

    if step_type == "scout":
        # Keep file list and pattern observations, trim file contents
        lines = output.splitlines()
        kept = []
        char_count = 0
        for line in lines:
            if char_count + len(line) > max_chars:
                kept.append("... (truncated)")
                break
            kept.append(line)
            char_count += len(line) + 1
        return "\n".join(kept)

    if step_type in ("test_fail", "exec"):
        # Use LLM to interpret test output and extract the important lines
        try:
            from arqitect.brain.helpers import llm_generate
            from arqitect.brain.config import BRAIN_MODEL
            snippet = output[:2000]  # limit input size
            prompt = (
                "Given the following test output, did the tests pass or fail? "
                "If they failed, extract ONLY the most important error/failure lines. "
                "Respond with either:\n"
                "PASS\n"
                "or:\n"
                "FAIL: <extracted error lines>\n\n"
                f"Test output:\n{snippet}"
            )
            result = llm_generate(BRAIN_MODEL, prompt).strip()
            if result.upper().startswith("PASS"):
                return output[-max_chars:]  # Keep tail for passing output
            # LLM identified failures — use its extraction
            extracted = result[5:].strip() if result.upper().startswith("FAIL") else result
            return extracted[:max_chars] if extracted else output[:max_chars]
        except Exception:
            # Fallback: return truncated output
            return output[:max_chars]

    if step_type == "code":
        return output[:max_chars]

    return output[:max_chars]


# ── TDD chain builder ─────────────────────────────────────────────────────

def is_coding_task(task: str) -> bool:
    """Check if the task looks like a coding/development request using LLM classification."""
    try:
        from arqitect.brain.helpers import llm_generate
        from arqitect.brain.config import BRAIN_MODEL
        prompt = (
            "Is this a coding/development task? Answer yes or no.\n\n"
            f"Task: {task}"
        )
        result = llm_generate(BRAIN_MODEL, prompt).strip().lower()
        return result.startswith("yes")
    except Exception:
        return False


def build_tdd_chain(task: str, project_facts: dict) -> tuple[dict, TaskChecklist]:
    """Build a 5-step TDD chain + checklist for a coding task.

    Returns (chain_decision, checklist) where chain_decision is compatible
    with the brain's chain_nerves handler.
    """
    import hashlib
    task_id = hashlib.md5(task.encode()).hexdigest()[:8]

    project_path = project_facts.get("path", ".")
    test_cmd = pick_test_command(project_facts)

    # Build nerve names from stack fingerprint
    scout_name = stack_fingerprint("scout", project_facts)
    writer_name = stack_fingerprint("test_writer", project_facts)
    impl_name = stack_fingerprint("implementer", project_facts)

    # Build nerve descriptions
    scout_desc = build_nerve_description("scout", project_facts)
    writer_desc = build_nerve_description("test_writer", project_facts)
    impl_desc = build_nerve_description("implementer", project_facts)

    # Format project context for nerves
    from arqitect.knowledge.project_profiler import format_profile_for_prompt
    project_context = format_profile_for_prompt(project_facts)

    # Build concise step args — must fit in ~600 tokens for 2K context models
    lang = project_facts.get("language", "")
    framework = project_facts.get("framework", "")
    test_fw = project_facts.get("test_framework", "")
    stack_label = "+".join(filter(None, [lang, framework]))

    # Derive a sensible module name from the task (strip the path first)
    import re as _re
    task_no_path = _re.sub(r'/[\w/.@-]+', '', task).strip()  # remove paths
    task_words = _re.findall(r'[A-Z][a-z]+|[a-z]+', task_no_path)
    _stop_words = {
        "add", "create", "implement", "build", "write", "modify", "update", "fix",
        "remove", "delete", "refactor", "change",
        "a", "an", "the", "to", "with", "and", "or", "for", "in", "on", "of",
        "component", "module", "class", "function", "method", "methods",
        "that", "which", "has", "have", "using", "use", "new", "some",
    }
    # Take the first meaningful word as the module name (it's usually the main noun)
    # If it's very short (<3 chars), take up to 2 words
    meaningful = [w.lower() for w in task_words if w.lower() not in _stop_words]
    if meaningful and len(meaningful[0]) >= 3:
        module_name = meaningful[0][:30]
    elif len(meaningful) >= 2:
        module_name = "_".join(meaningful[:2])[:30]
    else:
        module_name = meaningful[0][:30] if meaningful else "feature"
    if lang == "python":
        test_file = f"tests/test_{module_name}.py"
        impl_file = f"src/{module_name}.py"
    else:
        # PascalCase for JS/TS components
        pascal = "".join(w.capitalize() for w in module_name.split("_"))
        test_file = f"src/__tests__/{pascal}.test.tsx"
        impl_file = f"src/components/{pascal}.tsx"

    steps = [
        {
            "nerve": scout_name,
            "args": f"List files at {project_path}. What test patterns and code conventions exist?",
            "description": scout_desc,
            "step_type": "scout",
        },
        {
            "nerve": writer_name,
            "args": f"Write a {test_fw or 'unit'} test for: {task}. Output ONLY the test code.",
            "description": writer_desc,
            "step_type": "test_writer",
            "target_file": os.path.join(project_path, test_file),
        },
        {
            "nerve": "_touch_exec",
            "cmd": f"cd {project_path} && {test_cmd}",
            "step_type": "test_fail",
        },
        {
            "nerve": impl_name,
            "args": f"Write {stack_label} code for: {task}. Output ONLY the implementation code.",
            "description": impl_desc,
            "step_type": "implementer",
            "target_file": os.path.join(project_path, impl_file),
        },
        {
            "nerve": "_touch_exec",
            "cmd": f"cd {project_path} && {test_cmd}",
            "step_type": "verify",
        },
    ]

    # Build the chain_nerves decision
    decision = {
        "action": "chain_nerves",
        "goal": task,
        "steps": steps,
        "tdd": True,  # Flag for the chain handler
        "project_path": project_path,
        "project_facts": project_facts,
    }

    # Build the checklist
    checklist = TaskChecklist(task_id, task, [
        {"name": "Scout codebase"},
        {"name": "Write failing test"},
        {"name": "Run test (expect fail)"},
        {"name": "Implement"},
        {"name": "Verify tests pass"},
    ])

    return decision, checklist
