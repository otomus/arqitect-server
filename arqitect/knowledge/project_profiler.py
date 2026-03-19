"""
Project Profiler — deterministic scanner that extracts structured facts from any project.

No LLM needed. Reads config files (package.json, pyproject.toml, Cargo.toml, etc.),
maps directory structure, and produces compressed project facts stored in cold memory
under category="project:<abs_path>".

Usage:
    from arqitect.knowledge.project_profiler import profile_project
    facts = profile_project("/path/to/project")

Or as CLI:
    python knowledge/project_profiler.py /path/to/project
"""

import json
import os
import sys



# ── Config file detectors ────────────────────────────────────────────────────

def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _read_toml(path: str) -> dict:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {}
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _detect_from_package_json(project_path: str) -> dict:
    """Extract facts from package.json."""
    pj = _read_json(os.path.join(project_path, "package.json"))
    if not pj:
        return {}

    facts = {"language": "javascript"}
    deps = {**pj.get("dependencies", {}), **pj.get("devDependencies", {})}

    # Runtime / framework
    if "next" in deps:
        facts["framework"] = "next.js"
    elif "nuxt" in deps:
        facts["framework"] = "nuxt"
    elif "svelte" in deps or "@sveltejs/kit" in deps:
        facts["framework"] = "svelte"
    elif "react" in deps:
        facts["framework"] = "react"
    elif "vue" in deps:
        facts["framework"] = "vue"
    elif "express" in deps:
        facts["framework"] = "express"
    elif "fastify" in deps:
        facts["framework"] = "fastify"

    # TypeScript
    if "typescript" in deps or os.path.exists(os.path.join(project_path, "tsconfig.json")):
        facts["language"] = "typescript"

    # Bundler
    if "vite" in deps:
        facts["bundler"] = "vite"
    elif "webpack" in deps or "webpack-cli" in deps:
        facts["bundler"] = "webpack"
    elif "esbuild" in deps:
        facts["bundler"] = "esbuild"
    elif "turbopack" in deps or "turbo" in deps:
        facts["bundler"] = "turbo"

    # CSS / styling
    if "tailwindcss" in deps:
        facts["styling"] = "tailwind"
    elif "styled-components" in deps:
        facts["styling"] = "styled-components"
    elif "@emotion/react" in deps:
        facts["styling"] = "emotion"
    elif "sass" in deps or "node-sass" in deps:
        facts["styling"] = "sass"

    # Test framework
    if "vitest" in deps:
        facts["test_framework"] = "vitest"
    elif "jest" in deps:
        facts["test_framework"] = "jest"
    elif "mocha" in deps:
        facts["test_framework"] = "mocha"
    elif "@playwright/test" in deps:
        facts["test_framework"] = "playwright"
    elif "cypress" in deps:
        facts["test_framework"] = "cypress"

    # State management
    if "zustand" in deps:
        facts["state_mgmt"] = "zustand"
    elif "redux" in deps or "@reduxjs/toolkit" in deps:
        facts["state_mgmt"] = "redux"
    elif "mobx" in deps:
        facts["state_mgmt"] = "mobx"
    elif "pinia" in deps:
        facts["state_mgmt"] = "pinia"

    # Package manager
    if os.path.exists(os.path.join(project_path, "bun.lockb")) or os.path.exists(os.path.join(project_path, "bun.lock")):
        facts["pkg_manager"] = "bun"
    elif os.path.exists(os.path.join(project_path, "pnpm-lock.yaml")):
        facts["pkg_manager"] = "pnpm"
    elif os.path.exists(os.path.join(project_path, "yarn.lock")):
        facts["pkg_manager"] = "yarn"
    else:
        facts["pkg_manager"] = "npm"

    # Scripts
    scripts = pj.get("scripts", {})
    if scripts:
        facts["scripts"] = ",".join(sorted(scripts.keys()))

    # Entry point
    if pj.get("main"):
        facts["entry_point"] = pj["main"]

    # Key deps (top 15 by name, excluding dev tooling noise)
    dep_names = sorted(pj.get("dependencies", {}).keys())[:15]
    if dep_names:
        facts["dependencies"] = ",".join(dep_names)

    return facts


def _detect_from_pyproject(project_path: str) -> dict:
    """Extract facts from pyproject.toml or setup.py/setup.cfg."""
    facts = {"language": "python"}

    toml_path = os.path.join(project_path, "pyproject.toml")
    if os.path.exists(toml_path):
        data = _read_toml(toml_path)
        project = data.get("project", {})
        deps = project.get("dependencies", [])
        build = data.get("build-system", {})

        # Framework detection from deps
        dep_str = " ".join(deps).lower() if isinstance(deps, list) else ""
        if "django" in dep_str:
            facts["framework"] = "django"
        elif "flask" in dep_str:
            facts["framework"] = "flask"
        elif "fastapi" in dep_str:
            facts["framework"] = "fastapi"
        elif "starlette" in dep_str:
            facts["framework"] = "starlette"

        # Build system
        requires = build.get("requires", [])
        req_str = " ".join(requires).lower() if isinstance(requires, list) else ""
        if "hatchling" in req_str:
            facts["build_tool"] = "hatch"
        elif "setuptools" in req_str:
            facts["build_tool"] = "setuptools"
        elif "poetry" in req_str:
            facts["build_tool"] = "poetry"
        elif "flit" in req_str:
            facts["build_tool"] = "flit"

        # Test framework
        tool = data.get("tool", {})
        if "pytest" in tool or "pytest" in dep_str:
            facts["test_framework"] = "pytest"

        # Key deps
        if isinstance(deps, list) and deps:
            clean = [d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip() for d in deps[:15]]
            facts["dependencies"] = ",".join(clean)

    # requirements.txt fallback
    req_path = os.path.join(project_path, "requirements.txt")
    if "dependencies" not in facts and os.path.exists(req_path):
        try:
            with open(req_path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            deps = [l.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip() for l in lines[:15]]
            facts["dependencies"] = ",".join(deps)
            dep_str = " ".join(deps).lower()
            if "django" in dep_str:
                facts.setdefault("framework", "django")
            elif "flask" in dep_str:
                facts.setdefault("framework", "flask")
            elif "fastapi" in dep_str:
                facts.setdefault("framework", "fastapi")
        except Exception:
            pass

    # Virtual env
    if os.path.exists(os.path.join(project_path, ".venv")):
        facts["venv"] = ".venv"
    elif os.path.exists(os.path.join(project_path, "venv")):
        facts["venv"] = "venv"

    # UV / poetry
    if os.path.exists(os.path.join(project_path, "uv.lock")):
        facts["pkg_manager"] = "uv"
    elif os.path.exists(os.path.join(project_path, "poetry.lock")):
        facts["pkg_manager"] = "poetry"
    else:
        facts["pkg_manager"] = "pip"

    return facts


def _detect_from_cargo(project_path: str) -> dict:
    """Extract facts from Cargo.toml."""
    data = _read_toml(os.path.join(project_path, "Cargo.toml"))
    if not data:
        return {}
    facts = {"language": "rust"}
    deps = data.get("dependencies", {})
    if deps:
        facts["dependencies"] = ",".join(sorted(deps.keys())[:15])
    if "actix-web" in deps:
        facts["framework"] = "actix"
    elif "axum" in deps:
        facts["framework"] = "axum"
    elif "rocket" in deps:
        facts["framework"] = "rocket"
    return facts


def _detect_from_go_mod(project_path: str) -> dict:
    """Extract facts from go.mod."""
    mod_path = os.path.join(project_path, "go.mod")
    if not os.path.exists(mod_path):
        return {}
    facts = {"language": "go"}
    try:
        with open(mod_path) as f:
            content = f.read()
        # Extract module name
        for line in content.splitlines():
            if line.startswith("module "):
                facts["module"] = line.split()[1]
                break
        # Extract deps
        deps = []
        in_require = False
        for line in content.splitlines():
            if line.strip().startswith("require ("):
                in_require = True
                continue
            if in_require and line.strip() == ")":
                in_require = False
                continue
            if in_require:
                parts = line.strip().split()
                if parts:
                    deps.append(parts[0].split("/")[-1])
        if deps:
            facts["dependencies"] = ",".join(deps[:15])
        dep_str = content.lower()
        if "gin-gonic" in dep_str:
            facts["framework"] = "gin"
        elif "fiber" in dep_str:
            facts["framework"] = "fiber"
        elif "echo" in dep_str:
            facts["framework"] = "echo"
    except Exception:
        pass
    return facts


# ── Structure scanner ─────────────────────────────────────────────────────────

def _scan_structure(project_path: str, max_depth: int = 3) -> dict:
    """Scan directory structure up to max_depth. Returns summary facts."""
    top_dirs = []
    file_extensions = {}
    entry_candidates = []

    for root, dirs, files in os.walk(project_path):
        depth = root.replace(project_path, "").count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue
        # Skip common noise
        dirs[:] = [d for d in dirs if d not in (
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            ".next", ".nuxt", "dist", "build", ".cache", "coverage",
            ".pytest_cache", ".mypy_cache", "target", ".svelte-kit",
        )]

        if depth == 1:
            rel = os.path.relpath(root, project_path)
            top_dirs.append(rel)

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext:
                file_extensions[ext] = file_extensions.get(ext, 0) + 1

            # Common entry points
            if depth <= 1 and f in (
                "index.ts", "index.tsx", "index.js", "main.ts", "main.tsx",
                "main.py", "app.py", "manage.py", "main.go", "main.rs",
                "App.tsx", "App.jsx", "App.vue", "App.svelte",
            ):
                entry_candidates.append(os.path.join(os.path.relpath(root, project_path), f))

    facts = {}
    if top_dirs:
        facts["top_dirs"] = ",".join(sorted(top_dirs)[:15])

    # Most common extensions (top 5)
    sorted_ext = sorted(file_extensions.items(), key=lambda x: -x[1])[:5]
    if sorted_ext:
        facts["file_types"] = ",".join(f"{ext}({c})" for ext, c in sorted_ext)

    if entry_candidates:
        facts["entry_candidates"] = ",".join(entry_candidates[:5])

    return facts


# ── Convention detection ──────────────────────────────────────────────────────

def _detect_conventions(project_path: str) -> dict:
    """Detect code conventions from config files."""
    facts = {}

    # Linter / formatter
    for name, tool in [
        (".eslintrc.json", "eslint"), (".eslintrc.js", "eslint"), (".eslintrc.cjs", "eslint"),
        ("eslint.config.js", "eslint"), ("eslint.config.mjs", "eslint"),
        (".prettierrc", "prettier"), (".prettierrc.json", "prettier"),
        ("biome.json", "biome"),
        ("ruff.toml", "ruff"), (".flake8", "flake8"),
        (".rustfmt.toml", "rustfmt"),
    ]:
        if os.path.exists(os.path.join(project_path, name)):
            facts.setdefault("linter", []).append(tool)

    if "linter" in facts:
        facts["linter"] = ",".join(sorted(set(facts["linter"])))

    # Docker
    if os.path.exists(os.path.join(project_path, "Dockerfile")) or \
       os.path.exists(os.path.join(project_path, "docker-compose.yml")) or \
       os.path.exists(os.path.join(project_path, "docker-compose.yaml")):
        facts["containerized"] = "docker"

    # CI/CD
    if os.path.exists(os.path.join(project_path, ".github", "workflows")):
        facts["ci"] = "github-actions"
    elif os.path.exists(os.path.join(project_path, ".gitlab-ci.yml")):
        facts["ci"] = "gitlab-ci"
    elif os.path.exists(os.path.join(project_path, ".circleci")):
        facts["ci"] = "circleci"

    # Monorepo
    if os.path.exists(os.path.join(project_path, "lerna.json")) or \
       os.path.exists(os.path.join(project_path, "nx.json")) or \
       os.path.exists(os.path.join(project_path, "turbo.json")):
        facts["monorepo"] = "true"

    # Config files (vite, tailwind, etc.)
    for name, key in [
        ("vite.config.ts", "vite_config"), ("vite.config.js", "vite_config"),
        ("tailwind.config.ts", "tailwind_config"), ("tailwind.config.js", "tailwind_config"),
        ("next.config.js", "next_config"), ("next.config.mjs", "next_config"),
        ("tsconfig.json", "tsconfig"),
    ]:
        if os.path.exists(os.path.join(project_path, name)):
            facts[key] = name

    return facts


# ── Main profiler ─────────────────────────────────────────────────────────────

def profile_project(project_path: str) -> dict:
    """Profile a project directory. Returns dict of facts (all string values).

    Facts are designed to be compact — the full set fits in ~300 tokens,
    suitable for injection into a ~4K context window.
    """
    project_path = os.path.abspath(project_path)
    if not os.path.isdir(project_path):
        return {"error": f"Not a directory: {project_path}"}

    facts = {"path": project_path, "name": os.path.basename(project_path)}

    # Detect language + framework from config files (first match wins)
    detectors = [
        (os.path.join(project_path, "package.json"), _detect_from_package_json),
        (os.path.join(project_path, "pyproject.toml"), _detect_from_pyproject),
        (os.path.join(project_path, "setup.py"), _detect_from_pyproject),
        (os.path.join(project_path, "requirements.txt"), _detect_from_pyproject),
        (os.path.join(project_path, "Cargo.toml"), _detect_from_cargo),
        (os.path.join(project_path, "go.mod"), _detect_from_go_mod),
    ]

    for config_file, detector in detectors:
        if os.path.exists(config_file):
            detected = detector(project_path)
            if detected:
                facts.update(detected)
                break

    # If no config file found, guess from file extensions
    if "language" not in facts:
        struct = _scan_structure(project_path)
        ft = struct.get("file_types", "")
        if ".py(" in ft:
            facts["language"] = "python"
        elif ".ts(" in ft or ".tsx(" in ft:
            facts["language"] = "typescript"
        elif ".js(" in ft or ".jsx(" in ft:
            facts["language"] = "javascript"
        elif ".rs(" in ft:
            facts["language"] = "rust"
        elif ".go(" in ft:
            facts["language"] = "go"
        elif ".java(" in ft:
            facts["language"] = "java"
        elif ".rb(" in ft:
            facts["language"] = "ruby"

    # Structure + conventions
    facts.update(_scan_structure(project_path))
    facts.update(_detect_conventions(project_path))

    return facts


def store_profile(project_path: str, facts: dict | None = None) -> dict:
    """Profile a project and store facts in cold memory. Returns the facts."""
    from arqitect.memory.cold import ColdMemory

    if facts is None:
        facts = profile_project(project_path)

    project_path = os.path.abspath(project_path)
    category = f"project:{project_path}"
    cold = ColdMemory()

    for key, value in facts.items():
        cold.set_fact(category, key, str(value))

    return facts


def get_stored_profile(project_path: str) -> dict:
    """Retrieve a previously stored project profile from cold memory."""
    from arqitect.memory.cold import ColdMemory

    project_path = os.path.abspath(project_path)
    category = f"project:{project_path}"
    return ColdMemory().get_facts(category)


def format_profile_for_prompt(facts: dict) -> str:
    """Format project facts into a compact string for LLM prompt injection.

    Output is ~200-300 tokens, designed for small context windows.
    """
    if not facts or facts.get("error"):
        return ""

    lines = [f"Project: {facts.get('name', '?')} ({facts.get('path', '?')})"]

    # Core stack
    stack_parts = []
    if facts.get("language"):
        stack_parts.append(facts["language"])
    if facts.get("framework"):
        stack_parts.append(facts["framework"])
    if facts.get("bundler"):
        stack_parts.append(facts["bundler"])
    if facts.get("styling"):
        stack_parts.append(facts["styling"])
    if stack_parts:
        lines.append(f"Stack: {' + '.join(stack_parts)}")

    if facts.get("pkg_manager"):
        lines.append(f"Pkg manager: {facts['pkg_manager']}")
    if facts.get("test_framework"):
        lines.append(f"Tests: {facts['test_framework']}")
    if facts.get("linter"):
        lines.append(f"Linter: {facts['linter']}")
    if facts.get("ci"):
        lines.append(f"CI: {facts['ci']}")
    if facts.get("top_dirs"):
        lines.append(f"Dirs: {facts['top_dirs']}")
    if facts.get("entry_candidates"):
        lines.append(f"Entry: {facts['entry_candidates']}")
    if facts.get("file_types"):
        lines.append(f"Files: {facts['file_types']}")
    if facts.get("dependencies"):
        lines.append(f"Deps: {facts['dependencies']}")
    if facts.get("scripts"):
        lines.append(f"Scripts: {facts['scripts']}")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python knowledge/project_profiler.py <project_path>", file=sys.stderr)
        sys.exit(1)

    path = os.path.abspath(sys.argv[1])
    facts = profile_project(path)

    print(json.dumps(facts, indent=2))
    print()
    print("--- Prompt format ---")
    print(format_profile_for_prompt(facts))

    if "--store" in sys.argv:
        store_profile(path, facts)
        print(f"\nStored in cold memory as project:{path}")
