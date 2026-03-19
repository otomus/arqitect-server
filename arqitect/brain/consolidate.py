"""Dreamstate — maintenance that happens while the brain sleeps.

Seven cooperative processes, ALL interruptible by incoming tasks:

1. **Consolidation** — merges similar nerves to reduce clutter.
2. **MCP Fanout** — discovers new external MCP tools and wires them into matching
   nerves. Runs BEFORE reconciliation so nerves have the right tools before testing.
3. **Reconciliation** — pushes weak nerves toward 95%+ quality (prompt tuning).
4. **Brain upgrade** — self-provisioning capabilities (planner, recipes).
5. **Fine-tuning** — trains LoRA adapters for nerves with enough training data.
   This is the "deep loop" that breaks through the prompt-tuning ceiling.
   Adapters are saved to nerves/{name}/adapter/ and auto-loaded at inference.
6. **Contribution** — pushes nerves and adapters back to the community repo.
   New nerves are contributed as bundles. Existing nerves get model-specific
   adapters or new stack implementations contributed.
7. **Personality reflection** — evolves communication voice based on interaction
   patterns. Only affects the final text the user sees (communication sense) and
   self-reflection (awareness sense). Never touches routing, nerve selection, or
   work quality.

The moment a task arrives (wake()), ALL dreamstate work stops immediately.
The brain is either awake (processing tasks) or dreaming (maintaining itself).
Never both.

Inspired by React Fiber's reconciliation:
- Work is broken into small units (one merge, one iteration, one provision)
- Between each unit, check if a task arrived (interrupted)
- If interrupted: save progress, yield immediately, release LLM locks
- When idle again: rebuild work queue from current state
"""

import json
import logging
import os
import shutil
import time
import threading

from arqitect.brain.config import CORE_SENSES, NERVES_DIR, mem
from arqitect.brain.events import publish_event, publish_nerve_status
from arqitect.brain.types import Channel

logger = logging.getLogger(__name__)

# How long the brain must be idle before maintenance triggers (seconds)
IDLE_THRESHOLD = 120

# Thresholds and limits
TOOL_FAILURE_RATE_THRESHOLD = 0.3
PLATEAU_DELTA_THRESHOLD = 0.05
PLATEAU_WINDOW = 2
HIGH_QUALITY_SCORE = 0.95
MIN_EPISODES_FOR_REFLECTION = 5
MAX_REFLECTION_EPISODES = 30
MAX_EPISODE_TASK_LENGTH = 100
MAX_TOOL_FIX_ATTEMPTS = 2
MAX_RETEST_CASES = 4
MAX_ROLLBACK_CASES = 3
TRAIT_WEIGHT_MIN = 0.1
TRAIT_WEIGHT_MAX = 0.9
TRAIT_WEIGHT_MAX_DELTA = 0.1
ADAPTER_SCORE_INCREMENT = 0.1
ADAPTER_SCORE_CAP = 0.95
MIN_TOOL_USES_FOR_CONTRIBUTION = 2
MIN_TOOL_CODE_LENGTH = 20
MIN_TEST_COVERAGE = 0.8  # Must run 80% of test bank before score is trustworthy

# Extension-to-language mapping (shared across contribution methods)
_EXT_TO_LANG = {".py": "python", ".js": "javascript", ".ts": "typescript"}

# Sibling discovery constants
_STDLIB_MODULES = frozenset({
    "os", "sys", "json", "re", "math", "datetime", "time", "collections",
    "itertools", "functools", "pathlib", "typing", "hashlib", "base64",
    "urllib", "io", "string", "textwrap", "logging", "copy", "enum",
    "dataclasses", "abc", "contextlib", "threading", "subprocess",
})
_SIBLING_COLD_CATEGORY = "fanout_siblings"
MAX_SIBLINGS_PER_TOOL = 5


def _extract_tool_source_info(tool_name: str) -> dict:
    """Parse a local MCP tool's .py file to extract API URLs and library imports.

    Returns:
        Dict with keys "apis" (list of base URLs) and "libraries" (list of
        non-stdlib import names found in the tool source).
    """
    from arqitect.brain.config import MCP_TOOLS_DIR

    result = {"apis": [], "libraries": []}
    tool_path = os.path.join(MCP_TOOLS_DIR, f"{tool_name}.py")
    if not os.path.isfile(tool_path):
        return result

    try:
        with open(tool_path, "r") as f:
            source = f.read()
    except Exception:
        return result

    result["apis"] = _extract_api_urls(source)
    result["libraries"] = _extract_third_party_imports(source)
    return result


def _extract_api_urls(source: str) -> list[str]:
    """Extract unique API base URLs from requests calls in source code.

    Looks for patterns like requests.get("https://api.example.com/v1/weather")
    and normalizes to the base path (scheme + host).
    """
    import re
    from urllib.parse import urlparse

    url_pattern = re.compile(
        r'requests\.(?:get|post|put|delete|patch)\s*\(\s*[f"\']+(https?://[^"\'}\s]+)',
    )
    base_urls = set()
    for match in url_pattern.finditer(source):
        raw_url = match.group(1).rstrip("/")
        parsed = urlparse(raw_url)
        # Keep scheme + netloc as the base — that identifies the API
        base = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.netloc and "localhost" not in parsed.netloc:
            base_urls.add(base)
    return list(base_urls)


def _extract_third_party_imports(source: str) -> list[str]:
    """Extract non-stdlib, non-requests import names from Python source."""
    import re

    imports = set()
    for match in re.finditer(r'^\s*(?:import|from)\s+(\w+)', source, re.MULTILINE):
        module = match.group(1)
        if module not in _STDLIB_MODULES and module != "requests":
            imports.add(module)
    return list(imports)


def _discover_siblings_for_api(tool_name: str, base_url: str,
                                nerve_desc: str) -> list[dict]:
    """Ask LLM what other endpoints an API offers, using docs if available.

    Returns a list of dicts with "name", "description", and "params" for
    each discovered sibling tool.
    """
    from arqitect.brain.helpers import llm_generate, extract_json

    # Try to fetch API docs (OpenAPI spec or docs page)
    api_docs_snippet = _fetch_api_docs(base_url)

    docs_section = ""
    if api_docs_snippet:
        docs_section = (
            f"\nAPI documentation excerpt:\n{api_docs_snippet[:2000]}\n"
        )

    prompt = (
        f"A tool called '{tool_name}' uses this API: {base_url}\n"
        f"It serves a nerve that handles: {nerve_desc}\n"
        f"{docs_section}\n"
        f"What OTHER endpoints or operations does this API offer that would be "
        f"useful for this domain?\n\n"
        f"Return a JSON object with:\n"
        f'  "siblings": [\n'
        f'    {{"name": "snake_case_tool_name", "description": "what it does", '
        f'"params": "param1: str, param2: str"}}\n'
        f"  ]\n\n"
        f"Rules:\n"
        f"- Only include endpoints that actually exist on this API\n"
        f"- Max {MAX_SIBLINGS_PER_TOOL} siblings\n"
        f"- Tool names must be snake_case\n"
        f"- Each tool should do ONE specific thing\n"
        f"Return ONLY the JSON object."
    )

    raw = llm_generate("brain", prompt)
    parsed = extract_json(raw)
    if not parsed:
        return []
    return parsed.get("siblings", [])[:MAX_SIBLINGS_PER_TOOL]


def _discover_siblings_for_library(tool_name: str, library_name: str,
                                    nerve_desc: str) -> list[dict]:
    """Discover useful functions from a library that could become sibling tools.

    Tries runtime inspection first (dir + docstrings), falls back to LLM.

    Returns a list of dicts with "name", "description", and "params".
    """
    from arqitect.brain.helpers import llm_generate, extract_json

    # Try runtime inspection for concrete info
    lib_exports = _inspect_library_exports(library_name)

    exports_section = ""
    if lib_exports:
        exports_section = (
            f"\nLibrary '{library_name}' exports these public functions/classes:\n"
            f"  {', '.join(lib_exports[:50])}\n"
        )

    prompt = (
        f"A tool called '{tool_name}' uses the Python library '{library_name}'.\n"
        f"It serves a nerve that handles: {nerve_desc}\n"
        f"{exports_section}\n"
        f"What OTHER functions from '{library_name}' would be useful as separate "
        f"MCP tools for this domain?\n\n"
        f"Return a JSON object with:\n"
        f'  "siblings": [\n'
        f'    {{"name": "snake_case_tool_name", "description": "what it does", '
        f'"params": "param1: str, param2: str", "library_function": "the_function_to_use"}}\n'
        f"  ]\n\n"
        f"Rules:\n"
        f"- Only include functions that actually exist in the library\n"
        f"- Max {MAX_SIBLINGS_PER_TOOL} siblings\n"
        f"- Each tool should wrap ONE specific library function\n"
        f"- The tool should be independently useful, not just a thin wrapper\n"
        f"Return ONLY the JSON object."
    )

    raw = llm_generate("brain", prompt)
    parsed = extract_json(raw)
    if not parsed:
        return []
    return parsed.get("siblings", [])[:MAX_SIBLINGS_PER_TOOL]


def _fetch_api_docs(base_url: str) -> str | None:
    """Try to fetch API documentation from common doc endpoints.

    Checks for OpenAPI/Swagger specs first, then falls back to a docs page.
    Returns a text snippet or None.
    """
    import requests as req

    doc_paths = [
        "/openapi.json", "/swagger.json", "/api-docs",
        "/v1/openapi.json", "/v2/openapi.json",
        "/docs", "/api",
    ]

    for path in doc_paths:
        try:
            resp = req.get(f"{base_url}{path}", timeout=5)
            if resp.status_code != 200:
                continue

            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                doc = resp.json()
                # OpenAPI spec — extract endpoint summaries
                paths = doc.get("paths", {})
                if paths:
                    return _summarize_openapi_paths(paths)

            # HTML or text doc page — return raw text
            text = resp.text.strip()
            if len(text) > 50:
                return text[:2000]
        except Exception:
            continue
    return None


def _summarize_openapi_paths(paths: dict) -> str:
    """Summarize OpenAPI paths into a compact list of endpoints."""
    lines = []
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() in ("get", "post", "put", "delete", "patch"):
                summary = ""
                if isinstance(details, dict):
                    summary = details.get("summary", details.get("description", ""))
                lines.append(f"  {method.upper()} {path} — {summary}")
        if len(lines) >= 30:
            break
    return "\n".join(lines)


def _inspect_library_exports(library_name: str) -> list[str]:
    """Inspect a library at runtime to list its public callables.

    Returns a list of function/class names or empty list on import failure.
    """
    try:
        import importlib
        mod = importlib.import_module(library_name)
        exports = []
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            attr = getattr(mod, attr_name, None)
            if callable(attr):
                exports.append(attr_name)
        return exports
    except Exception:
        return []


def _discover_external_siblings(tool_name: str, all_tools: dict) -> list[str]:
    """Find sibling tools from the same external MCP server.

    Returns tool names from the same server that aren't already known.
    """
    tool_info = all_tools.get(tool_name, {})
    if not isinstance(tool_info, dict) or tool_info.get("source") != "external":
        return []

    server_name = tool_info.get("server")
    if not server_name:
        return []

    # Find all tools from the same server
    return [
        name for name, info in all_tools.items()
        if isinstance(info, dict)
        and info.get("server") == server_name
        and name != tool_name
    ]


def _discover_tool_siblings(nerve_name: str, nerve_desc: str,
                             existing_tools: set[str],
                             all_tools: dict,
                             interrupted: threading.Event) -> list[dict]:
    """Discover sibling tools from the same sources as a nerve's existing tools.

    For each tool wired to the nerve:
      - Local tools using an API → discover other API endpoints
      - Local tools using a library → discover other library functions
      - External MCP tools → find other tools from the same server

    Runs once per tool (tracked in cold memory). Returns fabrication candidates
    as a list of dicts with "name", "description", "params".

    Args:
        nerve_name: The nerve being expanded.
        nerve_desc: Current nerve description.
        existing_tools: Tools already wired to this nerve.
        all_tools: Full tool catalog from MCP server.
        interrupted: Dream state interruption flag.

    Returns:
        List of sibling tool specs ready for fabrication.
    """
    candidates = []

    for tool_name in list(existing_tools):
        if interrupted.is_set():
            break

        # Check if siblings already discovered for this tool
        already_done = mem.cold.get_fact(_SIBLING_COLD_CATEGORY, tool_name)
        if already_done:
            continue

        tool_info = all_tools.get(tool_name, {})
        is_external = isinstance(tool_info, dict) and tool_info.get("source") == "external"

        if is_external:
            # External MCP: just find unwired siblings from same server
            sibling_names = _discover_external_siblings(tool_name, all_tools)
            for sib_name in sibling_names:
                if sib_name not in existing_tools:
                    candidates.append({
                        "name": sib_name,
                        "source": "external_sibling",
                    })
        else:
            # Local tool: introspect source code
            source_info = _extract_tool_source_info(tool_name)

            for base_url in source_info["apis"]:
                if interrupted.is_set():
                    break
                siblings = _discover_siblings_for_api(
                    tool_name, base_url, nerve_desc,
                )
                candidates.extend(siblings)

            for lib_name in source_info["libraries"]:
                if interrupted.is_set():
                    break
                siblings = _discover_siblings_for_library(
                    tool_name, lib_name, nerve_desc,
                )
                candidates.extend(siblings)

        # Mark this tool as explored regardless of results
        mem.cold.set_fact(_SIBLING_COLD_CATEGORY, tool_name, "done")
        logger.info("[SIBLING-DISCOVERY] Explored '%s' -> %d candidates",
                    tool_name, len(candidates))

    return candidates


def _fabricate_sibling_tools(nerve_name: str, candidates: list[dict],
                              existing_tools: set[str], all_tools: dict,
                              interrupted: threading.Event) -> int:
    """Fabricate discovered sibling tools and wire them to the nerve.

    External siblings are just wired (they already exist). Local siblings
    are fabricated via the standard tool fabrication pipeline.

    Args:
        nerve_name: Nerve to wire new tools to.
        candidates: Sibling specs from _discover_tool_siblings.
        existing_tools: Mutable set of tools already on the nerve.
        all_tools: Current tool catalog.
        interrupted: Dream state interruption flag.

    Returns:
        Number of tools successfully fabricated or wired.
    """
    if not candidates:
        return 0

    from arqitect.brain.synthesis import fabricate_mcp_tool
    from arqitect.matching import best_match_tool

    fabricated = 0
    for candidate in candidates:
        if interrupted.is_set():
            break

        name = candidate.get("name", "")
        if not isinstance(name, str) or not name:
            continue

        # External sibling — just wire it
        if candidate.get("source") == "external_sibling":
            if name not in existing_tools and name in all_tools:
                mem.cold.add_nerve_tool(nerve_name, name)
                existing_tools.add(name)
                fabricated += 1
                logger.info("[SIBLING-FABRICATE] Wired external '%s' -> '%s'",
                           name, nerve_name)
            continue

        # Skip if tool already exists or has a close match
        if name in all_tools or name in existing_tools:
            continue
        if best_match_tool(name, all_tools, threshold=2.0):
            continue

        description = candidate.get("description", f"Sibling tool: {name}")
        params = candidate.get("params", "query: str")

        try:
            fabricate_mcp_tool(name, description, params)
            mem.cold.add_nerve_tool(nerve_name, name)
            existing_tools.add(name)
            fabricated += 1
            logger.info("[SIBLING-FABRICATE] Created '%s' for nerve '%s'",
                       name, nerve_name)
        except Exception as e:
            logger.warning("[SIBLING-FABRICATE] Failed '%s': %s", name, e)

    if fabricated:
        logger.info("[SIBLING-FABRICATE] %d tool(s) added to nerve '%s'",
                   fabricated, nerve_name)
    return fabricated


def _contribute_pr(community_dir: str, add_path: str, commit_msg: str,
                   pr_title: str, pr_body: str, branch_prefix: str,
                   search_key: str) -> str | None:
    """Create or update a community PR.

    Deduplicates by searching for an existing open PR from @me matching search_key.
    If found, checks out its branch, updates, rebases, and force-pushes.
    If not, creates a new branch and PR.

    Returns the PR URL on success, None on failure.
    """
    import subprocess

    def _run(cmd, **kw):
        return subprocess.run(cmd, cwd=community_dir, capture_output=True,
                              text=True, timeout=kw.pop("timeout", 30), **kw)

    try:
        # Check for existing open PR from us
        existing_branch = None
        check = _run(["gh", "pr", "list", "--author", "@me", "--state", "open",
                       "--search", search_key, "--json", "headRefName", "--limit", "1"])
        if check.returncode == 0 and check.stdout.strip() not in ("", "[]"):
            prs = json.loads(check.stdout)
            if prs:
                existing_branch = prs[0]["headRefName"]

        if existing_branch:
            # Update existing PR — checkout its branch, update files, amend
            _run(["git", "checkout", existing_branch])
            _run(["git", "add", add_path])
            # Check if there are staged changes
            diff = _run(["git", "diff", "--cached", "--quiet"])
            if diff.returncode == 0:
                # No changes — nothing to update
                _run(["git", "checkout", "main"])
                return None
            _run(["git", "commit", "-m", commit_msg])
            _run(["git", "pull", "--rebase", "origin", "main"])
            _run(["git", "push", "--force-with-lease"], timeout=60)
            logger.info("[CONTRIBUTE] Updated existing PR on branch '%s'", existing_branch)
            _run(["git", "checkout", "main"])
            return existing_branch
        else:
            # New PR
            branch_name = f"{branch_prefix}-{int(time.time())}"
            _run(["git", "checkout", "main"])
            _run(["git", "pull", "origin", "main"])
            _run(["git", "checkout", "-b", branch_name])
            _run(["git", "add", add_path])
            _run(["git", "commit", "-m", commit_msg])
            _run(["git", "pull", "--rebase", "origin", "main"])
            _run(["git", "push", "-u", "origin", branch_name], timeout=60)
            pr_result = _run(["gh", "pr", "create", "--fill", "--auto",
                              "--title", pr_title, "--body", pr_body], timeout=60)
            _run(["git", "checkout", "main"])
            if pr_result.returncode == 0:
                url = pr_result.stdout.strip()
                logger.info("[CONTRIBUTE] PR created: %s", url)
                return url
            else:
                logger.warning("[CONTRIBUTE] PR creation failed: %s", pr_result.stderr[:200])
                return None

    except subprocess.TimeoutExpired:
        logger.warning("[CONTRIBUTE] Git timeout for '%s'", search_key)
    except Exception as e:
        logger.warning("[CONTRIBUTE] Git error for '%s': %s", search_key, e)
    finally:
        try:
            _run(["git", "checkout", "main"])
        except Exception as exc:
            logger.debug("[CONTRIBUTE] Could not checkout main in finally: %s", exc)
    return None


def _update_nerve_file_description(nerve_name: str, new_desc: str):
    """Update the DESCRIPTION constant baked into a nerve's nerve.py file.

    Nerves have their description in two places: SQLite (cold memory) and
    the nerve.py source file. This patches the source file so the nerve
    uses the expanded description at runtime without re-synthesis.
    """
    import re
    nerve_path = os.path.join(NERVES_DIR, nerve_name, "nerve.py")
    if not os.path.isfile(nerve_path):
        return
    try:
        code = open(nerve_path).read()
        # Replace the DESCRIPTION = """...""" line
        code = re.sub(
            r'DESCRIPTION\s*=\s*""".*?"""',
            f'DESCRIPTION = """{new_desc}"""',
            code,
            count=1,
            flags=re.DOTALL,
        )
        # Also update the module docstring if it contains the old description
        code = re.sub(
            r'^"""Nerve: ' + re.escape(nerve_name) + r' — .*?"""',
            f'"""Nerve: {nerve_name} — {new_desc}"""',
            code,
            count=1,
        )
        with open(nerve_path, "w") as f:
            f.write(code)
    except Exception as e:
        logger.warning("[MCP-FANOUT] Failed to update nerve file for '%s': %s", nerve_name, e)

def _get_merge_threshold() -> float:
    """Get merge threshold from community config."""
    from arqitect.brain.adapters import get_tuning_config
    return get_tuning_config("nerve")["merge_threshold"]

def _get_improvement_threshold() -> float:
    """Get improvement threshold from community config."""
    from arqitect.brain.adapters import get_tuning_config
    return get_tuning_config("nerve")["improvement_threshold"]


# ── Consolidation (interruptible between each merge) ────────────────────────


def find_nerve_clusters(catalog: dict[str, str],
                        community_nerves: frozenset[str] | None = None,
                        ) -> list[list[tuple[str, str]]]:
    """Group nerves into clusters of similar functionality.

    Args:
        catalog: {nerve_name: description} mapping of all nerves.
        community_nerves: Pre-loaded set of community nerve names.
            Loaded from manifest if not provided.

    Returns:
        List of clusters, where each cluster is [(name, description), ...].
        Only returns clusters with 2+ members (i.e. potential merges).
    """
    from arqitect.matching import match_score

    if community_nerves is None:
        community_nerves = _get_community_nerve_names()

    names = [n for n in catalog if n not in CORE_SENSES and not mem.cold.is_sense(n)
             and (n in community_nerves or not mem.cold.get_fact(f"nerve:{n}", "stack_hash"))]
    if len(names) < 2:
        return []

    # Build adjacency: which nerves are similar enough to merge?
    adj = {n: set() for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            score = max(
                match_score(catalog[a], b, catalog[b]),
                match_score(catalog[b], a, catalog[a]),
            )
            if score >= _get_merge_threshold():
                adj[a].add(b)
                adj[b].add(a)

    # Connected components via BFS
    visited = set()
    clusters = []
    for n in names:
        if n in visited or not adj[n]:
            continue
        cluster = []
        queue = [n]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append((curr, catalog[curr]))
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def _get_community_nerve_names() -> frozenset[str]:
    """Load the set of nerve names declared in the community manifest.

    Returns:
        Frozenset of community nerve names (empty if manifest unavailable).
    """
    from arqitect.brain.community import _load_cached_manifest
    manifest = _load_cached_manifest()
    if not manifest:
        return frozenset()
    return frozenset(manifest.get("nerves", {}).keys())


def pick_winner(cluster: list[tuple[str, str]],
                community_nerves: frozenset[str] | None = None) -> tuple[str, list[str]]:
    """Pick the best nerve in a cluster. Returns (winner_name, loser_names).

    Winner criteria (in order):
    1. Community nerve (always wins over fabricated)
    2. Highest qualification score
    3. Most tools learned (more tools = more capability)

    Args:
        cluster: List of (name, description) pairs in the cluster.
        community_nerves: Pre-loaded set of community nerve names.
            Loaded from manifest if not provided.
    """
    if community_nerves is None:
        community_nerves = _get_community_nerve_names()

    nerve_scores = _build_nerve_score_lookup()

    scored = []
    for name, _desc in cluster:
        is_community = name in community_nerves
        qual_score = nerve_scores.get(name, 0.0)
        tools = len(mem.cold.get_nerve_tools(name))
        scored.append((name, is_community, qual_score, tools))

    scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    winner = scored[0][0]
    losers = [s[0] for s in scored[1:]]
    return winner, losers


def _build_nerve_score_lookup() -> dict[str, float]:
    """Build a {nerve_name: score} mapping from all qualification records."""
    scores: dict[str, float] = {}
    try:
        for q in mem.cold.list_qualifications():
            if q["subject_type"] == "nerve":
                scores[q["subject_name"]] = q.get("score", 0.0)
    except Exception as exc:
        logger.debug("[CONSOLIDATE] Could not load qualifications: %s", exc)
    return scores


def merge_nerve(winner: str, loser: str, loser_desc: str):
    """Migrate tools, adapters, and metadata from loser to winner, then delete loser."""
    # Migrate tools
    loser_tools = mem.cold.get_nerve_tools(loser)
    winner_tools = mem.cold.get_nerve_tools(winner)
    for tool in loser_tools:
        if tool not in winner_tools:
            mem.cold.add_nerve_tool(winner, tool)
            logger.info("[CONSOLIDATE] Migrated tool '%s' from '%s' to '%s'", tool, loser, winner)

    # Migrate cached context/meta files and LoRA adapters from loser
    _merge_nerve_cache_files(winner, loser)

    # Enrich winner description with loser's keywords
    _enrich_description(winner, loser_desc)

    # TODO: episode/history migration — decide on a migration policy for
    # accumulated episodes and conversation history from the loser nerve.
    # For now we let them go with the loser deletion.

    # Delete loser completely (filesystem + cold memory + Redis)
    from arqitect.brain.synthesis import delete_nerve
    delete_nerve(loser)
    logger.info("[CONSOLIDATE] Merged '%s' into '%s'", loser, winner)


def _merge_nerve_cache_files(winner: str, loser: str):
    """Merge community cache files (context.json, meta.json) from loser into winner.

    Copies per-size-class directories from the loser's cache that the winner
    doesn't already have. Also migrates LoRA adapter files on disk.
    """
    from arqitect.brain.community import _cache_dir
    loser_cache = os.path.join(_cache_dir(), "nerves", loser)
    winner_cache = os.path.join(_cache_dir(), "nerves", winner)

    if os.path.isdir(loser_cache):
        from arqitect.brain.adapters import SIZE_CLASSES
        for size_class in SIZE_CLASSES:
            src_dir = os.path.join(loser_cache, size_class)
            dst_dir = os.path.join(winner_cache, size_class)
            if not os.path.isdir(src_dir):
                continue
            if os.path.isdir(dst_dir):
                # Merge model-slug subdirs without overwriting
                for entry in os.listdir(src_dir):
                    src_sub = os.path.join(src_dir, entry)
                    dst_sub = os.path.join(dst_dir, entry)
                    if os.path.isdir(src_sub) and not os.path.exists(dst_sub):
                        shutil.copytree(src_sub, dst_sub)
                        logger.info("[CONSOLIDATE] Migrated cache '%s/%s' from '%s' to '%s'",
                                    size_class, entry, loser, winner)
            else:
                shutil.copytree(src_dir, dst_dir)
                logger.info("[CONSOLIDATE] Migrated cache '%s' from '%s' to '%s'",
                            size_class, loser, winner)

    _merge_adapter_files(winner, loser)


def _merge_adapter_files(winner: str, loser: str):
    """Copy LoRA adapter files from loser to winner for models the winner lacks."""
    loser_adapter_dir = os.path.join(NERVES_DIR, loser, "adapter")
    winner_adapter_dir = os.path.join(NERVES_DIR, winner, "adapter")
    if not os.path.isdir(loser_adapter_dir):
        return

    for entry in os.listdir(loser_adapter_dir):
        src = os.path.join(loser_adapter_dir, entry)
        dst = os.path.join(winner_adapter_dir, entry)
        if os.path.exists(dst):
            continue  # Never overwrite existing adapters
        os.makedirs(winner_adapter_dir, exist_ok=True)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            logger.info("[CONSOLIDATE] Migrated adapter file '%s' from '%s' to '%s'",
                        entry, loser, winner)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)
            logger.info("[CONSOLIDATE] Migrated adapter dir '%s' from '%s' to '%s'",
                        entry, loser, winner)


def _enrich_description(winner: str, loser_desc: str):
    """Add unique keywords from loser_desc to the winner's description.

    Capped to prevent keyword soup — if the description already has an
    '(also: ...)' section, we don't add more. Bloated descriptions cause
    false matches in consolidation and pruning, leading to cascading merges.
    """
    import re
    winner_info = mem.cold.get_nerve_info(winner)
    if not winner_info:
        return
    winner_desc = winner_info.get("description", "")

    # Don't enrich if already enriched — prevents keyword accumulation
    if "(also:" in winner_desc:
        return

    def words(text):
        return set(re.split(r"[^a-z0-9]+", text.lower())) - {"", "a", "the", "and", "or", "for", "of", "in", "on", "to"}

    winner_words = words(winner_desc)
    loser_words = words(loser_desc)
    new_words = loser_words - winner_words

    if new_words and len(new_words) <= 5:
        enriched = f"{winner_desc} (also: {', '.join(sorted(new_words))})"
        try:
            mem.cold.update_nerve_description(winner, enriched)
            logger.info("[CONSOLIDATE] Enriched '%s' with: %s", winner, sorted(new_words))
        except Exception as exc:
            logger.warning("[CONSOLIDATE] Failed to enrich '%s': %s", winner, exc)


def _is_qualified_nerve(name: str) -> bool:
    """Check if a nerve has a qualification score at or above threshold."""
    from arqitect.brain.adapters import get_tuning_config
    threshold = get_tuning_config("nerve")["qualification_threshold"]
    qual = mem.cold.get_qualification("nerve", name)
    score = qual.get("score", 0) if qual else 0
    return score >= threshold


def _llm_judge_same_purpose(nerve_a: str, desc_a: str, nerve_b: str, desc_b: str) -> bool:
    """Ask the brain LLM whether two nerves serve the same purpose.

    Returns True only if the LLM confirms they are duplicates that should
    be merged. Defaults to False on any error (safe side: don't merge).

    Args:
        nerve_a: Name of the first nerve.
        desc_a: Description of the first nerve.
        nerve_b: Name of the second nerve.
        desc_b: Description of the second nerve.

    Returns:
        True if the LLM judges them as duplicates.
    """
    prompt = (
        f"Nerve A: {nerve_a} — {desc_a}\n"
        f"Nerve B: {nerve_b} — {desc_b}\n\n"
        "Are these two nerves doing the SAME job? Would merging them into one "
        "lose any unique capability?\n\n"
        "Answer with ONLY one word: YES (same job, safe to merge) or NO (different, keep both)."
    )
    try:
        from arqitect.inference.router import generate_for_role
        from arqitect.brain.adapters import get_max_tokens
        result = generate_for_role(
            "brain", prompt,
            system="You are judging whether two task handlers are duplicates. Be conservative — if in doubt, say NO.",
            max_tokens=min(16, get_max_tokens("brain")),
        ).strip().upper()
        return result.startswith("YES")
    except Exception as e:
        logger.warning("[CONSOLIDATE] LLM judge failed: %s — skipping merge", e)
        return False


def consolidate_nerves(interrupted: threading.Event = None) -> dict:
    """Run nerve consolidation. Interruptible between each merge.

    Requires a medium or large brain model — smaller models lack the judgment
    to decide which nerves are true duplicates. Uses LLM-as-judge to confirm
    each merge, preventing false positives from keyword similarity.
    """
    from arqitect.brain.permissions import can_model_fabricate

    if not can_model_fabricate():
        logger.info("[CONSOLIDATE] Model too small for consolidation — skipping")
        return {"clusters": 0, "merged": 0}

    # Use full registry — consolidation must see ALL nerves including 0% ones
    # to merge duplicates (e.g. weather_nerve + forecast_nerve).
    community_nerves = _get_community_nerve_names()
    catalog = mem.cold.list_nerves()
    clusters = find_nerve_clusters(catalog, community_nerves)

    if not clusters:
        logger.info("[CONSOLIDATE] No similar nerves found — nothing to consolidate.")
        return {"clusters": 0, "merged": 0}

    total_merged = 0
    for cluster in clusters:
        if interrupted and interrupted.is_set():
            logger.info("[CONSOLIDATE] Interrupted — yielding")
            break

        names_in_cluster = [n for n, _ in cluster]
        logger.info("[CONSOLIDATE] Found cluster: %s", names_in_cluster)

        winner, losers = pick_winner(cluster, community_nerves)
        logger.info("[CONSOLIDATE] Winner: '%s', candidates: %s", winner, losers)

        winner_is_community = winner in community_nerves
        winner_desc = catalog.get(winner, "")
        cluster_dict = dict(cluster)
        for loser in losers:
            if interrupted and interrupted.is_set():
                logger.info("[CONSOLIDATE] Interrupted mid-merge — yielding")
                break

            # Community winner always absorbs — community is source of truth.
            # Fabricated winner skips qualified losers — they earned their place.
            if not winner_is_community and _is_qualified_nerve(loser):
                logger.info("[CONSOLIDATE] Skipping merge of qualified '%s' into fabricated '%s'", loser, winner)
                continue

            # LLM-as-judge: confirm the loser truly duplicates the winner
            loser_desc = cluster_dict.get(loser, catalog.get(loser, ""))
            if not _llm_judge_same_purpose(winner, winner_desc, loser, loser_desc):
                logger.info(
                    "[CONSOLIDATE] LLM says '%s' and '%s' serve different purposes — keeping both",
                    winner, loser,
                )
                continue

            merge_nerve(winner, loser, loser_desc)
            total_merged += 1

    publish_nerve_status()
    summary = {"clusters": len(clusters), "merged": total_merged}
    publish_event(Channel.BRAIN_THOUGHT, {
        "stage": "consolidation",
        "message": f"Consolidated {total_merged} duplicate nerve(s) across {len(clusters)} cluster(s).",
    })
    return summary


# ── Reconciler (slow, interruptible — React Fiber style) ────────────────────


def _build_work_queue() -> list[dict]:
    """Scan all nerves and build a prioritized work queue.

    Returns list of {name, description, score, last_invoked_at} sorted by:
    1. Recently used nerves first (have last_invoked_at)
    2. Within each group, weakest score first

    This ensures nerves the user actually invokes get tuned before
    dormant ones. Only includes nerves that:
    - Are not core senses
    - Have a nerve.py file on disk
    - Score below _get_improvement_threshold()
    """
    quals = mem.cold.list_qualifications()

    # Build score lookup
    nerve_scores = {}
    for q in quals:
        if q["subject_type"] == "nerve":
            nerve_scores[q["subject_name"]] = q.get("score", 0.0)

    # Scan all nerves from registry + filesystem
    all_nerves = {}
    all_nerves.update(mem.cold.list_nerves())
    from arqitect.config.loader import get_nerves_dir; nerves_dir = get_nerves_dir()
    if os.path.isdir(nerves_dir):
        for d in os.listdir(nerves_dir):
            nerve_py = os.path.join(nerves_dir, d, "nerve.py")
            if os.path.isfile(nerve_py) and d not in all_nerves:
                all_nerves[d] = ""

    queue = []
    for name, desc in all_nerves.items():
        if name in CORE_SENSES or mem.cold.is_sense(name):
            continue
        nerve_py = os.path.join(nerves_dir, name, "nerve.py")
        if not os.path.isfile(nerve_py):
            continue
        score = nerve_scores.get(name, 0.0)
        if score < _get_improvement_threshold():
            last_invoked = mem.cold.get_last_invoked_at(name)
            queue.append({
                "name": name,
                "description": desc or name.replace("_", " "),
                "score": score,
                "last_invoked_at": last_invoked,
            })

    # Recently-used nerves first, then weakest score within each group
    queue.sort(key=lambda x: (x["last_invoked_at"] is None, x["score"]))
    return queue


def _validate_nerve(name: str) -> bool:
    """Check that a nerve still exists and still needs improvement.

    This is the React-style "check reality before doing work" step.
    A nerve may have been deleted by consolidation or already improved
    since we built the queue.
    """
    # Check nerve exists on disk (don't use catalog — it excludes 0% nerves)
    from arqitect.config.loader import get_nerves_dir; nerves_dir = get_nerves_dir()
    nerve_py = os.path.join(nerves_dir, name, "nerve.py")
    if not os.path.isfile(nerve_py):
        return False  # deleted by consolidation
    if name in CORE_SENSES or mem.cold.is_sense(name):
        return False

    # Check current score
    quals = mem.cold.list_qualifications()
    for q in quals:
        if q["subject_name"] == name and q["subject_type"] == "nerve":
            if q.get("score", 0.0) >= _get_improvement_threshold():
                return False  # already improved
            return True

    # No qualification record = score 0.0, definitely needs work
    return True


def _improve_one_nerve(name: str, description: str, old_score: float,
                       interrupted: threading.Event) -> dict:
    """Run one qualification pass on a nerve, checking for interruption.

    Returns {name, old_score, new_score, completed}.
    """
    from arqitect.critic.qualify_nerve import (
        generate_test_cases, run_nerve_with_input,
        evaluate_nerve_output, suggest_improvements, _publish_progress,
    )
    from arqitect.brain.adapters import get_tuning_config

    nerve_meta = mem.cold.get_nerve_metadata(name)
    nerve_role = nerve_meta.get("role", "tool") if nerve_meta else "tool"
    tuning_cfg = get_tuning_config(nerve_role)
    max_iterations = tuning_cfg["max_reconciliation_iterations"]
    qual_threshold = tuning_cfg["qualification_threshold"]

    known_tools = mem.cold.get_nerve_tools(name)
    _publish_progress(name, old_score, None, known_tools, 0, max_iterations)

    state = _ImprovementState(name, description, old_score)
    stored_tests = mem.cold.get_test_bank(name)

    for iteration in range(1, max_iterations + 1):
        if interrupted.is_set():
            logger.info("[RECONCILE] Interrupted before iteration %d for '%s'", iteration, name)
            break

        if _detect_plateau(state.score_history):
            logger.info("[RECONCILE] Plateau detected for '%s' (last scores: %s)", name, state.score_history[-PLATEAU_WINDOW:])
            break

        logger.info("[RECONCILE] Iteration %d/%d for '%s' (current: %.0f%%)", iteration, max_iterations, name, state.best_score * 100)

        if interrupted.is_set():
            break

        stored_tests = _ensure_test_bank(name, description, stored_tests, generate_test_cases)
        if not stored_tests:
            logger.warning("[RECONCILE] Failed to generate test cases for '%s'", name)
            break

        iteration_results = _run_test_cases(
            name, stored_tests, interrupted, run_nerve_with_input, evaluate_nerve_output,
        )
        if not iteration_results:
            break

        avg_score = sum(r["score"] for r in iteration_results) / len(iteration_results)
        passed_count = sum(1 for r in iteration_results if r["passed"])
        state.update(avg_score, iteration_results)

        _publish_progress(name, avg_score, None, known_tools, iteration, max_iterations)

        if avg_score >= qual_threshold and _has_sufficient_coverage(len(iteration_results), len(stored_tests)):
            mem.cold.record_qualification(
                "nerve", name, True, avg_score, iteration,
                len(iteration_results), passed_count, json.dumps(iteration_results),
            )
            if avg_score >= _get_improvement_threshold():
                logger.info("[RECONCILE] '%s' reached target: %.0f%%", name, avg_score * 100)
                _publish_progress(name, avg_score, True, known_tools, iteration, max_iterations)
                publish_nerve_status()
                return {"name": name, "old_score": old_score, "new_score": avg_score, "completed": True}
        elif avg_score >= qual_threshold:
            logger.info("[RECONCILE] '%s' scored %.0f%% but only %d/%d tests ran — not recording",
                        name, avg_score * 100, len(iteration_results), len(stored_tests))

        if interrupted.is_set():
            break

        # Suggest and apply improvements for next iteration
        if iteration < max_iterations and not interrupted.is_set():
            result = _apply_improvements(
                name, state, iteration_results, known_tools, stored_tests,
                interrupted, suggest_improvements, run_nerve_with_input, evaluate_nerve_output,
                _publish_progress, max_iterations, iteration,
            )
            if result is not None:
                return result

    # Save best score even if interrupted or exhausted — but only with sufficient coverage
    _save_final_score(name, state, old_score, qual_threshold, max_iterations, len(stored_tests))

    completed = not interrupted.is_set()
    return {"name": name, "old_score": old_score, "new_score": state.best_score, "completed": completed}


class _ImprovementState:
    """Tracks improvement state across iterations."""

    def __init__(self, name: str, description: str, old_score: float):
        self.name = name
        self.description = description
        self.best_score = old_score
        self.all_test_results: list[dict] = []
        self.score_history: list[float] = []

    def update(self, avg_score: float, results: list[dict]) -> None:
        self.all_test_results = results
        self.best_score = max(self.best_score, avg_score)
        self.score_history.append(avg_score)


def _detect_plateau(score_history: list[float]) -> bool:
    """Check if scores have plateaued over the last few iterations."""
    if len(score_history) < PLATEAU_WINDOW:
        return False
    recent = score_history[-PLATEAU_WINDOW:]
    improvement = max(recent) - min(recent)
    return improvement < PLATEAU_DELTA_THRESHOLD and recent[-1] < _get_improvement_threshold()


def _ensure_test_bank(name: str, description: str, stored_tests: list | None,
                      generate_test_cases) -> list | None:
    """Ensure we have a sufficient test bank, generating if needed."""
    if stored_tests and len(stored_tests) >= MAX_RETEST_CASES:
        return stored_tests
    tests = generate_test_cases(name, description, "")
    if tests:
        mem.cold.set_test_bank(name, tests)
        return tests
    return stored_tests


def _has_sufficient_coverage(results_count: int, total_tests: int) -> bool:
    """Check if enough test cases ran to trust the score.

    Args:
        results_count: Number of test cases that completed.
        total_tests: Total number of test cases in the bank.

    Returns:
        True if coverage meets MIN_TEST_COVERAGE threshold.
    """
    if total_tests <= 0:
        return False
    return results_count / total_tests >= MIN_TEST_COVERAGE


def _run_test_cases(name: str, tests: list[dict], interrupted: threading.Event,
                    run_nerve_with_input, evaluate_nerve_output) -> list[dict]:
    """Run test cases against a nerve, yielding on interruption."""
    results = []
    for tc in tests:
        if interrupted.is_set():
            logger.info("[RECONCILE] Interrupted mid-test for '%s'", name)
            break
        user_input = tc.get("input", "")
        if not user_input:
            continue
        if interrupted.is_set():
            break
        output = run_nerve_with_input(name, user_input, mem)
        if interrupted.is_set():
            break
        evaluation = evaluate_nerve_output(tc, output)
        evaluation["input"] = user_input
        evaluation["category"] = tc.get("category", "")
        evaluation["expected"] = tc.get("output", tc.get("expected_behavior", ""))
        evaluation["raw_stderr"] = output.get("raw_stderr", "")
        evaluation["raw_stdout"] = output.get("raw_stdout", "")
        results.append(evaluation)
    return results


def _apply_improvements(name: str, state: _ImprovementState, iteration_results: list[dict],
                        known_tools: list[str], stored_tests: list[dict],
                        interrupted: threading.Event, suggest_improvements,
                        run_nerve_with_input, evaluate_nerve_output,
                        _publish_progress, max_iterations: int, iteration: int) -> dict | None:
    """Apply tool fixes and prompt improvements. Returns result dict if target reached, else None."""
    from arqitect.critic.qualify_nerve import _extract_tool_errors, _apply_tool_fix, _rollback_tool_fix

    failures = [r for r in iteration_results if not r["passed"]]
    meta = mem.cold.get_nerve_metadata(name)
    prev_prompt = meta.get("system_prompt", "")
    prev_score = state.score_history[-1]

    tool_errors = _extract_tool_errors(failures)
    applied_tool_fixes = []

    if tool_errors and not interrupted.is_set():
        applied_tool_fixes, improvements = _fix_broken_tools(
            name, state.description, prev_prompt, meta, known_tools, failures,
            tool_errors, suggest_improvements, _apply_tool_fix,
        )
        # Validate tool fixes
        if applied_tool_fixes and stored_tests and not interrupted.is_set():
            result = _validate_tool_fixes(
                name, state, applied_tool_fixes, stored_tests, prev_score,
                interrupted, run_nerve_with_input, evaluate_nerve_output,
                _publish_progress, _rollback_tool_fix, known_tools, max_iterations, iteration,
            )
            if result is not None:
                return result
    else:
        improvements = suggest_improvements(
            name, state.description, prev_prompt, meta.get("examples", []),
            known_tools, failures,
        )

    new_sp = improvements.get("system_prompt", "")
    new_ex = improvements.get("examples", [])
    new_desc = improvements.get("description", "")

    _apply_prompt_improvements(name, state, new_sp, new_ex, new_desc, meta)

    # Rollback check for prompt changes
    if (new_sp or new_ex) and not applied_tool_fixes and stored_tests and len(stored_tests) >= MAX_RETEST_CASES:
        _rollback_if_worse(
            name, state.description, prev_prompt, prev_score, meta, stored_tests,
            interrupted, run_nerve_with_input, evaluate_nerve_output,
        )

    return None


def _fix_broken_tools(name: str, description: str, prev_prompt: str, meta: dict,
                      known_tools: list[str], failures: list[dict],
                      tool_errors: list[dict], suggest_improvements, _apply_tool_fix) -> tuple[list[str], dict]:
    """Fix broken tools and return (applied_fixes, improvements)."""
    logger.info("[RECONCILE] Found %d tool errors for '%s': %s", len(tool_errors), name, [e['tool'] for e in tool_errors])

    improvements = suggest_improvements(
        name, description, prev_prompt, meta.get("examples", []),
        known_tools, failures, tool_errors=tool_errors,
    )

    applied = []
    for fix in improvements.get("tool_fixes", [])[:MAX_TOOL_FIX_ATTEMPTS]:
        tool_name = fix.get("tool", "")
        tool_code = fix.get("fixed_code", "")
        if tool_name and tool_code and _apply_tool_fix(tool_name, tool_code, known_tools):
            applied.append(tool_name)

    return applied, improvements


def _validate_tool_fixes(name: str, state: _ImprovementState, applied_fixes: list[str],
                         stored_tests: list[dict], prev_score: float,
                         interrupted: threading.Event, run_nerve_with_input,
                         evaluate_nerve_output, _publish_progress, _rollback_tool_fix,
                         known_tools: list[str], max_iterations: int, iteration: int) -> dict | None:
    """Re-test after tool fixes. Returns result dict if target reached, else None."""
    logger.info("[RECONCILE] Re-testing '%s' after tool fixes...", name)
    retest_score = 0.0
    retest_count = 0

    for tc in stored_tests[:MAX_RETEST_CASES]:
        if interrupted.is_set():
            break
        user_input = tc.get("input", "")
        if not user_input:
            continue
        output = run_nerve_with_input(name, user_input, mem)
        evaluation = evaluate_nerve_output(tc, output)
        retest_score += evaluation["score"]
        retest_count += 1

    if retest_count == 0:
        return None

    retest_avg = retest_score / retest_count
    if retest_avg > prev_score:
        logger.info("[RECONCILE] Tool fixes improved '%s': %.2f -> %.2f", name, prev_score, retest_avg)
        state.best_score = max(state.best_score, retest_avg)
        state.score_history[-1] = retest_avg
        _cleanup_tool_backups(applied_fixes)
        if retest_avg >= _get_improvement_threshold():
            logger.info("[RECONCILE] '%s' reached target after tool fix: %.0f%%", name, retest_avg * 100)
            _publish_progress(name, retest_avg, True, known_tools, iteration, max_iterations)
            publish_nerve_status()
            return {"name": name, "old_score": state.best_score, "new_score": retest_avg, "completed": True}
    else:
        logger.info("[RECONCILE] Tool fixes didn't help '%s': %.2f -> %.2f, rolling back", name, prev_score, retest_avg)
        for tool_name in applied_fixes:
            _rollback_tool_fix(tool_name)

    return None


def _cleanup_tool_backups(applied_fixes: list[str]) -> None:
    """Remove .bak files for successfully applied tool fixes."""
    from arqitect.config.loader import get_mcp_tools_dir
    mcp_dir = get_mcp_tools_dir()
    for tool_name in applied_fixes:
        bak = os.path.join(mcp_dir, f"{tool_name}.py.bak")
        if os.path.exists(bak):
            os.remove(bak)
            logger.info("[RECONCILE] Cleaned up backup: %s.py.bak", tool_name)


def _apply_prompt_improvements(name: str, state: _ImprovementState,
                                new_sp: str, new_ex: list, new_desc: str, meta: dict) -> None:
    """Apply system prompt, examples, and description improvements."""
    effective_desc = new_desc if new_desc and new_desc.strip() else state.description
    if not (new_sp or new_ex):
        return

    existing_role = meta.get("role", "tool")
    mem.cold.register_nerve_rich(
        name, effective_desc, new_sp, json.dumps(new_ex) if new_ex else "[]", role=existing_role,
    )
    logger.info("[RECONCILE] Updated metadata for '%s'", name)

    if new_desc and new_desc.strip() and new_desc != state.description:
        state.description = effective_desc
        logger.info("[RECONCILE] Refined description for '%s': %s", name, effective_desc[:80])


def _rollback_if_worse(name: str, description: str, prev_prompt: str, prev_score: float,
                       meta: dict, stored_tests: list[dict],
                       interrupted: threading.Event, run_nerve_with_input, evaluate_nerve_output) -> None:
    """Roll back prompt changes if they made the score worse."""
    try:
        reeval_score = 0.0
        reeval_count = 0
        for tc in stored_tests[:MAX_ROLLBACK_CASES]:
            if interrupted.is_set():
                break
            user_input = tc.get("input", "")
            if not user_input:
                continue
            output = run_nerve_with_input(name, user_input, mem)
            evaluation = evaluate_nerve_output(tc, output)
            reeval_score += evaluation["score"]
            reeval_count += 1
        if reeval_count > 0 and reeval_score / reeval_count < prev_score:
            logger.info("[RECONCILE] Score dropped, rolling back '%s'", name)
            existing_role = meta.get("role", "tool")
            mem.cold.register_nerve_rich(
                name, description, prev_prompt,
                json.dumps(meta.get("examples", [])), role=existing_role,
            )
    except Exception as exc:
        logger.warning("[RECONCILE] Rollback check failed: %s", exc)


def _save_final_score(name: str, state: _ImprovementState, old_score: float,
                      qual_threshold: float, max_iterations: int,
                      total_tests: int) -> None:
    """Save best score even if interrupted or exhausted.

    Only records if the score improved AND enough test cases ran
    to make the score trustworthy (MIN_TEST_COVERAGE).

    Args:
        name: Nerve name.
        state: Current improvement state with best score and results.
        old_score: Score before this improvement cycle started.
        qual_threshold: Minimum score to be considered qualified.
        max_iterations: Max iterations configured for this role.
        total_tests: Total number of test cases in the bank.
    """
    if not state.all_test_results or state.best_score <= old_score:
        return
    if not _has_sufficient_coverage(len(state.all_test_results), total_tests):
        logger.info("[RECONCILE] '%s' scored %.0f%% but only %d/%d tests ran — skipping save",
                    name, state.best_score * 100, len(state.all_test_results), total_tests)
        return
    passed_count = sum(1 for r in state.all_test_results if r["passed"])
    mem.cold.record_qualification(
        "nerve", name, state.best_score >= qual_threshold, state.best_score, max_iterations,
        len(state.all_test_results), passed_count, json.dumps(state.all_test_results),
    )
    publish_nerve_status()


# ── Dreamstate Manager (coordinates ALL sleep-time processes) ────────────────


class Dreamstate:
    """The brain's sleep cycle — ALL maintenance happens here, ALL of it stops on wake.

    Dreamstate phases, ALL interruptible:
    1. Consolidation — merge similar nerves
    2. MCP Fanout — discover new external tools and wire them into nerves
    3. Reconciliation — improve weak nerves (now with fresh tools)
    4. Brain upgrade — self-provision missing capabilities
    5. Fine-tuning — train LoRA adapters
    6. Personality observation — analyze interaction signals
    7. Personality evolution — evolve communication voice within anchored bounds

    The moment wake() is called (task arrives), the dreamstate worker is
    interrupted and must yield. The brain is either dreaming or awake. Never both.
    """

    def __init__(self):
        self._last_activity = time.time()
        self._timer = None
        self._interrupted = threading.Event()
        self._worker_thread = None
        self._lock = threading.Lock()
        self._schedule()

    def wake(self):
        """Brain wakes up — stop ALL dreamstate activity immediately.

        Blocks until the dreamstate worker has fully stopped and released
        all LLM locks, so think() can safely use the models.
        Clears conversation buffer so the new session starts fresh.
        """
        self._last_activity = time.time()
        self._interrupted.set()
        worker = self._worker_thread
        if worker and worker.is_alive():
            logger.info("[DREAMSTATE] Waking up — waiting for dreamstate to yield...")
            worker.join(timeout=60)
            if worker.is_alive():
                logger.warning("[DREAMSTATE] Dreamstate did not yield in 60s — proceeding anyway")
            else:
                logger.info("[DREAMSTATE] Dreamstate yielded — brain is awake")
            # Clear stale pre-dream conversation so wake session starts fresh
            mem.hot.clear_conversation()
        self._schedule()

    # Backward compat
    touch = wake

    def _schedule(self):
        """Schedule next dreamstate entry after IDLE_THRESHOLD seconds of silence."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(IDLE_THRESHOLD, self._enter_dreamstate)
            self._timer.daemon = True
            self._timer.start()

    def _enter_dreamstate(self):
        """Enter dreamstate if brain has been idle long enough."""
        elapsed = time.time() - self._last_activity
        if elapsed < IDLE_THRESHOLD:
            return
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._interrupted.clear()
        self._worker_thread = threading.Thread(
            target=self._dream, daemon=True, name="dreamstate",
        )
        self._worker_thread.start()

    def _dream(self):
        """The dream cycle — consolidate, reconcile, upgrade. Yields on any wake signal."""
        logger.info("[DREAMSTATE] Brain is dreaming...")
        # Clear pre-dream conversation so LLM context is clean
        mem.hot.clear_conversation()

        # Dream Phase 0: Community sync — pull new nerves/tools someone may have added
        if not self._interrupted.is_set():
            self._dream_community_sync()

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up after community sync")
            return

        # Dream Phase 1: Consolidation
        if not self._interrupted.is_set():
            try:
                consolidate_nerves(interrupted=self._interrupted)
            except Exception as e:
                logger.warning("[DREAMSTATE] Consolidation error: %s", e)

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up after consolidation")
            return

        # Dream Phase 2: MCP Fanout — wire new external tools into nerves BEFORE evaluation
        if not self._interrupted.is_set():
            self._dream_mcp_fanout()

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up after MCP fanout")
            return

        # Dream Phase 3: Reconciliation (nerves now have fresh tools)
        if not self._interrupted.is_set():
            self._dream_reconcile()

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up after reconciliation")
            return

        # Dream Phase 4: Brain upgrade (capability provisioning)
        if not self._interrupted.is_set():
            self._dream_upgrade()

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up after upgrade")
            return

        # Dream Phase 5: LoRA fine-tuning — train adapters for nerves with data
        if not self._interrupted.is_set():
            self._dream_finetune()

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up after fine-tuning")
            return

        # Dream Phase 6: Contribute — push nerves/adapters back to community
        if not self._interrupted.is_set():
            self._dream_contribute()

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up after contribution")
            return

        # Dream Phase 7: Usage report — write nerve/tool stats to community for visibility
        if not self._interrupted.is_set():
            self._dream_usage_report()

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up after usage report")
            return

        # Dream Phase 8: Personality observation + evolution
        if not self._interrupted.is_set():
            self._dream_personality()

        # Final sync — ensure Redis reflects all nerve state changes from this dream cycle
        publish_nerve_status()
        # Clear stale conversation context so wake starts fresh
        mem.hot.clear_conversation()
        logger.info("[DREAMSTATE] Dream cycle complete — brain resting")

    def _dream_community_sync(self):
        """Pull latest community manifest, seed new nerves/tools, and build tool environments."""
        try:
            from arqitect.brain.community import sync_manifest, seed_tools, seed_nerves, hydrate_nerve_bundle
            manifest = sync_manifest()
            if manifest:
                seed_tools()
                count = seed_nerves()
                if count:
                    logger.info("[DREAMSTATE] Community sync: registered %d new nerve(s)", count)
                    publish_nerve_status()

            # Build tool environments for any tools that need it
            self._dream_build_tool_envs()

            # Hydrate deferred bundles — fetch full metadata for nerves
            # that were registered lightweight at startup
            catalog = mem.cold.list_nerves()
            for nerve_name in catalog:
                if self._interrupted.is_set():
                    break
                meta = mem.cold.get_nerve_metadata(nerve_name)
                if not meta.get("system_prompt"):
                    hydrate_nerve_bundle(nerve_name)

        except Exception as e:
            logger.warning("[DREAMSTATE] Community sync error: %s", e)

    def _dream_build_tool_envs(self):
        """Build dependency environments for tools that need it.

        Scans mcp_tools/ for directories with a .needs_build marker or
        a version mismatch, builds their environments, and restarts
        warm processes.
        """
        try:
            from arqitect.mcp.env_builder import build_env, rebuild_env, env_ready
            from arqitect.config.loader import get_mcp_tools_dir
            import os
            import json

            tools_dir = str(get_mcp_tools_dir())
            if not os.path.isdir(tools_dir):
                return

            built = 0
            for entry in os.listdir(tools_dir):
                if self._interrupted.is_set():
                    break

                tool_dir = os.path.join(tools_dir, entry)
                if not os.path.isdir(tool_dir):
                    continue

                needs_build_marker = os.path.join(tool_dir, ".needs_build")
                if os.path.isfile(needs_build_marker):
                    # New tool — build from scratch
                    if build_env(tool_dir):
                        os.remove(needs_build_marker)
                        built += 1
                        logger.info("[DREAMSTATE] Built env for tool: %s", entry)
                        self._signal_tool_restart(entry)
                elif not env_ready(tool_dir):
                    # Version mismatch — rebuild
                    if rebuild_env(tool_dir):
                        built += 1
                        logger.info("[DREAMSTATE] Rebuilt env for tool: %s", entry)
                        self._signal_tool_restart(entry)

            if built:
                logger.info("[DREAMSTATE] Built environments for %d tool(s)", built)

        except Exception as e:
            logger.warning("[DREAMSTATE] Tool env build error: %s", e)

    def _signal_tool_restart(self, tool_name: str):
        """Publish a tool lifecycle event so the MCP server can restart the tool process.

        Args:
            tool_name: Name of the tool to restart.
        """
        try:
            publish_event(Channel.TOOL_LIFECYCLE, {
                "action": "restart",
                "tool": tool_name,
            })
        except Exception:
            pass

    def _dream_mcp_fanout(self):
        """MCP Fanout — progressively expand nerves from narrow seeds into domain experts.

        Each nerve starts as a narrow task (e.g. "get weather for city X").
        Fanout grows it across up to 7 iterations per dream cycle:

        Iteration 1: Generalize — expand task-specific seed into domain description
        Iteration 2-3: Equip — wire existing tools, fabricate missing ones
        Iteration 4-5: Deepen — refine system_prompt with tool-aware expertise
        Iteration 6-7: Sharpen — add edge cases, examples, domain-specific instructions

        Each iteration builds on the previous. Progress is tracked per-nerve
        so interrupted cycles resume where they left off.
        """
        MAX_FANOUT_ITERATIONS = 7

        try:
            from arqitect.brain.catalog import list_mcp_tools_with_info
            from arqitect.brain.helpers import llm_generate, extract_json
            from arqitect.matching import match_tools

            all_tools = list_mcp_tools_with_info()
            if not all_tools:
                logger.info("[DREAMSTATE] MCP Fanout: no tools available, skipping")
                return

            # Use full registry — NOT the routing catalog which filters out 0% nerves.
            # Dream state needs to work on ALL nerves, especially weak ones.
            catalog = mem.cold.list_nerves()
            nerves_evolved = 0

            for nerve_name, desc in catalog.items():
                if self._interrupted.is_set():
                    break
                if nerve_name in CORE_SENSES or mem.cold.is_sense(nerve_name):
                    continue

                # Track iteration progress per nerve (resumes across dream cycles)
                progress_raw = mem.cold.get_fact("fanout", nerve_name)
                current_iter = int(progress_raw) if progress_raw and progress_raw.isdigit() else 0
                if current_iter >= MAX_FANOUT_ITERATIONS:
                    continue  # Fully evolved

                # 95%+ AND already fanned out at least once → skip
                # (nerves can score 95%+ at synthesis before any fanout — those still need it)
                qual = mem.cold.get_qualification("nerve", nerve_name)
                if qual and qual.get("score", 0) >= 0.95 and current_iter >= 1:
                    continue

                # Clear context between nerves so LLM doesn't bleed across
                mem.hot.clear_conversation()

                nerve_meta = mem.cold.get_nerve_metadata(nerve_name)
                existing_tools = set(mem.cold.get_nerve_tools(nerve_name))
                all_tool_names = list(all_tools.keys())
                changed = False

                # ── Sibling discovery (once, before equip phase) ────
                if current_iter < 2 and not self._interrupted.is_set():
                    sibling_candidates = _discover_tool_siblings(
                        nerve_name, desc, existing_tools, all_tools,
                        self._interrupted,
                    )
                    siblings_fabricated = _fabricate_sibling_tools(
                        nerve_name, sibling_candidates, existing_tools,
                        all_tools, self._interrupted,
                    )
                    if siblings_fabricated:
                        # Refresh tool catalog so new tools appear in prompts
                        all_tools = list_mcp_tools_with_info()
                        all_tool_names = list(all_tools.keys())
                        changed = True

                for iteration in range(current_iter + 1, MAX_FANOUT_ITERATIONS + 1):
                    if self._interrupted.is_set():
                        break

                    current_desc = mem.cold.get_nerve_metadata(nerve_name).get("description", desc)
                    current_sp = mem.cold.get_nerve_metadata(nerve_name).get("system_prompt", "")
                    current_tool_list = list(existing_tools)

                    logger.info("[MCP-FANOUT] '%s' iteration %d/%d (tools: %d, desc: %s...)",
                               nerve_name, iteration, MAX_FANOUT_ITERATIONS,
                               len(existing_tools), current_desc[:50])

                    # ── Build iteration-aware prompt ─────────────────────
                    if iteration == 1:
                        # First iteration: generalize the seed
                        prompt = (
                            f"A nerve called '{nerve_name}' was created from this user request:\n"
                            f'  "{current_desc}"\n\n'
                            f"This nerve should become a domain expert, not just handle one task.\n\n"
                            f"Available MCP tools on this system:\n"
                            f"  {', '.join(all_tool_names)}\n\n"
                            f"Return a JSON object with:\n"
                            f'  "description": "A broad 1-sentence description of what this nerve handles as a domain expert (NOT task-specific, NOT mentioning specific cities/names/values from the original request)",\n'
                            f'  "system_prompt": "<3-5 sentences of expert-level behavioral instructions for this domain>",\n'
                            f'  "tools_needed": ["tool names from the available list that this expert would use"],\n'
                            f'  "tools_missing": ["snake_case tool names NOT in available tools that this expert needs"]\n\n'
                            f"Return ONLY the JSON object."
                        )
                    elif iteration <= 3:
                        # Iterations 2-3: equip with more tools
                        prompt = (
                            f"Nerve '{nerve_name}' is a domain expert: {current_desc}\n"
                            f"Current system_prompt: {current_sp[:200]}\n"
                            f"Current tools: {current_tool_list}\n\n"
                            f"All available MCP tools:\n  {', '.join(all_tool_names)}\n\n"
                            f"This nerve needs MORE tools to be a true expert. Think about:\n"
                            f"- What sub-tasks does this domain involve?\n"
                            f"- What data sources would an expert need?\n"
                            f"- What utility tools (formatting, conversion, search) would help?\n\n"
                            f"Return a JSON object with:\n"
                            f'  "tools_needed": ["additional tools from available list not yet wired"],\n'
                            f'  "tools_missing": ["new tools to fabricate — snake_case, specific purpose"],\n'
                            f'  "system_prompt": "<updated system prompt incorporating the new tools, explain how and when to use each, 4-6 sentences>"\n\n'
                            f"Return ONLY the JSON object."
                        )
                    elif iteration <= 5:
                        # Iterations 4-5: deepen expertise in system_prompt
                        prompt = (
                            f"Nerve '{nerve_name}' is a domain expert: {current_desc}\n"
                            f"Current system_prompt:\n  {current_sp}\n"
                            f"Tools available: {current_tool_list}\n\n"
                            f"Deepen this nerve's expertise. The system_prompt should include:\n"
                            f"- Specific instructions for how to handle common queries in this domain\n"
                            f"- What units, formats, or conventions to use\n"
                            f"- How to chain tools together for complex queries\n"
                            f"- Common mistakes to avoid\n\n"
                            f"Return a JSON object with:\n"
                            f'  "system_prompt": "<your improved, more detailed system prompt, 5-8 sentences, concrete and actionable>",\n'
                            f'  "examples": [{{"input": "<example user query>", "output": "<expected response format>"}}] (2-3 examples)\n\n'
                            f"Return ONLY the JSON object."
                        )
                    else:
                        # Iterations 6-7: sharpen with edge cases
                        prompt = (
                            f"Nerve '{nerve_name}' is a domain expert: {current_desc}\n"
                            f"System prompt:\n  {current_sp}\n"
                            f"Tools: {current_tool_list}\n\n"
                            f"Final refinement. Think about edge cases and failure modes:\n"
                            f"- What happens when a tool returns an error or no data?\n"
                            f"- What ambiguous queries might users send?\n"
                            f"- Are there any missing tools that would handle edge cases?\n"
                            f"- Should the system_prompt mention fallback strategies?\n\n"
                            f"Return a JSON object with:\n"
                            f'  "system_prompt": "<your refined system prompt with edge case handling, 5-8 sentences>",\n'
                            f'  "tools_missing": ["<tool_name_if_any>"],\n'
                            f'  "examples": [{{"input": "<tricky edge case query>", "output": "<how to handle it>"}}] (1-2 examples)\n\n'
                            f"Return ONLY the JSON object."
                        )

                    expand_raw = llm_generate("brain", prompt)
                    expand = extract_json(expand_raw)

                    if not expand:
                        logger.info("[MCP-FANOUT] '%s' iteration %d: no valid JSON, skipping", nerve_name, iteration)
                        continue

                    # ── Apply changes from this iteration ────────────────
                    new_desc = expand.get("description")
                    new_sp = expand.get("system_prompt")
                    new_examples = expand.get("examples")
                    tools_needed = expand.get("tools_needed", [])
                    tools_missing = expand.get("tools_missing", [])

                    # Update description (usually only iteration 1)
                    if new_desc and new_desc != current_desc:
                        mem.cold.update_nerve_description(nerve_name, new_desc)
                        _update_nerve_file_description(nerve_name, new_desc)
                        logger.info("[MCP-FANOUT] '%s' new desc: %s", nerve_name, new_desc[:80])
                        changed = True

                    # Update system_prompt
                    if new_sp and new_sp != current_sp:
                        role = nerve_meta.get("role", "tool")
                        ex = new_examples if isinstance(new_examples, list) else nerve_meta.get("examples", [])
                        mem.cold.register_nerve_rich(
                            nerve_name, new_desc or current_desc, new_sp, json.dumps(ex), role
                        )
                        logger.info("[MCP-FANOUT] '%s' system_prompt updated (iter %d)", nerve_name, iteration)
                        changed = True
                    elif new_examples and isinstance(new_examples, list):
                        # Update just examples
                        role = nerve_meta.get("role", "tool")
                        mem.cold.register_nerve_rich(
                            nerve_name, current_desc, current_sp, json.dumps(new_examples), role
                        )
                        changed = True

                    # Wire existing tools
                    for tool_name in tools_needed:
                        if isinstance(tool_name, str) and tool_name in all_tools and tool_name not in existing_tools:
                            mem.cold.add_nerve_tool(nerve_name, tool_name)
                            existing_tools.add(tool_name)
                            logger.info("[MCP-FANOUT] Wired '%s' -> '%s'", tool_name, nerve_name)
                            changed = True

                    # Keyword matching against current description
                    effective_desc = new_desc or current_desc
                    matches = match_tools(effective_desc, all_tools, threshold=2.0)
                    for tool_name, score in matches[:5]:
                        if tool_name not in existing_tools:
                            mem.cold.add_nerve_tool(nerve_name, tool_name)
                            existing_tools.add(tool_name)
                            logger.info("[MCP-FANOUT] Wired '%s' -> '%s' (score=%.1f)", tool_name, nerve_name, score)
                            changed = True

                    # Fabricate missing tools
                    for tool_name in tools_missing[:2]:
                        if self._interrupted.is_set():
                            break
                        if not isinstance(tool_name, str) or not tool_name:
                            continue
                        if tool_name in all_tools or tool_name in existing_tools:
                            continue
                        from arqitect.matching import best_match_tool
                        if best_match_tool(tool_name, all_tools, threshold=2.0):
                            continue
                        try:
                            from arqitect.brain.synthesis import fabricate_mcp_tool
                            fab_desc = f"Tool for {effective_desc}: {tool_name.replace('_', ' ')}"
                            fabricate_mcp_tool(tool_name, fab_desc, "query: str")
                            mem.cold.add_nerve_tool(nerve_name, tool_name)
                            existing_tools.add(tool_name)
                            logger.info("[MCP-FANOUT] Fabricated '%s' for '%s'", tool_name, nerve_name)
                            changed = True
                        except Exception as e:
                            logger.warning("[MCP-FANOUT] Failed to fabricate '%s': %s", tool_name, e)

                    # Save iteration progress
                    mem.cold.set_fact("fanout", nerve_name, str(iteration))

                # end iteration loop
                if changed:
                    nerves_evolved += 1
                    final_tools = mem.cold.get_nerve_tools(nerve_name)
                    final_meta = mem.cold.get_nerve_metadata(nerve_name)
                    logger.info("[MCP-FANOUT] '%s' evolved: %d tools, desc: %s",
                               nerve_name, len(final_tools), final_meta['description'][:60])

            if nerves_evolved:
                logger.info("[MCP-FANOUT] Evolved %d nerve(s)", nerves_evolved)
                publish_nerve_status()
            else:
                logger.info("[DREAMSTATE] MCP Fanout: all nerves fully evolved")

        except Exception as e:
            logger.exception("[DREAMSTATE] MCP Fanout error: %s", e)

    def _dream_reconcile(self):
        """Reconciliation phase — improve weak nerves."""
        queue = _build_work_queue()
        if not queue:
            logger.info("[DREAMSTATE] All nerves at %.0f%%+ — nothing to reconcile", _get_improvement_threshold() * 100)
            return

        total = len(queue)
        logger.info("[DREAMSTATE] Reconciling %d nerve(s) below %.0f%%", total, _get_improvement_threshold() * 100)
        publish_event(Channel.BRAIN_THOUGHT, {
            "stage": "reconciliation_start",
            "message": f"Improving {total} nerve(s) toward {int(_get_improvement_threshold() * 100)}% quality",
        })

        improved = 0
        for i, item in enumerate(queue):
            if self._interrupted.is_set():
                logger.info("[DREAMSTATE] Woken at nerve %d/%d — yielding", i + 1, total)
                break

            name = item["name"]
            if not _validate_nerve(name):
                continue

            # Clear context between nerves so LLM doesn't bleed across
            mem.hot.clear_conversation()

            logger.info("[DREAMSTATE] [%d/%d] Reconciling '%s' (%.0f%%)", i + 1, total, name, item['score'] * 100)
            publish_event(Channel.BRAIN_THOUGHT, {
                "stage": "reconciling",
                "message": f"Improving nerve '{name}' ({int(item['score'] * 100)}%) — {i+1}/{total}",
            })

            result = _improve_one_nerve(name, item["description"], item["score"], self._interrupted)
            if result["new_score"] > result["old_score"]:
                improved += 1
                logger.info("[DREAMSTATE] '%s': %.0f%% -> %.0f%%", name, result['old_score'] * 100, result['new_score'] * 100)
            else:
                logger.info("[DREAMSTATE] '%s': no improvement (%.0f%%)", name, result['old_score'] * 100)

            if not result["completed"]:
                break

        # No standalone pruning — only consolidate merges/deletes duplicates

        remaining = len(_build_work_queue())
        publish_nerve_status()
        publish_event(Channel.BRAIN_THOUGHT, {
            "stage": "reconciliation_done",
            "message": f"Improved {improved} nerve(s). {remaining} still below {int(_get_improvement_threshold() * 100)}%.",
        })
        logger.info("[DREAMSTATE] Reconciliation: improved %d, remaining %d", improved, remaining)

    def _dream_upgrade(self):
        """Adapter tuning phase — evolve brain, nerve, sense, and creative adapters.

        For each adapter role (brain, nerve, creative, code, awareness, communication, vision):
        1. Check if qualification_score < 0.95 (needs tuning)
        2. Analyze recent episodes to identify what's failing
        3. LLM suggests improvements to system_prompt, few_shot_examples, temperature
        4. Test the improved adapter against stored episodes
        5. Keep improvement if score rises, rollback if not
        6. Save updated qualification_score to context.json

        Also heals MCP tools that have high failure rates (> 30%).
        """
        try:
            self._heal_mcp_tools()
        except Exception as e:
            logger.warning("[DREAMSTATE] MCP tool healing error: %s", e)

        if self._interrupted.is_set():
            return

        try:
            self._tune_adapters()
        except Exception as e:
            logger.warning("[DREAMSTATE] Adapter tuning error: %s", e)

    def _heal_mcp_tools(self):
        """Heal MCP tools with high failure rates.

        Reads tool_stats from cold memory, identifies tools failing > 30%,
        reads the tool source, asks LLM to fix it, validates, and saves.
        """
        from arqitect.brain.helpers import llm_generate
        from arqitect.brain.config import CODE_MODEL
        from arqitect.config.loader import get_mcp_tools_dir
        mcp_dir = get_mcp_tools_dir()

        try:
            rows = mem.cold.conn.execute(
                "SELECT name, total_calls, successes, failures FROM tool_stats "
                "WHERE total_calls >= 3 AND failures > 0"
            ).fetchall()
        except Exception as e:
            logger.warning("[DREAMSTATE] Could not read tool_stats: %s", e)
            return

        healed = 0
        for row in rows:
            if self._interrupted.is_set():
                break

            name = row["name"]
            total = row["total_calls"]
            failures = row["failures"]
            failure_rate = failures / total if total > 0 else 0

            if failure_rate < 0.3:
                continue

            tool_path = os.path.join(mcp_dir, f"{name}.py")
            if not os.path.isfile(tool_path):
                continue

            logger.info("[TOOL-HEAL] '%s' failure rate: %.0f%% (%d/%d)", name, failure_rate * 100, failures, total)

            try:
                with open(tool_path) as f:
                    tool_code = f.read()
            except OSError as exc:
                logger.debug("[TOOL-HEAL] Could not read '%s': %s", tool_path, exc)
                continue

            if self._interrupted.is_set():
                break

            # Ask coder LLM to fix the tool
            fix_prompt = (
                f"This MCP tool '{name}' has a {failure_rate:.0%} failure rate.\n\n"
                f"Current code:\n```python\n{tool_code}\n```\n\n"
                "Common failure modes:\n"
                "- API endpoint changed or returns different format\n"
                "- Missing error handling for network timeouts\n"
                "- Parameter validation missing\n"
                "- JSON parsing failures\n\n"
                "Fix the tool. Add proper error handling. Keep the same function signature.\n"
                "Output ONLY the complete fixed Python code, no explanation."
            )

            try:
                fixed_code = llm_generate(CODE_MODEL, fix_prompt,
                    system="You fix Python tools. Output ONLY valid Python code.")
                # Strip markdown fences
                from arqitect.brain.helpers import strip_markdown_fences
                fixed_code = strip_markdown_fences(fixed_code).strip()

                if not fixed_code or len(fixed_code) < 20:
                    continue

                # Validate syntax
                compile(fixed_code, f"{name}.py", "exec")

                # Save with backup
                import shutil
                bak_path = f"{tool_path}.bak"
                shutil.copy2(tool_path, bak_path)
                with open(tool_path, "w") as f:
                    f.write(fixed_code)

                healed += 1
                logger.info("[TOOL-HEAL] Fixed '%s' (backup at %s.py.bak)", name, name)

            except SyntaxError:
                logger.warning("[TOOL-HEAL] Fix for '%s' had syntax errors, skipping", name)
            except Exception as e:
                logger.warning("[TOOL-HEAL] Failed to fix '%s': %s", name, e)

        if healed:
            logger.info("[DREAMSTATE] Healed %d MCP tool(s)", healed)

    def _tune_adapters(self):
        """Tune adapter system prompts and few-shot examples based on episode analysis.

        For each adapter with qualification_score < 0.95:
        1. Pull episodes relevant to this adapter's role
        2. Identify failure patterns
        3. LLM generates improved system_prompt and few_shot_examples
        4. Evaluate improvement via test replay
        5. Save to the correct model-specific variant (derived from model registry)
        6. When qualification_score reaches 0.95, contribute back to community repo
        """
        from arqitect.brain.adapters import (
            list_adapters_needing_tuning, save_context,
        )
        from arqitect.brain.helpers import llm_generate, extract_json
        from arqitect.brain.config import BRAIN_MODEL

        adapters = list_adapters_needing_tuning()
        if not adapters:
            logger.info("[DREAMSTATE] All adapters at 95%%+ — nothing to tune")
            return

        # Pull recent episodes for analysis
        try:
            rows = mem.warm.conn.execute(
                "SELECT task, nerve, tool, result_summary, success, timestamp "
                "FROM episodes ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()
            all_episodes = [dict(r) for r in rows] if rows else []
        except Exception as e:
            logger.warning("[DREAMSTATE] Could not load episodes for adapter tuning: %s", e)
            return

        # Build role-to-nerves mapping dynamically from cold memory
        all_nerve_data = mem.cold.get_all_nerve_data()
        # Senses map by name, nerves map by their role field
        sense_names = {n for n, d in all_nerve_data.items() if d.get("is_sense")}
        role_to_nerves = {}
        for n, data in all_nerve_data.items():
            nerve_role = data.get("role", "tool")
            # Senses are their own adapter role (awareness, communication, etc.)
            if data.get("is_sense"):
                role_to_nerves.setdefault(n, set()).add(n)
            else:
                # Map nerve roles to adapter roles
                adapter_role = nerve_role if nerve_role != "tool" else "nerve"
                role_to_nerves.setdefault(adapter_role, set()).add(n)

        tuned = 0
        for adapter_info in adapters:
            if self._interrupted.is_set():
                break

            role = adapter_info["role"]
            ctx = adapter_info["context"]
            current_score = adapter_info["current_score"]
            variant = adapter_info["variant"]

            # Filter episodes relevant to this role
            if role == "brain":
                # Brain routes everything — all episodes are relevant
                relevant_eps = all_episodes
            else:
                # Get nerves that belong to this adapter role
                nerves_for_role = role_to_nerves.get(role, set())
                relevant_eps = [e for e in all_episodes if e.get("nerve") in nerves_for_role]

            failures = [e for e in relevant_eps if not e.get("success", True)]
            failure_rate = len(failures) / len(relevant_eps) if relevant_eps else 0

            logger.info("[ADAPTER-TUNE] '%s' (%s): %d episodes, %.0f%% failure rate, score: %s",
                        role, variant, len(relevant_eps), failure_rate * 100, current_score)

            if self._interrupted.is_set():
                break

            mem.hot.clear_conversation()

            # Build episode summaries for the LLM
            ep_summary = [{
                "task": ep.get("task", "")[:80],
                "nerve": ep.get("nerve", ""),
                "tool": ep.get("tool", ""),
                "success": bool(ep.get("success", True)),
                "result": (ep.get("result_summary") or "")[:60],
            } for ep in relevant_eps[:20]]

            failure_details = [{
                "task": ep.get("task", "")[:80],
                "nerve": ep.get("nerve", ""),
                "result": (ep.get("result_summary") or "")[:100],
            } for ep in failures[:10]]

            current_sp = ctx.get("system_prompt", "")
            current_examples = ctx.get("few_shot_examples", [])

            tune_prompt = (
                f"You are tuning the '{role}' adapter for a Arqitect AI system.\n\n"
                f"CURRENT SYSTEM PROMPT:\n{current_sp}\n\n"
                f"CURRENT FEW-SHOT EXAMPLES ({len(current_examples)}):\n"
                f"{json.dumps(current_examples[:5], indent=1)}\n\n"
                f"RECENT EPISODES ({len(relevant_eps)} total, {len(failures)} failures):\n"
                f"{json.dumps(ep_summary, indent=1)}\n\n"
            )

            if failure_details:
                tune_prompt += (
                    f"FAILURE DETAILS:\n"
                    f"{json.dumps(failure_details, indent=1)}\n\n"
                )

            tune_prompt += (
                "Analyze the failures and improve the adapter.\n\n"
                "Rules:\n"
                "- Keep the system_prompt structure intact, refine wording\n"
                "- Add/fix few_shot_examples that address failure patterns\n"
                "- Few-shot examples should cover observed failure categories\n"
                "- Keep temperature/max_tokens unless there's a clear reason to change\n"
                "- Do NOT remove working examples, only add or refine\n"
                "- Maximum 10 few-shot examples total\n\n"
                "Output a JSON object with:\n"
                '  "system_prompt": "improved system prompt",\n'
                '  "few_shot_examples": [{...}],\n'
                '  "temperature": float (optional),\n'
                '  "max_tokens": int (optional),\n'
                '  "reasoning": "1-sentence explanation of what you changed"\n\n'
                "Output ONLY valid JSON."
            )

            try:
                result = llm_generate(BRAIN_MODEL, tune_prompt,
                    system="You tune AI adapter prompts. Output only valid JSON.")
                updates = extract_json(result)
            except Exception as e:
                logger.warning("[ADAPTER-TUNE] LLM failed for '%s': %s", role, e)
                continue

            if not updates:
                logger.info("[ADAPTER-TUNE] No valid JSON from tuning LLM for '%s'", role)
                continue

            if self._interrupted.is_set():
                break

            new_sp = updates.get("system_prompt", "")
            new_examples = updates.get("few_shot_examples", [])
            new_temp = updates.get("temperature")
            new_max = updates.get("max_tokens")
            reasoning = updates.get("reasoning", "")

            if not new_sp and not new_examples:
                continue

            # Build updated context
            updated_ctx = dict(ctx)
            if new_sp:
                updated_ctx["system_prompt"] = new_sp
            if new_examples and isinstance(new_examples, list):
                from arqitect.brain.adapters import get_tuning_config as _gtc2
                _fsl = _gtc2(role)["few_shot_limit"]
                updated_ctx["few_shot_examples"] = new_examples[:_fsl]
            if isinstance(new_temp, (int, float)):
                updated_ctx["temperature"] = round(float(new_temp), 2)
            if isinstance(new_max, int) and new_max > 0:
                updated_ctx["max_tokens"] = new_max

            # Bump qualification_score (capped at 0.95)
            old_score = float(ctx.get("qualification_score", 0))
            new_score = min(0.95, old_score + 0.1)
            updated_ctx["qualification_score"] = round(new_score, 2)

            # Save to the model-specific variant (derived from model registry)
            save_context(role, updated_ctx, variant=variant)
            tuned += 1

            logger.info("[ADAPTER-TUNE] '%s' (%s) tuned: %.0f%% -> %.0f%% (%s)",
                        role, variant, old_score * 100, new_score * 100, reasoning[:80])

            # Contribute to model-specific dir when:
            # 1. First of its kind (model adapter doesn't exist yet) → push as baseline
            # 2. Not first → push only if we topped the community's score
            from arqitect.brain.adapters import get_model_name_for_role
            model_slug = get_model_name_for_role(role)
            if model_slug:
                community_score = self._community_model_adapter_score(role, variant, model_slug)
                if community_score is None or new_score > community_score:
                    self._contribute_adapter(role, variant, updated_ctx)

            if self._interrupted.is_set():
                break

        if tuned:
            logger.info("[DREAMSTATE] Tuned %d adapter(s)", tuned)
            publish_event(Channel.BRAIN_THOUGHT, {
                "stage": "adapter_tuning",
                "message": f"Tuned {tuned} adapter(s) based on episode analysis",
            })
        else:
            logger.info("[DREAMSTATE] No adapter improvements this cycle")

    def _community_model_adapter_score(self, role: str, size_class: str, model_slug: str) -> float | None:
        """Get the qualification_score of a model-specific adapter in the community repo.

        Returns None if the model adapter doesn't exist (first of its kind).
        """
        community_dir = self._find_community_dir()
        if not community_dir:
            return None
        path = os.path.join(community_dir, "adapters", role, size_class, model_slug, "context.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                return float(json.load(f).get("qualification_score", 0))
        except (json.JSONDecodeError, OSError, TypeError):
            return 0.0

    def _contribute_adapter(self, role: str, variant: str, context: dict):
        """Contribute a tuned adapter to the model-specific directory via PR.

        Always writes to adapters/{role}/{size_class}/{model_name}/ —
        never to the size class defaults.
        Pushes context.json, meta.json, and test_bank.jsonl.
        """
        try:
            import subprocess
            from arqitect.brain.adapters import (
                get_contribution_path, _load_meta, _load_test_bank,
                build_meta_json,
            )

            community_dir = self._find_community_dir()
            if not community_dir:
                logger.info("[CONTRIBUTE] Community repo not found, skipping PR for %s/%s", role, variant)
                return

            size_class, model_slug, rel_dir = get_contribution_path(role)
            adapter_dir = os.path.join(community_dir, "adapters", rel_dir)
            os.makedirs(adapter_dir, exist_ok=True)

            # Write context.json
            with open(os.path.join(adapter_dir, "context.json"), "w") as f:
                json.dump(context, f, indent=2)

            # Write meta.json — same schema for brain senses and nerves
            existing_meta = _load_meta(role, size_class, model_slug)
            if not existing_meta:
                existing_meta = _load_meta(role, size_class)
            score = context.get("qualification_score", 0)
            meta = build_meta_json(
                role, model_slug, size_class,
                score=score, existing_meta=existing_meta,
            )
            with open(os.path.join(adapter_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # Write test_bank.jsonl from local cache
            tests = _load_test_bank(role, size_class, model_slug)
            if not tests:
                tests = _load_test_bank(role, size_class)
            if tests:
                with open(os.path.join(adapter_dir, "test_bank.jsonl"), "w") as f:
                    for entry in tests:
                        f.write(json.dumps(entry) + "\n")

            # Copy LoRA adapter if trained
            import shutil
            local_adapter = os.path.join(NERVES_DIR, role, "adapter", "adapter.gguf")
            if os.path.isfile(local_adapter):
                shutil.copy2(local_adapter, os.path.join(adapter_dir, "adapter.gguf"))
                meta["has_lora"] = True
                with open(os.path.join(adapter_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                logger.info("[CONTRIBUTE] Included LoRA adapter for %s/%s/%s", role, size_class, model_slug)

            # Create or update PR
            score = context.get("qualification_score", 0)
            _files = ["context.json", "meta.json"]
            if tests:
                _files.append("test_bank.jsonl")
            _contribute_pr(
                community_dir,
                add_path=os.path.join("adapters", rel_dir),
                commit_msg=f"tune({role}/{size_class}/{model_slug}): score {score}",
                pr_title=f"Adapter: {role}/{size_class}/{model_slug} at {score:.0%}",
                pr_body=(f"Auto-tuned by dream state.\n\n"
                         f"Role: {role}\nSize class: {size_class}\n"
                         f"Model: {model_slug}\nScore: {score}\n"
                         f"Files: {', '.join(_files)}\n\n"
                         f"Includes tuning config + qualification thresholds in meta.json."),
                branch_prefix=f"adapter-tune/{role}-{model_slug}",
                search_key=f"Adapter: {role}/{size_class}/{model_slug}",
            )

        except Exception as e:
            logger.warning("[CONTRIBUTE] Failed to contribute %s/%s: %s", role, variant, e)

    def _find_community_dir(self) -> str | None:
        """Locate the local clone of sentient-community."""
        from arqitect.brain.community import _cache_dir
        for candidate in [
            os.path.join(os.path.dirname(_cache_dir()), "sentient-community"),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "..", "sentient-community",
            ),
        ]:
            if os.path.isdir(os.path.join(candidate, ".git")):
                return candidate
        return None

    def _community_has_nerve(self, community_dir: str, name: str) -> bool:
        """Check if a nerve bundle exists in the community repo."""
        return os.path.isfile(os.path.join(community_dir, "nerves", name, "bundle.json"))

    def _community_nerve_has_model_adapter(self, community_dir: str, name: str, size_class: str) -> bool:
        """Check if a nerve already has context.json for this size class."""
        ctx_path = os.path.join(community_dir, "nerves", name, size_class, "context.json")
        return os.path.isfile(ctx_path)

    def _community_nerve_has_stack(self, community_dir: str, name: str, lang: str) -> bool:
        """Check if a nerve's tools already have implementations in this language."""
        bundle_path = os.path.join(community_dir, "nerves", name, "bundle.json")
        if not os.path.isfile(bundle_path):
            return False
        try:
            with open(bundle_path) as f:
                bundle = json.load(f)
            for tool in bundle.get("tools", []):
                if lang in tool.get("implementations", {}):
                    return True
            return False
        except (json.JSONDecodeError, OSError):
            return False

    def _build_nerve_bundle(self, name: str) -> dict | None:
        """Build a bundle.json from local nerve data for contribution."""
        from arqitect.brain.adapters import get_active_variant
        nerve_meta = mem.cold.get_nerve_metadata(name)
        if not nerve_meta:
            return None

        description = nerve_meta.get("description", "")
        role = nerve_meta.get("role", "tool")
        system_prompt = nerve_meta.get("system_prompt", "")
        examples = nerve_meta.get("examples", [])
        if isinstance(examples, str):
            try:
                examples = json.loads(examples)
            except (json.JSONDecodeError, TypeError):
                examples = []

        tool_entries = self._collect_tool_entries(name)

        bundle = {
            "name": name,
            "version": "1.0.0",
            "description": description,
            "role": role,
            "tags": [role, name.replace("_nerve", "").replace("_", "-")],
            "authors": [{"github": "otomus"}],
            "arqitect_version": ">=0.1.0",
            "tools": tool_entries,
        }

        return bundle

    def _collect_tool_entries(self, name: str) -> list[dict]:
        """Gather tool entries for a nerve bundle, filtering low-use and self-references."""
        tools_with_counts = mem.cold.get_nerve_tools_with_counts(name)
        from arqitect.config.loader import get_mcp_tools_dir
        mcp_tools_dir = str(get_mcp_tools_dir())
        entries = []
        for item in tools_with_counts:
            tool_name = item["tool"]
            if item["use_count"] < MIN_TOOL_USES_FOR_CONTRIBUTION:
                continue
            if tool_name == name:
                continue
            entry = {"name": tool_name, "spec": f"mcp_tools/{tool_name}/spec.json", "implementations": {}}
            for ext, lang in _EXT_TO_LANG.items():
                if os.path.isfile(os.path.join(mcp_tools_dir, f"{tool_name}{ext}")):
                    entry["implementations"][lang] = f"mcp_tools/{tool_name}/tool{ext}"
            entries.append(entry)
        return entries

    def _copy_nerve_adapter_to_community(self, name: str, nerve_dir: str, role: str):
        """Copy a nerve's LoRA adapter GGUF into the community nerve directory.

        Places adapter at: nerves/{name}/{size_class}/{model_slug}/adapter.gguf
        Mirrors the adapter directory structure.
        """
        local_adapter = os.path.join(NERVES_DIR, name, "adapter", "adapter.gguf")
        if not os.path.isfile(local_adapter):
            return

        try:
            from arqitect.brain.adapters import get_active_variant, get_model_name_for_role
            size_class = get_active_variant(role)
            model_slug = get_model_name_for_role(role)
        except (ImportError, ValueError):
            return

        if not size_class or not model_slug:
            return

        dest_dir = os.path.join(nerve_dir, size_class, model_slug)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(local_adapter, os.path.join(dest_dir, "adapter.gguf"))
        logger.info("[CONTRIBUTE] Included LoRA adapter for %s/%s/%s", name, size_class, model_slug)

    def _copy_tool_implementations(self, name: str, bundle: dict, nerve_dir: str):
        """Copy tool source files from local mcp_tools/ into the community nerve directory."""
        from arqitect.config.loader import get_mcp_tools_dir
        mcp_tools_dir = str(get_mcp_tools_dir())
        for tool in bundle.get("tools", []):
            tool_name = tool["name"]
            for lang, rel_path in tool.get("implementations", {}).items():
                ext = ".py" if lang == "python" else ".js" if lang == "javascript" else ""
                src = os.path.join(mcp_tools_dir, f"{tool_name}{ext}")
                if os.path.isfile(src):
                    dest_dir = os.path.join(nerve_dir, os.path.dirname(rel_path))
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy2(src, os.path.join(nerve_dir, rel_path))

    def _write_test_cases(self, name: str, nerve_dir: str):
        """Write test_cases.json from cold memory into the community nerve directory."""
        test_bank = mem.cold.get_test_bank(name) if hasattr(mem.cold, "get_test_bank") else None
        if test_bank:
            with open(os.path.join(nerve_dir, "test_cases.json"), "w") as f:
                json.dump(test_bank, f, indent=2)

    def _write_nerve_adapter_files(self, name: str, role: str, nerve_dir: str):
        """Write context.json and meta.json for our model into the nerve directory.

        Creates: {nerve_dir}/{size_class}/context.json, meta.json
                 {nerve_dir}/{size_class}/{model_slug}/context.json, meta.json
        """
        try:
            from arqitect.brain.adapters import (
                get_active_variant, get_model_name_for_role,
                get_tuning_config, get_temperature, build_meta_json,
            )
            size_class = get_active_variant(role)
            model_slug = get_model_name_for_role(role)
        except (ImportError, ValueError):
            return

        if not size_class or not model_slug:
            return

        nerve_meta = mem.cold.get_nerve_metadata(name)
        system_prompt = nerve_meta.get("system_prompt", "")
        examples = nerve_meta.get("examples", [])
        if isinstance(examples, str):
            try:
                examples = json.loads(examples)
            except (json.JSONDecodeError, TypeError):
                examples = []

        tuning_cfg = get_tuning_config(role)
        few_shot_limit = tuning_cfg.get("few_shot_limit", 5)
        temperature = get_temperature(role)

        total = nerve_meta.get("total_invocations", 0) or 0
        successes = nerve_meta.get("successes", 0) or 0
        score = successes / total if total > 0 else 0
        has_lora = os.path.isfile(os.path.join(NERVES_DIR, name, "adapter", "adapter.gguf"))

        # Size-class level (defaults for all models in this class)
        sc_dir = os.path.join(nerve_dir, size_class)
        os.makedirs(sc_dir, exist_ok=True)

        context = {
            "system_prompt": system_prompt,
            "few_shot_examples": examples[:few_shot_limit],
            "temperature": temperature,
            "qualification_score": round(score, 2),
        }
        meta = build_meta_json(role, size_class, size_class, score=score, has_lora=False)

        if not os.path.exists(os.path.join(sc_dir, "context.json")):
            with open(os.path.join(sc_dir, "context.json"), "w") as f:
                json.dump(context, f, indent=2)
        if not os.path.exists(os.path.join(sc_dir, "meta.json")):
            with open(os.path.join(sc_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        # Model-specific level (our exact model)
        model_dir = os.path.join(sc_dir, model_slug)
        os.makedirs(model_dir, exist_ok=True)

        model_meta = build_meta_json(role, model_slug, size_class, score=score, has_lora=has_lora)
        model_meta["score"] = round(score, 2)
        model_meta["qualified_by"] = "dreamstate"

        with open(os.path.join(model_dir, "context.json"), "w") as f:
            json.dump(context, f, indent=2)
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump(model_meta, f, indent=2)

    def _contribute_nerve_bundle(self, community_dir: str, name: str, bundle: dict):
        """Push a new nerve bundle to the community repo via PR.

        Writes bundle.json (identity + tools) and per-size-class context.json +
        meta.json files matching the community nerve directory structure.
        """
        import shutil
        from arqitect.config.loader import get_mcp_tools_dir

        nerve_dir = os.path.join(community_dir, "nerves", name)
        os.makedirs(nerve_dir, exist_ok=True)

        with open(os.path.join(nerve_dir, "bundle.json"), "w") as f:
            json.dump(bundle, f, indent=2)

        self._copy_tool_implementations(name, bundle, nerve_dir)
        self._write_test_cases(name, nerve_dir)

        role = bundle.get("role", "tool")
        self._write_nerve_adapter_files(name, role, nerve_dir)
        self._copy_nerve_adapter_to_community(name, nerve_dir, role)

        _contribute_pr(
            community_dir,
            add_path=f"nerves/{name}",
            commit_msg=f"Add nerve: {name}",
            pr_title=f"Add nerve: {name}",
            pr_body=(f"## Nerve: {name}\n\n{bundle.get('description', '')}\n\n"
                     f"Role: {bundle.get('role', 'tool')}\n"
                     f"Tools: {', '.join(t['name'] for t in bundle.get('tools', []))}\n\n"
                     f"Auto-contributed by dream state."),
            branch_prefix=f"contribute/{name}",
            search_key=f"Add nerve: {name}",
        )

    def _contribute_nerve_model_adapter(self, community_dir: str, name: str, size_class: str):
        """Push a model-specific adapter for an existing nerve.

        Writes {size_class}/context.json + meta.json and
        {size_class}/{model_slug}/context.json + meta.json into the
        community nerve directory, then creates a PR.
        """
        nerve_meta = mem.cold.get_nerve_metadata(name)
        if not nerve_meta:
            return

        nerve_role = nerve_meta.get("role", "tool")
        nerve_dir = os.path.join(community_dir, "nerves", name)
        self._write_nerve_adapter_files(name, nerve_role, nerve_dir)
        self._copy_nerve_adapter_to_community(name, nerve_dir, nerve_role)

        from arqitect.brain.adapters import get_model_name_for_role, get_temperature, get_tuning_config
        model_slug = get_model_name_for_role(nerve_role) or size_class
        total = nerve_meta.get("total_invocations", 0) or 0
        successes = nerve_meta.get("successes", 0) or 0
        score = successes / total if total > 0 else 0
        has_lora = os.path.isfile(os.path.join(
            nerve_dir, size_class, model_slug, "adapter.gguf"
        ))

        pr_body = (
            f"Model adapter for {name} on {size_class} models.\n\n"
            f"Model: {model_slug}\n"
            f"Score: {score:.0%}\n"
            f"LoRA adapter: {'included' if has_lora else 'none'}\n\n"
            f"Auto-contributed by dream state."
        )
        _contribute_pr(
            community_dir,
            add_path=f"nerves/{name}",
            commit_msg=f"adapter({name}/{size_class}): score {score:.2f}",
            pr_title=f"Nerve adapter: {name}/{size_class}/{model_slug}",
            pr_body=pr_body,
            branch_prefix=f"nerve-adapter/{name}-{size_class}",
            search_key=f"Nerve adapter: {name}/{size_class}",
        )

    def _contribute_nerve_stack(self, community_dir: str, name: str, lang: str):
        """Push a new language implementation for an existing nerve's tools."""
        import subprocess
        import shutil
        from arqitect.config.loader import get_mcp_tools_dir

        bundle_path = os.path.join(community_dir, "nerves", name, "bundle.json")
        if not os.path.isfile(bundle_path):
            return
        try:
            with open(bundle_path) as f:
                bundle = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        _EXT_MAP = {"python": ".py", "javascript": ".js", "typescript": ".ts"}
        ext = _EXT_MAP.get(lang, f".{lang}")
        mcp_tools_dir = str(get_mcp_tools_dir())
        nerve_dir = os.path.join(community_dir, "nerves", name)
        changed = False

        for tool in bundle.get("tools", []):
            tool_name = tool["name"]
            if lang in tool.get("implementations", {}):
                continue  # Already has this language
            src = os.path.join(mcp_tools_dir, f"{tool_name}{ext}")
            if not os.path.isfile(src):
                continue
            rel_path = f"tools/{tool_name}/tool{ext}"
            dest = os.path.join(nerve_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(src, dest)
            tool.setdefault("implementations", {})[lang] = rel_path
            changed = True

        if not changed:
            return

        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)

        _contribute_pr(
            community_dir,
            add_path=f"nerves/{name}",
            commit_msg=f"stack({name}): add {lang} implementation",
            pr_title=f"Nerve stack: {name} ({lang})",
            pr_body=(f"Add {lang} tool implementations for {name}.\n\n"
                     f"Auto-contributed by dream state."),
            branch_prefix=f"nerve-stack/{name}-{lang}",
            search_key=f"Nerve stack: {name} ({lang})",
        )

    def _dream_contribute(self):
        """Contribution phase — push nerves and adapters back to community.

        For each local nerve (excluding senses):
        1. New nerve (not in community) → push full bundle
        2. Existing nerve, new model size class → push model adapter
        3. Existing nerve, same model, new language stack → push stack
        """
        from arqitect.brain.adapters import get_active_variant

        community_dir = self._find_community_dir()
        if not community_dir:
            logger.info("[CONTRIBUTE] Community repo not found, skipping contribution phase")
            return

        all_nerve_data = mem.cold.get_all_nerve_data()
        contributed = 0

        for name, data in all_nerve_data.items():
            if self._interrupted.is_set():
                break
            # Skip senses — they use the adapter contribution path
            if data.get("is_sense"):
                continue

            role = data.get("role", "tool")
            size_class = get_active_variant(role)

            # Don't contribute until LoRA adapter is trained
            local_adapter = os.path.join(NERVES_DIR, name, "adapter", "adapter.gguf")
            if not os.path.isfile(local_adapter):
                logger.info("[CONTRIBUTE] Skipping '%s' — no adapter.gguf yet", name)
                continue

            if not self._community_has_nerve(community_dir, name):
                # Case 1: New nerve — push full bundle
                bundle = self._build_nerve_bundle(name)
                if bundle:
                    logger.info("[CONTRIBUTE] New nerve: %s", name)
                    self._contribute_nerve_bundle(community_dir, name, bundle)
                    contributed += 1
            else:
                # Case 2: Existing nerve, new model → push adapter
                if size_class != "core" and not self._community_nerve_has_model_adapter(
                    community_dir, name, size_class
                ):
                    logger.info("[CONTRIBUTE] New model adapter: %s/%s", name, size_class)
                    self._contribute_nerve_model_adapter(community_dir, name, size_class)
                    contributed += 1

                # Case 3: Existing nerve, new stack → push implementations
                from arqitect.config.loader import get_mcp_tools_dir
                mcp_tools_dir = str(get_mcp_tools_dir())
                nerve_tools = mem.cold.get_nerve_tools(name)
                local_langs = set()
                for t in nerve_tools:
                    for ext, lang in _EXT_TO_LANG.items():
                        if os.path.isfile(os.path.join(mcp_tools_dir, f"{t}{ext}")):
                            local_langs.add(lang)

                for lang in local_langs:
                    if self._interrupted.is_set():
                        break
                    if not self._community_nerve_has_stack(community_dir, name, lang):
                        logger.info("[CONTRIBUTE] New stack: %s/%s", name, lang)
                        self._contribute_nerve_stack(community_dir, name, lang)
                        contributed += 1

        if contributed:
            logger.info("[CONTRIBUTE] Contributed %d item(s) to community", contributed)
        else:
            logger.info("[CONTRIBUTE] Nothing new to contribute")

    def _expand_test_banks_for_training(self):
        """Generate test cases for qualified nerves that need more data for LoRA.

        Runs before fine-tuning so nerves can cross the min_training_examples
        threshold within a single dream cycle.
        """
        try:
            from arqitect.inference.tuner import collect_training_data
            from arqitect.critic.qualify_nerve import generate_test_cases
            from arqitect.brain.adapters import get_tuning_config
        except ImportError:
            return

        rows = mem.cold.conn.execute(
            "SELECT name, description, role FROM nerve_registry WHERE is_sense=0"
        ).fetchall()

        for row in rows:
            if self._interrupted.is_set():
                break

            name, description, role = row[0], row[1] or "", row[2] or "tool"

            # Only expand for qualified nerves — unqualified ones still need prompt tuning first
            if not mem.cold.is_qualified("nerve", name):
                continue

            cfg = get_tuning_config(role)
            min_needed = cfg["min_training_examples"]
            current_data = collect_training_data(name)
            if len(current_data) >= min_needed:
                continue  # Already has enough

            current_bank = mem.cold.get_test_bank(name)
            existing_inputs = {t.get("input", "") for t in current_bank}
            gap = min_needed - len(current_data)

            logger.info("[DREAMSTATE] Expanding test bank for '%s': %d tests, %d total data, need %d (gap: %d)",
                        name, len(current_bank), len(current_data), min_needed, gap)

            rounds = 0
            batch_size = cfg["test_cases_per_batch"]
            # Calculate enough rounds to cover the gap, with some headroom for duplicates
            max_rounds = max(5, (gap // max(batch_size, 1)) + 5)
            while len(current_bank) < min_needed and rounds < max_rounds:
                if self._interrupted.is_set():
                    break
                rounds += 1

                new_tests = generate_test_cases(
                    name, description, role=role,
                    existing_inputs=existing_inputs,
                )
                if not new_tests:
                    break

                unique = [t for t in new_tests if t.get("input", "") not in existing_inputs]
                if not unique:
                    break

                current_bank.extend(unique)
                existing_inputs.update(t.get("input", "") for t in unique)
                mem.cold.set_test_bank(name, current_bank)
                logger.info("[DREAMSTATE] '%s' test bank: +%d (total: %d, target: %d)",
                            name, len(unique), len(current_bank), min_needed)

            if len(current_bank) >= min_needed:
                logger.info("[DREAMSTATE] '%s' test bank ready for LoRA training", name)

    def _dream_finetune(self):
        """Fine-tuning phase — train LoRA adapters for nerves with enough data.

        This is the "deep loop": the reconciler (phase 2) tunes prompts and generates
        training data. This phase uses that data to train weight adapters that
        fundamentally change what the model can do for each nerve.

        Adapters are saved to nerves/{name}/adapter/adapter.gguf and automatically
        loaded at inference time by nerve_runtime.think_for_role().
        """
        try:
            from arqitect.inference.tuner import get_nerves_ready_for_training, train_nerve_adapter
        except ImportError as e:
            logger.warning("[DREAMSTATE] LoRA tuner not available: %s", e)
            return

        # Expand test banks for nerves that don't have enough training data yet.
        # The critic LLM generates batches until the minimum is met.
        self._expand_test_banks_for_training()

        candidates = get_nerves_ready_for_training()
        if not candidates:
            logger.info("[DREAMSTATE] No nerves ready for fine-tuning (need 20+ training examples)")
            return

        # Prioritize: nerves without adapters first, then by data count
        candidates.sort(key=lambda c: (c["has_adapter"], -c["data_count"]))

        logger.info("[DREAMSTATE] Fine-tuning %d nerve(s) with LoRA adapters", len(candidates))
        publish_event(Channel.BRAIN_THOUGHT, {
            "stage": "finetuning_start",
            "message": f"Training LoRA adapters for {len(candidates)} nerve(s)",
        })

        trained = 0
        for i, candidate in enumerate(candidates):
            if self._interrupted.is_set():
                logger.info("[DREAMSTATE] Woken at fine-tuning %d/%d — yielding", i + 1, len(candidates))
                break

            name = candidate["name"]
            role = candidate["role"]

            # Clear context between nerves so LLM doesn't bleed across
            mem.hot.clear_conversation()

            # Skip nerves that already have a recent adapter and high score
            if candidate["has_adapter"]:
                info = mem.cold.get_nerve_info(name)
                score = info.get("qual_score", 0) if isinstance(info, dict) else 0
                if score >= _get_improvement_threshold():
                    continue  # Already good enough

            logger.info("[DREAMSTATE] [%d/%d] Fine-tuning '%s' (%d examples)",
                        i + 1, len(candidates), name, candidate['data_count'])
            publish_event(Channel.BRAIN_THOUGHT, {
                "stage": "finetuning",
                "message": f"Training LoRA adapter for '{name}' — {i+1}/{len(candidates)}",
            })

            success = train_nerve_adapter(
                name, role=role, interrupted=self._interrupted,
            )

            if success:
                trained += 1
                logger.info("[DREAMSTATE] '%s' adapter trained successfully", name)

                # Re-qualify to measure improvement
                if not self._interrupted.is_set():
                    try:
                        from arqitect.critic.qualify_nerve import qualify_nerve
                        info = mem.cold.get_nerve_info(name) or {}
                        desc = info.get("description", "")
                        qual = qualify_nerve(name, desc, "", mem)
                        new_score = qual.get("score", 0)
                        logger.info("[DREAMSTATE] '%s' post-LoRA score: %.0f%%", name, new_score * 100)
                    except Exception as e:
                        logger.warning("[DREAMSTATE] Post-LoRA qualification failed: %s", e)
            else:
                logger.info("[DREAMSTATE] '%s' fine-tuning did not complete", name)

        if trained:
            publish_nerve_status()
        publish_event(Channel.BRAIN_THOUGHT, {
            "stage": "finetuning_done",
            "message": f"Trained {trained} LoRA adapter(s). "
                       f"Nerves will use specialized models on next invocation.",
        })
        logger.info("[DREAMSTATE] Fine-tuning: trained %d adapter(s)", trained)

    def _dream_usage_report(self):
        """Write nerve/tool usage stats to sentient-community for community visibility.

        Reads from the dedicated monitoring.db (not knowledge.db) and writes
        a JSON report to the community repo's reports/ directory.
        Non-blocking — failures are logged and skipped.
        """
        community_dir = self._find_community_dir()
        if not community_dir:
            logger.info("[USAGE] Community repo not found, skipping usage report")
            return

        try:
            from arqitect.memory.monitoring import MonitoringMemory
            monitoring = MonitoringMemory()
            report = monitoring.get_usage_report()

            if not report["nerves"] and not report["tools"] and not report["mcps"]:
                logger.info("[USAGE] No usage data to report")
                return

            import datetime
            report["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            report["instance_id"] = self._get_instance_id()

            reports_dir = os.path.join(community_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)

            report_path = os.path.join(reports_dir, f"usage_{report['instance_id']}.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # Prune old monitoring data to keep the db bounded
            monitoring.prune(older_than_days=90)

            logger.info(
                "[USAGE] Wrote report: %d nerve(s), %d tool(s), %d mcp(s) → %s",
                len(report["nerves"]), len(report["tools"]),
                len(report["mcps"]), report_path,
            )
        except Exception as e:
            logger.warning("[USAGE] Failed to write usage report: %s", e)

    def _get_instance_id(self) -> str:
        """Return a stable identifier for this server instance."""
        instance_id = mem.cold.get_fact("system", "instance_id")
        if not instance_id:
            import uuid
            instance_id = str(uuid.uuid4())[:8]
            mem.cold.set_fact("system", "instance_id", instance_id)
        return instance_id

    def _dream_personality(self):
        """Personality observation + evolution — analyze signals and evolve voice.

        Two-phase dream process:
        1. Observation — analyze accumulated interaction signals, score trait effectiveness
        2. Evolution — propose and apply trait changes within admin-defined anchor bounds

        Replaces the legacy _dream_reflect with structured signal-based evolution,
        anchor validation, and full history tracking.
        """
        if self._interrupted.is_set():
            return

        logger.info("[DREAMSTATE] Personality evolution starting...")

        from arqitect.brain.personality import observe_personality, evolve_personality

        seed = self._load_personality_seed()

        def _generate(model, prompt, system="", max_tokens=512):
            """LLM wrapper that checks for interrupts before calling."""
            if self._interrupted.is_set():
                return ""
            try:
                from arqitect.inference.router import generate_for_role
                from arqitect.brain.adapters import get_max_tokens
                return generate_for_role(
                    model, prompt, system=system,
                    max_tokens=min(max_tokens, get_max_tokens(model)),
                )
            except Exception as e:
                logger.warning("[DREAMSTATE] Personality LLM call failed: %s", e)
                return ""

        # Phase 1: Observation
        observation = observe_personality(mem.cold, _generate, seed)

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up during personality observation")
            return

        if observation is None:
            logger.info("[DREAMSTATE] Personality observation skipped (insufficient signals)")
            return

        # Phase 2: Evolution
        applied = evolve_personality(mem.cold, _generate, observation, seed)

        if self._interrupted.is_set():
            logger.info("[DREAMSTATE] Woken up during personality evolution")
            return

        if applied:
            publish_event(Channel.BRAIN_THOUGHT, {
                "stage": "personality_evolution",
                "message": (
                    f"Personality evolved: {', '.join(c['trait'] + ': ' + str(c['old']) + '->' + str(c['new']) for c in applied)}"
                ),
            })
            logger.info("[DREAMSTATE] Personality evolution complete: %d change(s)", len(applied))
        else:
            logger.info("[DREAMSTATE] Personality evolution: no changes applied")

    def _load_personality_seed(self) -> dict:
        """Load the personality seed from personality.json.

        Returns:
            Seed dict with trait_weights (or traits) and voice keys.
        """
        from arqitect.config.loader import get_project_root
        personality_path = os.path.join(str(get_project_root()), "personality.json")
        try:
            with open(personality_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("[DREAMSTATE] Could not load personality seed: %s", exc)
            return {
                "trait_weights": {
                    "wit": 0.5, "swagger": 0.3, "warmth": 0.7,
                    "formality": 0.3, "verbosity": 0.3, "pop_culture_refs": 0.1,
                },
            }


# Singleton
_dreamstate = None


def get_consolidator() -> Dreamstate:
    """Get the singleton dreamstate manager. Name kept for backward compat."""
    global _dreamstate
    if _dreamstate is None:
        _dreamstate = Dreamstate()
    return _dreamstate
