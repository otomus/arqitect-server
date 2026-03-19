"""
Nerve Qualification — closed-loop critic that tests nerves at creation time.
Uses LLM critic to evaluate nerve quality and correctness.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

import redis

from arqitect.config.loader import get_nerves_dir, get_sandbox_dir, get_mcp_tools_dir, get_redis_host_port

BRAIN_MODEL = "brain"
NERVES_DIR = get_nerves_dir()
SANDBOX_DIR = get_sandbox_dir()

_host, _port = get_redis_host_port()
_r = redis.Redis(host=_host, port=_port, decode_responses=True)


def _llm(prompt: str, system: str = "", role: str = "tool") -> str:
    """Call the brain model for critic reasoning via in-process inference.

    max_tokens is resolved from community tuning config (training_max_length).
    """
    try:
        from arqitect.inference.router import generate_for_role
        from arqitect.brain.adapters import get_tuning_config
        max_tokens = get_tuning_config(role)["training_max_length"]
        return generate_for_role("brain", prompt, system=system, max_tokens=max_tokens)
    except Exception as e:
        return f"Error: {e}"


def _extract_json(raw: str) -> dict | list | None:
    """Extract JSON from possibly noisy LLM output."""
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find a JSON array first, then object
    for open_ch, close_ch in [("[", "]"), ("{", "}")]:
        start = text.find(open_ch)
        if start < 0:
            continue
        # Track all bracket types for proper nesting
        depth_sq, depth_cu = 0, 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "[": depth_sq += 1
            elif c == "]": depth_sq -= 1
            elif c == "{": depth_cu += 1
            elif c == "}": depth_cu -= 1
            if depth_sq == 0 and depth_cu == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None


def _get_batch_size(role: str = "tool") -> int:
    """Get model-specific test cases per batch from community config."""
    from arqitect.brain.adapters import get_tuning_config
    return get_tuning_config(role)["test_cases_per_batch"]


def generate_test_cases(name: str, description: str, trigger_task: str = "",
                        count: int = 0, existing_inputs: set | None = None,
                        role: str = "tool") -> list[dict]:
    """Ask the critic LLM to generate domain test cases for a nerve.

    Each test case has: input, context, output, category.
    - input: the user query
    - context: runtime context (user details, location, time) that was present — use {} for no context
    - output: the expected response the nerve should produce
    - category: core|edge|boundary|negative

    count: how many to generate (0 = use model-specific default batch size)
    existing_inputs: set of inputs already in the test bank (to avoid duplicates)
    """
    batch_size = count or _get_batch_size(role)
    prompt = (
        f"You are a QA engineer. Generate test cases for a nerve agent called '{name}'.\n"
        f"Description: {description}\n"
    )
    if trigger_task:
        prompt += f"The user's original task was: \"{trigger_task}\"\n"
    if existing_inputs:
        prompt += f"\nAvoid these inputs that already exist: {list(existing_inputs)[:10]}\n"
    prompt += (
        f"\nGenerate {batch_size} test cases as a JSON array. Each test case has:\n"
        '  {"input": "user query", "context": {"userDetails": {"name": "Test User"}, "location": "New York, US", "timezone": "America/New_York", "messages": [{"role": "user", "content": "previous msg"}, {"role": "assistant", "content": "previous reply"}]}, "output": "expected response", "category": "core|edge|boundary|negative"}\n\n'
        "Fields:\n"
        "- input: the user's query string\n"
        "- context: runtime context object. Use {} when context doesn't matter. Fields:\n"
        "    - userDetails: user profile (name, language, preferences)\n"
        "    - location: city, country\n"
        "    - timezone: IANA timezone\n"
        "    - messages: recent conversation history [{role, content}, ...]\n"
        "- output: the actual expected text response the nerve should produce\n"
        "- category: core|edge|boundary|negative\n\n"
        "Categories:\n"
        f"- core ({max(batch_size // 3, 2)}): common queries this nerve must handle well\n"
        f"- edge ({max(batch_size // 4, 1)}): unusual but valid queries in this domain\n"
        "- boundary (1-2): empty or very long input\n"
        f"- negative ({max(batch_size // 4, 1)}): queries from a DIFFERENT domain the nerve should NOT handle\n\n"
        "Return ONLY the JSON array."
    )
    raw = _llm(prompt, role=role)
    print(f"[CRITIC] Raw test case output ({len(raw)} chars): {raw[:300]}")
    result = _extract_json(raw)
    if isinstance(result, list):
        return result
    # Fallback: try json.loads directly on stripped output
    try:
        stripped = raw.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            stripped = "\n".join(lines)
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    print(f"[CRITIC] Could not parse test cases from critic LLM output")
    return []


def run_nerve_with_input(nerve_name: str, user_input: str, mem_manager) -> dict:
    """Run a nerve subprocess with the given input and return structured results."""
    nerve_path = os.path.join(NERVES_DIR, nerve_name, "nerve.py")
    if not os.path.exists(nerve_path):
        return {"raw_stdout": "", "raw_stderr": f"Nerve '{nerve_name}' not found", "parsed": None, "exit_code": -1, "timed_out": False}

    env = os.environ.copy()
    env.update(mem_manager.get_env_for_nerve(nerve_name, user_input, user_id="system_test_user"))
    # Lazy-load only the model the nerve needs (not all 6)
    env["SYNAPSE_LAZY_LOAD"] = "1"
    # Skip expensive tool acquisition during qualification (HTTP searches, fabrication)
    env["SYNAPSE_NO_ACQUIRE"] = "1"
    # Skip fact-based answers during qualification — test the nerve's LLM, not fact recall
    env["SYNAPSE_SKIP_FACTS"] = "1"

    _nerve_timeout = 60
    try:
        result = subprocess.run(
            [sys.executable, nerve_path, user_input],
            capture_output=True, text=True, timeout=_nerve_timeout,
            cwd=SANDBOX_DIR, env=env,
        )
        parsed = None
        try:
            parsed = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "raw_stdout": result.stdout,
            "raw_stderr": result.stderr,
            "parsed": parsed,
            "exit_code": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {"raw_stdout": "", "raw_stderr": "Timed out", "parsed": None, "exit_code": -1, "timed_out": True}
    except Exception as e:
        return {"raw_stdout": "", "raw_stderr": str(e), "parsed": None, "exit_code": -1, "timed_out": False}


def _deterministic_check(output: str, user_input: str) -> float | None:
    """Quick deterministic checks. Returns score or None to defer to LLM."""
    if not output or len(output.strip()) < 2:
        return 0.0  # Empty output

    # Error detection
    error_markers = ["Error:", "Traceback", "timed out", "not found", "failed to"]
    if any(m.lower() in output.lower() for m in error_markers):
        return 0.1  # Error output

    # Echo detection
    from difflib import SequenceMatcher
    if SequenceMatcher(None, output.strip().lower(), user_input.strip().lower()).ratio() > 0.85:
        return 0.1  # Echoed input

    return None  # Defer to embedding/LLM evaluation



def evaluate_nerve_output(test_case: dict, nerve_output: dict) -> dict:
    """Evaluate nerve output using layered checks: deterministic, embedding, then LLM."""
    user_input = test_case.get("input", "")
    raw_stdout = nerve_output.get("raw_stdout", "")
    expected = test_case.get("output", test_case.get("expected_behavior", ""))

    # Layer 1: Deterministic check
    det_score = _deterministic_check(raw_stdout, user_input)
    if det_score is not None:
        passed = det_score >= 0.7
        reason = "empty output" if det_score == 0.0 else "error detected" if det_score == 0.1 else "deterministic pass"
        if det_score == 0.1 and raw_stdout and user_input:
            from difflib import SequenceMatcher
            if SequenceMatcher(None, raw_stdout.strip().lower(), user_input.strip().lower()).ratio() > 0.85:
                reason = "echoed input"
        return {"passed": passed, "score": det_score, "reasoning": reason, "issue": reason if not passed else ""}

    # Layer 2: LLM evaluation — the only reliable way to judge correctness
    expected_output = test_case.get("output", test_case.get("expected_behavior", ""))
    test_context = test_case.get("context", {})
    prompt = (
        f"Evaluate this nerve agent's response to a test case.\n\n"
        f"Test input: {test_case.get('input', '')}\n"
        f"Test context: {json.dumps(test_context) if test_context else 'none'}\n"
        f"Expected output: {expected_output}\n"
        f"Category: {test_case.get('category', '')}\n\n"
        f"Nerve stdout: {nerve_output.get('raw_stdout', '')[:500]}\n"
        f"Nerve stderr: {nerve_output.get('raw_stderr', '')[:300]}\n"
        f"Exit code: {nerve_output.get('exit_code', -1)}\n"
        f"Timed out: {nerve_output.get('timed_out', False)}\n\n"
        "Score the response:\n"
        '- Return a JSON object: {"passed": true/false, "score": 0.0-1.0, "reasoning": "why", "issue": "specific problem or empty string"}\n'
        "- score 1.0 = perfect, 0.7 = acceptable, 0.0 = completely wrong\n"
        "- For 'negative' category tests: the nerve SHOULD refuse or indicate it can't help (that's a PASS)\n"
        "- For 'boundary' tests: not crashing is a pass, graceful handling is bonus\n"
        "- If timed out or exit_code != 0 for core/edge: that's a fail\n"
        "Return ONLY the JSON object."
    )
    raw = _llm(prompt)
    result = _extract_json(raw)
    if isinstance(result, dict):
        return {
            "passed": bool(result.get("passed", False)),
            "score": float(result.get("score", 0.0)),
            "reasoning": result.get("reasoning", ""),
            "issue": result.get("issue", ""),
        }
    return {"passed": False, "score": 0.0, "reasoning": "Failed to parse evaluation", "issue": "evaluation_parse_error"}


def _is_junk_rule(rule: str) -> bool:
    """Detect rules that are vague filler or leak internal metrics.

    These patterns appear when the reconciler's evaluation feedback leaks into
    improvement suggestions, or when the LLM generates generic non-actionable rules.
    """
    r = rule.lower()
    # Internal metric references — nerve prompts should never mention these
    metric_leaks = ["embedding similarity", "embedding score", "cosine similarity",
                    "scoring", "low score", "high score", "improve score"]
    if any(m in r for m in metric_leaks):
        return True
    # Vague filler — rules that say nothing specific
    vague_patterns = [
        "ensure.*contextually appropriate",
        "ensure.*contextually relevant",
        "validate.*against.*request",
        "ensure.*outputs are.*relevant",
    ]
    import re
    for pat in vague_patterns:
        if re.search(pat, r):
            return True
    return False


def _is_duplicate_rule(new_rule: str, existing_prompt: str) -> bool:
    """Check if a new rule substantially overlaps with existing rules."""
    new_words = set(new_rule.lower().split())
    # Remove common filler words
    filler = {"the", "a", "an", "is", "are", "to", "and", "or", "of", "for",
              "in", "with", "that", "this", "be", "on", "it", "rule:", "ensure"}
    new_words -= filler
    if len(new_words) < 3:
        return True  # Too short to be meaningful

    for line in existing_prompt.split("\n"):
        if not line.startswith("Rule:"):
            continue
        existing_words = set(line.lower().split()) - filler
        if not existing_words:
            continue
        overlap = len(new_words & existing_words) / max(len(new_words), 1)
        if overlap > 0.6:
            return True
    return False


def _is_junk_description(desc: str) -> bool:
    """Detect descriptions polluted by reconciler jargon."""
    d = desc.lower()
    junk_terms = ["embedding similarity", "embedding score", "refine",
                  "scoring cases", "improve score", "low scoring",
                  "high embedding", "cosine"]
    return any(t in d for t in junk_terms)


def _extract_tool_errors(failures: list[dict]) -> list[dict]:
    """Extract tool-specific errors from test run stderr/stdout."""
    tool_errors = []
    seen = set()
    for f in failures:
        stderr = f.get("raw_stderr", "")
        stdout = f.get("raw_stdout", "")
        combined = stderr + "\n" + stdout

        for line in combined.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Pattern 1: [NERVE:name] Tool call failed (tool_name): ...
            m = re.search(r'Tool call failed \((\w+)\):\s*(.*)', line)
            if m:
                tool_name, error_msg = m.group(1), m.group(2)
                key = (tool_name, error_msg[:80])
                if key not in seen:
                    seen.add(key)
                    tool_errors.append({"tool": tool_name, "error": error_msg, "input": f.get("input", "")})
                continue

            # Pattern 1b: [NERVE:name] Tool call failed: MCP call error (tool_name): ...
            m = re.search(r'Tool call failed:.*MCP call error \((\w+)\):\s*(.*)', line)
            if m:
                tool_name, error_msg = m.group(1), m.group(2)
                key = (tool_name, error_msg[:80])
                if key not in seen:
                    seen.add(key)
                    tool_errors.append({"tool": tool_name, "error": error_msg, "input": f.get("input", "")})
                continue

            # Pattern 2: [NERVE:name] Tool call failed: {"error": "..."}  or  Tool call failed: Error ...
            m = re.search(r'Tool call failed:\s*(.*)', line)
            if m:
                error_text = m.group(1)
                # Try to extract tool name from earlier in stderr
                tm = re.search(r"Invoking.*?'(\w+)'|call.*?(\w+)\(", stderr)
                tool_name = (tm.group(1) or tm.group(2)) if tm else "unknown"
                key = (tool_name, error_text[:80])
                if key not in seen:
                    seen.add(key)
                    tool_errors.append({"tool": tool_name, "error": error_text, "input": f.get("input", "")})
                continue

            # Pattern 3: [NERVE:name] Tool returned error: ...
            m = re.search(r'Tool returned error:\s*(.*)', line)
            if m:
                error_text = m.group(1)
                tm = re.search(r"Invoking.*?'(\w+)'|call.*?(\w+)\(", stderr)
                tool_name = (tm.group(1) or tm.group(2)) if tm else "unknown"
                key = (tool_name, error_text[:80])
                if key not in seen:
                    seen.add(key)
                    tool_errors.append({"tool": tool_name, "error": error_text, "input": f.get("input", "")})
                continue

            # Pattern 4: [MCP] tool_name(...) → Error ... (from dashboard/bridge logs)
            m = re.search(r'\[MCP\]\s+(\w+)\(.*?\)\s*→\s*(Error.*)', line)
            if m:
                tool_name, error_msg = m.group(1), m.group(2)
                key = (tool_name, error_msg[:80])
                if key not in seen:
                    seen.add(key)
                    tool_errors.append({"tool": tool_name, "error": error_msg, "input": f.get("input", "")})
                continue

        # Pattern 5: Check stdout JSON for tool results containing "Error"
        # (tools that return errors as success, e.g. dateparser returning "Error parsing date...")
        try:
            parsed = json.loads(stdout) if stdout.strip().startswith("{") else None
            if parsed and isinstance(parsed, dict):
                answer = parsed.get("answer", "")
                if isinstance(answer, str) and answer.startswith("Error"):
                    # Try to find which tool was called from stderr
                    tool_calls = re.findall(r"Mapped '(\w+)'|call.*?'(\w+)'|MCP call.*?\((\w+)\)", stderr)
                    for tc in tool_calls:
                        tool_name = next((t for t in tc if t), "unknown")
                        key = (tool_name, answer[:80])
                        if key not in seen:
                            seen.add(key)
                            tool_errors.append({"tool": tool_name, "error": answer, "input": f.get("input", "")})
        except (json.JSONDecodeError, TypeError):
            pass

    return tool_errors[:5]  # Cap at 5


_MCP_TOOLS_DIR = get_mcp_tools_dir()


def _apply_tool_fix(tool_name: str, fixed_code: str, known_tools: list) -> bool:
    """Apply a tool code fix with safety validation. Returns True on success."""
    # Guard: only fix tools the nerve uses
    if tool_name not in known_tools:
        print(f"[CRITIC] Rejected tool fix for '{tool_name}' — not in nerve's known tools")
        return False

    # Guard: don't fix core tools
    try:
        from arqitect.mcp.server import CORE_TOOLS
        if tool_name in CORE_TOOLS:
            print(f"[CRITIC] Rejected tool fix for '{tool_name}' — core tool")
            return False
    except ImportError:
        pass

    tool_path = os.path.join(_MCP_TOOLS_DIR, f"{tool_name}.py")
    if not os.path.exists(tool_path):
        print(f"[CRITIC] Tool file not found: {tool_path}")
        return False

    # Syntax check
    try:
        compile(fixed_code, f"{tool_name}.py", "exec")
    except SyntaxError as e:
        print(f"[CRITIC] Syntax error in tool fix for '{tool_name}': {e}")
        return False

    # Must have a run() function
    if "def run(" not in fixed_code:
        print(f"[CRITIC] Tool fix for '{tool_name}' missing run() function")
        return False

    # Backup current version
    backup_path = tool_path + ".bak"
    shutil.copy2(tool_path, backup_path)

    # Write fix
    with open(tool_path, "w") as f:
        f.write(fixed_code)

    print(f"[CRITIC] Applied tool fix for '{tool_name}' (backup at {backup_path})")
    return True


def _rollback_tool_fix(tool_name: str) -> bool:
    """Restore a tool from its .bak file."""
    tool_path = os.path.join(_MCP_TOOLS_DIR, f"{tool_name}.py")
    backup_path = tool_path + ".bak"
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, tool_path)
        os.remove(backup_path)
        print(f"[CRITIC] Rolled back tool fix for '{tool_name}'")
        return True
    return False


def _cleanup_bak_files(known_tools: list):
    """Remove .bak files for tools that were successfully fixed."""
    for t in known_tools:
        bak = os.path.join(_MCP_TOOLS_DIR, f"{t}.py.bak")
        if os.path.exists(bak):
            os.remove(bak)
            print(f"[CRITIC] Cleaned up backup: {t}.py.bak")


def suggest_improvements(name: str, desc: str, system_prompt: str, examples: list, known_tools: list, failures: list[dict], tool_errors: list[dict] = None, role: str = "tool") -> dict:
    """Ask the critic LLM to analyze failures and suggest additive improvements.

    Instead of rewriting the entire system prompt, generates specific rules
    to append, a refined description for routing, and new examples.
    When tool errors are present, can also suggest tool code fixes.

    Guards against:
    - Internal metric leakage (rules about "embedding similarity")
    - Duplicate rules (same idea rephrased)
    - Vague filler rules ("ensure contextually appropriate")
    - Description pollution (reconciler jargon replacing what the nerve does)
    """
    failure_summary = "\n".join(
        f"- Input: {f.get('input', '')} | Issue: {f.get('issue', '')} | Score: {f.get('score', 0)}"
        for f in failures[:5]
    )

    # Build tool error section if present
    tool_error_section = ""
    tool_fix_instruction = ""
    if tool_errors:
        tool_error_lines = "\n".join(
            f"- Tool: {e['tool']} | Error: {e['error'][:200]} | Input: {e['input'][:100]}"
            for e in tool_errors
        )
        tool_error_section = f"\nTool errors observed during testing:\n{tool_error_lines}\n"

        # Read current tool source code for erroring tools
        tool_sources = {}
        for e in tool_errors:
            t = e["tool"]
            if t in tool_sources:
                continue
            tool_path = os.path.join(_MCP_TOOLS_DIR, f"{t}.py")
            if os.path.exists(tool_path):
                try:
                    with open(tool_path) as f:
                        tool_sources[t] = f.read()
                except Exception:
                    pass
        if tool_sources:
            source_section = "\n".join(
                f"--- {t}.py ---\n{src}\n--- end ---"
                for t, src in tool_sources.items()
            )
            tool_error_section += f"\nCurrent tool source code:\n{source_section}\n"

        tool_fix_instruction = (
            '"tool_fixes": [{"tool": "tool_name", "problem": "what is wrong", "fixed_code": "complete corrected Python source code"}], '
            "// Only include tool_fixes if the tool has a clear code bug. Include the COMPLETE file content, not just the changed part.\n"
        )

    prompt = (
        f"A nerve agent called '{name}' ({desc}) is failing some tests.\n\n"
        f"Current system_prompt: {system_prompt}\n"
        f"Current examples: {json.dumps(examples)}\n"
        f"Known tools: {known_tools}\n\n"
        f"Failures:\n{failure_summary}\n"
        f"{tool_error_section}\n"
        "Suggest improvements as a JSON object:\n"
        '{"rule": "a specific rule addressing the failures (1 sentence)", '
        '"examples": [new examples with input/output to add], '
        f'{tool_fix_instruction}'
        '"discover_tools": ["tool domain to acquire, e.g. date formatting"], '
        '"index_domains": ["knowledge domain to index, e.g. html, css, sql"], '
        '"description": "refined one-line description for routing (or empty string to keep current)"}\n\n'
        "Guidelines:\n"
        "- The rule must be a SPECIFIC directive about the nerve's BEHAVIOR, not about test scores\n"
        "- GOOD rules: 'Always include the unit in conversion results', 'Return just the number, no explanation'\n"
        "- BAD rules: 'Ensure outputs are contextually relevant', 'Improve embedding similarity'\n"
        "- NEVER reference internal metrics (scores, similarity, embeddings) in rules or descriptions\n"
        "- Do NOT duplicate rules already in the system prompt — only add genuinely new guidance\n"
        "- The description must say WHAT the nerve does for the user, not how to improve it\n"
        "- GOOD description: 'Solve math problems: arithmetic, algebra, word problems'\n"
        "- BAD description: 'Refined rule to improve scoring on math operations'\n"
        "- Keep ALL improvements focused on this nerve's SPECIFIC PURPOSE and DOMAIN\n"
        "- Do NOT add generic instructions like 'provide helpful responses' or 'assist the user'\n"
        "- The system_prompt must remain specific to ONLY this nerve's domain — if a rule could apply to any nerve, don't add it\n"
        "- Add examples that show the exact input→output for failing cases\n"
        "- If tool errors show a clear code bug (wrong parsing, missing library import, bad format handling), include a tool_fixes entry with the COMPLETE corrected Python file\n"
        "- Only fix tools that appear in Known tools\n"
        "Return ONLY the JSON object."
    )
    raw = _llm(prompt)
    result = _extract_json(raw)
    if isinstance(result, dict):
        # Build additive system prompt: append the rule to existing prompt
        rule = result.get("rule", "")
        new_system_prompt = system_prompt
        if rule and rule.strip():
            # Guard: reject junk rules and duplicates
            if _is_junk_rule(rule):
                print(f"[QUALIFY] Rejected junk rule for '{name}': {rule[:80]}")
            elif _is_duplicate_rule(rule, system_prompt):
                print(f"[QUALIFY] Rejected duplicate rule for '{name}': {rule[:80]}")
            else:
                if system_prompt:
                    new_system_prompt = f"{system_prompt}\nRule: {rule.strip()}"
                else:
                    new_system_prompt = f"Rule: {rule.strip()}"
                # Cap total system prompt length at 800 chars
                if len(new_system_prompt) > 800:
                    new_system_prompt = new_system_prompt[:800]

        # Guard: reject polluted descriptions
        new_desc = result.get("description", "")
        if new_desc and _is_junk_description(new_desc):
            print(f"[QUALIFY] Rejected junk description for '{name}': {new_desc[:80]}")
            new_desc = ""

        # Merge new examples with existing ones (additive)
        new_examples = result.get("examples", [])
        merged_examples = list(examples) if examples else []
        if isinstance(new_examples, list):
            merged_examples.extend(new_examples)
            # Cap examples at model-specific few_shot_limit
            from arqitect.brain.adapters import get_tuning_config
            _fsl = get_tuning_config(role)["few_shot_limit"]
            merged_examples = merged_examples[:_fsl]

        return {
            "system_prompt": new_system_prompt,
            "examples": merged_examples,
            "discover_tools": result.get("discover_tools", []),
            "index_domains": result.get("index_domains", []),
            "description": new_desc,
            "tool_fixes": result.get("tool_fixes", []),
        }
    return {"system_prompt": system_prompt, "examples": examples, "discover_tools": [], "index_domains": [], "description": "", "tool_fixes": []}


def _publish_progress(nerve_name: str, score: float, qualified: bool | None, tools: list, iteration: int, max_iter: int):
    """Publish qualification progress to Redis for dashboard."""
    status = "testing" if qualified is None else ("pass" if qualified else "fail")
    try:
        _r.publish("nerve:qualification", json.dumps({
            "nerves": [{
                "name": nerve_name,
                "score": round(score * 100),
                "qualified": qualified,
                "status": status,
                "tools": tools,
                "iteration": iteration,
                "max_iterations": max_iter,
            }]
        }))
    except Exception:
        pass


def qualify_nerve(name: str, description: str, trigger_task: str, mem_manager) -> dict:
    """Main qualification loop: generate tests, run nerve, evaluate, improve if needed.

    Returns {qualified, score, iterations, test_results}.
    Total timeout: 120 seconds. If exceeded, returns best score so far.
    """
    # Load model-specific config from community meta.json
    nerve_meta = mem_manager.cold.get_nerve_metadata(name)
    _nerve_role = nerve_meta.get("role", "tool")
    from arqitect.brain.adapters import get_tuning_config
    _tcfg = get_tuning_config(_nerve_role)
    MAX_ITERATIONS = _tcfg["max_qualification_iterations"]
    THRESHOLD = _tcfg["qualification_threshold"]
    TOTAL_TIMEOUT = _tcfg["qualification_timeout"]

    _start_time = time.time()

    known_tools = mem_manager.cold.get_nerve_tools(name)

    # Pre-acquire missing tools before qualification tests.
    # Tool-dependent nerves (e.g. weather) cannot pass qualification if their
    # declared tools don't actually exist — the nerve will hallucinate instead
    # of calling a real tool.  Acquire them upfront so tests exercise the full
    # nerve→tool pipeline.
    if known_tools:
        from arqitect.nerves.nerve_runtime import mcp_tool_exists
        missing = [t for t in known_tools if not mcp_tool_exists(t)]
        if missing:
            print(f"[CRITIC] Nerve '{name}' has missing tools: {missing} — acquiring before tests")
            import concurrent.futures
            from arqitect.nerves.nerve_runtime import acquire_tool
            for tool_need in missing:
                if time.time() - _start_time > 60:
                    print(f"[CRITIC] Tool pre-acquisition timeout, continuing with available tools")
                    break
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(acquire_tool, f"{description}: {tool_need}")
                        acquired = future.result(timeout=30)
                    if acquired:
                        mem_manager.cold.add_nerve_tool(name, acquired)
                        print(f"[CRITIC] Pre-acquired tool '{acquired}' for nerve '{name}'")
                except (concurrent.futures.TimeoutError, Exception) as e:
                    print(f"[CRITIC] Failed to pre-acquire tool '{tool_need}': {e}")
            known_tools = mem_manager.cold.get_nerve_tools(name)

    _publish_progress(name, 0.0, None, known_tools, 0, MAX_ITERATIONS)

    all_test_results = []

    # Stable test bank: reuse stored tests if available
    stored_tests = mem_manager.cold.get_test_bank(name)

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Check total timeout before each iteration
        elapsed = time.time() - _start_time
        if elapsed > TOTAL_TIMEOUT:
            print(f"[CRITIC] Qualification timeout ({elapsed:.0f}s > {TOTAL_TIMEOUT}s) for nerve '{name}'")
            break
        print(f"[CRITIC] Qualification iteration {iteration}/{MAX_ITERATIONS} for nerve '{name}'")

        # Use stored test bank if sufficient, otherwise generate new tests
        if stored_tests and len(stored_tests) >= 4:
            tests = stored_tests
            print(f"[CRITIC] Reusing stored test bank ({len(tests)} tests) for '{name}'")
        else:
            tests = generate_test_cases(name, description, trigger_task)
            if tests:
                # Store the generated tests for future reuse
                mem_manager.cold.set_test_bank(name, tests)
                stored_tests = tests
                print(f"[CRITIC] Generated and stored {len(tests)} tests for '{name}'")
        if not tests:
            print(f"[CRITIC] Failed to generate test cases for '{name}'")
            _publish_progress(name, 0.0, False, known_tools, iteration, MAX_ITERATIONS)
            return {"qualified": False, "score": 0.0, "iterations": iteration, "test_results": []}

        # Run and evaluate each test (with per-iteration timeout check)
        iteration_results = []
        total_score = 0.0
        for tc in tests:
            # Bail if we're running out of time
            if time.time() - _start_time > TOTAL_TIMEOUT:
                print(f"[CRITIC] Timeout mid-iteration, stopping tests for '{name}'")
                break

            user_input = tc.get("input", "")
            if not user_input:
                continue

            output = run_nerve_with_input(name, user_input, mem_manager)
            evaluation = evaluate_nerve_output(tc, output)
            evaluation["input"] = user_input
            evaluation["category"] = tc.get("category", "")
            evaluation["context"] = tc.get("context", {})
            evaluation["expected"] = tc.get("output", tc.get("expected_behavior", ""))
            evaluation["raw_stderr"] = output.get("raw_stderr", "")
            evaluation["raw_stdout"] = output.get("raw_stdout", "")
            iteration_results.append(evaluation)
            total_score += evaluation["score"]

        if not iteration_results:
            _publish_progress(name, 0.0, False, known_tools, iteration, MAX_ITERATIONS)
            return {"qualified": False, "score": 0.0, "iterations": iteration, "test_results": []}

        avg_score = total_score / len(iteration_results)
        passed_count = sum(1 for r in iteration_results if r["passed"])
        all_test_results = iteration_results

        print(f"[CRITIC] Nerve '{name}' iteration {iteration}: score={avg_score:.2f}, passed={passed_count}/{len(iteration_results)}")
        _publish_progress(name, avg_score, None, known_tools, iteration, MAX_ITERATIONS)

        # Persist progress to DB after each iteration so it survives restarts
        mem_manager.cold.record_qualification(
            "nerve", name, avg_score >= THRESHOLD, avg_score, iteration,
            len(iteration_results), passed_count,
            json.dumps(iteration_results),
        )

        # Check if qualified
        if avg_score >= THRESHOLD:
            print(f"[CRITIC] Nerve '{name}' QUALIFIED with score {avg_score:.2f}")
            _publish_progress(name, avg_score, True, known_tools, iteration, MAX_ITERATIONS)

            # Record in cold memory
            mem_manager.cold.record_qualification(
                "nerve", name, True, avg_score, iteration,
                len(iteration_results), passed_count,
                json.dumps(iteration_results),
            )

            # Expand test bank to meet tuning requirements for this model
            # The brain LLM generates batches until we have enough for LoRA training
            min_needed = _tcfg["min_training_examples"]

            current_bank = mem_manager.cold.get_test_bank(name)
            existing_inputs = {t.get("input", "") for t in current_bank}
            expansion_rounds = 0
            _batch = _tcfg["test_cases_per_batch"]
            _gap = min_needed - len(current_bank)
            max_expansion_rounds = max(5, (_gap // max(_batch, 1)) + 5)

            while len(current_bank) < min_needed and time.time() - _start_time < TOTAL_TIMEOUT - 30:
                expansion_rounds += 1
                if expansion_rounds > max_expansion_rounds:
                    break
                try:
                    extra_tests = generate_test_cases(
                        name, description, trigger_task,
                        existing_inputs=existing_inputs, role=_nerve_role,
                    )
                    if not extra_tests:
                        break
                    new_tests = [t for t in extra_tests if t.get("input", "") not in existing_inputs]
                    if not new_tests:
                        break
                    current_bank.extend(new_tests)
                    existing_inputs.update(t.get("input", "") for t in new_tests)
                    mem_manager.cold.set_test_bank(name, current_bank)
                    print(f"[CRITIC] Expanded test bank for '{name}': +{len(new_tests)} (total: {len(current_bank)}, target: {min_needed})")
                except Exception as e:
                    print(f"[CRITIC] Test bank expansion failed: {e}")
                    break

            # Clean up any .bak files from tool fixes that stuck
            _cleanup_bak_files(known_tools)

            return {
                "qualified": True,
                "score": avg_score,
                "iterations": iteration,
                "test_results": iteration_results,
            }

        # Not qualified yet — suggest improvements if more iterations remain
        if iteration < MAX_ITERATIONS:
            failures = [r for r in iteration_results if not r["passed"]]
            meta = mem_manager.cold.get_nerve_metadata(name)

            # --- Phase 1: Fix broken tools FIRST ---
            tool_errors = _extract_tool_errors(failures)
            _applied_tool_fixes = []

            if tool_errors:
                print(f"[CRITIC] Found {len(tool_errors)} tool errors for '{name}': {[e['tool'] for e in tool_errors]}")

                improvements = suggest_improvements(
                    name, description,
                    meta.get("system_prompt", ""),
                    meta.get("examples", []),
                    known_tools, failures,
                    tool_errors=tool_errors,
                    role=_nerve_role,
                )

                # Apply tool fixes immediately
                tool_fixes = improvements.get("tool_fixes", [])
                for fix in tool_fixes[:2]:
                    t_name = fix.get("tool", "")
                    t_code = fix.get("fixed_code", "")
                    if t_name and t_code:
                        if _apply_tool_fix(t_name, t_code, known_tools):
                            _applied_tool_fixes.append(t_name)

                # Re-test after tool fixes
                if _applied_tool_fixes and stored_tests:
                    print(f"[CRITIC] Re-testing '{name}' after tool fixes...")
                    retest_score = 0.0
                    retest_count = 0
                    for tc in stored_tests[:4]:
                        if time.time() - _start_time > TOTAL_TIMEOUT:
                            break
                        ui = tc.get("input", "")
                        if not ui:
                            continue
                        out = run_nerve_with_input(name, ui, mem_manager)
                        ev = evaluate_nerve_output(tc, out)
                        retest_score += ev["score"]
                        retest_count += 1
                    if retest_count > 0:
                        retest_avg = retest_score / retest_count
                        if retest_avg > avg_score:
                            print(f"[CRITIC] Tool fixes improved '{name}': {avg_score:.2f} -> {retest_avg:.2f}")
                            avg_score = retest_avg
                            _cleanup_bak_files(_applied_tool_fixes)
                            if retest_avg >= THRESHOLD:
                                print(f"[CRITIC] Nerve '{name}' QUALIFIED after tool fix: {retest_avg:.2f}")
                                _publish_progress(name, retest_avg, True, known_tools, iteration, MAX_ITERATIONS)
                                mem_manager.cold.record_qualification(
                                    "nerve", name, True, retest_avg, iteration,
                                    len(iteration_results), sum(1 for r in iteration_results if r["passed"]),
                                    json.dumps(iteration_results),
                                )
                                return {"qualified": True, "score": retest_avg, "iterations": iteration, "test_results": iteration_results}
                        else:
                            print(f"[CRITIC] Tool fixes didn't help '{name}': {avg_score:.2f} -> {retest_avg:.2f}, rolling back")
                            for t_name in _applied_tool_fixes:
                                _rollback_tool_fix(t_name)
                            _applied_tool_fixes = []
            else:
                improvements = suggest_improvements(
                    name, description,
                    meta.get("system_prompt", ""),
                    meta.get("examples", []),
                    known_tools, failures,
                    role=_nerve_role,
                )

            # --- Phase 2: Apply prompt improvements ---
            new_system_prompt = improvements.get("system_prompt", "")
            new_examples = improvements.get("examples", [])
            new_desc = improvements.get("description", "")
            effective_desc = new_desc if new_desc and new_desc.strip() else description
            if new_system_prompt or new_examples:
                examples_json = json.dumps(new_examples) if new_examples else "[]"
                existing_role = meta.get("role", "tool")
                mem_manager.cold.register_nerve_rich(name, effective_desc, new_system_prompt, examples_json, role=existing_role)
                print(f"[CRITIC] Updated metadata for nerve '{name}'")
                if new_desc and new_desc.strip() and new_desc != description:
                    description = effective_desc
                    print(f"[CRITIC] Refined description for nerve '{name}': {effective_desc[:80]}")

            # Discover/acquire missing tools — max 1 tool per iteration, with timeout
            _tools_to_try = improvements.get("discover_tools", [])[:1]
            for tool_domain in _tools_to_try:
                if time.time() - _start_time > TOTAL_TIMEOUT - 10:
                    print(f"[CRITIC] Skipping tool acquisition (timeout approaching)")
                    break
                print(f"[CRITIC] Attempting to acquire tool for: {tool_domain}")
                try:
                    import concurrent.futures
                    from arqitect.nerves.nerve_runtime import acquire_tool
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(acquire_tool, tool_domain)
                        acquired = future.result(timeout=30)
                    if acquired:
                        mem_manager.cold.add_nerve_tool(name, acquired)
                        known_tools = mem_manager.cold.get_nerve_tools(name)
                        print(f"[CRITIC] Acquired tool '{acquired}' for nerve '{name}'")
                except concurrent.futures.TimeoutError:
                    print(f"[CRITIC] Tool acquisition timed out for: {tool_domain}")
                except Exception as e:
                    print(f"[CRITIC] Tool acquisition failed: {e}")

            # Index missing domain knowledge
            for domain in improvements.get("index_domains", []):
                print(f"[CRITIC] Indexing domain knowledge: {domain}")
                try:
                    _domain_indexer = os.path.join(os.path.dirname(__file__), "..", "knowledge", "domain_indexer.py")
                    proc = subprocess.Popen(
                        [sys.executable, _domain_indexer, domain],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    proc.wait(timeout=30)
                    print(f"[CRITIC] Domain '{domain}' indexed (exit={proc.returncode})")
                    # Brief wait for facts to be available
                    time.sleep(2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print(f"[CRITIC] Domain indexing timed out for '{domain}'")
                except Exception as e:
                    print(f"[CRITIC] Domain indexing failed: {e}")

    # Exhausted iterations
    avg_score = total_score / len(all_test_results) if all_test_results else 0.0
    passed_count = sum(1 for r in all_test_results if r["passed"])
    qualified = avg_score >= THRESHOLD

    print(f"[CRITIC] Nerve '{name}' final: qualified={qualified}, score={avg_score:.2f}")
    _publish_progress(name, avg_score, qualified, known_tools, MAX_ITERATIONS, MAX_ITERATIONS)

    # Clean up .bak files if qualified, rollback if not
    if qualified:
        _cleanup_bak_files(known_tools)
    else:
        for t in known_tools:
            _rollback_tool_fix(t)  # no-op if no .bak exists

    mem_manager.cold.record_qualification(
        "nerve", name, qualified, avg_score, MAX_ITERATIONS,
        len(all_test_results), passed_count,
        json.dumps(all_test_results),
    )
    return {
        "qualified": qualified,
        "score": avg_score,
        "iterations": MAX_ITERATIONS,
        "test_results": all_test_results,
    }
