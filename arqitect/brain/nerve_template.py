"""Nerve template — the Python source code template used to synthesize new nerves."""

NERVE_TEMPLATE = '''"""Nerve: {{NERVE_NAME}} — {{DESCRIPTION}}"""
import json
import sys
import os

from arqitect.nerves.nerve_runtime import (
    think, think_for_role, mcp_call, mcp_list_tools, mcp_tool_exists,
    fabricate_tool, acquire_tool, respond, get_args,
    get_session_context, get_episode_hints, get_known_tools,
    get_user_facts, get_user_profile, get_nerve_meta, get_messages,
    can_answer_from_facts, substitute_fact_values,
    publish_tool_learned,
)
from arqitect.senses.sense_runtime import (
    see, see_screenshot, hear, speak, touch,
    check_awareness, express, call_sense,
)
from arqitect.matching import match_tools as _match_tools

NERVE_NAME = "{{NERVE_NAME}}"
NERVE_ROLE = "{{NERVE_ROLE}}"
DESCRIPTION = """{{DESCRIPTION}}"""

# Available senses this nerve can call directly
AVAILABLE_SENSES = {
    "see": "Analyze an image file (pass file path)",
    "see_screenshot": "Capture and analyze the screen",
    "hear": "Speech-to-text from audio file",
    "speak": "Text-to-speech (play audio or generate file)",
    "touch": "File/OS operations (read, write, list, delete, exec, sysinfo)",
    "check_awareness": "Permission check before destructive operations",
    "express": "Format a message with tone and style (formal, casual, emoji, gif)",
}


def get_tool_list():
    """Get this nerve's known tools (from env var) with descriptions from MCP.

    Nerves only see tools they have previously used or acquired.
    This prevents small models from misusing unrelated tools.
    Only includes tools that actually exist in MCP — phantom tools
    (registered but never fabricated/installed) are excluded to prevent
    the planner from hallucinating answers instead of acquiring real tools.
    """
    all_tools = mcp_list_tools()
    if not isinstance(all_tools, dict):
        all_tools = {}
    known = get_known_tools()
    tool_info = {}
    for name in known:
        if name in all_tools:
            info = all_tools[name]
            desc = info.get("description", "") if isinstance(info, dict) else ""
            params = info.get("params", []) if isinstance(info, dict) else []
            tool_info[name] = {"description": desc, "params": params}
    return tool_info


def get_tool_names():
    """Get just tool names (for matching and existence checks)."""
    return list(get_tool_list().keys())


_SENSE_ACTIONS_BLOCK = """
You also have access to SENSES — built-in capabilities you can invoke directly:
- "use_sense": invoke a sense when the task needs perception or physical interaction.
  {"action":"use_sense","sense":"SENSE_NAME","args":{"PARAM":"VALUE"}}

Available senses:
  see(image_path) — analyze an image file
  see_screenshot() — capture and analyze the screen
  hear(audio_path) — speech-to-text from audio file
  speak(text, voice) — text-to-speech playback
  touch(command, path, ...) — file/OS ops: read, write, list, delete, exec, sysinfo
  check_awareness(action, context) — permission check before destructive ops
  express(message, tone, format) — reformat a message with tone/style (formal, casual, emoji, gif)
"""


def get_effective_role(meta):
    """Get the nerve role: prefer runtime metadata from cold memory over baked-in constant.

    This ensures role corrections in SQLite take effect without re-synthesizing nerves.
    """
    meta_role = meta.get("role", "")
    return meta_role if meta_role else NERVE_ROLE


def build_planner_prompt(meta):
    """Build PLANNER_PROMPT with nerve-specific or community adapter prompt.

    The nerve's system_prompt evolves from the community adapter prompt through
    dream cycles. If the nerve has its own tuned prompt, use it. Otherwise
    fall back to the community adapter prompt as the starting skeleton.
    """
    effective_role = get_effective_role(meta)

    from arqitect.brain.adapters import resolve_prompt
    adapter = resolve_prompt(effective_role)

    prompt = ""
    sp = meta.get("system_prompt", "")

    if sp:
        # Nerve has a tuned prompt — use it (evolved from community base)
        prompt += sp
        # Use nerve-specific examples from cold memory
        examples = meta.get("examples", [])
    else:
        # No nerve prompt yet — fall back to community adapter prompt
        if adapter and adapter.get("system_prompt"):
            prompt += adapter["system_prompt"]
        examples = adapter.get("few_shot_examples", []) if adapter else []

    if examples:
        prompt += "\\nExamples of correct behavior:\\n"
        for ex in examples:
            if isinstance(ex, dict) and "input" in ex and "output" in ex:
                prompt += f'  Input: "{ex["input"]}" -> Output: {ex["output"]}\\n'

    # Append sense actions block so nerves know how to use senses
    prompt += _SENSE_ACTIONS_BLOCK

    return prompt


def parse_json(raw):
    """Extract first JSON object from a string."""
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def match_tool_name(model_name, available_tools):
    """Fuzzy match a tool name from the model to actual available tools.
    Small models often hallucinate similar but wrong tool names."""
    if model_name in available_tools:
        return model_name
    model_lower = model_name.lower()
    for tool in available_tools:
        if model_lower in tool.lower() or tool.lower() in model_lower:
            return tool
    model_words = set(model_lower.replace("_", " ").split())
    best, best_score = None, 0
    for tool in available_tools:
        tool_words = set(tool.lower().replace("_", " ").split())
        overlap = len(model_words & tool_words)
        if overlap > best_score:
            best, best_score = tool, overlap
    return best if best_score > 0 else model_name


def main():
    import sys as _sys
    _err = lambda msg: print(msg, file=_sys.stderr)

    user_input = get_args() or "(no input provided)"

    # Read session context from env
    session = get_session_context()
    episodes = get_episode_hints()
    facts = get_user_facts()
    user_profile = get_user_profile()
    nerve_meta = get_nerve_meta()
    messages = get_messages()

    # Phase 2: Try answering from stored facts before calling the LLM
    # Skip for TDD nerves (SYNAPSE_SKIP_FACTS=1) — they must always run their LLM
    fact_answer = None if os.environ.get("SYNAPSE_SKIP_FACTS") else can_answer_from_facts(user_input, facts, session)
    if fact_answer:
        _err(f"[NERVE:{NERVE_NAME}] Answered from facts: {fact_answer}")
        respond({"nerve": NERVE_NAME, "input": user_input, "response": fact_answer})
        return

    tool_info = get_tool_list()
    _err(f"[NERVE:{NERVE_NAME}] Tools loaded: {list(tool_info.keys())}")

    # Smart proactive acquire: if nerve has no relevant tools,
    # check if any global MCP tool matches and auto-acquire it.
    if not tool_info:
        all_tools = mcp_list_tools()
        from arqitect.matching import best_match_tool as _best_match
        best = _best_match(user_input, all_tools, threshold=1.5)
        if best:
            _err(f"[NERVE:{NERVE_NAME}] Auto-acquiring matching tool: {best}")
            publish_tool_learned(NERVE_NAME, best)
            # Build tool_info directly from live MCP data (env var is stale)
            info = all_tools.get(best, {})
            desc = info.get("description", "") if isinstance(info, dict) else ""
            params = info.get("params", []) if isinstance(info, dict) else []
            tool_info[best] = {"description": desc, "params": params}

    # Pre-filter tools by relevance to user input
    ranked = _match_tools(user_input, tool_info, threshold=0.5)
    if ranked:
        relevant = {name for name, _ in ranked[:5]}
        tool_info = {k: v for k, v in tool_info.items() if k in relevant}

    # Build context with live system data + session + episode hints
    import datetime as _dt
    _now = _dt.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
    context = f"LIVE SYSTEM DATA (always accurate, overrides your training data):\\n"
    context += f"  Current date/time: {_now}\\n"
    if session.get("city"):
        context += f"  User location: {session.get('city', '?')}, {session.get('country', '?')}\\n"
        context += f"  User timezone: {session.get('timezone', '?')}\\n"
    if user_profile:
        profile_parts = [f"{k}: {v}" for k, v in user_profile.items()]
        context += f"  User: {', '.join(profile_parts)}\\n"
    context += f"\\n"

    # Conversation history (sliding window)
    if messages:
        context += f"Conversation history:\\n"
        for msg in messages:
            context += f"  {msg.get('role', '?')}: {msg.get('content', '')}\\n"
        context += f"\\n"

    # Episode hints
    if episodes:
        context += f"Recent relevant episodes:\\n"
        for ep in episodes[:3]:
            status = "SUCCESS" if ep.get("success") else "FAILED"
            context += f"  - {ep.get('task', '?')} -> {ep.get('nerve', '?')}:{ep.get('tool', '?')} [{status}]\\n"
        context += f"\\n"

    context += f"Available tools:\\n"
    for tname, tinfo in tool_info.items():
        context += f"  - {tname}: {tinfo.get('description', 'no description')} (params: {tinfo.get('params', [])})\\n"
    context += f"\\nUser input: {user_input}"
    PLANNER_PROMPT = build_planner_prompt(nerve_meta)
    _effective_role = get_effective_role(nerve_meta)
    _err(f"[NERVE:{NERVE_NAME}] Planner context tools: {list(tool_info.keys())}")
    _json_mode = True
    try:
        from arqitect.brain.adapters import get_json_mode as _get_jm
        _json_mode = _get_jm(_effective_role)
    except Exception:
        pass
    plan_raw = think_for_role(_effective_role, PLANNER_PROMPT, context, json_mode=_json_mode)
    _err(f"[NERVE:{NERVE_NAME}] Planner raw output: {plan_raw[:300]}")
    plan = parse_json(plan_raw)
    _err(f"[NERVE:{NERVE_NAME}] Parsed plan: {plan}")

    if not plan:
        # Raw LLM output wasn't valid JSON — use it as a direct response
        # but strip any instruction/prompt leakage
        _clean = plan_raw.strip()
        if _clean.startswith(("```", "{")):
            _clean = ""  # garbage, don't use
        respond({"nerve": NERVE_NAME, "input": user_input, "response": _clean or "(no response)"})
        return

    # Issue 4 fix: Detect template placeholder strings echoed by small models
    _plan_str = json.dumps(plan).upper()
    _placeholders = ["TOOL_NAME_HERE", "TOOLNAME", "PARAM", "VALUE", "TOOL_NAME"]
    if any(ph in _plan_str for ph in _placeholders) and plan.get("action") == "call":
        _err(f"[NERVE:{NERVE_NAME}] Plan has template placeholders, falling back to matching")
        # Pick best tool from tool_info via matching
        if tool_info:
            _best = _match_tools(user_input, tool_info, threshold=0.5)
            if _best:
                plan = {"action": "call", "tool": _best[0][0], "args": {"query": user_input}}
            else:
                plan = {"action": "acquire", "need": user_input}
        else:
            plan = {"action": "acquire", "need": user_input}

    action = plan.get("action", "")

    # Normalize: small models sometimes put tool name in "action" instead of using "call"
    # e.g. {"action":"weather_tool","args":{...}} instead of {"action":"call","tool":"weather_tool","args":{...}}
    if action and action not in ("call", "answer", "use_sense", "acquire", "fabricate", "needs"):
        _all_tool_names = set(tool_info.keys()) | set(get_tool_names())
        if action in _all_tool_names or match_tool_name(action, list(_all_tool_names)) in _all_tool_names:
            _matched_action = match_tool_name(action, list(_all_tool_names))
            _err(f"[NERVE:{NERVE_NAME}] Normalized action '{action}' -> call:{_matched_action}")
            plan["tool"] = _matched_action
            plan["action"] = "call"
            action = "call"

    if action == "answer":
        # Guard: tool-dependent nerves must not hallucinate answers.
        # If this nerve has known tools but the LLM chose "answer" instead of
        # calling one, the LLM is likely hallucinating (e.g. making up weather
        # data).  Redirect to "acquire" so the nerve gets a real tool.
        _known = get_known_tools()
        if _known and not tool_info:
            _err(f"[NERVE:{NERVE_NAME}] LLM chose 'answer' but nerve has tool requirements {_known} — redirecting to acquire")
            plan = {"action": "acquire", "need": f"{DESCRIPTION}: {user_input}"}
            action = "acquire"
        else:
            _answer = plan.get("response", "")
            # Sanitize: if "response" contains a JSON plan, it's prompt leakage
            if isinstance(_answer, str) and _answer.strip().startswith(("{", "[")):
                try:
                    _leaked = json.loads(_answer)
                    if isinstance(_leaked, dict) and "action" in _leaked:
                        _answer = ""  # it's a leaked plan, not a real answer
                except (json.JSONDecodeError, TypeError):
                    pass
            if not _answer:
                _answer = plan_raw if not plan_raw.strip().startswith(("{", "[", "```")) else "(no response)"
            respond({"nerve": NERVE_NAME, "input": user_input, "response": _answer})
            return

    if action == "needs":
        respond({
            "nerve": NERVE_NAME,
            "input": user_input,
            "status": "needs_data",
            "needs": plan.get("missing", "unknown"),
            "reason": plan.get("reason", ""),
        })
        return

    if action == "use_sense":
        sense_name = plan.get("sense", "")
        sense_args = plan.get("args", {})
        if isinstance(sense_args, str):
            sense_args = {"input": sense_args}
        _err(f"[NERVE:{NERVE_NAME}] Invoking sense: {sense_name} with {sense_args}")

        # Map sense name to the imported function
        _SENSE_MAP = {
            "see": lambda a: see(a.get("image_path", "")),
            "see_screenshot": lambda a: see_screenshot(),
            "hear": lambda a: hear(a.get("audio_path", "")),
            "speak": lambda a: speak(a.get("text", ""), a.get("voice", "default")),
            "touch": lambda a: touch(a.get("command", "read"), a.get("path", ""), **{k: v for k, v in a.items() if k not in ("command", "path")}),
            "check_awareness": lambda a: check_awareness(a.get("action", ""), a.get("context", "")),
            "express": lambda a: express(a.get("message", ""), a.get("tone", "neutral"), a.get("format", "text")),
        }

        sense_fn = _SENSE_MAP.get(sense_name)
        if not sense_fn:
            # Fallback: try call_sense with raw name
            sense_result = call_sense(sense_name, sense_args)
        else:
            sense_result = sense_fn(sense_args)

        _err(f"[NERVE:{NERVE_NAME}] Sense result: {str(sense_result)[:200]}")

        # Interpret sense result with the nerve's model
        sense_output = json.dumps(sense_result) if isinstance(sense_result, dict) else str(sense_result)
        analysis = think_for_role(
            _effective_role,
            f"You are a {DESCRIPTION} nerve. Answer the user directly using the data below. "
            f"Do NOT output JSON or describe actions. Start with the actual answer. No preamble.",
            f"User asked: {user_input}\\n\\nSense ({sense_name}) returned:\\n{sense_output}"
        )

        respond({"nerve": NERVE_NAME, "input": user_input, "sense": sense_name,
                 "sense_result": sense_result, "response": analysis})
        return

    if action in ("acquire", "fabricate"):
        # During qualification (SYNAPSE_NO_ACQUIRE=1), skip expensive tool
        # acquisition (HTTP searches, fabrication) and answer directly instead
        if os.environ.get("SYNAPSE_NO_ACQUIRE") == "1":
            _err(f"[NERVE:{NERVE_NAME}] Skipping acquisition (qualification mode), answering directly")
            fallback = think_for_role(
                _effective_role,
                f"You are a {DESCRIPTION} nerve. Answer the user's question directly using your knowledge. "
                f"Do NOT output JSON. Do NOT describe actions. Just give the answer in plain text.",
                f"User asked: {user_input}"
            )
            _fb = fallback.strip()
            if _fb.startswith(("{", "[", "```")):
                _fb = ""
            respond({"nerve": NERVE_NAME, "input": user_input, "response": _fb or "(could not generate response)"})
            return

        need = plan.get("need", user_input) if action == "acquire" else (plan.get("description", "") or f"{DESCRIPTION}: {user_input}")
        acquired = acquire_tool(need)
        publish_tool_learned(NERVE_NAME, acquired)
        # Refresh tool list with acquired tool
        all_tools = mcp_list_tools()
        if acquired in all_tools:
            info = all_tools[acquired]
            desc = info.get("description", "") if isinstance(info, dict) else ""
            params = info.get("params", []) if isinstance(info, dict) else []
            tool_info[acquired] = {"description": desc, "params": params}
        context = f"Available tools: {tool_info}\\nUser input: {user_input}\\nNote: tool \\'{acquired}\\' is now available. Use it."
        plan_raw = think_for_role(_effective_role, PLANNER_PROMPT, context)
        plan = parse_json(plan_raw)
        if not plan or plan.get("action") != "call":
            respond({"nerve": NERVE_NAME, "input": user_input, "acquired": acquired, "response": plan_raw})
            return

    if plan.get("action") == "call":
        tool_name = plan["tool"]
        tool_args = plan.get("args", {})
        if isinstance(tool_args, str):
            tool_args = {"query": tool_args}

        # Detect placeholder arg keys/values echoed by small models
        _PH = {"PARAM", "VALUE", "TOOLNAME", "TOOL_NAME", "TOOL_NAME_HERE",
               "PARAM1", "PARAM2", "VALUE1", "VALUE2", "SENSE_NAME"}
        if isinstance(tool_args, dict):
            _has_ph = any(k.upper() in _PH or (isinstance(v, str) and v.upper() in _PH)
                         for k, v in tool_args.items())
            if _has_ph:
                _err(f"[NERVE:{NERVE_NAME}] Placeholder args detected: {tool_args}, replacing with user input")
                tool_args = {"query": user_input}

        available = get_tool_names()
        matched = match_tool_name(tool_name, available)
        if matched != tool_name:
            _err(f"[NERVE:{NERVE_NAME}] Mapped \\'{tool_name}\\' -> \\'{matched}\\'")
            tool_name = matched

        # Remap arg keys to match the tool's expected params
        if tool_name in tool_info and isinstance(tool_args, dict):
            expected = tool_info[tool_name].get("params", [])
            if expected and set(tool_args.keys()) != set(expected):
                if len(expected) == 1:
                    # Single-param tool: pass first value (or user_input) as that param
                    _val = next(iter(tool_args.values()), user_input) if tool_args else user_input
                    tool_args = {expected[0]: _val}
                    _err(f"[NERVE:{NERVE_NAME}] Remapped args to single param: {expected[0]}")

        if not mcp_tool_exists(tool_name):
            if os.environ.get("SYNAPSE_NO_ACQUIRE") == "1":
                _err(f"[NERVE:{NERVE_NAME}] Tool \\'{tool_name}\\' not found (qualification mode, skipping acquire)")
                respond({"nerve": NERVE_NAME, "input": user_input, "tool": tool_name,
                         "error": f"Tool \\'{tool_name}\\' not available",
                         "response": f"Tool \\'{tool_name}\\' is not available yet."})
                return
            _err(f"[NERVE:{NERVE_NAME}] Tool \\'{tool_name}\\' not found. Acquiring...")
            acquired = acquire_tool(f"{DESCRIPTION}: {user_input}")
            publish_tool_learned(NERVE_NAME, acquired)
            if mcp_tool_exists(acquired):
                tool_name = acquired
                _err(f"[NERVE:{NERVE_NAME}] Using acquired tool: {tool_name}")

        publish_tool_learned(NERVE_NAME, tool_name)
        # Phase 2: Fix garbled values from small model context
        tool_args = substitute_fact_values(tool_args, facts, session)
        tool_result = mcp_call(tool_name, tool_args) or ""

        if tool_result.startswith("MCP call error") or tool_result.startswith("Tool error") or tool_result.startswith("Error:") or tool_result.startswith("Error "):
            _err(f"[NERVE:{NERVE_NAME}] Tool call failed ({tool_name}): {tool_result}")
            respond({"nerve": NERVE_NAME, "input": user_input, "tool": tool_name, "error": tool_result,
                     "response": f"Tool \\'{tool_name}\\' returned an error. Please try again."})
            return

        try:
            result_data = json.loads(tool_result)
            if isinstance(result_data, dict) and "error" in result_data:
                _err(f"[NERVE:{NERVE_NAME}] Tool returned error: {result_data['error']}")
                respond({
                    "nerve": NERVE_NAME,
                    "input": user_input,
                    "tool": tool_name,
                    "status": "needs_data",
                    "needs": result_data["error"],
                    "reason": f"Tool \\'{tool_name}\\' could not complete without proper input",
                })
                return
            # Media result (image/audio/video) — return directly without LLM interpretation
            _MEDIA_KEYS = {"image_b64", "image_path", "audio_b64", "audio_path", "video_b64", "video_path"}
            if isinstance(result_data, dict) and _MEDIA_KEYS & set(result_data.keys()):
                _err(f"[NERVE:{NERVE_NAME}] Media result detected — returning directly")
                result_data["nerve"] = NERVE_NAME
                result_data["input"] = user_input
                result_data["tool"] = tool_name
                if "response" not in result_data:
                    result_data["response"] = result_data.get("response", "Here is the generated content.")
                respond(result_data)
                return
        except (json.JSONDecodeError, TypeError):
            pass

        analysis = think_for_role(
            _effective_role,
            f"You are a {DESCRIPTION} nerve. Answer the user directly using the tool data below. "
            f"Do NOT output JSON or describe actions. Start with the actual answer. No preamble.",
            f"User asked: {user_input}\\n\\nTool ({tool_name}) returned:\\n{tool_result}"
        )

        respond({"nerve": NERVE_NAME, "input": user_input, "tool": tool_name, "response": analysis})
        return

    # Unrecognized action — the LLM returned a plan it can't execute.
    # Re-prompt for a direct answer instead of echoing the plan.
    _err(f"[NERVE:{NERVE_NAME}] Unrecognized action '{action}'. Re-prompting for direct answer.")
    fallback = think_for_role(
        _effective_role,
        f"You are a {DESCRIPTION} nerve. Answer the user's question directly. "
        f"Do NOT output JSON. Do NOT describe actions. Just give the answer in plain text.",
        f"User asked: {user_input}"
    )
    # Strip any JSON that leaked into the response
    _fb = fallback.strip()
    if _fb.startswith(("{", "[", "```")):
        _fb = ""
    respond({"nerve": NERVE_NAME, "input": user_input, "response": _fb or "(could not generate response)"})


if __name__ == "__main__":
    main()
'''
