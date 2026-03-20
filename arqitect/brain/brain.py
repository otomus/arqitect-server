"""
The Progenitor (Brain) — high-level reasoning and nerve synthesis.
Uses the configured brain model via the inference router for reasoning and code generation.
Communicates with nerves over Redis Pub/Sub.
Three-tier memory: Hot (Redis) + Warm (SQLite episodes) + Cold (SQLite knowledge).
"""

import json
import logging
import os
import re
import subprocess
import sys
import time

# Force unbuffered output for daemon mode
sys.stdout.reconfigure(line_buffering=True)

# Configure logging so logger.info() calls in all modules are visible.
# The brain runs as a daemon with stdout/stderr redirected to brain.log.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)

import base64
import redis
import requests

from arqitect.brain.config import (
    BRAIN_MODEL, CODE_MODEL, COMMUNICATION_MODEL, NERVES_DIR, SANDBOX_DIR,
    DOMAIN_INDEXER_PATH, CORE_SENSES,
    r, mem,
)
from arqitect.types import Action, Channel, IntentType, NerveStatus, RedisKey
from arqitect.brain.helpers import (
    llm_generate, extract_json,
    _is_nerve_error, _graceful_failure_message, _substitute_fact_values_brain,
)
from arqitect.brain.events import publish_event, publish_memory_state, publish_nerve_status, publish_response, set_task_origin, get_task_origin
from arqitect.brain.catalog import list_nerves, discover_nerves
from arqitect.brain.prompt import get_system_prompt
from arqitect.brain.bootstrap import (
    bootstrap_session, bootstrap_senses, bootstrap_user_session,
    calibrate_sense, calibrate_all_senses, _store_calibration_in_memory,
    _prompt_calibration_config,
)
from arqitect.brain.invoke import invoke_nerve, _enrich_nerve_result_with_image, _translate_sense_args
from arqitect.brain.synthesis import synthesize_nerve, prune_degenerate_nerves
from arqitect.brain.consolidate import get_consolidator, consolidate_nerves
from arqitect.brain.circuit_breaker import is_available as _cb_is_available, record_success as _cb_success, record_failure as _cb_failure
from arqitect.brain.tdd import detect_project_path, build_tdd_chain, is_coding_task, compress_chain_output, stack_fingerprint
from arqitect.brain.checklist import TaskChecklist
from arqitect.brain.intent import classify_intent
from arqitect.brain.safety import check_input as _safety_check_input
from arqitect.brain.planner import plan_task
from arqitect.brain.community import (
    sync_manifest as _sync_community_manifest,
    seed_tools as _seed_community_tools,
    seed_nerves as _seed_community_nerves,
)
from arqitect.brain.adapters import (
    sync_all_adapters as _sync_community_adapters,
    get_temperature, get_max_tokens, get_conversation_window, get_message_truncation,
    resolve_prompt as _resolve_adapter,
)
from arqitect.brain.dispatch import DispatchContext, dispatch_action


# Maximum task length (chars) fed into the routing prompt
_MAX_TASK_LENGTH = 4000


def _reverse_geocode(lat: float, lon: float) -> str:
    """Reverse-geocode lat/lon to city name via Nominatim (best-effort)."""
    try:
        import requests
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 10},
            headers={"User-Agent": "Arqitect/1.0"},
            timeout=5,
        )
        data = resp.json()
        addr = data.get("address", {})
        return addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county") or ""
    except Exception:
        return ""



def _personality_media_enhancement(task: str, msg: str, nerve_result: dict) -> dict:
    """Let personality decide whether to enhance a response with media (GIF, emoji).

    Checks personality trait weights from cold memory. High wit/swagger = more
    likely to add a GIF or emoji. Structured data responses are never enhanced.
    Returns the (possibly enriched) nerve_result dict.
    """
    if not msg or not isinstance(nerve_result, dict):
        return nerve_result
    # Already has media — don't double up
    if nerve_result.get("gif_url") or nerve_result.get("image_b64") or nerve_result.get("image_path"):
        return nerve_result
    # Structured data — personality stays minimal
    if msg.strip().startswith(("{", "[", "```", "    ")):
        return nerve_result

    try:
        import random
        weights = {"wit": 0.5, "swagger": 0.3}  # defaults from seed
        raw = mem.cold.get_fact("personality", "trait_weights")
        if raw:
            weights.update(json.loads(raw))

        wit = weights.get("wit", 0.5)
        swagger = weights.get("swagger", 0.5)
        # Probability of adding media — kept low to avoid abuse
        # wit=0.7, swagger=0.6 → ~4% chance of GIF, ~6% chance of emoji
        gif_chance = max(0, (wit + swagger - 1.0) * 0.12)   # ~0.04 at defaults
        emoji_chance = max(0, (wit - 0.3) * 0.15)            # ~0.06 at defaults

        roll = random.random()

        if roll < gif_chance:
            # Try to invoke a GIF nerve
            catalog = discover_nerves()
            gif_nerves = [n for n, d in catalog.items() if "gif" in d.lower() and n not in CORE_SENSES]
            if gif_nerves:
                gif_query = task[:50]  # use task as search context
                gif_output = invoke_nerve(gif_nerves[0], json.dumps({"query": gif_query}))
                try:
                    gif_result = json.loads(gif_output)
                    url = gif_result.get("gif_url", "")
                    if url:
                        nerve_result["gif_url"] = url
                        print(f"[PERSONALITY] Enhanced response with GIF (nerve={gif_nerves[0]})")
                except (json.JSONDecodeError, TypeError):
                    pass
        elif roll < gif_chance + emoji_chance:
            # Add emoji enhancement via communication sense
            emoji_output = invoke_nerve("communication", json.dumps({
                "message": msg, "format": "emoji", "tone": "casual"
            }))
            try:
                emoji_result = json.loads(emoji_output)
                enhanced = emoji_result.get("response", "")
                if enhanced and enhanced != msg:
                    nerve_result["_personality_rewrite"] = enhanced
                    print(f"[PERSONALITY] Enhanced response with emojis")
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception as e:
        print(f"[PERSONALITY] Media enhancement failed: {e}")

    return nerve_result


def _trigger_domain_index_background(task: str):
    """Detect domain in the task and spawn background indexer if not yet indexed.

    Fire-and-forget: user never waits for indexing. Facts become available
    for the next question in that domain.
    """
    try:
        from arqitect.knowledge.domain_indexer import detect_domain, is_indexed
        domain = detect_domain(task)
        if domain and not is_indexed(domain):
            print(f"[BRAIN] Spawning background domain indexer for '{domain}'")
            subprocess.Popen(
                [sys.executable, DOMAIN_INDEXER_PATH, domain],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception as e:
        print(f"[BRAIN] Domain index trigger failed: {e}")


def _tdd_scout(project_path: str, project_facts: dict) -> str:
    """Scout step: deterministic file listing + read key files. No LLM needed."""
    results = []

    # List top-level structure
    list_args = json.dumps({"command": "list", "path": project_path})
    list_output = invoke_nerve("touch", list_args)
    try:
        parsed = json.loads(list_output) if isinstance(list_output, str) else list_output
        if isinstance(parsed, dict):
            list_output = parsed.get("response", list_output)
    except (json.JSONDecodeError, TypeError):
        pass
    results.append(f"Project structure:\n{str(list_output)[:500]}")

    # Read an existing test file if we can find one
    lang = project_facts.get("language", "")
    test_dirs = ["tests", "test", "src/__tests__", "__tests__", "spec"]
    for td in test_dirs:
        td_path = os.path.join(project_path, td)
        if os.path.isdir(td_path):
            # Find first test file
            for f in os.listdir(td_path):
                is_test = (f.startswith("test_") or f.endswith((".test.ts", ".test.tsx", ".test.js", ".spec.ts", ".spec.tsx")))
                if is_test:
                    sample_path = os.path.join(td_path, f)
                    read_args = json.dumps({"command": "read", "path": sample_path})
                    read_output = invoke_nerve("touch", read_args)
                    try:
                        parsed = json.loads(read_output) if isinstance(read_output, str) else read_output
                        if isinstance(parsed, dict):
                            read_output = parsed.get("response", read_output)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    results.append(f"Example test ({f}):\n{str(read_output)[:600]}")
                    break
            break

    # Read an existing source file for conventions
    src_dirs = ["src", "lib", "app", "components", "src/components"]
    for sd in src_dirs:
        sd_path = os.path.join(project_path, sd)
        if os.path.isdir(sd_path):
            for f in sorted(os.listdir(sd_path)):
                ext_match = (lang == "python" and f.endswith(".py")) or \
                            (lang in ("typescript", "javascript") and f.endswith((".ts", ".tsx", ".js", ".jsx")))
                if ext_match and not f.startswith((".", "_")):
                    sample_path = os.path.join(sd_path, f)
                    if os.path.isfile(sample_path):
                        read_args = json.dumps({"command": "read", "path": sample_path})
                        read_output = invoke_nerve("touch", read_args)
                        try:
                            parsed = json.loads(read_output) if isinstance(read_output, str) else read_output
                            if isinstance(parsed, dict):
                                read_output = parsed.get("response", read_output)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        results.append(f"Example source ({f}):\n{str(read_output)[:600]}")
                        break
            break

    return "\n\n".join(results)


def _tdd_llm_step(step_type: str, task: str, project_facts: dict,
                   chain_context: list[str], extra_context: str = "",
                   target_file: str = "", impl_file: str = "") -> str:
    """Run a TDD code generation step using the brain's in-process coder model."""
    from arqitect.brain.helpers import llm_generate, strip_markdown_fences
    from arqitect.knowledge.project_profiler import format_profile_for_prompt

    project_context = format_profile_for_prompt(project_facts)
    lang = project_facts.get("language", "")
    framework = project_facts.get("framework", "")
    test_fw = project_facts.get("test_framework", "")
    stack_label = "+".join(filter(None, [lang, framework]))
    project_path = project_facts.get("path", ".")

    # Build context from prior steps (compact)
    prior = ""
    if chain_context:
        prior = "\n".join(f"[Step {j+1}]: {ctx[:300]}" for j, ctx in enumerate(chain_context))
        if len(prior) > 800:
            prior = prior[:800] + "..."

    # Compute relative import path for the implementation module
    impl_rel = ""
    impl_module = ""
    if impl_file:
        impl_rel = os.path.relpath(impl_file, project_path)
        # Convert path to module: src/calculator.py -> src.calculator
        impl_module = impl_rel.replace(os.sep, ".").removesuffix(".py").removesuffix(".tsx").removesuffix(".ts")

    if step_type == "test_writer":
        system = (
            f"You are a {test_fw or 'unit'} test writer for {stack_label} projects. "
            f"Output ONLY valid {lang} test code. No explanations, no markdown fences."
        )
        import_hint = ""
        if lang == "python" and impl_module:
            import_hint = f"Import the module using: from {impl_module} import ...\n"
        elif lang in ("typescript", "javascript") and impl_rel:
            # Relative import from test file to impl file
            test_rel = os.path.relpath(target_file, project_path) if target_file else ""
            if test_rel and impl_rel:
                test_dir = os.path.dirname(test_rel)
                rel_import = os.path.relpath(impl_rel, test_dir).removesuffix(".tsx").removesuffix(".ts")
                import_hint = f"Import using: import {{ ... }} from '{rel_import}'\n"
        prompt = (
            f"{project_context}\n\n"
            f"Task: {task}\n"
            f"Implementation will be at: {impl_rel}\n"
            f"{import_hint}\n"
            f"Write a failing {test_fw or 'unit'} test for this task.\n"
        )
        if prior:
            prompt += f"\nProject context from scout:\n{prior}\n"

    elif step_type == "implementer":
        system = (
            f"You are a {stack_label} developer. "
            f"Output ONLY valid {lang} implementation code. No explanations, no markdown fences."
        )
        prompt = (
            f"{project_context}\n\n"
            f"Task: {task}\n"
            f"File path: {impl_rel}\n\n"
            f"Write the implementation code that makes the tests pass.\n"
        )
        if prior:
            prompt += f"\nPrior steps:\n{prior}\n"
        if extra_context:
            prompt += f"\n{extra_context}\n"

    else:
        system = f"You are a {stack_label} developer."
        prompt = f"{project_context}\n\nTask: {task}\n"
        if prior:
            prompt += f"\nContext:\n{prior}\n"

    output = llm_generate(CODE_MODEL, prompt, system=system)
    return strip_markdown_fences(output)


def _handle_tdd_chain(task: str, decision: dict, checklist: TaskChecklist) -> str:
    """Execute a TDD chain using in-process LLM — no nerve subprocesses needed.

    Scout step: deterministic file reading (no LLM).
    Code steps: direct llm_generate() calls using already-loaded coder model.
    Exec steps: direct touch sense invocation.
    """
    steps = decision.get("steps", [])
    project_path = decision.get("project_path", ".")
    project_facts = decision.get("project_facts", {})

    print(f"[BRAIN] TDD chain started: {len(steps)} steps for: {task}")
    publish_event(Channel.BRAIN_THOUGHT, {"stage": "tdd_chain", "steps": len(steps), "goal": task})
    publish_event(Channel.BRAIN_CHECKLIST, checklist.to_dict())

    chain_context = []
    test_file_path = ""
    retries = 0
    max_retries = 2

    # Pre-extract impl file path so test_writer knows the import path
    impl_file_path = ""
    for s in steps:
        if s.get("step_type") == "implementer" and s.get("target_file"):
            impl_file_path = s["target_file"]
            break

    i = 0
    while i < len(steps):
        chain_step = steps[i]
        nerve_name = chain_step.get("nerve", "")
        step_type = chain_step.get("step_type", "")
        target_file = chain_step.get("target_file", "")

        checklist.activate(i)
        publish_event(Channel.BRAIN_CHECKLIST, checklist.to_dict())

        # ── _touch_exec steps: direct command execution, no LLM ──
        if nerve_name == "_touch_exec":
            cmd = chain_step.get("cmd", "")
            if test_file_path and "{test_file}" in cmd:
                cmd = cmd.replace("{test_file}", test_file_path)
            elif "{test_file}" in cmd:
                checklist.check(i, False, "No test file path discovered")
                publish_event(Channel.BRAIN_CHECKLIST, checklist.to_dict())
                break

            print(f"[BRAIN] TDD step {i+1}/{len(steps)}: exec: {cmd[:100]}")
            publish_event(Channel.BRAIN_ACTION, {"nerve": "touch", "args": cmd, "chain_step": i+1})

            exec_args = json.dumps({"command": "exec", "cmd": cmd})
            output = invoke_nerve("touch", exec_args)

            # Touch exec returns {"stdout": "...", "stderr": "...", "returncode": N}
            step_output = output
            try:
                result = json.loads(output)
                if isinstance(result, dict):
                    step_output = result.get("stdout", "") + result.get("stderr", "")
            except (json.JSONDecodeError, TypeError):
                extracted = extract_json(output)
                if extracted and isinstance(extracted, dict):
                    step_output = extracted.get("stdout", "") + extracted.get("stderr", "")

            compressed = compress_chain_output(str(step_output), step_type)

            if step_type in ("test_fail", "verify"):
                passed, summary = TaskChecklist.verify_test_output(str(step_output))

                if step_type == "test_fail":
                    checklist.check(i, True, summary if not passed else "Test already passes (unexpected)")
                elif step_type == "verify":
                    if passed:
                        checklist.check(i, True, summary)
                    else:
                        checklist.check(i, False, summary)
                        if retries < max_retries:
                            retries += 1
                            impl_step_idx = i - 1
                            if impl_step_idx >= 0 and steps[impl_step_idx].get("step_type") == "implementer":
                                print(f"[BRAIN] TDD verify failed, retrying implementation (attempt {retries}/{max_retries})")
                                steps[impl_step_idx]["_retry_context"] = (
                                    f"PREVIOUS ATTEMPT FAILED. Test error:\n{compressed}\n"
                                    f"Previous code (fix it):\n{str(chain_context[-1] if chain_context else '')[:400]}"
                                )
                                i = impl_step_idx
                                publish_event(Channel.BRAIN_CHECKLIST, checklist.to_dict())
                                continue
            else:
                checklist.check(i, True, compressed[:200])

            chain_context.append(compressed)
            publish_event(Channel.NERVE_RESULT, {"nerve": "touch_exec", "output": compressed, "chain_step": i+1})

        # ── Scout step: deterministic file reading, no LLM ──
        elif step_type == "scout":
            print(f"[BRAIN] TDD step {i+1}/{len(steps)}: scouting {project_path}")
            publish_event(Channel.BRAIN_ACTION, {"nerve": "scout", "args": "file scan", "chain_step": i+1})

            scout_output = _tdd_scout(project_path, project_facts)
            compressed = compress_chain_output(scout_output, "scout")

            checklist.check(i, bool(scout_output), compressed[:200])
            chain_context.append(compressed)
            publish_event(Channel.NERVE_RESULT, {"nerve": "scout", "output": compressed[:200], "chain_step": i+1})

        # ── Code generation steps: in-process LLM, no subprocess ──
        elif step_type in ("test_writer", "implementer"):
            print(f"[BRAIN] TDD step {i+1}/{len(steps)}: {step_type} (in-process LLM)")
            publish_event(Channel.BRAIN_ACTION, {"nerve": step_type, "args": task[:200], "chain_step": i+1})

            extra = chain_step.get("_retry_context", "")
            code_output = _tdd_llm_step(
                step_type, task, project_facts, chain_context,
                extra_context=extra, target_file=target_file, impl_file=impl_file_path,
            )

            print(f"[BRAIN] TDD {step_type} output ({len(code_output)} chars): {code_output[:200]}")
            publish_event(Channel.NERVE_RESULT, {"nerve": step_type, "output": code_output[:200], "chain_step": i+1})

            if target_file and code_output and len(code_output) > 10:
                target_dir = os.path.dirname(target_file)
                os.makedirs(target_dir, exist_ok=True)

                write_args = json.dumps({"command": "write", "path": target_file, "content": code_output})
                invoke_nerve("touch", write_args)
                print(f"[BRAIN] TDD wrote {len(code_output)} chars to {target_file}")

                if step_type == "test_writer":
                    test_file_path = target_file
                    print(f"[BRAIN] TDD test file: {test_file_path}")

                checklist.check(i, True, f"Wrote {os.path.basename(target_file)} ({len(code_output)} chars)")
                chain_context.append(code_output[:500])
            else:
                checklist.check(i, False, "LLM produced no usable code")
                chain_context.append("(no code)")

        publish_event(Channel.BRAIN_CHECKLIST, checklist.to_dict())
        print(f"[BRAIN] TDD step {i+1} complete: {checklist.steps[i]['status']}")
        i += 1

    # Build final response
    summary = checklist.summary()
    if checklist.is_complete():
        msg = f"TDD chain completed successfully.\n\n{summary}"
    else:
        failed = checklist.failed_step()
        if failed is not None:
            msg = f"TDD chain stopped at step {failed + 1}.\n\n{summary}"
        else:
            msg = f"TDD chain finished.\n\n{summary}"

    mem.cold.set_fact(f"task:{checklist.task_id}", "checklist", json.dumps(checklist.to_dict()))

    _tdd_user_id = get_task_origin().get("user_id", "")
    mem.hot.add_message("user", task, user_id=_tdd_user_id)
    mem.hot.add_message("assistant", msg, user_id=_tdd_user_id)
    mem.record_episode({
        "task": task, "nerve": "tdd_chain",
        "tool": "chain", "success": checklist.is_complete(),
        "result_summary": msg[:200],
        "user_id": _tdd_user_id,
    })
    publish_response(msg)
    return msg


def _handle_recipe_chain(task: str, recipe_decision: dict) -> str:
    """Execute a planner-generated recipe as a nerve chain.

    The recipe_decision has the same format as chain_nerves:
    {"action": "chain_nerves", "steps": [...], "goal": "...", "recipe_id": "..."}

    Each step is a nerve invocation. Missing nerves are synthesized on the fly.
    After execution, the recipe is evaluated and stored/updated.
    """
    steps = recipe_decision.get("steps", [])
    goal = recipe_decision.get("goal", task)
    recipe_id = recipe_decision.get("recipe_id", "")
    nerve_catalog = discover_nerves()
    available = list(nerve_catalog.keys())

    print(f"[BRAIN] Recipe chain started: {len(steps)} steps for goal: {goal}")
    publish_event(Channel.BRAIN_THOUGHT, {"stage": "recipe_chain", "steps": len(steps), "goal": goal})

    chain_context = []
    final_output = ""
    last_nerve_result = {}
    step_results = []  # for eval

    for i, chain_step in enumerate(steps):
        nerve_name = re.sub(r"[^a-z0-9_]", "_", chain_step.get("nerve", "").lower()).strip("_")
        step_args = chain_step.get("args", task)
        step_desc = chain_step.get("description", "")

        if not nerve_name:
            print(f"[BRAIN] Recipe step {i+1}: missing nerve name, skipping")
            step_results.append({"step": i, "status": "skipped"})
            continue

        # For core senses: if args look like JSON, pass directly (recipe gave structured args)
        # Otherwise translate natural language args
        is_json_args = False
        if nerve_name in CORE_SENSES:
            try:
                json.loads(step_args)
                is_json_args = True
            except (json.JSONDecodeError, TypeError):
                step_args = _translate_sense_args(nerve_name, step_args, task)

        # Synthesize nerve if it doesn't exist (skip senses — they always exist)
        if nerve_name not in available and nerve_name not in CORE_SENSES:
            from arqitect.brain.permissions import can_model_fabricate
            if not can_model_fabricate():
                print(f"[BRAIN] Recipe step {i+1}: model too small to synthesize '{nerve_name}', skipping")
                continue
            desc = step_desc or step_args or goal
            print(f"[BRAIN] Recipe step {i+1}: synthesizing nerve '{nerve_name}'")
            publish_event(Channel.BRAIN_THOUGHT, {"stage": "recipe_synthesize", "nerve": nerve_name, "step": i+1})
            nerve_name, _ = synthesize_nerve(nerve_name, desc, trigger_task=task)
            available.append(nerve_name)

        # Inject previous chain context (only for non-JSON args)
        if chain_context and not is_json_args:
            context_summary = "\n".join(
                f"[Step {j+1} result]: {ctx}" for j, ctx in enumerate(chain_context)
            )
            step_args = f"{step_args}\n\nContext from previous steps:\n{context_summary}"

        print(f"[BRAIN] Recipe step {i+1}/{len(steps)}: invoking '{nerve_name}'")
        publish_event(Channel.BRAIN_ACTION, {"nerve": nerve_name, "args": step_args, "chain_step": i+1})
        output = invoke_nerve(nerve_name, step_args)
        publish_event(Channel.NERVE_RESULT, {"nerve": nerve_name, "output": output, "chain_step": i+1})

        # Extract response text
        try:
            result = json.loads(output)
            step_output = result.get("response", output) if isinstance(result, dict) else output
            if isinstance(result, dict):
                last_nerve_result = result
        except (json.JSONDecodeError, TypeError):
            step_output = output

        step_ok = not _is_nerve_error(str(step_output))
        chain_context.append(str(step_output)[:500])
        final_output = step_output
        step_results.append({"step": i, "nerve": nerve_name, "status": "ok" if step_ok else "error"})
        print(f"[BRAIN] Recipe step {i+1} complete: {str(step_output)[:100]}")

    # Build final response
    combined = str(final_output).strip() if len(chain_context) <= 1 else "\n".join(chain_context)
    _chain_failed = _is_nerve_error(combined)

    if _chain_failed:
        nerve_names = ", ".join(s.get("nerve", "?") for s in steps)
        msg = _graceful_failure_message(task, nerve_names)
    elif len(chain_context) > 1:
        chain_summary = "\n\n".join(f"Step {j+1}: {ctx}" for j, ctx in enumerate(chain_context))
        msg = llm_generate(
            COMMUNICATION_MODEL,
            f"The user asked: {task}\n\nData collected:\n{chain_summary}\n\n"
            f"Respond directly to the user. Start with the actual answer — "
            f"NO preamble, NO meta-commentary. Just give the combined answer.",
            system=_resolve_adapter("communication")["system_prompt"]
        ).strip()
    else:
        msg = combined

    # Evaluate and store recipe
    from arqitect.brain.planner import evaluate_and_store_recipe
    evaluate_and_store_recipe(recipe_decision, step_results, task, not _chain_failed)

    # Encode any image from the chain's last nerve output
    _enrich_nerve_result_with_image(last_nerve_result)

    _recipe_user_id = get_task_origin().get("user_id", "")
    mem.hot.add_message("user", task, user_id=_recipe_user_id)
    mem.hot.add_message("assistant", msg, user_id=_recipe_user_id)
    mem.record_episode({
        "task": task, "nerve": "recipe_chain",
        "tool": "chain", "success": not _chain_failed,
        "result_summary": str(msg)[:200],
        "user_id": _recipe_user_id,
    })
    publish_response(msg, nerve_result=last_nerve_result)
    return msg



def think(task: str, history: list[str] | None = None, depth: int = 0) -> str:
    """Main reasoning loop — take a task and process it."""
    user_id = get_task_origin().get("user_id", "")
    if depth == 0:
        get_consolidator().wake()
    if depth > 5:
        # Personality-flavored fallback instead of generic message
        publish_event(Channel.BRAIN_THOUGHT, {"stage": "depth_limit", "task": task})
        msg = (
            "I wasn't able to resolve this one. "
            "Could you try rephrasing or giving me a bit more detail?"
        )
        publish_response(msg)
        return msg

    # Safety filter — reject harmful input before any processing
    if depth == 0:
        is_safe, refusal = _safety_check_input(task)
        if not is_safe:
            publish_event(Channel.BRAIN_THOUGHT, {"stage": "safety_block", "task": task})
            publish_response(refusal)
            return refusal

    # Feedback detection is handled by the brain's context window — it sees
    # conversation history and emits {"action": "feedback"} when appropriate.
    # No separate pre-check needed.

    # Follow-up context is handled by the sliding conversation window injected
    # into the LLM prompt (see _build_conversation_context). No separate
    # embedding-based detection needed.

    # Pre-LLM: recalibration detection
    _recal_match = re.match(r"recalibrate\s+(senses|all|sight|hearing|touch|awareness|communication)", task.strip().lower())
    if _recal_match and not history:
        publish_event(Channel.BRAIN_THOUGHT, {"stage": "recalibration", "task": task})
        target = _recal_match.group(1)
        if target in ("senses", "all"):
            results = calibrate_all_senses()
            _store_calibration_in_memory(results)
        else:
            results = {target: calibrate_sense(target)}
            _store_calibration_in_memory(results)
        # Build summary
        lines = []
        for name, cal in sorted(results.items()):
            status = cal.get("status", "unavailable")
            caps = cal.get("capabilities", {})
            available = [k for k, v in caps.items() if v.get("available")]
            missing = [k for k, v in caps.items() if not v.get("available")]
            line = f"  {name} [{status}]: {', '.join(available)}"
            if missing:
                line += f" (missing: {', '.join(missing)})"
            lines.append(line)
        msg = "Recalibration complete:\n" + "\n".join(lines)
        mem.hot.add_message("user", task, user_id=user_id)
        mem.hot.add_message("assistant", msg, user_id=user_id)
        publish_response(msg)
        return msg

    # Add message to conversation buffer
    mem.hot.add_message("user", task, user_id=user_id)

    # Get memory context for this task
    context_data = mem.get_context_for_task(task, user_id=user_id)

    nerve_catalog = discover_nerves()
    available = list(nerve_catalog.keys())

    # TODO: We already fetch past episodes in context_data. Use episode history
    # to rank nerves that previously succeeded for similar tasks — show those
    # first to the brain. If the brain fails to match from the short list,
    # retry with the full catalog. This avoids flooding the context window
    # while still giving the LLM all options as a fallback.

    # Circuit breaker — filter out nerves with open circuits (core senses are exempt)
    blocked_nerves = [n for n in available if n not in CORE_SENSES and not _cb_is_available(n)]
    if blocked_nerves:
        for n in blocked_nerves:
            del nerve_catalog[n]
        available = list(nerve_catalog.keys())
        print(f"[CIRCUIT] Filtered {len(blocked_nerves)} nerve(s) with open circuits")

    # Context detection is handled by the LLM via the update_context action.
    # No pre-LLM regex bypass — the LLM decides what's a context statement.

    # Intent classification → recipe-based planner for workflows
    if not history:
        intent = classify_intent(task)
        intent_type = intent.get("type", IntentType.DIRECT)

        if intent_type == IntentType.WORKFLOW:
            category = intent.get("category", "")
            print(f"[BRAIN] Intent: workflow (category={category})")

            # Gather project context if available
            project_path = detect_project_path(task)
            project_facts = None
            if project_path:
                from arqitect.knowledge.project_profiler import get_stored_profile, store_profile
                project_facts = get_stored_profile(project_path) or store_profile(project_path)

            # TDD shortcut: coding workflow on existing project with known language
            if project_facts and project_facts.get("language") and is_coding_task(task):
                print(f"[BRAIN] TDD chain: {project_path} ({project_facts.get('language')}+{project_facts.get('framework', '?')})")
                decision, checklist = build_tdd_chain(task, project_facts)
                return _handle_tdd_chain(task, decision, checklist)

            # General workflow: planner generates/matches a recipe → chain_nerves
            planned = plan_task(task, category, project_facts)
            if planned:
                steps = planned.get("steps", [])
                goal = planned.get("goal", task)
                print(f"[BRAIN] Planner recipe: {len(steps)} steps")
                # Execute as chain_nerves — uses the existing nerve chain handler
                return _handle_recipe_chain(task, planned)

    # Truncate excessively long tasks before feeding into the routing prompt
    if len(task) > _MAX_TASK_LENGTH:
        task = task[:_MAX_TASK_LENGTH] + "... [truncated]"

    # Build context for LLM — full catalog, LLM decides routing
    nerve_list = "\n".join(f"  - {n}: {d}" for n, d in nerve_catalog.items())
    if nerve_catalog:
        context = f"Available nerves:\n{nerve_list}\n\nTask: {task}"
    else:
        context = f"No nerves exist yet. You MUST synthesize a new nerve.\n\nTask: {task}"

    # Inject episode hints — only successful episodes to guide routing
    episodes = [ep for ep in context_data.get("episodes", []) if ep.get("success")]
    if episodes:
        ep_hints = "\n".join(
            f"  - {ep.get('task', '?')} -> {ep.get('nerve', '?')}"
            f"{': ' + ep['result_summary'] if ep.get('result_summary') else ''}"
            for ep in episodes[:3]
        )
        context += f"\n\nRecent similar tasks:\n{ep_hints}"

    # Inject user profile (name, gender, preferences) for personalization
    if user_id:
        profile = mem.cold.get_user_profile(user_id)
        if profile:
            profile_str = ", ".join(f"{k}: {v}" for k, v in profile.items())
            context += f"\n\nUser profile: {profile_str}"

    # Inject user facts
    facts = context_data.get("facts", {})
    if facts:
        fact_str = ", ".join(f"{k}={v}" for k, v in facts.items())
        context += f"\n\nKnown user facts: {fact_str}"

    # Inject recent conversation history — sliding window sized by community config
    window_size = get_conversation_window("brain")
    max_msg_len = get_message_truncation("brain")
    recent_convo = mem.hot.get_conversation(limit=window_size, user_id=user_id)
    # Exclude the current message (already appended above) to avoid duplication
    if len(recent_convo) > 1:
        convo_lines = []
        for msg in recent_convo[:-1]:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if len(content) > max_msg_len:
                content = content[:max_msg_len] + "..."
            convo_lines.append(f"  [{role}]: {content}")
        if convo_lines:
            context += "\n\nRecent conversation:\n" + "\n".join(convo_lines)

    if history:
        context += "\n\nPrevious steps:\n" + "\n".join(history)

    print(f"[BRAIN] Thinking about: {task}")
    print(f"[BRAIN] Catalog ({len(nerve_catalog)}): {nerve_catalog}")
    print(f"[BRAIN] Context sent to LLM:\n{context[:500]}")
    publish_event(Channel.BRAIN_THOUGHT, {"stage": "thinking", "task": task, "catalog": nerve_catalog})

    raw = llm_generate(BRAIN_MODEL, context, system=get_system_prompt())
    print(f"[BRAIN] Raw response: {raw}")

    decision = extract_json(raw)
    if not decision:
        publish_response(raw.strip())
        return raw.strip()

    ctx = DispatchContext(
        task=task,
        decision=decision,
        user_id=user_id,
        history=history or [],
        depth=depth,
        nerve_catalog=nerve_catalog,
        available=available,
        think_fn=think,
    )
    return dispatch_action(ctx)


def listen_redis():
    """Listen for tasks on Redis Pub/Sub (daemon mode for dashboard)."""
    # Initialize OpenTelemetry tracing (writes to ./traces/*.jsonl)
    try:
        from arqitect.telemetry import init_telemetry
        trace_file = init_telemetry()
        print(f"[TELEMETRY] Tracing to {trace_file}")
    except Exception as e:
        print(f"[TELEMETRY] Init failed (tracing disabled): {e}")

    try:
        # Bootstrap core senses (immutable, always present)
        bootstrap_senses()

        # Sync community manifest, adapters, and seed missing tools
        _sync_community_manifest()
        _sync_community_adapters()
        _seed_community_tools()
        _seed_community_nerves()

        # Prune duplicate nerves at startup (senses are protected)
        prune_degenerate_nerves()

        # Consolidate similar nerves (only consolidate merges/deletes)
        consolidate_nerves()

        # Bootstrap session on startup
        bootstrap_session()

        # Clear stale conversations from previous session
        mem.hot.clear_all_conversations()

        from arqitect.config.loader import get_redis_host_port
        _host, _port = get_redis_host_port()
        sub = redis.Redis(host=_host, port=_port, decode_responses=True)
        pubsub = sub.pubsub()
        pubsub.subscribe(Channel.BRAIN_TASK, Channel.SYSTEM_KILL, Channel.MEMORY_EPISODE, Channel.MEMORY_TOOL_LEARNED, Channel.SENSE_PEEK, Channel.SENSE_VOICE, Channel.SENSE_IMAGE, Channel.SENSE_CONFIG)

        # Initialize idle manager — starts the first idle timer for reconciliation
        get_consolidator()

        # Sync nerve status to Redis (validates against filesystem)
        publish_nerve_status()
    except Exception as e:
        import traceback
        print(f"[BRAIN] FATAL: Bootstrap failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print(f"[BRAIN] Progenitor online. Nerves: {list_nerves()}")
    print("[BRAIN] Listening on Redis channels: brain:task, memory:episode, memory:tool_learned, sense:peek, sense:voice, sense:image, sense:config...")
    publish_event(Channel.SYSTEM_STATUS, {"state": "online", "nerves": list(discover_nerves().keys())})

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        channel = message["channel"]

        if channel == Channel.SYSTEM_KILL:
            print("[BRAIN] KILL SIGNAL RECEIVED — shutting down.")
            publish_event(Channel.SYSTEM_STATUS, {"state": "killed"})
            break

        if channel == Channel.MEMORY_EPISODE:
            # Nerve published an episode — record it in warm memory
            try:
                episode = json.loads(message["data"])
                mem.record_episode(episode)
            except Exception as e:
                print(f"[BRAIN] Episode record error: {e}")
            continue

        if channel == Channel.MEMORY_TOOL_LEARNED:
            # Nerve learned a new tool — update cold memory (deduplicated)
            try:
                data = json.loads(message["data"])
                nerve_name = data.get("nerve", "")
                tool_name = data.get("tool", "")
                if nerve_name and tool_name:
                    # Issue 6 fix: Only record if truly new
                    existing_tools = mem.cold.get_nerve_tools(nerve_name)
                    if tool_name not in existing_tools:
                        mem.cold.add_nerve_tool(nerve_name, tool_name)
                        print(f"[BRAIN] Recorded: {nerve_name} learned tool '{tool_name}'")
            except Exception as e:
                print(f"[BRAIN] Tool learned error: {e}")
            continue

        if channel == Channel.SENSE_PEEK:
            # Dashboard requested a sight peek — screenshot or camera capture
            try:
                import base64
                peek_data = json.loads(message["data"])
                source = peek_data.get("source", "screenshot")

                if source == "camera":
                    # Camera capture via ffmpeg
                    import platform
                    import shutil
                    os.makedirs(SANDBOX_DIR, exist_ok=True)
                    frame_path = os.path.join(SANDBOX_DIR, "camera_frame.jpg")
                    system = platform.system()
                    captured = False
                    if shutil.which("imagesnap"):
                        result = subprocess.run(
                            ["imagesnap", "-q", frame_path],
                            capture_output=True, timeout=10,
                        )
                        captured = result.returncode == 0
                    elif shutil.which("ffmpeg"):
                        if system == "Darwin":
                            result = subprocess.run(
                                ["ffmpeg", "-y", "-f", "avfoundation", "-framerate", "1",
                                 "-i", "0", "-frames:v", "1", frame_path],
                                capture_output=True, timeout=10,
                            )
                        else:
                            result = subprocess.run(
                                ["ffmpeg", "-y", "-f", "v4l2", "-i", "/dev/video0",
                                 "-frames:v", "1", frame_path],
                                capture_output=True, timeout=10,
                            )
                        captured = result.returncode == 0

                    if captured and os.path.exists(frame_path):
                        with open(frame_path, "rb") as f:
                            img_b64 = base64.b64encode(f.read()).decode("utf-8")
                        publish_event(Channel.SENSE_SIGHT_FRAME, {
                            "image": img_b64,
                            "timestamp": time.time(),
                            "source": "camera",
                        })
                        print("[BRAIN] Sight peek: camera frame captured and published")
                    else:
                        publish_event(Channel.SENSE_SIGHT_FRAME, {
                            "error": "Camera capture failed — no camera tool available or no camera detected",
                            "timestamp": time.time(),
                        })
                else:
                    # Default: screenshot
                    output = invoke_nerve("sight", json.dumps({"mode": "screenshot"}))
                    screenshot_path = os.path.join(SANDBOX_DIR, "screenshot.png")
                    if os.path.exists(screenshot_path):
                        with open(screenshot_path, "rb") as f:
                            img_b64 = base64.b64encode(f.read()).decode("utf-8")
                        publish_event(Channel.SENSE_SIGHT_FRAME, {
                            "image": img_b64,
                            "timestamp": time.time(),
                            "source": "screenshot",
                        })
                        print("[BRAIN] Sight peek: screenshot captured and published")
                    else:
                        publish_event(Channel.SENSE_SIGHT_FRAME, {
                            "error": "Screenshot capture failed",
                            "timestamp": time.time(),
                        })
            except Exception as e:
                print(f"[BRAIN] Sight peek error: {e}")
                publish_event(Channel.SENSE_SIGHT_FRAME, {"error": str(e)})
            continue

        if channel == Channel.SENSE_VOICE:
            # Voice message from dashboard — STT then process as task
            try:
                voice_data = json.loads(message["data"])
                audio_b64 = voice_data.get("audio_b64", "")
                if not audio_b64:
                    publish_response("No audio received.")
                    continue

                # Save audio to file
                os.makedirs(SANDBOX_DIR, exist_ok=True)
                audio_path = os.path.join(SANDBOX_DIR, "voice_input.webm")
                with open(audio_path, "wb") as f:
                    f.write(base64.b64decode(audio_b64))

                # Convert webm to wav for whisper (if ffmpeg available)
                wav_path = os.path.join(SANDBOX_DIR, "voice_input.wav")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
                        capture_output=True, timeout=15,
                    )
                except Exception:
                    wav_path = audio_path  # fallback, try raw

                # STT via hearing sense
                stt_output = invoke_nerve("hearing", json.dumps({"mode": "stt", "audio_path": wav_path}))
                try:
                    stt_result = json.loads(stt_output) if isinstance(stt_output, str) else stt_output
                except Exception:
                    stt_result = {}

                transcribed = stt_result.get("text", "").strip() if isinstance(stt_result, dict) else ""
                if not transcribed:
                    error_msg = stt_result.get("error", "Could not transcribe audio") if isinstance(stt_result, dict) else "STT failed"
                    publish_response(f"Couldn't understand the audio: {error_msg}")
                    continue

                # Publish transcription so dashboard shows what was said
                publish_event(Channel.SENSE_STT_RESULT, {"text": transcribed})
                print(f"[BRAIN] Voice transcribed: {transcribed}")

                # Process as a regular task
                result = think(transcribed)
                print(f"[RESULT] {result}\n")
            except Exception as e:
                print(f"[BRAIN] Voice processing error: {e}")
                publish_response(f"Voice processing error: {e}")
            continue

        if channel == Channel.SENSE_CONFIG:
            # Sense configuration from dashboard — persist to cold memory
            try:
                config_data = json.loads(message["data"])
                sense_name = config_data.get("sense", "")
                key = config_data.get("key", "")
                value = config_data.get("value", "")
                if sense_name and key and value:
                    mem.cold.set_fact("sense_config", f"{sense_name}.{key}", value, confidence=1.0)
                    print(f"[BRAIN] Sense config saved: {sense_name}.{key} = {value}")
            except Exception as e:
                print(f"[BRAIN] Sense config error: {e}")
            continue

        if channel == Channel.SENSE_IMAGE:
            # Image from dashboard — analyze with sight then respond
            try:
                img_data = json.loads(message["data"])
                image_b64 = img_data.get("image_b64", "")
                prompt = img_data.get("prompt", "Describe this image in detail.")
                if not image_b64:
                    publish_response("No image received.")
                    continue

                # Analyze with sight sense
                sight_output = invoke_nerve("sight", json.dumps({"base64": image_b64, "prompt": prompt}))
                try:
                    sight_result = json.loads(sight_output) if isinstance(sight_output, str) else sight_output
                except Exception:
                    sight_result = {}

                if isinstance(sight_result, dict) and sight_result.get("response"):
                    description = sight_result["response"]
                    print(f"[BRAIN] Image analyzed: {description[:100]}...")
                    publish_response(description)
                elif isinstance(sight_result, dict) and sight_result.get("error"):
                    publish_response(f"Sight error: {sight_result['error']}")
                else:
                    publish_response("My eyes are a bit blurry on that one — couldn't make sense of the image.")
            except Exception as e:
                print(f"[BRAIN] Image processing error: {e}")
                publish_response(f"Image processing error: {e}")
            continue

        try:
            raw_data = message["data"]
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                # Handle malformed JSON (e.g. unescaped characters)
                data = {"task": raw_data if isinstance(raw_data, str) else ""}
            task = data.get("task", "")
            source = data.get("source", "dashboard")
            chat_id = data.get("chat_id", "")
            connector_user_id = data.get("connector_user_id", "")

            # Onboarding: platform connectors (Telegram, WhatsApp) always send
            # connector_user_id. Bridge handles its own onboarding — sends
            # empty connector_user_id for anon users.
            user_id = ""
            if connector_user_id and source:
                from arqitect.brain.onboarding import handle_onboarding
                onboarding_msg, user_id = handle_onboarding(
                    mem.cold, source, connector_user_id, task
                )
                if onboarding_msg:
                    # User is in onboarding flow — send response and skip brain processing
                    set_task_origin(source, chat_id, "")
                    publish_response(onboarding_msg)
                    continue

            # Store sender info from connector (name, language)
            sender_name = data.get("sender_name", "")
            if user_id and sender_name:
                existing = mem.cold.get_user(user_id)
                if existing and not existing.get("display_name"):
                    mem.cold.set_user_display_name(user_id, sender_name)
                    print(f"[BRAIN] Stored display name '{sender_name}' for user {user_id[:8]}...")

            # Store language preference from Telegram language_code
            language_code = data.get("language_code", "")
            if user_id and language_code:
                existing_lang = mem.cold.get_user_facts(user_id).get("language")
                if not existing_lang:
                    mem.cold.set_user_fact(user_id, "language", language_code)
                    print(f"[BRAIN] Stored language '{language_code}' for user {user_id[:8]}...")

            # Store shared location in user session (WhatsApp/Telegram location messages)
            shared_location = data.get("location")
            if shared_location and shared_location.get("latitude"):
                lat = shared_location["latitude"]
                lon = shared_location["longitude"]
                loc_ctx = {
                    "latitude": str(lat),
                    "longitude": str(lon),
                }
                # Use provided name/address, or reverse-geocode
                if shared_location.get("name"):
                    loc_ctx["city"] = shared_location["name"]
                elif shared_location.get("address"):
                    loc_ctx["city"] = shared_location["address"]
                else:
                    city = _reverse_geocode(lat, lon)
                    if city:
                        loc_ctx["city"] = city
                if user_id:
                    mem.hot.update_session(loc_ctx, user_id=user_id)
                    for k, v in loc_ctx.items():
                        mem.cold.set_user_fact(user_id, k, v)
                else:
                    mem.hot.update_session(loc_ctx)
                print(f"[BRAIN] Stored location ({lat},{lon}) -> {loc_ctx.get('city', '?')}")

            # Bootstrap per-user session on first message
            if user_id:
                bootstrap_user_session(user_id)

            # Pre-route images from connectors (e.g. WhatsApp) to sight sense
            media = data.get("media")
            if media and media.get("type") == "image":
                set_task_origin(source, chat_id, user_id)
                image_b64 = media.get("image_b64", "")
                image_path = media.get("path", "")

                # Use caption as prompt if user wrote something; otherwise generic
                caption = task if task and not task.startswith("[Received") else ""
                prompt = caption if caption else "Describe this image in detail."

                # Build sight args
                sight_args = {"prompt": prompt}
                if image_b64:
                    sight_args["base64"] = image_b64
                elif image_path:
                    sight_args["image_path"] = image_path

                # Analyze with sight sense
                sight_output = invoke_nerve("sight", json.dumps(sight_args))
                try:
                    sight_result = json.loads(sight_output) if isinstance(sight_output, str) else sight_output
                except Exception:
                    sight_result = {}

                description = sight_result.get("response", "") if isinstance(sight_result, dict) else ""

                if caption and description:
                    # User asked a question about the image — feed description + question into think()
                    enriched = f"The user sent an image. Image analysis: {description}\n\nUser's question: {caption}"
                    think(enriched)
                elif description:
                    publish_response(description)
                else:
                    error = sight_result.get("error", "Couldn't read the image") if isinstance(sight_result, dict) else "Couldn't read the image"
                    publish_response(f"Sight error: {error}")
                continue

            if task:
                set_task_origin(source, chat_id, user_id)
                print(f"\n[BRAIN] Task from {source}: {task}")
                # Run think() in a thread with a 120s timeout so the brain never freezes
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(think, task)
                    try:
                        result = future.result(timeout=120)
                        print(f"[RESULT] {result}\n")
                    except concurrent.futures.TimeoutError:
                        print(f"[BRAIN] Task timed out after 120s: {task}")
                        publish_response("That request took too long to process. Could you try again, perhaps with a simpler request?")
        except Exception as e:
            print(f"[BRAIN] Error processing task: {e}")
            import traceback
            traceback.print_exc()
            publish_response("Something got tangled in the wires. Mind trying that again?")


def main():
    """Interactive loop — read tasks from stdin or Redis."""
    os.makedirs(NERVES_DIR, exist_ok=True)
    os.makedirs(SANDBOX_DIR, exist_ok=True)

    # Daemon mode — listen_redis() handles its own bootstrap
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        listen_redis()
        return

    # Bootstrap core senses (non-daemon modes)
    bootstrap_senses()

    # Prompt user for any pending calibration config (CLI mode)
    _prompt_calibration_config()

    print("[BRAIN] Progenitor online.")
    print("[BRAIN] Available nerves:", list_nerves())
    print()

    # Publish nerve status on startup — qualification happens in dream state
    publish_nerve_status()

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        bootstrap_session()
        result = think(task)
        print(f"\n[RESULT] {result}")
        return

    # Interactive mode
    bootstrap_session()
    while True:
        try:
            task = input("[YOU] > ").strip()
            if not task:
                continue
            if task.lower() in ("quit", "exit"):
                print("[BRAIN] Shutting down.")
                break
            result = think(task)
            print(f"\n[RESULT] {result}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n[BRAIN] Shutting down.")
            break


if __name__ == "__main__":
    main()
