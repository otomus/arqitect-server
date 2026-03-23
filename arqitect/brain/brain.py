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
import threading
import time
import uuid
from dataclasses import dataclass

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
from arqitect.brain.plan_session import PlanSession
from arqitect.brain.plan_router import classify_plan_message
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


# Module-level chain progress tracking — lets listen_redis() extend the
# deadline for long-running chains that are still making forward progress.
_chain_active = threading.Event()
_last_chain_progress: float = 0.0



# Max nerves shown to the routing LLM, keyed by model size class.
# Smaller models have tighter context budgets (tinylm=2048, small=4096).
_NERVE_LIMIT_BY_SIZE: dict[str, int] = {
    "tinylm": 5,
    "small": 8,
    "medium": 15,
    "large": 20,
}
_DEFAULT_NERVE_LIMIT = 20

# Maximum task length (chars) fed into the routing prompt
_MAX_TASK_LENGTH = 4000

# Base timeout for simple (non-chain) tasks
_BASE_TIMEOUT_S = 120
# How often we poll the future for completion
_POLL_INTERVAL_S = 5


def _await_future_with_chain_extension(future: "concurrent.futures.Future[str]") -> str:
    """Poll a future, extending the deadline while a chain makes progress.

    Simple tasks time out after _BASE_TIMEOUT_S (120s).  When a chain is
    active and reporting progress, the deadline resets from the last progress
    timestamp — so a chain step that takes up to 120s of *idle* time is still
    allowed, but one that stalls for 120s with no step completion gets killed.

    Args:
        future: The Future wrapping the think() call.

    Returns:
        The result string from think().

    Raises:
        concurrent.futures.TimeoutError: If the task exceeds its deadline.
    """
    import concurrent.futures

    start = time.time()
    deadline = start + _BASE_TIMEOUT_S

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise concurrent.futures.TimeoutError()

        poll_timeout = min(_POLL_INTERVAL_S, remaining)
        try:
            return future.result(timeout=poll_timeout)
        except concurrent.futures.TimeoutError:
            pass  # poll again — check whether chain extended the deadline

        # If a chain is actively running and recently made progress,
        # push the deadline out from the last progress timestamp.
        if _chain_active.is_set() and _last_chain_progress > 0:
            chain_deadline = _last_chain_progress + _BASE_TIMEOUT_S
            if chain_deadline > deadline:
                deadline = chain_deadline


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


@dataclass
class ChainCheckpoint:
    """Snapshot of chain progress for crash recovery.

    Attributes:
        task_id: Unique identifier for this chain execution.
        chain_type: "tdd" or "recipe".
        step_index: Zero-based index of the current step.
        total_steps: Total number of steps in the chain.
        chain_context: Accumulated output context from prior steps.
        checklist_dict: Serialized checklist (from TaskChecklist.to_dict()).
        decision: The original routing decision that spawned this chain.
        status: One of "running", "complete", or "failed".
    """

    task_id: str
    chain_type: str
    step_index: int
    total_steps: int
    chain_context: list
    checklist_dict: dict
    decision: dict
    status: str = "running"


def _checkpoint_chain(cp: ChainCheckpoint) -> None:
    """Persist a chain checkpoint to cold memory.

    Only the last 5 context entries are stored to keep the payload small.

    Args:
        cp: Snapshot of the current chain state.
    """
    state = {
        "task_id": cp.task_id,
        "chain_type": cp.chain_type,
        "step_index": cp.step_index,
        "total_steps": cp.total_steps,
        "chain_context": cp.chain_context[-5:],
        "checklist": cp.checklist_dict,
        "decision": cp.decision,
        "status": cp.status,
        "updated_at": time.time(),
    }
    mem.cold.set_fact("chain_state", cp.task_id, json.dumps(state))


def _mark_abandoned_chains() -> None:
    """Scan for chains left in 'running' state from a previous session.

    Called once at startup.  Any chain still marked as running could not
    have completed (the process crashed or was killed), so we mark it
    as abandoned and log a warning.
    """
    logger = logging.getLogger(__name__)
    chain_states = mem.cold.get_facts("chain_state")
    for task_id, raw in chain_states.items():
        try:
            state = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if state.get("status") != "running":
            continue
        state["status"] = "abandoned"
        state["updated_at"] = time.time()
        mem.cold.set_fact("chain_state", task_id, json.dumps(state))
        logger.warning(
            "[BRAIN] Abandoned chain detected: task_id=%s chain_type=%s step=%d/%d",
            task_id,
            state.get("chain_type", "unknown"),
            state.get("step_index", -1),
            state.get("total_steps", -1),
        )


def _handle_tdd_chain(task: str, decision: dict, checklist: TaskChecklist) -> str:
    """Execute a TDD chain using in-process LLM — no nerve subprocesses needed.

    Scout step: deterministic file reading (no LLM).
    Code steps: direct llm_generate() calls using already-loaded coder model.
    Exec steps: direct touch sense invocation.
    """
    steps = decision.get("steps", [])
    project_path = decision.get("project_path", ".")
    project_facts = decision.get("project_facts", {})
    chain_task_id = str(uuid.uuid4())

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

    global _last_chain_progress
    _chain_active.set()
    _last_chain_progress = time.time()

    i = 0
    while i < len(steps):
        chain_step = steps[i]
        nerve_name = chain_step.get("nerve", "")
        step_type = chain_step.get("step_type", "")
        target_file = chain_step.get("target_file", "")

        checklist.activate(i)
        publish_event(Channel.BRAIN_CHECKLIST, checklist.to_dict())
        get_consolidator().touch()

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
                                _last_chain_progress = time.time()
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
        _checkpoint_chain(ChainCheckpoint(
            task_id=chain_task_id, chain_type="tdd", step_index=i,
            total_steps=len(steps), chain_context=chain_context,
            checklist_dict=checklist.to_dict(), decision=decision,
        ))
        _last_chain_progress = time.time()
        print(f"[BRAIN] TDD step {i+1} complete: {checklist.steps[i]['status']}")
        i += 1

    _chain_active.clear()

    # Mark chain complete or failed
    _tdd_final_status = "complete" if checklist.is_complete() else "failed"
    _checkpoint_chain(ChainCheckpoint(
        task_id=chain_task_id, chain_type="tdd", step_index=len(steps),
        total_steps=len(steps), chain_context=chain_context,
        checklist_dict=checklist.to_dict(), decision=decision,
        status=_tdd_final_status,
    ))

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
    chain_task_id = str(uuid.uuid4())
    nerve_catalog = discover_nerves()
    available = list(nerve_catalog.keys())

    print(f"[BRAIN] Recipe chain started: {len(steps)} steps for goal: {goal}")
    publish_event(Channel.BRAIN_THOUGHT, {"stage": "recipe_chain", "steps": len(steps), "goal": goal})

    chain_context = []
    final_output = ""
    last_nerve_result = {}
    step_results = []  # for eval

    _chain_active.set()
    _last_chain_progress = time.time()

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
        get_consolidator().touch()
        _checkpoint_chain(ChainCheckpoint(
            task_id=chain_task_id, chain_type="recipe", step_index=i,
            total_steps=len(steps), chain_context=chain_context,
            checklist_dict={}, decision=recipe_decision,
        ))
        _last_chain_progress = time.time()
        print(f"[BRAIN] Recipe step {i+1} complete: {str(step_output)[:100]}")

    _chain_active.clear()

    # Build final response
    combined = str(final_output).strip() if len(chain_context) <= 1 else "\n".join(chain_context)
    _chain_failed = _is_nerve_error(combined)

    _checkpoint_chain(ChainCheckpoint(
        task_id=chain_task_id, chain_type="recipe", step_index=len(steps),
        total_steps=len(steps), chain_context=chain_context,
        checklist_dict={}, decision=recipe_decision,
        status="complete" if not _chain_failed else "failed",
    ))

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
    from arqitect.telemetry import span as _tspan
    with _tspan("brain.think", task=task[:500], depth=depth) as _ts:
        result = _think_inner(task, history=history, depth=depth)
        _ts.set_attribute("response_length", len(result) if result else 0)
        _ts.set_attribute("response_preview", (result or "")[:300])
        return result


def _gather_plan_context(plan: PlanSession, task: str, user_id: str) -> None:
    """Populate a PlanSession with context from past work and the project.

    Mutates plan in place — sets related_plans, related_episodes,
    project_facts, and matched_recipe.

    Args:
        plan: The PlanSession to enrich.
        task: The user's original message.
        user_id: The user who sent the message.
    """
    plan.related_plans = PlanSession.get_past_plans(mem, plan.category, limit=3)

    episodes = mem.warm.recall(task, user_id=user_id) if hasattr(mem, "warm") else []
    plan.related_episodes = episodes[:3] if episodes else []

    project_path = detect_project_path(task)
    if project_path:
        from arqitect.knowledge.project_profiler import get_stored_profile, store_profile
        plan.project_facts = get_stored_profile(project_path) or store_profile(project_path)

    matched = plan_task(task, plan.category, plan.project_facts)
    if matched:
        plan.matched_recipe = matched


def _build_planning_prompt(plan: PlanSession) -> str:
    """Build the LLM prompt from plan context.

    Args:
        plan: PlanSession with gathered context.

    Returns:
        Prompt string summarizing goal, past plans, project, and recipes.
    """
    parts = [f"The user wants to: {plan.goal}"]
    if plan.related_plans:
        summaries = [f"  - {p.get('goal', '?')} (status: {p.get('status', '?')})" for p in plan.related_plans[:3]]
        parts.append("Related past plans:\n" + "\n".join(summaries))
    if plan.project_facts:
        parts.append(f"Project: {plan.project_facts}")
    if plan.matched_recipe:
        parts.append(f"Matched existing recipe with {len(plan.matched_recipe.get('steps', []))} steps")
    return "\n\n".join(parts)


_PLAN_START_SYSTEM = (
    "You are starting a planning phase for a multi-step task. "
    "Summarize your understanding of what the user wants, reference any "
    "relevant past work if available, and ask clarifying questions to "
    "gather requirements before proposing a plan. Be conversational. "
    "Do NOT propose steps yet — just gather information."
)


def _handle_plan_start(task: str, intent: dict, user_id: str) -> str:
    """Start a new planning session when intent is classified as 'plan'.

    Args:
        task: The user's original message.
        intent: Intent classification dict with 'type' and optional 'category'.
        user_id: The user who sent the message.

    Returns:
        The brain's initial planning response.
    """
    category = intent.get("category", "")
    print(f"[BRAIN] Intent: plan (category={category})")
    publish_event(Channel.PLAN_UPDATE, {"stage": "plan_start", "task": task, "category": category})

    plan = PlanSession.create(user_id, goal=task, category=category)
    _gather_plan_context(plan, task, user_id)
    plan.add_message("user", task)

    planning_prompt = _build_planning_prompt(plan)

    response = llm_generate(BRAIN_MODEL, planning_prompt, system=_PLAN_START_SYSTEM)
    plan.add_message("assistant", response)
    plan.save(r)

    mem.hot.add_message("user", task, user_id=user_id)
    mem.hot.add_message("assistant", response, user_id=user_id)
    publish_event(Channel.PLAN_UPDATE, {"stage": "gathering", "plan_id": plan.plan_id})
    publish_response(response)
    return response


def _handle_plan_continue(task: str, plan: PlanSession, user_id: str) -> str:
    """Handle a message that continues an active plan's conversation.

    Adds the user's input to the plan context, generates a brain response that
    either asks more questions or proposes a plan.

    Args:
        task: The user's new message.
        plan: The active PlanSession.
        user_id: The user who sent the message.

    Returns:
        The brain's response (gathering more info or proposing a plan).
    """
    publish_event(Channel.PLAN_UPDATE, {"stage": "plan_continue", "plan_id": plan.plan_id})
    plan.add_requirement(task)
    plan.add_message("user", task)

    # Build context for the brain
    convo_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in plan.conversation_context
    )
    prompt = (
        f"Goal: {plan.goal}\n"
        f"Category: {plan.category}\n"
        f"Requirements gathered so far: {plan.requirements}\n"
        f"Conversation:\n{convo_text}\n\n"
        f"Based on the conversation, either ask more clarifying questions "
        f"or propose a concrete plan with numbered steps if you have enough info."
    )
    system = (
        "You are in a planning phase for a multi-step task. "
        "If you have enough requirements, propose a concrete numbered plan "
        "and ask the user to approve. Otherwise, ask more clarifying questions."
    )
    response = llm_generate(BRAIN_MODEL, prompt, system=system)

    # Check if the brain proposed a plan (heuristic: contains numbered steps)
    if plan.status == "gathering" and _response_contains_plan(response):
        steps = _extract_plan_steps(response)
        if steps:
            plan.propose(steps)
            publish_event(Channel.PLAN_UPDATE, {"stage": "proposed", "plan_id": plan.plan_id})

    plan.add_message("assistant", response)
    plan.save(r)

    mem.hot.add_message("user", task, user_id=user_id)
    mem.hot.add_message("assistant", response, user_id=user_id)
    publish_response(response)
    return response


def _handle_plan_approve(plan: PlanSession, user_id: str) -> str:
    """Handle user approval of a proposed plan — transition to execution.

    Args:
        plan: The active PlanSession (must be in 'proposed' status).
        user_id: The user who approved.

    Returns:
        The result of executing the plan as a recipe chain.
    """
    publish_event(Channel.PLAN_UPDATE, {"stage": "plan_approve", "plan_id": plan.plan_id})

    if plan.status != "proposed":
        # Not in a state to approve — treat as continue
        return _handle_plan_continue("I approve, let's go!", plan, user_id)

    plan.approve()
    plan.save(r)
    publish_event(Channel.PLAN_UPDATE, {"stage": "approved", "plan_id": plan.plan_id})

    recipe = plan.to_recipe()

    try:
        result = _handle_recipe_chain(plan.goal, recipe)
        plan.complete(success=True)
    except Exception as exc:
        print(f"[BRAIN] Plan execution failed: {exc}")
        plan.complete(success=False)
        result = f"The plan ran into an issue: {exc}"

    # Archive to cold memory and clean up Redis
    plan.archive(mem)
    plan.delete(r)
    publish_event(Channel.PLAN_UPDATE, {"stage": "done", "plan_id": plan.plan_id})
    return result


def _handle_plan_abort(plan: PlanSession, user_id: str) -> str:
    """Handle user aborting an active plan.

    Args:
        plan: The active PlanSession to abandon.
        user_id: The user who aborted.

    Returns:
        Confirmation message.
    """
    publish_event(Channel.PLAN_UPDATE, {"stage": "plan_abort", "plan_id": plan.plan_id})
    plan.abandon()
    plan.archive(mem)
    plan.delete(r)
    msg = "Plan cancelled. Let me know if you'd like to start something else."
    mem.hot.add_message("assistant", msg, user_id=user_id)
    publish_response(msg)
    return msg


def _response_contains_plan(response: str) -> bool:
    """Heuristic check: does the LLM response contain a numbered plan proposal?

    Args:
        response: The brain's response text.

    Returns:
        True if the response appears to contain a numbered plan.
    """
    import re
    # Look for at least 2 numbered items (e.g., "1." and "2.")
    numbered = re.findall(r"^\s*\d+\.\s", response, re.MULTILINE)
    return len(numbered) >= 2


def _extract_plan_steps(response: str) -> list[dict]:
    """Extract numbered steps from a plan proposal response.

    Args:
        response: The brain's response containing numbered steps.

    Returns:
        List of step dicts with 'step' number and 'description'.
    """
    import re
    steps = []
    for match in re.finditer(r"^\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\n\n|$)", response, re.MULTILINE | re.DOTALL):
        steps.append({
            "step": int(match.group(1)),
            "description": match.group(2).strip(),
        })
    return steps


def _think_inner(task: str, history: list[str] | None = None, depth: int = 0) -> str:
    """Inner think logic — wrapped by think() for tracing."""
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

    # STAGE 1: Plan gate — check for active plan BEFORE intent classification
    active_plan = PlanSession.get_active(user_id, r)
    if active_plan:
        plan_action = classify_plan_message(task, active_plan)
        if plan_action == "continue":
            return _handle_plan_continue(task, active_plan, user_id)
        elif plan_action == "approve":
            return _handle_plan_approve(active_plan, user_id)
        elif plan_action == "abort":
            return _handle_plan_abort(active_plan, user_id)
        # "aside" — fall through to normal routing

    # STAGE 2: Intent classification (no active plan, or aside message)
    if not history:
        intent = classify_intent(task)
        intent_type = intent.get("type", IntentType.DIRECT)

        if intent_type == IntentType.PLAN:
            return _handle_plan_start(task, intent, user_id)

    # Truncate excessively long tasks before feeding into the routing prompt
    if len(task) > _MAX_TASK_LENGTH:
        task = task[:_MAX_TASK_LENGTH] + "... [truncated]"

    # Pre-filter catalog — limit varies by model size class to respect
    # smaller context budgets (tinylm=2048, small=4096).
    from arqitect.matching import filter_nerve_catalog
    from arqitect.brain.adapters import get_active_variant

    size_class = get_active_variant("brain")
    nerve_limit = _NERVE_LIMIT_BY_SIZE.get(size_class, _DEFAULT_NERVE_LIMIT)
    nerve_catalog = filter_nerve_catalog(task, nerve_catalog, limit=nerve_limit)

    # Build context for LLM — filtered catalog, LLM decides routing
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

        # Mark any in-progress chains from a previous session as abandoned
        _mark_abandoned_chains()

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
            # Test bypass: if user_id is provided directly, skip onboarding.
            user_id = data.get("user_id", "")
            if not user_id and connector_user_id and source:
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
                # Run think() in a thread — chains that make progress can
                # exceed the base 120s deadline; idle tasks still time out.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(think, task)
                    try:
                        result = _await_future_with_chain_extension(future)
                        print(f"[RESULT] {result}\n")
                    except concurrent.futures.TimeoutError:
                        elapsed = time.time() - _last_chain_progress if _chain_active.is_set() else -1
                        print(f"[BRAIN] Task timed out: {task} (chain_idle={elapsed:.0f}s)")
                        publish_response("That request took too long to process. Could you try again, perhaps with a simpler request?")
        except Exception as e:
            _chain_active.clear()  # Ensure chain flag is reset on any error
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
