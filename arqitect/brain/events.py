"""Brain event publishing — Redis pub/sub for dashboard and inter-component communication."""

import base64
import json
import os
import threading

from arqitect.brain.adapters import resolve_prompt as _resolve_adapter, get_temperature, get_max_tokens
from arqitect.brain.config import r, mem
from arqitect.brain.safety import check_output as _safety_check_output
from arqitect.types import Channel, NerveRole, RedisKey, Tone
from arqitect.senses.communication.envelope import build_envelope, merge_nerve_result_into_envelope

# Task origin — simple dict, not thread-local, because think() runs in a
# ThreadPoolExecutor but there's only one task at a time.
_task_origin = {"source": "", "chat_id": "", "user_id": ""}


def set_task_origin(source: str = "", chat_id: str = "", user_id: str = ""):
    """Set the origin for the current task (called by brain before processing)."""
    _task_origin["source"] = source
    _task_origin["chat_id"] = chat_id
    _task_origin["user_id"] = user_id


def get_task_origin() -> dict:
    """Get the current task's origin."""
    return {
        "source": _task_origin.get("source", ""),
        "chat_id": _task_origin.get("chat_id", ""),
        "user_id": _task_origin.get("user_id", ""),
    }


def publish_event(channel: str, data: dict):
    """Publish an event to the synaptic bus."""
    r.publish(channel, json.dumps(data))


def publish_memory_state():
    """Publish current memory state to dashboard via Redis."""
    try:
        episodes = mem.warm.recall("", limit=5)
        ep_list = []
        for ep in episodes:
            ep_list.append({
                "task": ep.get("task", ""),
                "nerve": ep.get("nerve", ""),
                "tool": ep.get("tool", ""),
                "success": bool(ep.get("success", 1)),
            })
        facts = mem.cold.get_facts("user")
        r.publish(Channel.MEMORY_UPDATE, json.dumps({
            "episodes": ep_list,
            "facts": facts,
        }))
    except Exception as e:
        print(f"[WARN] publish_memory_state: {e}")


def publish_nerve_status():
    """Publish all nerve qualification data to Redis for dashboard.

    Validates nerves exist on disk — removes stale DB entries for deleted nerves.
    Always publishes (even if empty) so the dashboard stays in sync.

    Uses a single bulk query to avoid N+1 database access patterns.
    """
    try:
        import os
        from arqitect.brain.config import NERVES_DIR, SENSES_DIR

        # Single bulk load — replaces N+1 per-nerve queries
        all_nerve_data = mem.cold.get_all_nerve_data()

        nerves_list = []
        stale_nerves = []

        for name, data in all_nerve_data.items():
            # Validate nerve exists on disk
            is_sense = data["is_sense"]
            if is_sense:
                nerve_path = os.path.join(SENSES_DIR, name, "nerve.py")
            else:
                nerve_path = os.path.join(NERVES_DIR, name, "nerve.py")
            if not os.path.isfile(nerve_path):
                print(f"[EVENTS] Removing stale nerve '{name}' from registry (file missing)")
                stale_nerves.append(name)
                continue

            qual = data["qualification"] or {}
            nerves_list.append({
                "name": name,
                "score": round(qual.get("score", 0) * 100),
                "qualified": qual.get("qualified"),
                "status": "pass" if qual.get("qualified") else ("fail" if qual.get("qualified") is False else "unknown"),
                "tools": data["tools"],
                "iteration": qual.get("iterations", 0),
                "max_iterations": 3,
            })

        # Remove stale nerves from cold memory
        if stale_nerves:
            try:
                for name in stale_nerves:
                    mem.cold.delete_nerve(name)
            except Exception as e:
                print(f"[WARN] Failed to remove stale nerves: {e}")

        payload = json.dumps({"nerves": nerves_list})
        r.publish(Channel.NERVE_QUALIFICATION, payload)
        r.set(RedisKey.NERVE_STATUS, payload)
        _publish_all_nerve_details_bulk(all_nerve_data)
    except Exception as e:
        print(f"[EVENTS] publish_nerve_status failed: {e}")


def publish_nerve_details(name: str):
    """Store full nerve details in Redis for the dashboard deep-link page."""
    try:
        info = mem.cold.get_nerve_info(name) or {}
        qual = mem.cold.get_qualification("nerve", name)
        meta = mem.cold.get_nerve_metadata(name) or {}
        tools = mem.cold.get_nerve_tools(name)

        _overlay_community_metadata(meta, nerve_name=name)
        details = _build_nerve_details(name, info, meta, tools, qual)
        r.hset("synapse:nerve_details", name, json.dumps(details))
    except Exception as e:
        print(f"[WARN] publish_nerve_details({name}): {e}")


def _overlay_community_metadata(meta: dict, nerve_name: str = "") -> None:
    """For community nerves, resolve system_prompt and examples from cache.

    Mutates meta in place with values from the community cache context.json.
    Logs a warning if resolution fails — never silently drops errors.
    """
    if meta.get("origin") != "community":
        return
    try:
        from arqitect.brain.adapters import resolve_nerve_prompt
        ctx = resolve_nerve_prompt(nerve_name, meta.get("role", "tool"))
        if ctx:
            if ctx.get("system_prompt"):
                meta["system_prompt"] = ctx["system_prompt"]
            if ctx.get("few_shot_examples"):
                meta["examples"] = ctx["few_shot_examples"]
    except Exception as exc:
        print(f"[WARN] Failed to resolve community metadata for '{nerve_name}': {exc}")


def _build_nerve_details(name: str, info: dict, meta: dict, tools: list, qual: dict | None) -> dict:
    """Build the nerve details dict for the dashboard. Used by both single and bulk paths."""
    details = {
        "name": name,
        "description": info.get("description", ""),
        "role": meta.get("role", NerveRole.TOOL),
        "system_prompt": meta.get("system_prompt", ""),
        "examples": meta.get("examples", []),
        "tools": tools,
        "total_invocations": info.get("total_invocations", 0),
        "successes": info.get("successes", 0),
        "failures": info.get("failures", 0),
    }
    if qual:
        details["score"] = round(qual.get("score", 0) * 100)
        details["qualified"] = qual.get("qualified", False)
        details["iterations"] = qual.get("iterations", 0)
        details["test_count"] = qual.get("test_count", 0)
        details["pass_count"] = qual.get("pass_count", 0)
        details["test_results"] = qual.get("details", [])
        details["last_qualified"] = qual.get("timestamp", "")
    else:
        details["score"] = 0
        details["qualified"] = None
        details["test_results"] = []
    return details


def _publish_all_nerve_details_bulk(all_nerve_data: dict):
    """Store details for all nerves in Redis using pre-loaded bulk data."""
    try:
        for name, data in all_nerve_data.items():
            info = {
                "description": data["description"],
                "total_invocations": data["total_invocations"],
                "successes": data["successes"],
                "failures": data["failures"],
            }
            meta = {
                "role": data["role"],
                "system_prompt": data["system_prompt"],
                "examples": data["examples"],
                "origin": data.get("origin", "local"),
            }
            _overlay_community_metadata(meta, nerve_name=name)
            details = _build_nerve_details(name, info, meta, data["tools"], data["qualification"])
            r.hset("synapse:nerve_details", name, json.dumps(details))
    except Exception as e:
        print(f"[WARN] _publish_all_nerve_details_bulk: {e}")


def publish_all_nerve_details():
    """Store details for all nerves in Redis."""
    try:
        all_nerve_data = mem.cold.get_all_nerve_data()
        _publish_all_nerve_details_bulk(all_nerve_data)
    except Exception as e:
        print(f"[WARN] publish_all_nerve_details: {e}")


def _text_similarity(a: str, b: str) -> float:
    """Simple character-level similarity ratio."""
    if not a or not b:
        return 0.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _validate_response(msg: str, task: str = "") -> str | None:
    """Validate a response before sending to user.
    Returns None if OK, or a reason string if the response should be blocked."""
    if not msg or len(msg.strip()) < 2:
        return "empty"

    stripped = msg.strip()

    # Leaked JSON — response looks like a raw action object
    if stripped.startswith('{"action"') or stripped.startswith("{'action"):
        return "leaked_json"

    # Leaked JSON in code fence
    if stripped.startswith('```json') and '"action"' in stripped:
        return "leaked_json"

    # Leaked tool call syntax — catches "call:", "call,", "call " variants
    import re as _re
    if _re.match(r'^call[:\s,]', stripped, _re.IGNORECASE) and 'args' in stripped.lower():
        return "leaked_tool_call"

    # Echo detection — response is essentially the same as the input
    if task:
        norm_msg = stripped.lower().strip('.,!? ')
        norm_task = task.lower().strip('.,!? ')
        if norm_msg and norm_task and (norm_msg == norm_task or
            (len(norm_msg) < 100 and len(norm_task) < 100 and
             _text_similarity(norm_msg, norm_task) > 0.85)):
            return "echo"

    return None


def _apply_personality(msg: str, task: str = "") -> tuple[str, str]:
    """Apply a light personality touch to a response.

    Returns (flavored_msg, detected_tone). The original message is preserved
    almost verbatim — only small adjustments to word choice, a brief quip,
    or a greeting tweak are allowed.

    Structured data (code, JSON, tables) passes through untouched.
    Short messages (< 10 chars) are not worth touching.
    """
    if not msg or len(msg.strip()) < 10:
        return msg, Tone.NEUTRAL
    try:
        from arqitect.senses.communication.nerve import (
            _load_personality_traits, _build_personality_instruction, _is_structured_data,
        )
        if _is_structured_data(msg):
            return msg, Tone.NEUTRAL

        traits = _load_personality_traits()
        personality = _build_personality_instruction(traits)

        from arqitect.inference.router import generate_for_role
        _comm = _resolve_adapter("communication") or {}
        _comm_sys = _comm.get("system_prompt", "")
        _comm_temp = get_temperature("communication")
        result = generate_for_role(
            "communication",
            (
                f"Original message:\n{msg}\n\n"
                "Add a LIGHT personality touch. You may:\n"
                "- Adjust a few words for warmth or wit\n"
                "- Add a short quip (max 5 words) before or after\n"
                "- Soften or sharpen phrasing slightly\n"
                "You MUST keep 90%+ of the original text intact.\n"
                "Do NOT replace the message. Do NOT add new information.\n"
                "Output ONLY the lightly adjusted message."
            ),
            system=(
                f"{personality} {_comm_sys} "
                "You are adding a LIGHT personality flavor to an existing message. "
                "The message is already complete and correct. "
                "Make MINIMAL changes — a word here, a quip there. "
                "NEVER replace the message with something different. "
                "NEVER remove information. Keep it almost identical."
            ),
            max_tokens=max(len(msg) + 50, 150),
            temperature=_comm_temp,
        ).strip()

        if not result or len(result) < 5 or result.startswith("Error:"):
            return msg, Tone.NEUTRAL

        # Sanity: if the result diverged too much from original, keep original
        # Simple check: at least 40% of original words should appear in result
        orig_words = set(msg.lower().split())
        result_words = set(result.lower().split())
        if orig_words and len(orig_words & result_words) < len(orig_words) * 0.4:
            print(f"[PERSONALITY] Result diverged too much, keeping original")
            return msg, Tone.NEUTRAL

        # Detect tone from the result (lightweight keyword check, no LLM call)
        result_lower = result.lower()
        if any(w in result_lower for w in ("!", "awesome", "great", "excited")):
            tone = Tone.ENTHUSIASTIC
        elif any(w in result_lower for w in ("hey", "cool", "yeah", "btw")):
            tone = Tone.CASUAL
        else:
            tone = Tone.NEUTRAL

        return result, tone
    except Exception as e:
        print(f"[PERSONALITY] Apply failed: {e}")
    return msg, Tone.NEUTRAL


def _generate_blocked_message(task: str, issue: str) -> str:
    """Generate a personality-flavored message when a response is blocked by guardrails."""
    try:
        from arqitect.inference.router import generate_for_role
        from arqitect.senses.communication.nerve import (
            _load_personality_traits, _build_personality_instruction,
        )
        traits = _load_personality_traits()
        personality = _build_personality_instruction(traits)
        _comm = _resolve_adapter("communication") or {}
        _comm_sys = _comm.get("system_prompt", "")
        _comm_temp = get_temperature("communication")
        _comm_maxt = get_max_tokens("communication")
        result = generate_for_role(
            "communication",
            f"The user asked: \"{(task or 'something')[:80]}\"\nThe response was blocked ({issue}).",
            system=(
                f"{personality} {_comm_sys} "
                "Generate a SHORT (1 sentence) message asking the user to rephrase. "
                "Be friendly and clear. Do NOT mention technical details. "
                "Output ONLY the message."
            ),
            max_tokens=_comm_maxt,
            temperature=_comm_temp,
        ).strip()
        if result and not result.startswith("Error:") and len(result) > 5:
            return result
    except Exception as e:
        print(f"[WARN] _generate_blocked_message: {e}")
    try:
        from arqitect.inference.router import generate_for_role as _gfr
        _fb_comm = _resolve_adapter("communication") or {}
        _fb_sys = _fb_comm.get("system_prompt", "You are a helpful assistant.")
        _fb_temp = get_temperature("communication")
        _fb_maxt = get_max_tokens("communication")
        fallback = _gfr(
            "communication",
            f"The user asked: \"{(task or 'something')[:80]}\"\nGenerate a short, friendly message asking them to rephrase.",
            system=f"{_fb_sys} Output ONLY a single short sentence asking the user to rephrase. Be casual.",
            max_tokens=_fb_maxt,
            temperature=_fb_temp,
        ).strip()
        if fallback and len(fallback) > 5 and not fallback.startswith("Error:"):
            return fallback
    except Exception as e:
        print(f"[WARN] _generate_blocked_message fallback: {e}")
    return "Could you try putting that differently?"


def publish_response(msg: str, nerve_result: dict = None, tone: str = Tone.NEUTRAL, task: str = "", request_identity: bool = False):
    """Publish a rich response envelope to brain:response."""
    # Get current task for echo detection — prefer explicit param, fallback to conversation
    if not task:
        try:
            convo = mem.hot.get_conversation(limit=2)
            user_msgs = [m for m in convo if m.get("role") == "user"]
            if user_msgs:
                task = user_msgs[-1].get("content", "")
        except Exception as e:
            print(f"[WARN] publish_response task lookup: {e}")

    # Validate response quality
    issue = _validate_response(msg, task)
    if issue:
        print(f"[GUARDRAIL] Blocked response ({issue}): {msg[:80]}")
        msg = _generate_blocked_message(task, issue)

    # Safety filter — catch sensitive data leakage and inappropriate content
    media_urls = []
    if nerve_result and isinstance(nerve_result, dict):
        for key in ("gif_url", "image_url", "media_url"):
            if nerve_result.get(key):
                media_urls.append(nerve_result[key])
    is_safe, refusal = _safety_check_output(msg, media_urls)
    if not is_safe:
        print(f"[SAFETY] Output blocked in publish_response: {msg[:80]}")
        msg = refusal
        nerve_result = None  # strip any unsafe media

    from arqitect.brain.invoke import invoke_nerve

    # Apply personality rewriting + tone detection in a single LLM call
    if tone == Tone.NEUTRAL:
        msg, tone = _apply_personality(msg, task=task)

    envelope = build_envelope(message=msg, tone=tone, markdown=True)

    if request_identity:
        envelope["request_identity"] = True

    # Merge any media from nerve results (GIF, card, audio, etc.)
    if nerve_result:
        merge_nerve_result_into_envelope(envelope, nerve_result)

    # Attach task origin so connectors can route responses
    origin = get_task_origin()
    if origin["source"]:
        envelope["source"] = origin["source"]
    if origin["chat_id"]:
        envelope["chat_id"] = origin["chat_id"]

    # Send text response immediately — don't wait for TTS
    publish_event(Channel.BRAIN_RESPONSE, envelope)

    # Generate TTS audio in background thread, publish as separate event
    def _generate_tts():
        try:
            tts_result = invoke_nerve("hearing", json.dumps({"mode": "tts_file", "text": msg}))
            if isinstance(tts_result, str):
                tts_result = json.loads(tts_result)
            audio_path = tts_result.get("audio_path", "") if isinstance(tts_result, dict) else ""
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_data = base64.b64encode(f.read()).decode("utf-8")
                mime = "audio/x-aiff" if audio_path.endswith(".aiff") else "audio/wav"
                audio_payload = {
                    "audio_b64": audio_data,
                    "audio_mime": mime,
                }
                # Attach task origin so connectors can route audio to the right chat
                audio_origin = get_task_origin()
                if audio_origin["source"]:
                    audio_payload["source"] = audio_origin["source"]
                if audio_origin["chat_id"]:
                    audio_payload["chat_id"] = audio_origin["chat_id"]
                publish_event(Channel.BRAIN_AUDIO, audio_payload)
        except Exception as e:
            print(f"[WARN] TTS generation: {e}")
    threading.Thread(target=_generate_tts, daemon=True).start()
