"""Brain nerve invocation — running nerve scripts and processing their output."""

import json
import logging
import os
import re
import subprocess
import sys

from arqitect.brain.config import CORE_SENSES, NERVES_DIR, SENSES_DIR, SANDBOX_DIR, mem
from arqitect.brain.types import RedisKey

logger = logging.getLogger(__name__)

NERVE_EXECUTION_TIMEOUT_SECONDS = 90
STDERR_TRUNCATION_LENGTH = 500

EXTENSION_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


_NERVE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")

_monitoring = None


def _record_nerve_usage(name: str, success: bool, latency_ms: float = 0,
                        error_message: str = ""):
    """Record nerve invocation in monitoring database. Best-effort, never blocks."""
    global _monitoring
    try:
        if _monitoring is None:
            from arqitect.memory.monitoring import MonitoringMemory
            _monitoring = MonitoringMemory()
        _monitoring.record_call("nerve", name, success,
                                latency_ms=latency_ms, error_message=error_message)
    except Exception:
        logger.debug("Failed to record nerve usage for '%s'", name, exc_info=True)


def _sanitize_nerve_name(name: str) -> str | None:
    """Validate and sanitize a nerve name. Returns None if invalid."""
    name = name.removesuffix(".py").strip()
    if not name or not _NERVE_NAME_PATTERN.match(name):
        return None
    return name


def invoke_nerve(name: str, args: str, user_id: str = "") -> str:
    """Run a nerve script in the sandbox and return its output.

    Passes memory context via env vars.
    """
    name = _sanitize_nerve_name(name)
    if name is None:
        return json.dumps({"error": "Invalid nerve name — must contain only alphanumeric, underscore, or hyphen characters"})

    # Strip null bytes from args to prevent subprocess crashes
    args = args.replace("\x00", "")

    # Core senses live under nerves/senses/{name}/
    if name in CORE_SENSES:
        nerve_path = os.path.join(SENSES_DIR, name, "nerve.py")
    else:
        nerve_path = os.path.join(NERVES_DIR, name, "nerve.py")
    if not os.path.exists(nerve_path):
        return json.dumps({"error": f"Nerve '{name}' not found"})

    # Warn if invoking an unqualified nerve (skip core senses — they don't need qualification)
    if name not in CORE_SENSES and not mem.cold.is_qualified("nerve", name):
        logger.warning("[BRAIN] Invoking unqualified nerve '%s'", name)

    # Build env with memory context
    env = os.environ.copy()
    env.update(mem.get_env_for_nerve(name, args, user_id=user_id))
    env["SYNAPSE_USER_ID"] = user_id
    # Lazy-load only the model the nerve needs (not all 6)
    env["SYNAPSE_LAZY_LOAD"] = "1"
    # Ensure arqitect package is importable in subprocess
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _pythonpath = env.get("PYTHONPATH", "")
    if _project_root not in _pythonpath:
        env["PYTHONPATH"] = f"{_project_root}:{_pythonpath}" if _pythonpath else _project_root

    import time as _time
    t0 = _time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, nerve_path, args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=NERVE_EXECUTION_TIMEOUT_SECONDS,
            cwd=SANDBOX_DIR,
            env=env,
        )
        latency = (_time.monotonic() - t0) * 1000
        # Log nerve debug output (stderr) for visibility
        if result.stderr:
            for line in result.stderr.splitlines():
                if line.startswith("[NERVE:"):
                    logger.debug("%s", line)
        _record_nerve_usage(name, success=(result.returncode == 0), latency_ms=latency)
        return result.stdout or result.stderr
    except subprocess.TimeoutExpired as e:
        latency = (_time.monotonic() - t0) * 1000
        _record_nerve_usage(name, success=False, latency_ms=latency,
                            error_message=f"Timed out (>{NERVE_EXECUTION_TIMEOUT_SECONDS}s)")
        stderr = ""
        if hasattr(e, 'stderr') and e.stderr:
            stderr = e.stderr[:STDERR_TRUNCATION_LENGTH]
        return json.dumps({"error": f"Nerve '{name}' timed out (>{NERVE_EXECUTION_TIMEOUT_SECONDS}s)", "stderr": stderr})


def _enrich_nerve_result_with_image(nerve_result: dict) -> dict:
    """Ensure image_mime is set when image_path is present."""
    if not isinstance(nerve_result, dict):
        return nerve_result
    image_path = nerve_result.get("image_path", "")
    if image_path and not nerve_result.get("image_mime"):
        ext = os.path.splitext(image_path)[1].lower()
        nerve_result["image_mime"] = EXTENSION_TO_MIME.get(ext, "image/png")
    return nerve_result


def _translate_sense_args(sense_name: str, raw_args: str, task: str) -> str:
    """Use LLM to translate natural language args into structured JSON for a sense."""
    from arqitect.brain.config import r, BRAIN_MODEL
    from arqitect.brain.helpers import llm_generate

    # If raw_args is already well-structured JSON, pass through directly
    try:
        parsed = json.loads(raw_args)
        if isinstance(parsed, dict) and len(parsed) > 0:
            return raw_args
    except (json.JSONDecodeError, TypeError) as exc:
        logger.debug("Sense '%s' raw_args not valid JSON, will translate: %s", sense_name, exc)

    # Get sense capabilities from calibration
    try:
        cal_raw = r.hget(RedisKey.SENSE_CALIBRATION, sense_name)
        if cal_raw:
            cal = json.loads(cal_raw)
            caps = cal.get("capabilities", {})
            available = [k for k, v in caps.items() if v.get("available")]
        else:
            available = []
    except Exception:
        available = []

    if not available:
        return raw_args

    prompt = (
        f"Sense: {sense_name}\n"
        f"Available modes: {', '.join(available)}\n"
        f"User request: {task}\n"
        f"Current args: {raw_args}\n\n"
        f"Extract the correct JSON arguments for this sense. "
        f"Pick the mode that best matches the user's intent. "
        f"Include any relevant parameters (file paths, text, language, duration, etc).\n"
        f"Output ONLY a valid JSON object, nothing else."
    )

    try:
        result = llm_generate(
            BRAIN_MODEL, prompt,
            system="You extract structured parameters from natural language. Output only valid JSON."
        )
        # Validate it's JSON
        parsed = json.loads(result.strip())
        return json.dumps(parsed)
    except Exception as e:
        logger.warning("[BRAIN] Sense arg translation failed for '%s': %s", sense_name, e)
        return raw_args
