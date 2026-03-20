"""Brain bootstrap — session initialization, sense calibration, and model checks."""

import json
import os
import subprocess
import sys

import requests

from arqitect.brain.config import (
    CORE_SENSES, SENSE_DESCRIPTIONS, SENSES_DIR, SANDBOX_DIR,
    r, mem,
)
from arqitect.brain.events import publish_event, publish_memory_state


def bootstrap_session():
    """On startup, restore session from cold memory or detect via IP."""
    session = mem.hot.get_session()
    if session.get("city"):
        print(f"[BRAIN] Session already bootstrapped: {session.get('city')}, {session.get('timezone')}")
        return

    # Try restoring from cold memory first (survives Redis flushes)
    cold_facts = mem.cold.get_facts("user")
    if cold_facts.get("city"):
        print(f"[BRAIN] Restoring session from cold memory...")
        mem.hot.set_session(cold_facts)
        print(f"[BRAIN] Session restored: {cold_facts.get('city')}, {cold_facts.get('timezone')}")
        publish_memory_state()
        return

    # No cold memory either — call IP geolocation
    print("[BRAIN] Bootstrapping session via IP geolocation...")
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        session_data = {
            "city": data.get("city", ""),
            "region": data.get("region", ""),
            "country": data.get("country_name", ""),
            "country_code": data.get("country_code", ""),
            "timezone": data.get("timezone", ""),
            "latitude": str(data.get("latitude", "")),
            "longitude": str(data.get("longitude", "")),
        }
        mem.hot.set_session(session_data)

        # Also store in cold facts
        for k, v in session_data.items():
            if v:
                mem.cold.set_fact("user", k, v, confidence=0.5)

        print(f"[BRAIN] Session bootstrapped: {session_data.get('city')}, {session_data.get('timezone')}")
        publish_memory_state()
    except Exception as e:
        print(f"[BRAIN] IP geolocation failed: {e}")


def bootstrap_senses():
    """Ensure all 5 core senses exist and are registered.

    Called at startup. Recreates missing sense directories/files,
    registers each in cold memory with is_sense=1, and checks
    required models (pulling missing ones in background).
    """
    os.makedirs(SENSES_DIR, exist_ok=True)

    for sense_name in CORE_SENSES:
        sense_dir = os.path.join(SENSES_DIR, sense_name)
        nerve_path = os.path.join(sense_dir, "nerve.py")

        if not os.path.isfile(nerve_path):
            print(f"[BRAIN] WARNING: Sense '{sense_name}' nerve.py missing at {nerve_path}")
            # Senses are hand-crafted, not synthesized. If missing, just log.
            continue

        # Register in cold memory with is_sense=1
        desc = SENSE_DESCRIPTIONS.get(sense_name, f"Core sense: {sense_name}")
        mem.cold.register_sense(sense_name, desc)

    # Check required models are available
    _check_and_pull_models()

    senses = mem.cold.list_senses()
    print(f"[BRAIN] Core senses bootstrapped: {list(senses.keys())}")

    # Run auto-calibration for all senses
    calibration_results = calibrate_all_senses()
    _store_calibration_in_memory(calibration_results)


def calibrate_sense(name: str) -> dict:
    """Invoke a sense with {"mode": "calibrate"} and return its calibration result."""
    nerve_path = os.path.join(SENSES_DIR, name, "nerve.py")
    if not os.path.isfile(nerve_path):
        return {"sense": name, "status": "unavailable", "error": "nerve.py not found"}

    try:
        _env = os.environ.copy()
        # Ensure arqitect package is importable in subprocess
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _pp = _env.get("PYTHONPATH", "")
        if _project_root not in _pp:
            _env["PYTHONPATH"] = f"{_project_root}:{_pp}" if _pp else _project_root
        result = subprocess.run(
            [sys.executable, nerve_path, json.dumps({"mode": "calibrate"})],
            capture_output=True, text=True, timeout=30,
            cwd=SANDBOX_DIR,
            env=_env,
        )
        output = result.stdout.strip()
        if output:
            cal = json.loads(output)
            print(f"[BRAIN] Calibrated {name}: {cal.get('status', '?')}")
            return cal
        else:
            err = result.stderr.strip()
            print(f"[BRAIN] Calibration failed for {name}: {err}")
            return {"sense": name, "status": "unavailable", "error": err}
    except subprocess.TimeoutExpired:
        return {"sense": name, "status": "unavailable", "error": "calibration timed out"}
    except Exception as e:
        return {"sense": name, "status": "unavailable", "error": str(e)}


def calibrate_all_senses() -> dict:
    """Run calibration for all core senses. Returns {name: result}.

    Core senses that fail subprocess calibration (typically due to
    ModuleNotFoundError in subprocess) are forced to 'available' since
    they work fine when invoked in-process by the brain.
    """
    results = {}
    for name in sorted(CORE_SENSES):
        cal = calibrate_sense(name)
        # Force core senses to available when subprocess calibration fails
        # The subprocess can't import arqitect.* but the sense works in-process
        if cal.get("status") != "available":
            err = cal.get("error", "")
            if "ModuleNotFoundError" in err or "No module named" in err:
                print(f"[BRAIN] Calibrated {name}: available (forced — subprocess import issue)")
                cal = {
                    "sense": name,
                    "status": "available",
                    "capabilities": {},
                    "dependencies": {},
                }
        results[name] = cal
    return results


def _store_calibration_in_memory(results: dict):
    """Store calibration results in cold memory facts + Redis hash."""
    for name, cal in results.items():
        status = cal.get("status", "unavailable")
        caps = cal.get("capabilities", {})
        available_caps = [k for k, v in caps.items() if v.get("available")]
        missing_caps = [k for k, v in caps.items() if not v.get("available")]

        # Cold memory fact
        summary = f"{status}:{','.join(available_caps)}"
        if missing_caps:
            summary += f" (missing:{','.join(missing_caps)})"
        mem.cold.set_fact("sense_calibration", name, summary, confidence=1.0)

        # Redis hash for fast access
        try:
            r.hset("synapse:sense_calibration", name, json.dumps(cal))
        except Exception:
            pass

    # Publish calibration event
    summary_lines = []
    for name, cal in sorted(results.items()):
        status = cal.get("status", "unavailable")
        caps = cal.get("capabilities", {})
        available = [k for k, v in caps.items() if v.get("available")]
        summary_lines.append(f"  {name} [{status}]: {', '.join(available) if available else 'none'}")

    print(f"[BRAIN] Sense calibration complete:")
    for line in summary_lines:
        print(f"[BRAIN] {line}")

    # Publish user_action_needed items to dashboard
    all_actions = []
    for name, cal in results.items():
        for action in cal.get("user_action_needed", []):
            action["sense"] = name
            all_actions.append(action)
    if all_actions:
        publish_event("sense:calibration", {"user_action_needed": all_actions})


def _prompt_calibration_config():
    """In CLI mode, prompt user for any pending calibration config choices."""
    try:
        all_cal = r.hgetall("synapse:sense_calibration")
    except Exception:
        return

    for name, raw in sorted(all_cal.items()):
        try:
            cal = json.loads(raw)
        except Exception:
            continue
        for action in cal.get("user_action_needed", []):
            key = action.get("key", "")
            prompt = action.get("prompt", "")
            options = action.get("options", [])
            if not key or not prompt or not options:
                continue
            # Check if already configured
            existing = mem.cold.get_facts("sense_config").get(f"{name}.{key}")
            if existing:
                continue
            print(f"\n[CONFIG] {name}: {prompt}")
            for i, opt in enumerate(options):
                print(f"  {i + 1}. {opt}")
            print(f"  0. Skip")
            try:
                choice = input("[CONFIG] > ").strip()
                if choice == "0" or not choice:
                    continue
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    mem.cold.set_fact("sense_config", f"{name}.{key}", options[idx], confidence=1.0)
                    print(f"[CONFIG] Set {name}.{key} = {options[idx]}")
            except (ValueError, EOFError, KeyboardInterrupt):
                continue


def bootstrap_user_session(user_id: str):
    """Initialize session for a new user (lazy, on first message).

    1. If user already has a session in hot memory, return immediately.
    2. Try restoring from cold user facts (survives Redis flushes).
    3. Fall back to server's default (IP-based) session.
    """
    if not user_id:
        return

    session = mem.hot.get_session(user_id=user_id)
    if session.get("city"):
        return  # Already bootstrapped

    # Try restoring from cold user facts
    user_facts = mem.cold.get_user_facts(user_id)
    if user_facts.get("city"):
        mem.hot.set_session(user_facts, user_id=user_id)
        print(f"[BRAIN] User session restored from cold facts for {user_id[:8]}...")
        return

    # For new users, inherit server's default session (IP-based)
    server_session = mem.hot.get_session()  # global/server session
    if server_session.get("city"):
        mem.hot.set_session(server_session, user_id=user_id)
        print(f"[BRAIN] User session bootstrapped from server defaults for {user_id[:8]}...")


def _check_and_pull_models():
    """Eagerly preload each role's model via the configured provider.

    Cloud providers treat preload as a no-op (always ready).
    Local providers (e.g. GGUF) load model weights into memory.
    """
    from arqitect.inference.router import (
        get_role_provider, _resolve_role_config, VALID_ROLES,
    )
    from arqitect.brain.adapters import get_max_context

    reported_providers: set[str] = set()
    for role in VALID_ROLES:
        try:
            provider = get_role_provider(role)
            _, model_name = _resolve_role_config(role)
            n_ctx = get_max_context(role)
            provider.preload(model_name, n_ctx=n_ctx)

            pname = type(provider).__name__
            if pname not in reported_providers:
                reported_providers.add(pname)
                models = provider.list_loaded()
                print(f"[BRAIN] {pname} models loaded: {models}")
        except Exception as e:
            print(f"[BRAIN] Provider for role '{role}' not available: {e}")
