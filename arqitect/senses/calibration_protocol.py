"""Calibration Protocol — shared utilities for sense auto-calibration.

Each sense probes its own dependencies and returns a standardized result.
This module provides DRY helpers for those probes.
"""

import json
import os
import platform
import shutil
import time


def check_binary(name: str, install_hint: str = "") -> dict:
    """Check if a binary is available on PATH."""
    path = shutil.which(name)
    return {
        "installed": path is not None,
        "path": path or "",
        "install_hint": install_hint if not path else "",
    }


def check_ollama_model(model: str, url: str = "http://localhost:11434") -> dict:
    """Check if an Ollama model is available."""
    try:
        import requests
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        installed = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        found = model in installed
        return {
            "installed": found,
            "version": "latest" if found else "",
            "install_hint": f"ollama pull {model}" if not found else "",
        }
    except Exception:
        return {
            "installed": False,
            "version": "",
            "install_hint": f"Start Ollama, then: ollama pull {model}",
        }


def check_python_module(name: str, install_hint: str = "") -> dict:
    """Check if a Python module is importable."""
    try:
        __import__(name)
        return {"installed": True, "install_hint": ""}
    except ImportError:
        return {
            "installed": False,
            "install_hint": install_hint or f"pip install {name}",
        }


def derive_status(capabilities: dict) -> str:
    """Derive overall status from capabilities dict.

    All available -> operational, some -> degraded, none -> unavailable.
    """
    if not capabilities:
        return "unavailable"
    available = [c for c in capabilities.values() if c.get("available")]
    if len(available) == len(capabilities):
        return "operational"
    if available:
        return "degraded"
    return "unavailable"


def save_calibration(sense_dir: str, result: dict):
    """Write calibration.json to a sense's directory."""
    path = os.path.join(sense_dir, "calibration.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_calibration(sense_dir: str) -> dict | None:
    """Read cached calibration.json from a sense's directory."""
    path = os.path.join(sense_dir, "calibration.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def build_result(
    sense: str,
    capabilities: dict,
    deps: dict,
    config: dict | None = None,
    user_actions: list | None = None,
    auto_installable: list | None = None,
) -> dict:
    """Build a standardized calibration result."""
    return {
        "sense": sense,
        "timestamp": time.time(),
        "platform": platform.system(),
        "status": derive_status(capabilities),
        "capabilities": capabilities,
        "dependencies": deps,
        "config": config or {},
        "user_action_needed": user_actions or [],
        "auto_installable": auto_installable or [],
    }
