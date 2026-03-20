"""Inference configuration — delegates to arqitect.yaml via config.loader.

Provides convenience accessors for model readiness checks and setup guidance.
"""

import os
import sys

from arqitect.config.loader import (
    get_models_dir as _loader_get_models_dir,
    get_inference_provider,
    get_model_for_role,
)
from arqitect.inference.model_registry import MODEL_REGISTRY
from arqitect.types import InferenceRole

REQUIRED_ROLES = [*InferenceRole, "vision"]


def get_backend_type() -> str:
    """Return the configured inference provider from arqitect.yaml."""
    return get_inference_provider()


def get_model_name(role: str) -> str:
    """Get the configured model name/path for a logical role."""
    return get_model_for_role(role) or role


def get_models_dir() -> str:
    """Get the GGUF models directory from arqitect.yaml."""
    return _loader_get_models_dir()


def check_gguf_ready() -> tuple[bool, list[str]]:
    """Check which GGUF model files are present.

    Returns (all_ready, missing_models).
    """
    models_dir = get_models_dir()
    missing = []
    for role in REQUIRED_ROLES:
        model_file = get_model_name(role)
        path = os.path.join(models_dir, model_file)
        if not os.path.exists(path):
            source = MODEL_REGISTRY.get(role, {}).get("source", "?")
            missing.append(f"{role} ({model_file}) from {source}")
    return len(missing) == 0, missing


def print_setup_guide():
    """Print first-run setup instructions to stderr."""
    print("""
╔══════════════════════════════════════════════════════════╗
║              ARQITECT — First Run Setup                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  No inference models found.                              ║
║  Configure arqitect.yaml to get started.                 ║
║                                                          ║
║  1. pip install llama-cpp-python huggingface_hub          ║
║  2. Run the setup wizard:                                ║
║       python -m arqitect.cli.main setup                  ║
║  3. Models auto-download on first use (~8GB total)       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""", file=sys.stderr)


def print_status_report(backend: str, ready: bool, missing: list[str]):
    """Print a clear status report for the configured backend."""
    if ready:
        print(f"[INFERENCE] Backend: {backend} — all models ready", file=sys.stderr)
        return

    print(f"\n[INFERENCE] Backend: {backend} — missing models detected!", file=sys.stderr)
    print("", file=sys.stderr)
    for m in missing:
        print(f"  ✗ {m}", file=sys.stderr)
    print("", file=sys.stderr)

    models_dir = get_models_dir()
    print(f"  Fix: download the missing GGUF files to {models_dir}/", file=sys.stderr)
    print(f"  Or run: python -m arqitect.cli.main setup", file=sys.stderr)
    print("", file=sys.stderr)
