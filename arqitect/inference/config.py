"""Inference configuration — reads inference.conf to determine backend and model sources.

On first run, if inference.conf doesn't exist, guides the user through setup.
The config file is the single source of truth for how inference works.

Supported backends:
  - ollama: Uses Ollama HTTP API (models managed by `ollama pull`)
  - gguf:   Uses llama-cpp-python with local GGUF files (auto-downloaded from HuggingFace)

Config file format (inference.conf):
  [backend]
  type = ollama

  [ollama]
  host = http://localhost:11434

  [models]
  brain = phi4-mini
  nerve = qwen2.5:1.5b
  coder = qwen2.5-coder:1.5b
  creative = llama3.2:3b
  communication = gemma3:1b
  vision = moondream
"""

import configparser
import os
import sys

from arqitect.config.loader import (
    get_project_root,
    get_models_dir as _loader_get_models_dir,
    get_config as _get_yaml_config,
    get_inference_provider,
    get_model_for_role,
)

CONFIG_PATH = os.path.join(str(get_project_root()), "inference.conf")

# Logical role -> default Ollama model name
OLLAMA_DEFAULTS = {
    "brain": "qwen2.5-coder:32b",
    "nerve": "qwen2.5-coder:32b",
    "coder": "qwen2.5-coder:32b",
    "creative": "qwen2.5-coder:32b",
    "communication": "qwen2.5-coder:32b",
    "vision": "moondream",
}

# Logical role -> default GGUF filename + HuggingFace source
# Must stay in sync with inference/model_registry.py MODEL_REGISTRY
GGUF_DEFAULTS = {
    "brain": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "n_ctx": 4096,
    },
    "nerve": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "n_ctx": 4096,
    },
    "coder": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "n_ctx": 4096,
    },
    "creative": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "n_ctx": 4096,
    },
    "communication": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "n_ctx": 4096,
    },
    "vision": {
        "file": "moondream2-text-model-f16.gguf",
        "source": "moondream/moondream2-gguf",
        "n_ctx": 2048,
        "chat_handler": "moondream",
    },
}

REQUIRED_ROLES = ["brain", "nerve", "coder", "creative", "communication", "vision"]


def load_config() -> configparser.ConfigParser:
    """Load inference.conf. Returns parsed config or None if missing."""
    if not os.path.exists(CONFIG_PATH):
        return None
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config


def get_backend_type() -> str:
    """Return the configured backend type from arqitect.yaml or inference.conf."""
    # Check arqitect.yaml first
    yaml_provider = get_inference_provider()
    if yaml_provider and yaml_provider != "gguf":
        return yaml_provider
    # Fall back to inference.conf
    config = load_config()
    if config is None:
        return yaml_provider or ""
    return config.get("backend", "type", fallback=yaml_provider or "")


def get_ollama_host() -> str:
    config = load_config()
    if config is None:
        return "http://localhost:11434"
    return config.get("ollama", "host", fallback="http://localhost:11434")


def get_model_name(role: str) -> str:
    """Get the configured model name/path for a logical role.

    Priority: arqitect.yaml (via loader) > inference.conf > hardcoded defaults.
    """
    # Check arqitect.yaml first (the primary config source)
    yaml_model = get_model_for_role(role)
    if yaml_model:
        return yaml_model
    # Fall back to legacy inference.conf
    config = load_config()
    if config is None:
        return OLLAMA_DEFAULTS.get(role, role)
    return config.get("models", role, fallback=OLLAMA_DEFAULTS.get(role, role))


def get_models_dir() -> str:
    config = load_config()
    if config is None:
        return _loader_get_models_dir()
    return config.get("gguf", "models_dir",
                       fallback=_loader_get_models_dir())


def write_default_config(backend: str):
    """Write a default inference.conf for the chosen backend."""
    config = configparser.ConfigParser()
    config["backend"] = {"type": backend}

    if backend == "ollama":
        config["ollama"] = {"host": "http://localhost:11434"}
        config["models"] = dict(OLLAMA_DEFAULTS)
    elif backend == "gguf":
        config["gguf"] = {"models_dir": get_models_dir()}
        config["models"] = {role: entry["file"] for role, entry in GGUF_DEFAULTS.items()}

    with open(CONFIG_PATH, "w") as f:
        f.write("# Arqitect Inference Configuration\n")
        f.write("# Generated automatically — edit to customize.\n")
        f.write("#\n")
        if backend == "ollama":
            f.write("# Backend: ollama — uses Ollama HTTP API\n")
            f.write("# Install: https://ollama.com\n")
            f.write("# Pull models: ollama pull phi4-mini && ollama pull qwen2.5:1.5b ...\n")
        elif backend == "gguf":
            f.write("# Backend: gguf — uses llama-cpp-python with local GGUF files\n")
            f.write("# Models auto-download from HuggingFace on first use.\n")
            f.write("# pip install llama-cpp-python huggingface_hub\n")
        f.write("#\n\n")
        config.write(f)

    print(f"[CONFIG] Written {CONFIG_PATH}")


def check_ollama_ready() -> tuple[bool, list[str]]:
    """Check if Ollama is running and which required models are available.

    Returns (all_ready, missing_models).
    """
    import requests
    config = load_config()

    host = get_ollama_host()
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception:
        return False, list(REQUIRED_ROLES)

    available = set()
    for m in resp.json().get("models", []):
        name = m.get("name", "")
        # Normalize: "phi4-mini:latest" -> "phi4-mini"
        base = name.split(":")[0] if ":" in name else name
        available.add(name)
        available.add(base)

    missing = []
    for role in REQUIRED_ROLES:
        model_name = get_model_name(role)
        base = model_name.split(":")[0]
        if model_name not in available and base not in available:
            missing.append(f"{role} ({model_name})")

    return len(missing) == 0, missing


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
            source = GGUF_DEFAULTS.get(role, {}).get("source", "?")
            missing.append(f"{role} ({model_file}) from {source}")
    return len(missing) == 0, missing


def print_setup_guide():
    """Print first-run setup instructions to stderr and exit."""
    print("""
╔══════════════════════════════════════════════════════════╗
║              SENTIENT — First Run Setup                  ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  No inference backend configured.                        ║
║  Create inference.conf to get started.                   ║
║                                                          ║
║  Option 1: Ollama (recommended for local dev)            ║
║  ─────────────────────────────────────────               ║
║  1. Install Ollama: https://ollama.com                   ║
║  2. Pull required models:                                ║
║       ollama pull phi4-mini                               ║
║       ollama pull qwen2.5:1.5b                            ║
║       ollama pull qwen2.5-coder:1.5b                      ║
║       ollama pull llama3.2:3b                              ║
║       ollama pull gemma3:1b                                ║
║       ollama pull moondream                                ║
║  3. Generate config:                                     ║
║       python inference/setup.py ollama                    ║
║                                                          ║
║  Option 2: Local GGUF files (no Ollama needed)           ║
║  ─────────────────────────────────────────               ║
║  1. pip install llama-cpp-python huggingface_hub          ║
║  2. Generate config:                                     ║
║       python inference/setup.py gguf                      ║
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
    print(f"", file=sys.stderr)
    for m in missing:
        print(f"  ✗ {m}", file=sys.stderr)
    print(f"", file=sys.stderr)

    if backend == "ollama":
        print(f"  Fix: pull the missing models:", file=sys.stderr)
        for m in missing:
            # Extract the model name from "role (model_name)"
            model = m.split("(")[1].rstrip(")") if "(" in m else m
            print(f"    ollama pull {model}", file=sys.stderr)
    elif backend == "gguf":
        models_dir = get_models_dir()
        print(f"  Fix: download the missing GGUF files to {models_dir}/", file=sys.stderr)
        print(f"  Or run: python inference/setup.py gguf --download", file=sys.stderr)

    print(f"", file=sys.stderr)
