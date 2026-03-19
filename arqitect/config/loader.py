"""Project root finder and configuration loader.

Replaces all __file__/sys.path hacks with a single source of truth for paths.
Parses arqitect.yaml for all configuration, falling back to sensible defaults.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml

from arqitect.config.defaults import DEFAULTS


def find_project_root() -> Path:
    """Find the arqitect project root by walking up from cwd.

    Checks for arqitect.yaml first, then falls back to ARQITECT_PROJECT_ROOT env,
    then walks up from cwd looking for common project markers.
    """
    # Env override
    env_root = os.environ.get("ARQITECT_PROJECT_ROOT")
    if env_root:
        p = Path(env_root)
        if p.is_dir():
            return p

    # Walk up from cwd looking for arqitect.yaml
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "arqitect.yaml").exists():
            return parent
        # Also check for legacy inference.conf (old projects)
        if (parent / "inference.conf").exists():
            return parent

    # Fallback: cwd itself
    return cwd


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Cached version of find_project_root."""
    return find_project_root()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load arqitect.yaml from project root, merged with defaults.

    Falls back to DEFAULTS if no arqitect.yaml exists.
    Also reads legacy inference.conf if present and no arqitect.yaml.
    """
    root = get_project_root()
    yaml_path = root / "arqitect.yaml"

    if yaml_path.exists():
        with open(yaml_path) as f:
            user_config = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULTS, user_config)

    # Legacy: try reading inference.conf
    conf_path = root / "inference.conf"
    if conf_path.exists():
        return _load_legacy_config(conf_path)

    import copy
    return copy.deepcopy(DEFAULTS)


def _load_legacy_config(conf_path: Path) -> dict:
    """Read legacy inference.conf and map to the new config structure."""
    import configparser
    import copy
    cp = configparser.ConfigParser()
    cp.read(str(conf_path))

    config = copy.deepcopy(DEFAULTS)

    # Map backend type
    backend = cp.get("backend", "type", fallback="gguf")
    config["inference"] = dict(config.get("inference", {}))
    config["inference"]["provider"] = backend

    # Map models
    if cp.has_section("models"):
        config["inference"]["models"] = dict(cp.items("models"))

    # Map gguf models_dir
    if cp.has_section("gguf"):
        models_dir = cp.get("gguf", "models_dir", fallback="")
        if models_dir:
            config["inference"]["models_dir"] = models_dir

    # Map ollama host
    if cp.has_section("ollama"):
        config["inference"]["ollama_host"] = cp.get("ollama", "host", fallback="http://localhost:11434")

    return config


# ── Config accessors ─────────────────────────────────────────────────────

def get_config(path: str, default: Any = None) -> Any:
    """Get a nested config value by dot-separated path.

    Example: get_config("inference.provider") -> "gguf"
    """
    config = load_config()
    keys = path.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# ── Path accessors ───────────────────────────────────────────────────────

def get_nerves_dir() -> str:
    """Directory where synthesized nerves live."""
    return get_config("paths.nerves", str(get_project_root() / "nerves"))


def get_senses_dir() -> str:
    """Directory where core senses live (inside the arqitect package)."""
    return str(Path(__file__).parent.parent / "senses")


def get_mcp_tools_dir() -> str:
    """Directory where MCP tool plugins live."""
    return get_config("paths.mcp_tools", str(get_project_root() / "mcp_tools"))


def get_sandbox_dir() -> str:
    """Sandbox directory for file operations."""
    return get_config("paths.sandbox", str(get_project_root() / "sandbox"))


def get_memory_dir() -> str:
    """Directory for SQLite databases."""
    return get_config("paths.memory", str(get_project_root()))


def get_models_dir() -> str:
    """Directory for GGUF model files."""
    return get_config("inference.models_dir", str(get_project_root() / "models"))



def get_whatsapp_dir() -> str:
    """Directory containing WhatsApp connector."""
    return str(Path(__file__).parent.parent / "connectors" / "whatsapp")


def get_telegram_dir() -> str:
    """Directory containing Telegram connector."""
    return str(Path(__file__).parent.parent / "connectors" / "telegram")


# ── Secret accessors ─────────────────────────────────────────────────────

def get_secret(path: str, default: Any = "") -> Any:
    """Get a secret value by dot-separated path under the 'secrets' section.

    Example: get_secret("jwt_secret") -> "abc123..."
    """
    return get_config(f"secrets.{path}", default)


def get_connector_config(connector: str, key: str, default: Any = None) -> Any:
    """Get a connector config value.

    Example: get_connector_config("telegram", "bot_name") -> "Arqitect"
    """
    return get_config(f"connectors.{connector}.{key}", default)


# ── Storage accessors ────────────────────────────────────────────────────

def get_redis_url() -> str:
    """Redis URL from arqitect.yaml config."""
    return get_config("storage.hot.url", "redis://localhost:6379")


def get_redis_host_port() -> tuple[str, int]:
    """Parse Redis host and port from URL."""
    url = get_redis_url()
    url = url.replace("redis://", "")
    if ":" in url:
        host, port_str = url.split(":", 1)
        port_str = port_str.split("/")[0]
        return host, int(port_str)
    return url.split("/")[0], 6379


def get_redis_client():
    """Create a configured Redis client from arqitect.yaml / env."""
    import redis
    host, port = get_redis_host_port()
    return redis.Redis(host=host, port=port, decode_responses=True)


def get_cold_db_path() -> str:
    """Path to the cold memory SQLite database."""
    return os.path.join(
        get_config("paths.memory", str(get_project_root())),
        get_config("storage.cold.path", "arqitect_memory.db"),
    )


def get_warm_db_path() -> str:
    """Path to the warm memory SQLite database."""
    return os.path.join(
        get_config("paths.memory", str(get_project_root())),
        get_config("storage.warm.path", "episodes.db"),
    )


# ── Port accessors ───────────────────────────────────────────────────────

def get_mcp_port() -> int:
    """MCP server port."""
    return int(get_config("ports.mcp", 8100))


def get_mcp_url() -> str:
    """Full MCP server URL built from config."""
    host = get_config("ports.mcp_host", "127.0.0.1")
    port = get_mcp_port()
    return f"http://{host}:{port}"


def get_ssl_context():
    """Build an SSL context from config cert/key paths, or return None."""
    import ssl
    cert = get_config("ssl.cert", "")
    key = get_config("ssl.key", "")
    if not cert or not key:
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return ctx


def get_ssl_paths() -> tuple[str, str]:
    """Return (cert_path, key_path) from config. Both empty if not set."""
    return get_config("ssl.cert", ""), get_config("ssl.key", "")


def get_bridge_port() -> int:
    """WebSocket bridge port."""
    return int(get_config("ports.bridge", get_config("ports.dashboard", 3000)))


# ── Inference accessors ──────────────────────────────────────────────────

def get_inference_provider() -> str:
    """Primary inference provider name."""
    return get_config("inference.provider", "gguf")


def get_model_for_role(role: str) -> str:
    """Get the model name for a given role (brain, nerve, coder, creative, etc.)."""
    return get_config(f"inference.models.{role}", get_config("inference.models.brain", ""))


def get_per_role_provider(role: str) -> str | None:
    """Get provider override for a specific role, or None for default."""
    return get_config(f"inference.roles.{role}.provider")


def get_per_role_model(role: str) -> str | None:
    """Get model override for a specific role, or None for default."""
    return get_config(f"inference.roles.{role}.model")
