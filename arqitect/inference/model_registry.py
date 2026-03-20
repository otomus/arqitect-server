"""Model Registry — built from arqitect.yaml, zero hardcoded model knowledge.

The registry maps logical role names to model config (file, source, handler).
All model information comes from ``inference.models`` in arqitect.yaml.
Defaults live in ``arqitect/config/defaults.py`` — not here.

Context window (n_ctx) is NOT stored here — it comes from the community
adapter meta.json → capabilities.max_context, the single source of truth.
"""

# ── Named constants ──────────────────────────────────────────────────────
# Handler/backend identifiers used for comparison — not model opinions.

CHAT_HANDLER_MOONDREAM = "moondream"
BACKEND_STABLE_DIFFUSION = "stable_diffusion"

# ── Registry roles ───────────────────────────────────────────────────────

_REGISTRY_ROLES = (
    "brain", "nerve", "coder", "creative", "communication",
    "vision", "embedding", "image_gen",
)

# ── Lazy-built registry ──────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, dict] | None = None


def _build_registry() -> dict[str, dict]:
    """Build MODEL_REGISTRY from arqitect.yaml config.

    Reads ``inference.models.<role>`` for each role. Each entry can be
    a string (filename only) or a dict (file, source, chat_handler, etc.).
    """
    from arqitect.config.loader import get_model_config

    registry: dict[str, dict] = {}
    for role in _REGISTRY_ROLES:
        config = get_model_config(role)
        if config and config.get("file"):
            registry[role] = config
    return registry


def _get_registry() -> dict[str, dict]:
    """Get or build the model registry (lazy singleton)."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        _MODEL_REGISTRY = _build_registry()
    return _MODEL_REGISTRY


class _RegistryProxy(dict):
    """Dict-like proxy that builds the registry on first access."""

    def _ensure_built(self):
        if not super().__len__():
            super().update(_get_registry())

    def __getitem__(self, key):
        self._ensure_built()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_built()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._ensure_built()
        return super().get(key, default)

    def keys(self):
        self._ensure_built()
        return super().keys()

    def values(self):
        self._ensure_built()
        return super().values()

    def items(self):
        self._ensure_built()
        return super().items()

    def __iter__(self):
        self._ensure_built()
        return super().__iter__()

    def __len__(self):
        self._ensure_built()
        return super().__len__()

    def __repr__(self):
        self._ensure_built()
        return super().__repr__()


MODEL_REGISTRY = _RegistryProxy()

# Alias roles that differ between subsystems (adapters use "code",
# inference/config use "coder").
ROLE_TO_REGISTRY_KEY: dict[str, str] = {
    "tool": "nerve",
    "code": "coder",
    "awareness": "brain",
    "scheduler": "nerve",
    "generative": "creative",
}


def resolve_registry_key(role: str) -> str:
    """Translate a runtime role name to its MODEL_REGISTRY key.

    Returns the role unchanged when no alias exists.
    """
    return ROLE_TO_REGISTRY_KEY.get(role, role)


def find_registry_entry_by_file(filename: str) -> dict | None:
    """Find a MODEL_REGISTRY entry whose 'file' field matches a filename.

    Args:
        filename: GGUF filename to match (e.g. 'model-file.gguf').

    Returns:
        The first matching registry entry, or None.
    """
    for entry in MODEL_REGISTRY.values():
        if entry.get("file") == filename:
            return entry
    return None


def resolve_model_path(name: str, models_dir: str) -> str | None:
    """Resolve a model name to a GGUF file path.

    Checks absolute paths first, then looks for the filename in models_dir.

    Args:
        name: Model name — can be an absolute path or a filename.
        models_dir: Directory containing GGUF model files.

    Returns:
        Resolved file path, or None if not found.
    """
    import os

    if os.path.isabs(name) and os.path.exists(name):
        return name
    candidate = os.path.join(models_dir, name)
    if os.path.exists(candidate):
        return candidate
    return None
