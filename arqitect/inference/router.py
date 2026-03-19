"""Per-role inference router — central entry point for all LLM text generation.

Each role (brain, nerve, coder, creative, communication) can independently
use any provider (gguf, anthropic, groq, ollama, openai). Configuration lives
in ``inference.roles.<role>.provider`` and ``inference.roles.<role>.model``
in arqitect.yaml.

Backward compatible: falls back to flat ``inference.provider`` / ``inference.models.*``
when per-role config is not set.

No silent fallback: if a role is configured for a cloud provider, it uses that
provider or raises. API keys are validated at provider instantiation time.
"""

import threading
from typing import Optional

from arqitect.config.loader import (
    get_config,
    get_inference_provider,
    get_model_for_role as _get_flat_model,
    get_models_dir,
    get_per_role_model,
    get_per_role_provider,
    get_secret,
)
from arqitect.inference.providers import get_provider, PROVIDER_META, PROVIDER_REGISTRY
from arqitect.inference.providers.base import InferenceProvider

# Valid role names that the router recognises.
VALID_ROLES = frozenset({"brain", "nerve", "coder", "creative", "communication"})

# ── Provider cache ────────────────────────────────────────────────────────

_provider_cache: dict[str, InferenceProvider] = {}
_cache_lock = threading.Lock()


def _validate_api_key(provider_name: str) -> None:
    """Raise ``ValueError`` if a cloud provider's API key is missing.

    Reads auth requirements from PROVIDER_META — no hardcoded key map.
    """
    meta = PROVIDER_META.get(provider_name, {})
    if meta.get("auth_type") != "api_key":
        return  # local providers don't need keys
    secret_path = meta["secret_key"]
    key = get_secret(secret_path) or ""
    if not key.strip():
        raise ValueError(
            f"Provider '{provider_name}' requires secret '{secret_path}' "
            f"but it is empty. Set it in arqitect.yaml → secrets.{secret_path}"
        )


def _build_provider_kwargs(provider_name: str) -> dict:
    """Build kwargs for instantiating a provider, validating keys first.

    Reads auth and extra config from PROVIDER_META.  Provider-specific
    constructor kwargs (models_dir, host) are resolved from config.
    """
    _validate_api_key(provider_name)

    meta = PROVIDER_META.get(provider_name, {})
    kwargs: dict = {}

    # Pass API key if the provider requires one
    if meta.get("secret_key"):
        kwargs["api_key"] = get_secret(meta["secret_key"])

    # Pass extra config values (e.g. base_url, host) from arqitect.yaml
    for field in meta.get("extra_config", []):
        store_in = field["store_in"]
        key = field["key"]
        if store_in == "secrets":
            value = get_secret(key) or field.get("default", "")
        else:
            value = get_config(f"{store_in}.{key}", field.get("default", ""))
        if value:
            # Map config key to constructor kwarg name
            kwarg_name = _config_key_to_kwarg(key)
            kwargs[kwarg_name] = value

    # GGUF needs models_dir, Ollama needs host — both from inference config
    if provider_name == "gguf":
        kwargs["models_dir"] = get_models_dir()
    elif provider_name == "ollama" and "host" not in kwargs:
        kwargs["host"] = get_config("inference.ollama_host", "http://localhost:11434")

    return kwargs


def _config_key_to_kwarg(key: str) -> str:
    """Map a config key name to the provider constructor kwarg name.

    Examples: openai_base_url → base_url, ollama_host → host.
    """
    # Strip known provider prefixes to get the constructor parameter name
    prefixes = ("openai_", "ollama_", "azure_openai_")
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _get_or_create_provider(provider_name: str) -> InferenceProvider:
    """Return a cached provider instance, creating one if necessary.

    Thread-safe. One instance per provider name is shared across all roles.
    """
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    with _cache_lock:
        # Double-check after acquiring lock
        if provider_name in _provider_cache:
            return _provider_cache[provider_name]

        kwargs = _build_provider_kwargs(provider_name)
        instance = get_provider(provider_name, **kwargs)
        _provider_cache[provider_name] = instance
        return instance


# ── Role config resolution ────────────────────────────────────────────────

def _resolve_role_config(role: str) -> tuple[str, str]:
    """Resolve (provider_name, model_name) for a role.

    Resolution order:
    1. ``inference.roles.<role>.provider`` / ``.model`` (per-role config)
    2. ``inference.provider`` / ``inference.models.<role>`` (flat compat)
    3. Raise ``ValueError`` if nothing is configured.

    Returns:
        Tuple of (provider_name, model_name).
    """
    # 1. Per-role config
    role_provider = get_per_role_provider(role)
    role_model = get_per_role_model(role)

    if role_provider:
        if role_provider not in PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown provider '{role_provider}' for role '{role}'. "
                f"Available: {list(PROVIDER_REGISTRY.keys())}"
            )
        # Model: per-role model, or fall back to flat model, or role name itself
        model = role_model or _get_flat_model(role) or role
        return role_provider, model

    # 2. Flat backward-compat config
    flat_provider = get_inference_provider()
    if flat_provider:
        model = _get_flat_model(role) or role
        return flat_provider, model

    # 3. Nothing configured
    raise ValueError(
        f"No inference provider configured for role '{role}'. "
        "Set inference.roles.<role>.provider or inference.provider in arqitect.yaml"
    )


def get_role_provider(role: str) -> InferenceProvider:
    """Return the provider instance for a given role.

    Args:
        role: One of brain, nerve, coder, creative, communication.

    Returns:
        Cached ``InferenceProvider`` instance.

    Raises:
        ValueError: If the role has no provider configured or the API key is missing.
    """
    provider_name, _ = _resolve_role_config(role)
    return _get_or_create_provider(provider_name)


# ── Public API ────────────────────────────────────────────────────────────

def generate_for_role(
    role: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    json_mode: bool = False,
    lora_path: Optional[str] = None,
) -> str:
    """Generate text for a specific role using its configured provider.

    This is the single entry point that replaces all ``get_engine().generate()``
    calls for text generation.

    Args:
        role: Logical role name (brain, nerve, coder, creative, communication).
        prompt: User prompt text.
        system: System prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        json_mode: Request JSON output format.
        lora_path: Path to a LoRA adapter (only used by GGUF provider).

    Returns:
        Generated text string.
    """
    provider_name, model = _resolve_role_config(role)
    provider = _get_or_create_provider(provider_name)

    kwargs: dict = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "json_mode": json_mode,
    }

    # LoRA adapters are only supported by GGUF
    if lora_path and provider.supports_lora:
        kwargs["lora_path"] = lora_path

    return provider.generate(**kwargs)


def reset_cache() -> None:
    """Clear the provider cache. Useful for testing or config reload."""
    with _cache_lock:
        _provider_cache.clear()
