"""Inference providers — pluggable backends for LLM inference.

Provider metadata follows the MCP meta.json pattern from model-integration-recommendations.
Each entry declares its auth requirements, default model, and extra config fields so that
the wizard and router can be fully data-driven — no hardcoded per-provider branches.
"""

from arqitect.inference.providers.base import InferenceProvider

# ── Provider metadata (MCP meta.json pattern) ────────────────────────────
#
# Fields:
#   label          — Human-friendly name shown in wizard menus.
#   category       — "local" or "cloud"; controls wizard ordering.
#   auth_type      — "none" or "api_key".
#   secret_key     — Config path under ``secrets.*`` (None for local providers).
#   secret_label   — Human-friendly name for the API key prompt.
#   default_model  — Default model name (None triggers file picker for GGUF).
#   model_prompt   — "file_picker" (GGUF) or "text" (all others).
#   extra_config   — Additional config fields the provider needs.
#     key          — Config key name.
#     label        — Prompt label for wizard.
#     default      — Default value.
#     store_in     — "secrets" or "inference" — where in arqitect.yaml to store.
#     before_model — If True, prompt *before* model name (e.g. base_url).

PROVIDER_META: dict[str, dict] = {
    # ── Local providers ──────────────────────────────────────────────────
    "gguf": {
        "label": "GGUF (local)",
        "category": "local",
        "auth_type": "none",
        "secret_key": None,
        "secret_label": None,
        "default_model": None,
        "model_prompt": "file_picker",
        "extra_config": [],
    },
    "ollama": {
        "label": "Ollama (local)",
        "category": "local",
        "auth_type": "none",
        "secret_key": None,
        "secret_label": None,
        "default_model": "qwen2.5-coder:7b",
        "model_prompt": "text",
        "extra_config": [
            {
                "key": "ollama_host",
                "label": "Ollama host",
                "default": "http://localhost:11434",
                "store_in": "inference",
                "before_model": False,
            },
        ],
    },
    # ── Cloud providers (existing) ───────────────────────────────────────
    "anthropic": {
        "label": "Anthropic (cloud)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "anthropic_api_key",
        "secret_label": "Anthropic API key",
        "default_model": "claude-sonnet-4-20250514",
        "model_prompt": "text",
        "extra_config": [],
    },
    "openai": {
        "label": "OpenAI-compatible (cloud)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "openai_api_key",
        "secret_label": "OpenAI API key",
        "default_model": "gpt-4o",
        "model_prompt": "text",
        "extra_config": [
            {
                "key": "openai_base_url",
                "label": "API base URL",
                "default": "https://api.openai.com/v1",
                "store_in": "secrets",
                "before_model": True,
            },
        ],
    },
    "groq": {
        "label": "Groq (cloud)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "groq_api_key",
        "secret_label": "Groq API key",
        "default_model": "llama-3.3-70b-versatile",
        "model_prompt": "text",
        "extra_config": [],
    },
    "deepseek": {
        "label": "DeepSeek (cloud)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "deepseek_api_key",
        "secret_label": "DeepSeek API key",
        "default_model": "deepseek-chat",
        "model_prompt": "text",
        "extra_config": [],
    },
    "mistral": {
        "label": "Mistral (cloud)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "mistral_api_key",
        "secret_label": "Mistral API key",
        "default_model": "mistral-large-latest",
        "model_prompt": "text",
        "extra_config": [],
    },
    "openrouter": {
        "label": "OpenRouter (gateway)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "openrouter_api_key",
        "secret_label": "OpenRouter API key",
        "default_model": "anthropic/claude-sonnet-4-20250514",
        "model_prompt": "text",
        "extra_config": [],
    },
    "google_gemini": {
        "label": "Google Gemini (cloud)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "google_ai_api_key",
        "secret_label": "Google AI API key",
        "default_model": "gemini-2.0-flash",
        "model_prompt": "text",
        "extra_config": [],
    },
    "xai": {
        "label": "xAI / Grok (cloud)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "xai_api_key",
        "secret_label": "xAI API key",
        "default_model": "grok-2",
        "model_prompt": "text",
        "extra_config": [],
    },
    "together_ai": {
        "label": "Together AI (cloud)",
        "category": "cloud",
        "auth_type": "api_key",
        "secret_key": "together_api_key",
        "secret_label": "Together API key",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "model_prompt": "text",
        "extra_config": [],
    },
}


# ── Provider class registry ──────────────────────────────────────────────
# Maps provider name → "module:ClassName" for lazy import.

PROVIDER_REGISTRY: dict[str, str] = {
    "anthropic": "arqitect.inference.providers.anthropic:AnthropicProvider",
    "groq": "arqitect.inference.providers.groq:GroqProvider",
    "openai": "arqitect.inference.providers.openai_compat:OpenAICompatProvider",
    "openai_compat": "arqitect.inference.providers.openai_compat:OpenAICompatProvider",
    "ollama": "arqitect.inference.providers.ollama:OllamaProvider",
    "gguf": "arqitect.inference.providers.gguf:GGUFProvider",
    "deepseek": "arqitect.inference.providers.deepseek:DeepSeekProvider",
    "mistral": "arqitect.inference.providers.mistral:MistralProvider",
    "openrouter": "arqitect.inference.providers.openrouter:OpenRouterProvider",
    "google_gemini": "arqitect.inference.providers.google_gemini:GoogleGeminiProvider",
    "xai": "arqitect.inference.providers.xai:XAIProvider",
    "together_ai": "arqitect.inference.providers.together_ai:TogetherAIProvider",
}


def get_provider(name: str, **kwargs) -> InferenceProvider:
    """Get a provider instance by name.

    Args:
        name: Provider identifier (must exist in PROVIDER_REGISTRY).
        **kwargs: Passed to the provider constructor.

    Returns:
        Instantiated InferenceProvider.

    Raises:
        ValueError: If the provider name is not registered.
    """
    entry = PROVIDER_REGISTRY.get(name)
    if not entry:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDER_REGISTRY.keys())}")
    module_path, class_name = entry.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def get_registered_providers() -> list[str]:
    """Return provider names that have implementations (excluding aliases).

    Filters out ``openai_compat`` alias — the wizard should show ``openai`` only.
    """
    return [name for name in PROVIDER_META if name in PROVIDER_REGISTRY]


def get_wizard_providers(environment: str) -> list[tuple[str, str]]:
    """Build an ordered provider list for the wizard based on environment.

    Returns list of (provider_name, label) tuples. Local providers appear first
    for desktop/iot, cloud providers first for server environments.

    Args:
        environment: One of "desktop", "server", "iot".

    Returns:
        Ordered list of (name, label) tuples for providers with implementations.
    """
    available = get_registered_providers()
    local = [(n, PROVIDER_META[n]["label"]) for n in available if PROVIDER_META[n]["category"] == "local"]
    cloud = [(n, PROVIDER_META[n]["label"]) for n in available if PROVIDER_META[n]["category"] == "cloud"]

    if environment in ("desktop", "iot"):
        return local + cloud
    return cloud + local
