"""OpenRouter inference provider (OpenAI-compatible gateway)."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.openai_compat import OpenAICompatProvider


class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter gateway — routes to multiple model providers."""

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(
            api_key=api_key or get_secret("openrouter_api_key"),
            base_url="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4-20250514",
            **kwargs,
        )
