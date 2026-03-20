"""Mistral inference provider (OpenAI-compatible API)."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.openai_compat import OpenAICompatProvider


class MistralProvider(OpenAICompatProvider):
    """Mistral AI models via OpenAI-compatible endpoint."""

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(
            api_key=api_key or get_secret("mistral_api_key"),
            base_url="https://api.mistral.ai/v1",
            default_model="",
            **kwargs,
        )
