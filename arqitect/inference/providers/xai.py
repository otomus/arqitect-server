"""xAI / Grok inference provider (OpenAI-compatible API)."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.openai_compat import OpenAICompatProvider


class XAIProvider(OpenAICompatProvider):
    """xAI Grok models via OpenAI-compatible endpoint."""

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(
            api_key=api_key or get_secret("xai_api_key"),
            base_url="https://api.x.ai/v1",
            default_model="grok-2",
            **kwargs,
        )
