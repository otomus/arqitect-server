"""DeepSeek inference provider (OpenAI-compatible API)."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.openai_compat import OpenAICompatProvider


class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek code-optimised models via OpenAI-compatible endpoint."""

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(
            api_key=api_key or get_secret("deepseek_api_key"),
            base_url="https://api.deepseek.com",
            default_model="deepseek-chat",
            **kwargs,
        )
