"""Together AI inference provider (OpenAI-compatible API)."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.openai_compat import OpenAICompatProvider


class TogetherAIProvider(OpenAICompatProvider):
    """Together AI hosted open-source models via OpenAI-compatible endpoint."""

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(
            api_key=api_key or get_secret("together_api_key"),
            base_url="https://api.together.xyz/v1",
            default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            **kwargs,
        )
