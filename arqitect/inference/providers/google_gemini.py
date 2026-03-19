"""Google Gemini inference provider (OpenAI-compatible API)."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.openai_compat import OpenAICompatProvider


class GoogleGeminiProvider(OpenAICompatProvider):
    """Google Gemini models via OpenAI-compatible endpoint."""

    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(
            api_key=api_key or get_secret("google_ai_api_key"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            default_model="gemini-2.0-flash",
            **kwargs,
        )
