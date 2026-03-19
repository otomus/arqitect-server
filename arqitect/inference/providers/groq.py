"""Groq inference provider (OpenAI-compatible API)."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.openai_compat import OpenAICompatProvider


class GroqProvider(OpenAICompatProvider):
    def __init__(self, api_key: str = "", **kwargs):
        super().__init__(
            api_key=api_key or get_secret("groq_api_key"),
            base_url="https://api.groq.com/openai/v1",
            default_model="llama-3.3-70b-versatile",
            **kwargs,
        )
