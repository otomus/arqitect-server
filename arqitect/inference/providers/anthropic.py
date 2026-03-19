"""Anthropic Claude inference provider."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.base import InferenceProvider


class AnthropicProvider(InferenceProvider):
    def __init__(self, api_key: str = "", **kwargs):
        self.api_key = api_key or get_secret("anthropic_api_key")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 2048, temperature: float = 0.7,
                 json_mode: bool = False) -> str:
        client = self._get_client()
        kwargs = {
            "model": model or "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    @property
    def supports_vision(self) -> bool:
        return True

    def generate_vision(self, model: str, prompt: str, image_b64: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text
