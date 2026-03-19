"""Generic OpenAI-compatible inference provider."""
from arqitect.config.loader import get_secret
from arqitect.inference.providers.base import InferenceProvider


class OpenAICompatProvider(InferenceProvider):
    def __init__(self, api_key: str = "", base_url: str = "", default_model: str = "", **kwargs):
        self.api_key = api_key or get_secret("openai_api_key")
        self.base_url = base_url or get_secret("openai_base_url", "https://api.openai.com/v1")
        self.default_model = default_model or "gpt-4o-mini"
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 2048, temperature: float = 0.7,
                 json_mode: bool = False) -> str:
        client = self._get_client()
        messages = self._build_messages(prompt, system)
        kwargs = {"model": model or self.default_model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    @property
    def supports_vision(self) -> bool:
        return True

    def generate_vision(self, model: str, prompt: str, image_b64: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=model or self.default_model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content
