"""Abstract base class for inference providers."""
from abc import ABC, abstractmethod
from typing import Optional


class InferenceProvider(ABC):
    """Abstract inference provider interface."""

    @abstractmethod
    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 2048, temperature: float = 0.7,
                 json_mode: bool = False) -> str:
        """Generate text completion."""
        ...

    @staticmethod
    def _build_messages(prompt: str, system: str = "") -> list[dict]:
        """Build a standard messages list for chat completion APIs."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate_vision(self, model: str, prompt: str, image_b64: str) -> str:
        """Generate text from image + prompt. Override if supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support vision")

    def embed(self, text: str) -> list[float]:
        """Generate text embedding. Override if supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    @property
    def supports_vision(self) -> bool:
        return False

    @property
    def supports_embedding(self) -> bool:
        return False

    @property
    def supports_lora(self) -> bool:
        return False

    def preload(self, model: str, n_ctx: int = 2048) -> None:
        """Eagerly load a model so it's ready for generation.

        No-op for cloud providers. Local providers (e.g. GGUF) override
        this to load model weights into memory at startup.

        Args:
            model: Model name or path to preload.
            n_ctx: Context window size (from community adapter meta.json).
        """

    def list_loaded(self) -> list[str]:
        """Return names of currently loaded models.

        Returns an empty list for cloud providers (always ready).
        """
        return []
