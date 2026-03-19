"""Ollama inference provider — wraps OllamaEngine as an InferenceProvider."""
import base64
import os
import threading

from arqitect.inference.providers.base import InferenceProvider
from arqitect.inference.model_registry import OLLAMA_NAME_MAP


class OllamaProvider(InferenceProvider):
    """Inference via Ollama HTTP API. Zero startup cost -- models managed externally."""

    def __init__(self, host: str = "http://localhost:11434", **kwargs):
        self._host = host.rstrip("/")
        self._available: set[str] = set()
        self._lock = threading.Lock()
        self._refresh_models()

    # ── internal helpers ──────────────────────────────────────────────────

    def _refresh_models(self):
        try:
            import requests
            resp = requests.get(f"{self._host}/api/tags", timeout=5)
            resp.raise_for_status()
            for m in resp.json().get("models", []):
                name = m.get("name", "")
                self._available.add(name)
                self._available.add(name.split(":")[0])
        except Exception:
            pass

    def _resolve_model(self, name: str) -> str:
        """Map logical name -> Ollama model name via config."""
        from arqitect.inference.config import get_model_name
        return get_model_name(name)

    # ── InferenceProvider interface ───────────────────────────────────────

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 2048, temperature: float = 0.7,
                 json_mode: bool = False) -> str:
        import requests
        resolved = OLLAMA_NAME_MAP.get(model, model)
        ollama_name = self._resolve_model(resolved)

        payload = {
            "model": ollama_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            resp = requests.post(f"{self._host}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"

    @property
    def supports_vision(self) -> bool:
        return True

    def generate_vision(self, model: str, prompt: str, image_b64: str) -> str:
        import requests
        from arqitect.inference.config import get_model_name
        vision_model = get_model_name("vision")

        if not image_b64:
            return "Error: no image provided"

        payload = {
            "model": vision_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }

        try:
            resp = requests.post(f"{self._host}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"

    # ── extra helpers (non-interface) ─────────────────────────────────────

    def is_loaded(self, model: str) -> bool:
        resolved = OLLAMA_NAME_MAP.get(model, model)
        ollama_name = self._resolve_model(resolved)
        return ollama_name in self._available or ollama_name.split(":")[0] in self._available

    def list_loaded(self) -> list[str]:
        return list(self._available)

    def load_from_registry(self, names: list[str] = None):
        self._refresh_models()
