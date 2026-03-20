"""Inference Engine — GGUF-based singleton for vision, embedding, and image generation.

Text generation uses ``arqitect.inference.router.generate_for_role()``.
This module provides the ``get_engine()`` singleton for non-text tasks:
  - Vision (``generate_vision()``) — used by the sight sense
  - Embedding (``embed()``) — used by the matching module
  - Image generation (``generate_image()``) — used by the image_gen sense

GGUFEngine inherits all model loading, text generation, embedding, and image
generation from GGUFProvider. It adds only:
  - ``generate_vision()`` with file-path support (delegates to provider)
  - ``load_from_registry()`` for legacy startup loading
  - The singleton pattern via ``get_engine()``

Thread-safe: multiple nerves/senses can call methods concurrently.
"""

import os
import sys
import threading

from arqitect.inference.providers.gguf import GGUFProvider
from arqitect.config.loader import get_models_dir as _get_models_dir

_DEFAULT_MODELS_DIR = _get_models_dir()


class GGUFEngine(GGUFProvider):
    """Singleton GGUF engine — extends GGUFProvider with file-path vision."""

    def __init__(self, models_dir: str = _DEFAULT_MODELS_DIR):
        super().__init__(models_dir=models_dir)

    def generate_vision(self, image_path: str = "", base64_data: str = "",
                        prompt: str = "Describe this image in detail.",
                        max_tokens: int = 256) -> str:
        """Analyze an image from a file path or base64 data.

        This is the engine-specific signature used by sight/nerve.py.
        Delegates to GGUFProvider.generate_vision_from_path().
        """
        return self.generate_vision_from_path(
            image_path=image_path,
            base64_data=base64_data,
            prompt=prompt,
            max_tokens=max_tokens,
        )


# ── Singleton ─────────────────────────────────────────────────────────────

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> GGUFEngine:
    """Get or create the singleton GGUF inference engine."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE

        from .config import (
            get_models_dir, check_gguf_ready,
            print_setup_guide, print_status_report,
        )

        models_dir = get_models_dir()
        lazy = os.environ.get("SYNAPSE_LAZY_LOAD") == "1"
        if not lazy:
            ready, missing = check_gguf_ready()
            print_status_report("gguf", ready, missing)
            if not ready:
                print_setup_guide()
                raise RuntimeError(
                    f"GGUF models missing: {', '.join(missing)}. "
                    "Check arqitect.yaml inference config."
                )
        _ENGINE = GGUFEngine(models_dir)
        if not lazy:
            _ENGINE.load_from_registry()
        suffix = " (lazy)" if lazy else ""
        print(f"[ENGINE] GGUF backend ready ({models_dir}){suffix}")

    return _ENGINE


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_embedder():
    """Get the embedding function from the engine."""
    return get_engine().embed
