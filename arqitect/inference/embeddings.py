"""ONNX embedding utility — standalone sentence embeddings without the LLM engine.

Uses all-MiniLM-L6-v2 ONNX model (22MB, 384 dims) for fast, thread-safe
embedding generation. Singleton pattern with lazy loading.

Usage:
    from arqitect.inference.embeddings import get_embedder
    embed = get_embedder()
    vector = embed("hello world")  # -> list[float] of length 384
"""

import logging
import math
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_ONNX_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
_ONNX_MODEL_FILE = "onnx/model.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_EMBEDDING_DIM = 384
_MAX_LENGTH = 128


class ONNXEmbedder:
    """Singleton ONNX-based sentence embedder.

    Lazy-loads the ONNX runtime session and tokenizer on first embed() call.
    Thread-safe via a lock guarding initialization.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def _ensure_loaded(self) -> bool:
        """Load ONNX session and tokenizer if not already loaded.

        Returns True if the model is ready, False on failure.
        """
        if self._initialized:
            return True

        with self._lock:
            if self._initialized:
                return True
            try:
                self._load()
                self._initialized = True
                return True
            except Exception as e:
                logger.warning("ONNX embedder failed to load: %s", e)
                return False

    def _load(self):
        """Load ONNX model and tokenizer from disk."""
        from arqitect.inference.download import ensure_onnx_embedding_model
        model_dir = ensure_onnx_embedding_model()
        if model_dir is None:
            raise RuntimeError("ONNX embedding model not available")

        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = str(Path(model_dir) / "model.onnx")
        tokenizer_path = str(Path(model_dir) / "tokenizer.json")

        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=_MAX_LENGTH)
        self._tokenizer.enable_padding(length=_MAX_LENGTH)
        logger.info("ONNX embedder loaded: %s", model_path)

    def embed(self, text: str) -> list[float]:
        """Compute a normalized 384-dim embedding for the input text.

        Uses mean pooling over token embeddings followed by L2 normalization,
        matching the sentence-transformers convention.

        Args:
            text: Input text to embed.

        Returns:
            Normalized embedding vector as a list of floats.

        Raises:
            RuntimeError: If the ONNX model cannot be loaded.
        """
        if not self._ensure_loaded():
            raise RuntimeError("ONNX embedder not available")

        import numpy as np

        encoded = self._tokenizer.encode(text)
        input_ids = [encoded.ids]
        attention_mask = [encoded.attention_mask]

        outputs = self._session.run(
            None,
            {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.zeros_like(input_ids, dtype=np.int64),
            },
        )

        # Mean pooling: average token embeddings, masked by attention
        token_embeddings = outputs[0][0]  # shape: (seq_len, hidden_dim)
        mask = np.array(attention_mask[0], dtype=np.float32)
        mask_expanded = mask[:, np.newaxis]
        summed = np.sum(token_embeddings * mask_expanded, axis=0)
        count = np.clip(mask.sum(), a_min=1e-9, a_max=None)
        mean_pooled = summed / count

        # L2 normalize
        norm = math.sqrt(sum(float(x) ** 2 for x in mean_pooled))
        if norm > 0:
            mean_pooled = mean_pooled / norm

        return mean_pooled.tolist()


def get_embedder():
    """Return the singleton embed function.

    Returns:
        A callable ``(text: str) -> list[float]`` that produces 384-dim embeddings.
        Returns None if ONNX dependencies are not installed.
    """
    try:
        embedder = ONNXEmbedder()
        return embedder.embed
    except Exception:
        return None
