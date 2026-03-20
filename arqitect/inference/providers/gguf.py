"""GGUF inference provider — in-process inference via llama-cpp-python."""
import base64
import os
import sys
import threading
import time

from arqitect.inference.providers.base import InferenceProvider
from arqitect.inference.model_registry import (
    MODEL_REGISTRY,
    CHAT_HANDLER_MOONDREAM,
    BACKEND_STABLE_DIFFUSION,
    find_registry_entry_by_file,
    resolve_model_path,
)
from arqitect.config.loader import get_models_dir as _get_models_dir

_DEFAULT_MODELS_DIR = None


def _default_models_dir() -> str:
    global _DEFAULT_MODELS_DIR
    if _DEFAULT_MODELS_DIR is None:
        _DEFAULT_MODELS_DIR = _get_models_dir()
    return _DEFAULT_MODELS_DIR


class GGUFProvider(InferenceProvider):
    """In-process inference via llama-cpp-python with GGUF files."""

    _MAX_LORA_MODELS = 5

    def __init__(self, models_dir: str = None, **kwargs):
        if models_dir is None:
            models_dir = _default_models_dir()
        self._models = {}
        self._model_locks = {}
        self._models_dir = os.path.abspath(models_dir)
        self._global_lock = threading.Lock()
        self._path_to_name = {}
        self._lora_access_times = {}

    # ── model loading ─────────────────────────────────────────────────────

    def load_model(self, name: str, gguf_path: str = None,
                   n_ctx: int = 2048, n_gpu_layers: int = -1,
                   chat_handler: str = None):
        """Load a GGUF model into memory.

        Args:
            name: Logical model name.
            gguf_path: Path to GGUF file (auto-resolved if None).
            n_ctx: Context window size (from community adapter).
            n_gpu_layers: GPU layers (-1 = all).
            chat_handler: Special handler name (e.g. moondream).
        """
        from llama_cpp import Llama
        from arqitect.inference.download import ensure_model

        if gguf_path is None:
            gguf_path = ensure_model(name, self._models_dir)
            if gguf_path is None:
                print(f"[ENGINE] Skipping {name}: model file not available", file=sys.stderr)
                return

        if not os.path.exists(gguf_path):
            print(f"[ENGINE] Skipping {name}: {gguf_path} not found", file=sys.stderr)
            return

        resolved_path = os.path.realpath(gguf_path)
        if resolved_path in self._path_to_name:
            existing = self._path_to_name[resolved_path]
            if existing in self._models:
                with self._global_lock:
                    self._models[name] = self._models[existing]
                    self._model_locks[name] = self._model_locks[existing]
                print(f"[ENGINE] {name} sharing model with {existing} ({os.path.basename(gguf_path)})")
                return

        print(f"[ENGINE] Loading {name} from {os.path.basename(gguf_path)}...")

        kwargs = {
            "model_path": gguf_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }

        if chat_handler == CHAT_HANDLER_MOONDREAM:
            handler = self._build_moondream_handler(name)
            if handler is None:
                return
            kwargs["chat_handler"] = handler

        try:
            model = Llama(**kwargs)
            lock = threading.Lock()
            with self._global_lock:
                self._models[name] = model
                self._model_locks[name] = lock
                self._path_to_name[resolved_path] = name
            print(f"[ENGINE] {name} loaded ({n_ctx} ctx)")
        except Exception as e:
            print(f"[ENGINE] Failed to load {name}: {e}", file=sys.stderr)

    def _build_moondream_handler(self, name: str):
        """Build a MoondreamChatHandler, using local mmproj file if available."""
        try:
            from llama_cpp.llama_chat_format import MoondreamChatHandler
            mmproj_file = MODEL_REGISTRY.get(name, {}).get("mmproj")
            mmproj_path = os.path.join(self._models_dir, mmproj_file) if mmproj_file else None
            if mmproj_path and os.path.exists(mmproj_path):
                return MoondreamChatHandler(clip_model_path=mmproj_path)
            # Fallback: download from HuggingFace using source from yaml config
            source = MODEL_REGISTRY.get(name, {}).get("source")
            if not source:
                print(f"[ENGINE] No source configured for {name}. Set inference.models.{name}.source in arqitect.yaml",
                      file=sys.stderr)
                return None
            return MoondreamChatHandler.from_pretrained(
                repo_id=source, filename="*mmproj*",
            )
        except Exception as e:
            print(f"[ENGINE] Moondream handler failed: {e}. Vision unavailable.", file=sys.stderr)
            return None

    def _ensure_loaded(self, name: str, n_ctx: int = 2048):
        """Ensure a model is loaded and available for generation.

        Args:
            name: Model name, registry key, filename, or absolute path.
            n_ctx: Context window size (from community adapter).

        Resolution order:
        1. Already loaded under this name → return immediately.
        2. Name is a MODEL_REGISTRY key → load via registry entry.
        3. Name is a filename or path → resolve and load directly.
        4. Already loaded under the same resolved path → alias it.
        """
        if name in self._models:
            return
        if name in MODEL_REGISTRY:
            entry = MODEL_REGISTRY[name]
            self.load_model(
                name,
                n_ctx=n_ctx,
                chat_handler=entry.get("chat_handler"),
            )
            return
        gguf_path = resolve_model_path(name, self._models_dir)
        if gguf_path:
            resolved = os.path.realpath(gguf_path)
            if resolved in self._path_to_name:
                existing = self._path_to_name[resolved]
                if existing in self._models:
                    with self._global_lock:
                        self._models[name] = self._models[existing]
                        self._model_locks[name] = self._model_locks[existing]
                    return
            entry = find_registry_entry_by_file(os.path.basename(gguf_path)) or {}
            self.load_model(
                name, gguf_path=gguf_path,
                n_ctx=n_ctx,
                chat_handler=entry.get("chat_handler"),
            )

    # ── InferenceProvider interface ───────────────────────────────────────

    def preload(self, model: str, n_ctx: int = 2048) -> None:
        """Eagerly load a GGUF model into memory."""
        self._ensure_loaded(model, n_ctx=n_ctx)

    def list_loaded(self) -> list[str]:
        """Return names of currently loaded models."""
        return list(self._models.keys())

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 2048, temperature: float = 0.7,
                 json_mode: bool = False, lora_path: str = None) -> str:
        """Generate text. If lora_path is provided, loads a LoRA adapter model instance."""
        if lora_path and os.path.exists(lora_path):
            return self._generate_with_lora(model, prompt, system,
                                            max_tokens, temperature, lora_path)

        from arqitect.brain.adapters import get_max_context
        self._ensure_loaded(model, n_ctx=get_max_context(model))

        if model not in self._models:
            return f"Error: model '{model}' not loaded"

        lock = self._model_locks[model]
        llm = self._models[model]

        messages = self._build_messages(prompt, system)
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            with lock:
                resp = llm.create_chat_completion(**kwargs)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}"

    def _generate_with_lora(self, model: str, prompt: str, system: str,
                            max_tokens: int, temperature: float,
                            lora_path: str) -> str:
        """Generate using a model with a LoRA adapter applied."""
        import hashlib
        path_hash = hashlib.md5(lora_path.encode()).hexdigest()[:8]
        lora_key = f"{model}:lora:{path_hash}"

        if lora_key not in self._models:
            from llama_cpp import Llama
            from arqitect.inference.download import ensure_model

            gguf_path = ensure_model(model, self._models_dir)
            if not gguf_path or not os.path.exists(gguf_path):
                return f"Error: base model '{model}' not found for LoRA"

            from arqitect.brain.adapters import get_max_context
            n_ctx = get_max_context(model)

            self._evict_oldest_lora()

            print(f"[ENGINE] Loading {model} with LoRA adapter: {lora_path}",
                  file=sys.stderr)
            try:
                llm = Llama(
                    model_path=gguf_path, n_ctx=n_ctx,
                    n_gpu_layers=-1, verbose=False,
                    lora_path=lora_path,
                )
                with self._global_lock:
                    self._models[lora_key] = llm
                    self._model_locks[lora_key] = threading.Lock()
                    self._lora_access_times[lora_key] = time.monotonic()
                print(f"[ENGINE] {lora_key} loaded")
            except Exception as e:
                print(f"[ENGINE] LoRA load failed: {e}", file=sys.stderr)
                return self.generate(model, prompt, system, max_tokens, temperature)

        self._lora_access_times[lora_key] = time.monotonic()
        lock = self._model_locks[lora_key]
        llm = self._models[lora_key]

        messages = self._build_messages(prompt, system)
        try:
            with lock:
                resp = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}"

    def _evict_oldest_lora(self):
        """Evict the least recently used LoRA model if at capacity."""
        with self._global_lock:
            lora_keys = [k for k in self._lora_access_times if k in self._models]
            if len(lora_keys) >= self._MAX_LORA_MODELS:
                oldest = min(lora_keys, key=lambda k: self._lora_access_times[k])
                self._models.pop(oldest, None)
                self._model_locks.pop(oldest, None)
                self._lora_access_times.pop(oldest, None)
                print(f"[ENGINE] Evicted LoRA model {oldest} (capacity={self._MAX_LORA_MODELS})")

    # ── Vision ────────────────────────────────────────────────────────────

    @property
    def supports_vision(self) -> bool:
        return True

    def generate_vision(self, model: str, prompt: str, image_b64: str) -> str:
        """Generate text from image + prompt."""
        self._ensure_loaded("vision")

        if "vision" not in self._models:
            return "Error: vision model not loaded"
        if not image_b64:
            return "Error: no image provided"

        data_uri = f"data:image/png;base64,{image_b64}"
        lock = self._model_locks["vision"]
        llm = self._models["vision"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            with lock:
                resp = llm.create_chat_completion(messages=messages, max_tokens=256)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}"

    def generate_vision_from_path(self, image_path: str = "",
                                  base64_data: str = "",
                                  prompt: str = "Describe this image in detail.",
                                  max_tokens: int = 256) -> str:
        """Generate text from an image file path or base64 data.

        Extends generate_vision with file-path-to-base64 conversion.
        Used by the sight sense and the legacy engine singleton.
        """
        self._ensure_loaded("vision")

        if "vision" not in self._models:
            return "Error: vision model not loaded"

        if not base64_data and image_path:
            if not os.path.exists(image_path):
                return f"Error: image not found: {image_path}"
            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

        if not base64_data:
            return "Error: no image provided"

        data_uri = f"data:image/png;base64,{base64_data}"
        lock = self._model_locks["vision"]
        llm = self._models["vision"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            with lock:
                resp = llm.create_chat_completion(messages=messages, max_tokens=max_tokens)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}"

    # ── Embedding ─────────────────────────────────────────────────────────

    @property
    def supports_embedding(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        if not hasattr(self, '_embed_model') or self._embed_model is None:
            self._load_embed_model()
        if self._embed_model is None:
            raise RuntimeError("Embedding model failed to load")
        with self._embed_lock:
            result = self._embed_model.embed(text)
            if isinstance(result[0], list):
                return result[0]
            return result

    def _load_embed_model(self):
        """Load a dedicated embedding model (separate from chat models).

        Model file and config come from ``inference.models.embedding`` in yaml.
        Falls back to the nerve model if no embedding model is configured.
        """
        from llama_cpp import Llama
        from arqitect.inference.download import ensure_model

        embed_entry = MODEL_REGISTRY.get("embedding", {})
        embed_file = embed_entry.get("file", "")

        embed_path = None
        if embed_file:
            embed_path = os.path.join(self._models_dir, embed_file)
            if not os.path.exists(embed_path):
                embed_path = ensure_model("embedding", self._models_dir)

        if not embed_path or not os.path.exists(embed_path):
            embed_path = ensure_model("nerve", self._models_dir)

        if not embed_path or not os.path.exists(embed_path):
            print("[ENGINE] Embedding model not found", file=sys.stderr)
            return

        # Context size for embedding is small — 512 is standard
        from arqitect.brain.adapters import get_max_context
        n_ctx = get_max_context("embedding")

        print(f"[ENGINE] Loading embedding model from {os.path.basename(embed_path)}...")
        try:
            self._embed_model = Llama(
                model_path=embed_path, n_ctx=n_ctx, embedding=True,
                n_gpu_layers=-1, verbose=False,
            )
            self._embed_lock = threading.Lock()
            print("[ENGINE] Embedding model loaded")
        except Exception as e:
            self._embed_model = None
            print(f"[ENGINE] Failed to load embedding model: {e}", file=sys.stderr)

    @property
    def supports_lora(self) -> bool:
        return True

    # ── Helpers ────────────────────────────────────────────────────────────

    def is_loaded(self, model: str) -> bool:
        """Check if a model is currently loaded."""
        return model in self._models

    def unload(self, model: str):
        """Unload a model from memory."""
        with self._global_lock:
            self._models.pop(model, None)
            self._model_locks.pop(model, None)

    def load_image_gen(self, name: str = "image_gen"):
        """Load a Stable Diffusion model for image generation."""
        if name in self._models:
            return
        from arqitect.inference.download import ensure_model
        entry = MODEL_REGISTRY.get(name)
        if not entry or entry.get("backend") != BACKEND_STABLE_DIFFUSION:
            return
        gguf_path = ensure_model(name, self._models_dir)
        if not gguf_path:
            print(f"[ENGINE] Skipping {name}: model file not available", file=sys.stderr)
            return
        try:
            from stable_diffusion_cpp import StableDiffusion
            print(f"[ENGINE] Loading {name} from {os.path.basename(gguf_path)}...")
            sd = StableDiffusion(model_path=gguf_path, vae_decode_only=True)
            with self._global_lock:
                self._models[name] = sd
                self._model_locks[name] = threading.Lock()
            print(f"[ENGINE] {name} loaded (SD Turbo)")
        except Exception as e:
            print(f"[ENGINE] Failed to load {name}: {e}", file=sys.stderr)

    def generate_image(self, prompt: str, width: int = 512, height: int = 512,
                       steps: int = 4, seed: int = -1) -> str | None:
        """Generate an image from a text prompt. Returns the file path or None."""
        name = "image_gen"
        if name not in self._models:
            self.load_image_gen(name)
        if name not in self._models:
            return None

        import random
        if seed < 0:
            seed = random.randint(0, 2**31)

        lock = self._model_locks[name]
        sd = self._models[name]

        sandbox = os.path.join(self._models_dir, "..", "sandbox")
        os.makedirs(sandbox, exist_ok=True)
        output_path = os.path.join(sandbox, f"generated_{seed}.png")

        try:
            with lock:
                images = sd.generate_image(
                    prompt=prompt,
                    width=width,
                    height=height,
                    sample_steps=steps,
                    cfg_scale=1.0,
                    seed=seed,
                )
            if images and len(images) > 0:
                images[0].save(output_path)
                print(f"[ENGINE] Image generated: {output_path}")
                return output_path
            return None
        except Exception as e:
            print(f"[ENGINE] Image generation failed: {e}", file=sys.stderr)
            return None

    def load_from_registry(self, names: list[str] = None):
        """Load models from the registry, resolving n_ctx from community adapters."""
        from arqitect.brain.adapters import get_max_context
        targets = names or list(MODEL_REGISTRY.keys())
        for name in targets:
            if name in self._models:
                continue
            entry = MODEL_REGISTRY.get(name)
            if not entry:
                continue
            if entry.get("backend") == BACKEND_STABLE_DIFFUSION:
                continue
            self.load_model(
                name,
                n_ctx=get_max_context(name),
                chat_handler=entry.get("chat_handler"),
            )
