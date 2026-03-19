"""Inference Engine — legacy singleton for non-text-gen operations.

DEPRECATED for text generation: use ``arqitect.inference.router.generate_for_role()``
instead. All text-gen call sites have been migrated to the per-role router.

This module is still used for:
  - Vision (``generate_vision()``) — used by the sight sense
  - Embedding (``embed()``) — used by the matching module
  - Image generation (``generate_image()``) — used by the image_gen sense
  - Model loading / listing (``load_from_registry()``, ``list_loaded()``)

Thread-safe: multiple nerves/senses can call generate() concurrently.
Singleton pattern via get_engine() — one engine per process.
"""

import base64
import os
import sys
import threading
import time

from .model_registry import MODEL_REGISTRY, OLLAMA_NAME_MAP


from arqitect.config.loader import get_models_dir as _get_models_dir
_DEFAULT_MODELS_DIR = _get_models_dir()


# ── Ollama Backend ────────────────────────────────────────────────────────

class OllamaEngine:
    """Inference via Ollama HTTP API. Zero startup cost — models managed externally."""

    def __init__(self, host: str = "http://localhost:11434"):
        self._host = host.rstrip("/")
        self._available = set()  # cached set of model names
        self._lock = threading.Lock()
        self._refresh_models()

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
        from .config import get_model_name
        return get_model_name(name)

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 512, temperature: float = 0.7,
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

    def generate_vision(self, image_path: str = "", base64_data: str = "",
                        prompt: str = "Describe this image in detail.",
                        max_tokens: int = 256) -> str:
        import requests
        from .config import get_model_name
        vision_model = get_model_name("vision")

        if not base64_data and image_path:
            if not os.path.exists(image_path):
                return f"Error: image not found: {image_path}"
            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

        if not base64_data:
            return "Error: no image provided"

        payload = {
            "model": vision_model,
            "prompt": prompt,
            "images": [base64_data],
            "stream": False,
        }

        try:
            resp = requests.post(f"{self._host}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"Error: {e}"

    def is_loaded(self, model: str) -> bool:
        resolved = OLLAMA_NAME_MAP.get(model, model)
        ollama_name = self._resolve_model(resolved)
        return ollama_name in self._available or ollama_name.split(":")[0] in self._available

    def list_loaded(self) -> list[str]:
        return list(self._available)

    def generate_image(self, prompt: str, width: int = 512, height: int = 512,
                       steps: int = 4, seed: int = -1) -> str | None:
        """Image generation not supported on Ollama backend."""
        return None

    def load_from_registry(self, names: list[str] = None):
        self._refresh_models()


# ── GGUF Backend ──────────────────────────────────────────────────────────

class GGUFEngine:
    """In-process inference via llama-cpp-python with GGUF files."""

    _MAX_LORA_MODELS = 5  # Maximum number of cached LoRA-loaded model instances

    def __init__(self, models_dir: str = _DEFAULT_MODELS_DIR):
        self._models = {}           # name -> Llama instance
        self._model_locks = {}      # name -> threading.Lock (per-model)
        self._models_dir = os.path.abspath(models_dir)
        self._global_lock = threading.Lock()
        self._path_to_name = {}     # resolved gguf path -> first loaded name (for sharing)
        self._lora_access_times = {}  # lora_key -> last access timestamp

    def load_model(self, name: str, gguf_path: str = None,
                   n_ctx: int = 2048, n_gpu_layers: int = -1,
                   chat_handler: str = None):
        from llama_cpp import Llama
        from .download import ensure_model

        if gguf_path is None:
            gguf_path = ensure_model(name, self._models_dir)
            if gguf_path is None:
                print(f"[ENGINE] Skipping {name}: model file not available", file=sys.stderr)
                return

        if not os.path.exists(gguf_path):
            print(f"[ENGINE] Skipping {name}: {gguf_path} not found", file=sys.stderr)
            return

        # Model sharing — if this GGUF is already loaded under another name, reuse it
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

        if chat_handler == "moondream":
            try:
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                # Use local mmproj file instead of downloading from HuggingFace
                mmproj_entry = MODEL_REGISTRY.get(name, {}).get("mmproj")
                mmproj_path = os.path.join(self._models_dir, mmproj_entry) if mmproj_entry else None
                if mmproj_path and os.path.exists(mmproj_path):
                    handler = MoondreamChatHandler(clip_model_path=mmproj_path)
                else:
                    # Fallback: try from_pretrained with the GGUF repo
                    handler = MoondreamChatHandler.from_pretrained(
                        repo_id="moondream/moondream2-gguf",
                        filename="*mmproj*",
                    )
                kwargs["chat_handler"] = handler
            except Exception as e:
                print(f"[ENGINE] Moondream handler failed: {e}. Vision unavailable.", file=sys.stderr)
                return

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

    def _ensure_loaded(self, name: str):
        if name in self._models:
            return
        if name in MODEL_REGISTRY:
            entry = MODEL_REGISTRY[name]
            self.load_model(
                name,
                n_ctx=entry.get("n_ctx", 2048),
                chat_handler=entry.get("chat_handler"),
            )

    def generate(self, model: str, prompt: str, system: str = "",
                 max_tokens: int = 512, temperature: float = 0.7,
                 lora_path: str = None, json_mode: bool = False) -> str:
        """Generate text. If lora_path is provided, loads a LoRA adapter model instance."""
        resolved = OLLAMA_NAME_MAP.get(model, model)

        # LoRA adapter: load a separate model instance with the adapter applied
        if lora_path and os.path.exists(lora_path):
            return self._generate_with_lora(resolved, prompt, system,
                                            max_tokens, temperature, lora_path)

        self._ensure_loaded(resolved)

        if resolved not in self._models:
            return f"Error: model '{model}' (resolved: '{resolved}') not loaded"

        lock = self._model_locks[resolved]
        llm = self._models[resolved]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

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
        """Generate using a model with a LoRA adapter applied.

        LoRA-adapted models are cached separately as '{model}:lora:{path_hash}'
        so each nerve gets its own specialized instance.
        """
        import hashlib
        path_hash = hashlib.md5(lora_path.encode()).hexdigest()[:8]
        lora_key = f"{model}:lora:{path_hash}"

        if lora_key not in self._models:
            from llama_cpp import Llama
            from .download import ensure_model

            gguf_path = ensure_model(model, self._models_dir)
            if not gguf_path or not os.path.exists(gguf_path):
                return f"Error: base model '{model}' not found for LoRA"

            entry = MODEL_REGISTRY.get(model, {})
            n_ctx = entry.get("n_ctx", 2048)

            # Evict least recently used LoRA model if at capacity
            with self._global_lock:
                lora_keys = [k for k in self._lora_access_times if k in self._models]
                if len(lora_keys) >= self._MAX_LORA_MODELS:
                    oldest = min(lora_keys, key=lambda k: self._lora_access_times[k])
                    self._models.pop(oldest, None)
                    self._model_locks.pop(oldest, None)
                    self._lora_access_times.pop(oldest, None)
                    print(f"[ENGINE] Evicted LoRA model {oldest} (capacity={self._MAX_LORA_MODELS})")

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
                # Fallback to base model
                return self.generate(model, prompt, system, max_tokens, temperature)

        self._lora_access_times[lora_key] = time.monotonic()
        lock = self._model_locks[lora_key]
        llm = self._models[lora_key]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

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

    def generate_vision(self, image_path: str = "", base64_data: str = "",
                        prompt: str = "Describe this image in detail.",
                        max_tokens: int = 256) -> str:
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

    def is_loaded(self, model: str) -> bool:
        resolved = OLLAMA_NAME_MAP.get(model, model)
        return resolved in self._models

    def unload(self, model: str):
        resolved = OLLAMA_NAME_MAP.get(model, model)
        with self._global_lock:
            self._models.pop(resolved, None)
            self._model_locks.pop(resolved, None)

    def list_loaded(self) -> list[str]:
        return list(self._models.keys())

    def load_image_gen(self, name: str = "image_gen"):
        """Load a Stable Diffusion model for image generation."""
        if name in self._models:
            return
        from .download import ensure_model
        entry = MODEL_REGISTRY.get(name)
        if not entry or entry.get("backend") != "stable_diffusion":
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

    def _load_embed_model(self):
        """Load a dedicated embedding model (separate from chat models)."""
        from llama_cpp import Llama

        # Use dedicated lightweight embedding model (nomic-embed-text, 274MB)
        # instead of the main chat model which would double memory usage
        embed_path = os.path.join(self._models_dir, "nomic-embed-text.gguf")
        if not os.path.exists(embed_path):
            # Fallback to nerve model if nomic not available
            from .download import ensure_model
            embed_path = ensure_model("nerve", self._models_dir)
        if embed_path is None:
            embed_path = os.path.join(self._models_dir, MODEL_REGISTRY["nerve"]["file"])

        if not os.path.exists(embed_path):
            print(f"[ENGINE] Embedding model not found: {embed_path}", file=sys.stderr)
            return

        print(f"[ENGINE] Loading embedding model from {os.path.basename(embed_path)}...")
        try:
            self._embed_model = Llama(
                model_path=embed_path, n_ctx=512, embedding=True,
                n_gpu_layers=-1, verbose=False,
            )
            self._embed_lock = threading.Lock()
            print("[ENGINE] Embedding model loaded")
        except Exception as e:
            self._embed_model = None
            print(f"[ENGINE] Failed to load embedding model: {e}", file=sys.stderr)

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        if not hasattr(self, '_embed_model') or self._embed_model is None:
            self._load_embed_model()
        if self._embed_model is None:
            raise RuntimeError("Embedding model failed to load")
        with self._embed_lock:
            result = self._embed_model.embed(text)
            # result is a list of floats or list of list of floats
            if isinstance(result[0], list):
                return result[0]
            return result

    def load_from_registry(self, names: list[str] = None):
        targets = names or list(MODEL_REGISTRY.keys())
        for name in targets:
            if name in self._models:
                continue
            entry = MODEL_REGISTRY.get(name)
            if not entry:
                continue
            # Skip non-LLM models (loaded on demand)
            if entry.get("backend") == "stable_diffusion":
                continue
            self.load_model(
                name,
                n_ctx=entry.get("n_ctx", 2048),
                chat_handler=entry.get("chat_handler"),
            )


# ── Singleton ─────────────────────────────────────────────────────────────

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> OllamaEngine | GGUFEngine:
    """Get or create the singleton inference engine.

    Backend is determined by inference.conf. If no config exists,
    prints setup instructions and exits.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE

        from .config import (
            get_backend_type, get_ollama_host, get_models_dir,
            check_ollama_ready, check_gguf_ready,
            print_setup_guide, print_status_report,
        )

        backend = get_backend_type()

        if not backend:
            print_setup_guide()
            raise RuntimeError(
                "No inference backend configured. "
                "Run: python inference/setup.py ollama"
            )

        if backend == "ollama":
            host = get_ollama_host()
            ready, missing = check_ollama_ready()
            print_status_report("ollama", ready, missing)
            if not ready:
                raise RuntimeError(
                    f"Ollama models missing: {', '.join(missing)}. "
                    "Fix missing models before starting."
                )
            _ENGINE = OllamaEngine(host)
            print(f"[ENGINE] Ollama backend ready ({host})")

        elif backend == "gguf":
            models_dir = get_models_dir()
            lazy = os.environ.get("SYNAPSE_LAZY_LOAD") == "1"
            if not lazy:
                ready, missing = check_gguf_ready()
                print_status_report("gguf", ready, missing)
                if not ready:
                    raise RuntimeError(
                        f"GGUF models missing: {', '.join(missing)}. "
                        "Run: python inference/setup.py gguf --download"
                    )
            _ENGINE = GGUFEngine(models_dir)
            if not lazy:
                _ENGINE.load_from_registry()
            suffix = " (lazy)" if lazy else ""
            print(f"[ENGINE] GGUF backend ready ({models_dir}){suffix}")

        else:
            raise RuntimeError(
                f"Unknown inference backend '{backend}' in config. "
                "Supported: ollama, gguf"
            )

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
