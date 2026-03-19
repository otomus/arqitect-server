"""Model Registry — maps logical model names to GGUF files and HuggingFace sources.

Each entry defines:
  - file: expected GGUF filename in the models/ directory
  - n_ctx: context window size
  - source: HuggingFace repo ID for downloading
  - chat_handler: optional special handler (e.g. moondream for vision)

Logical names match the role system in nerve_runtime.py:
  brain       -> Qwen2.5-Coder-7B (routing, reasoning, judging)
  nerve       -> Qwen2.5-Coder-7B (shared — tool calling, structured queries)
  coder       -> Qwen2.5-Coder-7B (shared — code generation, tool fabrication)
  creative    -> Qwen2.5-Coder-7B (shared — writing, humor, summaries)
  communication -> Qwen2.5-Coder-7B (shared — tone rewriting, expression)
  vision      -> moondream2 (image analysis)
"""

MODEL_REGISTRY = {
    "brain": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "n_ctx": 4096,
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    },
    "nerve": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "n_ctx": 4096,
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    },
    "coder": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "n_ctx": 4096,
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    },
    "creative": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "n_ctx": 4096,
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    },
    "communication": {
        "file": "Qwen2.5-Coder-7B.gguf",
        "n_ctx": 4096,
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    },
    "vision": {
        "file": "moondream2-text-model-f16.gguf",
        "n_ctx": 2048,
        "source": "moondream/moondream2-gguf",
        "chat_handler": "moondream",
        "mmproj": "moondream2-mmproj-f16.gguf",
    },
    "image_gen": {
        "file": "stable-diffusion-v2-1-turbo-Q4_0.gguf",
        "source": "gpustack/stable-diffusion-v2-1-turbo-GGUF",
        "backend": "stable_diffusion",
    },
}

# Map old Ollama model names to logical names (used during migration)
OLLAMA_NAME_MAP = {
    "phi4-mini": "brain",
    "phi4-mini:latest": "brain",
    "qwen2.5:1.5b": "nerve",
    "qwen2.5-coder:1.5b": "coder",
    "llama3.2:3b": "creative",
    "gemma3:1b": "communication",
    "glm-4.7-flash": "brain",
    "glm-4.7-flash:latest": "brain",
    "moondream": "vision",
    "moondream2": "vision",
}
