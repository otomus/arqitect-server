"""Model downloader — fetches GGUF files from HuggingFace Hub."""

import os
import sys

from arqitect.inference.model_registry import MODEL_REGISTRY


def ensure_model(name: str, models_dir: str) -> str | None:
    """Ensure a model file exists locally, downloading if needed.

    Returns the path to the GGUF file, or None if unavailable.
    """
    entry = MODEL_REGISTRY.get(name)
    if not entry:
        return None

    filename = entry["file"]
    path = os.path.join(models_dir, filename)

    if os.path.exists(path):
        return path

    source = entry.get("source")
    if not source:
        return None

    return download_gguf(source, filename, models_dir)


def download_gguf(repo_id: str, filename: str, dest_dir: str) -> str | None:
    """Download a GGUF file from HuggingFace Hub.

    Returns the local path on success, None on failure.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    try:
        from huggingface_hub import hf_hub_download
        print(f"[DOWNLOAD] Fetching {filename} from {repo_id}...", file=sys.stderr)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[DOWNLOAD] {filename} ready at {dest_dir}", file=sys.stderr)
        return downloaded
    except Exception as e:
        print(f"[DOWNLOAD] Failed to download {filename}: {e}", file=sys.stderr)
        return None


# Known GGUF models available for download, grouped by size class.
# Each entry: (display_name, gguf_filename, hf_repo, size_label, n_ctx)
AVAILABLE_MODELS = [
    {
        "name": "Qwen2.5-Coder-1.5B",
        "file": "Qwen2.5-Coder-1.5B.gguf",
        "source": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "size_class": "tiny",
        "size_label": "~1.2 GB",
        "n_ctx": 2048,
    },
    {
        "name": "Qwen2.5-Coder-7B",
        "file": "Qwen2.5-Coder-7B.gguf",
        "source": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "size_class": "small",
        "size_label": "~4.7 GB",
        "n_ctx": 4096,
    },
    {
        "name": "Qwen2.5-Coder-14B",
        "file": "Qwen2.5-Coder-14B.gguf",
        "source": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "size_class": "medium",
        "size_label": "~9.0 GB",
        "n_ctx": 4096,
    },
    {
        "name": "Qwen2.5-Coder-32B",
        "file": "Qwen2.5-Coder-32B.gguf",
        "source": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "size_class": "large",
        "size_label": "~20 GB",
        "n_ctx": 4096,
    },
]
