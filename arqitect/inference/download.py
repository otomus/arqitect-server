"""Model downloader — fetches GGUF files from HuggingFace Hub.

All model information (file, source) comes from arqitect.yaml via MODEL_REGISTRY.
No hardcoded model lists — users configure what they need in yaml.
"""

import os
import sys

from arqitect.inference.model_registry import MODEL_REGISTRY


def ensure_model(name: str, models_dir: str) -> str | None:
    """Ensure a model file exists locally, downloading if needed.

    Looks up ``name`` in MODEL_REGISTRY (built from yaml config).
    If the file is missing and a ``source`` (HF repo) is configured,
    downloads it automatically.

    Returns the path to the GGUF file, or None if unavailable.
    """
    entry = MODEL_REGISTRY.get(name)
    if not entry:
        return None

    filename = entry.get("file", "")
    if not filename:
        return None

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
