"""Model downloader — fetches GGUF files from HuggingFace Hub.

All model information (file, source) comes from arqitect.yaml via MODEL_REGISTRY.
No hardcoded model lists — users configure what they need in yaml.
Also handles ONNX embedding model downloads.
"""

import os
import sys

from arqitect.inference.model_registry import MODEL_REGISTRY

_ONNX_EMBEDDING_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_ONNX_EMBEDDING_FILES = ["onnx/model.onnx", "tokenizer.json"]


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


def ensure_onnx_embedding_model() -> str | None:
    """Ensure the ONNX embedding model files are available locally.

    Downloads all-MiniLM-L6-v2 ONNX model and tokenizer from HuggingFace
    if not already present. Follows the same pattern as ensure_model().

    Returns:
        Path to the directory containing model.onnx and tokenizer.json,
        or None if download fails.
    """
    from arqitect.config.loader import get_models_dir

    dest_dir = os.path.join(get_models_dir(), "onnx-embeddings")
    model_path = os.path.join(dest_dir, "model.onnx")
    tokenizer_path = os.path.join(dest_dir, "tokenizer.json")

    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        return dest_dir

    os.makedirs(dest_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        for filename in _ONNX_EMBEDDING_FILES:
            # Determine the local name (flatten onnx/ prefix)
            local_name = os.path.basename(filename)
            local_path = os.path.join(dest_dir, local_name)
            if os.path.exists(local_path):
                continue

            print(f"[DOWNLOAD] Fetching {filename} from {_ONNX_EMBEDDING_REPO}...",
                  file=sys.stderr)
            downloaded = hf_hub_download(
                repo_id=_ONNX_EMBEDDING_REPO,
                filename=filename,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may place files in subdirs; move to flat layout
            if downloaded != local_path and os.path.exists(downloaded):
                import shutil
                shutil.move(downloaded, local_path)

        print(f"[DOWNLOAD] ONNX embedding model ready at {dest_dir}", file=sys.stderr)
        return dest_dir
    except Exception as e:
        print(f"[DOWNLOAD] Failed to download ONNX embedding model: {e}",
              file=sys.stderr)
        return None


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
