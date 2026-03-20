"""LoRA fine-tuning engine — trains nerve-specific adapters during dreamstate.

Uses the same GGUF model files as inference (no model duplication).
transformers >= 5.3 loads GGUF directly into PyTorch for training.
Adapters are saved alongside nerve.py in nerves/{name}/adapter/.

At inference time, llama-cpp-python loads the adapter via lora_path param.
Adapter format: GGUF (converted from safetensors after training).

Training is interruptible — checks the interrupted event between batches.
If interrupted mid-training, partial work is discarded (never saves garbage).
"""

import json
import os
import sys
import threading

def _nerves_dir() -> str:
    """Lazy accessor — avoids module-level calls that break import from subprocesses."""
    from arqitect.config.loader import get_nerves_dir
    return get_nerves_dir()


def _models_dir() -> str:
    """Lazy accessor — avoids module-level calls that break import from subprocesses."""
    from arqitect.config.loader import get_models_dir
    return get_models_dir()

def _get_min_examples(role: str) -> int:
    """Get minimum training examples from community config."""
    from arqitect.brain.adapters import get_tuning_config
    return get_tuning_config(role)["min_training_examples"]


def _log(msg: str):
    print(msg, file=sys.stderr)


# Per-size-class training data limits.
# Each tier trains on a prefix of the canonical test bank.
_SIZE_CLASS_LIMITS: dict[str, int] = {
    "tinylm": 50,
    "small": 100,
    "medium": 200,
    "large": 500,
}


def _slice_for_size_class(data: list[dict], size_class: str | None) -> list[dict]:
    """Slice training data to the limit for a size class.

    Returns the full dataset when size_class is None (backward compat).
    Always returns a prefix — same ordering, not random.
    """
    if size_class is None:
        return data
    limit = _SIZE_CLASS_LIMITS.get(size_class, len(data))
    return data[:limit]


def collect_training_data(nerve_name: str, size_class: str | None = None) -> list[dict]:
    """Gather input/output pairs from test bank and successful episodes.

    Sources:
        1. Test bank — expected behaviors from qualification (high quality).
        2. Successful episodes — real user interactions that worked.

    Args:
        nerve_name: Name of the nerve to collect data for.
        size_class: Optional size class to slice the data. Each tier gets
            a prefix of the canonical bank (tinylm~50, small~100,
            medium~200, large~500). None returns everything.

    Returns:
        List of {"input": str, "output": str} dicts.
    """
    data = []

    try:
        from arqitect.memory.cold import ColdMemory
        cold = ColdMemory()

        # 1. Test bank — expected behaviors as training targets
        test_bank = cold.get_test_bank(nerve_name)
        for tc in test_bank:
            inp = tc.get("input", "")
            out = tc.get("output", tc.get("expected_behavior", ""))
            if inp and out:
                data.append({"input": inp, "output": out})

        # 2. Successful episodes from warm memory (Redis)
        try:
            import redis
            from arqitect.config.loader import get_redis_host_port
            _host, _port = get_redis_host_port()
            r = redis.Redis(host=_host, port=_port, decode_responses=True)
            episodes_raw = r.lrange(f"synapse:episodes:{nerve_name}", 0, 50)
            for ep_str in episodes_raw:
                try:
                    ep = json.loads(ep_str)
                    if ep.get("success") and ep.get("task") and ep.get("result_summary"):
                        data.append({
                            "input": ep["task"],
                            "output": ep["result_summary"],
                        })
                except (json.JSONDecodeError, TypeError):
                    continue
        except Exception:
            pass  # Redis may not be available

    except Exception as e:
        _log(f"[TUNER] Failed to collect training data for '{nerve_name}': {e}")

    return _slice_for_size_class(data, size_class)


def _get_base_model_path(role: str) -> str | None:
    """Get the GGUF model path for a nerve role."""
    from .model_registry import MODEL_REGISTRY, resolve_registry_key

    model_name = resolve_registry_key(role)
    entry = MODEL_REGISTRY.get(model_name)
    if not entry:
        return None

    path = os.path.join(_models_dir(), entry["file"])
    return path if os.path.exists(path) else None


def train_nerve_adapter(
    nerve_name: str,
    role: str = "tool",
    training_data: list[dict] | None = None,
    interrupted: threading.Event | None = None,
    lora_rank: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
) -> bool:
    """Train a LoRA adapter for a nerve. Returns True if training completed.

    Steps:
    1. Load GGUF into PyTorch via transformers (gguf_file= parameter)
    2. Wrap with peft LoRA
    3. Train on input/output pairs
    4. Save adapter as safetensors
    5. Convert to GGUF format for llama-cpp-python
    6. Unload PyTorch model to free memory

    Interrupt safety: checks interrupted.is_set() between each training step.
    If interrupted, discards partial adapter (never saves garbage).
    """
    # Resolve LoRA params from community config
    from arqitect.brain.adapters import get_tuning_config
    _cfg = get_tuning_config(role)
    if lora_rank is None:
        lora_rank = _cfg["lora_rank"]
    if epochs is None:
        epochs = _cfg["lora_epochs"]
    if lr is None:
        lr = _cfg["lora_lr"]

    if training_data is None:
        training_data = collect_training_data(nerve_name)

    min_examples = _get_min_examples(role)
    if len(training_data) < min_examples:
        _log(f"[TUNER] '{nerve_name}': only {len(training_data)} examples "
             f"(need {min_examples} for {role}). Skipping.")
        return False

    base_model_path = _get_base_model_path(role)
    if not base_model_path:
        _log(f"[TUNER] No base model found for role '{role}'")
        return False

    adapter_dir = os.path.join(_nerves_dir(), nerve_name, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    _log(f"[TUNER] Training LoRA adapter for '{nerve_name}' "
         f"({len(training_data)} examples, rank={lora_rank}, epochs={epochs})")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        # Check for interrupt before heavy loading
        if interrupted and interrupted.is_set():
            return False

        # Load GGUF into PyTorch — transformers >= 5.3 supports this directly
        _log(f"[TUNER] Loading {os.path.basename(base_model_path)} into PyTorch...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            gguf_file=os.path.basename(base_model_path),
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            gguf_file=os.path.basename(base_model_path),
        )

        if interrupted and interrupted.is_set():
            del model, tokenizer
            return False

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=_cfg["lora_dropout"],
            target_modules=_cfg["lora_target_modules"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _log(f"[TUNER] LoRA: {trainable:,} trainable params / {total:,} total "
             f"({100 * trainable / total:.2f}%)")

        # Prepare training data
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            if interrupted and interrupted.is_set():
                _log(f"[TUNER] Interrupted at epoch {epoch}")
                del model, tokenizer, optimizer
                return False

            total_loss = 0
            for i, example in enumerate(training_data):
                if interrupted and interrupted.is_set():
                    _log(f"[TUNER] Interrupted at epoch {epoch}, step {i}")
                    del model, tokenizer, optimizer
                    return False

                # Format as instruction-response pair
                text = f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=_cfg["training_max_length"], padding="max_length")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()

                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(training_data)
            _log(f"[TUNER] Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

        # Save adapter (safetensors format)
        safetensors_dir = os.path.join(adapter_dir, "_safetensors")
        os.makedirs(safetensors_dir, exist_ok=True)
        model.save_pretrained(safetensors_dir)
        _log(f"[TUNER] Saved safetensors adapter to {safetensors_dir}")

        # Convert safetensors -> GGUF for llama-cpp-python
        gguf_adapter_path = os.path.join(adapter_dir, "adapter.gguf")
        converted = _convert_adapter_to_gguf(safetensors_dir, gguf_adapter_path,
                                              base_model_path)

        # Cleanup PyTorch model to free memory
        del model, tokenizer, optimizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if converted:
            _log(f"[TUNER] LoRA adapter ready: {gguf_adapter_path}")
            return True
        else:
            _log(f"[TUNER] GGUF conversion failed — safetensors saved but not usable "
                 f"at inference. Manual conversion needed.")
            return False

    except Exception as e:
        _log(f"[TUNER] Training failed for '{nerve_name}': {e}")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False


def _convert_adapter_to_gguf(safetensors_dir: str, output_path: str,
                              base_model_path: str) -> bool:
    """Convert a safetensors LoRA adapter to GGUF format.

    Uses llama.cpp's convert_lora_to_gguf.py if available,
    otherwise falls back to manual conversion.
    """
    import subprocess

    # Try llama.cpp converter (usually installed with llama-cpp-python)
    converter_paths = [
        # Common locations
        os.path.expanduser("~/.local/bin/convert_lora_to_gguf.py"),
        "/usr/local/bin/convert_lora_to_gguf.py",
    ]

    # Also search in site-packages
    try:
        import llama_cpp
        pkg_dir = os.path.dirname(os.path.dirname(llama_cpp.__file__))
        converter_paths.append(os.path.join(pkg_dir, "scripts", "convert_lora_to_gguf.py"))
    except Exception:
        pass

    for converter in converter_paths:
        if os.path.exists(converter):
            try:
                result = subprocess.run(
                    [sys.executable, converter,
                     "--base", base_model_path,
                     "--lora", safetensors_dir,
                     "--output", output_path],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0 and os.path.exists(output_path):
                    return True
                _log(f"[TUNER] Converter failed: {result.stderr[:200]}")
            except Exception as e:
                _log(f"[TUNER] Converter error: {e}")

    # Fallback: try python-based conversion
    try:
        from llama_cpp import llama_lora_adapter_init
        # If llama-cpp-python can load safetensors directly, we're good
        # Copy the safetensors as-is and let the engine handle it
        import shutil
        for f in os.listdir(safetensors_dir):
            if f.endswith(".safetensors") or f == "adapter_config.json":
                shutil.copy2(os.path.join(safetensors_dir, f),
                           os.path.join(os.path.dirname(output_path), f))
        _log(f"[TUNER] Copied safetensors adapter (GGUF conversion unavailable)")
        return False
    except Exception:
        pass

    _log(f"[TUNER] No GGUF converter found. Adapter saved as safetensors only.")
    return False


def get_nerves_ready_for_training() -> list[dict]:
    """Find all nerves that have enough training data for LoRA fine-tuning.

    Returns list of {"name": str, "role": str, "data_count": int, "has_adapter": bool}
    """
    results = []
    try:
        from arqitect.memory.cold import ColdMemory
        cold = ColdMemory()

        rows = cold.conn.execute(
            "SELECT name, description, role FROM nerve_registry WHERE is_sense=0"
        ).fetchall()

        for row in rows:
            name = row[0]
            role = row[2] or "tool"
            data = collect_training_data(name)
            adapter_dir = os.path.join(_nerves_dir(), name, "adapter")
            has_adapter = os.path.exists(os.path.join(adapter_dir, "adapter.gguf"))

            min_ex = _get_min_examples(role)
            if len(data) >= min_ex:
                results.append({
                    "name": name,
                    "role": role,
                    "data_count": len(data),
                    "has_adapter": has_adapter,
                })

    except Exception as e:
        _log(f"[TUNER] Error scanning nerves: {e}")

    return results
