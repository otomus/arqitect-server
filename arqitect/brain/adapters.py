"""Adapter resolver — loads community adapter prompts with fallback chain.

Directory structure in community repo:
    adapters/{role}/{size_class}/              — size class defaults
    adapters/{role}/{size_class}/{model_name}/ — model-specific tuning

Resolution order:
    {role}/{size_class}/{model_name} → {role}/{size_class}

Contributions always go to {role}/{size_class}/{model_name}/ — never to the
size class defaults.
"""

import json
import os
import urllib.request
import urllib.error

from arqitect.brain.community import _cache_dir, COMMUNITY_RAW_URL

# Roles that have community adapters
ADAPTER_ROLES = ("brain", "nerve", "creative", "code", "awareness", "communication", "vision")

# Valid size classes (directories in community repo)
SIZE_CLASSES = ("tinylm", "small", "medium", "large")

# ── Role-based LoRA target overrides ──────────────────────────────────
# The classification determines what behavior we're training for.
# Different roles benefit from adapting different layers:
#   - tool/nerve: precise structured I/O → attention steering (q, v)
#   - code: syntax + structure → attention + MLP for richer code patterns
#   - creative: diverse generation → wide targets, higher rank
#   - vision: different architecture (moondream2 uses fused projections)
#   - communication: tone/style → attention layers, lower rank
#   - brain: routing accuracy → attention focus, conservative rank
#
# These are architecture defaults only — the community meta.json
# overrides everything else. These just set lora_target_modules
# per model architecture when community hasn't specified them.

ROLE_TUNING_OVERRIDES = {
    "tool": {
        # Precise structured output — focus attention steering
        "lora_target_modules": ["q_proj", "v_proj"],
    },
    "code": {
        # Syntax accuracy + structured generation — add MLP layers
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_rank": 16,
        "default_temperature": 0.2,
    },
    "creative": {
        # Diverse generation — wider targets, higher rank, warmer temperature
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        "lora_rank": 16,
        "default_temperature": 0.7,
    },
    "vision": {
        # Moondream2 architecture — fused attention projections
        "lora_target_modules": ["qkv_proj", "out_proj"],
        "lora_rank": 8,
        "training_max_length": 384,
    },
    "communication": {
        # Tone/style adaptation — attention layers, conservative
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_rank": 4,
        "default_temperature": 0.5,
    },
    "brain": {
        # Routing decisions — attention focus, low rank to avoid overfitting
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_rank": 4,
        "lora_lr": 1e-4,
        "default_temperature": 0.2,
    },
    "awareness": {
        # Self-reflection — similar to brain
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_rank": 4,
        "default_temperature": 0.3,
    },
}

# Map arbitrary nerve roles to community adapter roles.
# Senses map to themselves (brain, awareness, etc.).
# All other nerve roles map to "nerve" (the generic nerve adapter).
_ROLE_TO_ADAPTER = {
    "tool": "nerve", "data": "nerve", "scheduler": "nerve",
    "generative": "creative",
}


def _resolve_adapter_role(role: str) -> str:
    """Map a nerve role to its community adapter role."""
    if role in ADAPTER_ROLES:
        return role
    return _ROLE_TO_ADAPTER.get(role, "nerve")


def _apply_meta_to_config(base: dict, meta: dict):
    """Extract tuning + qualification fields from a meta.json into config dict."""
    tuning = meta.get("tuning", {})
    for key in ("min_training_examples", "test_cases_per_batch", "few_shot_limit",
                 "lora_rank", "lora_epochs", "lora_lr", "lora_dropout",
                 "lora_target_modules", "training_max_length",
                 "warmup_steps", "eval_split", "scheduler", "quantization",
                 "batch_size", "gradient_accumulation_steps"):
        if key in tuning:
            base[key] = tuning[key]

    qual = meta.get("qualification", {})
    _QUAL_MAP = {
        "minimum_threshold": "qualification_threshold",
        "golden_threshold": "improvement_threshold",
        "golden_iterations": "max_qualification_iterations",
        "min_iterations": "max_reconciliation_iterations",
        "qualification_timeout": "qualification_timeout",
        "low_quality_threshold": "low_quality_threshold",
        "merge_threshold": "merge_threshold",
        "improvement_threshold": "improvement_threshold",
    }
    for src, dest in _QUAL_MAP.items():
        if src in qual:
            base[dest] = qual[src]


def get_tuning_config(role: str, nerve_name: str | None = None) -> dict:
    """Get tuning config from the community.

    Resolution chain (each layer overlays the previous):
    1. ROLE_TUNING_OVERRIDES — architecture-level LoRA targets per model type
    2. Adapter meta.json — adapters/{adapter_role}/{size_class}/meta.json
    3. Nerve meta.json — nerves/{nerve_name}/meta.json (if nerve_name given)
    4. Adapter context.json — temperature, max_tokens

    Arbitrary nerve roles are mapped to adapter roles:
        tool/data/scheduler → nerve, generative → creative, etc.
    Senses (brain, awareness, etc.) map to themselves.
    """
    adapter_role = _resolve_adapter_role(role)

    # 1. Architecture defaults (LoRA target modules per model type)
    base = dict(ROLE_TUNING_OVERRIDES.get(role, {}))

    # 2. Adapter-level meta.json (community source of truth)
    adapter_meta = resolve_meta(adapter_role)
    if adapter_meta:
        _apply_meta_to_config(base, adapter_meta)

    # 3. Nerve-specific meta.json overlays adapter defaults
    if nerve_name:
        nerve_meta_path = os.path.join(_cache_dir(), "nerves", nerve_name, "meta.json")
        nerve_meta = _load_json(nerve_meta_path)
        if nerve_meta:
            _apply_meta_to_config(base, nerve_meta)

    # 4. Overlay inference defaults from context.json
    ctx = resolve_prompt(adapter_role)
    if ctx:
        if "temperature" in ctx:
            base["default_temperature"] = ctx["temperature"]
        if "max_tokens" in ctx:
            base["default_max_tokens"] = ctx["max_tokens"]

    return base

# Files per adapter
_ADAPTER_FILES = ("context.json", "meta.json", "test_bank.jsonl")


def _model_slug(model_file: str) -> str:
    """Derive a stable directory-safe model slug from a model filename.

    e.g. 'Qwen2.5-Coder-7B.gguf' → 'qwen2.5-coder-7b'
         'moondream2-text-model-f16.gguf' → 'moondream2-text-model-f16'
    """
    slug = model_file.lower()
    for ext in (".gguf", ".bin", ".safetensors"):
        if slug.endswith(ext):
            slug = slug[:-len(ext)]
    return slug.strip().replace(" ", "-")


def _get_model_file_for_role(role: str) -> str | None:
    """Get the raw model filename for a role from registry or yaml config."""
    try:
        from arqitect.inference.model_registry import MODEL_REGISTRY
        entry = MODEL_REGISTRY.get(role)
        if entry:
            return entry.get("file", "")
    except ImportError:
        pass
    try:
        from arqitect.config.loader import get_model_for_role
        return get_model_for_role(role) or None
    except ImportError:
        pass
    return None


def get_model_name_for_role(role: str) -> str | None:
    """Get the model slug for a role (used as directory name for contributions)."""
    f = _get_model_file_for_role(role)
    return _model_slug(f) if f else None


def _adapter_cache_dir(role: str, *path_parts: str) -> str:
    """Build cache path: .community/cache/adapters/{role}/{parts...}"""
    return os.path.join(_cache_dir(), "adapters", role, *path_parts)


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    results = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    return results


def _load_context(role: str, *path_parts: str) -> dict | None:
    """Load a cached adapter context.json."""
    return _load_json(os.path.join(_adapter_cache_dir(role, *path_parts), "context.json"))


def _load_meta(role: str, *path_parts: str) -> dict | None:
    """Load a cached adapter meta.json."""
    return _load_json(os.path.join(_adapter_cache_dir(role, *path_parts), "meta.json"))


def _load_test_bank(role: str, *path_parts: str) -> list[dict]:
    """Load a cached adapter test_bank.jsonl."""
    return _load_jsonl(os.path.join(_adapter_cache_dir(role, *path_parts), "test_bank.jsonl"))


# ── Size class mapping ──────────────────────────────────────────────

def _model_to_size_class(model_name: str) -> str | None:
    """Map a model name to a size class for adapter resolution.

    Derives from parameter count in the model filename.
    Models without a recognizable count need an explicit entry
    in _MODEL_SIZE_OVERRIDES.
    """
    if not model_name:
        return None
    name = model_name.lower()
    _MODEL_SIZE_OVERRIDES = {
        "moondream": "small",
    }
    for pattern, size in _MODEL_SIZE_OVERRIDES.items():
        if pattern in name:
            return size
    if any(s in name for s in ("0.5b", "1b", "1.5b")):
        return "tinylm"
    if any(s in name for s in ("3b", "4b")):
        return "small"
    if any(s in name for s in ("7b", "8b", "13b")):
        return "medium"
    if any(s in name for s in ("30b", "34b", "70b")):
        return "large"
    return None


def get_model_size_class(role: str) -> str | None:
    """Get the size class for the model currently assigned to a role."""
    f = _get_model_file_for_role(role)
    return _model_to_size_class(f) if f else None


def get_active_variant(role: str) -> str:
    """Get the size class for this role's model.

    Always derived from config — never hardcoded.
    Raises ValueError if the model can't be mapped.
    """
    size = get_model_size_class(role)
    if not size:
        model = _get_model_file_for_role(role)
        raise ValueError(
            f"Cannot determine size class for role '{role}' "
            f"(model: '{model}'). Add a parameter count (e.g. 7B) "
            f"to the model name or update _model_to_size_class()."
        )
    return size


# ── Sync (pull from community) ──────────────────────────────────────

def sync_adapter(role: str, size_class: str, model_name: str | None = None) -> bool:
    """Download an adapter's files from the community repo.

    If model_name is given, syncs from {role}/{size_class}/{model_name}/.
    Otherwise syncs the size class defaults from {role}/{size_class}/.
    """
    parts = [size_class, model_name] if model_name else [size_class]
    dest_dir = _adapter_cache_dir(role, *parts)
    os.makedirs(dest_dir, exist_ok=True)

    remote_path = f"{role}/{'/'.join(parts)}"
    success = False
    for fname in _ADAPTER_FILES:
        url = f"{COMMUNITY_RAW_URL}/adapters/{remote_path}/{fname}"
        local_path = os.path.join(dest_dir, fname)
        try:
            urllib.request.urlretrieve(url, local_path)
            if fname == "context.json":
                success = True
        except urllib.error.HTTPError:
            pass

    if success:
        print(f"[ADAPTERS] Synced adapter: {remote_path}")
    return success


def sync_all_adapters() -> int:
    """Download adapters that runtime will actually use.

    For each role, syncs exactly what resolve_prompt would look up:
    1. {role}/{size_class}/{model_slug}/ — model-specific
    2. {role}/{size_class}/ — size class defaults (fallback)

    Called during brain bootstrap alongside manifest sync.
    Returns the number of adapters synced.
    """
    synced = 0
    for role in ADAPTER_ROLES:
        size = get_active_variant(role)
        model_slug = get_model_name_for_role(role)
        # Size class defaults (fallback)
        if sync_adapter(role, size):
            synced += 1
        # Model-specific
        if model_slug:
            if sync_adapter(role, size, model_slug):
                synced += 1
    if synced:
        print(f"[ADAPTERS] Synced {synced} adapter(s)")
    return synced


# ── Resolve (read at runtime) ───────────────────────────────────────

def resolve_prompt(role: str, model_name: str | None = None, size_class: str | None = None) -> dict | None:
    """Resolve the best adapter prompt for a given role.

    Fallback chain:
        {role}/{size_class}/{model_name} → {role}/{size_class}
    """
    effective_size = size_class or get_model_size_class(role) or get_active_variant(role)
    model_slug = model_name or get_model_name_for_role(role)

    # 1. Model-specific
    if model_slug:
        ctx = _load_context(role, effective_size, model_slug)
        if ctx:
            return ctx

    # 2. Size class defaults
    ctx = _load_context(role, effective_size)
    if ctx:
        return ctx

    return None


def resolve_meta(role: str) -> dict | None:
    """Load the adapter meta.json for a role.

    Merges base (role/size_class) with model-specific (role/size_class/model)
    so base fields aren't lost when the model-specific meta only overrides some.
    """
    size = get_active_variant(role)
    model_slug = get_model_name_for_role(role)

    base = _load_meta(role, size)
    if not model_slug:
        return base

    specific = _load_meta(role, size, model_slug)
    if not specific:
        return base
    if not base:
        return specific

    # Deep merge: base fields are kept unless overridden by model-specific
    merged = dict(base)
    for key, val in specific.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def resolve_test_bank(role: str) -> list[dict]:
    """Load the adapter test_bank.jsonl for a role."""
    size = get_active_variant(role)
    model_slug = get_model_name_for_role(role)

    if model_slug:
        tests = _load_test_bank(role, size, model_slug)
        if tests:
            return tests

    return _load_test_bank(role, size)


# ── Nerve-specific resolution ─────────────────────────────────────

def _nerve_cache_dir(nerve_name: str, *path_parts: str) -> str:
    """Build cache path: .community/cache/nerves/{nerve_name}/{parts...}"""
    return os.path.join(_cache_dir(), "nerves", nerve_name, *path_parts)


def resolve_nerve_prompt(nerve_name: str, role: str) -> dict | None:
    """Resolve the best prompt for a nerve from its cached community directory.

    Fallback chain (mirrors resolve_prompt):
        nerves/{name}/{size_class}/{model_slug}/context.json
      → nerves/{name}/{size_class}/context.json
    """
    try:
        size_class = get_active_variant(role)
    except ValueError:
        return None
    model_slug = get_model_name_for_role(role)

    if model_slug:
        ctx = _load_json(_nerve_cache_dir(nerve_name, size_class, model_slug, "context.json"))
        if ctx:
            return ctx

    return _load_json(_nerve_cache_dir(nerve_name, size_class, "context.json"))


def resolve_nerve_meta(nerve_name: str, role: str) -> dict | None:
    """Resolve the best meta.json for a nerve from its cached community directory.

    Merges base (size_class) with model-specific override, same as resolve_meta.
    """
    try:
        size_class = get_active_variant(role)
    except ValueError:
        return None
    model_slug = get_model_name_for_role(role)

    base = _load_json(_nerve_cache_dir(nerve_name, size_class, "meta.json"))
    if not model_slug:
        return base

    specific = _load_json(_nerve_cache_dir(nerve_name, size_class, model_slug, "meta.json"))
    if not specific:
        return base
    if not base:
        return specific

    merged = dict(base)
    for key, val in specific.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


# ── Convenience getters ─────────────────────────────────────────────

def get_temperature(role: str) -> float:
    ctx = resolve_prompt(role)
    return ctx["temperature"]


def get_max_tokens(role: str) -> int:
    """Get max output tokens from the community adapter for a role."""
    ctx = resolve_prompt(role)
    return ctx["max_tokens"]


_DEFAULT_CONVERSATION_WINDOW = 10
_DEFAULT_MESSAGE_TRUNCATION = 200


def get_conversation_window(role: str) -> int:
    """Get the sliding conversation window size from the community adapter.

    Controls how many recent messages are injected into the LLM prompt
    for multi-turn context. Smaller models get fewer messages to save
    context budget.
    """
    ctx = resolve_prompt(role)
    if ctx:
        return ctx.get("conversation_window", _DEFAULT_CONVERSATION_WINDOW)
    return _DEFAULT_CONVERSATION_WINDOW


def get_message_truncation(role: str) -> int:
    """Get max character length per conversation message from the community adapter.

    Longer messages are truncated to fit within the model's context window.
    """
    ctx = resolve_prompt(role)
    if ctx:
        return ctx.get("message_truncation", _DEFAULT_MESSAGE_TRUNCATION)
    return _DEFAULT_MESSAGE_TRUNCATION


def get_json_mode(role: str) -> bool:
    meta = resolve_meta(role)
    if meta:
        caps = meta.get("capabilities", {})
        return caps.get("json_mode", False)
    return False


def get_description(role: str) -> str:
    meta = resolve_meta(role)
    if meta and meta.get("description"):
        return meta["description"]
    return ""


def get_qualification_score(role: str) -> float:
    ctx = resolve_prompt(role)
    return float(ctx.get("qualification_score", 0)) if ctx else 0.0


# ── Save (local persistence for dream state) ────────────────────────

def _save_to(dest_dir: str, context: dict | None = None, meta: dict | None = None, tests: list[dict] | None = None):
    """Write adapter files to a local cache directory."""
    os.makedirs(dest_dir, exist_ok=True)
    if context is not None:
        with open(os.path.join(dest_dir, "context.json"), "w") as f:
            json.dump(context, f, indent=2)
    if meta is not None:
        with open(os.path.join(dest_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    if tests is not None:
        with open(os.path.join(dest_dir, "test_bank.jsonl"), "w") as f:
            for entry in tests:
                f.write(json.dumps(entry) + "\n")


def save_model_adapter(role: str, context: dict | None = None, meta: dict | None = None, tests: list[dict] | None = None):
    """Save tuned adapter files to the model-specific cache directory.

    Always writes to {role}/{size_class}/{model_name}/ — never to the
    size class defaults.
    """
    size = get_active_variant(role)
    model_slug = get_model_name_for_role(role)
    if not model_slug:
        raise ValueError(f"Cannot determine model name for role '{role}'")
    dest_dir = _adapter_cache_dir(role, size, model_slug)
    _save_to(dest_dir, context, meta, tests)


# Back-compat aliases used by consolidate.py
def save_context(role: str, context: dict, variant: str):
    """Save context.json to model-specific directory."""
    model_slug = get_model_name_for_role(role)
    if not model_slug:
        raise ValueError(f"Cannot determine model name for role '{role}'")
    dest_dir = _adapter_cache_dir(role, variant, model_slug)
    _save_to(dest_dir, context=context)


# ── Meta builder ─────────────────────────────────────────────────────

def build_meta_json(role: str, model_slug: str, size_class: str,
                    score: float = 0, has_lora: bool = False,
                    existing_meta: dict | None = None) -> dict:
    """Build a meta.json dict following adapter_meta.schema.json.

    Identical for nerves and system senses (brain, awareness, etc.) —
    both are LLM-based and need the same tuning + qualification config
    so anyone pulling can tune for their model.
    """
    cfg = get_tuning_config(role)

    meta = dict(existing_meta) if existing_meta else {}
    meta["model"] = model_slug
    meta["size_class"] = size_class
    meta["provider"] = "gguf"
    meta["has_lora"] = has_lora
    if "contributor" not in meta:
        meta["contributor"] = {"github": "otomus"}

    meta["tuning"] = {
        "min_training_examples": cfg["min_training_examples"],
        "test_cases_per_batch": cfg["test_cases_per_batch"],
        "few_shot_limit": cfg["few_shot_limit"],
        "lora_rank": cfg["lora_rank"],
        "lora_epochs": cfg["lora_epochs"],
        "lora_lr": cfg["lora_lr"],
        "lora_dropout": cfg["lora_dropout"],
        "lora_target_modules": cfg["lora_target_modules"],
        "training_max_length": cfg["training_max_length"],
    }
    for opt_key in ("warmup_steps", "eval_split", "scheduler",
                    "quantization", "batch_size", "gradient_accumulation_steps"):
        if opt_key in cfg:
            meta["tuning"][opt_key] = cfg[opt_key]

    meta["qualification"] = {
        "minimum_threshold": cfg["qualification_threshold"],
        "golden_threshold": cfg["improvement_threshold"],
        "golden_iterations": cfg["max_qualification_iterations"],
        "min_iterations": cfg["max_reconciliation_iterations"],
        "qualification_timeout": cfg["qualification_timeout"],
        "low_quality_threshold": cfg["low_quality_threshold"],
        "merge_threshold": cfg["merge_threshold"],
        "improvement_threshold": cfg["improvement_threshold"],
    }

    if score:
        meta["qualification"]["current_score"] = round(score, 4)

    return meta


# ── Contribution helpers ─────────────────────────────────────────────

def get_contribution_path(role: str) -> tuple[str, str, str]:
    """Return (size_class, model_slug, relative_dir) for contribution.

    The relative_dir is the path under the community repo's adapters/ dir
    where this role's model-specific adapter should be contributed.
    """
    size = get_active_variant(role)
    model_slug = get_model_name_for_role(role)
    if not model_slug:
        raise ValueError(f"Cannot determine model name for role '{role}'")
    rel_dir = os.path.join(role, size, model_slug)
    return size, model_slug, rel_dir


# ── Tuning query ─────────────────────────────────────────────────────

def list_adapters_needing_tuning() -> list[dict]:
    """Return adapters whose qualification_score < improvement_threshold.

    Each entry: {role, context, current_score, variant, model_slug}.
    Loads from model-specific dir first, falls back to size class defaults
    as starting point for tuning.
    """
    result = []
    for role in ADAPTER_ROLES:
        variant = get_active_variant(role)
        model_slug = get_model_name_for_role(role)

        # Try model-specific first
        ctx = None
        if model_slug:
            ctx = _load_context(role, variant, model_slug)

        # Fall back to size class defaults as starting point
        if not ctx:
            ctx = _load_context(role, variant)

        if not ctx:
            continue

        cfg = get_tuning_config(role)
        score = float(ctx.get("qualification_score", 0))
        if score < cfg["improvement_threshold"]:
            result.append({
                "role": role,
                "context": ctx,
                "current_score": score,
                "variant": variant,
                "model_slug": model_slug,
            })
    return result
