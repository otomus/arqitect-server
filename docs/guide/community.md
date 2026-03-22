# Community

Arqitect instances are a family. They share nerve bundles, adapters, and tools through a shared GitHub repository. Small instances consume what the community provides. Large instances synthesize, tune, and contribute back.

## The Community Repo

The community lives at `otomus/sentient-community` on GitHub. Its raw content is fetched via `https://raw.githubusercontent.com/otomus/sentient-community/main/`.

The local cache is stored at `{project_root}/.community/cache/`.

## Manifest

The manifest is the entry point. A single `manifest.json` at the repo root declares everything available.

```json
{
  "nerves": {
    "weather_nerve": {
      "description": "Weather forecasts and conditions",
      "role": "tool",
      "tags": ["weather", "api"],
      "tools": ["get_weather", "get_forecast"]
    }
  },
  "tools": {
    "get_weather": {
      "version": "1.0.0",
      "files": ["tool.json", "tool.py"],
      "runtime": "python"
    }
  },
  "adapters": { ... },
  "connectors": { ... }
}
```

### Nerve Entries

Each nerve entry in the manifest has:

| Field | Purpose |
|---|---|
| `description` | Domain-level description |
| `role` | `tool`, `creative`, or `code` |
| `tags` | Tags for filtering (including environment-exclusive tags) |
| `tools` | List of tool names this nerve uses |

### Environment-Exclusive Tags

Tags `iot` and `desktop` lock a nerve to a specific environment. A nerve tagged `iot` is only seeded when the configured environment is `iot`. Nerves with no exclusive tags are universal.

When the environment changes, nerves from previous seeds that no longer match are pruned — deleted from disk and cold memory.

## Sync Flow

### When Manifests Are Pulled

1. **Brain bootstrap** — `sync_manifest()` on startup
2. **Dream state** — community sync phase at the start of each dream cycle
3. **Manual** — `sync_all()` for full sync

### Bootstrap Sequence

On startup:

1. Fetch `manifest.json` from GitHub (10-second timeout)
2. Cache it locally at `.community/cache/manifest.json`
3. Seed tools — download missing tool directories from the manifest
4. Seed nerves — register community nerves in cold memory and write `nerve.py` files

**Lightweight seeding:** Nerve seeding uses only manifest metadata (description, role, tools). No network IO per nerve. Full bundles (system prompts, examples, adapters) are deferred to dream state or first invocation via `hydrate_nerve_bundle()`.

### Dream State Sync

During each dream cycle:

1. Pull latest manifest
2. Seed any new tools and nerves
3. Build dependency environments for tools with `.needs_build` markers
4. **Smart hydration** — only download full bundles for nerves that need work:
   - Invoked nerves scoring below 95%
   - Nerves with no adapter for the current model
   - Community nerves overlapping locally fabricated nerves (consolidation candidates)

## Nerve Bundles

A bundle is the full package for a nerve. Stored in `nerves/{name}/` in the community repo.

### Bundle Structure

```
nerves/weather_nerve/
├── bundle.json          # Identity, tools, metadata
├── test_cases.json      # Qualification test bank
├── tools/               # Tool implementations
│   └── get_weather/
│       └── tool.json
├── small/               # Size-class defaults
│   ├── context.json     # System prompt, few-shot examples, temperature
│   └── meta.json        # Tuning config, qualification thresholds
├── small/               # Model-specific overrides
│   └── qwen2.5-3b/
│       ├── context.json
│       ├── meta.json
│       └── adapter.gguf # LoRA adapter
└── medium/
    ├── context.json
    └── meta.json
```

### bundle.json

The nerve's identity and tool declarations.

```json
{
  "name": "weather_nerve",
  "version": "1.0.0",
  "description": "Weather forecasts and conditions",
  "role": "tool",
  "tags": ["weather", "api"],
  "authors": [{"github": "otomus"}],
  "arqitect_version": ">=0.1.0",
  "tools": [
    {
      "name": "get_weather",
      "spec": "mcp_tools/get_weather/spec.json",
      "implementations": {
        "python": "mcp_tools/get_weather/tool.py"
      }
    }
  ]
}
```

### test_cases.json

Qualification test bank. Each entry has an input and expected output/behavior.

```json
[
  {
    "input": "What's the weather in London?",
    "output": "{\"action\": \"call\", \"tool\": \"get_weather\", ...}",
    "category": "basic"
  }
]
```

### context.json

Per-size-class prompt configuration.

```json
{
  "system_prompt": "You are a weather expert. ...",
  "few_shot_examples": [...],
  "temperature": 0.3,
  "qualification_score": 0.85,
  "max_tokens": 2048,
  "conversation_window": 10,
  "message_truncation": 200
}
```

### meta.json

Tuning configuration and qualification thresholds.

```json
{
  "model": "qwen2.5-3b",
  "size_class": "small",
  "provider": "gguf",
  "has_lora": false,
  "tuning": {
    "min_training_examples": 20,
    "test_cases_per_batch": 5,
    "few_shot_limit": 5,
    "lora_rank": 8,
    "lora_epochs": 3,
    "lora_lr": 2e-4,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj"],
    "training_max_length": 512
  },
  "qualification": {
    "minimum_threshold": 0.7,
    "golden_threshold": 0.95,
    "golden_iterations": 5
  }
}
```

## Adapters

Adapters are model-specific prompts and tuning configurations for system roles (brain, nerve, creative, code, awareness, communication, vision).

### Directory Structure

```
adapters/
├── brain/
│   ├── small/
│   │   ├── context.json
│   │   ├── meta.json
│   │   └── test_bank.jsonl
│   └── small/
│       └── qwen2.5-3b/
│           ├── context.json
│           ├── meta.json
│           └── adapter.gguf
├── nerve/
│   ├── tinylm/
│   ├── small/
│   ├── medium/
│   └── large/
└── communication/
    └── ...
```

### Resolution Chain

When arqitect needs a prompt for a role, it follows this fallback chain:

```
{role}/{size_class}/{model_slug}/context.json
  → {role}/{size_class}/context.json
```

Model-specific first, then size-class defaults. The same chain applies to `meta.json`.

For tuning config, the resolution is deeper:

1. `ROLE_TUNING_OVERRIDES` — architecture-level LoRA targets per model type
2. Adapter `meta.json` — community source of truth
3. Nerve-specific `meta.json` — per-nerve overrides
4. Adapter `context.json` — temperature, max_tokens

### Role Mapping

Nerve roles map to adapter roles:

| Nerve Role | Adapter Role |
|---|---|
| `tool`, `data`, `scheduler` | `nerve` |
| `generative` | `creative` |
| Senses (brain, awareness, etc.) | themselves |

### Size Classes

Models are bucketed by parameter count:

| Size Class | Parameter Range | Example Models |
|---|---|---|
| `tinylm` | < 3B | 0.5B, 1B, 1.5B, 2B |
| `small` | 3B – 6B | 3B, 3.8B, 4B |
| `medium` | 6B – 32B | 7B, 8B, 13B, 14B |
| `large` | > 32B | 30B, 70B, 405B |

Cloud providers (Anthropic, OpenAI, etc.) always map to `large`.

Size class is derived from the model filename using regex to extract parameter counts (e.g., `qwen2.5-7b.gguf` → 7B → `medium`).

## Contribution

Arqitect instances contribute back to the community during dream state.

### What Gets Pushed

For each local nerve:

1. **New nerve** (not in community): Full bundle — `bundle.json`, `test_cases.json`, tool implementations, per-size-class `context.json` and `meta.json`, LoRA adapter
2. **Existing nerve, new model**: Model-specific adapter — `context.json`, `meta.json`, `adapter.gguf`
3. **Existing nerve, new language**: Tool implementations in the new language

For adapters: When an adapter's `qualification_score` exceeds the community's score for that model, the improved adapter is pushed.

### Contribution Gate

Nerves are not contributed until they have a trained LoRA adapter (`adapter.gguf` must exist on disk). This ensures only battle-tested nerves reach the community.

Contributions always go to `{role}/{size_class}/{model_name}/` — never to the size-class defaults.

### PR Flow

All contributions use GitHub PRs via `gh`:

1. Search for existing open PR from `@me` matching the nerve/adapter
2. If found: checkout the branch, update files, rebase, force-push
3. If not: create new branch, commit, push, create PR with `--fill --auto`

## Family Dynamics

### Size Classes and Capability Gating

Not all instances are equal. The permission system gates capabilities by model size:

| Capability | Required Size |
|---|---|
| Invoke nerves | Any |
| Use community bundles | Any |
| Synthesize new nerves | Medium, Large |
| Fabricate tools | Medium, Large |
| Consolidate (merge) nerves | Medium, Large |
| Generate test cases for LoRA | Medium, Large |
| Contribute to community | Any (with LoRA adapter) |

### How Small and Large Help Each Other

**Large instances** run frontier models. They:
- Synthesize new nerves from scratch
- Generate high-quality test banks
- Train LoRA adapters
- Contribute bundles, adapters, and tool implementations back to the community

**Small instances** run lightweight models. They:
- Consume community bundles and adapters tuned for their size class
- Benefit from LoRA adapters trained by larger instances
- Report usage data that helps prioritize community development
- Contribute model-specific adapters once they've trained their own LoRA

The community adapters bridge the gap. A nerve contributed by a large instance includes size-class-specific `context.json` files with prompts optimized for smaller models — shorter, more structured, with more examples. The `meta.json` includes tuning configs calibrated per size class (smaller LoRA rank, fewer training examples required).

### Tuning: Local vs Cloud

**Local models (GGUF)** get the full tuning pipeline: LoRA adapter training, few-shot optimization, and prompt refinement — all automated during dream state.

**Cloud providers (Anthropic, OpenAI, Gemini, etc.)** are tuned at the prompt level only — temperature, max_tokens, system prompts, and few-shot examples via `context.json`. LoRA training does not apply to cloud models. Cloud instances always map to the `large` size class and benefit from community-contributed prompts optimized for frontier models.

::: tip Related
See [Dream State](/guide/dream-state) for when sync and contribution happen, and [Architecture Overview](/architecture/overview) for how the community fits into the nervous system.
:::
