# Configuration

All configuration lives in `arqitect.yaml` at the project root. The quickest way to generate it is `make init`, which runs the interactive wizard. You can also copy the example and edit manually:

```bash
cp arqitect.yaml.example arqitect.yaml
```

## How Config Loading Works

Configuration is resolved in three layers, merged with a deep-merge strategy:

1. **Defaults** — hardcoded in `arqitect/config/defaults.py`. Every key has a value.
2. **arqitect.yaml** — your overrides. Only specify what you want to change.
3. **Environment variables** — `ARQITECT_PROJECT_ROOT` overrides the project root path.

The loader finds the project root by walking up from `cwd` looking for `arqitect.yaml`. If not found, it checks `ARQITECT_PROJECT_ROOT`, then falls back to `cwd`.

Config values are accessed via dot-separated paths:

```python
from arqitect.config.loader import get_config
provider = get_config("inference.provider")  # "gguf"
```

## Top-Level Settings

```yaml
name: Arqitect          # Instance name
environment: server     # server | desktop | iot
```

The `environment` value drives defaults for the touch sense — `server` sandboxes filesystem access, `desktop` gives full access.

## Inference

The core decision: what powers the brain and each specialized role.

```yaml
inference:
  provider: gguf        # Default provider for all roles

  models:
    brain:
      file: 'Qwen2.5-Coder-7B.gguf'
      source: 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF'
    nerve:
      file: ''
      source: ''
    coder:
      file: ''
      source: ''
    creative:
      file: ''
      source: ''
    communication:
      file: ''
      source: ''
    vision:
      file: ''
      source: ''
      chat_handler: ''   # e.g. 'moondream' for moondream2
      mmproj: ''          # multimodal projection file
    embedding:
      file: ''
      source: ''
    image_gen:
      file: ''
      source: ''
      backend: ''         # e.g. 'stable_diffusion'

  roles:
    brain:
      provider: null      # Override provider per role
      model: null          # Override model per role
    nerve:
      provider: null
      model: null
    coder:
      provider: null
      model: null
    creative:
      provider: null
      model: null
    communication:
      provider: null
      model: null
```

### Model Format

Each model entry can be a simple string or a dict:

```yaml
# String — just the filename
brain: 'Qwen2.5-Coder-7B.gguf'

# Dict — filename plus metadata for auto-download
brain:
  file: 'Qwen2.5-Coder-7B.gguf'
  source: 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF'
```

If a model file is missing, the `source` field is used to auto-download from HuggingFace.

### Per-Role Providers

Mix providers across roles. Run the brain on Anthropic, nerves on local GGUF, code on Groq:

```yaml
inference:
  provider: gguf          # Default
  roles:
    brain:
      provider: anthropic
      model: claude-sonnet-4-20250514
    coder:
      provider: groq
      model: llama-3.3-70b-versatile
```

### Supported Providers

| Provider | Key | Type |
|---|---|---|
| GGUF | `gguf` | Local |
| Anthropic | `anthropic` | Cloud |
| OpenAI | `openai` | Cloud |
| Groq | `groq` | Cloud |
| Google Gemini | `google_gemini` | Cloud |
| DeepSeek | `deepseek` | Cloud |
| Mistral | `mistral` | Cloud |
| OpenRouter | `openrouter` | Cloud |
| xAI | `xai` | Cloud |
| Together AI | `together_ai` | Cloud |

## Personality

```yaml
personality:
  name: ''                  # Display name for the AI
  preset: ''                # professional | friendly | technical | custom
  tone: ''                  # Overall tone descriptor
  traits: []                # List of trait strings
  communication_style:
    emoji_level: moderate   # none | minimal | moderate | heavy
    formality: casual       # formal | casual | adaptive
    verbosity: concise      # concise | balanced | verbose
```

Personality affects how the communication sense rewrites responses. The `preset` value loads a predefined set of traits. Use `custom` to define your own.

## Senses

Each sense can be enabled or disabled. Disabled senses still exist but report as unavailable during calibration.

```yaml
senses:
  sight:
    enabled: true
    provider: ''            # anthropic | moondream | custom endpoint

  hearing:
    enabled: true
    stt: ''                 # whisper | deepgram | custom
    tts: ''                 # openai | piper | elevenlabs | custom

  touch:
    enabled: true
    environment: server
    filesystem:
      access: sandboxed     # full | sandboxed | readonly | none
      root: ./sandbox       # Root path for sandboxed access
    execution:
      enabled: true
      allowlist: []         # Empty = allow all; list = restrict to these commands

  awareness:
    enabled: true

  communication:
    enabled: true
```

### Touch Filesystem Modes

| Mode | Description |
|---|---|
| `full` | Home directory access |
| `sandboxed` | Project directory only (default for `server` environment) |
| `readonly` | Read-only access |
| `none` | No filesystem access |

## Connectors

```yaml
connectors:
  telegram:
    enabled: false
    bot_name: Arqitect
    bot_aliases: []              # Alternative names the bot responds to
    whitelisted_users: []        # Empty = allow all
    whitelisted_groups: []
    monitor_groups: []           # Groups to monitor passively

  whatsapp:
    enabled: false
    bot_name: Arqitect
    bot_aliases: []
    whitelisted_users: []
    whitelisted_groups: []
    monitor_groups: []
```

## Embeddings

```yaml
embeddings:
  provider: gguf               # gguf | openai | none
  model: Qwen2.5-Coder-7B.gguf
```

Used for semantic search across memory and nerve matching. Set `provider: none` to fall back to keyword matching.

## Storage

```yaml
storage:
  hot:
    url: redis://localhost:6379  # Redis URL for hot memory
  cold:
    path: arqitect_memory.db    # SQLite file for cold memory (knowledge graph)
  warm:
    path: episodes.db           # SQLite file for warm memory (task episodes)
```

All SQLite paths are relative to the project root (or the `paths.memory` config if set).

## Ports

```yaml
ports:
  mcp: 8100                     # MCP tool server
  bridge: 3000                  # WebSocket bridge (dashboard)
```

## SSL

```yaml
ssl:
  cert: ''                      # Path to SSL certificate
  key: ''                       # Path to SSL private key
```

When both are set, the bridge serves over WSS.

## Admin

```yaml
admin:
  name: ''                      # Your name
  email: ''                     # Your email
```

## Secrets

```yaml
secrets:
  jwt_secret: ''                # Auto-generated by wizard
  telegram_bot_token: ''        # From @BotFather
  anthropic_api_key: ''
  openai_api_key: ''
  openai_base_url: https://api.openai.com/v1
  groq_api_key: ''
  deepseek_api_key: ''
  mistral_api_key: ''
  openrouter_api_key: ''
  google_ai_api_key: ''
  xai_api_key: ''
  together_api_key: ''
  smtp:
    host: smtp.gmail.com
    port: 587
    user: ''
    password: ''
    from: ''
```

Never commit real secret values. The wizard generates `arqitect.yaml` which is gitignored. The `.example` file shows the structure with empty values.

Secrets are accessed via:

```python
from arqitect.config.loader import get_secret
key = get_secret("anthropic_api_key")
```

::: tip Related
See [Getting Started](/guide/getting-started) for the interactive wizard walkthrough.
:::
