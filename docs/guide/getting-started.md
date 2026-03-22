# Getting Started

## Prerequisites

- **Python 3.x**
- **Redis** — required for hot memory
- **Make** — for the setup commands

## Quick Start

The fastest way to get running:

```bash
make init
```

This launches the interactive setup wizard that walks you through every configuration choice. When it's done, it installs dependencies, syncs the community, and starts all services.

## The Init Wizard

The wizard is a 10-step interactive flow that generates your `arqitect.yaml`. It handles everything — from choosing your inference provider to configuring your AI's personality.

### Step 1: Environment

Where is arqitect running?

- **Desktop/Laptop** — full filesystem access available
- **Server** — sandboxed by default
- **IoT/Embedded** — minimal footprint, execution disabled

This choice drives defaults for the touch sense and resource allocation.

### Step 2: Inference Provider

The core decision. Choose what powers the brain and specialized roles.

**Uniform mode** — one provider for everything. Simple.

**Per-role mode** — mix providers. Run the brain on Anthropic, nerves on local GGUF, code generation on Groq. Each role (brain, nerve, coder, creative, communication) gets its own provider and model.

Supported providers:

| Provider | Type | Notes |
|---|---|---|
| **GGUF** | Local | No API key, runs on your hardware |
| **Anthropic** | Cloud | Claude models |
| **OpenAI** | Cloud | GPT models, configurable base URL |
| **Groq** | Cloud | Fast inference |
| **Google Gemini** | Cloud | Gemini models |
| **DeepSeek** | Cloud | |
| **Mistral** | Cloud | |
| **OpenRouter** | Cloud | Multi-model gateway |
| **xAI** | Cloud | Grok models |
| **Together AI** | Cloud | |

For GGUF, the wizard opens a native file picker to select your model file. For cloud providers, it asks for your API key.

### Step 3: Vision

Enable the sight sense. Three options:

- **Anthropic Claude Vision** — if you're already using Anthropic
- **Local moondream2** — ~2GB model, runs locally
- **Custom endpoint** — point to your own vision API

Or skip it entirely.

### Step 4: Hearing

Configure audio input/output.

**Speech-to-text**: OpenAI Whisper (local, free) or a cloud provider like Deepgram.

**Text-to-speech**: OpenAI TTS, Piper (local, free), ElevenLabs, or custom.

### Step 5: Personality

Give your arqitect a name and a voice.

Choose a preset:
- **Professional assistant** — clear, direct, helpful
- **Friendly companion** — warm, approachable
- **Technical expert** — precise, analytical
- **Custom** — define your own tone and traits

Set communication formality: formal, casual, or adaptive.

### Step 6: Connectors

How will users talk to arqitect? The dashboard is always included.

- **Telegram** — provide your bot token from @BotFather
- **WhatsApp** — enable and configure
- **Web SDK** — embed arqitect into your own website or app via the WebSocket bridge. Connect to `ws://your-host:3000` (or `wss://` with SSL), authenticate with a JWT, and exchange messages using the [bridge protocol](/guide/bridge). Supports text, voice, image, and real-time streaming of brain activity

### Step 7: Touch

What can arqitect access on the filesystem?

- **Full access** — home directory
- **Sandboxed** — project directory only (default)
- **Read-only**
- **No filesystem access**

Command execution can be enabled with an optional allowlist.

### Step 8: Embeddings

Choose the embedding model for semantic search and memory matching.

- **nomic-embed-text** — recommended, local
- **all-MiniLM-L6-v2** — lighter alternative
- **GGUF** — use the same model as the brain
- **OpenAI embeddings** — cloud
- **None** — keyword matching only

### Step 9: Storage

Configure where data lives.

- **Hot memory** — Redis URL (default: `redis://localhost:6379`)
- **Cold memory** — SQLite file or PostgreSQL connection
- **Warm memory** — SQLite file for episodes
- **Ports** — MCP server (default: 8100), dashboard bridge (default: 3000)

### Step 10: Admin

- Your name and email
- JWT secret — auto-generated, cryptographically secure
- Optional SMTP configuration for email notifications — highly recommended when using the Web SDK, since only users with verified identity can request fabricated nerves on demand (prevents anonymous users from overloading the system)

## What Happens Next

After the wizard finishes, `make setup` runs automatically:

1. Creates a virtual environment and installs dependencies
2. Syncs the community manifest — nerve bundles, tools, adapters
3. Seeds community tools
4. Starts all services:
   - **Redis** — hot memory
   - **MCP Server** — tool host
   - **Brain daemon** — the orchestrator
   - **WebSocket Bridge** — dashboard communication
   - **Connectors** — Telegram, WhatsApp (if enabled)

## Bootstrap Sequence

On startup, the brain initializes itself:

1. **Session bootstrap** — detects location and timezone (from cold memory or geolocation API)
2. **Sense bootstrap** — ensures all 5 senses exist, registers them, runs calibration
3. **Model preload** — loads model weights for each role (local) or validates API keys (cloud)
4. **Community sync** — pulls the latest manifest

## Connect

Once services are running:

1. Open the [Arqitect Dashboard](https://otomus.github.io/arqitect-dashboard/)
2. Go to Settings and add your server address
3. Start talking

## Manual Setup

If you prefer to skip the wizard:

```bash
# Clone and install
git clone https://github.com/otomus/arqitect-server.git
cd arqitect-server
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Copy and edit the config manually
cp arqitect.yaml.example arqitect.yaml

# Start services
make start
```

## Configuration File

The wizard generates `arqitect.yaml` in your project root. You can edit it at any time — changes take effect on restart. See [Configuration](/guide/configuration) for the full reference.

::: tip Reconfigure at any time
You can re-run `make init` whenever you want to change your configuration. The wizard will walk you through the same steps and regenerate `arqitect.yaml` with your new choices.
:::
