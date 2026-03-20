# Senses

Five core senses. Immutable. They can never be deleted, and they are always registered in the catalog. Senses are the platform's built-in perception and interaction capabilities.

Senses can be disabled individually via `senses.<name>.enabled: false` in `arqitect.yaml`, but they cannot be removed from the codebase.

## The 5 Senses

| Sense | Purpose | Model/Provider |
|---|---|---|
| **Sight** | Image analysis, screenshots, video streams | moondream (vision GGUF) |
| **Hearing** | Speech-to-text, text-to-speech, recording, playback | Whisper (STT), system TTS |
| **Touch** | File CRUD, OS operations, shell execution | Filesystem + subprocess |
| **Awareness** | Identity, permissions, self-reflection | Rules engine + brain LLM |
| **Communication** | Tone rewriting, personality, translation, summarization | Communication model |

---

## Sight

Vision and image understanding. Image generation is handled by the `image_generator` MCP tool, not this sense.

### Modes

| Mode | Input | Output |
|---|---|---|
| `image` (default) | `{"image_path": "..."}` or `{"base64": "..."}` | Description string from vision model |
| `screenshot` | `{"mode": "screenshot"}` | Captures screen, then analyzes |
| `stream` | `{"mode": "stream", "source": "camera\|url", "interval": 5}` | Background process publishing to `sense:sight:stream` Redis channel |
| `stop_stream` | `{"mode": "stop_stream"}` | Kills background stream by PID |

### Image Analysis

- Resolves relative paths against the sandbox directory
- Supports both file path and base64 input
- Uses the `vision` role model via `get_engine().generate_vision()`
- Vision prompt is loaded from the community adapter, falling back to `"Describe this image in detail."`

### Screenshot Capture

- macOS: `screencapture -x`
- Linux: tries `scrot`, then `gnome-screenshot`, then `import` (ImageMagick)

### Video Streaming

Launches a background worker that:
1. Captures frames at a configurable interval (default 5s)
2. Analyzes each frame with the vision model
3. Publishes observations to Redis `sense:sight:stream`
4. Camera sources: system camera (imagesnap/ffmpeg) or URL

---

## Hearing

Audio input and output. Supports multiple STT and TTS backends with graceful fallback.

### Modes

| Mode | Input | Output |
|---|---|---|
| `stt` | `{"mode": "stt", "audio_path": "..."}` | `{text, language, engine}` |
| `tts` | `{"mode": "tts", "text": "...", "voice": "default"}` | Plays audio through system speaker |
| `tts_file` | `{"mode": "tts_file", "text": "..."}` | `{audio_path}` — generates file for remote playback |
| `voices` | `{"mode": "voices"}` | `{voices, voices_detailed}` — lists available TTS voices |
| `record` | `{"mode": "record", "duration": 5}` | `{audio_path, duration}` — records from microphone |
| `play` | `{"mode": "play", "audio_path": "..."}` | Plays an audio file |

### STT Providers

| Provider | Detection | Notes |
|---|---|---|
| **whisper (Python)** | `import whisper` | OpenAI Whisper. Model size configurable (tiny/base/small). |
| **whisper.cpp** | `which whisper-cpp` or `which main` | C++ implementation, faster on CPU. |

### TTS Providers

| Provider | Platform | Detection |
|---|---|---|
| **say** | macOS | Pre-installed. Supports voice selection and speech rate. |
| **espeak / espeak-ng** | Linux | `which espeak` or `which espeak-ng` |

Voice preferences and speech rate are persisted in Redis (`synapse:sense_config`) and loaded on each invocation.

### Recording

Tries providers in order: `sox`, `ffmpeg`, `arecord`. Records mono 16kHz WAV to the sandbox directory.

### Playback

macOS uses `afplay`. Linux tries `aplay`, then `ffplay`.

### Voice Listing

Returns voices sorted by type: human voices first, then novelty voices (Zarvox, Bells, etc.). Each voice includes name, locale, sample text, and type classification.

---

## Touch

File system operations and OS actions. Sandboxed by default.

### Commands

| Command | Aliases | Description | Awareness-gated |
|---|---|---|---|
| `read` | `cat`, `show`, `open`, `view` | Read file contents | No |
| `write` | `save`, `create` | Write content to file | Yes (outside sandbox) |
| `append` | | Append content to file | Yes (outside sandbox) |
| `list` | `ls`, `dir` | List directory contents | No |
| `tree` | | Recursive directory listing (max depth 3) | No |
| `search` | `find`, `grep` | Glob pattern search (max 100 results) | No |
| `copy` | `cp` | Copy file or directory | No |
| `move` | `mv`, `rename` | Move or rename | No |
| `exists` | `check` | Check if path exists | No |
| `delete` | `rm`, `remove` | Delete file or directory | Always |
| `mkdir` | | Create directory | No |
| `info` | | File/directory metadata (size, modified, permissions) | No |
| `exec` | `run`, `execute` | Execute shell command | Always |
| `sysinfo` | | System info (platform, architecture, Python version, hostname) | No |

### Sandboxing

- All relative paths resolve against the sandbox directory
- System paths are always denied: `/etc`, `/usr`, `/bin`, `/sbin`, `/var`, `/System`, `/Library`, `/boot`, `/dev`, `/proc`, `/sys`
- Writes outside the sandbox require an awareness check
- Deletes always require an awareness check
- Shell execution always requires an awareness check, runs in the sandbox directory with a 30-second timeout

### Natural Language Parsing

Touch accepts both JSON and natural language input. Plain text like `read /some/path/file.py` or `list /tmp` is parsed into structured commands automatically.

---

## Awareness

Self-identity, ethical boundaries, and the permission system.

### Two Modes

**Permission check** — called by touch before destructive operations:

```json
{"action": "delete", "context": "path=/etc/hosts"}
```

Returns `{allowed, denied, reason}`. Checks against:
- `never_execute` — forbidden command patterns
- `never_delete` — protected paths
- `require_confirmation` — actions that need user confirmation

Rules are loaded from `awareness_rules.json` in the sense directory.

**Self-reflection** — answers identity questions ("who are you?"):

```json
{"query": "who are you?"}
```

Uses the brain LLM with personality data to generate natural, personality-infused responses. Context includes:
- Personality seed from `personality.json`, overlaid with evolved traits from cold memory
- Live inventory of nerves, senses, and tools
- Total task invocation count
- User profile

The system prompt is loaded from the community awareness adapter, injecting the runtime name and voice.

### Personality Seed

The personality seed (`personality.json`) defines:
- `core_identity` — name, archetype, fixed traits
- `voice` — default tone, humor style
- `trait_weights` — wit, swagger, warmth, formality (0-1 floats)

Evolved traits from cold memory (`facts` table, category `personality`) overlay the seed. Dream state personality reflection updates these over time.

---

## Communication

The voice layer. Rewrites every response for tone and personality. Also handles translation and summarization.

### Tones

| Tone | Style |
|---|---|
| `formal` | Professional, proper grammar, no contractions |
| `casual` | Conversational, contractions, relaxed |
| `enthusiastic` | Energetic, upbeat, dynamic language |
| `empathetic` | Warm, supportive, acknowledging |
| `neutral` | Straightforward, concise |

### Formats

| Format | Output |
|---|---|
| `text` | Plain text with tone-appropriate rewrite + optional emoji suffix |
| `card` | Markdown card with title, body, and footer |
| `emoji` | LLM-enhanced emoji insertion throughout the message |
| `gif` | Text response + GIF URL from Tenor free API |
| `translate` | Language translation (tone field carries target language) |
| `summarize` | Condenses long text to 1-3 sentences |

### Personality Evolution

Communication loads personality traits from two sources:
1. `personality.json` seed (archetype, humor style, trait weights)
2. Cold memory `personality` facts (evolved through dream state)

Trait weights control the personality instruction:
- `wit >= 0.6` activates full humor; `>= 0.3` allows light humor
- `swagger >= 0.5` makes responses confident and direct
- `warmth >= 0.5` adds genuine warmth
- `formality <= 0.3` keeps it casual; `>= 0.7` maintains professional register
- `verbosity <= 0.3` enforces concise short sentences

### Structured Data Protection

When the message is detected as structured data (code blocks, JSON, numeric results, file listings), the personality layer stays minimal. It may add a one-liner quip but never alters the data itself.

### Safety Filter

Communication output passes through a safety filter (`brain.safety.check_output`) that catches inappropriate content and sensitive data. Unsafe content is replaced with a refusal message.

---

## Sense Runtime

The sense runtime (`arqitect/senses/sense_runtime.py`) provides convenience wrappers for nerves to invoke senses without knowing the subprocess mechanics.

| Function | Sense | Default args |
|---|---|---|
| `see(image_path, prompt)` | Sight | prompt: "Describe this image" |
| `see_screenshot(prompt)` | Sight | mode: screenshot |
| `hear(audio_path)` | Hearing | mode: stt |
| `speak(text, voice)` | Hearing | mode: tts, voice: "default" |
| `touch(command, path, **kwargs)` | Touch | command: "read" |
| `check_awareness(action, context)` | Awareness | |
| `express(message, tone, fmt)` | Communication | tone: "neutral", format: "text" |
| `call_sense(sense_name, args)` | Any | Generic invocation by name |

Each function calls `_invoke_sense()`, which:
1. Resolves the sense path: `senses/{name}/nerve.py`
2. Runs it as a subprocess with `json.dumps(args)` as the argument
3. Parses the JSON output (60-second timeout)
4. Returns a dict

## Calibration Protocol

Every sense auto-calibrates on startup. The calibration protocol (`arqitect/senses/calibration_protocol.py`) provides shared utilities:

- `check_binary(name)` — checks if a binary is on PATH
- `check_python_module(name)` — checks if a Python module is importable
- `derive_status(capabilities)` — all available = `operational`, some = `degraded`, none = `unavailable`
- `build_result()` — standardized calibration result with sense name, timestamp, platform, status, capabilities, dependencies, config, user actions needed, and auto-installable packages

### Calibration Result Structure

```json
{
  "sense": "sight",
  "timestamp": 1711000000.0,
  "platform": "Darwin",
  "status": "operational",
  "capabilities": {
    "image_analysis": {"available": true, "provider": "vision", "notes": ""},
    "screenshot": {"available": true, "provider": "screencapture", "notes": ""}
  },
  "dependencies": {
    "vision": {"installed": true, "version": "latest"},
    "ffmpeg": {"installed": true, "path": "/usr/local/bin/ffmpeg"}
  },
  "config": {},
  "user_action_needed": [],
  "auto_installable": []
}
```

Each sense writes its calibration result to `calibration.json` in its directory and can report user actions needed (e.g., "Which camera device should I use?").

## Integration with Nerves

Every synthesized nerve has access to all senses through the template. The nerve's planner prompt includes a `_SENSE_ACTIONS_BLOCK` that teaches the LLM how to invoke senses:

```json
{"action": "use_sense", "sense": "see", "args": {"image_path": "/path/to/image.jpg"}}
```

When the planner returns a `use_sense` action, the nerve:
1. Maps the sense name to the corresponding runtime function
2. Calls the sense
3. Feeds the sense result back to the LLM for interpretation
4. Returns the interpreted response

::: tip Related
- [Nerves](/guide/nerves) — how nerves use senses
- [Getting Started](/guide/getting-started) — configuring senses in the wizard
- [Architecture Overview](/architecture/overview) — senses in the nervous system
:::
