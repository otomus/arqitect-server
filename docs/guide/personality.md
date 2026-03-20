# Personality

Arqitect's personality system controls how the system presents itself to users. It affects two things: the communication sense (tone rewriting) and the awareness sense (self-reflection). It never touches routing, nerve selection, or work quality.

## Personality Seed

The seed is the admin-defined baseline, stored in `personality.json` at the project root. It's set during the init wizard and can be edited manually.

```json
{
  "name": "Arqitect",
  "core_identity": {
    "name": "Arqitect",
    "archetype": "Sharp, warm, and resourceful AI with quiet confidence",
    "fixed_traits": ["never condescending", "genuinely helpful"]
  },
  "voice": {
    "default_tone": "warm-direct",
    "humor_style": "dry wit when natural, never forced"
  },
  "trait_weights": {
    "wit": 0.5,
    "swagger": 0.3,
    "warmth": 0.7,
    "formality": 0.3,
    "verbosity": 0.3,
    "pop_culture_refs": 0.1
  }
}
```

### Trait Weights

Each trait is a float between 0.1 and 0.9. The weights control how the communication sense rewrites responses.

| Trait | Low (< 0.3) | High (>= 0.6) |
|---|---|---|
| `wit` | Light humor is fine but brief | Be witty, use configured humor style |
| `swagger` | — | Confident and direct |
| `warmth` | — | Genuinely warm and helpful |
| `formality` | Casual, contractions, conversational | Professional register despite personality |
| `verbosity` | Concise, short sentences | Detailed and expressive |
| `pop_culture_refs` | — | Reference culture when natural |

## Evolved Traits

Trait weights evolve over time through dream state reflection. Evolved traits are stored in cold memory's `facts` table under the `personality` category:

| Key | Value |
|---|---|
| `trait_weights` | JSON dict of current weights |
| `humor_style` | Updated humor style string |
| `learned_preferences` | JSON dict of user-learned preferences |

The seed is the floor. Evolution can adjust weights within bounds, but the admin's seed is always loaded first, then overlaid with evolved values from cold memory.

## Awareness Sense

The awareness sense handles identity and self-reflection. When a user asks "who are you?", this sense builds context and generates a response.

**Context assembly:**
- Name and voice description from personality
- Inventory of active nerves, senses, and tools from cold memory
- Total task invocations
- Known user profile (name, preferences)
- Awareness rules (permission boundaries, protected paths)

**Response generation:** Uses the brain model with either a community adapter prompt or a hardcoded fallback. The response is personality-infused but never exposes internal details like nerve counts or tool names.

**Permission checks:** The awareness sense also gates destructive operations. Before deleting files or executing dangerous commands, it checks against rules:
- `never_delete`: protected paths that can never be removed
- `never_execute`: forbidden command patterns
- `require_confirmation`: operations that need user approval

Rules are defined in `awareness_rules.json` co-located with the awareness sense.

## Communication Sense

The communication sense rewrites every outgoing response for tone and personality. It supports multiple output formats.

### Rewrite Pipeline

1. Load personality traits (seed + evolved from cold memory)
2. Build a personality instruction string from trait weights
3. Load the community communication adapter prompt (if available)
4. Detect if the message is structured data (code, JSON, tables)
5. Choose the rewrite strategy:
   - **Structured data**: Deliver data exactly as-is, at most a one-liner quip before or after
   - **With adapter**: Use community prompt with personality injection
   - **Without adapter**: Use tone template + personality instruction
6. Call the communication model for rewriting
7. Run safety filter on the output

### Output Formats

| Format | Description |
|---|---|
| `text` | Plain text with tone adjustment and optional emoji suffix |
| `card` | Markdown card with title, body, and footer |
| `emoji` | LLM-enhanced emoji insertion |
| `gif` | Text response with an accompanying GIF (Tenor API) |
| `translate` | Language translation (tone field carries target language) |
| `summarize` | Condense long text into 1–3 sentences |

### Tone Templates

Five built-in tones control the rewrite prompt:

| Tone | Style |
|---|---|
| `formal` | Proper grammar, no contractions, respectful register |
| `casual` | Conversational, contractions, relaxed |
| `enthusiastic` | Dynamic language, upbeat, positive |
| `empathetic` | Warm, supportive, acknowledges feelings |
| `neutral` | Clear, concise, straightforward |

### Structured Data Detection

The communication sense detects structured data and avoids personality-altering it:

- Code blocks (backtick-fenced or indented)
- Valid JSON objects/arrays
- Numeric results
- File listings, paths, and table rows (60%+ of lines match)

When structured data is detected, the rewrite prompt switches to a protective mode that preserves the data verbatim.

## Personality Evolution

Evolution happens exclusively during dream state to prevent mid-conversation instability.

### Signal Collection

During live conversation, interaction signals are appended to the `personality_signals` table in cold memory. Signals are append-only and never read during live interaction. They include context like user tone shifts, topic domains, and explicit feedback.

### Observation Phase

Requires at least 10 accumulated signals. The LLM analyzes the signals and produces:

- `trait_scores`: effectiveness score (0.0–1.0) per trait
- `insights`: 2–4 observations about what's working
- `explicit_feedback_summary`: summary of direct user feedback
- `recommendation`: suggested direction

Explicit user feedback is weighted higher than implicit signals.

### Evolution Phase

The LLM proposes trait weight adjustments based on the observation report. Constraints:

| Constraint | Value |
|---|---|
| Max drift per cycle | ±0.1 |
| Trait minimum | 0.1 |
| Trait maximum | 0.9 |
| Min confidence to apply | 0.6 |

Admin-defined anchors are enforced. The `never` anchor list prevents certain trait values. The `bounds` anchor sets per-trait min/max ranges.

**Anti-oscillation:** Recent evolution history is included in the prompt so the LLM avoids flip-flopping (e.g., raising warmth one cycle and lowering it the next).

### History

Every evolution event is recorded in the `personality_history` table with:
- Old and new trait weights
- Individual changes with reasons
- Observation summary
- LLM confidence score

This provides a full audit trail with rollback capability.

### Storage

Evolved traits are persisted as cold memory facts:

```python
cold.set_fact("personality", "trait_weights", json.dumps(new_weights))
cold.set_fact("personality", "humor_style", "dry wit with occasional tech puns")
```

On next load, both the awareness and communication senses read the evolved traits from cold memory, overlaying them on top of the seed.

::: tip Related
See [Dream State](/guide/dream-state) for when personality evolution runs, and [Community](/guide/community) for how adapters shape the communication model's behavior.
:::
