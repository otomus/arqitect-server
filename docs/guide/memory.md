# Memory

Arqitect uses a three-tier memory architecture. Each tier has a distinct purpose, lifetime, and storage engine. Data flows downward: hot → warm → cold. Never backward.

## Hot Memory (Redis)

Session-scoped, ephemeral. Wiped on restart.

Hot memory stores real-time conversation state and session context. It is the only tier that supports pub/sub for live event streaming.

### Keys

| Key Pattern | Type | Purpose |
|---|---|---|
| `synapse:session` | Hash | Global session context (location, timezone, country) |
| `synapse:session:{user_id}` | Hash | Per-user session context |
| `synapse:conversation` | List | Global conversation buffer (last 20 messages) |
| `synapse:conversation:{user_id}` | List | Per-user conversation buffer |
| `synapse:sense_calibration` | String | Cached sense calibration results |
| `synapse:nerve_status` | String | Current nerve registry snapshot for the dashboard |

### Conversation Buffer

Messages are stored as JSON objects with `role` and `content` fields. The buffer is capped at 20 messages via `LTRIM` — oldest messages are evicted automatically.

The sliding window size is configurable per role through community adapters. Smaller models get fewer messages to conserve context budget.

```python
# Add a message
mem.hot.add_message("user", "what's the weather?", user_id="abc")

# Retrieve last N
messages = mem.hot.get_conversation(limit=10, user_id="abc")
```

### Session Data

Session context is bootstrapped at startup from cold memory or a geolocation API. It includes location, timezone, and country — used for contextual awareness.

## Warm Memory (SQLite Episodes)

Medium-term episodic memory. Stores task execution history for recall and pattern matching.

### Storage

SQLite database at `{memory_dir}/episodes.db`. Single table:

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `timestamp` | REAL | Unix epoch of the episode |
| `task` | TEXT | The user's original request |
| `nerve` | TEXT | Which nerve handled it |
| `tool` | TEXT | Which tool was invoked |
| `args` | TEXT | JSON-serialized arguments |
| `result_summary` | TEXT | Summary of the result |
| `success` | INTEGER | 1 = success, 0 = failure |
| `tokens` | INTEGER | Token count |
| `user_id` | TEXT | Scoped to specific user |

### Retention

Capped at 500 episodes. When the limit is exceeded, the oldest episodes are pruned. No TTL — pruning is count-based.

### Recall

Episodes are recalled by keyword similarity scoring with a recency boost. Episodes less than 1 hour old get +2 score, less than 24 hours get +1. No vector DB required — uses `matching.py` for keyword scoring.

```python
# Find relevant past episodes
episodes = mem.warm.recall("weather forecast", limit=5, user_id="abc")
```

## Cold Memory (SQLite Knowledge Graph)

Permanent storage. Survives full restarts. This is the system's long-term knowledge.

SQLite database at `{memory_dir}/knowledge.db`.

### Tables

#### `facts`

General-purpose key-value store with categories.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER | Primary key |
| `category` | TEXT | Namespace (e.g., `user`, `personality`, `fanout`) |
| `key` | TEXT | Fact identifier |
| `value` | TEXT | Fact value |
| `confidence` | REAL | 0.0–1.0, defaults to 1.0 |

Unique constraint on `(category, key)` — upserts on conflict.

User-scoped facts use the category `user:{user_id}`. Personality traits use `personality`.

#### `nerve_registry`

The definitive registry of all nerves.

| Column | Type | Purpose |
|---|---|---|
| `name` | TEXT | Primary key — nerve identifier |
| `description` | TEXT | Human-readable description |
| `total_invocations` | INTEGER | Lifetime invocation count |
| `successes` | INTEGER | Successful invocations |
| `failures` | INTEGER | Failed invocations |
| `origin` | TEXT | `local` or `community` |
| `system_prompt` | TEXT | LLM system prompt (empty for community nerves) |
| `examples` | TEXT | JSON array of few-shot examples |
| `is_sense` | INTEGER | 1 if this is a protected core sense |
| `role` | TEXT | `tool`, `creative`, or `code` |
| `embedding` | TEXT | Cached description embedding |
| `model_adapters` | TEXT | JSON dict of model-specific overrides |
| `last_invoked_at` | TEXT | ISO timestamp of last invocation |

Community nerves store `system_prompt` and `examples` in the community cache directory, not in SQLite. The registry only holds identity and role.

#### `tool_stats`

Per-tool invocation statistics.

| Column | Type | Purpose |
|---|---|---|
| `name` | TEXT | Primary key — tool identifier |
| `total_calls` | INTEGER | Lifetime call count |
| `successes` | INTEGER | Successful calls |
| `failures` | INTEGER | Failed calls |

#### `nerve_tools`

Many-to-many relationship between nerves and tools.

| Column | Type | Purpose |
|---|---|---|
| `nerve` | TEXT | Nerve name |
| `tool` | TEXT | Tool name |
| `use_count` | INTEGER | How many times this nerve used this tool |

Primary key on `(nerve, tool)`. Use count increments on each invocation.

#### `qualification_results`

Test results from the qualification critic.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER | Primary key |
| `subject_type` | TEXT | `nerve` or `tool` |
| `subject_name` | TEXT | Name of the subject |
| `qualified` | INTEGER | 1 = passed |
| `score` | REAL | 0.0–1.0 qualification score |
| `iterations` | INTEGER | Number of improvement iterations |
| `test_count` | INTEGER | Total test cases run |
| `pass_count` | INTEGER | Test cases passed |
| `details` | TEXT | JSON with per-test results |
| `timestamp` | DATETIME | When qualification ran |

Unique constraint on `(subject_type, subject_name)`.

#### `users`

Canonical user records.

| Column | Type | Purpose |
|---|---|---|
| `user_id` | TEXT | Primary key — UUID |
| `display_name` | TEXT | User's display name |
| `email` | TEXT | Verified email address |
| `role` | TEXT | `anon`, `user`, or `admin` |
| `created_at` | DATETIME | Account creation time |
| `last_seen` | DATETIME | Last interaction time |
| `secrets` | TEXT | JSON dict of per-user secrets (MCP integrations) |

#### `user_links`

Maps connector identities to canonical users.

| Column | Type | Purpose |
|---|---|---|
| `connector` | TEXT | Platform (e.g., `telegram`, `whatsapp`, `dashboard`) |
| `connector_id` | TEXT | Platform-specific user ID |
| `user_id` | TEXT | Foreign key to `users.user_id` |
| `linked_at` | DATETIME | When the link was created |

Primary key on `(connector, connector_id)`. A single user can have multiple connector identities linked via email verification.

#### `personality_signals`

Append-only buffer of interaction signals for personality evolution.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER | Primary key |
| `timestamp` | REAL | Unix epoch |
| `data` | TEXT | JSON-serialized signal |

Signals accumulate during live conversation and are flushed after dream state processes them.

#### `personality_history`

Audit trail of personality evolution events.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER | Primary key |
| `timestamp` | REAL | Unix epoch |
| `old_traits` | TEXT | JSON of trait weights before change |
| `new_traits` | TEXT | JSON of trait weights after change |
| `changes` | TEXT | JSON array of individual changes |
| `observation_summary` | TEXT | LLM's analysis summary |
| `confidence` | REAL | LLM's confidence in the changes |

## Data Flow Between Tiers

### Hot → Warm

Every completed task is recorded as an episode. The `MemoryManager.record_episode()` method writes to warm memory and simultaneously updates cold stats:

1. Episode dict is inserted into the `episodes` table (warm)
2. Nerve invocation count is incremented in `nerve_registry` (cold)
3. Tool call count is incremented in `tool_stats` (cold)
4. Nerve-tool relationship is recorded in `nerve_tools` (cold)

### Warm → Cold

Promotion happens through dream state processes:

- **Reconciliation** reads episodes to identify failure patterns and generate improvement suggestions
- **Adapter tuning** analyzes recent episodes to identify what's failing per role, then adjusts system prompts
- **Fine-tuning** uses successful episodes as training data for LoRA adapters
- **Personality observation** reads interaction signals to score trait effectiveness

### Context Assembly

When a nerve is invoked, `MemoryManager.get_env_for_nerve()` assembles context from all three tiers:

- **Hot**: current session, recent conversation messages
- **Warm**: relevant past episodes (keyword-matched with recency boost)
- **Cold**: known tools, user facts, nerve metadata (system prompt + examples), user profile

This context is serialized as environment variables and passed to the nerve subprocess.

## MemoryManager API

The `MemoryManager` class is a facade over all three tiers.

| Method | Purpose |
|---|---|
| `get_context_for_task(task, user_id)` | Build full context from all tiers |
| `record_episode(episode)` | Write to warm + update cold stats |
| `get_env_for_nerve(nerve, task, ...)` | Build env vars for nerve subprocess |

::: tip Related
See [Architecture Overview](/architecture/overview) for how memory fits into the nervous system, and [Dream State](/guide/dream-state) for how data is promoted during sleep.
:::
