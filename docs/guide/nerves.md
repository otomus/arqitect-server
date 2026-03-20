# Nerves

Nerves are autonomous AI agents. Each nerve is a Python subprocess with its own system prompt, few-shot examples, pre-seeded tools, and access to all 5 senses. The brain never implements — it routes tasks to nerves.

## Two Origins

Nerves come from two sources:

| Source | Description | When used |
|---|---|---|
| **Community bundles** | Curated, tested, versioned. Pulled from the arqitect-community repo. | Checked first. If a bundle exists for the nerve name, it's used as-is. |
| **LLM synthesis** | Generated on the fly by the brain's LLM. | Fallback when no community bundle matches. |

Community bundles carry their own description, system prompt, examples, role, and declared tool list. LLM synthesis guesses all of these via inference.

## Synthesis Pipeline

When the brain decides a new nerve is needed and no community bundle exists, the synthesis pipeline runs:

### 1. Name Guards

Before anything else, the proposed nerve name is validated. Three guards fire in sequence:

- **Catch-all rejection** — names like `general_knowledge_nerve`, `utility_nerve`, `misc_nerve` are rejected. These become black holes during consolidation, absorbing every other nerve.
- **Sense collision** — names matching a core sense (`sight`, `hearing`, `touch`, `awareness`, `communication`) or their `_nerve` variants are rejected.
- **Tool-as-nerve** — if the name matches an existing MCP tool, the tool is pre-seeded instead of creating a duplicate nerve.

When a name is rejected, a new name is derived from the first meaningful word in the description.

### 2. Description Generalization

The brain's LLM often passes the raw user query as the nerve description (e.g., "Calculate 2+2" instead of "Handles arithmetic and mathematical calculations"). A narrow description causes the nerve's self-awareness gate to reject valid tasks later.

The synthesizer detects task-specific descriptions by checking for domain markers (`handles`, `provides`, `manages`, etc.). If none are found, it calls the LLM to generalize:

```
"Calculate 2+2"
  -> "Solves arithmetic and algebraic expressions, returning numeric results
     with step-by-step breakdowns"
```

Generalized descriptions must be between 10 and 200 characters and must not look like JSON.

### 3. Role Classification

Every nerve is classified into one of three roles:

| Role | Use case |
|---|---|
| `tool` | Structured I/O, precise outputs, API calls, data processing |
| `creative` | Generative content, writing, brainstorming, reflection |
| `code` | Programming, syntax generation, code review |

The brain LLM picks the role. Invalid or unexpected roles are clamped to `tool`. Each role maps to a different tuning config — model selection, temperature, token limits, and qualification thresholds.

### 4. Rich Metadata Generation

The LLM generates two things:

- **System prompt** — 3-4 sentences of behavioral instructions specific to the nerve's domain. Must include concrete output format requirements and at least one boundary (what the nerve does NOT do). Generic prompts are detected and regenerated with a stricter prompt.
- **Examples** — 2-3 input/output pairs showing correct behavior.

Generic prompt detection triggers when 2+ phrases from a blocklist appear (`"provide helpful"`, `"assist the user"`, `"help the user"`, etc.).

### 5. File Creation

A `nerve.py` is written from the nerve template into `nerves/{name}/nerve.py`, along with a `meta.json` for tuning configuration.

## The Nerve Template

Every synthesized nerve uses the same template. The template provides:

- **Runtime imports** — `think`, `think_for_role`, `mcp_call`, `respond`, `get_args`, and all sense wrappers (`see`, `hear`, `touch`, `check_awareness`, `express`).
- **Tool management** — `get_tool_list()`, `get_tool_names()`, `mcp_tool_exists()`, `acquire_tool()`.
- **Planner loop** — the nerve calls `think_for_role()` with its system prompt and context (live system data, conversation history, episode hints, available tools). The LLM returns a JSON plan with one of these actions:

| Action | Behavior |
|---|---|
| `answer` | Direct response from LLM knowledge |
| `call` | Invoke an MCP tool with arguments |
| `use_sense` | Invoke a core sense (see, hear, touch, etc.) |
| `acquire` | Search for and install a missing tool |
| `fabricate` | Generate a new tool via LLM |
| `needs` | Signal that data is missing |

The template also handles small-model quirks: placeholder detection, action normalization (tool name in `action` field instead of `call`), argument remapping, and fuzzy tool name matching.

## Execution Model

Nerves run as **isolated Python subprocesses**:

- Invoked with `subprocess.run([python, nerve.py, args])`
- Working directory is always the sandbox
- 90-second timeout (`NERVE_EXECUTION_TIMEOUT_SECONDS`)
- Memory context passed via environment variables (`SYNAPSE_*`)
- `SYNAPSE_LAZY_LOAD=1` ensures only the nerve's own model loads, not all roles
- `PYTHONPATH` is set so the arqitect package is importable
- Null bytes are stripped from args to prevent subprocess crashes

### Environment Variables

The nerve receives its full context through env vars set by `mem.get_env_for_nerve()`:

- Session context (location, timezone)
- User facts and profile
- Conversation history
- Episode hints (recent task outcomes)
- Known tools list
- Nerve metadata (system prompt, examples, role)

## Output Format

Nerves output JSON to stdout. The `respond()` helper ensures a consistent structure:

```json
{
  "nerve": "weather_nerve",
  "input": "weather in Paris",
  "response": "Paris: 18C, partly cloudy, humidity 65%.",
  "tool": "weather_tool"
}
```

Optional fields: `sense`, `sense_result`, `status`, `needs`, `error`, `image_path`, `image_mime`, `audio_path`.

Media results (images, audio, video) are returned directly without LLM interpretation.

## Qualification

Every nerve — community or synthesized — goes through a **closed-loop qualification** process in a background thread. The nerve is usable immediately; qualification runs asynchronously.

### The Loop

1. **Generate test cases** — the critic LLM produces a batch of test cases, each with an input, expected output, context, and category (`core`, `edge`, `boundary`, `negative`). Only medium/large brain models can generate tests; small models produce garbage.

2. **Run the nerve** — each test case is fed to the nerve subprocess with `SYNAPSE_NO_ACQUIRE=1` (skip expensive tool acquisition) and `SYNAPSE_SKIP_FACTS=1` (test the LLM, not fact recall).

3. **Evaluate** — layered evaluation:
   - **Deterministic** — empty output scores 0.0, error markers score 0.1, echoed input scores 0.1.
   - **LLM critic** — the brain model scores the output 0.0-1.0 against the expected response and category rules. Negative tests pass when the nerve refuses. Boundary tests pass when the nerve doesn't crash.

4. **Improve if needed** — when the score is below threshold, the critic suggests:
   - A specific behavioral rule to append to the system prompt
   - New examples
   - Tool fixes (if tool errors were detected in stderr)
   - Missing tools to acquire
   - Domain knowledge to index

   Rules are guarded against junk (vague filler, internal metric leakage, duplicates). When 5+ rules accumulate, the prompt is consolidated into a coherent rewrite.

5. **Repeat** — up to `max_qualification_iterations` (model-specific), with a total timeout (model-specific, typically 120s).

### Thresholds

The qualification threshold is model-specific, loaded from the community tuning config. The target is 95%. Nerves that score below a separate `low_quality_threshold` after qualification are pruned at startup.

### Tool Fixing

When qualification detects tool errors in the nerve's stderr, it can:

1. Read the tool's source code
2. Ask the critic to generate a fix
3. Apply the fix (with syntax validation and `run()` function check)
4. Re-test
5. Roll back if the fix didn't help

### Test Bank

Test cases are stored in cold memory and reused across qualification iterations and restarts. After qualification succeeds, the test bank is expanded to meet the minimum training example count for LoRA fine-tuning.

## Nerve Matching

When the brain needs to route a task to a nerve, it uses hybrid scoring from `arqitect/matching.py`:

### Keyword Scoring

Each query token is scored against the nerve's name, description, and parameters:

| Match type | Weight |
|---|---|
| Exact match in name | 3.0 |
| Stem match in name | 2.0 |
| Exact match in description | 1.0 |
| Stem match in description | 0.5 |
| Exact match in parameters | 0.5 |
| Stem match in parameters | 0.25 |

Stem matching requires both tokens to be at least 4 characters long and share a common prefix of at least 75% of the shorter token's length. This catches morphological variants (`calculate`/`calculating`) while preventing spurious matches (`do`/`document`).

### Embedding Similarity

When available, embedding similarity is blended with keyword scores:

```
final = 0.4 * keyword_score + 0.6 * embedding_similarity * max_keyword_score
```

### Sense Boost

Core senses receive a +2.0 scoring bonus to ensure they are preferred over regular nerves for tasks in their domain (e.g., "read file" routes to the touch sense, not a `file_reader` nerve).

## Tool Pre-seeding

When a nerve is synthesized, it's pre-seeded with MCP tools from three sources (in order):

1. **Explicit** — tools named in the synthesis request
2. **Mentioned** — tools whose names appear literally in the trigger task
3. **Matched** — tools matched by keyword similarity to the nerve description (max 3, above merge threshold)

## Nerve Pruning

Two pruning mechanisms run at startup:

- **Duplicate pruning** — cross-scores all nerve descriptions. When two nerves score above threshold, the one with fewer invocations (or lower qualification score) is deleted. Core senses are never pruned.
- **Low-quality pruning** — nerves that scored below `low_quality_threshold` after qualification are removed.

::: tip Related
- [Senses](/guide/senses) — the 5 core senses available to every nerve
- [Tools](/guide/tools) — MCP tools that nerves invoke
- [Architecture Overview](/architecture/overview) — how nerves fit into the nervous system
:::
