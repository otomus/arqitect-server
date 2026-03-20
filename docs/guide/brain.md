# Brain

The brain is the orchestrator. It receives a task, classifies intent, asks an LLM for a routing decision, normalizes the output, and dispatches to the correct handler. It never implements — it routes.

## The think() Loop

Every task enters through `think(task, history, depth)`. The flow:

1. **Safety check** — input is filtered for harmful content (depth 0 only).
2. **Recalibration detection** — messages like `recalibrate senses` bypass the LLM entirely.
3. **Conversation buffer** — the task is added to hot memory.
4. **Memory context** — episodes, facts, and user profile are loaded for this task.
5. **Nerve catalog** — all qualified nerves are discovered. Circuit-broken nerves are filtered out (core senses are exempt).
6. **Intent classification** — the task is classified as `workflow` or `direct`.
7. **Workflow shortcut** — workflow intents go through the planner or TDD chain builder.
8. **LLM routing** — the brain model receives the full context (nerve catalog, episodes, conversation history, user facts) and the system prompt, and returns a JSON decision.
9. **Dispatch** — the JSON is parsed into a typed `Decision`, normalized, and dispatched.

If the LLM fails to produce valid JSON, the raw response is published directly.

### Re-thinking

When a handler determines the decision was wrong — nerve doesn't exist, nerve rejected the task, data is missing — it calls `think()` again with accumulated context in `history`. Each re-think increments `depth`.

**Depth limit: 5.** Beyond that, the brain returns a fallback message asking the user to rephrase. This is the circuit breaker for infinite re-think loops.

## Intent Classification

Intent classification runs in a separate LLM call with its own system prompt, isolated from the brain's routing prompt to avoid bias.

Two intent types:

| Type | Description | Examples |
|---|---|---|
| `workflow` | Multi-step structured work | Development, debugging, setup, planning, migration, deployment, refactoring |
| `direct` | Everything else | Greetings, questions, single requests, creative tasks, lookups |

Workflow intents include a `category` field (e.g. `"development"`, `"debugging"`) and are routed to the planner. Direct intents go through standard LLM routing.

If the classifier fails to produce valid JSON, it defaults to `direct` (conservative — don't hijack normal routing).

## Typed Decisions

The LLM returns a JSON object with an `action` field. This is parsed into one of 8 typed dataclasses:

| Decision | Action | Fields | When Used |
|---|---|---|---|
| `InvokeDecision` | `invoke_nerve` | `name`, `args` | Route to an existing nerve |
| `SynthesizeDecision` | `synthesize_nerve` | `name`, `description`, `mcp_tools` | Create a new nerve |
| `ChainDecision` | `chain_nerves` | `steps[]` (nerve + args), `goal` | Multi-nerve pipeline |
| `ClarifyDecision` | `clarify` | `message`, `suggestions[]` | Ask user for more info |
| `FeedbackDecision` | `feedback` | `sentiment`, `message` | User gave positive/negative feedback |
| `UpdateContextDecision` | `update_context` | `context{}`, `message` | User shared personal facts |
| `RespondDecision` | `respond` | `message` | Brain tried to respond directly (always redirected) |
| `UseSenseDecision` | `use_sense` | `sense`, `name`, `args` | Non-standard action (normalized to invoke) |

The `parse_decision()` function handles malformed LLM output gracefully — unknown keys are ignored, missing fields get defaults. Unknown actions fall back to `InvokeDecision`.

## The Dispatch Pipeline

Every decision passes through three stages before reaching a handler.

### 1. Normalize

`normalize_action()` cleans up the LLM's output:

- **use_sense to invoke_nerve** — `{"action": "use_sense", "sense": "sight"}` becomes `{"action": "invoke_nerve", "name": "sight"}`.
- **Nerve name as action** — if the action field contains a nerve or sense name (e.g. `"sight"` instead of `"invoke_nerve"`), it's rewritten to `invoke_nerve` with that name.
- **Typo correction** — fuzzy-matches action strings to valid actions using `difflib.get_close_matches` with a 0.6 cutoff.
- **Dict args** — dict args are coerced to JSON strings.

### 2. Validate (Synthesize Redirect)

`resolve_synthesize_redirect()` prevents re-synthesis of existing nerves:

- **Exact match** — if a nerve with the requested name already exists, redirect to `invoke_nerve`.
- **Fuzzy match** — if the description matches an existing nerve's description (score >= 2.5 with margin >= 0.5), redirect to `invoke_nerve`.
- **No match** — allow synthesis to proceed.

### 3. Dispatch

`dispatch_action()` routes to the correct handler:

| Action | Handler | Behavior |
|---|---|---|
| `invoke_nerve` | `_handle_invoke` | Permission check, calibration guard, invoke subprocess, parse output, record episode |
| `synthesize_nerve` | `_handle_synthesize` | Model capability gate, user permission gate, synthesize, re-think to invoke |
| `chain_nerves` | `_handle_chain` | Execute steps sequentially with context accumulation; single-step chains become invoke |
| `clarify` | `_handle_clarify` | Return clarification message with optional suggestions |
| `feedback` | `_handle_feedback` | Record sentiment on most recent episode, update circuit breaker |
| `update_context` | `_handle_update_context` | Store user facts in session and cold memory |
| `respond` | `_handle_respond` | Always re-thinks — brain must never respond directly, routes to awareness sense |
| `fabricate_tool` | re-think | Tool fabrication is handled by nerves, not the brain |
| unknown | re-think | Re-thinks with hint listing valid actions |

### Invoke Details

The invoke handler does more than just call a nerve:

1. **Args enrichment** — if the LLM's args are shorter than the original task, the task is appended as context.
2. **Sense arg translation** — core senses get specialized arg formatting.
3. **Existence check** — if the nerve doesn't exist and hasn't been synthesized in this re-think chain, re-think with "synthesize it first".
4. **Permission check** — user role is validated against the nerve's required role.
5. **Calibration guard** — for core senses, checks Redis calibration status. Unavailable senses return an error with missing dependency info.
6. **Invocation** — runs the nerve as a subprocess.
7. **Output parsing** — handles JSON output, `needs_data` status (re-think to resolve), `wrong_nerve` status (re-think to find another).
8. **Communication rewrite** — non-error text responses are rewritten through the communication model for personality and tone.
9. **Episode recording** — success/failure is recorded in warm memory and the circuit breaker is updated.

### Chain Details

Chains execute steps sequentially. Each step's output is collected and passed as context to subsequent steps:

- Missing nerves are synthesized on the fly (subject to model capability and user permission gates).
- Multi-step results are summarized by the communication model.
- Single-step chains are optimized to a direct invoke.

## System Prompt Construction

The brain's system prompt is assembled from four sources:

1. **Community adapter** — the base prompt loaded from `.community/cache/adapters/brain/core/context.json`. This is the routing instruction set.
2. **Calibration status** — live status of each core sense (operational/degraded/unavailable with capability details).
3. **Few-shot examples** — from the community adapter, formatted as `User: "input" -> output`.
4. **Session context** — known user facts (city, country, timezone) from the current session.

If the brain adapter is missing, startup fails with a clear error.

## Permissions

Access control operates at two independent levels.

### User Level

Role hierarchy: `anon` (0) < `user` (1) < `admin` (2) < `owner` (3).

| Permission | Required Level |
|---|---|
| Invoke most nerves | `anon` (0) |
| Invoke `touch` sense | `user` (1) |
| Invoke `code`-role nerves | `user` (1) |
| Synthesize new nerves | `user` (1) |
| All access | `admin` (2+) |

Anonymous users who try restricted operations get a message asking them to identify themselves via email.

### Model Level

Only `medium` and `large` model size classes can fabricate (synthesize) new nerves. Small models and tinylm are restricted to invoking existing nerves and using community-provided capabilities.

The size class is determined by the community adapter's model classification.

## Events

The brain publishes events to Redis pub/sub channels. The dashboard and connectors subscribe to these.

| Channel | Data | When |
|---|---|---|
| `brain:thought` | `{stage, task, catalog}` | Thinking starts, recalibration, depth limit, chain start |
| `brain:action` | `{nerve, args}` | About to invoke a nerve |
| `brain:response` | Response envelope (message, tone, media, source, chat_id) | Final response ready |
| `brain:audio` | `{audio_b64, audio_mime}` | TTS audio generated (background thread) |
| `brain:task` | Task payload | Incoming task from dashboard/connector |
| `brain:checklist` | Checklist state | TDD workflow progress |
| `nerve:result` | `{nerve, output}` | Nerve subprocess completed |
| `nerve:qualification` | `{nerves[]}` with scores | Nerve status sync |
| `system:status` | `{state}` | Brain online/killed |
| `system:kill` | — | Shutdown signal |
| `memory:update` | `{episodes[], facts{}}` | Memory state changed |
| `memory:episode` | Episode data | New episode recorded |
| `memory:tool_learned` | Tool data | New tool discovered |
| `sense:calibration` | Calibration data | Sense calibrated |
| `sense:config` | Config change | Sense config updated |
| `sense:peek` | Frame data | Sight sense peek |
| `sense:voice` | Voice data | Voice input |
| `sense:image` | Image data | Image input |
| `sense:sight:frame` | Frame data | Sight sense continuous frame |
| `sense:stt:result` | STT result | Speech-to-text completed |
| `tool:lifecycle` | Tool event | Tool discovered/removed |

## Response Validation

Before any response reaches the user, `_validate_response()` checks for four issues:

| Check | Trigger | Result |
|---|---|---|
| **Empty** | Response is under 2 characters | Blocked — regenerated with personality |
| **Leaked JSON** | Response starts with `{"action"` or contains action JSON in a code fence | Blocked — the LLM leaked its routing decision |
| **Leaked tool call** | Response starts with `call:` pattern with `args` | Blocked |
| **Echo** | Response is > 85% similar to the user's input (character-level `SequenceMatcher`) | Blocked — the LLM parroted the question back |

Blocked responses are replaced by a personality-flavored message asking the user to rephrase. A separate safety filter catches sensitive data leakage and inappropriate content.

::: tip Related
See [Architecture Overview](/architecture/overview) for how the brain fits into the full system, and [Data Flow](/architecture/data-flow) for the complete task lifecycle.
:::
