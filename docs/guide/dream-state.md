# Dream State

When the brain is idle for 120 seconds, it enters dream state. Eight autonomous maintenance processes run, each interruptible. The moment a task arrives, everything stops.

## The Idle Threshold

The `Dreamstate` class tracks `_last_activity`. A `threading.Timer` fires after `IDLE_THRESHOLD` (120 seconds). If the brain has truly been silent for that long, a daemon thread starts the dream cycle.

When `wake()` is called (task arrives), the `_interrupted` event is set. The dream worker checks this event between every unit of work, saves progress, and yields. The brain blocks until the worker has fully stopped and released all LLM locks.

```
idle 120s → _enter_dreamstate() → _dream() → [phases 1-8]
                                                ↑
task arrives → wake() → _interrupted.set() → join(timeout=60)
```

## The Eight Processes

### 1. Consolidation

Merges duplicate nerves to reduce clutter.

**How duplicates are detected:**

1. Build a catalog of all nerves from cold memory
2. Compare every pair using `match_score()` (keyword similarity)
3. Pairs scoring above the merge threshold form connected components via BFS
4. Each cluster is a group of nerves doing the same thing

**Winner selection** (in priority order):
- Community nerves always win over fabricated ones
- Highest qualification score
- Most tools learned

**Merge process:**
- Tools from the loser are migrated to the winner
- Cached context/meta files and LoRA adapters are copied
- Winner description is enriched with unique keywords from the loser
- Loser is deleted from disk, cold memory, and Redis

**Safety:** An LLM-as-judge confirms each merge. If the LLM says two nerves serve different purposes, the merge is skipped. Qualified losers are never merged into unqualified winners. Requires a medium or large brain model — small models lack the judgment.

### 2. MCP Fanout

Discovers new tools and wires them into nerves. Runs before reconciliation so nerves have the right tools before testing.

Each nerve starts as a narrow task seed. Fanout grows it across up to 7 iterations per dream cycle:

| Iterations | Phase | What happens |
|---|---|---|
| 1 | Generalize | Expand task-specific seed into a broad domain description |
| 2–3 | Equip | Wire existing MCP tools, fabricate missing ones |
| 4–5 | Deepen | Refine system prompt with tool-aware expertise |
| 6–7 | Sharpen | Add edge cases, examples, fallback strategies |

Progress is tracked per-nerve in cold memory (`fanout` category). Interrupted cycles resume where they left off.

**Sibling discovery:** For each tool already wired to a nerve, fanout inspects the tool's source code to find related tools:
- **API-based tools**: Fetches OpenAPI specs or doc pages to discover other endpoints
- **Library-based tools**: Inspects the library at runtime (`dir()`) to find useful functions
- **External MCP tools**: Finds other tools from the same server

Discovered siblings are fabricated and wired to the nerve automatically.

### 3. Reconciliation

Pushes weak nerves toward 95%+ quality through prompt tuning.

**Work queue:** Nerves are prioritized by recency of use (recently invoked first), then by weakest score. Only nerves that have actually been invoked are reconciled — improving never-used nerves wastes LLM calls.

**Per-nerve improvement loop:**

1. Ensure the nerve has a test bank (generate if missing)
2. Run test cases against the nerve
3. Evaluate results — score and pass/fail each test
4. If score meets the threshold with sufficient coverage (80%+ of tests), record qualification
5. If not, ask the LLM to suggest improvements (prompt changes, tool fixes)
6. Apply improvements, re-test, rollback if the score dropped

**Plateau detection:** If the score hasn't improved by more than 5% over the last 2 iterations, reconciliation stops for that nerve. This prevents infinite loops on nerves that have hit their ceiling with prompt tuning alone.

**Tool healing:** During the brain upgrade phase, tools with failure rates above 30% are automatically fixed. The coder LLM reads the tool source, identifies common failure modes (API changes, missing error handling, JSON parsing), and generates a fixed version. The old version is backed up. If the fix makes things worse, it's rolled back.

### 4. Brain Upgrade

Self-provisioning for the brain and adapter system.

**Adapter tuning:** For each adapter role (brain, nerve, creative, code, awareness, communication, vision):

1. Check if `qualification_score < 0.95`
2. Pull recent episodes relevant to this role
3. Identify failure patterns
4. LLM generates improved `system_prompt` and `few_shot_examples`
5. Save updated adapter to the model-specific variant directory
6. When score reaches 0.95, contribute the adapter back to the community

**MCP tool healing:** Tools with high failure rates (>30% over 3+ calls) are read, fixed by the coder LLM, syntax-validated, and saved with a backup.

### 5. Fine-tuning

Trains LoRA adapters for nerves with enough training data. This is the "deep loop" that breaks through the prompt-tuning ceiling.

**When it triggers:** A nerve needs:
- At least `min_training_examples` data points (configurable per role via community adapters)
- To be qualified (prompt tuning done first)
- Training data from test banks and successful episodes

**Test bank expansion:** Before training, the system generates additional test cases for qualified nerves that don't have enough data. Only medium/large brains can do this — small models produce garbage.

**Per-size-class data limits:**

| Size Class | Max Training Examples |
|---|---|
| tinylm | 50 |
| small | 100 |
| medium | 200 |
| large | 500 |

**Training process:**
1. Load the GGUF model into PyTorch via `transformers` (GGUF direct loading)
2. Wrap with PEFT LoRA
3. Train on input/output pairs formatted as instruction-response
4. Save adapter as safetensors
5. Convert to GGUF format for llama-cpp-python
6. Unload PyTorch model to free memory

**Tuning configs per role:**

| Role | Target Modules | Rank | Notes |
|---|---|---|---|
| tool/nerve | `q_proj`, `v_proj` | default | Precise structured output |
| code | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj` | 16 | Syntax accuracy + structure |
| creative | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` | 16 | Diverse generation |
| vision | `qkv_proj`, `out_proj` | 8 | Moondream2 fused projections |
| communication | `q_proj`, `v_proj` | 4 | Tone/style, conservative |
| brain | `q_proj`, `v_proj` | 4 | Routing decisions, low rank |
| awareness | `q_proj`, `v_proj` | 4 | Self-reflection |

These are architecture defaults. Community `meta.json` overrides everything.

**After training:** The nerve is re-qualified to measure improvement. Adapters are saved to `nerves/{name}/adapter/adapter.gguf` and auto-loaded at inference time.

### 6. Contribution

Pushes improved nerves and adapters back to the community repo.

**What gets contributed:**

| Condition | Action |
|---|---|
| New nerve (not in community) | Push full bundle (bundle.json, test_cases.json, tools, context.json, meta.json, adapter.gguf) |
| Existing nerve, new model size class | Push model-specific adapter (context.json, meta.json) |
| Existing nerve, new language stack | Push tool implementations in the new language |
| Adapter score > community score | Push updated adapter via PR |

**Gate:** Nerves are not contributed until they have a trained LoRA adapter (`adapter.gguf` must exist).

**PR deduplication:** Before creating a new PR, the system searches for existing open PRs matching the nerve name. If found, the existing branch is updated in-place instead of creating a duplicate.

**Unified PRs:** Each nerve produces one PR containing the full bundle, adapter, and tool stack. No more scattered PRs per artifact type.

**Authentication:** Contributions use the server's [GitHub App](/guide/getting-started#step-11-github-integration) identity. Each server mints short-lived installation tokens — no personal access tokens needed.

### 7. PR Review

Maintains open PRs and cleans up stale branches.

**Fix review feedback:** PRs with `CHANGES_REQUESTED` status are automatically addressed:

1. Read all review comments on the PR
2. Checkout the PR branch in a temporary worktree
3. Use the LLM to generate fixes based on the feedback
4. Commit and push the changes

**Cleanup stale PRs:** PRs that have been open and inactive for 30+ days are closed with a comment explaining the timeout.

**Delete merged branches:** Local and remote branches for merged or closed PRs are cleaned up to prevent accumulation.

### 8. Personality Reflection

Evolves the communication voice based on interaction patterns.

**Scope:** Only affects the final text the user sees (communication sense) and self-reflection (awareness sense). Never touches routing, nerve selection, or work quality.

**Two-phase process:**

1. **Observation:** Reads accumulated interaction signals from cold memory. An LLM analyzes them and produces trait effectiveness scores (0.0–1.0) plus insights. Requires at least 10 signals.

2. **Evolution:** The LLM proposes trait weight adjustments within admin-defined anchor bounds. Changes are capped at ±0.1 per dream cycle. All weights stay between 0.1 and 0.9. The LLM must be at least 60% confident. Full audit trail is recorded in `personality_history`.

## Interruptibility Pattern

Inspired by React Fiber's reconciliation model:

- Work is broken into small units (one merge, one test case, one training step)
- Between each unit, check `interrupted.is_set()`
- If interrupted: save progress, yield immediately, release LLM locks
- When idle again: rebuild work queue from current state (not from saved progress)

This means no long-running operation ever blocks an incoming task. The maximum latency between `wake()` and the brain being available is the time to finish one atomic unit of work — typically a single LLM call.

**Progress survival:** Most progress is persisted to cold memory between units:
- Consolidation: merged nerves stay merged
- Fanout: iteration count is saved per-nerve in the `fanout` facts category
- Reconciliation: qualification scores are recorded after each improvement
- Fine-tuning: complete adapters are saved; interrupted training is discarded (never saves partial weights)
- Personality: signals are only flushed after full processing

**Work queue rebuild:** When dream state resumes, it does not replay from a checkpoint. It rebuilds the entire work queue from the current state of cold memory. This means:
- A nerve deleted by consolidation won't appear in the reconciliation queue
- A nerve that improved past the threshold won't be reconciled again
- A nerve that gained tools from fanout will be tested with those tools

::: tip Related
See [Architecture Overview](/architecture/overview) for how dream state fits into the nervous system, and [Memory](/guide/memory) for the data tiers that dream state operates on.
:::
