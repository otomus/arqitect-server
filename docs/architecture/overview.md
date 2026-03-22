# Architecture Overview

Arqitect is a nervous system. Not a framework, not a pipeline — a living system that senses, thinks, acts, and evolves.

## The Nervous System

Every part of arqitect maps to a biological analogy, and that analogy is intentional. It shapes how the system grows.

| Component | Role | Biological Analogy |
|---|---|---|
| **Brain** | Orchestrates. Routes tasks, never implements. | Central nervous system |
| **Nerves** | Autonomous workers. Execute domain-specific tasks. | Peripheral nerves |
| **Senses** | Immutable platform capabilities. See, hear, touch, know, speak. | Sensory organs |
| **Memory** | Three tiers. Hot, warm, cold. | Short-term, episodic, long-term memory |
| **Tools (MCP)** | Structured integrations. Language-agnostic. | Muscles and limbs |
| **Dream State** | Reflects, tunes, consolidates, contributes. | Sleep and dreaming |

## Brain

The brain is the orchestrator. It receives a task and makes one decision: what to do with it. It never implements — it routes.

Every task flows through a typed pipeline: intent classification, action decision, normalization, dispatch. The brain speaks in typed decisions (`InvokeDecision`, `SynthesizeDecision`, `ChainDecision`, etc.) — not raw strings.

When things go wrong, the brain re-thinks with new context. Circuit breakers prevent infinite loops.

::: tip Deep dive
See [Brain](/guide/brain) for the full decision pipeline, and [Data Flow](/architecture/data-flow) for the complete task lifecycle.
:::

## Nerves

Nerves are autonomous AI agents that run as isolated Python subprocesses. Each has a system prompt, few-shot examples, pre-seeded tools, and access to all 5 senses.

Nerves come from two places: **community bundles** (curated, tested, versioned) or **LLM synthesis** (generated on the fly when no bundle exists). Every nerve gets qualified by a closed-loop critic — threshold is 95%.

::: tip Deep dive
See [Nerves](/guide/nerves) for synthesis, qualification, and name guards.
:::

## Senses

Five core senses. Immutable. Can never be deleted.

- **Sight** — vision via moondream. Images, screenshots, video streams.
- **Hearing** — speech-to-text, text-to-speech, recording, playback.
- **Touch** — file system and OS. Sandboxed. Calls awareness before destructive ops.
- **Awareness** — identity and boundaries. Permission checks, self-reflection.
- **Communication** — the voice. Rewrites every response for tone and personality.

Each sense auto-calibrates on startup, reporting its status as operational, degraded, or unavailable.

::: tip Deep dive
See [Senses](/guide/senses) for calibration, modes, and integration.
:::

## Memory

Three tiers, each with a distinct purpose and lifetime.

| Tier | Storage | Lifetime | Purpose |
|---|---|---|---|
| **Hot** | Redis | Session | Conversations, state, real-time data |
| **Warm** | SQLite | Days/weeks | Task episodes, searchable history |
| **Cold** | SQLite | Permanent | Knowledge graph — nerves, tools, users, personality |

Data flows downward: hot → warm → cold. Never backward. Hot memory is wiped on restart. Cold survives forever.

::: tip Deep dive
See [Memory Tiers](/architecture/memory-tiers) for the flow diagram, and [Memory](/guide/memory) for implementation.
:::

## Tools (MCP)

Language-agnostic tool processes speaking JSON-RPC over stdio. Each tool runs in an isolated subprocess — a crash in one tool never takes down another.

The MCP server discovers tools on startup, routes calls, and tracks usage. The brain can also fabricate new tools via LLM when needed.

::: tip Deep dive
See [Tool Isolation](/architecture/tool-isolation) for the isolation model, and [Tools](/guide/tools) for building tools.
:::

## Dream State

When the brain is idle for 120 seconds, it enters dream state. Seven interruptible processes run:

1. **Consolidation** — merge duplicate nerves
2. **MCP Fanout** — discover new tools, wire into nerves
3. **Reconciliation** — prompt-tune weak nerves toward 95%+
4. **Brain Upgrade** — self-provision capabilities
5. **Fine-tuning** — train LoRA adapters
6. **Contribution** — push improvements to the community
7. **Personality Reflection** — evolve the voice

All processes are interruptible — a fiber-inspired pattern checks for incoming tasks between each unit of work. No long-running operation ever blocks an incoming task.

::: tip Deep dive
See [Dream State](/guide/dream-state) for the full consolidation loop.
:::

## Community

Arqitect instances are a family. They share nerve bundles, adapters, and tools through the arqitect-community repo.

Not all instances are the same — some run large models, some small. The permission system gates capabilities by model size. Small models use what the community provides. Large models synthesize and contribute back.

::: tip Deep dive
See [Community](/guide/community) for sync, contribution, and family dynamics.
:::

## Embeddings

Arqitect uses ONNX-based embeddings (all-MiniLM-L6-v2, 384 dimensions) for semantic matching throughout the system. Rather than relying on keyword matching alone, the brain uses a hybrid scoring system to find the right nerve for a task.

### Hybrid Matching

When a task arrives, the nerve catalog is pre-filtered before the LLM sees it:

1. **Embed the task** — compute a 384-dim vector via the ONNX model
2. **Score each nerve** — blend keyword matching (40%) with embedding cosine similarity (60%)
3. **Filter to top 20** — only the most relevant nerves enter the LLM routing context
4. **Boost core senses** — senses always make the cut regardless of score

This keeps the LLM's context window focused and reduces hallucinated nerve selections.

### Caching

Nerve embeddings are cached at three levels:

1. **In-memory LRU** — 100 most recent, sub-millisecond lookup
2. **Cold memory (SQLite)** — persists across restarts in the `nerve_registry` table
3. **Compute fresh** — via ONNX embedder on cache miss, then persisted back

If ONNX is unavailable, the system falls back to the LLM engine's embed function. If both are unavailable, keyword-only matching is used.

## Permissions

Access control at two levels:

- **User level** — anonymous users are limited to safe nerves. Identified users can access the filesystem, code nerves, and synthesize new ones.
- **Model level** — small models invoke and use. Medium/large models synthesize and fabricate.
