# What is Arqitect?

Arqitect is a self-evolving AI agent server modeled as a nervous system. It receives tasks, routes them to specialized agents (nerves), and grows new capabilities on the fly when none exist.

It is not a framework you build on. It is a running system that builds itself.

## The Biological Analogy

Every component in arqitect maps to a part of the human nervous system. This is not decoration — the analogy drives the architecture.

| Component | Biological Analogy | What It Does |
|---|---|---|
| **Brain** | Central nervous system | Receives tasks, classifies intent, routes to nerves. Never implements. |
| **Nerves** | Peripheral nerves | Autonomous AI agents that execute domain-specific tasks in isolated subprocesses. |
| **Senses** | Sensory organs | Five immutable capabilities: sight, hearing, touch, awareness, communication. |
| **Memory** | Short-term / episodic / long-term | Three tiers: Hot (Redis), Warm (SQLite episodes), Cold (SQLite knowledge graph). |
| **Tools (MCP)** | Muscles and limbs | Structured integrations via JSON-RPC. Language-agnostic. Crash-isolated. |
| **Dream State** | Sleep and dreaming | Consolidation, tuning, contribution — runs when idle. |
| **Community** | Family | Instances share nerves, tools, and adapters through a shared repo. |

The nervous system model matters because it defines how the system grows. Nerves are synthesized, qualified, and pruned. Senses never change. Memory flows in one direction (hot to cold, never backward). The brain never implements — it only routes.

## Key Concepts

**Brain** — The orchestrator. Classifies user intent as `workflow` or `direct`, builds a context window with the nerve catalog and conversation history, and asks the LLM to produce a typed JSON decision. That decision gets normalized, validated, and dispatched. The brain re-thinks with accumulated context when things go wrong, up to a depth limit of 5.

**Nerves** — Autonomous agents that run as Python subprocesses. Each has a system prompt, few-shot examples, and access to MCP tools. Nerves come from two sources: community bundles (curated, tested, versioned) or LLM synthesis (generated when no existing nerve matches). Every nerve must pass a 95% qualification threshold before it routes traffic.

**Senses** — Five immutable core capabilities: `sight` (image analysis), `hearing` (STT/TTS), `touch` (filesystem/OS), `awareness` (identity and persona), `communication` (personality-driven response rewriting). Senses auto-calibrate on startup and report their status as operational, degraded, or unavailable.

**Memory** — Hot memory (Redis) holds conversations and session state. Warm memory (SQLite) stores task episodes for pattern matching. Cold memory (SQLite) is the permanent knowledge graph — nerves, tools, users, personality traits.

**Tools** — MCP-compliant processes that speak JSON-RPC over stdio. Each runs in its own subprocess. A crash in one tool never affects another.

**Dream State** — When idle for 120 seconds, the brain enters dream state. Seven processes run: consolidation (merge duplicate nerves), MCP fanout (discover new tools), reconciliation (tune weak nerves toward 95%+), brain upgrade, fine-tuning, community contribution, and personality reflection. All are interruptible — any incoming task stops dream work immediately.

**Community** — Arqitect instances share nerve bundles, adapters, and tools through the arqitect-community repo. Small models consume what the community provides. Large models synthesize and contribute back.

## How It Differs

Most agent frameworks give you a toolkit and leave the wiring to you. Arqitect is different in three ways.

**Autonomous evolution.** When a user asks for something no nerve can handle, the brain synthesizes a new nerve — writes the code, registers it, qualifies it, and invokes it. The next time someone asks a similar question, the nerve already exists.

**Community sharing.** Capabilities flow between instances. A nerve synthesized on one server can be qualified, bundled, and published to the community. Other instances pick it up on their next sync.

**Dream state.** The system improves while idle. Duplicate nerves get consolidated. Weak nerves get prompt-tuned. Personality traits evolve based on interaction patterns. This happens without human intervention.

## Who It's For

Arqitect is for developers who want an AI agent that runs as a persistent server — not a library call. It handles its own infrastructure: model loading, tool isolation, memory management, and capability growth.

Typical use cases:

- A personal AI assistant running on a home server or VPS
- A team-facing agent accessible via Telegram, WhatsApp, or the web dashboard
- An IoT controller that manages devices through nerves and MCP tools
- A development environment that writes, tests, and deploys code through TDD chains

::: tip Next steps
See [Getting Started](/guide/getting-started) to install and configure, or [Architecture Overview](/architecture/overview) for the full system diagram.
:::
