# Memory Tiers

arqitect doesn't dump everything into one database. Memory is split into three tiers — hot, warm, and cold — each optimized for a different access pattern and lifecycle. This is deliberate. A single store would either be too slow for real-time state or too volatile for permanent knowledge.

<MemoryTiersDiagram />

## Why Three Tiers

A conversation needs sub-millisecond access to the current session. An episode lookup needs indexed search over recent history. A nerve catalog needs permanent, structured storage that survives anything. One database can't optimize for all three. Three tiers, each with the right storage engine, can.

## Tier 1: Hot Memory (Redis)

Hot memory is the live workspace. It holds everything the brain needs right now:

- **Session state** — current user, active conversation, authentication context
- **Conversation history** — the rolling message window for the current interaction
- **Ephemeral state** — task progress, re-think counters, circuit breaker status

Hot memory lives in Redis. It's fast, it's in-memory, and it's disposable. When the process restarts, hot memory is gone. That's fine — sessions are rebuilt from the next user message, and conversation history is bounded anyway.

**What it doesn't store**: anything you'd miss if it disappeared. No learned knowledge, no nerve definitions, no user preferences.

## Tier 2: Warm Memory (SQLite Episodes)

Warm memory is the brain's short-term recall. Every task execution is recorded as an **episode** — a structured record of what was asked, what was tried, and what happened.

Before making a decision, the brain searches warm memory: *"have I handled something like this before?"* If it finds a matching episode, it can skip the expensive reasoning step and reuse the previous approach.

Episodes contain:

- The original user request
- Which nerve was selected and why
- The parameters passed
- The result (success or failure)
- Timing and cost metadata

Warm memory uses SQLite. It survives restarts. It's indexed for fast similarity search. But it's not permanent in the way cold memory is — episodes can be pruned, rotated, or consolidated over time.

## Tier 3: Cold Memory (SQLite Knowledge)

Cold memory is the knowledge graph. It's the largest and most permanent tier, storing everything arqitect has learned:

- **Nerves** — every nerve definition, system prompt, tool binding, and calibration score
- **Tools** — MCP tool registrations and capability metadata
- **Users** — identity, preferences, permission grants
- **Personality** — the system's configured voice and behavior rules
- **Community catalog** — published nerves from the community registry
- **Permissions** — which users can invoke which nerves

Cold memory is the source of truth. It survives restarts, upgrades, and redeployments. When the brain boots, cold memory is the first thing it reads. Everything else is derived from it.

## Data Flow Between Tiers

Data flows in one direction: hot to warm to cold.

1. A task arrives and enters **hot memory** as part of the active session
2. The brain checks **warm memory** for similar past episodes
3. The task executes — the result is written back to **warm memory** as a new episode
4. Over time, knowledge is extracted from episodes and written to **cold memory** — new nerve calibration scores, updated user preferences, refined tool metadata

Data never flows backward. Cold memory doesn't write to warm. Warm memory doesn't write to hot. Each tier feeds the next, building progressively more permanent knowledge.

## What Survives a Restart

| Tier | Storage | Survives Restart |
|---|---|---|
| Hot | Redis | No — rebuilt from next interaction |
| Warm | SQLite | Yes — episodes persist |
| Cold | SQLite | Yes — permanent knowledge |

Hot memory loss is invisible to users. The brain picks up where it left off using warm and cold memory. The only thing lost is the in-flight conversation context, which the user naturally re-establishes by sending their next message.

---

For implementation details — episode schemas, search algorithms, knowledge extraction pipelines — see the [Memory deep dive](/guide/memory).
