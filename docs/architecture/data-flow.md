# Data Flow

Every task follows the same lifecycle. User input enters, flows through a typed pipeline, and a response comes back. No magic, no hidden branches. Here's exactly what happens.

<DataFlowDiagram />

## The Task Lifecycle

A task enters `Brain.think()` and passes through a series of stages. Each stage is isolated, typed, and deterministic (aside from the LLM calls). The pipeline never shortcuts — every task walks the full path.

### 1. Intent Classification

The first decision: is this a **workflow** or a **direct** request?

- **Workflow** — multi-step tasks that require planning. "Book a flight and add it to my calendar" triggers the planner, which decomposes the task into ordered steps.
- **Direct** — single-step tasks. "What's the weather?" goes straight to action decision.

Intent classification happens before the LLM is called. It's a lightweight check that shapes the rest of the pipeline.

### 2. Embedding Pre-Filter

Before the LLM decides, the nerve catalog is narrowed down using hybrid matching — a blend of keyword scoring (40%) and ONNX embedding cosine similarity (60%). Only the top 20 most relevant nerves (plus all core senses) are included in the LLM's routing context. This keeps decisions fast and focused. See [Architecture Overview — Embeddings](/architecture/overview#embeddings) for details.

### 3. Typed Action Decisions

The brain doesn't generate free-form text. It generates a **typed decision** — a structured object that the dispatch layer can validate and route.

Seven decision types:

| Decision | What it does |
|---|---|
| `InvokeDecision` | Call an existing nerve by name |
| `SynthesizeDecision` | Create a new nerve from scratch when no match exists |
| `ChainDecision` | Execute multiple nerves in sequence, passing results forward |
| `ClarifyDecision` | Ask the user for missing information before proceeding |
| `UseSenseDecision` | Invoke a core sense directly (sight, hearing, touch, awareness) |
| `FeedbackDecision` | Record positive or negative feedback about a previous result |
| `RespondDecision` | Respond directly via the awareness sense — no nerve needed |

The LLM picks exactly one. The decision carries all the data needed for execution — nerve name, parameters, clarification questions, whatever the type demands.

### 4. Normalize, Validate, Dispatch

The typed decision passes through a three-stage pipeline before execution:

**Normalize** — fixes common LLM mistakes. Fuzzy-matches nerve names (so `calender` still hits `calendar`), redirects misclassified decisions (a sense invocation masquerading as a nerve call), and corrects typos in parameter names.

**Validate** — checks that the decision is structurally sound. Does the named nerve exist? Are required parameters present? Is the user allowed to invoke this nerve? Validation failures short-circuit to a helpful error.

**Dispatch** — routes the validated decision to the correct handler function. Each decision type has a dedicated handler. No switch statements, no type-checking at runtime — the dispatch table is built at startup.

### 5. Execution

The handler runs the decision. For `InvokeDecision`, this means:

1. Spawn the nerve as an isolated Python subprocess
2. Inject the sense runtime (all 5 senses available)
3. Attach the nerve's MCP tools
4. Pass the task and wait for a result

The nerve has full autonomy within its subprocess. It can call tools, use senses, make LLM calls — whatever its system prompt dictates. When it finishes, it returns a structured result.

### 6. Result Processing

The result flows back through the brain:

- **Episode recording** — every task execution is saved as an episode in warm memory. The brain checks these before future decisions: *"have I seen this before?"*
- **Communication rewrite** — the communication sense rewrites the raw result for tone, format, and personality. A code nerve's output gets formatted differently than a creative nerve's.

## Re-thinking

Not every task succeeds on the first try. When a nerve returns an error — wrong nerve selected, missing data, tool failure — the brain doesn't give up. It **re-thinks**.

Re-thinking feeds the error context back into `Brain.think()`:

```
Original task + error details + what was tried → new decision
```

The brain can choose a different nerve, ask for clarification, or try a completely different approach. This is not retry logic — it's a new decision with more information.

### Circuit Breakers

Re-thinking has limits. A circuit breaker tracks:

- **Attempt count** — max re-thinks per task (default: 3)
- **Nerve repetition** — won't invoke the same nerve twice for the same error
- **Time budget** — total elapsed time across all attempts

When the circuit breaker trips, the brain responds with what it knows — partial results, an honest error message, or a clarification request. It never spins in a loop.

## Error Boundaries

Each stage of the pipeline has its own error handling:

- **Intent classification failure** — defaults to "direct" and continues
- **LLM decision failure** — returns a `RespondDecision` with a fallback message
- **Normalization failure** — passes the decision through unchanged
- **Validation failure** — returns a structured error to the user
- **Execution failure** — triggers re-thinking (subject to circuit breaker)
- **Communication failure** — returns the raw result without personality rewrite

No silent failures. Every error is logged, tracked, and either recovered from or surfaced to the user.
