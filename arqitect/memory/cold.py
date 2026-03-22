"""
Cold Memory — SQLite knowledge graph.
Stores permanent facts, nerve registry, and tool statistics.
Survives full restarts.
"""

import json
import os
import sqlite3

from arqitect.config.loader import get_memory_dir
from arqitect.types import NerveRole

_DB_PATH = os.path.join(get_memory_dir(), "knowledge.db")


def _ensure_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            UNIQUE(category, key)
        );

        CREATE TABLE IF NOT EXISTS nerve_registry (
            name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            total_invocations INTEGER DEFAULT 0,
            successes INTEGER DEFAULT 0,
            failures INTEGER DEFAULT 0,
            origin TEXT DEFAULT 'local'
        );

        CREATE TABLE IF NOT EXISTS tool_stats (
            name TEXT PRIMARY KEY,
            total_calls INTEGER DEFAULT 0,
            successes INTEGER DEFAULT 0,
            failures INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS nerve_tools (
            nerve TEXT NOT NULL,
            tool TEXT NOT NULL,
            use_count INTEGER DEFAULT 0,
            PRIMARY KEY (nerve, tool)
        );

        CREATE TABLE IF NOT EXISTS qualification_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_type TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            qualified INTEGER NOT NULL DEFAULT 0,
            score REAL NOT NULL DEFAULT 0.0,
            iterations INTEGER DEFAULT 1,
            test_count INTEGER DEFAULT 0,
            pass_count INTEGER DEFAULT 0,
            details TEXT DEFAULT '{}',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(subject_type, subject_name)
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            display_name TEXT DEFAULT '',
            role TEXT DEFAULT 'user',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            secrets TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS user_links (
            connector TEXT NOT NULL,
            connector_id TEXT NOT NULL,
            user_id TEXT NOT NULL REFERENCES users(user_id),
            linked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (connector, connector_id)
        );

        CREATE INDEX IF NOT EXISTS idx_user_links_user ON user_links(user_id);

        CREATE TABLE IF NOT EXISTS verification_codes (
            connector TEXT NOT NULL,
            connector_id TEXT NOT NULL,
            email TEXT NOT NULL,
            code TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (connector, connector_id)
        );

        CREATE TABLE IF NOT EXISTS personality_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            data TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS personality_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            old_traits TEXT NOT NULL,
            new_traits TEXT NOT NULL,
            changes TEXT NOT NULL,
            observation_summary TEXT DEFAULT '',
            confidence REAL DEFAULT 0.0
        );
    """)
    # Migrate: add email column to users if missing
    try:
        conn.execute("SELECT email FROM users LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''")
    # Migrate: add system_prompt and examples columns if missing
    try:
        conn.execute("SELECT system_prompt FROM nerve_registry LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE nerve_registry ADD COLUMN system_prompt TEXT DEFAULT ''")
    try:
        conn.execute("SELECT examples FROM nerve_registry LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE nerve_registry ADD COLUMN examples TEXT DEFAULT '[]'")
    # Migrate: add is_sense column if missing
    try:
        conn.execute("SELECT is_sense FROM nerve_registry LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE nerve_registry ADD COLUMN is_sense INTEGER DEFAULT 0")
    # Migrate: add role column if missing
    try:
        conn.execute("SELECT role FROM nerve_registry LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE nerve_registry ADD COLUMN role TEXT DEFAULT 'tool'")
    # Migrate: add embedding column for cached description embeddings
    try:
        conn.execute("SELECT embedding FROM nerve_registry LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE nerve_registry ADD COLUMN embedding TEXT DEFAULT ''")
    # Migrate: add model_adapters column (JSON) for model-specific overrides
    try:
        conn.execute("SELECT model_adapters FROM nerve_registry LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE nerve_registry ADD COLUMN model_adapters TEXT DEFAULT '{}'")
    # Migrate: add last_invoked_at for usage-based reconciliation priority
    try:
        conn.execute("SELECT last_invoked_at FROM nerve_registry LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE nerve_registry ADD COLUMN last_invoked_at TEXT DEFAULT NULL")
    # Migrate: add origin column (local | community)
    try:
        conn.execute("SELECT origin FROM nerve_registry LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE nerve_registry ADD COLUMN origin TEXT DEFAULT 'local'")
    conn.commit()


class ColdMemory:
    def __init__(self):
        self.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = __import__('threading').Lock()
        _ensure_db(self.conn)

    # ── Facts ────────────────────────────────────────────────────────────

    def set_fact(self, category: str, key: str, value: str, confidence: float = 1.0):
        """Store or update a permanent fact."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO facts (category, key, value, confidence) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(category, key) DO UPDATE SET value=excluded.value, confidence=excluded.confidence",
                (category, key, value, confidence),
            )
            self.conn.commit()

    def get_fact(self, category: str, key: str) -> str | None:
        """Retrieve a single fact."""
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM facts WHERE category=? AND key=?", (category, key)
            ).fetchone()
        return row["value"] if row else None

    def get_facts(self, category: str) -> dict:
        """Retrieve all facts in a category."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT key, value FROM facts WHERE category=?", (category,)
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ── Per-User Facts ────────────────────────────────────────────────────

    def set_user_fact(self, user_id: str, key: str, value: str, confidence: float = 1.0):
        """Store a fact scoped to a specific user."""
        self.set_fact(f"user:{user_id}", key, value, confidence)

    def get_user_facts(self, user_id: str) -> dict:
        """Get all facts for a specific user."""
        return self.get_facts(f"user:{user_id}")

    # ── Nerve Registry ───────────────────────────────────────────────────

    def register_nerve(self, name: str, description: str, origin: str = "local"):
        """Register or update a nerve.

        Args:
            name: Nerve identifier.
            description: Human-readable description.
            origin: Provenance — 'local' for user-created, 'community' for arqitect-community.
        """
        with self._lock:
            self.conn.execute(
                "INSERT INTO nerve_registry (name, description, origin) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                "origin=excluded.origin",
                (name, description, origin),
            )
            self.conn.commit()

    def record_nerve_invocation(self, name: str, success: bool):
        """Increment invocation stats and update last_invoked_at timestamp.

        Uses UPSERT so nerves that exist on disk but were never register_nerve'd
        still get a row created — otherwise the UPDATE matches 0 rows and
        last_invoked_at stays NULL, preventing dreamstate from tuning the nerve.
        """
        col = "successes" if success else "failures"
        other = "failures" if success else "successes"
        with self._lock:
            self.conn.execute(
                f"INSERT INTO nerve_registry (name, description, total_invocations, "
                f"{col}, {other}, last_invoked_at) VALUES (?, '', 1, 1, 0, datetime('now')) "
                f"ON CONFLICT(name) DO UPDATE SET total_invocations = total_invocations + 1, "
                f"{col} = {col} + 1, last_invoked_at = datetime('now')",
                (name,),
            )
            self.conn.commit()

    def get_nerve_info(self, name: str) -> dict | None:
        """Get nerve registry info including origin."""
        with self._lock:
            row = self.conn.execute("SELECT * FROM nerve_registry WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d.setdefault("origin", "local")
        return d

    def register_nerve_rich(self, name: str, description: str, system_prompt: str = "",
                            examples_json: str = "[]", role: str = NerveRole.TOOL,
                            origin: str = "local", **_kwargs):
        """Register or update a nerve with rich metadata.

        Routes to the appropriate persistence strategy based on origin.

        Args:
            name: Nerve identifier.
            description: Human-readable description.
            system_prompt: LLM system prompt (ignored for community nerves).
            examples_json: JSON array of few-shot examples (ignored for community nerves).
            role: Nerve role classification (tool/creative/code).
            origin: Provenance — 'local' or 'community'.
        """
        if origin == "community":
            self._register_community_nerve(name, description, role)
        else:
            self._register_local_nerve(name, description, system_prompt, examples_json, role)

    def _register_community_nerve(self, name: str, description: str, role: str):
        """Persist a community nerve — identity and role only, no prompts.

        System prompt and examples live in the community cache
        (.community/cache/nerves/{name}/) and are resolved at read time.
        """
        with self._lock:
            self.conn.execute(
                "INSERT INTO nerve_registry (name, description, role, origin) "
                "VALUES (?, ?, ?, 'community') "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                "role=excluded.role, origin='community'",
                (name, description, role),
            )
            self.conn.commit()

    def _register_local_nerve(self, name: str, description: str,
                              system_prompt: str, examples_json: str, role: str):
        """Persist a local nerve with full metadata including prompts."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO nerve_registry (name, description, system_prompt, examples, role, origin) "
                "VALUES (?, ?, ?, ?, ?, 'local') "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                "system_prompt=excluded.system_prompt, examples=excluded.examples, "
                "role=excluded.role, origin='local'",
                (name, description, system_prompt, examples_json, role),
            )
            self.conn.commit()

    def get_nerve_metadata(self, name: str) -> dict:
        """Get nerve metadata: system_prompt, examples, role, origin.

        For community nerves, system_prompt and examples will be empty here —
        callers should resolve them from the community cache.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT description, system_prompt, examples, role, origin "
                "FROM nerve_registry WHERE name=?", (name,)
            ).fetchone()
        if not row:
            return {"description": "", "system_prompt": "", "examples": [],
                    "role": NerveRole.TOOL, "origin": "local"}
        examples = []
        try:
            examples = json.loads(row["examples"]) if row["examples"] else []
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "description": row["description"],
            "system_prompt": row["system_prompt"] or "",
            "examples": examples,
            "role": row["role"] or NerveRole.TOOL,
            "origin": row["origin"] or "local",
        }

    def get_test_bank(self, nerve_name: str) -> list[dict]:
        """Get stored test cases for a nerve."""
        raw = self.get_fact("test_bank", nerve_name)
        if raw:
            try:
                bank = json.loads(raw)
                if isinstance(bank, list):
                    return bank
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    def set_test_bank(self, nerve_name: str, tests: list[dict]):
        """Store test cases for a nerve."""
        self.set_fact("test_bank", nerve_name, json.dumps(tests))

    def list_nerves(self) -> dict[str, str]:
        """Return {name: description} from the registry."""
        with self._lock:
            rows = self.conn.execute("SELECT name, description FROM nerve_registry").fetchall()
        return {r["name"]: r["description"] for r in rows}

    def get_last_invoked_at(self, name: str) -> str | None:
        """Return the last_invoked_at timestamp for a nerve, or None if never invoked."""
        with self._lock:
            row = self.conn.execute(
                "SELECT last_invoked_at FROM nerve_registry WHERE name=?", (name,)
            ).fetchone()
        return row["last_invoked_at"] if row and row["last_invoked_at"] else None

    # ── Nerve Embeddings ──────────────────────────────────────────────────

    def get_nerve_embedding(self, name: str) -> list[float] | None:
        """Retrieve cached embedding vector for a nerve. Returns None if not cached."""
        with self._lock:
            row = self.conn.execute(
                "SELECT embedding FROM nerve_registry WHERE name=?", (name,)
            ).fetchone()
        if not row or not row["embedding"]:
            return None
        try:
            emb = json.loads(row["embedding"])
            if isinstance(emb, list) and len(emb) > 0:
                return emb
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def set_nerve_embedding(self, name: str, embedding: list[float]):
        """Cache an embedding vector for a nerve in the registry."""
        with self._lock:
            self.conn.execute(
                "UPDATE nerve_registry SET embedding=? WHERE name=?",
                (json.dumps(embedding), name),
            )
            self.conn.commit()

    # ── Senses (Protected Nerves) ─────────────────────────────────────────

    def register_sense(self, name: str, description: str, system_prompt: str = "", examples_json: str = "[]"):
        """Register a sense (protected nerve) with is_sense=1. Senses are always local."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO nerve_registry (name, description, system_prompt, examples, is_sense, origin) "
                "VALUES (?, ?, ?, ?, 1, 'local') "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                "system_prompt=excluded.system_prompt, examples=excluded.examples, "
                "is_sense=1, origin='local'",
                (name, description, system_prompt, examples_json),
            )
            self.conn.commit()

    def is_sense(self, name: str) -> bool:
        """Check if a nerve is a protected sense."""
        with self._lock:
            row = self.conn.execute(
                "SELECT is_sense FROM nerve_registry WHERE name=?", (name,)
            ).fetchone()
        return bool(row["is_sense"]) if row else False

    def list_senses(self) -> dict[str, str]:
        """Return {name: description} for all senses."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT name, description FROM nerve_registry WHERE is_sense=1"
            ).fetchall()
        return {r["name"]: r["description"] for r in rows}

    # ── Tool Stats ───────────────────────────────────────────────────────

    def record_tool_call(self, name: str, success: bool):
        """Record a tool call (legacy — kept for backward compat with record_episode)."""
        col = "successes" if success else "failures"
        with self._lock:
            self.conn.execute(
                f"INSERT INTO tool_stats (name, total_calls, {col}) VALUES (?, 1, 1) "
                f"ON CONFLICT(name) DO UPDATE SET total_calls = total_calls + 1, {col} = {col} + 1",
                (name,),
            )
            self.conn.commit()

    # ── Nerve-Tool Relationships ─────────────────────────────────────────

    def add_nerve_tool(self, nerve: str, tool: str):
        """Record that a nerve knows/uses a tool."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO nerve_tools (nerve, tool, use_count) VALUES (?, ?, 1) "
                "ON CONFLICT(nerve, tool) DO UPDATE SET use_count = use_count + 1",
                (nerve, tool),
            )
            self.conn.commit()

    def get_nerve_tools(self, nerve: str) -> list[str]:
        """Get list of tool names a nerve knows about."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT tool FROM nerve_tools WHERE nerve=? ORDER BY use_count DESC", (nerve,)
            ).fetchall()
        return [r["tool"] for r in rows]

    def get_nerve_tools_with_counts(self, nerve: str) -> list[dict]:
        """Get tools with use counts for a nerve, ordered by use_count descending."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT tool, use_count FROM nerve_tools WHERE nerve=? ORDER BY use_count DESC",
                (nerve,),
            ).fetchall()
        return [{"tool": r["tool"], "use_count": r["use_count"]} for r in rows]

    def update_nerve_description(self, name: str, description: str):
        """Update only the description of an existing nerve.

        Used by fanout to expand narrow task-specific descriptions into
        broad domain descriptions without overwriting system_prompt,
        examples, or role metadata.
        """
        with self._lock:
            self.conn.execute(
                "UPDATE nerve_registry SET description=? WHERE name=?",
                (description, name),
            )
            self.conn.commit()

    def delete_nerve(self, name: str):
        """Remove a nerve from cold memory (registry + qualification results)."""
        with self._lock:
            self.conn.execute("DELETE FROM nerve_registry WHERE name=?", (name,))
            self.conn.execute("DELETE FROM qualification_results WHERE subject_name=?", (name,))
            self.conn.commit()

    def get_nerve_origin(self, name: str) -> str:
        """Return the origin of a nerve ('local' or 'community')."""
        with self._lock:
            row = self.conn.execute(
                "SELECT origin FROM nerve_registry WHERE name=?", (name,)
            ).fetchone()
        return row["origin"] if row and row["origin"] else "local"

    def is_community_nerve(self, name: str) -> bool:
        """Check if a nerve originates from the community cache."""
        return self.get_nerve_origin(name) == "community"

    # ── Qualification Results ─────────────────────────────────────────────

    def record_qualification(self, subject_type: str, subject_name: str,
                             qualified: bool, score: float, iterations: int,
                             test_count: int, pass_count: int, details: str = "{}"):
        """Record or update qualification results for a nerve or tool."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO qualification_results "
                "(subject_type, subject_name, qualified, score, iterations, test_count, pass_count, details, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(subject_type, subject_name) DO UPDATE SET "
                "qualified=excluded.qualified, score=excluded.score, iterations=excluded.iterations, "
                "test_count=excluded.test_count, pass_count=excluded.pass_count, details=excluded.details, "
                "timestamp=CURRENT_TIMESTAMP",
                (subject_type, subject_name, int(qualified), score, iterations, test_count, pass_count, details),
            )
            self.conn.commit()

    def get_qualification(self, subject_type: str, subject_name: str) -> dict | None:
        """Get qualification result for a specific subject."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM qualification_results WHERE subject_type=? AND subject_name=?",
                (subject_type, subject_name),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["qualified"] = bool(d["qualified"])
        try:
            d["details"] = json.loads(d["details"])
        except (json.JSONDecodeError, TypeError):
            d["details"] = {}
        return d

    def is_qualified(self, subject_type: str, subject_name: str) -> bool:
        """Check if a subject has passed qualification."""
        with self._lock:
            row = self.conn.execute(
                "SELECT qualified FROM qualification_results WHERE subject_type=? AND subject_name=?",
                (subject_type, subject_name),
            ).fetchone()
        return bool(row["qualified"]) if row else False

    def was_qualified(self, subject_type: str, subject_name: str) -> bool:
        """Check if a subject has been through qualification (pass or fail).

        Unlike is_qualified() which only returns True for passed subjects,
        this returns True if any qualification attempt was made. Used to
        avoid re-running qualification on every restart for nerves that
        already went through the process.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM qualification_results WHERE subject_type=? AND subject_name=?",
                (subject_type, subject_name),
            ).fetchone()
        return row is not None

    def get_all_nerve_data(self) -> dict:
        """Load all nerve data in bulk: registry info, tools, and qualifications.

        Returns a dict keyed by nerve name, each containing:
          - description, is_sense, total_invocations, successes, failures
          - system_prompt, examples, role
          - tools: list[str]
          - qualification: {qualified, score, iterations, test_count, pass_count, details, timestamp} or None
        """
        with self._lock:
            # All nerves from registry
            nerves = {}
            for row in self.conn.execute(
                "SELECT name, description, is_sense, total_invocations, successes, failures, "
                "system_prompt, examples, role, origin FROM nerve_registry"
            ).fetchall():
                examples = []
                try:
                    examples = json.loads(row["examples"]) if row["examples"] else []
                except (json.JSONDecodeError, TypeError):
                    pass
                nerves[row["name"]] = {
                    "description": row["description"],
                    "is_sense": bool(row["is_sense"]),
                    "total_invocations": row["total_invocations"] or 0,
                    "successes": row["successes"] or 0,
                    "failures": row["failures"] or 0,
                    "system_prompt": row["system_prompt"] or "",
                    "examples": examples,
                    "role": row["role"] or NerveRole.TOOL,
                    "origin": row["origin"] or "local",
                    "tools": [],
                    "qualification": None,
                }

            # All nerve-tool relationships
            for row in self.conn.execute(
                "SELECT nerve, tool FROM nerve_tools ORDER BY use_count DESC"
            ).fetchall():
                if row["nerve"] in nerves:
                    nerves[row["nerve"]]["tools"].append(row["tool"])

            # All nerve qualifications
            for row in self.conn.execute(
                "SELECT subject_name, qualified, score, iterations, test_count, pass_count, "
                "details, timestamp FROM qualification_results WHERE subject_type='nerve'"
            ).fetchall():
                if row["subject_name"] in nerves:
                    details = {}
                    try:
                        details = json.loads(row["details"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                    nerves[row["subject_name"]]["qualification"] = {
                        "qualified": bool(row["qualified"]),
                        "score": row["score"],
                        "iterations": row["iterations"],
                        "test_count": row["test_count"],
                        "pass_count": row["pass_count"],
                        "details": details,
                        "timestamp": row["timestamp"],
                    }

            return nerves

    def list_qualifications(self) -> list[dict]:
        """List all qualification results."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT subject_type, subject_name, qualified, score, iterations, "
                "test_count, pass_count, timestamp FROM qualification_results ORDER BY timestamp DESC"
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["qualified"] = bool(d["qualified"])
            results.append(d)
        return results

    # ── Personality Signals ───────────────────────────────────────────────

    def append_personality_signal(self, signal: dict):
        """Append an interaction signal for personality observation.

        Signals are append-only during live conversation and flushed
        after dream state processes them.

        Args:
            signal: Signal dict (serialized to JSON for storage).
        """
        import time as _time
        with self._lock:
            self.conn.execute(
                "INSERT INTO personality_signals (timestamp, data) VALUES (?, ?)",
                (_time.time(), json.dumps(signal)),
            )
            self.conn.commit()

    def get_personality_signals(self) -> list[dict]:
        """Retrieve all accumulated personality signals.

        Returns:
            List of signal dicts ordered by timestamp ascending.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT data FROM personality_signals ORDER BY timestamp ASC"
            ).fetchall()
        results = []
        for row in rows:
            try:
                results.append(json.loads(row["data"]))
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    def flush_personality_signals(self):
        """Delete all personality signals after dream state has processed them."""
        with self._lock:
            self.conn.execute("DELETE FROM personality_signals")
            self.conn.commit()

    # ── Personality History ───────────────────────────────────────────────

    def append_personality_history(self, entry: dict):
        """Record a personality evolution event.

        Args:
            entry: Dict with timestamp, old_traits, new_traits, changes,
                   observation_summary, confidence.
        """
        with self._lock:
            self.conn.execute(
                "INSERT INTO personality_history "
                "(timestamp, old_traits, new_traits, changes, observation_summary, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.get("timestamp", 0.0),
                    json.dumps(entry.get("old_traits", {})),
                    json.dumps(entry.get("new_traits", {})),
                    json.dumps(entry.get("changes", [])),
                    entry.get("observation_summary", ""),
                    entry.get("confidence", 0.0),
                ),
            )
            self.conn.commit()

    def get_personality_history(self, limit: int = 0) -> list[dict]:
        """Retrieve personality evolution history.

        Args:
            limit: Max entries to return (0 = all).

        Returns:
            List of history entry dicts ordered by timestamp ascending.
        """
        query = "SELECT * FROM personality_history ORDER BY timestamp ASC"
        if limit > 0:
            query += f" LIMIT {limit}"
        with self._lock:
            rows = self.conn.execute(query).fetchall()
        results = []
        for row in rows:
            entry = {
                "timestamp": row["timestamp"],
                "observation_summary": row["observation_summary"],
                "confidence": row["confidence"],
            }
            for json_field in ("old_traits", "new_traits", "changes"):
                try:
                    entry[json_field] = json.loads(row[json_field])
                except (json.JSONDecodeError, TypeError):
                    entry[json_field] = {}
            results.append(entry)
        return results

    # ── Users ────────────────────────────────────────────────────────────

    def resolve_user(self, connector: str, connector_id: str) -> str:
        """Resolve a connector identity to a canonical user_id.
        Returns user_id if found, empty string if not linked yet."""
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id FROM user_links WHERE connector=? AND connector_id=?",
                (connector, connector_id),
            ).fetchone()
            if row:
                self.conn.execute(
                    "UPDATE users SET last_seen=CURRENT_TIMESTAMP WHERE user_id=?",
                    (row["user_id"],),
                )
                self.conn.commit()
                return row["user_id"]
            return ""

    def create_user_with_email(self, email: str, connector: str, connector_id: str) -> str:
        """Create a new verified user with email, or link to existing user with same email.
        Returns the canonical user_id."""
        email = email.lower().strip()
        with self._lock:
            # Check if a user with this email already exists
            existing = self.conn.execute(
                "SELECT user_id FROM users WHERE email=?", (email,),
            ).fetchone()
            if existing:
                # Link this connector to the existing user
                user_id = existing["user_id"]
                self.conn.execute(
                    "INSERT INTO user_links (connector, connector_id, user_id) VALUES (?, ?, ?) "
                    "ON CONFLICT(connector, connector_id) DO UPDATE SET user_id=excluded.user_id",
                    (connector, connector_id, user_id),
                )
                self.conn.execute(
                    "UPDATE users SET last_seen=CURRENT_TIMESTAMP WHERE user_id=?",
                    (user_id,),
                )
                self.conn.commit()
                return user_id
            # Create new user with email
            import uuid
            user_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO users (user_id, email) VALUES (?, ?)", (user_id, email),
            )
            self.conn.execute(
                "INSERT INTO user_links (connector, connector_id, user_id) VALUES (?, ?, ?)",
                (connector, connector_id, user_id),
            )
            self.conn.commit()
            return user_id

    def get_user(self, user_id: str) -> dict | None:
        """Get user record."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM users WHERE user_id=?", (user_id,),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["secrets"] = json.loads(d["secrets"])
        except (json.JSONDecodeError, TypeError):
            d["secrets"] = {}
        return d

    def get_user_profile(self, user_id: str) -> dict:
        """Get user profile: name, gender, and any known facts."""
        profile = {}
        user = self.get_user(user_id)
        if user:
            if user.get("display_name"):
                profile["name"] = user["display_name"]
        facts = self.get_user_facts(user_id)
        if facts.get("name"):
            profile["name"] = facts["name"]  # user-stated name takes priority
        if facts.get("gender"):
            profile["gender"] = facts["gender"]
        if facts.get("language"):
            profile["language"] = facts["language"]
        # Include any other personal facts
        for k, v in facts.items():
            if k not in profile and k not in ("city", "country", "timezone"):
                profile[k] = v
        return profile

    def get_user_role(self, user_id: str) -> str:
        """Get user role. Returns 'anon' if user not found."""
        with self._lock:
            row = self.conn.execute(
                "SELECT role FROM users WHERE user_id=?", (user_id,),
            ).fetchone()
        return row["role"] if row else "anon"

    def set_user_role(self, user_id: str, role: str):
        """Set user role."""
        with self._lock:
            self.conn.execute(
                "UPDATE users SET role=? WHERE user_id=?", (role, user_id),
            )
            self.conn.commit()

    def set_user_display_name(self, user_id: str, name: str):
        """Set user display name."""
        with self._lock:
            self.conn.execute(
                "UPDATE users SET display_name=? WHERE user_id=?", (name, user_id),
            )
            self.conn.commit()

    def set_user_secret(self, user_id: str, key: str, value: str):
        """Set a user-specific secret (for MCP integrations)."""
        with self._lock:
            row = self.conn.execute(
                "SELECT secrets FROM users WHERE user_id=?", (user_id,),
            ).fetchone()
            if not row:
                return
            try:
                secrets = json.loads(row["secrets"])
            except (json.JSONDecodeError, TypeError):
                secrets = {}
            secrets[key] = value
            self.conn.execute(
                "UPDATE users SET secrets=? WHERE user_id=?",
                (json.dumps(secrets), user_id),
            )
            self.conn.commit()

    def link_user_connector(self, user_id: str, connector: str, connector_id: str):
        """Link an additional connector identity to an existing user."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO user_links (connector, connector_id, user_id) VALUES (?, ?, ?) "
                "ON CONFLICT(connector, connector_id) DO UPDATE SET user_id=excluded.user_id",
                (connector, connector_id, user_id),
            )
            self.conn.commit()

    def get_user_links(self, user_id: str) -> list[dict]:
        """Get all connector links for a user."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT connector, connector_id FROM user_links WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Email Verification ───────────────────────────────────────────────

    def get_user_by_email(self, email: str) -> dict | None:
        """Find a user by email address."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM users WHERE email=?", (email.lower().strip(),),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["secrets"] = json.loads(d["secrets"])
        except (json.JSONDecodeError, TypeError):
            d["secrets"] = {}
        return d

    def set_user_email(self, user_id: str, email: str):
        """Set user email."""
        with self._lock:
            self.conn.execute(
                "UPDATE users SET email=? WHERE user_id=?", (email.lower().strip(), user_id),
            )
            self.conn.commit()

    def is_user_verified(self, user_id: str) -> bool:
        """Check if user has a verified email."""
        with self._lock:
            row = self.conn.execute(
                "SELECT email FROM users WHERE user_id=?", (user_id,),
            ).fetchone()
        return bool(row and row["email"])

    def store_verification_code(self, connector: str, connector_id: str, email: str, code: str):
        """Store a pending verification code."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO verification_codes (connector, connector_id, email, code) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(connector, connector_id) DO UPDATE SET email=excluded.email, code=excluded.code, created_at=CURRENT_TIMESTAMP",
                (connector, connector_id, email.lower().strip(), code),
            )
            self.conn.commit()

    def verify_code(self, connector: str, connector_id: str, code: str) -> str | None:
        """Verify a code. Returns email if valid, None if invalid. Deletes code on success."""
        with self._lock:
            row = self.conn.execute(
                "SELECT email, code, created_at FROM verification_codes WHERE connector=? AND connector_id=?",
                (connector, connector_id),
            ).fetchone()
            if not row:
                return None
            # Check expiry (10 minutes)
            import datetime
            created = datetime.datetime.fromisoformat(row["created_at"]).replace(tzinfo=datetime.timezone.utc)
            if (datetime.datetime.now(datetime.timezone.utc) - created).total_seconds() > 600:
                self.conn.execute(
                    "DELETE FROM verification_codes WHERE connector=? AND connector_id=?",
                    (connector, connector_id),
                )
                self.conn.commit()
                return None
            if row["code"] != code.strip():
                return None
            email = row["email"]
            self.conn.execute(
                "DELETE FROM verification_codes WHERE connector=? AND connector_id=?",
                (connector, connector_id),
            )
            self.conn.commit()
            return email

    def delete_verification_code(self, connector: str, connector_id: str):
        """Delete a pending verification code."""
        with self._lock:
            self.conn.execute(
                "DELETE FROM verification_codes WHERE connector=? AND connector_id=?",
                (connector, connector_id),
            )
            self.conn.commit()
