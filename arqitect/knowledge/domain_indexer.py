"""
Domain Knowledge Indexer — fetches authoritative sources and extracts structured facts.

Run as a standalone script: python domain_indexer.py <domain>
Stores facts in cold memory (SQLite) under category="domain:<domain>".
"""

import json
import os
import re
import sqlite3
import sys

import requests

from arqitect.config.loader import get_project_root

_DB_PATH = os.path.join(str(get_project_root()), "memory", "knowledge.db")
BRAIN_MODEL = "brain"

# ── Domain Detection ──────────────────────────────────────────────────────────

DOMAIN_MAP = {
    # keyword -> domain
    "html": "html", "<div": "html", "<span": "html", "<section": "html",
    "<header": "html", "<footer": "html", "<nav": "html", "<main": "html",
    "<article": "html", "<aside": "html", "<h1": "html", "<h2": "html",
    "<h3": "html", "<p>": "html", "<ul": "html", "<ol": "html", "<li": "html",
    "<a ": "html", "<img": "html", "<form": "html", "<input": "html",
    "<table": "html", "<button": "html", "doctype": "html", "hypertext": "html",
    "css": "css", "stylesheet": "css", "flexbox": "css", "grid layout": "css",
    "selector": "css", "margin": "css", "padding": "css",
    "javascript": "javascript", "ecmascript": "javascript", "async await": "javascript",
    "promise": "javascript", "callback": "javascript",
    "python": "python", "def ": "python", "import ": "python",
    "list comprehension": "python", "decorator": "python",
    "sql": "sql", "select ": "sql", "insert ": "sql", "update ": "sql",
    "delete from": "sql", "join": "sql", "where clause": "sql",
    "http": "http", "rest api": "http", "status code": "http",
    "get request": "http", "post request": "http",
    "git": "git", "commit": "git", "branch": "git", "merge": "git", "rebase": "git",
    "linux": "linux", "bash": "linux", "chmod": "linux", "grep": "linux",
    "math": "math", "algebra": "math", "calculus": "math", "matrix": "math",
}

AUTHORITY_URLS = {
    "html": "https://developer.mozilla.org/en-US/docs/Web/HTML/Element",
    "css": "https://developer.mozilla.org/en-US/docs/Web/CSS/Reference",
    "javascript": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference",
    "python": "https://docs.python.org/3/library/index.html",
    "sql": "https://www.w3schools.com/sql/default.asp",
    "http": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status",
    "git": "https://git-scm.com/docs",
    "linux": "https://www.gnu.org/software/bash/manual/bash.html",
    "math": "https://en.wikipedia.org/wiki/Mathematics",
}


def detect_domain(user_input: str) -> str | None:
    """Keyword-based domain detection. Returns domain name or None."""
    inp_lower = user_input.lower()
    for keyword, domain in DOMAIN_MAP.items():
        if keyword in inp_lower:
            return domain
    return None


def is_indexed(domain: str) -> bool:
    """Check if domain facts already exist in cold memory."""
    if not os.path.exists(_DB_PATH):
        return False
    try:
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE category=?", (f"domain:{domain}",)
        ).fetchone()
        conn.close()
        return row[0] > 0
    except Exception:
        return False


def _fetch_url(url: str) -> str:
    """Fetch a URL and strip HTML to plain text."""
    try:
        resp = requests.get(
            url, timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Arqitect/1.0)"},
        )
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]
    except Exception as e:
        return f"Error: {e}"


def _extract_facts_with_llm(domain: str, source_text: str) -> list[dict]:
    """Use LLM to extract structured facts from source text."""
    prompt = (
        f"You are a knowledge extractor. From the following reference text about '{domain}', "
        f"extract up to 30 key facts as a JSON array.\n\n"
        f"Each fact should be: {{\"key\": \"concept_name\", \"value\": \"concise factual description\"}}\n\n"
        f"Rules:\n"
        f"- key should be snake_case like 'div_purpose', 'h1_usage', 'join_inner'\n"
        f"- value should be a single clear sentence (max 100 chars)\n"
        f"- Focus on definitions, purposes, and common usage patterns\n"
        f"- Only include facts that are objectively true\n\n"
        f"Source text:\n{source_text[:3000]}\n\n"
        f"Return ONLY the JSON array, no other text."
    )
    try:
        from arqitect.inference.router import generate_for_role
        raw = generate_for_role("brain", prompt, max_tokens=1024)

        # Parse JSON from response
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            facts = json.loads(text[start:end])
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, dict) and f.get("key") and f.get("value")]
    except Exception as e:
        print(f"[INDEXER] LLM extraction failed: {e}", file=sys.stderr)
    return []


def _store_facts(domain: str, facts: list[dict]):
    """Store extracted facts in cold memory SQLite."""
    conn = sqlite3.connect(_DB_PATH)
    # Ensure facts table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            UNIQUE(category, key)
        )
    """)
    conn.commit()

    category = f"domain:{domain}"
    for fact in facts:
        key = fact["key"].strip().lower().replace(" ", "_")
        value = fact["value"].strip()
        try:
            conn.execute(
                "INSERT INTO facts (category, key, value, confidence) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(category, key) DO UPDATE SET value=excluded.value, confidence=excluded.confidence",
                (category, key, value, 0.9),
            )
        except Exception as e:
            print(f"[INDEXER] Failed to store fact {key}: {e}", file=sys.stderr)
    conn.commit()
    conn.close()


def index_domain(domain: str):
    """Main indexing pipeline for a domain."""
    if is_indexed(domain):
        print(f"[INDEXER] Domain '{domain}' already indexed, skipping.")
        return

    print(f"[INDEXER] Indexing domain: {domain}")

    # Fetch authoritative source
    url = AUTHORITY_URLS.get(domain)
    if url:
        source_text = _fetch_url(url)
    else:
        # Fallback: DuckDuckGo search for the domain
        try:
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": f"{domain} reference documentation", "format": "json", "no_html": 1},
                timeout=10,
            )
            data = resp.json()
            source_text = data.get("AbstractText", "")
            # Also grab related topics
            for topic in data.get("RelatedTopics", [])[:10]:
                if isinstance(topic, dict) and topic.get("Text"):
                    source_text += " " + topic["Text"]
        except Exception as e:
            print(f"[INDEXER] Fallback search failed: {e}", file=sys.stderr)
            return

    if not source_text or source_text.startswith("Error:"):
        print(f"[INDEXER] Could not fetch source for domain '{domain}'")
        return

    # Extract facts with LLM
    facts = _extract_facts_with_llm(domain, source_text)
    if not facts:
        print(f"[INDEXER] No facts extracted for domain '{domain}'")
        return

    # Store in cold memory
    _store_facts(domain, facts)
    print(f"[INDEXER] Stored {len(facts)} facts for domain '{domain}'")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python domain_indexer.py <domain>", file=sys.stderr)
        sys.exit(1)

    domain_arg = sys.argv[1].lower().strip()
    index_domain(domain_arg)
