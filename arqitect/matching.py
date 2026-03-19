"""
Deterministic keyword-scoring for tool and nerve matching.
Hybrid: keyword scoring + optional embedding similarity.
Replaces all LLM-based matching/routing calls.
"""

import re
from collections import OrderedDict

# ── Scoring weights (tune these to adjust matching sensitivity) ──────────────
NAME_TOKEN_WEIGHT = 3.0       # Exact match in tool/nerve name
STEM_MATCH_WEIGHT = 2.0       # Stem variant match in name
DESC_TOKEN_WEIGHT = 1.0       # Exact match in description
DESC_STEM_WEIGHT = 0.5        # Stem variant match in description
PARAM_TOKEN_WEIGHT = 0.5      # Exact match in parameters
PARAM_STEM_WEIGHT = 0.25      # Stem variant match in parameters
MIN_SUBSTR_LEN = 4            # Minimum token length for stem matching
PREFIX_OVERLAP_RATIO = 0.75   # Required prefix overlap for stem matching
SCORE_NORMALIZE_FACTOR = 0.15 # Min fraction of max possible score per token
BEST_MATCH_THRESHOLD = 0.5    # Min fraction of token count for best_match_tool
KEYWORD_WEIGHT = 0.4          # Keyword score weight in hybrid blend
EMBEDDING_WEIGHT = 0.6        # Embedding similarity weight in hybrid blend
SENSE_BOOST = 2.0             # Scoring boost for core senses over regular nerves

# Minimal stopword list — only articles, prepositions, and conjunctions.
# Kept small because the embedding engine handles semantic similarity;
# keyword matching only needs to avoid the most generic function words.
_STOPWORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "it",
    "and", "or", "but", "if", "so", "as", "by", "be", "no", "not",
    "i", "me", "my", "we", "our",
})


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords and short tokens.

    If aggressive stopword removal leaves fewer than 2 tokens, falls back to
    a lenient mode that only removes the most generic words. This prevents
    short queries like "who are you?" or "tell me a joke" from scoring 0.
    """
    if not text:
        return set()
    words = set(re.split(r"[^a-z0-9]+", text.lower()))
    filtered = {w for w in words if len(w) > 1 and w not in _STOPWORDS}
    if len(filtered) >= 2:
        return filtered
    # Lenient fallback — only remove the absolute minimum function words
    _HARD_STOPS = frozenset({"the", "a", "an", "is", "of", "in", "on", "at", "to", "for", "by", "be"})
    lenient = {w for w in words if len(w) > 1 and w not in _HARD_STOPS}
    return lenient if lenient else filtered



def _is_stem_match(a: str, b: str) -> bool:
    """Check if two tokens are stem variants of the same word.

    Only matches if:
    - Both tokens are at least MIN_SUBSTR_LEN characters long
    - They share a common prefix of at least PREFIX_OVERLAP_RATIO of the shorter
      token's length AND at least MIN_SUBSTR_LEN characters
    This catches morphological variants (calculate/calculating, translate/translation)
    while preventing spurious matches (do/document, info/information, log/location).
    """
    if len(a) < MIN_SUBSTR_LEN or len(b) < MIN_SUBSTR_LEN:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    # Find shared prefix length
    prefix_len = 0
    for ca, cb in zip(shorter, longer):
        if ca != cb:
            break
        prefix_len += 1
    # Require shared prefix covers at least 75% of the shorter word
    # and is at least MIN_SUBSTR_LEN characters
    return prefix_len >= MIN_SUBSTR_LEN and prefix_len >= len(shorter) * PREFIX_OVERLAP_RATIO


def match_score(query: str, name: str, description: str, params=None) -> float:
    """Score a query against a tool/nerve.

    Weighting (configured via module-level constants):
      - name tokens: NAME_TOKEN_WEIGHT (exact), STEM_MATCH_WEIGHT (stem)
      - description tokens: DESC_TOKEN_WEIGHT (exact), DESC_STEM_WEIGHT (stem)
      - param tokens: PARAM_TOKEN_WEIGHT (exact), PARAM_STEM_WEIGHT (stem)

    Substring matching is restricted to stem variants (e.g. "translate" /
    "translation") to prevent false positives from short token containment.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0

    name_tokens = _tokenize(name.replace("_", " "))
    desc_tokens = _tokenize(description)

    score = 0.0
    for qt in q_tokens:
        # Check name tokens (exact and stem match)
        for nt in name_tokens:
            if qt == nt:
                score += NAME_TOKEN_WEIGHT
                break
            if _is_stem_match(qt, nt):
                score += STEM_MATCH_WEIGHT
                break
        # Check description tokens
        for dt in desc_tokens:
            if qt == dt:
                score += DESC_TOKEN_WEIGHT
                break
            if _is_stem_match(qt, dt):
                score += DESC_STEM_WEIGHT
                break

    # Check param tokens
    if params:
        param_str = str(params) if not isinstance(params, str) else params
        param_tokens = _tokenize(param_str.replace("_", " "))
        for qt in q_tokens:
            for pt in param_tokens:
                if qt == pt:
                    score += PARAM_TOKEN_WEIGHT
                    break
                if _is_stem_match(qt, pt):
                    score += PARAM_STEM_WEIGHT
                    break

    return score


def match_tools(query: str, tools_dict: dict, threshold: float = 1.0) -> list[tuple[str, float]]:
    """Rank all tools against a query. Returns [(name, score)] descending.

    Applies both an absolute threshold and a normalized check: a tool's
    score must reach at least SCORE_NORMALIZE_FACTOR of the query's max
    possible score. This prevents long queries from accumulating
    enough noise matches to pass threshold while still allowing legitimate
    matches where 1-2 domain-specific tokens overlap strongly.
    """
    q_tokens = _tokenize(query)
    q_count = len(q_tokens)
    # Minimum score = SCORE_NORMALIZE_FACTOR of max possible (each token can
    # score up to NAME_TOKEN_WEIGHT + DESC_TOKEN_WEIGHT).
    max_per_token = NAME_TOKEN_WEIGHT + DESC_TOKEN_WEIGHT
    normalized_min = q_count * SCORE_NORMALIZE_FACTOR * max_per_token if q_count > 0 else threshold

    effective_threshold = max(threshold, normalized_min)

    scored = []
    for name, info in tools_dict.items():
        desc = ""
        params = None
        if isinstance(info, dict):
            desc = info.get("description", "")
            params = info.get("params", None)
        s = match_score(query, name, desc, params)
        if s >= effective_threshold:
            scored.append((name, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# Core senses get a scoring boost so they are preferred over regular nerves
# for overlapping domains (e.g. "read file" → touch sense wins over a file_reader nerve)
from arqitect.brain.types import Sense
CORE_SENSES = frozenset(Sense)


# Cache for nerve description embeddings — avoids re-embedding on every call
_EMBEDDING_CACHE_MAX = 100


class _LRUCache:
    """Simple LRU cache for embeddings."""
    def __init__(self, maxsize=_EMBEDDING_CACHE_MAX):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value


_nerve_embedding_cache = _LRUCache()


def _get_nerve_embedding(name: str, description: str) -> list[float] | None:
    """Get or compute embedding for a nerve description. Returns None on failure.

    Lookup order: in-memory LRU → cold memory (SQLite) → compute + persist.
    """
    cache_key = f"{name}:{description[:80]}"
    cached = _nerve_embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    # Check cold memory for persisted embedding
    try:
        from arqitect.brain.config import mem
        cold_emb = mem.cold.get_nerve_embedding(name)
        if cold_emb:
            _nerve_embedding_cache.put(cache_key, cold_emb)
            return cold_emb
    except Exception:
        pass
    # Compute fresh and persist
    try:
        from arqitect.inference.engine import get_engine
        emb = get_engine().embed(description)
        _nerve_embedding_cache.put(cache_key, emb)
        # Persist to cold memory (best-effort)
        try:
            from arqitect.brain.config import mem as _mem
            _mem.cold.set_nerve_embedding(name, emb)
        except Exception:
            pass
        return emb
    except Exception:
        return None


def match_nerves(task: str, nerve_catalog: dict, threshold: float = 1.0) -> list[tuple[str, float]]:
    """Rank all nerves against a task. Returns [(name, score)] descending.

    Uses hybrid scoring: keyword matching + embedding similarity.
    Core senses receive a SENSE_BOOST bonus to ensure they are preferred
    over regular nerves for tasks in their domain.
    Senses with zero raw score (no keyword overlap) are excluded — the boost
    alone is not enough to constitute a real match.
    """
    # First pass: keyword scoring
    keyword_scored = {}
    max_keyword_score = 0.0
    for name, description in nerve_catalog.items():
        s = match_score(task, name, description)
        if name in CORE_SENSES:
            s += SENSE_BOOST
        keyword_scored[name] = s
        if s > max_keyword_score:
            max_keyword_score = s

    # Second pass: embedding similarity (graceful fallback to keyword-only)
    task_emb = None
    try:
        from arqitect.inference.engine import get_engine, cosine_similarity
        task_emb = get_engine().embed(task)
    except Exception:
        pass  # Fall back to keyword-only

    scored = []
    for name, kw_score in keyword_scored.items():
        final_score = kw_score

        if task_emb is not None and max_keyword_score > 0:
            description = nerve_catalog.get(name, "")
            nerve_emb = _get_nerve_embedding(name, description)
            if nerve_emb is not None:
                try:
                    embed_sim = cosine_similarity(task_emb, nerve_emb)
                    # Blend keyword + embedding, normalized to keyword scale
                    final_score = KEYWORD_WEIGHT * kw_score + EMBEDDING_WEIGHT * embed_sim * max_keyword_score
                except Exception:
                    pass  # Keep keyword score on failure

        if final_score >= threshold:
            scored.append((name, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def best_match_tool(query: str, tools_dict: dict, threshold: float = 1.0) -> str | None:
    """Return the best matching tool name, or None if nothing meets threshold.

    Also applies a normalized check: the best score must be >= BEST_MATCH_THRESHOLD
    of the query's meaningful token count to avoid weak spurious matches.
    """
    ranked = match_tools(query, tools_dict, threshold)
    if not ranked:
        return None
    # Normalized check — score must cover at least half the query tokens
    q_token_count = len(_tokenize(query))
    if q_token_count > 0 and ranked[0][1] < q_token_count * BEST_MATCH_THRESHOLD:
        return None
    return ranked[0][0]


def best_match_nerve(task: str, nerve_catalog: dict, threshold: float = 1.0) -> str | None:
    """Return the best matching nerve name, or None if nothing meets threshold."""
    ranked = match_nerves(task, nerve_catalog, threshold)
    return ranked[0][0] if ranked else None




def find_duplicate_nerves(catalog: dict[str, str], threshold: float = 3.0) -> list[tuple[str, str, float]]:
    """Find pairs of nerves whose descriptions are too similar.

    Cross-scores all nerve descriptions against each other.
    Returns [(nerve_a, nerve_b, score)] for pairs exceeding the threshold.
    """
    names = list(catalog.keys())
    duplicates = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            # Score description of A against name+description of B
            score_ab = match_score(catalog[a], b, catalog[b])
            score_ba = match_score(catalog[b], a, catalog[a])
            score = max(score_ab, score_ba)
            if score >= threshold:
                duplicates.append((a, b, score))
    duplicates.sort(key=lambda x: x[2], reverse=True)
    return duplicates


