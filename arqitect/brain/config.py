"""Brain configuration — constants, Redis connection, and memory manager."""

import logging
import os
import redis
from arqitect.config.loader import (
    get_nerves_dir, get_senses_dir, get_sandbox_dir,
    get_mcp_tools_dir, get_redis_host_port,
)
from arqitect.memory import MemoryManager
from arqitect.brain.types import Sense

BRAIN_MODEL = "brain"
NERVE_MODEL = "nerve"
CODE_MODEL = "coder"
COMMUNICATION_MODEL = "communication"
CREATIVE_MODEL = "creative"
NERVES_DIR = get_nerves_dir()
SENSES_DIR = get_senses_dir()
SANDBOX_DIR = get_sandbox_dir()
MCP_TOOLS_DIR = get_mcp_tools_dir()
DOMAIN_INDEXER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "domain_indexer.py"
)

# The 5 immutable core senses — can never be pruned or deleted
CORE_SENSES = frozenset(Sense)

# Hardcoded fallback descriptions — only used if community adapter meta is missing
_FALLBACK_SENSE_DESCRIPTIONS = {
    Sense.SIGHT: "Image analysis and description — examines photos, screenshots, and visual content.",
    Sense.HEARING: "Audio input/output — speech-to-text, text-to-speech, audio recording and playback.",
    Sense.TOUCH: "File system and OS operations — read, write, list, delete, exec, sysinfo.",
    Sense.AWARENESS: "Sentient's own identity and persona — who Sentient is, what it can do, its personality.",
    Sense.COMMUNICATION: "Personality-driven voice rewriting — rewrites messages to match personality tone. Does NOT translate.",
}

# Map sense names to adapter role names
_SENSE_TO_ROLE = {
    Sense.SIGHT: "vision",
    Sense.HEARING: "hearing",  # no adapter yet — will use fallback
    Sense.TOUCH: "touch",       # no adapter yet — will use fallback
    Sense.AWARENESS: "awareness",
    Sense.COMMUNICATION: "communication",
}


_log = logging.getLogger(__name__)


def _adapter_description(role: str) -> str | None:
    """Try to load a description from the community adapter meta for *role*.

    Returns the description string, or ``None`` if unavailable.
    """
    try:
        from arqitect.brain.adapters import _load_meta
        meta = _load_meta(role, "core")
        if meta and meta.get("description"):
            return meta["description"]
    except Exception:
        _log.debug("failed to load community adapter meta for role=%s", role, exc_info=True)
    return None


def _load_sense_descriptions() -> dict:
    """Load sense descriptions from community adapter meta.json files.

    Falls back to hardcoded descriptions if no adapter is cached.
    """
    descriptions = {}
    for sense in Sense:
        role = _SENSE_TO_ROLE.get(sense)
        desc = _adapter_description(role) if role else None
        descriptions[sense] = desc or _FALLBACK_SENSE_DESCRIPTIONS.get(sense, f"Core sense: {sense}")
    return descriptions


SENSE_DESCRIPTIONS = _load_sense_descriptions()

_host, _port = get_redis_host_port()
r = redis.Redis(host=_host, port=_port, decode_responses=True)
mem = MemoryManager(r)
