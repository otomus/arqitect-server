"""Brain catalog — nerve and MCP tool discovery."""

import logging
import os
import time

import requests

from arqitect.config.loader import get_config, get_mcp_url
from arqitect.brain.config import NERVES_DIR, SENSES_DIR, SENSE_DESCRIPTIONS, mem
from arqitect.types import Sense

logger = logging.getLogger(__name__)

SENSES_SUBDIR = "senses"

# MCP tool discovery settings
MCP_MAX_RETRIES = 3
MCP_REQUEST_TIMEOUT_S = 10
MCP_RETRY_DELAY_S = 1


def list_nerves() -> list[str]:
    """Return names of available nerves (each is a directory containing nerve.py)."""
    os.makedirs(NERVES_DIR, exist_ok=True)
    return [d for d in os.listdir(NERVES_DIR)
            if os.path.isfile(os.path.join(NERVES_DIR, d, "nerve.py"))]


def _scan_filesystem_nerves(nerves: dict[str, str]) -> None:
    """Scan NERVES_DIR for nerve directories and enrich descriptions from manifest.

    Each directory containing a nerve.py is added to the result. Descriptions
    are sourced from the community manifest cache when available, falling back
    to any existing cold-memory description, then to the directory name.

    Args:
        nerves: Mutable dict to populate with {name: description}.
    """
    manifest_descs = _load_manifest_descriptions()

    for entry in os.listdir(NERVES_DIR):
        if entry.startswith(".") or entry == SENSES_SUBDIR:
            continue
        nerve_py = os.path.join(NERVES_DIR, entry, "nerve.py")
        if not os.path.isfile(nerve_py):
            continue
        if entry not in nerves:
            desc = manifest_descs.get(entry) or entry
            mem.cold.register_nerve(entry, desc)
            nerves[entry] = desc


def _load_manifest_descriptions() -> dict[str, str]:
    """Load nerve descriptions from the community manifest cache.

    Returns:
        Mapping of nerve name to description string.
    """
    try:
        from arqitect.brain.community import _load_cached_manifest
        manifest = _load_cached_manifest()
        if manifest:
            return {
                name: info.get("description", name)
                for name, info in manifest.get("nerves", {}).items()
            }
    except ImportError:
        pass
    return {}


def _ensure_core_senses(nerves: dict[str, str]) -> None:
    """Guarantee enabled core senses are registered and present in the result.

    Senses disabled via ``senses.<name>.enabled: false`` in arqitect.yaml
    are excluded from the catalog.

    Args:
        nerves: Mutable dict to populate with {name: description}.
    """
    if os.path.isdir(SENSES_DIR):
        for entry in os.listdir(SENSES_DIR):
            if not get_config(f"senses.{entry}.enabled", True):
                continue
            nerve_py = os.path.join(SENSES_DIR, entry, "nerve.py")
            if os.path.isfile(nerve_py) and entry not in nerves:
                desc = SENSE_DESCRIPTIONS.get(entry, f"Core sense: {entry}")
                mem.cold.register_sense(entry, desc)
                nerves[entry] = desc

    for sense in Sense:
        if not get_config(f"senses.{sense}.enabled", True):
            continue
        if sense not in nerves:
            desc = SENSE_DESCRIPTIONS.get(sense, f"Core sense: {sense}")
            mem.cold.register_sense(sense, desc)
            nerves[sense] = desc


def discover_nerves() -> dict[str, str]:
    """Discover nerves from filesystem and community manifest.

    Sources:
      1. Filesystem (NERVES_DIR) — already filtered at seed time
      2. Community manifest — for descriptions
      3. Core senses (SENSES_DIR) — only enabled ones

    No qualification filtering — every nerve on disk is returned.
    Descriptions are enriched from the manifest when available.
    Filesystem-only nerves are auto-registered in cold memory.
    Enabled core senses are always included.
    """
    os.makedirs(NERVES_DIR, exist_ok=True)

    nerves: dict[str, str] = {}
    _scan_filesystem_nerves(nerves)
    _ensure_core_senses(nerves)

    return nerves


def list_mcp_tools() -> list[str]:
    """Query the MCP server for available tools."""
    tools = list_mcp_tools_with_info()
    return list(tools.keys())


def list_mcp_tools_with_info() -> dict[str, dict]:
    """Query the MCP server for available tools with descriptions and params.

    Returns:
        A mapping of tool name to its info dict (description, parameters, etc.).
        An empty dict is returned when the server is reachable but exposes no
        tools.

    Raises:
        ConnectionError: The MCP server could not be reached after all retries.
    """
    last_exc: Exception | None = None
    for attempt in range(MCP_MAX_RETRIES):
        try:
            resp = requests.get(
                f"{get_mcp_url()}/tools", timeout=MCP_REQUEST_TIMEOUT_S
            )
            resp.raise_for_status()
            tools = resp.json().get("tools", {})
            if tools:
                return tools
            logger.info(
                "[MCP-CLIENT] Server returned empty tools (attempt %d/%d)",
                attempt + 1, MCP_MAX_RETRIES,
            )
            return {}
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[MCP-CLIENT] Failed to reach MCP server (attempt %d/%d): %s",
                attempt + 1, MCP_MAX_RETRIES, exc,
            )
        time.sleep(MCP_RETRY_DELAY_S)

    raise ConnectionError(
        f"Could not reach MCP server after {MCP_MAX_RETRIES} attempts"
    ) from last_exc
