"""Post-fabrication tool code validator.

Catches hardcoded placeholder credentials and quality issues before
a tool is written to disk. Prevents LLM-generated code from shipping
with ``YOUR_API_KEY``, ``example.com``, or similar dummy values.
"""

from __future__ import annotations

import re

PLACEHOLDER_PATTERNS: list[tuple[str, str]] = [
    (r"YOUR_[A-Z_]+", "hardcoded YOUR_* placeholder"),
    (r"DUMMY_[A-Z_]+", "hardcoded DUMMY_* placeholder"),
    (r"REPLACE_[A-Z_]+", "hardcoded REPLACE_* placeholder"),
    (r"INSERT_[A-Z_]+_HERE", "hardcoded INSERT_*_HERE placeholder"),
    (r"['\"]placeholder['\"]", "literal 'placeholder' string"),
    (r"example\.com", "example.com URL"),
    (r"sk-[a-zA-Z0-9]{20,}", "hardcoded API key (sk-...)"),
]

_COMPILED = [(re.compile(p), desc) for p, desc in PLACEHOLDER_PATTERNS]


def validate_tool_code(code: str) -> tuple[bool, list[str]]:
    """Validate fabricated tool code for placeholder credentials and quality issues.

    Args:
        code: The generated Python source code.

    Returns:
        Tuple of (is_valid, warnings). ``is_valid`` is False if any
        blocking issue is found. ``warnings`` lists human-readable
        descriptions of every detected problem.
    """
    warnings: list[str] = []
    for pattern, description in _COMPILED:
        matches = pattern.findall(code)
        if matches:
            sample = matches[0] if len(matches[0]) < 60 else matches[0][:57] + "..."
            warnings.append(f"{description}: found '{sample}'")
    is_valid = len(warnings) == 0
    return is_valid, warnings


def detect_credential_deps(code: str) -> list[dict]:
    """Scan tool code for ``get_credential()`` calls and extract the schema.

    Args:
        code: Python source code to scan.

    Returns:
        List of dicts with ``service`` and ``keys`` fields describing
        which credentials the tool requires at runtime.
    """
    pattern = re.compile(r'get_credential\(\s*["\'](\w+)["\']\s*,\s*["\'](\w+)["\']\s*\)')
    deps: dict[str, set[str]] = {}
    for service, key in pattern.findall(code):
        deps.setdefault(service, set()).add(key)
    return [{"service": svc, "keys": sorted(keys)} for svc, keys in sorted(deps.items())]
