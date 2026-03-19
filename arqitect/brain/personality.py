"""Personality Evolution — observe, reflect, evolve during dream state.

The system's personality starts from an admin-seeded baseline and evolves
based on interaction signals collected during live conversations. Evolution
happens exclusively during dream state to prevent mid-conversation instability.

Architecture:
  signals -> observation -> evolution -> anchor check -> apply

  - Signals: append-only during live conversation, flushed after dream
  - Observation: LLM analyzes signals, scores trait effectiveness
  - Evolution: LLM proposes changes within anchor bounds
  - History: full audit trail with rollback capability
"""

import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MIN_SIGNALS_FOR_OBSERVATION = 10
"""Minimum interaction signals needed before running observation."""

MIN_EVOLUTION_CONFIDENCE = 0.6
"""LLM must be at least this confident to apply changes."""

MAX_DRIFT_PER_CYCLE = 0.1
"""Maximum a single trait can change per dream cycle."""

TRAIT_MIN = 0.1
"""Absolute minimum for any trait weight."""

TRAIT_MAX = 0.9
"""Absolute maximum for any trait weight."""

MAX_SIGNALS_IN_PROMPT = 50
"""Cap signals sent to LLM to keep prompt manageable."""

DEFAULT_SEED_WEIGHTS = {
    "wit": 0.5, "swagger": 0.3, "warmth": 0.7,
    "formality": 0.3, "verbosity": 0.3, "pop_culture_refs": 0.1,
}
"""Fallback trait weights when no seed is available."""


# ── Prompts ──────────────────────────────────────────────────────────────────

OBSERVATION_PROMPT = """\
You are analyzing interaction signals to evaluate how a personality's \
communication style is landing with users.

Current personality trait weights: {traits}

Interaction signals since last observation ({signal_count} interactions):
{signals}

Analyze these signals and produce a JSON report with:
- trait_scores: dict mapping each trait name to an effectiveness score (0.0-1.0)
- insights: list of 2-4 concise observations about what is working or not
- explicit_feedback_summary: summary of any direct user feedback about personality
- recommendation: one sentence describing suggested direction

Rules:
- Score traits based on evidence in the signals, not assumptions
- Weight explicit user feedback (explicit_feedback field) higher than implicit signals
- If a trait has no relevant signals, score it 0.5 (neutral)
- Output ONLY valid JSON, no explanation
"""

EVOLUTION_PROMPT = """\
You are evolving a personality's communication style based on observation data.

Current trait weights: {traits}
Observation report: {observation}
Admin-defined anchors (MUST NOT violate): {anchors}
Recent evolution history (last {history_count} changes): {history}

Propose trait weight adjustments. Output a JSON object with:
- changes: list of {{"trait": "trait_name", "old": current_value, \
"new": proposed_value, "reason": "brief reason"}}
- unchanged: list of trait names that should stay the same
- confidence: float 0.0-1.0 indicating how confident you are in these changes

Rules:
- Never change a weight by more than {max_drift} from current value
- Keep all weights between {trait_min} and {trait_max}
- Only propose changes with evidence from the observation report
- If the observation shows traits are working well, propose no changes
- Check the "never" anchor list - never propose values matching those descriptors
- Check the "bounds" anchor - stay within min/max ranges
- Consider recent history to avoid oscillation (flip-flopping)
- Output ONLY valid JSON, no explanation
"""


# ── Signal Collection ────────────────────────────────────────────────────────

def record_signal(cold, signal: dict) -> None:
    """Append an interaction signal for later dream-state analysis.

    Called after each conversation turn during live interaction.
    Signals are append-only and never read or acted on during live conversation.

    Args:
        cold: ColdMemory instance.
        signal: Signal dict with keys like user_tone_shift, topic_domain, etc.
    """
    cold.append_personality_signal(signal)


# ── Observation (Dream Phase) ────────────────────────────────────────────────

def observe_personality(
    cold, generate_fn: Callable, seed: dict,
) -> dict | None:
    """Analyze accumulated signals and produce a personality fit report.

    Dream-state phase that runs before evolution. Requires a minimum number
    of signals to avoid overreacting to sparse data.

    Args:
        cold: ColdMemory instance.
        generate_fn: Callable with signature (model, prompt, system=, max_tokens=).
        seed: Personality seed dict (baseline traits).

    Returns:
        Observation report dict, or None if insufficient signals.
    """
    signals = cold.get_personality_signals()
    if len(signals) < MIN_SIGNALS_FOR_OBSERVATION:
        logger.info(
            "[PERSONALITY] Too few signals for observation (%d/%d)",
            len(signals), MIN_SIGNALS_FOR_OBSERVATION,
        )
        return None

    current_traits = load_current_traits(cold, seed)

    prompt = OBSERVATION_PROMPT.format(
        traits=json.dumps(current_traits),
        signal_count=len(signals),
        signals=json.dumps(signals[:MAX_SIGNALS_IN_PROMPT], indent=1),
    )

    result = generate_fn(
        "brain", prompt,
        system="You are a personality analyst. Output only valid JSON.",
        max_tokens=512,
    )

    return _parse_llm_json(result)


# ── Evolution (Dream Phase) ──────────────────────────────────────────────────

def evolve_personality(
    cold, generate_fn: Callable, observation: dict, seed: dict,
) -> list[dict]:
    """Propose, validate, and apply personality trait changes.

    Dream-state phase that runs after observation. Applies anchor validation
    and drift limits to ensure gradual, bounded evolution.

    Args:
        cold: ColdMemory instance.
        generate_fn: Callable with signature (model, prompt, system=, max_tokens=).
        observation: Observation report from observe_personality.
        seed: Personality seed dict.

    Returns:
        List of applied changes (may be empty).
    """
    if not is_evolution_enabled(cold):
        logger.info("[PERSONALITY] Evolution paused by admin")
        return []

    current_traits = load_current_traits(cold, seed)
    anchors = load_anchors(cold)
    history = cold.get_personality_history(limit=5)

    prompt = EVOLUTION_PROMPT.format(
        traits=json.dumps(current_traits),
        observation=json.dumps(observation),
        anchors=json.dumps(anchors),
        history_count=len(history),
        history=json.dumps(history[-5:]) if history else "[]",
        max_drift=MAX_DRIFT_PER_CYCLE,
        trait_min=TRAIT_MIN,
        trait_max=TRAIT_MAX,
    )

    result = generate_fn(
        "brain", prompt,
        system="You are a personality tuner. Output only valid JSON.",
        max_tokens=512,
    )

    proposal = _parse_llm_json(result)
    if not proposal:
        logger.warning("[PERSONALITY] Could not parse evolution proposal")
        return []

    confidence = proposal.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    if confidence < MIN_EVOLUTION_CONFIDENCE:
        logger.info(
            "[PERSONALITY] Evolution confidence too low (%.2f < %.2f)",
            confidence, MIN_EVOLUTION_CONFIDENCE,
        )
        return []

    raw_changes = proposal.get("changes", [])
    if not isinstance(raw_changes, list) or not raw_changes:
        logger.info("[PERSONALITY] No changes proposed")
        return []

    validated = validate_against_anchors(raw_changes, anchors)
    if not validated:
        logger.info("[PERSONALITY] All proposed changes blocked by anchors")
        return []

    return _apply_validated_changes(cold, current_traits, validated, seed, observation, confidence)


def _apply_validated_changes(
    cold, current_traits: dict, validated: list[dict],
    seed: dict, observation: dict, confidence: float,
) -> list[dict]:
    """Clamp, persist, and record validated trait changes.

    Args:
        cold: ColdMemory instance.
        current_traits: Current trait weight dict.
        validated: List of anchor-validated change proposals.
        seed: Personality seed dict.
        observation: Observation report.
        confidence: Evolution confidence score.

    Returns:
        List of actually applied changes.
    """
    seed_weights = seed.get("trait_weights", seed.get("traits", DEFAULT_SEED_WEIGHTS))
    new_traits = dict(current_traits)
    applied_changes = []

    for change in validated:
        trait = change.get("trait", "")
        new_val = change.get("new")
        if not trait or not isinstance(new_val, (int, float)):
            continue
        old_val = current_traits.get(trait, seed_weights.get(trait, 0.5))
        clamped = clamp_trait(new_val, old_val)
        if abs(clamped - old_val) < 0.01:
            continue
        new_traits[trait] = clamped
        applied_changes.append({
            "trait": trait,
            "old": old_val,
            "new": clamped,
            "reason": change.get("reason", ""),
        })

    if not applied_changes:
        return []

    cold.set_fact("personality", "trait_weights", json.dumps(new_traits))
    cold.set_fact("personality", "evolved_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

    cold.append_personality_history({
        "timestamp": time.time(),
        "old_traits": current_traits,
        "new_traits": new_traits,
        "changes": applied_changes,
        "observation_summary": observation.get("recommendation", ""),
        "confidence": confidence,
    })

    cold.flush_personality_signals()

    logger.info(
        "[PERSONALITY] Evolved %d trait(s): %s",
        len(applied_changes),
        ", ".join(f"{c['trait']}: {c['old']:.2f}->{c['new']:.2f}" for c in applied_changes),
    )

    return applied_changes


# ── Anchor Validation ────────────────────────────────────────────────────────

def validate_against_anchors(changes: list[dict], anchors: dict) -> list[dict]:
    """Filter out changes that violate admin-defined anchors.

    Args:
        changes: List of proposed change dicts with trait, old, new, reason.
        anchors: Anchor dict with never, always, bounds keys.

    Returns:
        Filtered list of changes that pass anchor validation.
    """
    never_list = anchors.get("never", [])
    bounds = anchors.get("bounds", {})
    validated = []

    for change in changes:
        trait = change.get("trait", "")
        new_value = change.get("new")

        if _is_blocked_by_never_list(trait, never_list):
            logger.info("[PERSONALITY] Anchor blocked: %s (in 'never' list)", trait)
            continue

        if _is_blocked_by_bounds(trait, new_value, bounds):
            continue

        validated.append(change)

    return validated


def _is_blocked_by_never_list(trait: str, never_list: list) -> bool:
    """Check if a trait name matches any entry in the never list."""
    trait_lower = trait.lower()
    return any(n.lower() in trait_lower for n in never_list)


def _is_blocked_by_bounds(trait: str, new_value: Any, bounds: dict) -> bool:
    """Check if a new trait value violates configured bounds."""
    if trait not in bounds:
        return False

    bound = bounds[trait]
    if "min" in bound and isinstance(new_value, (int, float)) and new_value < bound["min"]:
        logger.info(
            "[PERSONALITY] Anchor blocked: %s=%.2f (below min %.2f)",
            trait, new_value, bound["min"],
        )
        return True
    if "max" in bound and isinstance(new_value, (int, float)) and new_value > bound["max"]:
        logger.info(
            "[PERSONALITY] Anchor blocked: %s=%.2f (above max %.2f)",
            trait, new_value, bound["max"],
        )
        return True
    if "allowed" in bound and new_value not in bound["allowed"]:
        logger.info(
            "[PERSONALITY] Anchor blocked: %s=%s (not in allowed list)",
            trait, new_value,
        )
        return True

    return False


# ── Trait Clamping ───────────────────────────────────────────────────────────

def clamp_trait(value: float, old_value: float) -> float:
    """Clamp a trait value within absolute bounds and max drift per cycle.

    Args:
        value: Proposed new value.
        old_value: Current value.

    Returns:
        Clamped value within [TRAIT_MIN, TRAIT_MAX] and max drift of MAX_DRIFT_PER_CYCLE.
    """
    delta = max(-MAX_DRIFT_PER_CYCLE, min(MAX_DRIFT_PER_CYCLE, value - old_value))
    return round(max(TRAIT_MIN, min(TRAIT_MAX, old_value + delta)), 2)


# ── History & Rollback ───────────────────────────────────────────────────────

def get_history(cold, limit: int = 0) -> list[dict]:
    """Get personality evolution history.

    Args:
        cold: ColdMemory instance.
        limit: Max entries to return (0 = all).

    Returns:
        List of history entries ordered by timestamp ascending.
    """
    return cold.get_personality_history(limit=limit)


def rollback(cold, seed: dict, steps: int = 1) -> dict:
    """Revert personality to a previous state.

    Args:
        cold: ColdMemory instance.
        seed: Personality seed for loading current traits.
        steps: Number of evolution steps to revert.

    Returns:
        Dict with old_traits and new_traits (the reverted-to state).

    Raises:
        ValueError: If not enough history to rollback the requested steps.
    """
    history = cold.get_personality_history()
    if len(history) < steps:
        raise ValueError(
            f"Cannot rollback {steps} step(s) — only {len(history)} entries in history"
        )

    current_traits = load_current_traits(cold, seed)
    target_entry = history[-(steps)]
    target_traits = target_entry.get("old_traits", {})

    cold.set_fact("personality", "trait_weights", json.dumps(target_traits))
    cold.set_fact("personality", "evolved_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

    cold.append_personality_history({
        "timestamp": time.time(),
        "old_traits": current_traits,
        "new_traits": target_traits,
        "changes": [{"type": "rollback", "steps": steps}],
        "observation_summary": f"Manual rollback by admin ({steps} step(s))",
        "confidence": 1.0,
    })

    return {"old_traits": current_traits, "new_traits": target_traits}


# ── Admin Controls ───────────────────────────────────────────────────────────

def set_trait(cold, trait_name: str, value: Any, seed: dict) -> None:
    """Immediately set a personality trait value (admin override).

    Bypasses dream state and takes effect immediately.

    Args:
        cold: ColdMemory instance.
        trait_name: Trait key (e.g. "warmth", "formality").
        value: New value for the trait.
        seed: Personality seed for loading current traits.
    """
    current = load_current_traits(cold, seed)
    old_value = current.get(trait_name)
    current[trait_name] = value
    cold.set_fact("personality", "trait_weights", json.dumps(current))

    cold.append_personality_history({
        "timestamp": time.time(),
        "old_traits": {trait_name: old_value},
        "new_traits": {trait_name: value},
        "changes": [{"trait": trait_name, "old": old_value, "new": value, "reason": "admin override"}],
        "observation_summary": "Admin override",
        "confidence": 1.0,
    })


def add_anchor(cold, anchor_type: str, value: str) -> None:
    """Add a value to an anchor list (never/always).

    Args:
        cold: ColdMemory instance.
        anchor_type: "never" or "always".
        value: Trait descriptor to add.
    """
    anchors = load_anchors(cold)
    anchor_list = anchors.setdefault(anchor_type, [])
    if value not in anchor_list:
        anchor_list.append(value)
    cold.set_fact("personality", "anchors", json.dumps(anchors))


def remove_anchor(cold, anchor_type: str, value: str) -> None:
    """Remove a value from an anchor list.

    Args:
        cold: ColdMemory instance.
        anchor_type: "never" or "always".
        value: Trait descriptor to remove.
    """
    anchors = load_anchors(cold)
    anchor_list = anchors.get(anchor_type, [])
    if value in anchor_list:
        anchor_list.remove(value)
    cold.set_fact("personality", "anchors", json.dumps(anchors))


def set_anchor_bounds(cold, trait: str, bounds: dict) -> None:
    """Set bounds for a trait (min, max, or allowed values).

    Args:
        cold: ColdMemory instance.
        trait: Trait name.
        bounds: Dict with min, max, and/or allowed keys.
    """
    anchors = load_anchors(cold)
    anchors.setdefault("bounds", {})[trait] = bounds
    cold.set_fact("personality", "anchors", json.dumps(anchors))


def pause_evolution(cold) -> None:
    """Disable personality evolution while keeping current traits."""
    cold.set_fact("personality", "evolution_enabled", "false")


def resume_evolution(cold) -> None:
    """Re-enable personality evolution."""
    cold.set_fact("personality", "evolution_enabled", "true")


def is_evolution_enabled(cold) -> bool:
    """Check if personality evolution is enabled (default: True)."""
    val = cold.get_fact("personality", "evolution_enabled")
    return val != "false"


def reset_to_seed(cold, seed: dict) -> None:
    """Reset personality to the initial seed values.

    Args:
        cold: ColdMemory instance.
        seed: Personality seed dict with trait_weights or traits key.
    """
    current = load_current_traits(cold, seed)
    seed_weights = seed.get("trait_weights", seed.get("traits", DEFAULT_SEED_WEIGHTS))

    cold.set_fact("personality", "trait_weights", json.dumps(seed_weights))
    cold.set_fact("personality", "evolved_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

    cold.append_personality_history({
        "timestamp": time.time(),
        "old_traits": current,
        "new_traits": seed_weights,
        "changes": [{"type": "reset", "reason": "Reset to seed by admin"}],
        "observation_summary": "Reset to initial seed",
        "confidence": 1.0,
    })


# ── Internal Helpers ─────────────────────────────────────────────────────────

def load_current_traits(cold, seed: dict) -> dict:
    """Load current trait weights from cold memory, falling back to seed.

    Args:
        cold: ColdMemory instance.
        seed: Personality seed dict.

    Returns:
        Dict of trait name to weight value.
    """
    raw = cold.get_fact("personality", "trait_weights")
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return dict(seed.get("trait_weights", seed.get("traits", DEFAULT_SEED_WEIGHTS)))


def load_anchors(cold) -> dict:
    """Load personality anchors from cold memory.

    Args:
        cold: ColdMemory instance.

    Returns:
        Anchor dict with never, always, bounds keys.
    """
    raw = cold.get_fact("personality", "anchors")
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return {"never": [], "always": [], "bounds": {}}


def _parse_llm_json(text: str) -> dict | None:
    """Parse JSON from LLM output, handling markdown code fences.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict or None if parsing fails.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
