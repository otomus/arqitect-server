"""Sense: Communication — tone, personality, multi-modal expression, translation, summarization.

Input: {"message": "...", "tone": "formal|casual|enthusiastic|empathetic|neutral", "format": "text|card|emoji|gif|translate|summarize"}
Wraps messages in tone-appropriate LLM prompts.
Format modes: plain text, markdown card, emoji-enhanced, GIF-accompanied.
Special modes: translate (language translation), summarize (condense long text).
GIF mode uses Tenor free API to find a relevant reaction GIF.
"""

import json
import os
import sqlite3
import sys

_SENSE_DIR = os.path.dirname(os.path.abspath(__file__))
from arqitect.config.loader import get_project_root, get_sandbox_dir as _get_sandbox_dir
_PROJECT_ROOT = str(get_project_root())
_PERSONALITY_PATH = os.path.join(_PROJECT_ROOT, "personality.json")
_COLD_DB_PATH = os.path.join(str(get_project_root()), "memory", "knowledge.db")

SENSE_NAME = "communication"
COMM_MODEL = "communication"
def _load_adapter_description() -> str:
    try:
        from arqitect.brain.adapters import get_description
        desc = get_description("communication")
        if desc:
            return desc
    except Exception:
        pass
    return "Personality-driven voice rewriting — rewrites messages to match personality tone."

DESCRIPTION = _load_adapter_description()

# Personality configuration
PERSONALITY = {
    "name": "Arqitect",
    "default_tone": "neutral",
    "emoji_preference": "moderate",
}


def _load_personality_traits() -> dict:
    """Load personality traits from cold memory, falling back to personality.json seed.

    Returns a dict with keys: core_identity, voice, trait_weights.
    """
    seed = {}
    try:
        with open(_PERSONALITY_PATH, "r") as f:
            seed = json.load(f)
    except Exception:
        seed = {
            "core_identity": {"name": "Arqitect", "archetype": "Sharp, warm, and resourceful AI with quiet confidence"},
            "voice": {"default_tone": "warm-direct", "humor_style": "dry wit when natural, never forced"},
            "trait_weights": {"wit": 0.5, "swagger": 0.3, "warmth": 0.7, "formality": 0.3, "verbosity": 0.3},
        }

    # Overlay evolved traits from cold memory
    if os.path.exists(_COLD_DB_PATH):
        try:
            conn = sqlite3.connect(_COLD_DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT key, value FROM facts WHERE category='personality'"
            ).fetchall()
            conn.close()
            for row in rows:
                key, val = row["key"], row["value"]
                if key == "trait_weights":
                    try:
                        seed["trait_weights"] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif key == "humor_style":
                    seed.setdefault("voice", {})["humor_style"] = val
                elif key == "learned_preferences":
                    try:
                        seed["learned_preferences"] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
        except Exception:
            pass

    return seed


def _build_personality_instruction(traits: dict) -> str:
    """Build a personality instruction string for the LLM tone prompt.

    Intensity is controlled by trait_weights. When result looks like structured
    data, personality stays minimal.
    """
    core = traits.get("core_identity", {})
    voice = traits.get("voice", {})
    # Scaffolder writes voice as a plain string; evolved personality uses a dict
    if isinstance(voice, str):
        voice = {"humor_style": "dry wit", "default_tone": voice}
    weights = traits.get("trait_weights", traits.get("traits", {}))

    archetype = core.get("archetype", "")
    humor = voice.get("humor_style", "sarcastic")
    fixed = core.get("fixed_traits", [])

    wit = weights.get("wit", 0.5)
    swagger = weights.get("swagger", 0.5)
    warmth = weights.get("warmth", 0.4)
    formality = weights.get("formality", 0.2)
    verbosity = weights.get("verbosity", 0.3)

    name = traits.get("name") or core.get("name", "Arqitect")
    lines = [f"You are {name}, a {archetype}."] if archetype else [f"You are {name}."]

    if wit >= 0.6:
        lines.append(f"Humor style: {humor}. Be witty.")
    elif wit >= 0.3:
        lines.append("Light humor is fine but keep it brief.")

    if swagger >= 0.5:
        lines.append("Confident and direct.")
    if warmth >= 0.5:
        lines.append("Be genuinely warm and helpful.")
    if formality <= 0.3:
        lines.append("Keep it casual. Contractions, conversational.")
    elif formality >= 0.7:
        lines.append("Keep a professional register despite the personality.")
    if verbosity <= 0.3:
        lines.append("Be concise. Short sentences.")
    elif verbosity >= 0.7:
        lines.append("You can be detailed and expressive.")

    if fixed:
        lines.append("Core traits: " + "; ".join(fixed[:4]) + ".")

    lines.append(
        "IMPORTANT: If the content is structured data (code, math, lists, file output), "
        "deliver the data cleanly. At most add a one-liner quip before or after — never alter the data itself."
    )

    return " ".join(lines)


def _is_structured_data(message: str) -> bool:
    """Detect if a message is structured data that should not be personality-altered."""
    stripped = message.strip()
    # Code blocks
    if stripped.startswith("```") or stripped.startswith("    "):
        return True
    # JSON
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    # Numeric results
    if stripped.replace(".", "").replace("-", "").replace(",", "").isdigit():
        return True
    # File listings, paths, tables
    lines = stripped.split("\n")
    if len(lines) >= 3:
        # Check if most lines look like paths or table rows
        path_like = sum(1 for l in lines if l.strip().startswith("/") or l.strip().startswith("- ") or "|" in l)
        if path_like >= len(lines) * 0.6:
            return True
    return False

# Tone templates for LLM prompting
_TONE_PROMPTS = {
    "formal": (
        "Rewrite the following message in a formal, professional tone. "
        "Use proper grammar, avoid contractions, and maintain a respectful register."
    ),
    "casual": (
        "Rewrite the following message in a casual, friendly tone. "
        "Use conversational language, contractions are fine, keep it relaxed."
    ),
    "enthusiastic": (
        "Rewrite the following message with enthusiasm and energy. "
        "Show excitement, use dynamic language, be upbeat and positive."
    ),
    "empathetic": (
        "Rewrite the following message with empathy and warmth. "
        "Show understanding, be supportive, acknowledge feelings."
    ),
    "neutral": (
        "Deliver the following message clearly and concisely. "
        "Keep it straightforward and informative."
    ),
}

# Emoji sets by tone
_TONE_EMOJIS = {
    "formal": "",
    "casual": " 👋",
    "enthusiastic": " 🎉✨",
    "empathetic": " 💙",
    "neutral": "",
}


def _load_communication_adapter() -> dict | None:
    """Load the communication adapter prompt from community cache."""
    try:
        from arqitect.brain.adapters import resolve_prompt
        return resolve_prompt("communication")
    except Exception:
        return None


def _try_llm_rewrite(message: str, tone: str) -> str:
    """Use the communication model for tone rewriting with personality injection.

    When the message is structured data (code, math, JSON), personality stays
    minimal — a one-liner quip at most, with the data untouched.
    Falls back to passthrough if no LLM available.
    """
    try:
        from arqitect.inference.router import generate_for_role
        prompt_text = _TONE_PROMPTS.get(tone, _TONE_PROMPTS["neutral"])

        # Load personality traits and build instruction
        traits = _load_personality_traits()
        personality_instruction = _build_personality_instruction(traits)

        # Try community adapter for base system prompt
        adapter = _load_communication_adapter()
        adapter_system = adapter.get("system_prompt", "") if adapter else ""

        structured = _is_structured_data(message)
        if structured:
            system = (
                f"{personality_instruction} "
                "The following message contains structured data. "
                "Deliver the data exactly as-is. You may add a one-liner quip before or after, "
                "but NEVER modify the actual data. Output ONLY the final message."
            )
        elif adapter_system:
            # Use community adapter prompt with personality injection
            system = f"{adapter_system}\n\nPersonality: {personality_instruction}"
        else:
            system = (
                f"{personality_instruction} "
                f"{prompt_text} "
                "Output ONLY the rewritten message, nothing else."
            )

        result = generate_for_role(
            "communication",
            f"Message: {message}",
            system=system,
            max_tokens=256,
        )
        if result.startswith("Error:"):
            return message
        return result.strip()
    except Exception:
        return message


def format_text(message: str, tone: str) -> dict:
    """Format as plain text with optional tone adjustment."""
    rewritten = _try_llm_rewrite(message, tone)
    emoji_suffix = _TONE_EMOJIS.get(tone, "")
    if emoji_suffix and PERSONALITY["emoji_preference"] != "none":
        rewritten = rewritten.rstrip(".!") + emoji_suffix
    return {
        "format": "text",
        "tone": tone,
        "original": message,
        "response": rewritten,
    }


def format_card(message: str, tone: str) -> dict:
    """Format as a markdown card with title/body/footer."""
    rewritten = _try_llm_rewrite(message, tone)
    # Extract or generate a title (first sentence or summary)
    sentences = rewritten.split(". ")
    title = sentences[0].strip(".") if sentences else "Message"
    body = ". ".join(sentences[1:]).strip() if len(sentences) > 1 else rewritten
    card = {
        "format": "card",
        "tone": tone,
        "original": message,
        "card": {
            "title": title,
            "body": body or rewritten,
            "footer": f"— {PERSONALITY['name']}",
        },
        "response": f"**{title}**\n\n{body or rewritten}\n\n---\n*— {PERSONALITY['name']}*",
    }
    return card


def format_emoji(message: str, tone: str) -> dict:
    """Format with emoji enhancement."""
    rewritten = _try_llm_rewrite(message, tone)
    # Try LLM for emoji enhancement, fall back to suffix
    try:
        from arqitect.inference.router import generate_for_role
        enhanced = generate_for_role(
            "communication",
            f"Add relevant emojis to this message to make it more expressive. Keep the text, just add emojis:\n\n{rewritten}",
            system="Output ONLY the emoji-enhanced message. No explanation.",
            max_tokens=256,
        )
        if enhanced.startswith("Error:"):
            enhanced = rewritten + " " + _TONE_EMOJIS.get(tone, "✨")
        else:
            enhanced = enhanced.strip()
    except Exception:
        enhanced = rewritten + " " + _TONE_EMOJIS.get(tone, "✨")

    return {
        "format": "emoji",
        "tone": tone,
        "original": message,
        "response": enhanced,
    }


def _search_gif(query: str) -> str:
    """Search for a GIF using Tenor free API. Returns URL or empty string."""
    try:
        import requests
        resp = requests.get(
            "https://g.tenor.com/v1/search",
            params={"q": query, "key": "LIVDSRZULELA", "limit": 5, "media_filter": "minimal"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            import random
            pick = random.choice(results)
            media = pick.get("media", [{}])[0]
            gif_data = media.get("gif", media.get("tinygif", {}))
            return gif_data.get("url", "")
    except Exception:
        pass
    return ""


def _extract_gif_query(message: str, tone: str) -> str:
    """Use LLM to extract a short GIF search query from the message, or fall back to keywords."""
    try:
        from arqitect.inference.router import generate_for_role
        query = generate_for_role(
            "communication",
            f"Extract 1-3 keywords for a GIF search that matches the mood/topic of this message:\n\n\"{message}\"\n\nTone: {tone}",
            system="Output ONLY the search keywords, nothing else. Example: 'happy dance' or 'thumbs up' or 'mind blown'",
            max_tokens=32,
        ).strip().strip('"\'')
        if query and not query.startswith("Error:") and len(query) < 50:
            return query
    except Exception:
        pass
    words = message.split()[:3]
    return " ".join(words) if words else "reaction"


def format_gif(message: str, tone: str) -> dict:
    """Format response with an accompanying GIF."""
    rewritten = _try_llm_rewrite(message, tone)
    gif_query = _extract_gif_query(message, tone)
    gif_url = _search_gif(gif_query)
    return {
        "format": "gif",
        "tone": tone,
        "original": message,
        "response": rewritten,
        "gif_url": gif_url,
        "gif_query": gif_query,
    }


def format_translate(message: str, tone: str) -> dict:
    """Translate a message to a target language."""
    # tone field is reused to carry target language (e.g. "spanish", "french")
    target_lang = tone if tone not in _TONE_PROMPTS else "English"
    try:
        from arqitect.inference.router import generate_for_role
        result = generate_for_role(
            "communication",
            f"Translate the following text to {target_lang}:\n\n{message}",
            system=f"You are a translator. Output ONLY the translation, nothing else.",
            max_tokens=512,
        )
        if result.startswith("Error:"):
            return {"format": "translate", "original": message, "response": message,
                    "error": result, "sense": SENSE_NAME}
        return {
            "format": "translate",
            "target_language": target_lang,
            "original": message,
            "response": result.strip(),
        }
    except Exception as e:
        return {"format": "translate", "original": message, "response": message,
                "error": str(e), "sense": SENSE_NAME}


def format_summarize(message: str, tone: str) -> dict:
    """Summarize a long message into a concise version."""
    try:
        from arqitect.inference.router import generate_for_role
        result = generate_for_role(
            "communication",
            f"Summarize the following text concisely in 1-3 sentences:\n\n{message}",
            system="Output ONLY the summary, nothing else.",
            max_tokens=256,
        )
        if result.startswith("Error:"):
            return {"format": "summarize", "original": message, "response": message,
                    "error": result, "sense": SENSE_NAME}
        return {
            "format": "summarize",
            "original": message,
            "response": result.strip(),
        }
    except Exception as e:
        return {"format": "summarize", "original": message, "response": message,
                "error": str(e), "sense": SENSE_NAME}


_FORMATTERS = {
    "text": format_text,
    "card": format_card,
    "emoji": format_emoji,
    "gif": format_gif,
    "translate": format_translate,
    "summarize": format_summarize,
}


def calibrate() -> dict:
    """Probe communication sense capabilities."""
    from arqitect.senses.calibration_protocol import build_result, save_calibration

    # Check model availability via file existence (avoids loading full engine in subprocess)
    has_llm = False
    try:
        from arqitect.inference.config import get_backend_type, get_model_name, get_models_dir
        backend = get_backend_type()
        if backend == "gguf":
            model_file = get_model_name(COMM_MODEL)
            has_llm = os.path.exists(os.path.join(get_models_dir(), model_file))
        elif backend == "ollama":
            has_llm = True  # Assume available if ollama backend configured
        else:
            from arqitect.inference.engine import get_engine
            has_llm = get_engine().is_loaded(COMM_MODEL)
    except Exception:
        has_llm = False
    gemma_status = {"installed": has_llm, "version": "latest" if has_llm else "", "install_hint": ""}

    capabilities = {
        "tone_rewrite": {
            "available": has_llm,
            "provider": COMM_MODEL if has_llm else None,
            "notes": "" if has_llm else "Communication model not available — tone rewriting will use passthrough mode",
        },
        "text_format": {
            "available": True,
            "provider": COMM_MODEL if has_llm else "passthrough",
            "notes": "",
        },
        "card_format": {
            "available": True,
            "provider": COMM_MODEL if has_llm else "passthrough",
            "notes": "",
        },
        "emoji_format": {
            "available": True,
            "provider": COMM_MODEL if has_llm else "passthrough",
            "notes": "",
        },
        "gif_format": {
            "available": True,
            "provider": "tenor_api",
            "notes": "GIF search via Tenor free API — no key required",
        },
        "translate": {
            "available": has_llm,
            "provider": COMM_MODEL if has_llm else None,
            "notes": "" if has_llm else "Requires communication model for translation",
        },
        "summarize": {
            "available": has_llm,
            "provider": COMM_MODEL if has_llm else None,
            "notes": "" if has_llm else "Requires communication model for summarization",
        },
    }

    deps = {
        COMM_MODEL: gemma_status,
    }

    result = build_result(SENSE_NAME, capabilities, deps)
    save_calibration(_SENSE_DIR, result)
    return result


def main():
    raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "{}"
    try:
        input_data = json.loads(raw)
    except json.JSONDecodeError:
        input_data = {"message": raw, "tone": "neutral", "format": "text"}

    # Calibration mode
    if input_data.get("mode") == "calibrate":
        print(json.dumps(calibrate()))
        return

    message = input_data.get("message", "")
    if not message:
        print(json.dumps({"sense": SENSE_NAME, "error": "No message provided"}))
        return

    fmt = input_data.get("format", "text").lower()

    tone = input_data.get("tone", PERSONALITY["default_tone"]).lower()
    # For translate, tone carries the target language
    if fmt == "translate":
        tone = input_data.get("target_language", input_data.get("tone", "English"))
    elif tone not in _TONE_PROMPTS:
        tone = PERSONALITY["default_tone"]

    formatter = _FORMATTERS.get(fmt, format_text)

    result = formatter(message, tone)
    result["sense"] = SENSE_NAME

    # Safety filter — catch inappropriate content or sensitive data in output
    try:
        from arqitect.brain.safety import check_output as _safety_check_output
        media_urls = [result.get("gif_url", "")] if result.get("gif_url") else []
        is_safe, refusal = _safety_check_output(result.get("response", ""), media_urls)
        if not is_safe:
            result["response"] = refusal
            result.pop("gif_url", None)
            result.pop("card", None)
    except Exception:
        pass  # safety module unavailable — allow through

    print(json.dumps(result))


if __name__ == "__main__":
    main()
