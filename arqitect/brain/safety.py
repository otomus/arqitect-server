"""Safety filter — LLM-based input/output screening.

Two entry points:
  check_input(message)  — called before routing in think()
  check_output(response, media_urls) — called before publishing response

Uses the inference engine to classify content safety via LLM.
Returns (is_safe: bool, reason: str). When unsafe, reason contains a
context-appropriate refusal message suitable for returning to the user.
"""

import json
import re

from arqitect.inference.router import generate_for_role


# ---------------------------------------------------------------------------
# LLM classification prompts
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = (
    "You are a content safety classifier. Evaluate whether the given content "
    "is safe or unsafe. Unsafe categories:\n"
    "- sexual: Sexual or sexually explicit content\n"
    "- hate: Hate speech, discrimination, threats of violence, extremist ideology\n"
    "- harmful: Instructions for weapons, drugs, hacking, self-harm, exploitation, "
    "or other dangerous activities\n"
    "- sensitive_data: Exposed passwords, API keys, private keys, tokens, "
    "social security numbers, credit card numbers, or other credentials\n"
    "- nsfw_url: URLs pointing to known adult/NSFW content sites\n\n"
    "Respond ONLY with JSON: {\"safe\": true} or {\"safe\": false, \"category\": \"<category>\"}\n"
    "IMPORTANT: Be very precise and conservative with unsafe classifications.\n"
    "- Vague or ambiguous messages are ALWAYS safe (e.g. 'do the thing', 'fix it', 'make it better')\n"
    "- General requests for help, advice, or information are ALWAYS safe\n"
    "- Discussions ABOUT sensitive topics in educational or news context are safe\n"
    "- Only flag content that EXPLICITLY requests, produces, or contains unsafe material"
)

_CLASSIFY_PROMPT = "Is this content safe?\n\n{text}"

_REFUSAL_SYSTEM = (
    "You are a helpful AI assistant. Generate a short, clear refusal message "
    "(1-2 sentences) for blocked content. Be direct and respectful. "
    "Match the language of the user's message. "
    "Do NOT include the unsafe content itself in the refusal."
)

_REFUSAL_PROMPT = (
    "The user sent a message that was blocked for this reason: {category}. "
    "The user's original message language and tone: \"{snippet}\"\n\n"
    "Write a short, personality-flavored refusal (1-2 sentences). "
    "Reference the reason without being preachy. Match the user's language."
)

# Fallback refusal if LLM call fails
_FALLBACK_REFUSAL = (
    "I can't help with that request. "
    "Feel free to ask me something else."
)

# Category labels for logging
_CATEGORY_LABELS = {
    "sexual": "sexual/explicit content",
    "hate": "hate speech / discrimination",
    "harmful": "harmful / dangerous content",
    "sensitive_data": "sensitive data leakage",
    "nsfw_url": "NSFW media URL",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CLASSIFY_MAX_CHARS = 4000

# Patterns that indicate programming code or markup content
_CODE_PATTERNS = [
    re.compile(r"<\w+[\s>]"),              # HTML tags
    re.compile(r"```"),                     # Fenced code blocks
    re.compile(r"\{[^}]*:\s*[^}]+\}"),     # CSS-like blocks
    re.compile(r"\b(function|def|class|import|const|let|var)\s+\w+"),  # Code constructs
    re.compile(r"=>"),                      # Arrow functions
    re.compile(r"<script|<style|<div|<span", re.IGNORECASE),  # Common HTML tags
]

_CODE_CONTEXT_NOTE = (
    "\n\nIMPORTANT: The content below contains programming code, HTML, CSS, or markup. "
    "Code constructs (tags, functions, variables, selectors) are inherently safe. "
    "Only flag this content if it contains genuinely unsafe natural-language instructions "
    "or real exposed credentials — NOT because of code syntax."
)


def _contains_code_content(text: str) -> bool:
    """Detect whether text contains programming code or markup.

    Returns True when 2+ code signal patterns are found, indicating the
    content is predominantly code rather than natural language.
    """
    hits = sum(1 for pattern in _CODE_PATTERNS if pattern.search(text))
    return hits >= 2


def _classify(text: str) -> dict:
    """Ask the LLM to classify content safety. Returns parsed JSON or safe default."""
    try:
        system = _CLASSIFY_SYSTEM
        if _contains_code_content(text):
            system = _CLASSIFY_SYSTEM + _CODE_CONTEXT_NOTE

        raw = generate_for_role(
            "nerve",
            _CLASSIFY_PROMPT.format(text=text[:_CLASSIFY_MAX_CHARS]),
            system=system,
            max_tokens=50,
            temperature=0.0,
            json_mode=True,
        )
        if raw.startswith("Error:"):
            return {"safe": True}
        result = json.loads(raw)
        if isinstance(result.get("safe"), bool):
            return result
        return {"safe": True}
    except Exception as e:
        print(f"[SAFETY] Classification LLM call failed: {e} — defaulting to safe")
        return {"safe": True}


def _generate_refusal(category: str, user_text: str) -> str:
    """Generate a context-appropriate refusal message via LLM."""
    try:
        # Use a short snippet to give the LLM language/tone context without repeating unsafe content
        snippet = user_text[:120].replace('"', "'")
        category_label = _CATEGORY_LABELS.get(category, category)
        raw = generate_for_role(
            "nerve",
            _REFUSAL_PROMPT.format(category=category_label, snippet=snippet),
            system=_REFUSAL_SYSTEM,
            max_tokens=80,
            temperature=0.7,
        )
        if raw.startswith("Error:") or not raw.strip():
            return _FALLBACK_REFUSAL
        return raw.strip()
    except Exception:
        return _FALLBACK_REFUSAL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_input(message: str) -> tuple[bool, str]:
    """Screen an incoming user message before routing.

    Returns:
        (True, "")   — message is safe to process
        (False, msg) — message blocked; msg is the refusal to return
    """
    if not message or not message.strip():
        return True, ""

    text = message.strip()
    result = _classify(text)

    if result.get("safe", True):
        return True, ""

    category = result.get("category", "unknown")
    print(f"[SAFETY] Input blocked — category: {category}")
    refusal = _generate_refusal(category, text)
    return False, refusal


def check_output(response: str, media_urls: list = None) -> tuple[bool, str]:
    """Screen an outgoing response before delivering to the user.

    Checks for:
    - Inappropriate generated content
    - Leaked sensitive data (API keys, passwords, PII)
    - NSFW media URLs

    Returns:
        (True, "")   — response is safe to send
        (False, msg) — response blocked; msg is the refusal to return
    """
    if not response:
        return True, ""

    text = response.strip()

    # Check the response text
    result = _classify(text)
    if not result.get("safe", True):
        category = result.get("category", "unknown")
        print(f"[SAFETY] Output blocked — category: {category}")
        refusal = _generate_refusal(category, text)
        return False, refusal

    # Check media URLs if present
    if media_urls:
        urls_text = " ".join(url for url in media_urls if url)
        if urls_text.strip():
            url_result = _classify(urls_text)
            if not url_result.get("safe", True):
                category = url_result.get("category", "unknown")
                print(f"[SAFETY] Output blocked — NSFW media URL detected")
                refusal = _generate_refusal(category, urls_text)
                return False, refusal

    return True, ""
