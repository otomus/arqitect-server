"""
Known Python libraries the brain can use when fabricating MCP tools.
The coder model references this during tool generation to pick the right library.
Libraries are grouped by capability domain.
"""

KNOWN_LIBRARIES = {
    # Web & Data Fetching
    "httpx": "Modern async HTTP client with HTTP/2 — fetch URLs, call REST APIs, download files",
    "beautifulsoup4": "HTML/XML parsing and data extraction — scrape content, extract links/tables",
    "feedparser": "Parse RSS and Atom feeds — monitor news, aggregate blog updates",
    "trafilatura": "Extract main article text from web pages, strips boilerplate",
    "duckduckgo-search": "Search DuckDuckGo programmatically, no API key needed",
    "newspaper3k": "Article extraction from news sites — title, authors, text, keywords",
    "waybackpy": "Access Wayback Machine / Internet Archive for archived web pages",

    # Text Processing & NLP
    "textblob": "Simple NLP: sentiment analysis, noun phrases, spelling correction, translation",
    "rapidfuzz": "Fast fuzzy string matching and similarity scoring",
    "markdown": "Convert Markdown to HTML",
    "chardet": "Detect character encoding of text/files",
    "langdetect": "Detect language of a text string",
    "sumy": "Automatic text summarization using LSA, LexRank, Luhn algorithms",
    "ftfy": "Fix garbled/mojibake Unicode text",

    # Data & File Formats
    "openpyxl": "Read/write Excel .xlsx files — generate spreadsheets, parse uploads",
    "pdfplumber": "Extract text and tables from PDF files",
    "python-docx": "Create and modify Word .docx documents",
    "pyyaml": "Parse and emit YAML — config files, data serialization",
    "tabulate": "Pretty-print tabular data as ASCII, markdown, or HTML tables",
    "csvkit": "Advanced CSV processing — analyze, convert, query CSV files",
    "xmltodict": "Convert XML to Python dicts and back",

    # Math & Science
    "sympy": "Symbolic math — solve equations, simplify expressions, calculus, algebra",
    "pint": "Physical unit conversion and arithmetic — 1300+ units, composable",
    "uncertainties": "Error propagation in scientific calculations",

    # Media
    "pillow": "Image creation, manipulation, format conversion, thumbnails, watermarks",
    "qrcode": "Generate QR codes from text, URLs, or contact cards",
    "python-barcode": "Generate various barcode formats",
    "cairosvg": "Convert SVG to PNG/PDF",
    "colorthief": "Extract dominant colors from images",

    # System & DevOps
    "psutil": "System and process monitoring — CPU, RAM, disk, network usage",
    "watchdog": "Monitor filesystem for changes, trigger actions on file events",
    "paramiko": "SSH2 protocol — remote command execution, SFTP file transfer",
    "schedule": "Simple in-process job scheduling — periodic tasks",
    "sh": "Subprocess replacement — call shell commands as Python functions",

    # Security & Crypto
    "cryptography": "Encryption, decryption, signing, key generation, password hashing",
    "python-jose": "JSON Web Token (JWT) creation and validation",

    # Communication
    "apprise": "Push notifications to 90+ services — Slack, Telegram, Discord, email",
    "icalendar": "Parse and create iCalendar (.ics) files for calendar events",
    "yagmail": "Simple Gmail sending with OAuth or app password",

    # Date/Time
    "python-dateutil": "Powerful date parsing and relative deltas — 'next Tuesday', '3 weeks ago'",
    "arrow": "Human-friendly datetime manipulation and timezone handling",
    "croniter": "Parse and iterate cron expressions — compute next/previous fire times",

    # Productivity
    "tinydb": "Lightweight document-oriented JSON database — persistent key-value store",
    "python-slugify": "Generate URL-safe slugs from strings — clean filenames, IDs",
    "jinja2": "Template engine for generating formatted text, HTML, reports",

    # Code & Development
    "black": "Opinionated Python code formatter",
    "astroid": "Advanced Python AST analysis — extract functions, classes, imports",
    "gitpython": "Programmatic Git operations — commit, diff, log, branch",
}


def suggest_libraries(task_description: str, max_results: int = 5) -> list[tuple[str, str]]:
    """Given a task description, suggest relevant libraries.
    Returns [(library_name, description)] sorted by relevance."""
    from arqitect.matching import _tokenize, match_score
    task_tokens = _tokenize(task_description)
    if not task_tokens:
        return []

    scored = []
    for lib, desc in KNOWN_LIBRARIES.items():
        s = match_score(task_description, lib.replace("-", " "), desc)
        if s > 0:
            scored.append((lib, desc, s))

    scored.sort(key=lambda x: x[2], reverse=True)
    return [(name, desc) for name, desc, _ in scored[:max_results]]
