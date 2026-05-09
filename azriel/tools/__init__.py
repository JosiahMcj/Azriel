"""Azriel tool registry.

Tools are pure Python functions (str -> str) plus a tiny metadata block.
The runtime advertises tools to the model via the system prompt and routes
calls to the implementations here.
"""
from .bible_lookup import bible_lookup
from .crossref_lookup import crossref_lookup
from .memory_search import memory_search
from .strongs_lookup import strongs_lookup
from .web_search import web_search
from .web_fetch import web_fetch
from .conversation_search import conversation_search
from .memory_insert import memory_insert
from .image_search import image_search
from .pdf_extract import pdf_extract
from .filesystem import fs_list, fs_read, fs_write
from .weather import weather
from .document_create import document_create
from .visualize import visualize
from .commentary_lookup import commentary_lookup
from .doctrinal_check import doctrinal_check
from .image_describe import image_describe
from .pdf_create import pdf_create
from .propose_skill import propose_skill

# NOTE: github_query and cloudflare_query are NOT in the base REGISTRY.
# They live in azriel/connectors.py and only join the active registry
# when the user has connected them via /connectors. This keeps personal
# API tokens out of any public release: a fresh fork shows zero
# connector tools to the model until someone plugs one in.

REGISTRY = {
    "bible_lookup": {
        "fn": bible_lookup,
        "signature": "bible_lookup(ref: str) -> str",
        "doc": (
            "Returns the verse text for a reference like \"John 3:16\" or "
            "\"Romans 8:28-30\". Defaults to BSB; pass \"ref|kjv\" or "
            "\"ref|web\" for alternate translations."
        ),
    },
    "crossref_lookup": {
        "fn": crossref_lookup,
        "signature": "crossref_lookup(ref: str) -> str",
        "doc": (
            "Returns the top 5 cross-references for a verse like \"John 3:16\" "
            "with vote scores from the openbible.info CC-BY dataset. Pass "
            "\"ref|N\" to override the limit (1-50)."
        ),
    },
    "memory_search": {
        "fn": memory_search,
        "signature": "memory_search(query: str) -> str",
        "doc": (
            "Searches Azriel's persistent memory (notes, prior conversations) "
            "via SQLite FTS5 full-text search. Returns top 3 matches by "
            "relevance, one per line. Pass \"query|N\" to override limit (1-20)."
        ),
    },
    "strongs_lookup": {
        "fn": strongs_lookup,
        "signature": "strongs_lookup(ref: str) -> str",
        "doc": (
            "Looks up Strong's Hebrew lexicon by Strong's number, e.g. \"H1\". "
            "Returns the original word, transliteration, pronunciation, and "
            "KJV usage. Hebrew (H####) only for now; Greek (G####) queued."
        ),
    },
    "web_search": {
        "fn": web_search,
        "signature": "web_search(query: str) -> str",
        "doc": (
            "Searches the public web via DuckDuckGo. Returns top 5 results "
            "with title, URL, and snippet. Pass \"query|N\" for top N (1-10). "
            "Use for current events, fact-checking, finding specific articles."
        ),
    },
    "web_fetch": {
        "fn": web_fetch,
        "signature": "web_fetch(url: str) -> str",
        "doc": (
            "Fetches an http(s) URL and returns its readable text content "
            "(HTML stripped). Truncates at 6000 chars. Use after web_search "
            "to read a specific result."
        ),
    },
    "conversation_search": {
        "fn": conversation_search,
        "signature": "conversation_search(query: str) -> str",
        "doc": (
            "Searches the user's prior conversations with Azriel. Returns "
            "matching message snippets. Use when the user references past "
            "discussion (\"what did I tell you about my mother last week?\")."
        ),
    },
    "memory_insert": {
        "fn": memory_insert,
        "signature": "memory_insert(text: str) -> str",
        "doc": (
            "Save a USER-CONTEXT fact to persistent memory (preferences, "
            "personal context, things they've told you to remember). "
            "Max 500 chars. Do NOT use this for general doctrinal facts, "
            "system architecture, release history, or system internals -- "
            "those belong in your training, the constitution, or the repo "
            "docs, not in user memory. Never insert credentials or "
            "secrets. Examples that BELONG: 'user prefers BSB translation', "
            "'user has a presentation Tuesday'. Examples that DO NOT: "
            "'Pentecost is celebrated 50 days after Easter' (general fact)."
        ),
    },
    "image_search": {
        "fn": image_search,
        "signature": "image_search(query: str) -> str",
        "doc": "Top web image results via DuckDuckGo. 'q|N' for top N (1-12).",
    },
    "pdf_extract": {
        "fn": pdf_extract,
        "signature": "pdf_extract(path_or_url: str) -> str",
        "doc": "Extract text from a PDF (sandboxed local path or http(s) URL). 'arg|3-7' for pages.",
    },
    "fs_list": {
        "fn": fs_list,
        "signature": "fs_list(dir: str) -> str",
        "doc": "List a directory inside the ~/azriel-files sandbox. Use '.' for root.",
    },
    "fs_read": {
        "fn": fs_read,
        "signature": "fs_read(path: str) -> str",
        "doc": "Read a file inside the ~/azriel-files sandbox.",
    },
    "fs_write": {
        "fn": fs_write,
        "signature": "fs_write(arg: str) -> str",
        "doc": "Write a file inside the sandbox. Format: 'relative/path|file contents'.",
    },
    "weather": {
        "fn": weather,
        "signature": "weather(location: str) -> str",
        "doc": "Current conditions + 3-day forecast for a city, via open-meteo (no key).",
    },
    "document_create": {
        "fn": document_create,
        "signature": "document_create(arg: str) -> str",
        "doc": "Generate docx/pptx/xlsx in the sandbox. Format: 'docx|name|content' (paragraphs separated by blank lines for docx; CSV rows for xlsx; '---' between slides for pptx).",
    },
    "visualize": {
        "fn": visualize,
        "signature": "visualize(html_or_svg: str) -> str",
        "doc": "Render an inline widget (sanitized SVG/HTML) in the chat. Use for charts, simple diagrams, formatted tables.",
    },
    "commentary_lookup": {
        "fn": commentary_lookup,
        "signature": "commentary_lookup(query: str) -> str",
        "doc": (
            "FTS5 search over indexed public-domain commentary chunks "
            "(~/.azriel/data/docs). Pass a passage reference, topic, or "
            "phrase. Returns top 3 matches with source label and excerpt. "
            "Pass \"query|N\" (1-5) for a different limit. The index "
            "builds lazily on first call (~15s one-time)."
        ),
    },
    "doctrinal_check": {
        "fn": doctrinal_check,
        "signature": "doctrinal_check(claim: str) -> str",
        "doc": (
            "Heuristic term-hit classifier across 10 doctrinal axes. "
            "Pass a claim or short passage; returns per-axis matches and "
            "an aggregate verdict (position-a-leaning / position-b-leaning "
            "/ mixed / unclear) where the two position lists are neutral "
            "labels. Fast self-check, not authoritative judgment -- "
            "term-hit agrees with teacher rubric ~75% of the time."
        ),
    },
    "image_describe": {
        "fn": image_describe,
        "signature": "image_describe(arg: str) -> str",
        "doc": (
            "Describe an uploaded image. Pass a sandbox-relative path "
            "like \"uploads/photo.jpg\" (or \"uploads/photo.jpg|custom "
            "prompt\" to override the default describe prompt). Routes "
            "the image to vision API via the the vision provider "
            "API key at ~/.azriel-secrets/vision_api.json. Returns the "
            "model's textual description, or an ERROR string if no "
            "key, file missing, or unsupported format. Allowed: jpg, "
            "jpeg, png, gif, webp; max 5MB."
        ),
    },
    "pdf_create": {
        "fn": pdf_create,
        "signature": "pdf_create(arg: str) -> str",
        "doc": (
            "Generate a PDF in the sandbox. Format: \"name|content\". "
            "Name is the basename (.pdf appended automatically). "
            "Content is free text; paragraphs separated by blank lines. "
            "First paragraph rendered as title, rest as body. Auto-"
            "paginates if content overflows. Pure-Python (no external "
            "deps), so no PDF tooling required on the host. Returns "
            "the file path + size."
        ),
    },
    "propose_skill": {
        "fn": propose_skill,
        "signature": "propose_skill(arg: str) -> str",
        "doc": (
            "Offer to save the current workflow as a reusable skill. "
            "Fire ONLY after a multi-turn process has produced "
            "something reusable (sermon outline, prayer guide, study "
            "plan, journaling routine, file-and-archive workflow, "
            "etc.). Format: \"name|kickoff_prompt\" or "
            "\"name|kickoff_prompt|style\" or "
            "\"name|kickoff_prompt|style|persona_mix_json\". The "
            "kickoff is what a future user sees pre-filled when they "
            "launch this skill. The runtime intercepts this call and "
            "shows the user a 'Save this as a skill?' card -- you do "
            "NOT need to also write a confirmation in your own text; "
            "just fire the tool and acknowledge briefly. Do NOT fire "
            "for routine Q&A or one-shot answers."
        ),
    },
}


def get_active_registry() -> dict:
    """Base REGISTRY plus any currently-connected connector tools. Called
    fresh on every tool dispatch and primer render, so plug-in / unplug
    via /connectors is effective immediately without a server restart."""
    # Lazy import to avoid a circular reference (connectors.py uses lazy
    # imports back into tools/* for its tool functions).
    from ..connectors import active_tools
    merged = dict(REGISTRY)
    merged.update(active_tools())
    return merged


def call(name: str, arg: str) -> str:
    """Dispatch a tool call. Returns either the tool result or an
    ERROR-prefixed string per the v0.7 protocol."""
    spec = get_active_registry().get(name)
    if spec is None:
        return f"ERROR: unknown or unconnected tool '{name}'"
    try:
        return spec["fn"](arg)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def system_prompt_block() -> str:
    """Render the tools section to inject into the system prompt. Reads
    the *active* registry so disconnected connectors don't leak into the
    model's awareness."""
    lines = ["Available tools (call when needed; do not call gratuitously):"]
    for name, spec in get_active_registry().items():
        lines.append(f"- {spec['signature']}")
        lines.append(f" {spec['doc']}")
    lines.append("")
    lines.append(
        "To call a tool, emit a single line in this exact form on its own line:\n"
        "<tool>NAME(ARG)</tool>\n"
        "Then STOP and wait for the runtime to inject the result. After the "
        "result arrives as a <tool_result> block, integrate it into your answer."
    )
    return "\n".join(lines)
