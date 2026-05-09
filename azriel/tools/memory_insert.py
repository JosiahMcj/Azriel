"""memory_insert tool -- model-callable variant.

Memory is for USER-CONTEXT ONLY: preferences, personal facts the user
told us to remember, prior conversation references, scheduled events.

Memory is NOT for:
  - General doctrinal facts -- the model already knows from training
  - System architecture / release history -- those live in repo docs
  - Credentials / SSH endpoints / API keys -- never anywhere reachable

This module enforces those rules at the tool layer:

  - Soft block: if the text looks like a system / release / doctrine
    fact rather than user context, reject with a hint.
  - Hard block: if the text contains anything secret-shaped (SSH, key
    paths, ports, "Bearer xxx", etc.), reject and log nothing.
"""
import re
from .memory_search import insert as _admin_insert

MAX_TEXT_CHARS = 500

# Hard block: never accept secret-shaped strings into memory.
_SECRET_PATTERNS = [
    r"\bssh\b.*(\bport\b|\bkey\b|@)", # "ssh ... port" / "ssh ... key" / "ssh user@host"
    r"id_(ed25519|rsa|ecdsa|dsa)", # private-key file references
    r"-----BEGIN[A-Z ]*PRIVATE KEY",
    r"\bsk-[A-Za-z0-9]{20,}", # OpenAI-style keys
    r"\bsk-ant-[A-Za-z0-9_\-]{20,}", # vision-API-style keys
    r"\bghp_[A-Za-z0-9]{30,}", # GitHub PATs
    r"\bAKIA[A-Z0-9]{16}\b", # AWS access key IDs
    r"\bbore\.pub\b|161\.35\.\d+\.\d+", # the project's bore relay
    r"\bbearer\s+[A-Za-z0-9_\-\.]+", # bearer tokens
    r"\bapi[_-]?key\s*[:=]", # api_key = ..., api-key: ...
    r"\bpassword\s*[:=]", # password = ...
]
_SECRET_RE = re.compile("|".join(_SECRET_PATTERNS), re.IGNORECASE)


def looks_secret(text: str) -> bool:
    """Public helper for any other code path that wants to validate
    text BEFORE writing it to memory. Used by autoresearch promotion."""
    return bool(_SECRET_RE.search(text or ""))

# Soft block: shapes of "this is system knowledge, not user context".
# Pattern is anchored to start of text so it only fires on declarative
# system-fact-shaped inserts ("wraps Qwen3..."), not on user
# context that happens to mention these words ("user attends...").
_SYSTEM_FACT_PATTERNS = [
    r"^(Azriel|The model)\s+(uses|wraps|tags|is)\s+",
    r"^Phase\s+(α|β|γ|δ|ε|ζ|alpha|beta|gamma|delta)",
    r"^v0\.[0-9]+(\.[0-9]+)?\s+(was|broke|holds|is|tagged)",
    r"^(MLX|MLX-LM|LoRA|LTI|FTS5|MemPalace)\s+",
    r"^(Continuationism|Dispensationalism)\s+",
    r"^Pentecost\s+is\s+celebrated",
    r"^(Acts|Genesis|Romans|John|Hebrews)\s+\d+\s+(records|shows|teaches)",
    r"^The\s+(safety floor|doctrinal benchmark|teacher rubric|training data mix|Treasury)",
    r"^(Bible JSONL|openbible)",
    r"^The Phase",
]
_SYSTEM_RE = re.compile("|".join(_SYSTEM_FACT_PATTERNS), re.IGNORECASE)


def memory_insert(text: str) -> str:
    if not isinstance(text, str):
        return "ERROR: memory_insert expects a string."
    text = text.strip()
    if not text:
        return "ERROR: empty memory text."
    if len(text) > MAX_TEXT_CHARS:
        return f"ERROR: memory text too long ({len(text)} chars; max {MAX_TEXT_CHARS}). Summarize first."

    # Hard block: secret-shaped content is never written.
    if _SECRET_RE.search(text):
        return ("ERROR: refused -- text matches a secret/credential pattern. "
                "Memory is not the right place for SSH endpoints, API keys, "
                "or private-key references. Don't try to save these.")

    # Soft block: system-knowledge-shaped content gets a hint, no insert.
    if _SYSTEM_RE.match(text):
        return ("ERROR: refused -- this looks like system / release / "
                "doctrinal-general knowledge, not user context. Memory is "
                "for USER-specific facts (preferences, personal events, "
                "things they told you to remember). Doctrinal facts live "
                "in your training; system facts live in the repo docs.")

    rid = _admin_insert(text, source="model")
    return f"saved as memory #{rid}"


if __name__ == "__main__":
    import sys
    print(memory_insert(sys.argv[1] if len(sys.argv) > 1 else "test memory entry"))
