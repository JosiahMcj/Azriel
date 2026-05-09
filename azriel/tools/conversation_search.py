"""conversation_search tool.

Search past conversation transcripts (assistant + user turns across all
sessions). Reads the FastAPI server's conversations.db SQLite store.

Reference forms:
  "Pentecost" -> top 5 messages mentioning Pentecost
  "John 3:16|10" -> top 10 matches
"""
import re
import sqlite3
from pathlib import Path

DB = Path.home() / ".azriel" / "data" / "conversations.db"
DEFAULT_LIMIT = 5
MAX_LIMIT = 20


def _safe(query: str) -> str:
    return re.sub(r"[%_]", " ", query)


def conversation_search(query: str) -> str:
    if not isinstance(query, str):
        return "ERROR: conversation_search expects a string query."
    if "|" in query:
        q_part, n = query.rsplit("|", 1)
        try:
            limit = max(1, min(MAX_LIMIT, int(n.strip())))
        except ValueError:
            return f"ERROR: limit '{n}' is not a number."
    else:
        q_part, limit = query, DEFAULT_LIMIT
    q = _safe(q_part.strip())
    if not q:
        return "ERROR: empty query."
    if not DB.exists():
        return "(no conversation history yet)"

    c = sqlite3.connect(str(DB))
    try:
        rows = c.execute(
            "SELECT role, text, ts, session_id FROM messages "
            "WHERE text LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{q}%", limit),
        ).fetchall()
    finally:
        c.close()

    if not rows:
        return f"(no conversation matches for '{q_part}')"
    lines = []
    for role, text, ts, sid in rows:
        snippet = text.strip()
        # show ~200 chars centered around the match if possible
        idx = snippet.lower().find(q.lower())
        if idx > 60:
            snippet = "..." + snippet[idx - 60 : idx + 140]
        elif len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(f"- [{role}] {snippet}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(conversation_search(sys.argv[1] if len(sys.argv) > 1 else "Pentecost"))
