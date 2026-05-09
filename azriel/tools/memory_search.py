"""memory_search tool.

Persistent memory backend using SQLite FTS5 for full-text search. The DB
lives at ~/.azriel/data/memory/memory.db on the host machine.

Schema: a single FTS5 virtual table `memory` with columns
  - text (indexed)
  - source (UNINDEXED metadata)
  - ts (UNINDEXED unix timestamp)

The MCP-facing tool is `memory_search(query) -> str` which returns the top
matches by BM25 rank, one per line. Insert / delete / list are admin
primitives (not exposed to the model in v0.7).

Future steps:
  - Wire MemPalace as the canonical backend with this SQLite store as a
    fallback / cache (deferred)
  - Auto-ingest from session transcripts (deferred to γ.7+)
"""
import re
import sqlite3
import time
from pathlib import Path

DB_DIR = Path.home() / ".azriel" / "data" / "memory"
DB_PATH = DB_DIR / "memory.db"
DEFAULT_LIMIT = 3
MAX_LIMIT = 20


def _conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    fresh = not DB_PATH.exists()
    c = sqlite3.connect(str(DB_PATH))
    if fresh:
        c.execute(
            "CREATE VIRTUAL TABLE memory USING fts5("
            "text, source UNINDEXED, ts UNINDEXED, "
            "tokenize='porter unicode61')"
        )
        c.commit()
    return c


def insert(text: str, source: str = "manual", ts: int | None = None) -> int:
    if not text or not text.strip():
        raise ValueError("text must be non-empty")
    if ts is None:
        ts = int(time.time())
    c = _conn()
    cur = c.execute(
        "INSERT INTO memory(text, source, ts) VALUES (?, ?, ?)",
        (text.strip(), source, ts),
    )
    c.commit()
    rowid = cur.lastrowid
    c.close()
    return rowid


def delete(rowid: int) -> bool:
    c = _conn()
    cur = c.execute("DELETE FROM memory WHERE rowid = ?", (rowid,))
    deleted = cur.rowcount > 0
    c.commit()
    c.close()
    return deleted


def list_all(limit: int = 100):
    c = _conn()
    rows = c.execute(
        "SELECT rowid, text, source, ts FROM memory ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    c.close()
    return rows


def _sanitize_query(q: str) -> str:
    """Strip FTS5-special characters that aren't useful for typical queries."""
    return re.sub(r'["\']', "", q.strip())


def memory_search(query: str) -> str:
    if not isinstance(query, str):
        return "ERROR: memory_search expects a string query."
    if "|" in query:
        q_part, limit_str = query.rsplit("|", 1)
        try:
            limit = max(1, min(MAX_LIMIT, int(limit_str.strip())))
        except ValueError:
            return f"ERROR: limit '{limit_str}' is not a number."
    else:
        q_part, limit = query, DEFAULT_LIMIT

    q = _sanitize_query(q_part)
    if not q:
        return "ERROR: empty query."

    if not DB_PATH.exists():
        return "(memory empty)"

    c = _conn()
    try:
        rows = c.execute(
            "SELECT text, source, ts, rank FROM memory "
            "WHERE memory MATCH ? ORDER BY rank LIMIT ?",
            (q, limit),
        ).fetchall()
    except sqlite3.OperationalError as e:
        c.close()
        return f"ERROR: invalid FTS query '{q_part}': {e}"
    c.close()

    if not rows:
        return f"(no matches for '{q_part}')"
    out = []
    for text, source, ts, _rank in rows:
        out.append(f"- [{source}] {text}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(memory_search(sys.argv[1] if len(sys.argv) > 1 else "Pentecost"))
