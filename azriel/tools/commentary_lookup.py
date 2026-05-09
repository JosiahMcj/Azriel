"""commentary_lookup -- tool.

Search the indexed Missler / Henry / Calvin / public-domain commentary
chunks ingested at ~/.azriel/data/docs/documents.jsonl.

The source file is 249MB / 27k chunks; we don't load it on every call.
Instead we build a SQLite FTS5 index (~/.azriel/data/docs/commentaries.db)
on first use and query it from then on. Index build is ~15s one-time;
queries are sub-100ms thereafter.

Returns up to 3 best-matching chunks with source label + path + excerpt.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DOCS_PATH = Path.home() / ".azriel" / "data" / "docs" / "documents.jsonl"
INDEX_PATH = Path.home() / ".azriel" / "data" / "docs" / "commentaries.db"


def _build_index() -> None:
    """One-time indexer. Call when the FTS db doesn't exist."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    c = sqlite3.connect(str(INDEX_PATH))
    c.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5("
        "source, path, section, body, "
        "tokenize='porter unicode61')"
    )
    if not DOCS_PATH.exists():
        c.commit()
        c.close()
        return
    rows = []
    with DOCS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            body = (rec.get("response") or "").strip()
            if not body:
                continue
            rows.append((
                rec.get("source", "") or "",
                rec.get("path", "") or "",
                str(rec.get("section", "") or ""),
                body,
            ))
            if len(rows) >= 500:
                c.executemany(
                    "INSERT INTO chunks(source, path, section, body) "
                    "VALUES (?,?,?,?)",
                    rows,
                )
                rows.clear()
    if rows:
        c.executemany(
            "INSERT INTO chunks(source, path, section, body) "
            "VALUES (?,?,?,?)",
            rows,
        )
    c.commit()
    c.close()


def _ensure_index() -> None:
    if not INDEX_PATH.exists():
        _build_index()


def _quote_for_fts(query: str) -> str:
    """FTS5 prefers tokens; punctuation in the user query confuses it.
    We strip non-alphanumeric and re-join with OR for any-of semantics
    (more forgiving than implicit AND for short biblical-reference
    style queries)."""
    import re
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", query) if len(t) >= 2]
    if not tokens:
        return ""
    # Quote each token to avoid FTS5 syntax quirks (e.g., "and", "or").
    quoted = [f'"{t}"' for t in tokens]
    return " OR ".join(quoted)


def commentary_lookup(query: str) -> str:
    """Search the commentary corpus. Pass a passage reference, topic,
    keyword, or short phrase. Returns up to 3 best-matching chunks.

    Pass "query|N" to override the limit (1-5).
    """
    if not query or not query.strip():
        return "ERROR: empty query."
    parts = query.rsplit("|", 1)
    q = parts[0].strip()
    n = 3
    if len(parts) == 2 and parts[1].strip().isdigit():
        n = max(1, min(5, int(parts[1].strip())))
    fts_q = _quote_for_fts(q)
    if not fts_q:
        return f"ERROR: no usable tokens in query: {q}"
    try:
        _ensure_index()
    except Exception as e:
        return f"ERROR: failed to build commentary index: {type(e).__name__}: {e}"
    try:
        c = sqlite3.connect(str(INDEX_PATH))
        rows = c.execute(
            "SELECT source, path, section, "
            "snippet(chunks, 3, '[', ']', '...', 24) AS excerpt, "
            "rank "
            "FROM chunks WHERE chunks MATCH ? "
            "ORDER BY rank LIMIT ?",
            (fts_q, n),
        ).fetchall()
        c.close()
    except sqlite3.OperationalError as e:
        return f"ERROR: commentary search failed: {e}"
    if not rows:
        return f"No commentary matches for: {q}"
    out = [f"Top {len(rows)} commentary match(es) for '{q}':"]
    for i, (source, path, section, excerpt, rank) in enumerate(rows, 1):
        label = source or (Path(path).stem if path else "(unknown)")
        if section:
            label = f"{label} §{section}"
        excerpt = " ".join(excerpt.split())
        if len(excerpt) > 480:
            excerpt = excerpt[:480] + "..."
        out.append(f"\n{i}. {label}\n {excerpt}")
    return "\n".join(out)
