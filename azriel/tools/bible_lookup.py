"""bible_lookup tool.

Reference forms supported:
  "John 3:16" -> single verse
  "Romans 8:28-30" -> verse range
  "Genesis 1" -> whole chapter
  "John 3:16|kjv" -> single verse, KJV translation

Translations: bsb (default), kjv, web. Data lives at
~/.azriel/data/bible/bible_<trans>.jsonl on the host machine.

Known data gap (γ.2.b followup):
  Books with numeric prefixes (1 Samuel, 2 Samuel, 1 Kings, 2 Kings,
  1 Chronicles, 2 Chronicles, 1 Corinthians, 2 Corinthians,
  1 Thessalonians, 2 Thessalonians, 1 Timothy, 2 Timothy, 1 Peter,
  2 Peter, 1 John, 2 John, 3 John, Song of Solomon) are missing from
  the current BSB / KJV / WEB JSONL. The 48 books that ARE present work
  fine; lookups against missing books return the standard "unknown
  book" ERROR. Need to ingest these from another public-domain source.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path.home() / ".azriel" / "data" / "bible"
FILES = {"bsb": "bible_bsb.jsonl", "kjv": "bible_kjv.jsonl", "web": "bible_web.jsonl"}
DEFAULT = "bsb"

REF_RE = re.compile(
    r"^\s*([1-3]?\s*[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d+)(?::(\d+)(?:-(\d+))?)?\s*$"
)


@lru_cache(maxsize=3)
def _load(translation: str):
    """Returns ({(book_canonical, chapter, verse_or_None): text}, {book_lower_normalized: book_canonical})."""
    path = DATA_DIR / FILES[translation]
    if not path.exists():
        return {}, {}
    index = {}
    book_map = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            book = r["book"]
            ch = r["chapter"]
            v = r.get("verse")
            index[(book, ch, v)] = r["response"]
            book_map.setdefault(_book_key(book), book)
    return index, book_map


def _book_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _parse_ref(ref: str):
    m = REF_RE.match(ref)
    if not m:
        return None
    book = re.sub(r"\s+", " ", m.group(1).strip())
    chapter = int(m.group(2))
    v_start = int(m.group(3)) if m.group(3) else None
    v_end = int(m.group(4)) if m.group(4) else v_start
    return book, chapter, v_start, v_end


def bible_lookup(ref: str) -> str:
    if not isinstance(ref, str):
        return "ERROR: bible_lookup expects a string reference."
    # Optional |TRANSLATION suffix
    if "|" in ref:
        ref_part, trans = ref.rsplit("|", 1)
        trans = trans.strip().lower()
    else:
        ref_part, trans = ref, DEFAULT
    if trans not in FILES:
        return f"ERROR: unknown translation '{trans}'. Use bsb, kjv, or web."

    parsed = _parse_ref(ref_part)
    if not parsed:
        return (
            f"ERROR: malformed reference '{ref_part}'. "
            "Expected 'Book Chapter:Verse', 'Book Chapter:Verse-Verse', or 'Book Chapter'."
        )
    book_user, chapter, v_start, v_end = parsed
    index, book_map = _load(trans)
    if not index:
        return f"ERROR: bible data not loaded; expected file at {DATA_DIR / FILES[trans]}"

    book = book_map.get(_book_key(book_user))
    if book is None:
        return f"ERROR: unknown book '{book_user}'."

    if v_start is None:
        text = index.get((book, chapter, None))
        if text is None:
            return f"ERROR: {book} {chapter} not found in {trans.upper()}."
        return text

    verses = []
    for v in range(v_start, v_end + 1):
        t = index.get((book, chapter, v))
        if t is not None:
            verses.append(f"{v} {t}" if v_end != v_start else t)
    if not verses:
        return f"ERROR: {book} {chapter}:{v_start}-{v_end} not found in {trans.upper()}."
    return "\n".join(verses)


if __name__ == "__main__":
    import sys
    print(bible_lookup(sys.argv[1] if len(sys.argv) > 1 else "John 3:16"))
