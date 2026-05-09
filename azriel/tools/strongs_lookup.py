"""strongs_lookup tool.

Looks up Strong's Hebrew lexicon entries by Strong's number (e.g. "H1",
"H7225"). Returns the original word, transliteration, pronunciation, and
the KJV usage / etymology summary. Data lives at
~/.azriel/data/concordances/strongs_hebrew.jsonl on the host machine -- ingested
by scripts/48_ingest_concordances.py from openscriptures HebrewLexicon
(MIT-licensed).

Reference forms supported:
  "H1" -> entry for Strong's H1 (ʼâb, "father")
  "h7225" -> case-insensitive
  "1" -> bare number assumes Hebrew

Greek (G####) is queued for ingest in γ.4.c followups.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path.home() / ".azriel" / "data" / "concordances" / "strongs_hebrew.jsonl"


@lru_cache(maxsize=1)
def _load_index():
    if not DATA.exists():
        return {}
    index = {}
    with DATA.open() as f:
        for line in f:
            r = json.loads(line)
            # the prompt is "What does Strong's Hebrew H<N> ... mean?"
            m = re.search(r"H(\d+)", r["prompt"])
            if m:
                index[f"H{int(m.group(1))}"] = r["response"]
    return index


def strongs_lookup(ref: str) -> str:
    if not isinstance(ref, str):
        return "ERROR: strongs_lookup expects a string Strong's number."
    s = ref.strip().upper()
    # Allow bare digits to default to Hebrew
    if re.fullmatch(r"\d+", s):
        s = f"H{int(s)}"
    elif re.fullmatch(r"H\d+", s):
        s = f"H{int(s[1:])}"
    elif s.startswith("G"):
        return ("ERROR: Strong's Greek lexicon not yet ingested (queued as "
                "γ.4.c followup). Hebrew (H####) entries available.")
    else:
        return f"ERROR: malformed Strong's number '{ref}'. Use H#### (Hebrew)."

    index = _load_index()
    if not index:
        return f"ERROR: strongs data not loaded; expected {DATA}"
    text = index.get(s)
    if text is None:
        return f"ERROR: {s} not found in Strong's Hebrew (range H1-H8674)."
    return text


if __name__ == "__main__":
    import sys
    print(strongs_lookup(sys.argv[1] if len(sys.argv) > 1 else "H1"))
