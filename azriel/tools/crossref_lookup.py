"""crossref_lookup tool.

Returns cross-references for a Bible verse, drawn from the openbible.info
CC-BY cross-references dataset (a TSK-equivalent with vote-based quality
scoring). Data lives at ~/.azriel/data/crossref/cross_references.tsv on
the host machine -- TSV with columns: From Verse, To Verse, Votes.

Reference forms supported:
  "John 3:16" -> top 5 cross-refs by votes
  "John 3:16|10" -> top 10 cross-refs by votes (limit override)
"""
import re
from functools import lru_cache
from pathlib import Path

DATA = Path.home() / ".azriel" / "data" / "crossref" / "cross_references.tsv"
DEFAULT_LIMIT = 5

OSIS_BOOKS = [
    "Gen", "Exod", "Lev", "Num", "Deut", "Josh", "Judg", "Ruth",
    "1Sam", "2Sam", "1Kgs", "2Kgs", "1Chr", "2Chr",
    "Ezra", "Neh", "Esth", "Job", "Ps", "Prov", "Eccl", "Song",
    "Isa", "Jer", "Lam", "Ezek", "Dan",
    "Hos", "Joel", "Amos", "Obad", "Jonah", "Mic", "Nah", "Hab",
    "Zeph", "Hag", "Zech", "Mal",
    "Matt", "Mark", "Luke", "John", "Acts", "Rom",
    "1Cor", "2Cor", "Gal", "Eph", "Phil", "Col",
    "1Thess", "2Thess", "1Tim", "2Tim", "Titus", "Phlm",
    "Heb", "Jas", "1Pet", "2Pet", "1John", "2John", "3John", "Jude", "Rev",
]

DISPLAY_NAMES = {
    "Gen": "Genesis", "Exod": "Exodus", "Lev": "Leviticus", "Num": "Numbers",
    "Deut": "Deuteronomy", "Josh": "Joshua", "Judg": "Judges", "Ruth": "Ruth",
    "1Sam": "1 Samuel", "2Sam": "2 Samuel", "1Kgs": "1 Kings", "2Kgs": "2 Kings",
    "1Chr": "1 Chronicles", "2Chr": "2 Chronicles",
    "Ezra": "Ezra", "Neh": "Nehemiah", "Esth": "Esther", "Job": "Job",
    "Ps": "Psalms", "Prov": "Proverbs", "Eccl": "Ecclesiastes", "Song": "Song of Solomon",
    "Isa": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations", "Ezek": "Ezekiel", "Dan": "Daniel",
    "Hos": "Hosea", "Joel": "Joel", "Amos": "Amos", "Obad": "Obadiah",
    "Jonah": "Jonah", "Mic": "Micah", "Nah": "Nahum", "Hab": "Habakkuk",
    "Zeph": "Zephaniah", "Hag": "Haggai", "Zech": "Zechariah", "Mal": "Malachi",
    "Matt": "Matthew", "Mark": "Mark", "Luke": "Luke", "John": "John",
    "Acts": "Acts", "Rom": "Romans",
    "1Cor": "1 Corinthians", "2Cor": "2 Corinthians", "Gal": "Galatians",
    "Eph": "Ephesians", "Phil": "Philippians", "Col": "Colossians",
    "1Thess": "1 Thessalonians", "2Thess": "2 Thessalonians",
    "1Tim": "1 Timothy", "2Tim": "2 Timothy", "Titus": "Titus", "Phlm": "Philemon",
    "Heb": "Hebrews", "Jas": "James", "1Pet": "1 Peter", "2Pet": "2 Peter",
    "1John": "1 John", "2John": "2 John", "3John": "3 John", "Jude": "Jude", "Rev": "Revelation",
}

USER_TO_OSIS = {}
for osis in OSIS_BOOKS:
    full = DISPLAY_NAMES[osis]
    USER_TO_OSIS[osis.lower()] = osis
    USER_TO_OSIS[full.lower().replace(" ", "")] = osis

REF_RE = re.compile(r"^\s*([1-3]?\s*[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d+):(\d+)\s*$")


def _book_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _parse_user_ref(ref: str):
    m = REF_RE.match(ref)
    if not m:
        return None
    book = USER_TO_OSIS.get(_book_key(m.group(1)))
    if not book:
        return None
    return f"{book}.{m.group(2)}.{m.group(3)}"


def _osis_to_display(osis: str) -> str:
    """Gen.1.1 -> 'Genesis 1:1'. Also handles ranges Gen.1.1-Gen.1.2."""
    if "-" in osis:
        # range; take both endpoints
        a, b = osis.split("-", 1)
        a_d = _osis_to_display(a)
        # b may be either full ref or just "verse" continuation
        if "." in b:
            b_d = _osis_to_display(b)
            return f"{a_d}-{b_d}"
        return f"{a_d}-{b}"
    parts = osis.split(".")
    if len(parts) == 3:
        book, ch, v = parts
        return f"{DISPLAY_NAMES.get(book, book)} {ch}:{v}"
    return osis


@lru_cache(maxsize=1)
def _load_index():
    """Returns {osis_from: [(osis_to, votes), ...]} sorted by votes desc."""
    if not DATA.exists():
        return {}
    index = {}
    with DATA.open() as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("From Verse"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            src, dst, votes = parts[0], parts[1], parts[2]
            try:
                v = int(votes)
            except ValueError:
                continue
            index.setdefault(src, []).append((dst, v))
    for k in index:
        index[k].sort(key=lambda x: -x[1])
    return index


def crossref_lookup(ref: str) -> str:
    if not isinstance(ref, str):
        return "ERROR: crossref_lookup expects a string reference."
    if "|" in ref:
        ref_part, limit_str = ref.rsplit("|", 1)
        try:
            limit = max(1, min(50, int(limit_str.strip())))
        except ValueError:
            return f"ERROR: limit '{limit_str}' is not a number."
    else:
        ref_part, limit = ref, DEFAULT_LIMIT

    osis = _parse_user_ref(ref_part)
    if osis is None:
        return f"ERROR: malformed reference '{ref_part}'. Expected 'Book Chapter:Verse'."

    index = _load_index()
    if not index:
        return f"ERROR: crossref data not loaded; expected file at {DATA}"

    refs = index.get(osis, [])
    if not refs:
        return f"No cross-references found for {_osis_to_display(osis)}."

    out = []
    for dst, votes in refs[:limit]:
        out.append(f"{_osis_to_display(dst)} (votes={votes})")
    return "; ".join(out)


if __name__ == "__main__":
    import sys
    print(crossref_lookup(sys.argv[1] if len(sys.argv) > 1 else "John 3:16"))
