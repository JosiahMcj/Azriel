"""γ.4.c-γ.4.g concordance ingest.

Targets the 5 most-cited Bible concordances/dictionaries:
  1. Strong's Hebrew Lexicon (openscriptures HebrewStrong.xml -- live)
  2. Strong's Greek Lexicon (queued: needs alternate source)
  3. Naves Topical Bible (queued: needs biblehub-style scrape)
  4. Easton's Illustrated Bible Dictionary (queued: try Gutenberg id 9418)
  5. Smith's Bible Dictionary (queued: alternate source TBD)

Each concordance gets converted to prompt/response JSONL records compatible
with the existing data mix at ~/.azriel/data/synthetic/pairs.jsonl.

Run on a development machine:
    PYTHONPATH=. ~/.azriel/.venv/bin/python scripts/48_ingest_concordances.py strongs_hebrew
    PYTHONPATH=. ~/.azriel/.venv/bin/python scripts/48_ingest_concordances.py easton
    ... etc

Output: ~/.azriel/data/concordances/<source>.jsonl
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

OUT_DIR = Path.home() / ".azriel" / "data" / "concordances"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ingest_strongs_hebrew():
    """Strong's Hebrew Lexicon via openscriptures (MIT-licensed XML).
    Schema: <entry strongs="N"><w xlit="..." pron="...">word</w><note...>defs</note></entry>
    Output: prompt/response pairs like
      "What is the meaning of Strong's Hebrew H1?" -> "ʾāb (Strong's H1) means..."
    """
    url = "https://raw.githubusercontent.com/openscriptures/HebrewLexicon/master/HebrewStrong.xml"
    print(f"fetching {url}", flush=True)
    raw = urlopen(url, timeout=120).read()
    print(f"got {len(raw):,} bytes; parsing", flush=True)

    # Strip the default namespace to keep XPath simple
    raw_text = raw.decode("utf-8")
    raw_text = re.sub(r'\sxmlns="[^"]+"', "", raw_text, count=1)
    root = ET.fromstring(raw_text)

    out_path = OUT_DIR / "strongs_hebrew.jsonl"
    n = 0
    with out_path.open("w") as f:
        for entry in root.iter("entry"):
            sid = entry.attrib.get("id") or entry.attrib.get("strongs")
            if not sid:
                continue
            # word + transliteration
            w = entry.find("w")
            word = w.text if w is not None and w.text else ""
            xlit = w.attrib.get("xlit", "") if w is not None else ""
            pron = w.attrib.get("pron", "") if w is not None else ""
            # collect note text
            notes = []
            for note in entry.iter("note"):
                if note.text:
                    notes.append(note.text.strip())
            # source / meaning
            source = entry.find("source")
            meaning = entry.find("meaning")
            usage = entry.find("usage")
            src_t = "".join(source.itertext()).strip() if source is not None else ""
            mean_t = "".join(meaning.itertext()).strip() if meaning is not None else ""
            usage_t = usage.text.strip() if usage is not None and usage.text else ""

            response_parts = []
            if word:
                response_parts.append(f"{word} ({xlit}, pronounced {pron})")
            if mean_t:
                response_parts.append(mean_t)
            if src_t:
                response_parts.append(f"Source: {src_t}")
            if usage_t:
                response_parts.append(f"KJV usage: {usage_t}")
            response = " ".join(response_parts).strip()
            if not response or len(response) < 20:
                continue

            rec = {
                "prompt": f"What does Strong's Hebrew {sid} ({xlit}) mean?",
                "response": response,
                "source": "concordance:strongs_hebrew",
                "category": "concordance",
                "run_id": "concordance.strongs_hebrew",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n:,} entries to {out_path}", flush=True)


HANDLERS = {
    "strongs_hebrew": ingest_strongs_hebrew,
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "strongs_hebrew"
    if target not in HANDLERS:
        print(f"unknown target '{target}'. Available: {list(HANDLERS.keys())}", file=sys.stderr)
        sys.exit(2)
    HANDLERS[target]()


if __name__ == "__main__":
    main()
