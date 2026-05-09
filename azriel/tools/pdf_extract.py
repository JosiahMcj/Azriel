"""pdf_extract tool.

Extracts text from a PDF -- accepts either a sandboxed local path
(under ~/azriel-files/, including symlinked virtual mounts like
'missler/') or an http(s) URL. Uses pdfplumber.

Usage:
  pdf_extract("https://example.com/sermon.pdf")
  pdf_extract("notes/sermon-2026-04.pdf")
  pdf_extract("missler/65_Jude/65_Jude_Commentary_Handbook.pdf|1-3")
  pdf_extract("notes/sermon.pdf|3-7") -> only pages 3 through 7
"""
import io
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "Mozilla/5.0 Azriel/0.7"
SANDBOX = Path.home() / "azriel-files"
MAX_BYTES = 25_000_000 # 25 MB
MAX_OUT_CHARS = 12_000


def _resolve_local(path_str: str) -> Path:
    """Resolve a path inside the sandbox. Blocks traversal textually so
    symlinks the user placed inside the sandbox (e.g. missler -> real
    handbook dir) are honored when the OS opens the file."""
    base = SANDBOX.absolute()
    raw = (SANDBOX / path_str).absolute()
    p = Path(os.path.normpath(raw))
    base_s = str(base).rstrip(os.sep) + os.sep
    if str(p) != str(base) and not str(p).startswith(base_s):
        raise ValueError(f"path '{path_str}' escapes the sandbox")
    return p


def _fetch_remote(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(MAX_BYTES)


def _parse_pages(spec: str):
    """'3' -> [3]; '3-7' -> [3..7]; None for all."""
    if not spec:
        return None
    s = spec.strip()
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return list(range(a, b + 1))
    if s.isdigit():
        return [int(s)]
    return None


def pdf_extract(arg: str) -> str:
    if not isinstance(arg, str):
        return "ERROR: pdf_extract expects a string path or URL."
    if "|" in arg:
        loc, page_spec = arg.rsplit("|", 1)
        pages = _parse_pages(page_spec)
        if pages is None:
            return f"ERROR: bad page spec '{page_spec}'. Use 'N' or 'A-B'."
    else:
        loc, pages = arg, None
    loc = loc.strip()

    try:
        import pdfplumber
    except ImportError:
        return "ERROR: pdfplumber not installed. Run: uv pip install pdfplumber"

    try:
        if loc.startswith("http://") or loc.startswith("https://"):
            data = _fetch_remote(loc)
            stream = io.BytesIO(data)
        else:
            p = _resolve_local(loc)
            if not p.exists():
                return f"ERROR: file not found: {loc}"
            stream = open(p, "rb")
    except Exception as e:
        return f"ERROR: open failed ({type(e).__name__}: {e})"

    try:
        with pdfplumber.open(stream) as pdf:
            total = len(pdf.pages)
            target = pages if pages else range(1, total + 1)
            chunks = []
            for n in target:
                if n < 1 or n > total:
                    continue
                page = pdf.pages[n - 1]
                txt = (page.extract_text() or "").strip()
                if txt:
                    chunks.append(f"--- page {n} ---\n{txt}")
            text = "\n\n".join(chunks)
    except Exception as e:
        return f"ERROR: extract failed ({type(e).__name__}: {e})"
    finally:
        try: stream.close()
        except Exception: pass

    if not text:
        return "(no extractable text)"
    if len(text) > MAX_OUT_CHARS:
        text = text[:MAX_OUT_CHARS] + f"\n\n[...truncated, {len(text):,} chars total]"
    return text


if __name__ == "__main__":
    import sys
    print(pdf_extract(sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"))
