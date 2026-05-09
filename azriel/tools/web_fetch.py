"""web_fetch tool.

Fetch a URL and return its main readable text. No external deps -- uses
stdlib urllib + a simple HTML-to-text strip. Truncates at ~6000 chars
to keep tool-result blocks reasonable.
"""
import html as html_mod
import re
import urllib.parse
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Azriel/0.7"
)
MAX_CHARS = 6000

SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _strip_html(s: str) -> str:
    s = SCRIPT_RE.sub("", s)
    s = TAG_RE.sub(" ", s)
    s = html_mod.unescape(s)
    s = WS_RE.sub(" ", s).strip()
    return s


def web_fetch(url: str) -> str:
    if not isinstance(url, str):
        return "ERROR: web_fetch expects a string URL."
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "ERROR: web_fetch only supports http(s) URLs."
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as r:
            ctype = r.headers.get("Content-Type", "")
            body = r.read(2_000_000).decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        return f"ERROR: HTTP {e.code} from {url}"
    except Exception as e:
        return f"ERROR: fetch failed ({type(e).__name__}: {e})"

    if "html" in ctype:
        text = _strip_html(body)
    else:
        text = body

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n\n[...truncated, original {len(text):,} chars]"
    return text


if __name__ == "__main__":
    import sys
    print(web_fetch(sys.argv[1] if len(sys.argv) > 1 else "https://example.com"))
