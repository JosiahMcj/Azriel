"""image_search tool.

Top image results for a query via DuckDuckGo's vqd-token API. Returns
a list of (title, image_url, source_url, dimensions) entries.

Usage:
  "stained glass crucifixion" -> top 5
  "dove descending|10" -> top 10
"""
import json
import re
import urllib.parse
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Azriel/0.7"
)
DEFAULT_LIMIT = 5
MAX_LIMIT = 12


def _vqd(query: str) -> str:
    url = "https://duckduckgo.com/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode("utf-8", errors="ignore")
    m = re.search(r"vqd=['\"]([^'\"]+)['\"]", body)
    return m.group(1) if m else ""


def image_search(query: str) -> str:
    if not isinstance(query, str):
        return "ERROR: image_search expects a string query."
    if "|" in query:
        q_part, n = query.rsplit("|", 1)
        try:
            limit = max(1, min(MAX_LIMIT, int(n.strip())))
        except ValueError:
            return f"ERROR: limit '{n}' is not a number."
    else:
        q_part, limit = query, DEFAULT_LIMIT
    q = q_part.strip()
    if not q:
        return "ERROR: empty query."

    try:
        token = _vqd(q)
        if not token:
            return "ERROR: failed to obtain DDG token."
        params = urllib.parse.urlencode({
            "l": "us-en", "o": "json", "q": q, "vqd": token,
            "f": ",,,", "p": "1",
        })
        url = "https://duckduckgo.com/i.js?" + params
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Referer": "https://duckduckgo.com/",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        return f"ERROR: image search failed ({type(e).__name__}: {e})"

    results = data.get("results", [])[:limit]
    if not results:
        return f"(no images for '{q}')"
    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()[:100]
        img = r.get("image") or ""
        src = r.get("url") or ""
        w, h = r.get("width") or 0, r.get("height") or 0
        lines.append(f"{i}. {title}\n image: {img}\n source: {src}\n {w}x{h}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    import sys
    print(image_search(sys.argv[1] if len(sys.argv) > 1 else "dove descending"))
