"""web_search tool.

Top web results for a query via DuckDuckGo HTML lite (no API key, no
external deps beyond stdlib). Returns up to N titles + URLs + snippets.

Reference forms:
  "what does the parousia mean" -> top 5 results
  "fasting in the early church|10" -> top 10 results
"""
import html as html_mod
import re
import urllib.parse
import urllib.request
from typing import List, Tuple

DEFAULT_LIMIT = 5
MAX_LIMIT = 10
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Azriel/0.7"
)
ENDPOINT = "https://html.duckduckgo.com/html/"

RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.S,
)
TAG_RE = re.compile(r"<[^>]+>")


def _strip(s: str) -> str:
    return html_mod.unescape(TAG_RE.sub("", s)).strip()


def _unwrap_ddg(href: str) -> str:
    if href.startswith("//duckduckgo.com/l/?"):
        q = urllib.parse.urlparse("https:" + href).query
        params = urllib.parse.parse_qs(q)
        if "uddg" in params:
            return urllib.parse.unquote(params["uddg"][0])
    if href.startswith("/l/?"):
        q = urllib.parse.urlparse("https://duckduckgo.com" + href).query
        params = urllib.parse.parse_qs(q)
        if "uddg" in params:
            return urllib.parse.unquote(params["uddg"][0])
    return href


def _search(query: str, limit: int) -> List[Tuple[str, str, str]]:
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=data,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode("utf-8", errors="ignore")
    results = []
    for m in RESULT_RE.finditer(body):
        url = _unwrap_ddg(m.group(1))
        title = _strip(m.group(2))
        snippet = _strip(m.group(3))
        if title and url:
            results.append((title, url, snippet))
        if len(results) >= limit:
            break
    return results


def web_search(query: str) -> str:
    if not isinstance(query, str):
        return "ERROR: web_search expects a string query."
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
        results = _search(q, limit)
    except Exception as e:
        return f"ERROR: search failed ({type(e).__name__}: {e})"
    if not results:
        return f"(no results for '{q}')"
    out = []
    for i, (title, url, snippet) in enumerate(results, 1):
        out.append(f"{i}. {title}\n {url}\n {snippet[:200]}")
    return "\n\n".join(out)


if __name__ == "__main__":
    import sys
    print(web_search(sys.argv[1] if len(sys.argv) > 1 else "spurgeon prayer"))
