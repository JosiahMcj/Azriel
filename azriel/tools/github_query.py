"""github_query tool -- search GitHub via PAT, no gh CLI dependency.

Reads a GitHub Personal Access Token from ~/.azriel-secrets/github.json
(format: {"token": "..."}). The connector framework writes that file
when the user plugs in via /connectors/github/connect.

Query routing:
  github_query("repo:foo/bar fastest") -> code search inside that repo
  github_query("user:torvalds") -> user's repos
  github_query("issue is:open auth") -> issues
  github_query("nanoGPT") -> repository search

If the secret is missing, returns a clear ERROR pointing the user at
the connector flow rather than the old gh-CLI instructions.
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

SECRET_PATH = Path.home() / ".azriel-secrets" / "github.json"
USER_AGENT = "Azriel/0.7"
API = "https://api.github.com"


def _load_token() -> str | None:
    if not SECRET_PATH.exists():
        return None
    try:
        d = json.loads(SECRET_PATH.read_text())
        t = (d.get("token") or "").strip()
        return t or None
    except Exception:
        return None


def _http_get(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _format_repo_hit(it: dict) -> str:
    name = it.get("full_name") or it.get("name", "?")
    desc = (it.get("description") or "").replace("\n", " ").strip()[:80]
    stars = it.get("stargazers_count")
    star_part = f" ★{stars}" if stars is not None else ""
    return f"- {name}{star_part}{(' -- ' + desc) if desc else ''}"


def _format_issue_hit(it: dict) -> str:
    n = it.get("number", "?")
    title = (it.get("title") or "").strip()[:90]
    state = it.get("state", "?")
    url = it.get("html_url", "")
    return f"- #{n} [{state}] {title} {url}"


def _format_code_hit(it: dict) -> str:
    repo = (it.get("repository") or {}).get("full_name", "?")
    path = it.get("path", "?")
    url = it.get("html_url", "")
    return f"- {repo}: {path} {url}"


def github_query(query: str) -> str:
    if not isinstance(query, str):
        return "ERROR: github_query expects a string query."
    q = query.strip()
    if not q:
        return "ERROR: empty query."

    token = _load_token()
    if not token:
        return ("ERROR: github connector not connected. "
                "Plug it in via the dashboard Connectors tab "
                "(or POST /connectors/github/connect with a PAT).")

    try:
        if q.startswith("issue") or "is:issue" in q or "is:pr" in q:
            sq = q[6:].lstrip() if q.startswith("issue") else q
            url = f"{API}/search/issues?q={urllib.parse.quote(sq)}&per_page=10"
            d = _http_get(url, token)
            items = d.get("items", [])
            if not items:
                return f"(no github issues for '{q}')"
            return "\n".join(_format_issue_hit(it) for it in items[:10])[:6000]

        if q.startswith("repo:") or q.startswith("user:") or q.startswith("org:"):
            url = f"{API}/search/repositories?q={urllib.parse.quote(q)}&per_page=10"
            d = _http_get(url, token)
            items = d.get("items", [])
            if not items:
                return f"(no github repos for '{q}')"
            return "\n".join(_format_repo_hit(it) for it in items[:10])[:6000]

        # Default: code search
        url = f"{API}/search/code?q={urllib.parse.quote(q)}&per_page=10"
        d = _http_get(url, token)
        items = d.get("items", [])
        if not items:
            return f"(no github code matches for '{q}')"
        return "\n".join(_format_code_hit(it) for it in items[:10])[:6000]

    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            body = ""
        return f"ERROR: HTTP {e.code} from GitHub API. {body}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    import sys
    print(github_query(sys.argv[1] if len(sys.argv) > 1 else "user:karpathy nanoGPT"))
