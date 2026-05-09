"""Connector framework -- MCP style "plug it in" tools.

A *tool* in the base REGISTRY is always-on (no external auth required).
A *connector* is a tool that ONLY exists in the live registry once the
user has plugged in credentials. Until connected, the model never sees
its signature in the primer -- so a fresh fork with no secrets cannot
even attempt to call it.

Connecting persists config to ~/.azriel-secrets/<name>.json (0600).
Disconnecting removes that file. Status is recomputed on every call to
get_active_registry() (and therefore on every chat turn) so plug-in /
unplug is effective immediately, no server restart.

Public-release safety:
  - Secrets dir is .gitignored at the repo root.
  - Connector tool functions are looked up lazily, so importing this
    module never reaches into anyone's home dir.
  - Status checks are pure file-existence + token-non-empty; they do
    not make network calls (so they can't leak the token by accident).
"""
import json
import os
from pathlib import Path

SECRETS_DIR = Path.home() / ".azriel-secrets"


def _ensure_secrets_dir():
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(SECRETS_DIR, 0o700)
    except OSError:
        pass


def _secret_path(name: str) -> Path:
    return SECRETS_DIR / f"{name}.json"


def _read_secret(name: str) -> dict | None:
    p = _secret_path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_secret(name: str, data: dict) -> None:
    _ensure_secrets_dir()
    p = _secret_path(name)
    p.write_text(json.dumps(data))
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def _delete_secret(name: str) -> None:
    p = _secret_path(name)
    if p.exists():
        p.unlink()


# ===== generic per-connector helpers =====

def _token_status(secret_name: str, hint: str) -> dict:
    d = _read_secret(secret_name)
    if d and d.get("token"):
        return {"connected": True, "via": hint}
    return {"connected": False}


def _token_connect(secret_name: str, config: dict) -> dict:
    token = (config or {}).get("token", "").strip()
    if not token:
        return {"ok": False, "error": "missing 'token'"}
    _write_secret(secret_name, {"token": token})
    return {"ok": True}


def _token_disconnect(secret_name: str) -> dict:
    _delete_secret(secret_name)
    return {"ok": True}


# ===== tool wrappers (lazy import to avoid circular deps) =====

def _github_tool(arg: str) -> str:
    from .tools.github_query import github_query
    return github_query(arg)


def _cloudflare_tool(arg: str) -> str:
    from .tools.cloudflare_query import cloudflare_query
    return cloudflare_query(arg)


# the vision provider chat-time connector removed -- Azriel shouldn't
# advertise a competing AI as a "senior teacher" in his own settings;
# it dilutes the identity story. The offline teacher module
# azriel/tools/image_describe.py is still in place for rubric-scoring /
# synthetic-data scripts (the hybrid decision); it just isn't
# exposed to the model as a chat-time tool.


# ===== registry =====

CONNECTORS = {
    "github": {
        "label": "GitHub",
        "doc": "Search GitHub repos / users / issues / code via your PAT.",
        "tool_name": "github_query",
        "signature": "github_query(query: str) -> str",
        "tool_doc": (
            "Search GitHub via the user's PAT. 'repo:foo/bar', 'user:X', "
            "'issue is:open', or freeform code search."
        ),
        "tool_fn": _github_tool,
        "status": lambda: _token_status("github", "PAT in ~/.azriel-secrets/github.json"),
        "connect": lambda cfg: _token_connect("github", cfg),
        "disconnect": lambda: _token_disconnect("github"),
        "config_schema": [
            {
                "key": "token",
                "type": "password",
                "label": "GitHub Personal Access Token",
                "help": (
                    "Create at github.com/settings/tokens. Classic PAT with "
                    "scopes: repo:read, read:user. Stored locally at "
                    "~/.azriel-secrets/github.json (0600)."
                ),
            },
        ],
    },
    "cloudflare": {
        "label": "Cloudflare",
        "doc": "Read-only Cloudflare API -- list zones, DNS records, tunnels.",
        "tool_name": "cloudflare_query",
        "signature": "cloudflare_query(query: str) -> str",
        "tool_doc": (
            "Read-only Cloudflare API. Queries: 'zones', 'dns:<zone>', "
            "'tunnels'. Requires a scoped read-only token."
        ),
        "tool_fn": _cloudflare_tool,
        "status": lambda: _token_status("cloudflare", "token in ~/.azriel-secrets/cloudflare.json"),
        "connect": lambda cfg: _token_connect("cloudflare", cfg),
        "disconnect": lambda: _token_disconnect("cloudflare"),
        "config_schema": [
            {
                "key": "token",
                "type": "password",
                "label": "Cloudflare API Token",
                "help": (
                    "Create at dash.cloudflare.com/profile/api-tokens. "
                    "Use the 'Read all resources' template, or custom with "
                    "Zone.Read + DNS.Read + Account.Read. Stored locally."
                ),
            },
        ],
    },
}


def list_connectors() -> list[dict]:
    """Public list of connectors with current status. Never returns the
    actual secret -- only whether one is configured."""
    out = []
    for name, c in CONNECTORS.items():
        try:
            st = c["status"]()
        except Exception as e:
            st = {"connected": False, "error": str(e)}
        out.append({
            "name": name,
            "label": c["label"],
            "doc": c["doc"],
            "tool_name": c["tool_name"],
            "config_schema": c["config_schema"],
            "status": st,
        })
    return out


def active_tools() -> dict:
    """{tool_name: spec} for connectors currently connected. spec mirrors
    the shape used by tools/__init__.py REGISTRY entries."""
    out = {}
    for name, c in CONNECTORS.items():
        try:
            if c["status"]().get("connected"):
                out[c["tool_name"]] = {
                    "fn": c["tool_fn"],
                    "signature": c["signature"],
                    "doc": c["tool_doc"],
                }
        except Exception:
            pass
    return out


def connect(name: str, config: dict) -> dict:
    c = CONNECTORS.get(name)
    if not c:
        return {"ok": False, "error": f"unknown connector '{name}'"}
    try:
        return c["connect"](config)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def disconnect(name: str) -> dict:
    c = CONNECTORS.get(name)
    if not c:
        return {"ok": False, "error": f"unknown connector '{name}'"}
    try:
        return c["disconnect"]()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
