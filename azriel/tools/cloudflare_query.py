"""cloudflare_query tool.

Reads a Cloudflare API token from ~/.azriel-secrets/cloudflare.json
(format: {"token": "..."}). Supports a small set of read-only queries:

  cloudflare_query("zones") -> list zones
  cloudflare_query("dns:your-domain.example") -> DNS records for a zone
  cloudflare_query("tunnels") -> list cf-tunnels for the account

If the token file is missing, returns a clear instruction.
"""
import json
import urllib.request
from pathlib import Path

TOKEN_PATH = Path.home() / ".azriel-secrets" / "cloudflare.json"
API = "https://api.cloudflare.com/client/v4"


def _get(path: str, token: str):
    req = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _load_token() -> str | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        d = json.loads(TOKEN_PATH.read_text())
        return d.get("token")
    except Exception:
        return None


def cloudflare_query(query: str) -> str:
    if not isinstance(query, str):
        return "ERROR: cloudflare_query expects a string."
    q = query.strip()
    token = _load_token()
    if not token:
        return ("ERROR: Cloudflare token not configured. "
                f"Create {TOKEN_PATH} with {{\"token\":\"YOUR_API_TOKEN\"}} "
                "(scoped read-only Zone.Read + DNS.Read + Account.Read).")

    try:
        if q == "zones":
            d = _get("/zones?per_page=20", token)
            zones = d.get("result", [])
            if not zones:
                return "(no zones)"
            return "\n".join(f"- {z['name']} (id={z['id'][:8]}...)" for z in zones)

        if q.startswith("dns:"):
            zone_name = q[4:].strip()
            d = _get(f"/zones?name={zone_name}", token)
            zones = d.get("result", [])
            if not zones:
                return f"(zone not found: {zone_name})"
            zid = zones[0]["id"]
            d = _get(f"/zones/{zid}/dns_records?per_page=50", token)
            recs = d.get("result", [])
            if not recs:
                return f"(no DNS records for {zone_name})"
            return "\n".join(
                f"- {r['type']:6s} {r['name']} -> {r['content']}" for r in recs
            )

        if q == "tunnels":
            # Need account id; resolve from any zone
            d = _get("/zones?per_page=1", token)
            zones = d.get("result", [])
            if not zones:
                return "ERROR: no zones available to derive account id"
            acct = zones[0]["account"]["id"]
            d = _get(f"/accounts/{acct}/cfd_tunnel?is_deleted=false", token)
            tunnels = d.get("result", [])
            if not tunnels:
                return "(no tunnels)"
            return "\n".join(
                f"- {t['name']} ({t['id'][:8]}...) conns={t.get('connections', [])}"
                for t in tunnels
            )

        return ("ERROR: unsupported query. Use 'zones', 'dns:<zone>', or 'tunnels'.")
    except urllib.error.HTTPError as e:
        return f"ERROR: HTTP {e.code} from Cloudflare API"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


if __name__ == "__main__":
    import sys
    print(cloudflare_query(sys.argv[1] if len(sys.argv) > 1 else "zones"))
