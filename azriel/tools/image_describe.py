"""image_describe -- vision tool.

Sends an image from the sandbox to a vision API for description, or
to a local Ollama vision model if one is pulled.

The base model has no vision encoder, so this tool calls out. There
are two backends, tried in this order:

  1. Local vision via Ollama (free, ~30s, no auth). If you have a
     vision-capable model pulled (`ollama pull llava` /
     `ollama pull qwen2.5vl` etc.), this is the default path.
  2. Custom vision API. Drop a key at ~/.azriel-secrets/vision_api.json
     with the schema below and the tool will use it. The default
     payload format follows the most common chat-completions vision
     API shape; override AZRIEL_VISION_API_URL, AZRIEL_VISION_MODEL,
     and AZRIEL_VISION_AUTH_HEADER if your provider uses a different
     header name.

Secret JSON schema:
    {
      "api_key": "your-key-here",
      "url": "https://your-vision-api.example/v1/chat/completions",
      "model": "your-vision-model-id",
      "header": "Authorization" // or "x-api-key" etc.
    }

Input:
  - sandbox-relative path (relative to ~/azriel-files), e.g.
    "uploads/photo.jpg"
  - optionally, "path|prompt" to override the default describe prompt

Output:
  - either the text description, or an ERROR string with diagnostic
    info (no key, image too large, API error, unsupported format)
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
import urllib.error
from pathlib import Path

SANDBOX = Path.home() / "azriel-files"
SECRET_PATH = Path.home() / ".azriel-secrets" / "vision_api.json"

# Generic chat-completions vision payload defaults. Override via env
# vars or the secret JSON file if your provider uses something else.
DEFAULT_API_URL = os.environ.get("AZRIEL_VISION_API_URL", "")
DEFAULT_MODEL = os.environ.get("AZRIEL_VISION_MODEL", "")
DEFAULT_AUTH_HEADER = os.environ.get("AZRIEL_VISION_AUTH_HEADER", "Authorization")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
}
MAX_BYTES = 5 * 1024 * 1024 # 5 MB


def _load_secret() -> dict | None:
    if not SECRET_PATH.exists():
        return None
    try:
        return json.loads(SECRET_PATH.read_text())
    except Exception:
        return None


def _resolve_in_sandbox(rel: str) -> Path | None:
    """Resolve a sandbox-relative path. Reject anything that escapes
    the sandbox via .. or absolute paths outside it."""
    rel = rel.strip().lstrip("/")
    p = (SANDBOX / rel).resolve()
    try:
        p.relative_to(SANDBOX.resolve())
    except ValueError:
        return None
    return p if p.exists() and p.is_file() else None


_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
_OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
_VISION_MODEL_HINTS = (
    "llava", "qwen2.5vl", "qwen2-vl", "qwen3.6", "gemma3", "gemma:7b-vision",
    "minicpm-v", "moondream",
)


def _detect_local_vision_model() -> str | None:
    """Return the first locally-pulled Ollama tag that looks vision-
    capable, or None. Heuristic on tag name."""
    try:
        with urllib.request.urlopen(_OLLAMA_TAGS_URL, timeout=5) as r:
            d = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None
    for m in d.get("models", []):
        name = (m.get("name") or "").lower()
        if any(h in name for h in _VISION_MODEL_HINTS):
            return m.get("name")
    return None


def _try_ollama_vision(image_path: Path, prompt: str) -> str | None:
    """Best-effort local-vision via Ollama. Returns the description
    string on success, or None if Ollama is unreachable / no vision
    model pulled."""
    model = _detect_local_vision_model()
    if not model:
        return None
    try:
        b = image_path.read_bytes()
    except OSError:
        return None
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(b).decode("ascii")],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 600},
    }).encode()
    req = urllib.request.Request(
        _OLLAMA_GEN_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return f"ERROR: local vision via Ollama failed: {e}"
    text = (d.get("response") or "").strip()
    if not text:
        return None
    return f"{text}\n\n[generated locally via Ollama {model}]"


def _try_remote_vision(image_path: Path, prompt: str, ext: str) -> str | None:
    """Call the configured remote vision API. Returns the description
    text on success or an ERROR string on failure. Returns None if no
    secret / endpoint is configured (caller falls back)."""
    secret = _load_secret() or {}
    key = (secret.get("api_key") or secret.get("token") or "").strip()
    url = (secret.get("url") or DEFAULT_API_URL or "").strip()
    model = (secret.get("model") or DEFAULT_MODEL or "").strip()
    header_name = (secret.get("header") or DEFAULT_AUTH_HEADER or "Authorization").strip()
    if not (key and url and model):
        return None
    try:
        b = image_path.read_bytes()
    except OSError as e:
        return f"ERROR: cannot read image: {e}"
    media_type = MEDIA_TYPES.get(ext, "image/jpeg")
    data_url = f"data:{media_type};base64,{base64.b64encode(b).decode('ascii')}"
    # Generic chat-completions vision shape (OpenAI-compatible).
    payload = {
        "model": model,
        "max_tokens": 1000,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    auth_value = key if header_name.lower() != "authorization" else f"Bearer {key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            header_name: auth_value,
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", "replace")
        except Exception:
            err_body = ""
        return f"ERROR: HTTP {e.code} from vision API: {err_body[:300]}"
    except urllib.error.URLError as e:
        return f"ERROR: network error contacting vision API: {e}"
    except OSError as e:
        return f"ERROR: I/O error: {e}"
    # Try chat-completions response shape, then fall back to a few
    # common alternatives (different providers nest differently).
    choices = body.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        text = (msg.get("content") or "").strip() if isinstance(msg.get("content"), str) else ""
        if not text and isinstance(msg.get("content"), list):
            text = "\n".join(
                (c.get("text") or "") for c in msg["content"] if isinstance(c, dict)
            ).strip()
        if text:
            return text
    # Fallback for content-block style responses.
    blocks = body.get("content") or []
    text_parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    text = "\n".join(t for t in text_parts if t).strip()
    if text:
        return text
    return f"ERROR: vision API returned no text content. Raw: {json.dumps(body)[:300]}"


def image_describe(arg: str) -> str:
    """Return a textual description of an image in the sandbox.

    arg: "path" or "path|prompt"
      path: relative to ~/azriel-files (e.g. "uploads/photo.jpg")
      prompt: optional override, default describes what's in the image
    """
    if not arg or not arg.strip():
        return "ERROR: empty path. Pass a sandbox-relative image path like uploads/photo.jpg."
    if "|" in arg:
        rel, prompt = arg.split("|", 1)
        rel = rel.strip()
        prompt = prompt.strip() or "Describe what you see in this image."
    else:
        rel = arg.strip()
        prompt = (
            "Describe what you see in this image. Be specific about visible "
            "objects, text, people, scene, and overall composition. If the "
            "image contains text, transcribe it. If it depicts a person, "
            "describe what they appear to be doing without making "
            "identifying claims."
        )
    ext = Path(rel).suffix.lower()
    if ext not in ALLOWED_EXT:
        return f"ERROR: unsupported image format '{ext}'. Allowed: jpg, jpeg, png, gif, webp."
    p = _resolve_in_sandbox(rel)
    if p is None:
        return f"ERROR: image not found in sandbox: {rel}"
    try:
        size = p.stat().st_size
    except OSError as e:
        return f"ERROR: cannot stat {rel}: {e}"
    if size > MAX_BYTES:
        return f"ERROR: image too large ({size:,} bytes; max {MAX_BYTES:,})."
    if size < 32:
        return f"ERROR: image suspiciously small ({size} bytes); likely placeholder."

    # Try local Ollama vision first (free, no key needed).
    local = _try_ollama_vision(p, prompt)
    if local is not None:
        return local

    # Fall back to a remote vision API if configured.
    remote = _try_remote_vision(p, prompt, ext)
    if remote is not None:
        return remote

    return (
        "ERROR: no vision backend available. Either pull a local "
        "vision-capable Ollama model (`ollama pull llava` / `ollama "
        "pull qwen2.5vl`) OR drop a vision API key at "
        "~/.azriel-secrets/vision_api.json with shape "
        '{"api_key": "...", "url": "https://...", "model": "..."}.'
    )
