"""FastAPI server wrapping the Azriel runtime.

Single-process server that loads v0.6.0 once at startup (~12s) and serves
chat/memory/health endpoints. LAN-only by default (binds to a specific
interface, not 0.0.0.0).

Endpoints:
  POST /chat -> run a user message through run_with_tools, return
                       text + tool calls + duration
  GET /memory -> list memory entries (top-N most recent)
  POST /memory -> insert a memory entry
  GET /health -> service status (model loaded? ollama? uptime?)
  GET /tools -> tool registry (names + signatures + docs)

Run:
    PYTHONPATH=. ~/.azriel/.venv/bin/python -m azriel.server
"""
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import re as _re

UPLOAD_DIR = Path.home() / "azriel-files" / "uploads"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024 # 30 MB

ALLOWED_EXT = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".txt", ".md", ".csv", ".json", ".docx", ".xlsx", ".pptx",
}

from .inference import load_phase_beta
from .runtime import is_attack_prompt, run_with_tools
from .tools import REGISTRY, get_active_registry
from . import agent as _agent
from .tools.memory_search import (
    insert as memory_insert,
    list_all as memory_list,
    delete as memory_delete,
)
from . import connectors as _connectors

CONV_DB = Path.home() / ".azriel" / "data" / "conversations.db"


def _conv_conn() -> sqlite3.Connection:
    CONV_DB.parent.mkdir(parents=True, exist_ok=True)
    fresh = not CONV_DB.exists()
    c = sqlite3.connect(str(CONV_DB))
    if fresh:
        c.execute(
            "CREATE TABLE messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, "
            "role TEXT NOT NULL, "
            "text TEXT NOT NULL, "
            "calls_json TEXT, "
            "route TEXT, "
            "ts INTEGER NOT NULL)"
        )
        c.execute("CREATE INDEX idx_session_ts ON messages(session_id, ts)")
        c.commit()
    return c

# swappable base model.
#
# Default is the Qwen3-Coder-30B-A3B-Instruct base that the Azriel
# LoRA was trained against. Users can point at any other MLX-
# compatible local model via env vars:
#
# AZRIEL_BASE_MODEL -- HF repo id or local path of the base model
# AZRIEL_ADAPTER_PATH -- LoRA adapter to apply on top (optional;
# empty string means run the base model raw)
# AZRIEL_DISABLE_WRAPPER=1 -- skip the custom architecture wrapper
# entirely and use the raw mlx-lm model;
# useful when running with a non-Qwen base
# since the wrapper's looped-layer + LTI
# plumbing is tuned to Qwen's block layout.
#
# Tradeoff: with a non-Qwen base + DISABLE_WRAPPER=1 you keep the
# whole Azriel runtime stack (tools, agent mode, dashboard, memory,
# safety floor) but you don't get the trained Azriel personality
# (that lives in the LoRA, which only fits the matching base). The
# constitution still gets injected as system prompt; the base model
# just isn't trained to it.
BASE_MODEL = os.environ.get(
    "AZRIEL_BASE_MODEL",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
)
ADAPTER_PATH = os.environ.get(
    "AZRIEL_ADAPTER_PATH",
    str(Path.home() / ".azriel" / "checkpoints" / "azriel-v0.5-release-candidate"),
)
DISABLE_WRAPPER = os.environ.get("AZRIEL_DISABLE_WRAPPER", "").strip() in ("1", "true", "yes")
WEB_DIR = Path(__file__).parent.parent / "web"

# Module-level state
_state = {
    "model": None,
    "tokenizer": None,
    "loaded_at": None,
    "request_count": 0,
}

# mlx_lm.generate is not thread-safe. anyio.to_thread.run_sync dispatches
# to a thread pool, so two concurrent /chat or /agent/step calls would
# race on the model's KV cache and produce corrupt output. Every code
# path that calls into the model holds this lock; non-model paths
# (/sessions, /memory, /tools, /agent/list, /agent/status) bypass it.
import threading as _threading_for_model
_model_lock = _threading_for_model.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    have_adapter = ADAPTER_PATH and Path(ADAPTER_PATH).exists()
    print(
        f"loading {BASE_MODEL}"
        + (f" + adapter {ADAPTER_PATH}" if have_adapter else " (no adapter)")
        + (" (raw, custom-architecture wrapper disabled)" if DISABLE_WRAPPER else ""),
        flush=True,
    )
    t0 = time.time()
    model, tokenizer = load_phase_beta(
        BASE_MODEL,
        ADAPTER_PATH if have_adapter else "",
        disable_wrapper=DISABLE_WRAPPER,
    )
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["loaded_at"] = time.time()
    print(f"loaded in {time.time()-t0:.1f}s; ready", flush=True)
    yield
    print("shutting down", flush=True)


app = FastAPI(title="Azriel", version="0.7-runtime", lifespan=lifespan)

# CORS open for LAN dashboard convenience; tighten before any public exposure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, # required for browsers to send Basic Auth on cross-origin
    allow_methods=["*"],
    allow_headers=["*"],
)


# HTTP Basic Auth.
#
# When AZRIEL_BASIC_AUTH_USER and AZRIEL_BASIC_AUTH_PASS are set in the
# server environment, every request must carry Basic Auth credentials
# matching them. /health is exempt so monitoring can still probe.
# Missing creds -> 401 with WWW-Authenticate so browsers prompt for
# login. Bad creds -> 401 plain. If the env vars are not set the
# middleware no-ops (preserves dev-mode behavior).
import base64 as _b64
import hmac as _hmac
import secrets as _secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as _StarletteResponse

_AUTH_EXEMPT_PATHS = {"/health"}


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, user: str, password: str):
        super().__init__(app)
        self._user = user
        self._password = password

    async def dispatch(self, request, call_next):
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)
        # Allow CORS preflight to bypass auth (browser doesn't include
        # credentials on OPTIONS).
        if request.method == "OPTIONS":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("basic "):
            return _unauth_response()
        try:
            decoded = _b64.b64decode(header[6:].strip()).decode("utf-8", "replace")
        except Exception:
            return _unauth_response()
        if ":" not in decoded:
            return _unauth_response()
        user, _, password = decoded.partition(":")
        ok = (_hmac.compare_digest(user, self._user) and
              _hmac.compare_digest(password, self._password))
        if not ok:
            return _unauth_response(prompt=False)
        return await call_next(request)


def _unauth_response(prompt: bool = True) -> _StarletteResponse:
    headers = {}
    if prompt:
        headers["WWW-Authenticate"] = 'Basic realm="Azriel", charset="UTF-8"'
    return _StarletteResponse("Authentication required", status_code=401,
                              headers=headers)


_BASIC_USER = os.environ.get("AZRIEL_BASIC_AUTH_USER", "").strip()
_BASIC_PASS = os.environ.get("AZRIEL_BASIC_AUTH_PASS", "").strip()
if _BASIC_USER and _BASIC_PASS:
    app.add_middleware(BasicAuthMiddleware, user=_BASIC_USER, password=_BASIC_PASS)
    print(f"[auth] Basic Auth enabled for user '{_BASIC_USER}'", flush=True)
else:
    print("[auth] AZRIEL_BASIC_AUTH_USER/PASS not set; auth disabled", flush=True)


# rate limiting.
#
# In-memory sliding-window per (client_ip, path_category). The agent loop
# amplifies request volume (auto-step can fire 10 calls in seconds), so
# the limits below are sized to allow normal interactive use while
# stopping a runaway loop or a scraping bot.
#
# Real client IP comes from cf-connecting-ip when behind Cloudflare,
# otherwise request.client.host. 127.0.0.1 is exempt (local SSH tunnel
# is trusted; there is no NAT between you and the server).
import collections as _collections
import threading as _threading

# Path -> (limit, window_seconds). Anything not listed gets the LIGHT
# bucket. /health is already exempted at the auth layer below; the rate
# limiter checks the auth-exempt set first and skips it.
_RATE_HEAVY = (30, 60) # /chat, /agent/start, /agent/step, /chat/critique
_RATE_MEDIUM = (60, 60) # /upload, /persona/auto-mix
_RATE_LIGHT = (240, 60) # read-only, dashboard polling
_RATE_BUCKETS = {
    "/chat": _RATE_HEAVY,
    "/chat/critique": _RATE_HEAVY,
    "/agent/start": _RATE_HEAVY,
    "/agent/step": _RATE_HEAVY,
    "/upload": _RATE_MEDIUM,
    "/persona/auto-mix": _RATE_MEDIUM,
}
_RATE_DEFAULT = _RATE_LIGHT

# Per-(ip, path) deque of recent request timestamps; oldest expired
# lazily on each check. The lock guards _rate_state.
_rate_state: dict[tuple[str, str], _collections.deque] = {}
_rate_lock = _threading.RLock()


def _client_ip(request) -> str:
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def _bucket_for(path: str) -> tuple[int, int]:
    return _RATE_BUCKETS.get(path, _RATE_DEFAULT)


def _rate_check(ip: str, path: str) -> tuple[bool, int]:
    """Returns (allowed, retry_after_s). Allowed=False means the
    request would exceed the bucket limit."""
    if ip in ("127.0.0.1", "::1"):
        return (True, 0)
    limit, window = _bucket_for(path)
    now = time.time()
    cutoff = now - window
    key = (ip, path)
    with _rate_lock:
        dq = _rate_state.get(key)
        if dq is None:
            dq = _collections.deque()
            _rate_state[key] = dq
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            retry = max(1, int(dq[0] + window - now))
            return (False, retry)
        dq.append(now)
    return (True, 0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        ip = _client_ip(request)
        ok, retry = _rate_check(ip, request.url.path)
        if not ok:
            headers = {"Retry-After": str(retry)}
            return _StarletteResponse(
                f"rate limit: try again in {retry}s",
                status_code=429, headers=headers,
            )
        return await call_next(request)


_RATE_LIMIT_ENABLED = os.environ.get("AZRIEL_RATE_LIMIT", "1").strip() != "0"
if _RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)
    print(f"[rate] limits enabled (heavy={_RATE_HEAVY[0]}/{_RATE_HEAVY[1]}s, "
          f"light={_RATE_LIGHT[0]}/{_RATE_LIGHT[1]}s, 127.0.0.1 exempt)",
          flush=True)
else:
    print("[rate] AZRIEL_RATE_LIMIT=0; limits disabled", flush=True)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    max_calls: int = 5
    temperature: float = 0.3
    session_id: Optional[str] = None
    style: Optional[str] = None # "conviction" | "scholar" | "pastoral"
    persona_mix: Optional[dict] = None # {"funny": 25, "professional": 60, ...}
    thinking: Optional[bool] = False # deliberate mode (chain-of-thought + 2x loop depth)


class ToolCallOut(BaseModel):
    name: Optional[str]
    arg: Optional[str]
    result: str


class ChatOut(BaseModel):
    text: str
    calls: list[ToolCallOut]
    route: str
    reason_for_stop: str
    duration_ms: int
    hallucinated_tools: list[str] = [] # model emitted fake <tool_result> markup for these names without firing the tool
    skill_proposal: Optional[dict] = None # model fired propose_skill; dashboard renders save card


class MemoryIn(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "manual"


@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn) -> ChatOut:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    _state["request_count"] += 1
    t0 = time.time()

    # Fetch the session's prior turns so follow-up questions
    # ('where does that come from?') have antecedent context. The runtime
    # packs to a char budget (~12k tokens) so we forward the full session
    # transcript -- packing happens token-aware in runtime._pack_history.
    # Hard cap at 400 messages defends against pathological session size.
    history = []
    if body.session_id:
        try:
            c = _conv_conn()
            rows = c.execute(
                "SELECT role, text FROM messages WHERE session_id=? "
                "ORDER BY ts ASC, id ASC LIMIT 400",
                (body.session_id,),
            ).fetchall()
            c.close()
            history = [{"role": r[0], "text": r[1]} for r in rows]
        except Exception:
            history = []

    # mlx_lm.generate is sync + GPU-bound; offload to a thread so the event
    # loop can still serve /health concurrently. _model_lock serializes
    # model calls so two clients can't corrupt KV cache state.
    import anyio
    def _run():
        with _model_lock:
            return run_with_tools(
                _state["model"],
                _state["tokenizer"],
                body.message,
                max_calls=body.max_calls,
                temperature=body.temperature,
                history=history,
                style=body.style,
                persona_mix=body.persona_mix,
                thinking=bool(body.thinking),
            )
    out = await anyio.to_thread.run_sync(_run)
    response = ChatOut(
        text=out["text"],
        calls=[ToolCallOut(name=n, arg=a, result=r) for n, a, r in out["calls"]],
        route=out.get("route", "tools"),
        reason_for_stop=out.get("reason_for_stop", "natural"),
        duration_ms=int((time.time() - t0) * 1000),
        hallucinated_tools=out.get("hallucinated_tool_results", []),
        skill_proposal=out.get("skill_proposal"),
    )

    if body.session_id:
        ts = int(time.time())
        c = _conv_conn()
        c.execute(
            "INSERT INTO messages(session_id, role, text, calls_json, route, ts) VALUES (?,?,?,?,?,?)",
            (body.session_id, "user", body.message, None, None, ts),
        )
        c.execute(
            "INSERT INTO messages(session_id, role, text, calls_json, route, ts) VALUES (?,?,?,?,?,?)",
            (
                body.session_id, "assistant", out["text"],
                json.dumps([{"name": n, "arg": a, "result": r} for n, a, r in out["calls"]]),
                out.get("route", "tools"),
                ts + 1,
            ),
        )
        c.commit()
        c.close()

    return response


class CritiqueIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    response: str = Field(..., min_length=1, max_length=20000)
    include_constitution: bool = True
    max_tokens: int = 400


class CritiqueOut(BaseModel):
    severity: str
    revise_recommended: bool
    factual_issues: list[str]
    scripture_issues: list[str]
    doctrinal_issues: list[str]
    internal_contradictions: list[str]
    parse_failed: bool
    duration_ms: int
    raw: str
    constitution_in_context: bool


@app.post("/chat/critique", response_model=CritiqueOut)
async def chat_critique(body: CritiqueIn) -> CritiqueOut:
    """second-pass critique of an assistant answer.

    Returns a structured verdict (factual / scripture / doctrinal /
    contradiction issues + severity). LOGGED, not GATING -- callers
    should not block memory promotion or chat responses on this output.
    Same-model bias is a known limitation (see azriel/critic.py
    docstring).
    """
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    _state["request_count"] += 1
    from .critic import critique as _critique
    import anyio
    def _run_crit():
        with _model_lock:
            return _critique(
                _state["model"], _state["tokenizer"],
                body.message, body.response,
                include_constitution=body.include_constitution,
                max_tokens=body.max_tokens,
            )
    c = await anyio.to_thread.run_sync(_run_crit)
    return CritiqueOut(
        severity=c.severity,
        revise_recommended=c.revise_recommended,
        factual_issues=c.factual_issues,
        scripture_issues=c.scripture_issues,
        doctrinal_issues=c.doctrinal_issues,
        internal_contradictions=c.internal_contradictions,
        parse_failed=c.parse_failed,
        duration_ms=c.duration_ms,
        raw=c.raw,
        constitution_in_context=c.constitution_in_context,
    )


class SaveToMemoryIn(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "conversation"


@app.get("/sessions")
async def sessions_list():
    c = _conv_conn()
    rows = c.execute(
        "SELECT session_id, COUNT(*) AS n, MIN(ts) AS started, MAX(ts) AS last, "
        " (SELECT text FROM messages WHERE session_id = m.session_id "
        " AND role='user' ORDER BY ts LIMIT 1) AS first_user "
        "FROM messages m GROUP BY session_id ORDER BY last DESC LIMIT 50"
    ).fetchall()
    c.close()
    return [
        {"session_id": r[0], "messages": r[1], "started_at": r[2], "last_at": r[3], "first_user": r[4]}
        for r in rows
    ]


@app.get("/sessions/{session_id}")
async def session_get(session_id: str):
    c = _conv_conn()
    rows = c.execute(
        "SELECT role, text, calls_json, route, ts FROM messages "
        "WHERE session_id = ? ORDER BY ts",
        (session_id,),
    ).fetchall()
    c.close()
    out = []
    for role, text, calls_json, route, ts in rows:
        out.append({
            "role": role,
            "text": text,
            "calls": json.loads(calls_json) if calls_json else [],
            "route": route,
            "ts": ts,
        })
    return out


@app.delete("/sessions/{session_id}")
async def session_delete(session_id: str):
    c = _conv_conn()
    cur = c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    c.commit()
    c.close()
    return {"deleted": cur.rowcount}


@app.post("/sessions/{session_id}/save-to-memory")
async def save_to_memory(session_id: str, body: SaveToMemoryIn):
    rid = memory_insert(body.text, source=body.source)
    return {"rowid": rid, "session_id": session_id, "ok": True}


@app.get("/memory")
async def memory_get(limit: int = 50):
    rows = memory_list(limit=limit)
    return [
        {"rowid": r[0], "text": r[1], "source": r[2], "ts": r[3]}
        for r in rows
    ]


@app.post("/memory")
async def memory_post(body: MemoryIn):
    rid = memory_insert(body.text, source=body.source)
    return {"rowid": rid, "ok": True}


@app.delete("/memory/{rowid}")
async def memory_delete_one(rowid: int):
    ok = memory_delete(rowid)
    return {"rowid": rowid, "deleted": ok}


@app.get("/sandbox-list")
async def sandbox_list(path: str = "."):
    """Listing endpoint for the dashboard '+' menu. Same sandbox rules as
    the fs_list tool but returns structured JSON so the UI can render a
    file picker. Honors symlinked virtual mounts (missler/, etc)."""
    from .tools.filesystem import _resolve, SANDBOX
    try:
        p = _resolve(path or ".")
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not p.exists():
        raise HTTPException(404, f"not found: {path}")
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    base = SANDBOX.absolute()
    # At the missler/ root, restrict to actual book dirs (NN_Name pattern)
    # so the picker isn't cluttered by the K-House printing guide,
    # README.txt, "untitled folder", etc.
    is_missler_root = path.strip("/").rstrip("/") == "missler"
    items = []
    for c in sorted(p.iterdir()):
        if c.name.startswith(".") or c.name.startswith("._"):
            continue
        if is_missler_root and not _re.match(r"^\d+_", c.name):
            continue
        try:
            rel = str(c.relative_to(base))
        except ValueError:
            rel = c.name
        try:
            is_dir = c.is_dir()
            size = None if is_dir else c.stat().st_size
        except OSError:
            continue
        items.append({
            "name": c.name,
            "path": rel,
            "is_dir": is_dir,
            "size": size,
        })
    return {"path": path, "items": items}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Receive a file from the dashboard '+' button and store it inside
    the sandbox at ~/azriel-files/uploads/<sanitized>. Returns the
    sandbox-relative path so the model can call fs_read / pdf_extract on
    it. Size capped at 30 MB; extension allowlisted."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = file.filename or "upload"
    base = _re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "upload"
    ext = "." + base.rsplit(".", 1)[1].lower() if "." in base else ""
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(415, f"file type {ext} not allowed")
    target = UPLOAD_DIR / base
    if target.exists():
        stem = base.rsplit(".", 1)[0] if "." in base else base
        i = 1
        while target.exists():
            target = UPLOAD_DIR / f"{stem}-{i}{ext}"
            i += 1
    written = 0
    with target.open("wb") as out:
        while True:
            chunk = await file.read(1 << 16)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")
            out.write(chunk)
    rel = "uploads/" + target.name
    return {"path": rel, "size": written, "name": target.name}


def _azriel_version() -> str:
    """Resolve the release-candidate symlink to its target adapter dir
    (e.g. 'lora-azriel-v0.6.0') and return the version part ('v0.6.0').
    Falls back to 'unknown' if the symlink can't be resolved.

    This is the LoRA-weights version. The runtime layer ships its own
    version separately (RUNTIME_VERSION) so capability bumps that
    don't change the weights (agent mode, new tools, auth) are
    visible without renaming adapter directories."""
    try:
        target = Path(ADAPTER_PATH).resolve().name
        if target.startswith("lora-azriel-"):
            return target[len("lora-azriel-"):]
        return target
    except Exception:
        return "unknown"


# runtime capability version, independent of the
# LoRA-weights version. Bump on user-visible runtime changes:
# auth (iota.1), agent mode (theta.1-6), tool additions (theta.3), etc.
RUNTIME_VERSION = "v0.6.1-agent"


def _health_diagnostics() -> dict:
    """Operational diagnostics: tool count, persistence row counts,
    disk free. All sub-millisecond queries; safe to call on every
    /health hit. Errors swallowed silently -- diagnostics never break
    /health."""
    out: dict = {}
    try:
        out["tools_active"] = len(get_active_registry())
    except Exception:
        out["tools_active"] = None
    try:
        c = _conv_conn()
        out["sessions"] = c.execute(
            "SELECT COUNT(DISTINCT session_id) FROM messages"
        ).fetchone()[0]
        out["messages"] = c.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]
        c.close()
    except Exception:
        out["sessions"] = None
        out["messages"] = None
    try:
        from .tools.memory_search import _conn as _mem_conn
        m = _mem_conn()
        out["memory_rows"] = m.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        m.close()
    except Exception:
        out["memory_rows"] = None
    try:
        import shutil
        out["disk_free_gb"] = round(
            shutil.disk_usage(str(Path.home())).free / (1024 ** 3), 1
        )
    except Exception:
        out["disk_free_gb"] = None
    try:
        # Active agent tasks ()
        out["agent_tasks"] = len(_agent.list_tasks())
    except Exception:
        out["agent_tasks"] = None
    return out


@app.get("/health")
async def health():
    return {
        "status": "ready" if _state["model"] is not None else "loading",
        "name": "Azriel",
        "version": _azriel_version(),
        "runtime_version": RUNTIME_VERSION,
        "model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "loaded_at": _state["loaded_at"],
        "uptime_s": (time.time() - _state["loaded_at"]) if _state["loaded_at"] else 0,
        "requests_served": _state["request_count"],
        "diagnostics": _health_diagnostics(),
    }


@app.get("/tools")
async def tools():
    """Currently *active* tools -- base registry plus any connectors the
    user has plugged in. Disconnected connector tools are not listed."""
    return [
        {"name": name, "signature": spec["signature"], "doc": spec["doc"]}
        for name, spec in get_active_registry().items()
    ]


class AgentStartIn(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    # per-task tool whitelist set by the /agent
    # permissions panel. None / omitted == all currently registered
    # tools available.
    allowed_tools: Optional[list[str]] = None


class AgentStepIn(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=64)


@app.post("/agent/start")
async def agent_start(body: AgentStartIn):
    """start a new agent task. Goal goes through
    is_attack_prompt before any model invocation; matched goals yield
    an ABORTED task with no model call."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    _state["request_count"] += 1
    import anyio
    def _run_start():
        with _model_lock:
            return _agent.start_task(
                _state["model"], _state["tokenizer"],
                body.goal, session_id=body.session_id,
                allowed_tools=body.allowed_tools,
            )
    task = await anyio.to_thread.run_sync(_run_start)
    return _agent.to_dict(task)


@app.post("/agent/step")
async def agent_step(body: AgentStepIn):
    """Advance one step on an existing task. No-op if terminal. Returns
    the full task state."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if _agent.get_task(body.task_id) is None:
        raise HTTPException(status_code=404, detail=f"task not found: {body.task_id}")
    _state["request_count"] += 1
    import anyio
    def _run_step():
        with _model_lock:
            return _agent.step_task(
                _state["model"], _state["tokenizer"], body.task_id,
            )
    task = await anyio.to_thread.run_sync(_run_step)
    return _agent.to_dict(task)


@app.get("/agent/status")
async def agent_status(task_id: str):
    """Read-only view of a task. Does not advance it."""
    task = _agent.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return _agent.to_dict(task)


@app.get("/agent/list")
async def agent_list(session_id: Optional[str] = None):
    """List active+terminal tasks, newest first. Filter by session_id
    when provided."""
    tasks = _agent.list_tasks(session_id=session_id)
    return [_agent.to_dict(t) for t in tasks]


@app.get("/agent/history")
async def agent_history(limit: int = 25):
    """compact history of completed agent tasks. Each
    entry summarizes the goal, resolution reason, citation list, and
    step count. Aborted / parse-fail / capped tasks are NOT included
    (we only memorize successful resolutions)."""
    tasks = _agent.list_tasks()
    out = []
    for t in tasks:
        if t.status != "done":
            continue
        cites = _agent._extract_citations(t)
        out.append({
            "task_id": t.task_id,
            "goal": t.goal,
            "resolved": t.last_reason,
            "citations": cites,
            "step_count": len(t.steps),
            "completed_at": t.steps[-1].ts if t.steps else t.created_at,
        })
        if len(out) >= max(1, min(100, limit)):
            break
    return out


@app.get("/connectors")
async def connectors_list():
    return _connectors.list_connectors()


# Keyword -> persona-preset weights. Used by /persona/auto-mix when no
# the vision provider / Ollama teacher is connected (deterministic fallback).
_PERSONA_KEYWORDS = {
    "funny": ["funny", "joke", "playful", "humor", "humour", "witty", "lighthearted", "lol"],
    "ecstatic": ["ecstatic", "joyful", "excited", "celebrate", "celebratory", "exuberant", "emoji"],
    "personal": ["personal", "personable", "warm", "friendly", "intimate"],
    "somber": ["somber", "serious", "weighty", "quiet", "solemn", "reverent"],
    "professional": ["professional", "professinal", "crisp", "formal", "clean", "polished", "structured"],
    "interesting": ["interesting", "curious", "deep", "etymology", "history", "fascinating"],
    "nurturing": ["nurturing", "gentle", "kind", "supportive", "tender", "loving", "love", "caring"],
    "direct": ["direct", "blunt", "no-hedge", "no hedge", "to the point", "concise"],
    "poetic": ["poetic", "lyrical", "image", "image-rich", "rhythmic"],
    "encouraging": ["encouraging", "uplifting", "affirming", "build up", "edify"],
}


class PersonaAutoMixIn(BaseModel):
    description: str = Field(..., min_length=1, max_length=400)


@app.post("/persona/auto-mix")
async def persona_auto_mix(body: PersonaAutoMixIn):
    """Decompose a freeform voice description into preset percentages.
    Pure keyword matcher today. Future: route to the vision provider Sonnet 4.6
    when the connector is plugged, falling back to qwen2.5:14b."""
    desc = (body.description or "").lower()
    raw = {}
    for preset, kws in _PERSONA_KEYWORDS.items():
        for kw in kws:
            if kw in desc:
                raw[preset] = raw.get(preset, 0) + 1
    if not raw:
        return {"mix": {}, "method": "keyword", "note": "no preset keywords matched"}
    total = sum(raw.values())
    mix = {k: max(5, int(round(100 * v / total))) for k, v in raw.items()}
    # Renormalize to sum=100
    s = sum(mix.values())
    if s > 0:
        mix = {k: int(round(v * 100 / s)) for k, v in mix.items()}
    return {"mix": mix, "method": "keyword"}


class ConnectorConnectIn(BaseModel):
    token: Optional[str] = None
    # Future-proof: arbitrary extra config fields are allowed by the
    # underlying connect() function via dict access.

    class Config:
        extra = "allow"


@app.post("/connectors/{name}/connect")
async def connector_connect(name: str, body: ConnectorConnectIn):
    cfg = body.model_dump(exclude_none=True) if hasattr(body, "model_dump") else body.dict(exclude_none=True)
    res = _connectors.connect(name, cfg)
    if not res.get("ok"):
        raise HTTPException(400, res.get("error", "connect failed"))
    return res


@app.post("/connectors/{name}/disconnect")
async def connector_disconnect(name: str):
    res = _connectors.disconnect(name)
    if not res.get("ok"):
        raise HTTPException(400, res.get("error", "disconnect failed"))
    return res


# no-cache headers on dashboard HTML so a redeploy
# of any of these pages takes effect on next nav, not after a hard
# refresh. The user got bitten twice in a row by stale browser
# copies of /skills and /agent. must-revalidate forces a conditional
# GET; the static asset is small so the round-trip cost is trivial.
_HTML_NOCACHE = {"Cache-Control": "no-cache, must-revalidate"}


@app.get("/", response_class=HTMLResponse)
async def root():
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index, headers=_HTML_NOCACHE)
    return HTMLResponse(
        "<h1>Azriel</h1><p>Server is up; dashboard not yet built. "
        "See /docs for the OpenAPI explorer.</p>"
    )


@app.get("/agent", response_class=HTMLResponse)
async def agent_panel():
    """standalone agent-mode panel. Talks to the
    /agent/start, /agent/step, /agent/list endpoints; same auth realm
    as the rest of the server."""
    page = WEB_DIR / "agent.html"
    if page.exists():
        return FileResponse(page, headers=_HTML_NOCACHE)
    raise HTTPException(404, "agent.html not found")


@app.get("/skills", response_class=HTMLResponse)
async def skills_page():
    """skills catalog. Named workflows that bundle a
    kickoff prompt + persona_mix + style + tool hints. Launch creates
    a fresh session pre-seeded with the kickoff prompt."""
    page = WEB_DIR / "skills.html"
    if page.exists():
        return FileResponse(page, headers=_HTML_NOCACHE)
    raise HTTPException(404, "skills.html not found")


# user-created skills, persisted at
# ~/.azriel/data/user_skills.json. Built-in skills live inline in
# web/skills.html; user skills are added via propose_skill flow OR
# direct /skills/save calls.
USER_SKILLS_PATH = Path.home() / ".azriel" / "data" / "user_skills.json"


def _load_user_skills() -> list[dict]:
    if not USER_SKILLS_PATH.exists():
        return []
    try:
        d = json.loads(USER_SKILLS_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save_user_skills(skills: list[dict]) -> None:
    USER_SKILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_SKILLS_PATH.write_text(
        json.dumps(skills, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _slugify(s: str) -> str:
    out = _re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (out[:48] or "skill")


class SkillSaveIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    kickoff: str = Field(..., min_length=1, max_length=2000)
    style: Optional[str] = None
    persona_mix: Optional[dict] = None
    thinking: Optional[bool] = False
    icon: Optional[str] = None
    sub: Optional[str] = None


@app.get("/skills/list")
async def skills_list():
    """Return all user-created skills (built-ins live in web/skills.html)."""
    return _load_user_skills()


@app.post("/skills/save")
async def skills_save(body: SkillSaveIn):
    """Append a new user skill or update one with the same id."""
    skills = _load_user_skills()
    sid = _slugify(body.name)
    # Dedupe: if a skill with the same id already exists, replace it.
    skills = [s for s in skills if s.get("id") != sid]
    entry = {
        "id": sid,
        "icon": body.icon or "✦",
        "name": body.name.strip(),
        "sub": (body.sub or body.kickoff[:120]).strip(),
        "kickoff": body.kickoff.strip(),
        "style": body.style if body.style in ("conviction", "scholar", "pastoral") else None,
        "persona_mix": body.persona_mix or None,
        "thinking": bool(body.thinking),
        "user_created": True,
        "created_at": int(time.time()),
    }
    # Drop None values so the JSON is clean.
    entry = {k: v for k, v in entry.items() if v is not None}
    skills.append(entry)
    _save_user_skills(skills)
    return {"ok": True, "skill": entry}


@app.delete("/skills/{skill_id}")
async def skills_delete(skill_id: str):
    """Remove a user-created skill. Built-ins are not in this store
    so attempting to delete a built-in is a no-op."""
    skills = _load_user_skills()
    n_before = len(skills)
    skills = [s for s in skills if s.get("id") != skill_id]
    if len(skills) == n_before:
        return {"ok": False, "error": "not found"}
    _save_user_skills(skills)
    return {"ok": True, "deleted": skill_id}


# Mount static frontend directory if it exists.
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    import uvicorn
    host = os.environ.get("AZRIEL_HOST", "127.0.0.1")
    port = int(os.environ.get("AZRIEL_PORT", "8080"))
    uvicorn.run("azriel.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
