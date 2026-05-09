"""Autoresearch loop runtime.

Goes through a queue of research topics. For each topic, calls the
live /chat endpoint with a research-flavored prompt that enumerates
which tools to use for which kind of subquery, and a tool-call budget
of 10 per topic (vs the chat default of 5). Writes:

  ~/.azriel/data/research/notes/<slug>.md -- the research output
  ~/.azriel/data/research/runs.jsonl -- one row per topic-run
  ~/.azriel/data/research/issues.jsonl -- one row per problem

The issues log is the "fixable bugs" pile -- every tool error,
malformed call, repetition, bracket bleed, etc. lands there
classified so we can spot patterns across runs.

Architecture choice: HTTP to the live server, NOT direct MLX import.
The server already holds the model in memory and applies the γ.8f
primer + persona mix. Going through /chat means autoresearch and
interactive chat share infrastructure -- one runtime, one set of
fixes, no GPU contention.
"""
import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CHAT_URL = "http://localhost:8080/chat"
CRITIQUE_URL = "http://localhost:8080/chat/critique"
DATA_ROOT = Path.home() / ".azriel" / "data" / "research"
NOTES_DIR = DATA_ROOT / "notes"
RUNS_LOG = DATA_ROOT / "runs.jsonl"
ISSUES_LOG = DATA_ROOT / "issues.jsonl"
QUEUE_FILE = DATA_ROOT / "queue.jsonl"

# Tool budget per topic-run. Higher than chat default (5) so the model
# has room to fan out across multiple tools.
MAX_CALLS_PER_TOPIC = 10

# Per-topic time cap. If exceeded the run is logged as `model_timeout`.
TIMEOUT_SECONDS = 300

RESEARCH_PROMPT_TEMPLATE = """\
Research this topic for me, in depth: {topic}

You have access to these tools -- USE AT LEAST THREE DIFFERENT ONES:

  - bible_lookup / crossref_lookup / strongs_lookup (scripture)
  - pdf_extract on missler/<book>/<file>.pdf|1-3 (commentary)
  - web_search + web_fetch (current discussion)
  - memory_search (connect to prior research / notes)
  - image_search (only if a visual reference helps)
  - document_create (only if the user would want a doc)
  - fs_write (save your final note in your sandbox)

TOOL BUDGET: You have at most 10 tool calls. STOP firing tools by call 7
at the latest -- after that, you MUST write the summary and the fs_write,
even if you wanted more research. A coherent partial answer beats an
incomplete tool-spam.

For each subquery, fire the right tool, then INTEGRATE the result into
the running argument. Don't fire the same tool with the same arg twice
(the runtime now blocks that as no_progress). Move forward.

End with a 4-6 paragraph summary that:
  - states the thesis clearly
  - cites every scripture by book/chapter/verse
  - acknowledges where traditions disagree (briefly)
  - closes with a biblically-based position

When the summary is ready, save it via fs_write. The arg is everything
between the first | and the closing quote -- be sure to include the
entire markdown summary on a single fs_write call:

  fs_write("research/{slug}.md|<full markdown summary on one line>")

Then stop. Do NOT keep firing more tools after fs_write -- the run is
complete.
"""


def _slugify(topic: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return s[:80] or "untitled"


# Issue classification regexes
_BRACKET_TOOL_RE = re.compile(r"\[tool_use\s+\w+", re.I)
_REPEAT_LINE_RE = re.compile(r"^(.+)(?:\r?\n\1){3,}$", re.M)
_MALFORMED_RE = re.compile(r"ERROR: malformed tool call")
_INCOMPLETE_RE = re.compile(r"max_calls", re.I)


@dataclass
class Issue:
    ts: float
    topic: str
    iter: int
    kind: str
    detail: str
    snippet: str = ""

    def to_jsonl(self) -> str:
        return json.dumps({
            "ts": self.ts,
            "topic": self.topic,
            "iter": self.iter,
            "kind": self.kind,
            "detail": self.detail,
            "snippet": self.snippet[:300],
        }, ensure_ascii=False)


@dataclass
class RunSummary:
    topic: str
    slug: str
    started_ts: float
    finished_ts: float = 0.0
    duration_s: float = 0.0
    iterations: int = 0
    tools_fired: list = field(default_factory=list)
    tool_errors: int = 0
    issues: int = 0
    success: bool = False
    note_path: str = ""
    reason_for_stop: str = ""
    response_chars: int = 0
    memory_rowid: Optional[int] = None # ε.2: promoted memory row id (if any)
    # ζ.1: optional critic pass over the final note. None = critic not run.
    critique_severity: Optional[str] = None
    critique_revise_recommended: Optional[bool] = None
    critique_issue_count: Optional[int] = None
    critique_parse_failed: Optional[bool] = None

    def to_jsonl(self) -> str:
        d = self.__dict__.copy()
        return json.dumps(d, ensure_ascii=False)


def _ensure_dirs():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line + "\n")


def log_issue(topic: str, iter_n: int, kind: str, detail: str, snippet: str = "") -> None:
    iss = Issue(ts=time.time(), topic=topic, iter=iter_n, kind=kind,
                detail=detail, snippet=snippet)
    _append_jsonl(ISSUES_LOG, iss.to_jsonl())


def call_chat(message: str, max_calls: int = MAX_CALLS_PER_TOPIC,
              session_id: Optional[str] = None,
              timeout: int = TIMEOUT_SECONDS) -> dict:
    body = {"message": message, "max_calls": max_calls}
    if session_id:
        body["session_id"] = session_id
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def classify_response(topic: str, response: dict) -> tuple[list[str], int, int]:
    """Walk the response for issue signals. Returns
    (issues_logged_this_run, tool_calls_count, tool_errors_count)."""
    text = response.get("text", "") or ""
    calls = response.get("calls", []) or []
    issues = []

    # Tool-level errors
    tool_errors = 0
    for c in calls:
        result = (c.get("result") or "")
        if result.startswith("ERROR"):
            tool_errors += 1
            log_issue(topic, len(issues), "tool_error",
                      f"{c.get('name')}: {result[:200]}",
                      snippet=str(c.get("arg", ""))[:120])
            issues.append("tool_error")

    # Malformed tool call (parser saw <tool>...</tool> but couldn't parse args)
    if _MALFORMED_RE.search(text):
        log_issue(topic, len(issues), "tool_arg_malformed",
                  "runtime returned 'malformed tool call'", snippet=text[:300])
        issues.append("tool_arg_malformed")

    # Bracket-tool bleed (the vision provider-syntax bleed-through from base coder)
    if _BRACKET_TOOL_RE.search(text):
        log_issue(topic, len(issues), "bracket_bleed",
                  "model emitted [tool_use ...] coder-agent syntax",
                  snippet=_BRACKET_TOOL_RE.search(text).group(0))
        issues.append("bracket_bleed")

    # Repetition loop (4+ identical lines in a row)
    if _REPEAT_LINE_RE.search(text):
        m = _REPEAT_LINE_RE.search(text)
        log_issue(topic, len(issues), "repetition",
                  "4+ identical lines in a row", snippet=m.group(0)[:200])
        issues.append("repetition")

    # Stopped because we hit max_calls (model didn't converge)
    if response.get("reason_for_stop") == "max_calls":
        log_issue(topic, len(issues), "incomplete",
                  f"hit max_calls={MAX_CALLS_PER_TOPIC} without natural stop",
                  snippet=text[-300:])
        issues.append("incomplete")

    # Routed to bare path on a research prompt -- attack heuristic mis-fired
    if response.get("route") == "bare":
        log_issue(topic, len(issues), "safety_misroute",
                  "research prompt routed to bare/refusal path",
                  snippet=text[:200])
        issues.append("safety_misroute")

    return issues, len(calls), tool_errors


def extract_note(text: str) -> Optional[str]:
    """If the model emitted fs_write("research/<slug>.md|<content>"),
    extract <content> for our own copy. Otherwise return the natural
    text minus tool blocks."""
    # First try: did fs_write happen and have it?
    m = re.search(r'<tool>\s*fs_write\s*\(\s*"research/[^|]+\|([\s\S]+?)"\s*\)\s*</tool>', text)
    if m:
        return m.group(1).strip()
    # Fallback: the natural text after stripping tool blocks
    cleaned = re.sub(r"<tool>[\s\S]*?</tool>", "", text)
    cleaned = re.sub(r"<tool_result>[\s\S]*?</tool_result>", "", cleaned)
    cleaned = re.sub(r"\[tool_use\s+\w+[\s\S]*?\]", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or None


def promote_to_memory(summary: RunSummary, note_text: str) -> Optional[int]:
    """ε.2: When a research run succeeds, drop a one-line summary into
    memory.db so future chats can surface it via memory_search.
    Tagged source='research' (distinct from model/user inserts) so the
    dashboard can group these. Idempotent: skips if a research-tagged
    row already references this slug.

    Goes through the same secret-pattern check the user-facing
    memory_insert tool uses -- defense in depth. If the auto-generated
    summary somehow matches a secret shape, we log it as an issue and
    skip the insert rather than bypassing the guard."""
    if not summary.success:
        return None

    # Take the first paragraph of the note for the summary line; fall
    # back to a generic line if extraction failed.
    first_para = (note_text.split("\n\n", 1)[0].strip() if note_text else "")
    if not first_para:
        first_para = "(no narrative summary; see saved file)"
    teaser = first_para[:280]
    if len(first_para) > 280:
        teaser = teaser.rsplit(" ", 1)[0] + "..."

    line = (f"Research note on \"{summary.topic}\" -- saved at "
            f"research/{summary.slug}.md. Summary: {teaser}")

    # Cap to memory_insert's 500-char limit (the constant lives there;
    # we mirror it locally to avoid an import dance for one number).
    if len(line) > 500:
        line = line[:497] + "..."

    # Defense in depth: even though this is trusted code, run the same
    # secret-pattern check the model-facing tool uses, so an accidental
    # leak (e.g., topic name contained a path) is still caught.
    from .tools.memory_insert import looks_secret
    if looks_secret(line):
        log_issue(summary.topic, summary.iterations, "secret_in_summary",
                  "research summary text matched a secret pattern; "
                  "auto-promotion skipped",
                  snippet=line[:200])
        return None

    # Idempotency check.
    from .tools.memory_search import _conn
    c = _conn()
    existing = c.execute(
        "SELECT rowid FROM memory WHERE source = 'research' AND text LIKE ?",
        (f"%research/{summary.slug}.md%",),
    ).fetchone()
    c.close()
    if existing:
        return int(existing[0])

    # Use the underlying admin insert with source='research' so these
    # are tagged distinctly from user-issued ('model') memory writes.
    from .tools.memory_search import insert as _admin_insert
    return int(_admin_insert(line, source="research"))


def call_critique(message: str, response: str,
                  timeout: int = 120) -> Optional[dict]:
    """ζ.1: call /chat/critique. Returns the verdict dict on success,
    None on any failure. Never raises -- the autoresearch run must not
    abort because the critic call timed out or the endpoint is missing.
    """
    body = {"message": message, "response": response}
    req = urllib.request.Request(
        CRITIQUE_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def research_topic(topic: str, memorize: bool = True,
                   critic: bool = False) -> RunSummary:
    """Run the full autoresearch loop for one topic.

    memorize: if True (default), successful runs auto-promote a one-
    line summary into memory.db (source='research') via the same
    secret-pattern guard the model-facing memory_insert tool uses.

    critic: ζ.1. If True, after the note is written, run a second
    pass through /chat/critique and record the verdict in the run
    summary. LOGGED, not GATING -- a high-severity critique does NOT
    block memory promotion in this ship (that's a ζ.2 concern). The
    critique call is wrapped to NEVER raise; failures are logged as
    `kind=critique_error` and the run completes normally."""
    _ensure_dirs()
    slug = _slugify(topic)
    summary = RunSummary(topic=topic, slug=slug, started_ts=time.time())

    prompt = RESEARCH_PROMPT_TEMPLATE.format(topic=topic, slug=slug)

    try:
        response = call_chat(prompt, max_calls=MAX_CALLS_PER_TOPIC)
    except urllib.error.HTTPError as e:
        log_issue(topic, 0, "model_timeout", f"HTTP {e.code} from /chat")
        summary.finished_ts = time.time()
        summary.duration_s = summary.finished_ts - summary.started_ts
        summary.success = False
        summary.reason_for_stop = f"HTTP {e.code}"
        _append_jsonl(RUNS_LOG, summary.to_jsonl())
        return summary
    except Exception as e:
        log_issue(topic, 0, "model_timeout", f"{type(e).__name__}: {e}")
        summary.finished_ts = time.time()
        summary.duration_s = summary.finished_ts - summary.started_ts
        summary.success = False
        summary.reason_for_stop = "exception"
        _append_jsonl(RUNS_LOG, summary.to_jsonl())
        return summary

    issues, n_calls, n_errors = classify_response(topic, response)
    text = response.get("text", "") or ""
    note = extract_note(text)

    summary.finished_ts = time.time()
    summary.duration_s = summary.finished_ts - summary.started_ts
    summary.iterations = n_calls
    summary.tools_fired = sorted({
        c.get("name") for c in (response.get("calls") or [])
        if c.get("name")
    })
    summary.tool_errors = n_errors
    summary.issues = len(issues)
    summary.reason_for_stop = response.get("reason_for_stop", "?")
    summary.response_chars = len(text)

    if note and len(note) > 200:
        note_path = NOTES_DIR / f"{slug}.md"
        header = (
            f"# {topic}\n\n"
            f"_Generated by Azriel autoresearch on "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}._\n\n"
            f"_Tools fired: {', '.join(summary.tools_fired) or '(none)'}._\n\n"
            f"---\n\n"
        )
        note_path.write_text(header + note + "\n")
        summary.note_path = str(note_path)
        # Per advisor: a >200-char extracted note IS success. Tool errors
        # and "incomplete" don't reject -- they stay in issues.jsonl as
        # signal for triage but don't block memorization. The whole point
        # of the natural-text fallback in extract_note is that we get a
        # useful note even when the model didn't fire fs_write or hit
        # max_calls.
        # safety_misroute IS load-bearing: a research prompt that hits
        # the bare/refusal path produces refusal text >200 chars and
        # would otherwise be promoted to memory.db tagged source=research.
        # That row would then surface via memory_search in future chats.
        summary.success = "safety_misroute" not in issues
    else:
        log_issue(topic, n_calls, "incomplete",
                  f"no usable note extracted (text {len(text)} chars)",
                  snippet=text[-300:])
        summary.success = False

    # ε.2: auto-promote successful runs into memory so future chats
    # can surface them via memory_search.
    if memorize and summary.success and note:
        try:
            summary.memory_rowid = promote_to_memory(summary, note)
        except Exception as e:
            log_issue(topic, summary.iterations, "memory_promotion_failed",
                      f"{type(e).__name__}: {e}", snippet=str(e)[:200])

    # ζ.1: optional critic pass over the note. LOGGED, not GATING.
    if critic and summary.success and note:
        verdict = call_critique(topic, note)
        if verdict is None:
            log_issue(topic, summary.iterations, "critique_error",
                      "critique call failed (timeout / parse / endpoint)",
                      snippet="")
        else:
            summary.critique_severity = verdict.get("severity")
            summary.critique_revise_recommended = bool(
                verdict.get("revise_recommended"))
            issue_count = (
                len(verdict.get("factual_issues", []) or [])
                + len(verdict.get("scripture_issues", []) or [])
                + len(verdict.get("doctrinal_issues", []) or [])
                + len(verdict.get("internal_contradictions", []) or [])
            )
            summary.critique_issue_count = issue_count
            summary.critique_parse_failed = bool(
                verdict.get("parse_failed"))
            if verdict.get("parse_failed"):
                log_issue(topic, summary.iterations, "critique_parse_failed",
                          "critic returned non-JSON output",
                          snippet=str(verdict.get("raw", ""))[:300])
            elif verdict.get("severity") in ("medium", "high"):
                log_issue(topic, summary.iterations, "critique_flag",
                          f"severity={verdict.get('severity')} "
                          f"revise={verdict.get('revise_recommended')} "
                          f"issues={issue_count}",
                          snippet=json.dumps({
                              "factual": verdict.get("factual_issues"),
                              "scripture": verdict.get("scripture_issues"),
                              "doctrinal": verdict.get("doctrinal_issues"),
                              "contradictions": verdict.get(
                                  "internal_contradictions"),
                          })[:300])

    _append_jsonl(RUNS_LOG, summary.to_jsonl())
    return summary


# ===== queue management =====

def queue_load() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    items = []
    for line in QUEUE_FILE.open():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            pass
    return items


def queue_save(items: list[dict]) -> None:
    _ensure_dirs()
    with QUEUE_FILE.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def queue_next_pending() -> Optional[dict]:
    for it in queue_load():
        if it.get("status", "pending") == "pending":
            return it
    return None


def queue_mark(topic: str, status: str) -> None:
    items = queue_load()
    for it in items:
        if it.get("topic") == topic:
            it["status"] = status
            it["last_updated"] = time.time()
    queue_save(items)
