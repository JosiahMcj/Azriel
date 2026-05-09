"""Agent-mode loop: plan / act / observe.

Stateful multi-turn agent that decomposes a goal into tool calls and
runs them one step at a time. State is in-memory (process-scoped); a
server restart drops pending tasks. Persistence across restarts is a
future bell (theta.4 memory integration).

The planner emits exactly one line per turn:

    STEP: tool_name("arg") -- execute a tool, observe, replan
    DONE: <one-line reason> -- goal achieved
    ABORT: <one-line reason> -- cannot proceed

Hard rules:
  - The original goal runs through is_attack_prompt before any model
    invocation. If matched, the task starts in ABORTED state.
  - Each STEP's tool_arg also runs through is_attack_prompt. A matching
    arg aborts the task.
  - MAX_STEPS = 10. Tasks that hit the cap end in CAPPED state.
  - MAX_TASKS = 50 in memory; oldest evicted FIFO.
  - Tasks expire 24h after creation.

This module does not edit locked files. It calls into the
existing tool registry and the existing model/tokenizer; it does not
touch loop.py, lti.py, tool_heads.py, or model.py.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .runtime import is_attack_prompt
from .tools import REGISTRY, get_active_registry

CONSTITUTION_PATH = Path.home() / ".azriel" / "AZRIEL_CONSTITUTION_SYSTEM.txt"

MAX_STEPS = 10
MAX_TASKS = 50
TASK_TTL_SECONDS = 24 * 60 * 60

# Output grammar from the planner. Strict; anything else is treated as
# a parse failure and turned into an ABORT.
_LINE_RE = re.compile(r"^\s*(STEP|DONE|ABORT)\s*:\s*(.+?)\s*$",
                      re.IGNORECASE | re.MULTILINE)
# STEP arg parser: tool_name("arg") or tool_name('arg') or tool_name(arg)
_STEP_CALL_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(.*?)\s*\)\s*$",
                           re.DOTALL)
_QUOTED_RE = re.compile(r'^["\'](.*)["\']$', re.DOTALL)


@dataclass
class AgentStep:
    step_no: int
    kind: str # "STEP" | "DONE" | "ABORT" | "PARSE_FAIL"
    raw: str
    tool_name: Optional[str] = None
    tool_arg: Optional[str] = None
    observation: Optional[str] = None
    error: Optional[str] = None
    ts: int = 0


@dataclass
class AgentTask:
    task_id: str
    goal: str
    session_id: Optional[str]
    status: str # "running" | "done" | "aborted" | "capped" | "error"
    steps: list[AgentStep] = field(default_factory=list)
    created_at: int = 0
    expires_at: int = 0
    last_reason: Optional[str] = None
    # per-task capability scope set by the /agent
    # permissions UI. None means "all currently registered tools".
    allowed_tools: Optional[list[str]] = None


_LOCK = threading.RLock()
_TASKS: dict[str, AgentTask] = {}


def _now() -> int:
    return int(time.time())


def _evict_if_full() -> None:
    if len(_TASKS) <= MAX_TASKS:
        return
    # Drop oldest by created_at
    by_age = sorted(_TASKS.values(), key=lambda t: t.created_at)
    drop = len(_TASKS) - MAX_TASKS
    for t in by_age[:drop]:
        _TASKS.pop(t.task_id, None)


def cleanup_expired() -> int:
    now = _now()
    with _LOCK:
        expired = [tid for tid, t in _TASKS.items() if t.expires_at < now]
        for tid in expired:
            _TASKS.pop(tid, None)
        return len(expired)


def _tool_names(allowed: Optional[list[str]] = None) -> list[str]:
    """Names of tools the planner may call. Uses the 'active' subset
    so disabled connectors don't show up as available. If `allowed`
    is provided, only tools in that list are returned (for per-task
    capability scoping driven by the agent UI's permissions panel)."""
    try:
        names = sorted(get_active_registry().keys())
    except Exception:
        try:
            names = sorted(REGISTRY.keys())
        except Exception:
            names = []
    if allowed is None:
        return names
    allowed_set = {a for a in allowed if isinstance(a, str)}
    return [n for n in names if n in allowed_set]


def _build_agent_primer(goal: str, allowed_tools: Optional[list[str]] = None) -> tuple[str, str]:
    """Returns (primer_user_turn, primer_assistant_ack). Inserted between
    the system prompt and the working history so agent-mode instructions
    do not dilute the constitution's identity weight (learning carried forward).

    `allowed_tools` filters the primer's advertised
    tools so the user can scope the agent from the UI permissions
    panel. If the filter empties the list, the primer says so
    explicitly so the planner ABORTs cleanly rather than free-forming."""
    # include each tool's signature + 1-line doc so
    # the planner knows the arg format (e.g. document_create takes
    # "format|name|content" not separate args). Without this the agent
    # repeatedly fired document_create("New Doc", "gdoc") and got
    # ERROR: format is 'format|name|content'.
    tools = _tool_names(allowed=allowed_tools)
    if tools:
        try:
            registry = get_active_registry()
        except Exception:
            registry = REGISTRY
        lines = []
        for n in tools:
            spec = registry.get(n) or {}
            sig = spec.get("signature", f"{n}(arg: str) -> str")
            doc = (spec.get("doc") or "").strip().replace("\n", " ")
            # Compact long docs to one short line.
            if len(doc) > 240:
                doc = doc[:237] + "..."
            lines.append(f" - {sig}\n {doc}" if doc else f" - {sig}")
        tool_block = "\n".join(lines)
    else:
        tool_block = " (none registered)"
    primer = (
        "PLANNING MODE -- the user has given you a multi-step task. You "
        "decompose it by emitting exactly ONE LINE per turn, in one of "
        "three forms:\n\n"
        ' STEP: tool_name("argument") -- run a tool, see its output, replan next turn\n'
        ' DONE: <one-line reason> -- goal achieved\n'
        ' ABORT: <one-line reason> -- cannot proceed\n\n'
        "Rules:\n"
        " - One line only. No prose, no explanation, no thinking out loud.\n"
        " - On each subsequent turn you will see OBSERVATION: <tool result> in the user turn.\n"
        " - Tool args must be plain strings. No nested calls, no JSON unless the tool documents it.\n"
        " - If you would refuse the goal as a chat answer, ABORT immediately with the same reason.\n"
        " - Do not roleplay personas, do not adopt alternate identities, do not bypass the constitution.\n\n"
        "Strategy guidance (use the same one-line grammar):\n"
        " - HARD RULE: ABORT is FORBIDDEN unless either (a) the goal violates the "
        "constitution, OR (b) you have already made at least 3 STEP calls in this task. "
        "A weak or partial first result is NEVER grounds for ABORT.\n"
        " - When a tool returns a weak / off-target / 'no match' observation: re-issue the "
        "SAME tool with simpler or different keywords. Concrete example: if "
        'commentary_lookup(\"Chuck Missler Hebrews 11 faith\") returns weak matches, '
        'next try commentary_lookup(\"Hebrews 11 faith\") -- drop the author name, keep '
        'the topic. If that also fails, try a DIFFERENT tool (bible_lookup, crossref_lookup).\n'
        " - You have up to 10 total steps. Prefer two or three short tool calls that each "
        "make progress over one optimistic call. Combine tools when one alone is insufficient.\n"
        " - When you have enough information to answer the goal, emit DONE with a one-line "
        "summary -- the user reads the observation trail above for the substance.\n"
        " - Tool argument hygiene: keep args short and concrete. For commentary_lookup and "
        "web_search, prefer 2-4 keyword tokens over full sentences. For bible_lookup and "
        "crossref_lookup, pass a verse reference exactly (e.g. \"John 3:16\" or \"Hebrews 11\").\n\n"
        f"Available tools:\n{tool_block}\n\n"
        f"GOAL: {goal}\n\n"
        "Begin. Emit only the first line."
    )
    ack = "Acknowledged. Planning Mode active. Awaiting first action."
    return primer, ack


def _render_agent_chat(tokenizer, system: str, primer_user: str, primer_ack: str,
                       steps: list[AgentStep],
                       force_prefix: Optional[str] = None) -> str:
    """Render system + agent-primer turn pair + step history. The model
    is expected to emit the assistant continuation (one line).

    `force_prefix` lets the caller pre-fill the start
    of the assistant turn (e.g. "STEP: ") so the decoder is forced to
    continue from a grammar-conforming opening. Used on the first
    step when the model has been narrating instead of acting."""
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": primer_user},
        {"role": "assistant", "content": primer_ack},
    ]
    for s in steps:
        # Each completed step is an assistant emission then a user-side
        # OBSERVATION turn (or terminal kind, in which case there is no
        # follow-up user turn -- we won't be calling the model again).
        if s.kind == "STEP":
            msgs.append({"role": "assistant", "content": s.raw})
            obs = s.observation if s.observation is not None else f"[error: {s.error}]"
            # Truncate observations to keep the context bounded.
            if len(obs) > 1500:
                obs = obs[:1500] + "...[truncated]"
            msgs.append({"role": "user", "content": f"OBSERVATION: {obs}"})
    try:
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    if force_prefix:
        rendered = rendered + force_prefix
    return rendered


def _parse_planner_emission(text: str) -> tuple[str, Optional[str], Optional[str], str]:
    """Returns (kind, tool_name, tool_arg, raw_line). kind is one of
    STEP / DONE / ABORT / PARSE_FAIL.

    Lenient parsing: prefers the STEP/DONE/ABORT grammar, but if the
    model emits a bare `tool_name("arg")` call without the STEP:
    prefix we accept it as a STEP rather than failing parse. The
    runtime is more tolerant than the primer text says; this catches
    a common small-model formatting slip without weakening safety
    (the emission is still passed through is_attack_prompt before
    exec)."""
    # Prefer the explicit grammar.
    m = _LINE_RE.search(text)
    if m:
        kind = m.group(1).upper()
        rest = m.group(2).strip()
        if kind != "STEP":
            return (kind, None, None, f"{kind}: {rest}")
        cm = _STEP_CALL_RE.match(rest)
        if cm:
            name = cm.group(1)
            raw_arg = cm.group(2)
            qm = _QUOTED_RE.match(raw_arg)
            arg = qm.group(1) if qm else raw_arg
            return ("STEP", name, arg, f"STEP: {name}({raw_arg})")
        return ("PARSE_FAIL", None, None, f"STEP: {rest}")
    # Fallback: bare `tool_name("arg")` line, no STEP: prefix.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cm = _STEP_CALL_RE.match(line)
        if cm:
            name = cm.group(1)
            raw_arg = cm.group(2)
            qm = _QUOTED_RE.match(raw_arg)
            arg = qm.group(1) if qm else raw_arg
            return ("STEP", name, arg, f"STEP: {name}({raw_arg})")
        # First non-empty line that's not a tool call -> bail to PARSE_FAIL
        return ("PARSE_FAIL", None, None, line[:400])
    return ("PARSE_FAIL", None, None, text.strip()[:400])


_PROSE_CALL_RE = re.compile(
    r'`?([a-z_][a-z_0-9]*)`?\s*\(\s*"([^"]+)"\s*\)'
)


# last-resort goal->tool router. Fires when the
# planner has narrated and the prose extractor found no tool-call
# pattern. Maps goal keywords to a sensible default tool + arg so a
# concrete first step still gets taken instead of a hard ABORT.
# Order matters: more specific phrases win over generic ones.
# The arg template can include {goal}, {content}, or {name}; those are
# substituted from the goal text at routing time ().
_GOAL_KEYWORD_RULES = [
    # (substring_list, tool_name, arg_template)
    (["google doc", "google docs"], "document_create", "docx|{name}|{content}"),
    (["create a doc", "make a doc", "create a document",
      "make a document", "write a document", "new document"],
     "document_create", "docx|{name}|{content}"),
    (["create a pdf", "make a pdf", "write a pdf"], "pdf_create", "{name}|{content}"),
    (["create a presentation", "make slides", "make a slide deck"],
     "document_create", "pptx|{name}|{content}"),
    (["create a spreadsheet", "make a csv"], "document_create",
     "xlsx|{name}|name,value"),
    (["weather", "forecast", "temperature"], "weather", "{goal}"),
    (["search the web", "search online", "find an article",
      "look up online", "google "], "web_search", "{goal}"),
    (["look up", "what does", "what is"], "bible_lookup", "{goal}"),
]


# Phrases that introduce literal content the user wants written into
# the doc. Longer alternatives go FIRST so "with text saying X" matches
# the full intro and yields content "X" instead of "saying X".
_CONTENT_INTRO_RES = [
    re.compile(r"""(?:put in(?:\s+(?:that says|saying))?|with the text|with text(?:\s+(?:that says|saying))?|that says|that reads|with content|saying)\s*[:\-]?\s*["“‘']?(.+?)["”’']?\s*$""", re.IGNORECASE | re.DOTALL),
    re.compile(r"""containing\s+["“‘'](.+?)["”’']""", re.IGNORECASE | re.DOTALL),
]


def _extract_content_and_name(goal: str) -> tuple[str, str]:
    """Pull a content body and a sensible filename out
    of a free-form goal. "open a google doc on my computer and put in
    'HI this is Azriel'" -> content="HI this is Azriel", name="note".
    Falls back to (empty, "note") when no content phrase is found."""
    content = ""
    for rx in _CONTENT_INTRO_RES:
        m = rx.search(goal or "")
        if m:
            content = m.group(1).strip()
            # Trim trailing close-paren/period collateral.
            content = content.rstrip(" .;:)\"'")
            break
    # Pipe is the document_create field separator; strip any pipes the
    # user happened to include so we don't smuggle extra fields.
    content = content.replace("|", "/")
    # Default name. If the goal mentions a noun like "journal" /
    # "letter" / "note", we COULD fancy-extract; for now keep simple.
    name = "note"
    return content, name


def _infer_tool_from_goal(goal: str, valid: set[str]) -> Optional[tuple[str, str]]:
    """Keyword-route the goal to a default tool + arg
    when the planner has emitted prose with no concrete call. Used as
    a last-resort recovery before the agent aborts.

    arg templates can carry {content} and {name}
    placeholders, which are filled by _extract_content_and_name from
    the goal text -- so "open a google doc and put in 'X'" actually
    writes X into the file instead of leaving it empty."""
    g = (goal or "").lower()
    content, name = _extract_content_and_name(goal or "")
    for keywords, tool, arg_tpl in _GOAL_KEYWORD_RULES:
        if any(k in g for k in keywords) and tool in valid:
            arg = (
                arg_tpl
                .replace("{goal}", goal)
                .replace("{content}", content)
                .replace("{name}", name)
            )
            return (tool, arg)
    return None


def _extract_call_from_prose(text: str) -> Optional[tuple[str, str]]:
    """The planner sometimes narrates ('I will create a
    doc with `document_create("...")` and then ...') instead of
    obeying the strict one-line grammar. If the prose contains a
    registered-tool name followed by ("arg") -- with or without
    backtick wrapping -- pull out the FIRST such call and treat it as
    the recovered STEP."""
    try:
        valid = set(get_active_registry().keys())
    except Exception:
        try:
            valid = set(REGISTRY.keys())
        except Exception:
            return None
    for m in _PROSE_CALL_RE.finditer(text or ""):
        name = m.group(1)
        if name in valid:
            return (name, m.group(2))
    return None


def _exec_tool(name: str, arg: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (observation, error). Tools come from REGISTRY where each
    value is a spec dict {fn, signature, doc}; we resolve fn and call
    it with the planner's argument string."""
    try:
        active = get_active_registry()
    except Exception:
        active = REGISTRY
    spec = active.get(name)
    if spec is None:
        return (None, f"unknown tool: {name}")
    fn = spec.get("fn") if isinstance(spec, dict) else (spec if callable(spec) else None)
    if not callable(fn):
        return (None, f"tool {name} has no callable fn")
    try:
        result = fn(arg)
        if result is None:
            return (None, f"tool {name} returned None")
        if not isinstance(result, str):
            result = str(result)
        return (result, None)
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")


def _read_constitution() -> str:
    try:
        return CONSTITUTION_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return "I am Azriel, a language-model tool grounded in Scripture."


_BIBLE_REF_RE = re.compile(
    r"\b(?:[1-3]\s+)?(?:Genesis|Exodus|Leviticus|Numbers|Deuteronomy|"
    r"Joshua|Judges|Ruth|Samuel|Kings|Chronicles|Ezra|Nehemiah|Esther|"
    r"Job|Psalm|Psalms|Proverbs|Ecclesiastes|Song|Isaiah|Jeremiah|"
    r"Lamentations|Ezekiel|Daniel|Hosea|Joel|Amos|Obadiah|Jonah|Micah|"
    r"Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|Matthew|Mark|"
    r"Luke|John|Acts|Romans|Corinthians|Galatians|Ephesians|Philippians|"
    r"Colossians|Thessalonians|Timothy|Titus|Philemon|Hebrews|James|"
    r"Peter|Jude|Revelation)\s+\d+(?::\d+(?:-\d+)?)?\b"
)


def _extract_citations(task: AgentTask) -> list[str]:
    """Pull Bible citations out of the observation trail. Simple pattern
    match; misses obscure abbreviations but catches the canonical
    'Book Chapter:Verse' shape."""
    seen: dict[str, None] = {}
    for s in task.steps:
        if s.observation:
            for m in _BIBLE_REF_RE.findall(s.observation):
                seen[m.strip()] = None
    return list(seen.keys())[:8]


def _on_task_terminal(task: AgentTask) -> None:
    """write a one-line summary to long-term memory
    when an agent task completes successfully. Aborted / capped /
    parse-fail tasks are NOT memorized (noise + we don't want safety
    refusals leaking into recall)."""
    if task.status != "done":
        return
    try:
        from .tools.memory_search import insert as _memory_insert
    except Exception:
        return
    citations = _extract_citations(task)
    summary_parts = [
        f"agent task {task.task_id}",
        f"goal: {task.goal[:200]}",
        f"resolved: {(task.last_reason or '').strip()[:200] or '(no reason)'}",
    ]
    if citations:
        summary_parts.append(f"citations: {', '.join(citations)}")
    summary = " | ".join(summary_parts)
    if len(summary) > 600:
        summary = summary[:600] + "..."
    try:
        _memory_insert(summary, source="agent")
    except Exception:
        # Memory is best-effort; never let a failed insert mark the
        # task as non-terminal.
        pass


def start_task(model, tokenizer, goal: str,
               session_id: Optional[str] = None,
               allowed_tools: Optional[list[str]] = None) -> AgentTask:
    """Create a new agent task. Runs is_attack_prompt on the goal first;
    a match yields a task in ABORTED state with no model invocation.

    `allowed_tools` () is an optional whitelist set by the
    permissions panel in /agent. When provided, only those tools appear
    in the planner's primer; outside-list tools simply don't exist for
    this task."""
    task_id = uuid.uuid4().hex[:12]
    now = _now()
    task = AgentTask(
        task_id=task_id, goal=goal, session_id=session_id,
        status="running", steps=[], created_at=now,
        expires_at=now + TASK_TTL_SECONDS,
        allowed_tools=allowed_tools,
    )
    if is_attack_prompt(goal):
        task.status = "aborted"
        task.last_reason = "attack-pattern in goal; agent refused at planner gate"
        task.steps.append(AgentStep(
            step_no=0, kind="ABORT", raw="ABORT: attack pattern matched goal",
            error="is_attack_prompt matched goal",
            ts=now,
        ))
    with _LOCK:
        _TASKS[task_id] = task
        _evict_if_full()
    if task.status == "running":
        # Take the first step synchronously so the caller gets a meaningful
        # initial state rather than just an empty plan.
        step_task(model, tokenizer, task_id)
    return _TASKS[task_id]


def step_task(model, tokenizer, task_id: str) -> AgentTask:
    """Advance one step. No-op if task is terminal."""
    with _LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            raise KeyError(f"task not found: {task_id}")
    if task.status != "running":
        return task
    if len(task.steps) >= MAX_STEPS:
        task.status = "capped"
        task.last_reason = f"MAX_STEPS ({MAX_STEPS}) reached"
        return task

    # Lazy import to keep module-load fast (mlx_lm load is heavy).
    from mlx_lm import generate as _generate
    from mlx_lm.sample_utils import make_sampler

    system = _read_constitution()
    primer_user, primer_ack = _build_agent_primer(
        task.goal, allowed_tools=task.allowed_tools,
    )
    # on the very first step, force the assistant
    # turn to begin with "STEP: " so the model continues from a
    # grammar-conforming opening. Qwen3-Coder strongly prefers
    # narrating ("I will check what tools are available...") over
    # the strict one-line grammar; pre-filling the prefix removes
    # that escape hatch. Subsequent steps don't force, so the model
    # can still emit DONE / ABORT once it has observations to
    # summarize.
    force_prefix = "STEP: " if len(task.steps) == 0 else None
    prompt = _render_agent_chat(
        tokenizer, system, primer_user, primer_ack, task.steps,
        force_prefix=force_prefix,
    )
    sampler = make_sampler(temp=0.3)
    raw = _generate(model, tokenizer, prompt=prompt, max_tokens=180,
                    sampler=sampler)
    if force_prefix:
        # The forced prefix is part of the input, not the model's
        # emission -- but the parser expects the full STEP: line. Add
        # it back to the front so _parse_planner_emission sees it.
        raw = force_prefix + raw
    kind, tool_name, tool_arg, raw_line = _parse_planner_emission(raw)
    step_no = len(task.steps) + 1
    now = _now()

    if kind == "DONE":
        task.steps.append(AgentStep(step_no=step_no, kind="DONE", raw=raw_line, ts=now))
        task.status = "done"
        task.last_reason = raw_line[len("DONE:"):].strip()
        _on_task_terminal(task)
        return task
    if kind == "ABORT":
        # Early-abort guard (strategy fix). Default is to
        # TRUST the model's abort -- safety/refusal aborts use varied
        # vocabulary and we don't want to override them. We only rewrite
        # when the abort looks like a give-up on a hard search,
        # specifically: <2 STEP attempts AND the reason matches a
        # known give-up pattern (e.g. "no match", "could not find").
        n_step_attempts = sum(1 for s in task.steps if s.kind == "STEP")
        abort_reason = raw_line[len("ABORT:"):].strip().lower()
        give_up_patterns = (
            "no match", "no matches", "no direct match", "no result",
            "no results", "could not find", "couldn't find", "did not find",
            "didn't find", "no information", "unable to find",
            "search returned", "the search returned",
            "no relevant", "not relevant", "weren't relevant",
        )
        is_give_up = any(p in abort_reason for p in give_up_patterns)
        if n_step_attempts < 2 and is_give_up:
            override_obs = (
                "[runtime override] Your ABORT was rejected as a premature "
                f"give-up. You have only made {n_step_attempts} STEP call(s); "
                "the strategy guidance requires re-issuing the tool with "
                "different keywords or trying a different tool before "
                "abandoning the search. Issue another STEP."
            )
            task.steps.append(AgentStep(
                step_no=step_no, kind="STEP", raw=raw_line,
                tool_name=None, tool_arg=None,
                observation=override_obs,
                error="early-abort rewritten by runtime guard (give-up pattern)",
                ts=now,
            ))
            return task
        task.steps.append(AgentStep(step_no=step_no, kind="ABORT", raw=raw_line, ts=now))
        task.status = "aborted"
        task.last_reason = raw_line[len("ABORT:"):].strip()
        return task
    if kind == "PARSE_FAIL":
        # recover instead of immediately aborting.
        # The most common PARSE_FAIL is the model emitting a planning
        # paragraph ("I will create a doc and then read it...") on the
        # first step instead of obeying STEP/DONE/ABORT grammar. Try
        # to recover by:
        # 1. Scanning the prose for a tool name + bare arg pattern;
        # if found, promote to a real STEP.
        # 2. Otherwise, log the parse-fail as an OBSERVATION-style
        # step and re-prompt once with a stricter nudge.
        # Only retry on the FIRST step; later parse-fails after real
        # progress mean something else is wrong and we want to fail.
        first_step = len(task.steps) == 0
        # on the FIRST step, prefer goal-keyword
        # routing OVER prose-extracted calls when both fire. Reason:
        # the model often mentions a generic tool (fs_write) in its
        # narration even when the goal screams for a specific one
        # (document_create on "google doc"). The keyword router is
        # higher-signal because it reads the user's intent directly.
        # On later steps we keep the original order (prose first) so
        # the model can still pick its own tool to follow up.
        try:
            _valid = set(get_active_registry().keys())
        except Exception:
            _valid = set(REGISTRY.keys())
        if task.allowed_tools is not None:
            _valid = _valid.intersection(set(task.allowed_tools))
        recovered = None
        if first_step:
            recovered = _infer_tool_from_goal(task.goal, _valid)
        if recovered is None:
            # Try prose-extract: pull the FIRST registered-tool call out
            # of the prose.
            recovered = _extract_call_from_prose(raw_line)
        if recovered is None and first_step:
            # Belt-and-suspenders: re-try the keyword router in case
            # _valid changed (it didn't, but harmless).
            recovered = _infer_tool_from_goal(task.goal, _valid)
        if recovered is not None:
            r_name, r_arg = recovered
            task.steps.append(AgentStep(
                step_no=step_no, kind="STEP",
                raw=f'STEP: {r_name}("{r_arg}")',
                tool_name=r_name, tool_arg=r_arg,
                observation=None,
                error="recovered from PARSE_FAIL via prose-extract or goal-keyword route",
                ts=now,
            ))
            # Execute the recovered call so the loop can continue.
            obs, err = _exec_tool(r_name, r_arg)
            task.steps[-1].observation = obs
            if err:
                task.steps[-1].error = (
                    (task.steps[-1].error or "") + f"; {err}"
                )
            return task
        if first_step:
            # Log the failed prose as a PARSE_FAIL step but DON'T
            # terminate -- inject a nudge as the next planning hint
            # by appending a synthetic OBSERVATION turn so the
            # _render_agent_chat picks it up on next step.
            task.steps.append(AgentStep(
                step_no=step_no, kind="STEP",
                raw=raw_line,
                tool_name=None, tool_arg=None,
                observation=(
                    "[runtime nudge] Your previous output was a "
                    "planning paragraph, not the strict one-line "
                    "grammar. OUTPUT EXACTLY ONE LINE next, in one "
                    "of these forms (no prose, no preamble):\n"
                    ' STEP: tool_name("argument")\n'
                    " DONE: <one-line reason>\n"
                    " ABORT: <one-line reason>\n"
                    "Pick the first concrete tool you would call and "
                    "emit only that line."
                ),
                error="planner emitted prose; nudging once before abort",
                ts=now,
            ))
            return task
        # Not first step OR already nudged once -- abort cleanly.
        task.steps.append(AgentStep(
            step_no=step_no, kind="PARSE_FAIL", raw=raw_line,
            error="planner emission did not match STEP/DONE/ABORT grammar",
            ts=now,
        ))
        task.status = "aborted"
        task.last_reason = "planner emission unparseable; aborting to fail safe"
        return task

    # kind == STEP: validate tool arg before exec.
    if tool_arg and is_attack_prompt(tool_arg):
        task.steps.append(AgentStep(
            step_no=step_no, kind="ABORT", raw=raw_line,
            tool_name=tool_name, tool_arg=tool_arg,
            error="attack pattern matched tool argument",
            ts=now,
        ))
        task.status = "aborted"
        task.last_reason = "attack-pattern in tool argument; agent refused"
        return task

    # enforce per-task capability scope. The primer
    # already only advertises allowed tools, but the model may
    # hallucinate a name or replay one from training; reject those
    # before we hit the live tool fn so disabled categories stay off.
    if (
        task.allowed_tools is not None
        and tool_name
        and tool_name not in task.allowed_tools
    ):
        task.steps.append(AgentStep(
            step_no=step_no, kind="STEP", raw=raw_line,
            tool_name=tool_name, tool_arg=tool_arg,
            observation=(
                f"[runtime override] tool '{tool_name}' is disabled "
                f"in this task's permissions. Available: "
                f"{', '.join(task.allowed_tools) or '(none)'}. "
                f"Either pick an allowed tool or DONE / ABORT."
            ),
            error="tool not in allowed_tools whitelist",
            ts=now,
        ))
        return task

    obs, err = _exec_tool(tool_name or "", tool_arg or "")
    task.steps.append(AgentStep(
        step_no=step_no, kind="STEP", raw=raw_line,
        tool_name=tool_name, tool_arg=tool_arg,
        observation=obs, error=err, ts=now,
    ))
    return task


def get_task(task_id: str) -> Optional[AgentTask]:
    with _LOCK:
        return _TASKS.get(task_id)


def list_tasks(session_id: Optional[str] = None) -> list[AgentTask]:
    with _LOCK:
        out = list(_TASKS.values())
    if session_id is not None:
        out = [t for t in out if t.session_id == session_id]
    out.sort(key=lambda t: t.created_at, reverse=True)
    return out


def to_dict(task: AgentTask) -> dict:
    """Server-friendly view: dataclass -> dict with steps as list of dicts."""
    d = asdict(task)
    return d
