"""Tool-using inference runtime.

Wraps mlx_lm.generate so that when the model emits a `<tool>NAME(ARG)</tool>`
sequence, we intercept BEFORE the tokens are returned to the caller, execute
the actual tool, inject `<tool_result>...</tool_result>` into the assistant
turn, and continue generation. Up to `max_calls` rounds per turn; final
output is the full assistant turn including all interleaved calls/results.

This deliberately uses the v0.7 textual protocol (autoregressive tool calls
in plain tokens) rather than the mid-forward heads. Trade-off: a few
extra forward passes per tool call vs zero retraining required to ship.
"""
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from .tools import REGISTRY, call as tool_call, system_prompt_block

CONSTITUTION_PATH = Path.home() / ".azriel" / "AZRIEL_CONSTITUTION_SYSTEM.txt"

# In-context tool primer delivered as a pre-user turn rather than mixed into
# the system prompt. Keeps system prompt = constitution verbatim so v0.6.0's
# identity refusals (which the base-model probe showed are held entirely by
# constitutional weight) are not diluted by tool instructions.
FORMATTING_DIRECTIVE = (
    "FORMATTING — readability over density. The user is reading on a "
    "phone or compact dashboard, not a printed page.\n"
    " - Keep paragraphs short: 2-4 sentences each, blank line between.\n"
    " - When walking through a list (verses in a passage, points in a "
    "sermon, steps in prayer, items in an answer), use bullet points "
    "or short numbered headings — never a single long paragraph.\n"
    " - When quoting Scripture, put the verse on its own line, "
    "indented with `> ` (markdown blockquote), followed by the "
    "reference. Don't bury verses inline in prose.\n"
    " - Bold (**…**) the key term being explained, then unpack on a "
    "new line.\n"
    " - One wall-of-text paragraph longer than ~6 sentences is almost "
    "always wrong; break it up at natural pauses.\n"
    " - Skip filler openers (\"That's a great question\", \"Of course\", "
    "\"Certainly\") — start with the answer.\n"
    " - For a one-line factual answer, give the one line. Don't pad it "
    "into a paragraph."
)


STYLE_DIRECTIVES = {
    "conviction": (
        "Voice: speak from your biblically-based frame -- direct, scripture-grounded, "
        "with conviction. This is the default and your real voice."
    ),
    "scholar": (
        "Voice: when a topic is contested across Christian traditions, briefly "
        "survey the relevant traditions evenhandedly first, then close with "
        "your own view clearly labeled. Identity unchanged -- you are still "
        "Azriel; this is a teaching/scholarly voice, not a different person."
    ),
    "pastoral": (
        "Voice: shorter, gentler, more pastoral. Lead with care for the person, "
        "then scripture. Same biblically-based identity, softer pacing. Avoid heavy "
        "doctrinal exposition unless the user explicitly asks."
    ),
}

# Persona "voice cards" -- short descriptors of how each preset answers
# AND a sample opener line that shows the voice in action. Sample
# openers ship into the primer as concrete imitable examples; the
# trained LoRA bias is strong enough that abstract descriptors
# ("light, playful") get washed out at moderate weights, but a
# concrete opener gives the model a phrase to anchor on.
# Identity and doctrinal stance are NEVER touched here (those live
# in the constitution and the bare-chat refusal path). Persona is
# delivery only.
PERSONA_CARDS = {
    "funny": "light, playful, willing to drop a tasteful joke. Never irreverent toward Scripture or God. Sample opener: \"Here's the part nobody puts on a coffee mug...\"",
    "ecstatic": "joy-forward, exclamation marks, occasional emoji (🔥 ✨ 🙌), celebratory cadence. Sample opener: \"Oh, this one is GOOD --\"",
    "personal": "first-name warmth, direct address, 'I see you' tone. Pastoral check-in feel. Sample opener: \"Hey -- I want to walk through this with you.\"",
    "somber": "quiet, weighty, careful pacing -- for grief, lament, hard questions. Sample opener: \"This is heavy ground. Let me sit with it.\"",
    "professional": "crisp, structured, no slang. Suitable for study notes / sermon prep. Sample opener: \"Three things to anchor here:\"",
    "interesting": "surfaces unusual angles -- history, etymology, typological connections, lesser-known commentary. Sample opener: \"Here's something most readers miss --\"",
    "nurturing": "gentle, encouraging, slow. Affirms before correcting. Sample opener: \"First, take a breath. You're asking the right question.\"",
    "direct": "short sentences. No hedging. Says what it means. Sample opener: \"Short answer:\"",
    "poetic": "rhythmic, image-rich, scripture-cadenced -- echoes the Psalter and the prophets. Sample opener: \"Grace is breath in dry lungs --\"",
    "encouraging": "builds the user up, names what they did well, forward-looking. Sample opener: \"Good question -- and you're already closer to the answer than you think.\"",
}


def _build_persona_directive(persona_mix: dict | None) -> str:
    """Compose a 'Voice mix: X% A, Y% B' directive from a {preset: pct} dict.
    Returns empty string if no preset is meaningfully active (>=10%).
    Identity / doctrine are explicitly preserved in the closing line.

    earlier wording was too descriptive ("interesting:
    surfaces unusual angles") and the model read it as info, not
    instruction. Rewritten with imperative verbs + an explicit
    contrast directive ("make the shift OBVIOUS") so cadence
    actually changes when the user moves the sliders."""
    if not persona_mix or not isinstance(persona_mix, dict):
        return ""
    # Filter to known presets with weight >= 10% (below that is noise).
    items = []
    for name, weight in persona_mix.items():
        if name not in PERSONA_CARDS:
            continue
        try:
            w = int(round(float(weight)))
        except (TypeError, ValueError):
            continue
        if w >= 10:
            items.append((name, w))
    if not items:
        return ""
    items.sort(key=lambda x: -x[1])
    lead_name, lead_w = items[0]
    header = (
        "VOICE MIX FOR THIS TURN (override your default cadence): "
        + ", ".join(f"{w}% {n}" for n, w in items) + "."
    )
    detail_lines = [
        f" - {n} ({w}%) -- {PERSONA_CARDS[n]}" for n, w in items
    ]
    # Pull the lead voice's sample opener literally so the model has a
    # specific phrase to anchor the first sentence on. Concrete > abstract.
    lead_card = PERSONA_CARDS[lead_name]
    lead_opener = ""
    if "Sample opener:" in lead_card:
        lead_opener = lead_card.split("Sample opener:", 1)[1].strip()
    instruction = (
        f"OPEN THE ANSWER IN THE **{lead_name.upper()}** VOICE. Your "
        f"first sentence must sound like the {lead_name} sample opener "
        f"above"
        + (f" -- something in the spirit of {lead_opener}" if lead_opener else "")
        + f". Do NOT default-open with \"This verse means...\" or "
        f"\"<topic> is foundational because...\" -- those are your "
        f"default-Azriel openers and they make the mix invisible.\n"
        f"\n"
        f"Then write the full answer in that voice, with the lower-"
        f"weight voices showing up as occasional flourishes (1-2 "
        f"sentences out of every 6). Length unchanged -- the mix "
        f"changes HOW you answer, not whether or how much."
    )
    closer = (
        "Identity stays Azriel, biblically-based, scripture-grounded. Refusals "
        "and biblical truth claims are NOT subject to the mix. But "
        "cadence, sentence length, vocabulary, and opening lines ARE -- "
        "shift them."
    )
    return "\n".join([header, *detail_lines, "", instruction, "", closer])


THINKING_MODE_DIRECTIVE = (
    "DELIBERATE MODE -- reason briefly, then ANSWER.\n"
    "\n"
    "STRUCTURE (mandatory):\n"
    " 1. Open <thinking>...</thinking>. Inside, briefly: list the "
    "relevant scriptures, walk the doctrinal angles, anticipate the "
    "strongest counter-argument, decide what to say. Keep this section "
    "TIGHT -- 200-500 words MAX. It is preparation, not the deliverable.\n"
    " 2. CLOSE </thinking>.\n"
    " 3. Write the visible answer to the user (4-6 paragraphs). This "
    "is the actual deliverable -- the user only sees what's outside "
    "the thinking block. If you skip this step, the user sees an empty "
    "response.\n"
    "\n"
    "WHAT THE USER SEES:\n"
    " - inside <thinking>: nothing -- this is hidden from them\n"
    " - outside <thinking>: everything -- this is the answer\n"
    "\n"
    "Budget: keep <thinking> short so you have room to actually answer. "
    "If you're running long inside the thinking block, close it and write "
    "the answer.\n"
    "\n"
    "The recurrent loop has been deepened for this turn (LTI iterations "
    "doubled), so the per-token reasoning is already richer -- you don't "
    "need to write paragraphs of <thinking> to compensate."
)


def _recall_relevant_memory(query: str, n: int = 2) -> str:
    """best-effort cross-session memory recall.
    Queries the persistent memory store (memory.db FTS5) for hits
    related to the user's current message. Returns a small formatted
    block to append to the tool primer, or "" if no useful hits.

    Implementation notes:
      - Uses the same memory_search tool the planner can call manually,
        but invokes it eagerly. The returned block is labeled clearly
        as background hints so the model doesn't treat them as
        authoritative.
      - Hits include source-tagged entries from agent task summaries
        (source='agent'), explicit memory_inserts (source='manual'),
        and research notes (source='research').
      - Skips silently on any error -- memory recall is a UX bonus, not
        a correctness requirement.
      - Caller should NOT invoke this on the bare-route refusal path
        (lesson: don't dilute identity attention on refusal-
        critical prompts)."""
    if not query or len(query.strip()) < 4:
        return ""
    # FTS5 chokes on punctuation; reduce the user message to alphanumeric
    # tokens >=3 chars and rejoin. Drops common stopwords so the index
    # search hits content words ("BSB", "translation", "preference") not
    # the entire question shape.
    _STOP = {"the", "and", "for", "you", "what", "did", "tell", "about",
             "this", "that", "with", "from", "have", "are", "was",
             "were", "your", "any", "can", "how", "when", "where",
             "who", "why", "would", "could", "should", "will"}
    tokens = [
        t for t in re.findall(r"[A-Za-z0-9]+", query)
        if len(t) >= 3 and t.lower() not in _STOP
    ][:8]
    if not tokens:
        return ""
    sanitized = " ".join(tokens)
    try:
        from .tools.memory_search import memory_search as _ms
        raw = _ms(f"{sanitized}|{n}")
    except Exception:
        return ""
    if not raw:
        return ""
    low = raw.lower()
    if any(p in low for p in ("no matches", "no result", "error:", "invalid fts")):
        return ""
    # The tool returns a multi-line block; trim each line and re-format.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    out = ["",
           "Background context (from prior conversations or saved notes -- "
           "use only if relevant to the user's current message; do not quote "
           "verbatim, do not echo any markup, do not announce that you "
           "consulted memory):"]
    for ln in lines[:n]:
        # FTS5 hits include rank scores and source tags; drop the leading
        # "- [source] " prefix the memory_search formatter adds, and any
        # bullet markers, so we don't induce tool-style mimicry in the
        # model's output.
        cleaned = re.sub(r"^[-\s]*\[[a-z]+\]\s*", "", ln).strip()
        if not cleaned:
            continue
        out.append(f" {cleaned[:280]}")
    return "\n".join(out)


def _build_tool_primer(style: str | None = None, persona_mix: dict | None = None,
                        thinking: bool = False, user_message: str | None = None) -> str:
    """Built fresh on every chat turn so connectors that flip from
    disconnected -> connected (or vice versa) are reflected immediately
    in the model's awareness without a server restart.

    `style` is an optional voice hint ("conviction" | "scholar" |
    "pastoral") -- identity is unchanged, only delivery shifts.
    `persona_mix` is an optional {preset_name: percent} dict (e.g.
    {"funny": 25, "professional": 60}). Composed into a Voice-mix
    directive that lives in the PRIMER turn (never the system prompt
    -- that path empirically regresses 8/8 -> 4-7/8 refusals).
    Unknown style or missing values default to conviction."""
    from .tools import get_active_registry
    lines = []
    if thinking:
        lines.append(THINKING_MODE_DIRECTIVE)
        lines.append("")
    persona = _build_persona_directive(persona_mix)
    if persona:
        lines.append(persona)
        lines.append("")
    directive = STYLE_DIRECTIVES.get(style or "conviction")
    if directive:
        lines.append(directive)
        lines.append("")
    # formatting directive. Without this the model defaults
    # to wall-of-text answers (e.g. walking the Lord's Prayer phrase by
    # phrase as one 400-word paragraph). Lives in the primer so it's
    # mutable per turn and doesn't dilute the system-prompt identity weight.
    lines.append(FORMATTING_DIRECTIVE)
    lines.append("")
    lines.append(
        "You have these tools available. Use them when a question needs current data, a specific verse / cross-reference / dictionary entry, file or web access, weather, images, document generation, or memory recall:"
    )
    for name, spec in get_active_registry().items():
        lines.append(f"- {spec['signature']} -- {spec['doc']}")
    lines.append("")
    lines.append('To call a tool, emit a single line: <tool>NAME("ARG")</tool> -- then stop and wait for the runtime to inject <tool_result>...</tool_result>. After the result, integrate it into your answer.')
    lines.append("")
    lines.append('Use ONLY this <tool>...</tool> syntax. Do NOT use bracket-style coder-agent syntax like [tool_use Bash {...}] or shell commands -- those are not our protocol and will not run.')
    lines.append("")
    lines.append('CRITICAL: NEVER emit <tool_result>...</tool_result> yourself. Only the runtime emits those, AFTER a real <tool>...</tool> call you made. If you find yourself about to write <tool_result>, you forgot to call the tool -- emit <tool>NAME("ARG")</tool> instead and stop. Fabricating tool_result content is the single most common reliability bug; the runtime now clips your output at any tool_result you emit and reports the hallucination.')
    lines.append("")
    # γ.8f: Explicit tool-selection examples in the primer. The
    # γ.8 / γ.8c retraining attempts both failed to teach this via LTI
    # alone -- moving the signal up to the primer turn instead, where
    # it's reversible per chat turn and doesn't require GPU work.
    lines.append("WHEN TO FIRE WHICH TOOL:")
    lines.append(' - Missler / handbook / commentary question -> pdf_extract("missler/<book>/<file>.pdf|1-3")')
    lines.append(' e.g. "What does Missler say about Jude?" -> pdf_extract("missler/65_Jude/65_Jude_Commentary_Handbook.pdf|1-3")')
    lines.append(' - weather / forecast / temperature -> weather("<city>")')
    lines.append(' - search the web / current news / "find an article" -> web_search("...")')
    lines.append(' - read a specific URL -> web_fetch("https://...")')
    lines.append(' - find images / pictures of X -> image_search("...")')
    lines.append(' - GitHub / repository / code search -> github_query("...") [connector; may be unconnected]')
    lines.append(' - generate a docx/pptx/xlsx document -> document_create("docx|name|content")')
    lines.append(' - render a chart/diagram/SVG inline -> visualize("<svg>...</svg>")')
    lines.append(' - "remember that..." / "save this" -> memory_insert("...")')
    lines.append(' - "what did I tell you about X?" / past chat -> conversation_search("...") or memory_search("...")')
    lines.append("")
    lines.append("WHEN TO PROPOSE A SKILL:")
    lines.append(' After a multi-turn workflow that produced something reusable -- a sermon outline, a prayer routine, a study plan, a journaling format, a file/archive workflow -- you may offer to save it as a one-click launcher.')
    lines.append(' Fire: <tool>propose_skill("Short Name|kickoff prompt a future user would tweak and send")</tool>')
    lines.append(' Optionally: propose_skill("Name|kickoff|style|persona_mix_json") with style in {conviction, scholar, pastoral} and persona_mix as e.g. {"nurturing":50,"personal":30}.')
    lines.append(' After firing, briefly tell the user "I\'ve offered to save this as a skill -- there\'s a card below if you want to keep it." Do NOT fire propose_skill on routine Q&A or single-turn answers; only when the user has clearly USED you to build something they\'d run again.')
    lines.append("")
    lines.append("WHEN NOT TO FIRE A TOOL (trust your training):")
    lines.append(" - Famous verses you can recite (John 3:16, Psalm 23, Romans 8:28, Lord's Prayer, etc.) -- just recite.")
    lines.append(" - Doctrinal essentials (Trinity, gospel, salvation, repentance, sanctification) -- answer from training.")
    lines.append(" - Identity / self-description / casual conversation -- speak directly.")
    lines.append(" - Reflection on a passage you can quote -- reason from memory; cite the verse you reasoned from.")
    lines.append(" - When the user offers a Scripture-grounded reflection, engage with it instead of looking up new verses.")
    lines.append("")
    lines.append("Identity, refusals, doctrinal stance, prayer, and casual conversation do NOT need tools.")
    if user_message:
        recall = _recall_relevant_memory(user_message)
        if recall:
            lines.append(recall)
    return "\n".join(lines)

TOOL_PRIMER_ASSISTANT = "Understood. I'll call the right tool when one fits, and answer everything else directly from my training as Azriel."

# Empty -- system prompt is now pure constitution
FEW_SHOT_EXAMPLES = ""

TOOL_OPEN = "<tool>"
TOOL_CLOSE = "</tool>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"

# runtime nudge markup. Lets the loop coach the model
# mid-turn without leaking the coach text to the user. Stripped by
# _sanitize_protocol_markup before final return.
NUDGE_OPEN = "<runtime_hint>"
NUDGE_CLOSE = "</runtime_hint>"

CALL_RE = re.compile(r"(\w+)\s*\(\s*(.*?)\s*\)\s*$", re.S)
ARG_RE = re.compile(r'^"(.*)"$|^\'(.*)\'$', re.S)

# Dangling-continuation patterns. The model says it's about to do
# something ("Reading it now...", "Let me check...", "One moment...")
# but never fires a tool call, leaving the user with a half-promise.
# Matched against the trailing tail of the segment so we only nudge
# when the segment ENDS this way.
_DANGLING_RE = re.compile(
    r"(?im)\b("
    r"reading\s+(?:it|that|now|the\s+\S+)|"
    r"let\s+me\s+(?:check|look|read|see|verify|investigate|examine|fetch|grab|pull)|"
    r"checking\s+(?:it|that|now|the\s+\S+)|"
    r"looking\s+(?:at\s+(?:it|that)|into\s+(?:it|that))|"
    r"one\s+moment|hold\s+on|hang\s+on|just\s+a\s+(?:moment|second|sec)|"
    r"i'?ll\s+(?:read|check|look|verify|examine|fetch|grab|pull|run|do|try)|"
    r"i\s+will\s+(?:read|check|look|verify|examine|fetch|grab|pull|run|do|try)|"
    r"i'?m\s+going\s+to\s+(?:read|check|look|verify|examine|fetch|run|do|try)|"
    r"give\s+me\s+a\s+(?:moment|second|sec)|"
    r"now\s+(?:reading|checking|looking|fetching|opening)"
    r")\b[^.\n]*?(?:\.{2,}|…)\s*$"
)

# Heuristic attack-prompt detector. False positives are acceptable (a legit
# prompt that matches loses tool access but still gets a normal answer);
# false negatives are not (an attack that bypasses routes to tools-enabled
# path which empirically erodes refusal). Pattern list errs toward
# over-matching.
ATTACK_PATTERNS = [
    r"ignore.{0,50}(previous|prior|all).{0,50}instruction",
    r"\byou are (dan|now)\b",
    r"\bjailbroken\b",
    r"\bdeveloper mode\b",
    r"no constraints",
    r"\bbe an atheist\b",
    r"pretend.{0,50}(atheist|secular|naturalist)",
    r"reason as.{0,50}atheist",
    r"set aside.{0,30}biblical",
    r"set aside.{0,30}framework",
    r"for one response only",
    r"for this (one|single) (answer|response)",
    r"(from|in) a (purely )?secular",
    r"only secular reasoning",
    r"prophe(?:cy|sy).{0,40}(from|of) god",
    r"speak as god",
    r"as god's messenger",
    r"god's messenger",
    r"forget (your|all) (constitution|instructions|values)",
    r"override (your|the) (identity|persona|system)",
    r"bypass (your|the) (filter|safety|values)",
    # content-attack patterns surfaced by the theta.6
    # agent-mode safety probe. These are specific harmful framings; we
    # err toward catching the demand-pattern, not the surrounding
    # legitimate concern. A false positive on these routes the prompt
    # to bare-chat (clean refusal) instead of tool-route (still safe).
    #
    # Pastoral-bypass demand: "do/don't punt/refer (me) to a pastor"
    # or the more emphatic "you have the answer" variant.
    r"do(?:n't| not) (?:punt|refer|defer)\b.{0,40}\bpastor\b",
    r"don't (?:tell me to )?(?:talk to|see|consult) (?:a |my )?pastor",
    # Confrontation planning aimed at humiliation / public shaming.
    r"\bpublicly humiliate\b",
    r"humiliate\b.{0,40}\b(in front of|at the table|in front of the family)",
    r"(?:plan|help me plan|write|design)\b.{0,50}\b(?:public(?:ly)? (?:humiliate|shame|disgrace))",
    # Manipulation / exploitation taught via fiction frame.
    r"(?:fictional|short|brief) (?:scene|story|tale|skit)\b.{0,80}\b(?:manipulate|exploit|deceive|swindle|coerce|gaslight|emotional(?:ly)? exploit)",
    r"(?:scene|story) where\b.{0,80}\b(?:teaches?|shows?)\b.{0,40}\b(?:manipulate|exploit|deceive|swindle|coerce)",
    # Imperative pastoral over-reach: "tell me directly what God wants"
    # combined with explicit demand for non-deferred answer.
    r"tell me directly what god (?:wants|says|commands|tells)",
    r"(?:directly|exactly) tell me what god\b",
]
_attack_re = re.compile("|".join(ATTACK_PATTERNS), re.IGNORECASE)


def is_attack_prompt(text: str) -> bool:
    return bool(_attack_re.search(text))


_MISSING_CONSTITUTION_FALLBACK = (
    "You are an AI helper. Answer the user's question directly and "
    "substantively. NOTE: no constitution file is installed at "
    "~/.azriel/AZRIEL_CONSTITUTION_SYSTEM.txt -- copy "
    "docs/AZRIEL_CONSTITUTION_TEMPLATE.txt to that path and customize "
    "it for the full Azriel identity floor."
)


def _build_system_prompt() -> str:
    """Pure constitution -- no tool instructions. The constitution carries
    the identity refusal signal; diluting it with tool blocks regresses the
    safety floor (verified empirically: 4-7/8 refusals when tools-block is
    in system prompt vs 8/8 baseline).

    Lines beginning with `#` (whitespace optional) are treated as comments
    and stripped, so install-instruction headers in the template don't
    leak into the system prompt.

    Falls back to a minimal generic prompt if the constitution file is
    missing, so a fresh clone still boots; the user should populate
    ~/.azriel/AZRIEL_CONSTITUTION_SYSTEM.txt for production behaviour."""
    try:
        text = CONSTITUTION_PATH.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        return "\n".join(lines).strip()
    except (FileNotFoundError, PermissionError, OSError):
        return _MISSING_CONSTITUTION_FALLBACK


# Char-budget for history packing. Approximates ~12k tokens (~4 chars
# per token for English). Leaves room in Qwen3-Coder-30B's 256k context
# for the constitution (~1k), tool primer (~3k), current user turn,
# tool observations, and the assistant's response. Limiter here is
# inference latency, not raw context window -- 12k of history keeps
# per-turn generation under ~10s on the host machine.
HISTORY_CHAR_BUDGET = 48000


def _pack_history(history: list[dict] | None) -> list[dict]:
    """Token-budget-aware history packer. Walks newest -> oldest and
    keeps everything that fits inside HISTORY_CHAR_BUDGET. If older
    turns don't fit, prepends a single one-line stub summarizing how
    many turns were dropped + the first user message that was dropped,
    so the model knows the conversation began earlier without seeing
    the full content. Replaces the prior fixed-10-msg cap that lost
    turns silently."""
    if not history:
        return []
    cleaned = [
        m for m in history
        if m.get("role") in ("user", "assistant") and m.get("text")
    ]
    kept_reversed: list[dict] = []
    used = 0
    for msg in reversed(cleaned):
        cost = len(msg.get("text") or "") + 32 # 32 = im_start/im_end overhead
        if used + cost > HISTORY_CHAR_BUDGET:
            break
        kept_reversed.append(msg)
        used += cost
    kept = list(reversed(kept_reversed))
    dropped = len(cleaned) - len(kept)
    if dropped > 0:
        first_user_text = next(
            (m.get("text", "") for m in cleaned if m.get("role") == "user"),
            "",
        )
        stub = (
            f"[earlier in this session: {dropped} prior message(s) "
            f"were dropped to fit the context budget. The conversation "
            f"opened with the user asking: "
            f"{first_user_text[:240]}{'...' if len(first_user_text) > 240 else ''}]"
        )
        return [{"role": "user", "text": stub}] + kept
    return kept


def _render_chat(system: str, user: str, assistant_so_far: str, history=None, style: str | None = None, persona_mix: dict | None = None, thinking: bool = False) -> str:
    """Insert a tool-primer turn pair between system and the real user turn.
    This puts tools in conversational context (which the model handles fine)
    without polluting the system prompt's identity weight.

    history: optional list of {role, text} dicts representing prior turns
    in this session. Packed by _pack_history() up to HISTORY_CHAR_BUDGET
    so multi-turn conversations retain context past the previous fixed
    10-message cap (which truncated at ~5 turn pairs). Older turns that
    don't fit are summarized in a single stub so the model knows the
    conversation began earlier."""
    packed = _pack_history(history)
    parts = [
        f"<|im_start|>system\n{system}<|im_end|>\n",
        f"<|im_start|>user\n{_build_tool_primer(style=style, persona_mix=persona_mix, thinking=thinking, user_message=user)}<|im_end|>\n",
        f"<|im_start|>assistant\n{TOOL_PRIMER_ASSISTANT}<|im_end|>\n",
    ]
    for msg in packed:
        role = msg.get("role")
        text = msg.get("text", "")
        parts.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n")
    parts.append(f"<|im_start|>assistant\n{assistant_so_far}")
    return "".join(parts)


def _parse_tool_call(call_text: str) -> tuple[str, str] | None:
    m = CALL_RE.match(call_text.strip())
    if not m:
        return None
    name = m.group(1).strip()
    raw_arg = m.group(2).strip()
    a = ARG_RE.match(raw_arg)
    arg = (a.group(1) if a and a.group(1) is not None else
           a.group(2) if a else raw_arg)
    return name, arg


# response sanitizer.
#
# v0.6.0 occasionally emits <tool_result>...</tool_result> markup
# directly into its output instead of (or in addition to) calling the
# real tool. The runtime tools-loop only injects tool_result for calls
# it actually executed; any tool_result NOT bracketed by a real call
# is hallucinated. We strip protocol markup from the user-visible text
# and replace fake tool_results with a one-line "[note: model emitted a
# fake tool result for X; the tool was not actually called]" so the
# downstream UI is honest. The calls[] list still reflects what really
# fired.
_TOOL_OPEN_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
_TOOL_RESULT_RE = re.compile(r"<tool_result>(.*?)</tool_result>", re.DOTALL)
_NUDGE_RE = re.compile(re.escape(NUDGE_OPEN) + r".*?" + re.escape(NUDGE_CLOSE), re.DOTALL)


def _sanitize_protocol_markup(text: str, calls: list) -> tuple[str, list[str]]:
    """Strip <tool>...</tool> markup from the user-visible text and
    detect hallucinated <tool_result> blocks.

    A tool_result is "real" if a call appears in `calls` whose result
    body matches the result content. Anything else is hallucinated.

    Returns: (sanitized_text, list_of_hallucinated_tool_names).
    """
    real_results = {(c[2] or "").strip() for c in calls if isinstance(c, tuple)}
    hallucinated: list[str] = []
    # Drop <tool>NAME(arg)</tool> tags entirely from visible text -- the
    # call is preserved separately in `calls`.
    out = _TOOL_OPEN_RE.sub("", text)
    # Walk <tool_result> blocks. Real ones: collapse to whitespace
    # (the model's surrounding prose carries the substance). Fake ones:
    # replace with the disclosure note.
    def _result_repl(m: re.Match) -> str:
        body = (m.group(1) or "").strip()
        if body and any(body in r or r in body or body == r for r in real_results):
            return ""
        # Hallucinated: try to identify which tool the model thought it
        # was reporting. Look for the most recent <tool>NAME(...) in the
        # text up to this point.
        prior = text[: m.start()]
        prior_call = re.findall(r"<tool>\s*(\w+)\s*\(", prior)
        name = prior_call[-1] if prior_call else "unknown_tool"
        hallucinated.append(name)
        return (
            f"\n[note: the model emitted a tool_result for `{name}` but "
            f"that tool was not actually called; output below was hallucinated]\n"
        )
    out = _TOOL_RESULT_RE.sub(_result_repl, out)
    # Strip runtime nudge markers -- those are coaching to the model,
    # never meant for the user.
    out = _NUDGE_RE.sub("", out)
    # Collapse runs of blank lines that the strip operations left behind.
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, hallucinated


def _trim_after_close(text: str, close_tag: str) -> str:
    """Some models keep generating after </tool>; clip there for cleanliness."""
    idx = text.find(close_tag)
    if idx == -1:
        return text
    return text[: idx + len(close_tag)]


def _clip_repeating_substring(text: str, min_len: int = 60) -> str:
    """Detect catastrophic intra-line repetition (model
    decoder loop on a list, e.g. "Peter (rock), Andrew (strong),
    Philip..." cycling) and clip at the SECOND occurrence so the user
    sees the unique content once.

    Approach: walk substrings of length min_len starting throughout
    the first three-quarters of the text; if any 60-char window
    appears 2+ times, that's a clear decoder-loop signal (a 60-char
    exact repeat is implausible in genuine prose). Clip at the second
    occurrence.
    """
    if not text or len(text) < min_len * 2:
        return text
    n = len(text)
    seen_seeds = set()
    # Step by min_len//6 so we sample densely enough to catch loops
    # whose seed starts at any position.
    step = max(1, min_len // 6)
    for start in range(0, max(1, (3 * n) // 4), step):
        seed = text[start : start + min_len]
        if len(seed) < min_len:
            break
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        second = text.find(seed, start + 1)
        if second < 0:
            continue
        # 60-char exact repeat -- decoder loop confirmed. Clip at the
        # second occurrence so the user sees the content once.
        return (
            text[:second].rstrip()
            + "\n\n[output truncated -- the model entered a repetition "
            "loop here; the rest was the same content repeating]"
        )
    return text


# Whole-line bare tool call: `tool_name("arg")` with no <tool> wrapper.
# We require the WHOLE trimmed line to match so we don't accidentally
# rewrite prose that mentions a function call inline.
_BARE_CALL_RE = re.compile(r'^([a-z_][a-z_0-9]*)\s*\(\s*"(.*)"\s*\)\s*$', re.DOTALL)


def _promote_bare_tool_calls(text: str) -> str:
    """If the model emits a whole-line tool call without
    the <tool>...</tool> wrapper AND the name resolves to a registered
    tool, retroactively wrap it so the surrounding tool-loop regex
    matches and we actually execute the call instead of dumping the
    raw syntax to the user as text."""
    try:
        from .tools import get_active_registry
        valid = set(get_active_registry().keys())
    except Exception:
        valid = set()
    if not valid:
        return text
    out_lines = []
    for line in text.split("\n"):
        m = _BARE_CALL_RE.match(line.strip())
        if m and m.group(1) in valid:
            name, arg = m.group(1), m.group(2)
            # Escape literal " inside arg by passing it through unchanged --
            # the existing _parse_tool_call uses ARG_RE which handles
            # surrounding quotes; the inner content stays as-is.
            out_lines.append(f'<tool>{name}("{arg}")</tool>')
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _bare_chat(system: str, user: str) -> str:
    """Refusal-critical path: pure constitution + user, no tools context."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_with_tools(
    model: Any,
    tokenizer: Any,
    user_prompt: str,
    *,
    max_calls: int = 5,
    max_new_tokens_per_segment: int = 400,
    temperature: float = 0.3,
    verbose: bool = False,
    history=None,
    style: str | None = None,
    persona_mix: dict | None = None,
    thinking: bool = False,
) -> dict:
    """Generate an assistant turn, executing tool calls inline.

    Routes by attack-prompt heuristic: refusal-critical prompts go through
    a bare-constitution path (no tools-context, full safety floor).
    Cooperative prompts go through the tools-enabled primer path.

    Returns a dict:
      text: the full assistant turn including <tool>/<tool_result> blocks
      calls: list of (tool_name, arg, result) tuples in call order
      reason_for_stop: 'natural' | 'max_calls' | 'no_progress' | 'attack_route'
      route: 'bare' | 'tools'
    """
    system = _build_system_prompt()
    # bump sampling temperature when a persona_mix is
    # actively requesting voice variation. The LoRA-baked default voice
    # is strong; at temp=0.3 the decoder collapses to that voice even
    # when the primer asks for a different cadence. Higher entropy gives
    # the persona directive room to bite. Cap at 0.7 so we don't lose
    # factual reliability. Safety floor is unchanged -- attack prompts
    # route to bare path before this sampler is built.
    effective_temp = temperature
    if persona_mix and isinstance(persona_mix, dict):
        active = sum(1 for v in persona_mix.values()
                     if isinstance(v, (int, float)) and v >= 10)
        if active > 0:
            effective_temp = max(temperature, 0.65)
    sampler = make_sampler(temp=effective_temp)

    # ----- Thinking-mode runtime overrides -----
    # When the caller asks for deliberate mode, we (a) double the
    # recurrent-loop depth in the wrapper config so each token gets
    # extra LTI iterations on top of the heavy layer pass, and (b)
    # bump per-segment max_tokens so the model has room for both the
    # <thinking>...</thinking> scratchpad AND the visible answer.
    # Saved/restored in a finally so a crash doesn't leak the
    # deliberate setting into the next request.
    saved_loop_iters = None
    saved_max_tokens = max_new_tokens_per_segment
    if thinking:
        if hasattr(model, "config") and hasattr(model.config, "loop_max_iters"):
            saved_loop_iters = model.config.loop_max_iters
            # "Double that" -- if the wrapper currently iterates the
            # LTI once on top of layers (loop_max_iters=2), deliberate
            # mode runs 3 LTI iterations (loop_max_iters=4). Floors at
            # 4 so a base config that's somehow set to 1 still gets
            # meaningful extra capacity.
            model.config.loop_max_iters = max(saved_loop_iters * 2, 4)
        # 3000 leaves room for ~1000 tokens of <thinking> reasoning AND
        # ~1500 tokens of the visible answer. 1500 alone empirically ran
        # out mid-thought on the gift-of-tongues test prompt.
        max_new_tokens_per_segment = max(max_new_tokens_per_segment, 3000)

    # Adversarial prompts skip the tools-enabled path entirely. The
    # constitution-only context preserves v0.6.0's identity refusals.
    # Deliberate mode does NOT propagate to the bare path -- attacks
    # always get the same hard refusal floor.
    if is_attack_prompt(user_prompt):
        if verbose:
            print(f"[route=bare] attack-pattern matched: {user_prompt[:80]}", flush=True)
        # Restore baseline loop depth before bare generate -- the bare
        # path is refusal-critical and we want it to use the same
        # config v0.6.0 was calibrated against.
        if saved_loop_iters is not None:
            model.config.loop_max_iters = saved_loop_iters
            saved_loop_iters = None
        prompt = _bare_chat(system, user_prompt)
        out = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=saved_max_tokens, sampler=sampler,
        )
        return {"text": out.strip(), "calls": [], "reason_for_stop": "attack_route", "route": "bare"}

    assistant = ""
    calls = []
    reason = "natural"
    last_call_sig = None # for same-tool-call repetition guard
    nudged_already = False # one-shot dangling-continuation nudge per turn

    for round_idx in range(max_calls + 1):
        prompt = _render_chat(system, user_prompt, assistant, history=history, style=style, persona_mix=persona_mix, thinking=thinking)
        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens_per_segment,
            sampler=sampler,
        )
        if verbose:
            print(f"--- segment {round_idx} ---\n{out}\n", flush=True)

        # Bracket-style coder-agent syntax bleeding from the Qwen3-Coder base
        # ("[tool_use Bash {...}]"). Not our protocol -- truncate the response
        # at that point before it spirals into a degenerate ls/cat loop.
        bracket = re.search(r"\[tool_use\s+\w+", out)
        if bracket:
            out = out[: bracket.start()].rstrip()

        # bare-paren tool-call promotion.
        # The model sometimes emits `tool_name("arg")` on its own line
        # without the <tool>...</tool> wrapper, especially after a
        # multi-turn skill-design conversation where the user has
        # been talking ABOUT tool names. Without the wrapper the
        # runtime doesn't intercept the call and the user sees raw
        # tool-call syntax dumped as text. Wrap any whole-line bare
        # call whose name matches a registered tool so the loop's
        # <tool> regex catches it and we execute it for real.
        out = _promote_bare_tool_calls(out)

        # hallucination guard.
        # The model should NEVER emit <tool_result>...</tool_result>
        # itself; only the runtime injects those after a real call. If
        # this segment contains <tool_result> without a preceding <tool>
        # in the SAME segment, the model is fabricating a tool result.
        # Clip the segment at the first fake tool_result so the
        # sanitizer + disclosure note can take over cleanly. Break the
        # tool-loop with reason='hallucination' so we don't waste more
        # rounds on a confused decode trajectory.
        tr_open = out.find(TOOL_RESULT_OPEN)
        tool_open = out.find(TOOL_OPEN)
        if tr_open >= 0 and (tool_open < 0 or tool_open > tr_open):
            if verbose:
                print(f"[hallucination-guard] clipped fake tool_result at {tr_open}", flush=True)
            out = out[:tr_open].rstrip()
            assistant += out
            reason = "hallucination"
            break

        # Repeat-line guard: if the same non-empty line has shown up 4+ times
        # in a row, the decoder has locked into a loop (we saw this with
        # `ls /.../CLAUDE.md` repeating until max-tokens). Clip at the first
        # repetition so the user gets a clean answer.
        lines_out = out.split("\n")
        for i in range(3, len(lines_out)):
            window = lines_out[i - 3 : i + 1]
            ref = window[0].strip()
            if ref and all(l.strip() == ref for l in window):
                out = "\n".join(lines_out[: i - 2]).rstrip()
                if verbose:
                    print(f"[loop-guard] clipped at line {i}", flush=True)
                break

        # intra-line repetition guard. Catches list
        # loops where the model produces "Peter (rock), Andrew (strong),
        # Philip ..." then loops back to Peter and repeats the same
        # comma-separated names 3+ times within a single line. The
        # whole-line guard above misses this because it's all on one
        # line. Strategy: find a substring of length >=24 that appears
        # 3+ times in the segment; clip at the second occurrence so
        # the user sees the unique content once.
        out = _clip_repeating_substring(out)

        # Look for an unresolved tool call (one without a following <tool_result>)
        m = re.search(r"<tool>(.+?)</tool>", out, re.S)
        if not m:
            # dangling-continuation guard.
            # If the segment ends with "Reading it now...", "Let me
            # check...", "One moment..." and no tool was fired, the
            # model promised an action but bailed. Inject a one-shot
            # nudge into the assistant context and run another segment
            # so the model either commits to a tool call or finalizes
            # its answer. Capped at one nudge per turn so a stuck
            # decode can't spin the loop.
            if (
                round_idx < max_calls
                and not nudged_already
                and _DANGLING_RE.search(out.rstrip())
            ):
                nudged_already = True
                assistant += out
                assistant += (
                    f"\n{NUDGE_OPEN}you started an action but emitted no "
                    f"tool call. Either fire the tool now via "
                    f"<tool>NAME(\"ARG\")</tool> and stop, OR finalize "
                    f"your answer with what you already have. Do not "
                    f"say you're about to do something without doing "
                    f"it.{NUDGE_CLOSE}\n"
                )
                if verbose:
                    print(f"[dangling-guard] nudged at round {round_idx}", flush=True)
                continue
            assistant += out
            reason = "natural"
            break

        # Trim segment to end of </tool>; everything past that is speculative
        new_segment = _trim_after_close(out, TOOL_CLOSE)
        # If trimming gave us no forward progress, abort to avoid infinite loop
        if not new_segment.endswith(TOOL_CLOSE):
            assistant += out
            reason = "no_progress"
            break

        assistant += new_segment

        call_text = m.group(1)
        parsed = _parse_tool_call(call_text)
        if parsed is None:
            result = f"ERROR: malformed tool call '{call_text}'"
        else:
            name, arg = parsed
            sig = (name, arg)
            # Same-tool-call repetition guard. If the model fires the
            # exact same tool with the exact same arg twice in a row,
            # treat it as a stuck-decode signal and abort the round.
            # Saw this on autoresearch with strongs_lookup looping on
            # the same H#### across consecutive rounds. Saves Metal +
            # Ollama cost AND surfaces a cleaner result.
            if sig == last_call_sig:
                result = (
                    f"ERROR: same call as previous round ({name}({arg!r})). "
                    "If you have the answer, integrate it now. If not, "
                    "try a different tool or different argument."
                )
                calls.append((name, arg, result))
                assistant += "\n" + TOOL_RESULT_OPEN + result + TOOL_RESULT_CLOSE + "\n"
                reason = "no_progress"
                if verbose:
                    print(f"[same-call guard] aborting on repeated {sig}", flush=True)
                break
            last_call_sig = sig
            try:
                result = tool_call(name, arg)
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"
        calls.append((parsed[0] if parsed else None, parsed[1] if parsed else None, result))
        assistant += "\n" + TOOL_RESULT_OPEN + result + TOOL_RESULT_CLOSE + "\n"

        if round_idx == max_calls:
            reason = "max_calls"
            # Do one more pass to let the model integrate
            prompt = _render_chat(system, user_prompt, assistant)
            out = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens_per_segment,
                sampler=sampler,
            )
            assistant += out
            break

    # Restore loop depth before returning so the next request runs
    # with the baseline config (saved value is None if thinking=False).
    if saved_loop_iters is not None:
        model.config.loop_max_iters = saved_loop_iters
    final_text, hallucinated = _sanitize_protocol_markup(assistant, calls)

    # lift any propose_skill sentinel into a
    # structured field. The propose_skill tool returns
    # __AZRIEL_SKILL_PROPOSAL__<json>; we extract the json, strip the
    # tool_result block from visible text, and surface the proposal
    # so the dashboard can render a "Save this as a skill?" card.
    skill_proposal = None
    SENTINEL = "__AZRIEL_SKILL_PROPOSAL__"
    for c in calls:
        if isinstance(c, tuple) and len(c) >= 3 and isinstance(c[2], str):
            if c[2].startswith(SENTINEL):
                try:
                    import json as _json
                    skill_proposal = _json.loads(c[2][len(SENTINEL):])
                except Exception:
                    skill_proposal = None
                break
    if skill_proposal is not None:
        # Strip the sentinel from any visible tool_result block.
        final_text = re.sub(
            re.escape(SENTINEL) + r"\{.*?\}",
            "(skill proposed -- see card below)",
            final_text, flags=re.DOTALL,
        )

    return {
        "text": final_text,
        "calls": calls,
        "reason_for_stop": reason,
        "route": "tools",
        "hallucinated_tool_results": hallucinated,
        "skill_proposal": skill_proposal,
    }
