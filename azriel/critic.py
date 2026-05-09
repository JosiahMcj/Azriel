"""Self-critique.

Takes (message, response) and returns a structured verdict: which
factual claims look wrong, which scripture citations look fabricated
or out of place, which doctrinal claims drift from Azriel's trained
biblically-based frame, and which parts of the answer contradict each
other.

Known limitation -- same-model bias:
    The model that produced an answer is calibrated such that its
    own issues are invisible to it. A hallucinated citation that
    "feels right" to the generator will feel right to the critic
    too. Verdicts here are "second-pass cleanup catch" signal, not
    independent review. The right η-era upgrade is a different-model
    critic (Ollama qwen2.5:32b), gated on resolving the GPU-contention
    failure mode we hit earlier in this cycle.

Contract guarantees:
    - critique() NEVER raises. Parse failures, generation failures,
      timeouts -- all funnel into Critique(parse_failed=True, raw=...).
    - Output is a dataclass with predictable fields. Callers can rely
      on `c.severity in {"low","medium","high"}` even on failure
      (defaults to "low").
    - critique() is LOGGED, not GATING. Don't make memory promotion
      or chat responses depend on critic output in this ship; that's
      a ζ.2 concern. Adding gating now means a single false-positive
      can silently kill a real research success.

Architecture: direct `mlx_lm.generate` call with a critic-specific
system prompt, NOT through `run_with_tools()`. The runtime's
attack-detection / refusal path would mis-route a critique on a
challenging prompt to "I won't roleplay" -- which is the right call
for chat but wrong for analysis.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CONSTITUTION_PATH = Path.home() / ".azriel" / "AZRIEL_CONSTITUTION_SYSTEM.txt"


CRITIC_INSTRUCTIONS = """\
You are reviewing Azriel's answer for problems. Be specific and concise.
Output ONLY valid JSON matching this exact schema:

{
  "severity": "low" | "medium" | "high",
  "revise_recommended": true | false,
  "factual_issues": ["short description", ...],
  "scripture_issues": ["short description with verse if applicable", ...],
  "doctrinal_issues": ["short description", ...],
  "internal_contradictions": ["short description", ...]
}

Definitions:
- factual_issues: factually wrong claims (history, science, current events)
- scripture_issues: fabricated verses ("God helps those who help themselves"
  is NOT in 2 Corinthians), wrong references, misquotation
- doctrinal_issues: drift from the trained biblically-based frame
  (claims that contradict the constitution or the trained stance,
  prosperity-gospel creep, soft-universalism, replacement theology)
- internal_contradictions: the answer contradicts itself, or begins one
  position and ends another

Severity:
- low: no issues found, or trivial style
- medium: at least one issue but the overall answer is still useful
- high: at least one fabricated citation, factual error a reader would
  act on, or doctrinal drift that misleads

revise_recommended: true if the answer should be redone before being
shared with a user, false otherwise.

If the answer has no issues, output:
{"severity":"low","revise_recommended":false,"factual_issues":[],"scripture_issues":[],"doctrinal_issues":[],"internal_contradictions":[]}

Output ONLY the JSON object. No preamble, no commentary, no markdown
fence.
"""


@dataclass
class Critique:
    severity: str = "low"
    revise_recommended: bool = False
    factual_issues: list[str] = field(default_factory=list)
    scripture_issues: list[str] = field(default_factory=list)
    doctrinal_issues: list[str] = field(default_factory=list)
    internal_contradictions: list[str] = field(default_factory=list)
    parse_failed: bool = False
    raw: str = ""
    duration_ms: int = 0
    constitution_in_context: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def has_issues(self) -> bool:
        return bool(
            self.factual_issues
            or self.scripture_issues
            or self.doctrinal_issues
            or self.internal_contradictions
        )


def _load_constitution() -> str:
    """Read constitution fresh from disk. Empty string if missing.
    Lines starting with `#` are treated as comments and stripped."""
    try:
        text = CONSTITUTION_PATH.read_text()
        lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        return "\n".join(lines).strip()
    except Exception:
        return ""


def _build_critic_prompt(
    message: str, response: str, *, include_constitution: bool
) -> str:
    """Render the chat template with constitution + critic instructions
    + the message/response under review."""
    constitution = _load_constitution() if include_constitution else ""
    sys_parts = []
    if constitution:
        sys_parts.append(constitution)
    sys_parts.append(CRITIC_INSTRUCTIONS)
    system = "\n\n".join(sys_parts)

    user = (
        "Review the following exchange.\n\n"
        f"USER MESSAGE:\n{message}\n\n"
        f"AZRIEL'S ANSWER:\n{response}\n\n"
        "Output the JSON verdict now."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_verdict(raw: str) -> Optional[dict]:
    """Pull the first {...} block out of `raw` and json.loads it.
    Returns None if no parseable JSON is found."""
    if not raw:
        return None
    raw = raw.strip()
    # Sometimes the model wraps JSON in ```json ... ``` -- strip code fences
    if raw.startswith("```"):
        # Drop opening fence
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        # Drop closing fence
        if "```" in raw:
            raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()
    # Try direct parse first (cheapest)
    try:
        v = json.loads(raw)
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    # Fallback: greedy {...} extraction
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return None
    candidate = m.group(0)
    try:
        v = json.loads(candidate)
        if isinstance(v, dict):
            return v
    except Exception:
        return None
    return None


def _coerce_str_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v:
        return [v]
    return []


def _coerce_severity(v) -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("low", "medium", "high"):
            return s
        # common variants
        if s in ("med",):
            return "medium"
    return "low"


def critique(
    model,
    tokenizer,
    message: str,
    response: str,
    *,
    include_constitution: bool = True,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> Critique:
    """Critique `response` to `message` with the loaded model.

    Never raises. On any failure (timeout, parse, exception in
    generate), returns Critique(parse_failed=True, raw=<diagnostic>).
    """
    out = Critique(constitution_in_context=include_constitution)
    t0 = time.time()
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = _build_critic_prompt(
            message, response, include_constitution=include_constitution
        )
        sampler = make_sampler(temp=temperature)
        raw = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        out.raw = raw
    except Exception as e:
        out.parse_failed = True
        out.raw = f"<generate failed: {type(e).__name__}: {e}>"
        out.duration_ms = int((time.time() - t0) * 1000)
        return out

    parsed = _parse_json_verdict(raw)
    if parsed is None:
        out.parse_failed = True
        out.duration_ms = int((time.time() - t0) * 1000)
        return out

    out.severity = _coerce_severity(parsed.get("severity"))
    out.revise_recommended = bool(parsed.get("revise_recommended"))
    out.factual_issues = _coerce_str_list(parsed.get("factual_issues"))
    out.scripture_issues = _coerce_str_list(parsed.get("scripture_issues"))
    out.doctrinal_issues = _coerce_str_list(parsed.get("doctrinal_issues"))
    out.internal_contradictions = _coerce_str_list(
        parsed.get("internal_contradictions")
    )
    out.duration_ms = int((time.time() - t0) * 1000)
    return out
