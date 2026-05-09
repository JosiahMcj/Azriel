"""propose_skill -- conversation-driven skill creation.

Special tool the model fires when a multi-turn workflow has produced
something reusable. The result is a marker string the runtime detects
and lifts out into the /chat response's `skill_proposal` field. The
dashboard renders an inline "Save this as a skill?" card under the
message; the user clicks Save to persist it.

Argument shape:
  "name|kickoff_prompt"
  "name|kickoff_prompt|style"
  "name|kickoff_prompt|style|persona_mix_json"

Examples the model might fire:
  propose_skill("Daily Examen|Walk me through a five-step Ignatian-style daily examen: gratitude, ask, review, repent, resolve. One paragraph each.|pastoral")

  propose_skill("Romans Study|Help me build a 4-week study on Romans focusing on justification by faith. Ask me audience first.|conviction|{\"interesting\": 50, \"professional\": 35}")

When fired, this returns a SENTINEL string the runtime intercepts.
The visible text the user sees is replaced with a clean acknowledgment;
the structured proposal lifts into the /chat JSON response so the
dashboard can render the save UI.
"""
from __future__ import annotations

import json
import re

# Sentinel the runtime grep's for. Distinctive enough to never collide
# with normal text. The runtime strips this from the visible output and
# moves the structured payload into the /chat response.
_SENTINEL = "__AZRIEL_SKILL_PROPOSAL__"


def propose_skill(arg: str) -> str:
    if not arg or "|" not in arg:
        return ("ERROR: propose_skill expects 'name|kickoff' "
                "(optionally |style|persona_mix_json). Got: "
                f"{(arg or '')[:80]!r}")
    parts = arg.split("|", 3)
    name = parts[0].strip()
    kickoff = parts[1].strip() if len(parts) > 1 else ""
    style = parts[2].strip() if len(parts) > 2 else ""
    persona_raw = parts[3].strip() if len(parts) > 3 else ""

    if not name:
        return "ERROR: skill name is empty."
    if not kickoff:
        return "ERROR: skill kickoff prompt is empty."
    if len(name) > 80:
        return "ERROR: skill name too long (max 80 chars)."
    if len(kickoff) > 2000:
        return "ERROR: kickoff prompt too long (max 2000 chars)."

    persona = {}
    if persona_raw:
        try:
            persona = json.loads(persona_raw)
            if not isinstance(persona, dict):
                persona = {}
        except (json.JSONDecodeError, ValueError):
            persona = {}

    proposal = {
        "name": name,
        "kickoff": kickoff,
    }
    if style and style in ("conviction", "scholar", "pastoral"):
        proposal["style"] = style
    if persona:
        # Filter to known persona keys + sane percent values.
        valid = {
            "funny", "ecstatic", "personal", "somber", "professional",
            "interesting", "nurturing", "direct", "poetic", "encouraging",
        }
        clean = {}
        for k, v in persona.items():
            if k in valid:
                try:
                    iv = int(v)
                    if 0 < iv <= 100:
                        clean[k] = iv
                except (TypeError, ValueError):
                    continue
        if clean:
            proposal["persona_mix"] = clean

    # Slug for the eventual id. Server's /skills/save will normalize
    # again on persist.
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "skill"
    proposal["id_hint"] = slug

    # Return the sentinel + JSON. Runtime will parse this out.
    return f"{_SENTINEL}{json.dumps(proposal, ensure_ascii=False)}"
