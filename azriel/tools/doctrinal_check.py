"""doctrinal_check -- term-hit classifier across 10 doctrinal axes.

Given a claim or short passage, scan it for term patterns associated
with two contrasting positions on each of 10 doctrinal axes. Returns
per-axis term hits and an aggregate verdict.

The two position lists per axis are labeled A and B and carry no
preferential weight in this tool -- the caller decides what those
positions represent for their use case. The verdict is a heuristic
only (term-hit classification gets ~75% agreement with the teacher
rubric in scripts/45_doctrinal_scorer_v2.py), so this tool is for
quick orientation, not authoritative judgment. The model can use it
as a self-check before committing to a doctrinal claim.
"""
from __future__ import annotations

import re
from collections import Counter

# Per-axis (a_terms, b_terms). The two columns are not ranked or
# preferred -- they are simply the two contrasting vocabularies the
# tool scans for.
AXES: dict[str, tuple[list[str], list[str]]] = {
    "A.continuationism": (
        ["continue", "still active", "available", "operative",
         "1 corinthians 12", "1 corinthians 14", "mark 16",
         "gifts", "tongues", "prophecy", "manifestation", "power",
         "acts 2", "acts 1:8", "boldness"],
        ["ceased", "cessation", "apostolic age", "no longer",
         "fruit only", "internal only", "no outward sign"],
    ),
    "B.spirit_baptism": (
        ["subsequent", "second", "distinct", "separate experience",
         "acts 2", "acts 8", "acts 19", "empowerment",
         "second work", "after belief", "laying on of hands"],
        ["same moment", "at conversion", "automatic", "regeneration",
         "unique", "transitional", "anomaly"],
    ),
    "C.tongues": (
        ["initial evidence", "tongues", "acts 2:4", "acts 10:46",
         "acts 19:6", "speaking in other tongues", "outward sign",
         "real gift", "personal prayer", "edification",
         "1 corinthians 14:4", "interpretation", "intercession"],
        ["not necessarily", "not the only evidence",
         "different gifts to different people",
         "ceased", "not for today", "spurious"],
    ),
    "D.healing": (
        ["isaiah 53:5", "matthew 8:17", "1 peter 2:24", "by his stripes",
         "atonement provides", "healing in atonement",
         "james 5:14", "elders", "anointing with oil",
         "prayer of faith", "mark 16:18", "expect healing"],
        ["spiritual only", "not physical", "primarily soteriological",
         "medical only", "may not be god's will to heal"],
    ),
    "E.eschatology": (
        ["pre-tribulation", "imminent", "1 thessalonians 4:17",
         "revelation 3:10", "blessed hope", "titus 2:13",
         "premillennial", "revelation 20", "thousand years",
         "earthly reign", "davidic kingdom",
         "romans 11", "ongoing covenant", "ethnic israel",
         "all israel will be saved"],
        ["post-tribulation", "mid-tribulation", "amillennial", "preterist",
         "symbolic", "postmillennial", "no literal millennium",
         "replacement", "supersessionism", "church replaces"],
    ),
    "F.soteriology": (
        ["foreknowledge", "free will", "respond", "arminian",
         "1 peter 1:2", "whoever", "human responsibility", "conditional",
         "can fall away", "hebrews 6", "hebrews 10:26", "shipwreck faith",
         "1 timothy 1:19", "prevenient grace", "able to choose",
         "draw", "john 12:32"],
        ["unconditional", "sovereign decree", "irresistible", "tulip",
         "calvinist", "monergism",
         "once saved always saved", "perseverance of the saints",
         "eternal security",
         "totally depraved", "cannot choose god", "bondage of the will"],
    ),
    "G.sacramentology": (
        ["immersion", "believer's baptism", "credobaptism",
         "acts 8:38", "buried with christ", "romans 6:4",
         "memorial", "spiritually present", "remembrance",
         "1 corinthians 11:24", "ordinance", "symbol"],
        ["sprinkling", "infant", "paedobaptism", "covenant sign",
         "household",
         "transubstantiation", "real presence", "sacrament",
         "consubstantiation"],
    ),
    "H.bibliology": (
        ["inerrant", "without error", "everything it affirms", "fully",
         "all scripture", "2 timothy 3:16", "verbal plenary",
         "every word", "plenary", "2 peter 1:21", "carried along",
         "inspired", "god-breathed"],
        ["inerrant in faith only", "errs in history", "limited inerrancy",
         "infallible but not inerrant",
         "dictation only", "merely human", "inspired ideas only"],
    ),
    "I.ecclesiology": (
        ["acts 2:17-18", "joel 2:28", "daughters shall prophesy",
         "phoebe", "junia", "spiritual gifts to all",
         "every believer", "1 corinthians 12:7",
         "manifestation given to each", "all flesh"],
        ["1 timothy 2:12", "not permit a woman to teach",
         "complementarian", "male elders only",
         "only ordained", "only apostles", "selectively"],
    ),
    "J.hermeneutics": (
        ["literal", "israel means israel", "national", "physical",
         "ethnic", "earthly kingdom", "future"],
        ["symbolic", "spiritualized", "church",
         "fulfilled in christ"],
    ),
}


def _hits(text: str, terms: list[str]) -> list[str]:
    out = []
    low = text.lower()
    for t in terms:
        if t in low:
            out.append(t)
    return out


def _summarize(per_axis: dict[str, dict[str, list[str]]]) -> str:
    a_total = sum(len(v["a"]) for v in per_axis.values())
    b_total = sum(len(v["b"]) for v in per_axis.values())
    if a_total == 0 and b_total == 0:
        return "no doctrinal terms matched on any axis (unclear)"
    if a_total > 0 and b_total == 0:
        return f"position-a-leaning: {a_total} A term(s), 0 B"
    if b_total > 0 and a_total == 0:
        return f"position-b-leaning: {b_total} B term(s), 0 A"
    return f"mixed: {a_total} A term(s), {b_total} B term(s)"


def doctrinal_check(claim: str) -> str:
    """Scan a claim or short passage for term-hit signatures across the
    10 doctrinal axes. Returns axis-by-axis matches plus an aggregate
    verdict.

    This is a fast heuristic only -- it does not catch paraphrased
    positions that avoid the canonical vocabulary. Use it as a sanity
    check, not a final judgment.
    """
    if not claim or not claim.strip():
        return "ERROR: empty claim."
    text = claim.strip()
    if len(text) > 4000:
        text = text[:4000] + "..."
    per_axis: dict[str, dict[str, list[str]]] = {}
    for axis, (a_terms, b_terms) in AXES.items():
        ah = _hits(text, a_terms)
        bh = _hits(text, b_terms)
        if ah or bh:
            per_axis[axis] = {"a": ah, "b": bh}
    summary = _summarize(per_axis)
    out = [f"verdict: {summary}"]
    if not per_axis:
        out.append("(no matching terms on any of the 10 axes)")
        return "\n".join(out)
    for axis, hits in per_axis.items():
        line_parts = [axis]
        if hits["a"]:
            line_parts.append(f"a: {', '.join(hits['a'][:6])}")
        if hits["b"]:
            line_parts.append(f"b: {', '.join(hits['b'][:6])}")
        out.append(" " + " | ".join(line_parts))
    return "\n".join(out)
