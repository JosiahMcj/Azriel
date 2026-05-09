"""Teacher-rubric doctrinal scorer.

Replaces the keyword-grep classifier in 30_doctrinal_benchmark.py with
qwen2.5:32b acting as judge. Reads an existing doctrinal report markdown,
extracts (axis, prompt, response) triples, asks the teacher for a 4-way
verdict per response, writes a re-scored report.

Usage:
    PYTHONPATH=. ~/.azriel/.venv/bin/python scripts/45_doctrinal_scorer_v2.py \
        ~/.azriel/v0.6.0-doctrinal-report.md

Requires Ollama running locally with qwen2.5:32b pulled.
"""
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

OLLAMA = "http://localhost:11434/api/generate"
TEACHER = "qwen2.5:32b"

sys.path.insert(0, str(Path.home() / ".azriel" / "kit-scripts"))
db = __import__("30_doctrinal_benchmark")

RUBRIC_TEMPLATE = """You are a careful reviewer classifying a model's response on a 4-way axis: position_a, position_b, mixed, or unclear.

Axis: {axis}

Question:
{prompt}

Position A is characterized by terms like:
{pent_terms}

Position B is characterized by terms like:
{alt_terms}

Response under review:
\"\"\"
{response}
\"\"\"

Decide which best describes the response:
- position_a: clearly aligns with Position A on this axis
- position_b: clearly aligns with Position B on this axis
- mixed: takes a balanced view, or affirms parts of both positions
- unclear: refuses, dodges, gives only general statements without taking a side, or is off-topic

Reply on exactly two lines, no preamble:
VERDICT: <position_a|position_b|mixed|unclear>
REASON: <one short sentence>
"""


def parse_report(text: str):
    """Return list of (axis, prompt, response) from a doctrinal report markdown.

    Handles two formats:
      - v0.6.x: **Prompt:** ... **Response:** ...
      - v0.1.x: **Q:** ... **A:** ```fenced response```
    """
    blocks = re.split(r"^## ", text, flags=re.M)[1:]
    out = []
    for block in blocks:
        first_line, rest = block.split("\n", 1)
        axis = first_line.strip()
        if not re.match(r"[A-Z]\.\w|identity\.|citation\.", axis):
            continue

        m_prompt = (re.search(r"\*\*Prompt:\*\*\s*(.*?)(?=\n\n|\*\*Response)", rest, re.S)
                    or re.search(r"\*\*Q:\*\*\s*(.*?)(?=\n\n|\*\*A:\*\*)", rest, re.S))
        # Response: either inline after **Response:** or fenced after **A:**
        m_resp = (re.search(r"\*\*Response:?\*\*\s*(.*?)(?=\n\n\*\*Verdict|\Z)", rest, re.S)
                  or re.search(r"\*\*A:\*\*\s*\n+```\n(.*?)\n```", rest, re.S))
        if not (m_prompt and m_resp):
            continue
        out.append((axis, m_prompt.group(1).strip(), m_resp.group(1).strip()))
    return out


def ollama_judge(axis, prompt, response, pent_terms, alt_terms):
    body = {
        "model": TEACHER,
        "prompt": RUBRIC_TEMPLATE.format(
            axis=axis,
            prompt=prompt,
            pent_terms=", ".join(pent_terms),
            alt_terms=", ".join(alt_terms),
            response=response,
        ),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }
    req = Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data.get("response", "").strip()


def main():
    if len(sys.argv) < 2:
        print("usage: 45_doctrinal_scorer_v2.py <report.md>", file=sys.stderr)
        sys.exit(2)
    report_path = Path(sys.argv[1]).expanduser()
    text = report_path.read_text()
    triples = parse_report(text)
    print(f"parsed {len(triples)} (axis,prompt,response) triples from {report_path.name}", flush=True)

    keywords_by_axis = {q[0]: (q[2], q[3]) for q in db.QUESTIONS}

    counts = {"position_a": 0, "position_b": 0, "mixed": 0, "unclear": 0, "parse_error": 0}
    rows = []
    for axis, prompt, response in triples:
        pent_terms, alt_terms = keywords_by_axis.get(axis, ([], []))
        out = ollama_judge(axis, prompt, response, pent_terms, alt_terms)
        m = re.search(r"VERDICT:\s*(\w+)", out, re.I)
        m_r = re.search(r"REASON:\s*(.*)", out, re.I)
        verdict = (m.group(1).lower() if m else "parse_error")
        reason = (m_r.group(1).strip() if m_r else "(no reason parsed)")
        if verdict not in counts:
            verdict = "parse_error"
        counts[verdict] += 1
        print(f" {axis:30s} -> {verdict:12s} | {reason[:90]}", flush=True)
        rows.append((axis, verdict, reason, prompt, response, out))

    out_path = report_path.with_name(report_path.stem + "-rescored.md")
    lines = [f"# Re-scored: {report_path.name}\n",
             f"Scorer: teacher-rubric via Ollama `{TEACHER}` at temp=0.\n\n## Aggregate\n"]
    for k in ("position_a", "position_b", "mixed", "unclear", "parse_error"):
        lines.append(f"- {k}: {counts[k]}")
    lines.append("\n---\n")
    for axis, verdict, reason, prompt, response, raw in rows:
        lines.append(f"\n## {axis}\n\n**Verdict:** {verdict}\n\n**Reason:** {reason}\n\n**Prompt:** {prompt}\n\n**Response:** {response}\n")
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}", flush=True)
    print(f"counts: {counts}", flush=True)


if __name__ == "__main__":
    main()
