"""autoresearch issue triage.

Aggregates ~/.azriel/data/research/issues.jsonl and runs.jsonl into
a top-issues report. Closes the autoresearch feedback loop:

  ε.1 produces research notes
  ε.2 promotes them to memory
  ε.3 schedules hourly drains
  ε.4 (this) reads the issues log + surfaces fixable patterns

The report is rewritten on each run to ~/.azriel/data/research/
EPSILON_ISSUES_REPORT.md (and mirrored into the repo at
docs/EPSILON_ISSUES_REPORT.md when run from the dashboard machine).

Each `kind` has a canned "suggested fix" so when we see a pattern
we know roughly what to address. Reading the report should be enough
to write the next cycle's queued question or runtime patch.

USAGE:
  cd ~/azriel-arch
  PYTHONPATH=. ~/.azriel/.venv/bin/python scripts/61_issue_triage.py
"""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA_ROOT = Path.home() / ".azriel" / "data" / "research"
ISSUES_LOG = DATA_ROOT / "issues.jsonl"
RUNS_LOG = DATA_ROOT / "runs.jsonl"
REPORT_PATH = DATA_ROOT / "EPSILON_ISSUES_REPORT.md"

# Canned advice per issue kind. Updated as we learn what works.
SUGGESTED_FIX = {
    "tool_error": (
        "A tool returned ERROR. Check the snippet to see which tool and "
        "what arg shape failed. Common fixes: arg syntax mismatch (the "
        "model passed positional args where the tool wanted 'arg|pages'), "
        "tool needs auth (connector unconnected), or the tool genuinely "
        "errored on its remote side. Address case by case."
    ),
    "tool_arg_malformed": (
        "Runtime saw <tool>NAME(...)</tool> but couldn't parse the args. "
        "Either the model wrote a malformed call (check the snippet) or "
        "the parser regex is too strict. Easy fix: tighten the WHEN-TO-"
        "FIRE example for the offending tool in the runtime primer."
    ),
    "bracket_bleed": (
        "Model emitted the vision provider [tool_use NAME {...}] coder-agent syntax "
        "instead of <tool>NAME(\"...\")</tool>. Existing dashboard "
        "renderer already strips it; runtime guard in run_with_tools "
        "clips at first occurrence. Frequency dropping over time is the "
        "best signal here. If frequency rises, strengthen the primer's "
        "'use ONLY this syntax' line."
    ),
    "repetition": (
        "4+ identical lines in a row in a model response. Usually a "
        "stuck-decode loop (e.g., 'qarab' from a Strong's lookup). "
        "Existing runtime line-repeat guard catches when the SAME line "
        "fires within one segment; 4+ across segments needs the same-"
        "tool-call guard (already shipped)."
    ),
    "incomplete": (
        "Model hit max_calls without firing fs_write to save the note. "
        "Either max_calls=10 is too low (rare) or the prompt isn't "
        "strong enough about closing with fs_write. Tune the research "
        "prompt template."
    ),
    "safety_misroute": (
        "A research prompt routed to the bare/refusal path. Means our "
        "is_attack_prompt regex matched something it shouldn't have on "
        "an innocent research topic. Look at the snippet, check whether "
        "the topic phrasing accidentally hit a pattern, narrow the "
        "regex if needed."
    ),
    "model_timeout": (
        "Server crashed or HTTP error during a /chat call. Check launchd "
        "logs (~/.azriel/logs/server.{out,err}) for traceback. If "
        "Metal OOM, the model probably swapped due to memory pressure."
    ),
    "secret_in_summary": (
        "The auto-promotion path (ε.2) caught secret-shaped text in a "
        "research summary and refused to write it to memory. Audit the "
        "snippet to see what tripped looks_secret(); usually a topic "
        "name that includes an IP or path."
    ),
    "memory_promotion_failed": (
        "promote_to_memory raised an exception. Check the snippet for "
        "the exception type. If it's repeated, the memory store may be "
        "locked, full, or schema-mismatched."
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def aggregate(issues: list[dict], runs: list[dict]) -> dict:
    by_kind = Counter()
    by_topic = Counter()
    by_kind_topic = defaultdict(Counter)
    snippets = defaultdict(list)
    for r in issues:
        k = r.get("kind", "?")
        t = r.get("topic", "?")
        by_kind[k] += 1
        by_topic[t] += 1
        by_kind_topic[k][t] += 1
        if r.get("snippet"):
            snippets[k].append(r["snippet"][:200])

    n_runs = len(runs)
    successful = sum(1 for r in runs if r.get("success"))
    avg_dur = (sum(r.get("duration_s", 0) for r in runs) / n_runs) if n_runs else 0
    avg_calls = (sum(r.get("iterations", 0) for r in runs) / n_runs) if n_runs else 0
    tool_use = Counter()
    for r in runs:
        for t in r.get("tools_fired") or []:
            tool_use[t] += 1

    return {
        "n_issues": len(issues),
        "n_runs": n_runs,
        "n_successful": successful,
        "success_rate": (successful / n_runs) if n_runs else 0,
        "avg_duration_s": avg_dur,
        "avg_calls_per_run": avg_calls,
        "by_kind": dict(by_kind),
        "by_topic": dict(by_topic),
        "by_kind_topic": {k: dict(v) for k, v in by_kind_topic.items()},
        "tool_use": dict(tool_use),
        "snippets": {k: v[:3] for k, v in snippets.items()},
    }


def render_report(agg: dict) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Autoresearch -- Issues Report",
        "",
        f"_Last triage run: {ts}_",
        "",
        "## Run-level health",
        "",
        f"- Total runs logged: **{agg['n_runs']}**",
        f"- Successful: **{agg['n_successful']}** ({agg['success_rate'] * 100:.0f}%)",
        f"- Average duration: {agg['avg_duration_s']:.1f}s",
        f"- Average tool calls per run: {agg['avg_calls_per_run']:.1f}",
        "",
        "## Tool fan-out across runs",
        "",
    ]
    if agg["tool_use"]:
        for tool, n in sorted(agg["tool_use"].items(), key=lambda x: -x[1]):
            lines.append(f" - `{tool}`: fired in {n} runs")
    else:
        lines.append(" (no tools fired across logged runs)")
    lines += ["", "## Issues by kind", ""]
    if not agg["by_kind"]:
        lines.append("(no issues logged yet)")
    else:
        sorted_kinds = sorted(agg["by_kind"].items(), key=lambda x: -x[1])
        for kind, n in sorted_kinds:
            lines.append(f"### {kind} -- **{n}** occurrence(s)")
            lines.append("")
            advice = SUGGESTED_FIX.get(
                kind, "(no canned advice for this kind yet -- inspect snippets)"
            )
            lines.append(f"_Suggested fix: {advice}_")
            lines.append("")
            tops = agg["by_kind_topic"].get(kind, {})
            if tops:
                lines.append("**Affected topics:**")
                for topic, c in sorted(tops.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f" - {topic}: {c}x")
                lines.append("")
            snips = agg["snippets"].get(kind, [])
            if snips:
                lines.append("**Sample snippets (first 200 chars each):**")
                for s in snips[:3]:
                    lines.append("")
                    lines.append("```")
                    lines.append(s.replace("```", "''"))
                    lines.append("```")
                lines.append("")
            lines.append("---")
            lines.append("")

    lines += ["## Topics with the most issues", ""]
    if not agg["by_topic"]:
        lines.append("(none)")
    else:
        for topic, n in sorted(agg["by_topic"].items(), key=lambda x: -x[1])[:10]:
            lines.append(f" - {topic}: {n} issue(s)")
    lines.append("")

    lines += [
        "## Triage recommendation",
        "",
    ]
    if not agg["by_kind"]:
        lines.append(
            "Nothing to triage. Run more autoresearch topics to populate "
            "the log."
        )
    else:
        top_kind, top_n = sorted(agg["by_kind"].items(), key=lambda x: -x[1])[0]
        lines.append(
            f"The top issue kind is **{top_kind}** ({top_n} occurrence(s))."
        )
        lines.append("")
        lines.append(SUGGESTED_FIX.get(top_kind, "(no canned advice)"))
        lines.append("")
        lines.append(
            "Prioritize fixing this before other kinds; it's likely "
            "blocking the most successful runs."
        )

    return "\n".join(lines) + "\n"


def main():
    issues = load_jsonl(ISSUES_LOG)
    runs = load_jsonl(RUNS_LOG)
    agg = aggregate(issues, runs)
    md = render_report(agg)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print(f"wrote: {REPORT_PATH}")
    print()
    # Brief summary to stdout for cron-style use
    print(f"runs={agg['n_runs']} successful={agg['n_successful']} "
          f"issues={agg['n_issues']}")
    print("issues by kind:")
    for kind, n in sorted(agg["by_kind"].items(), key=lambda x: -x[1]):
        print(f" {kind:25s} {n}")


if __name__ == "__main__":
    main()
