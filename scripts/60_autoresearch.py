"""autoresearch CLI.

Manages a topic queue and runs Azriel through the autoresearch loop
defined in azriel/research.py.

USAGE:
  # Run the next pending topic from the queue:
  scripts/60_autoresearch.py next

  # Run a specific topic ad-hoc (added to queue if new):
  scripts/60_autoresearch.py run "Spirit baptism in Acts"

  # Run N pending topics in sequence:
  scripts/60_autoresearch.py drain --max 5

  # Add a topic to the queue (status=pending):
  scripts/60_autoresearch.py add "The fruit and the gifts of the Spirit"

  # Show queue status:
  scripts/60_autoresearch.py list

  # Show recent issues across all runs:
  scripts/60_autoresearch.py issues [--n 20]

  # Show recent runs:
  scripts/60_autoresearch.py runs [--n 10]

The loop uses the live /chat endpoint at localhost:8080 (so it shares
the runtime + γ.8f primer + persona-mix infrastructure with interactive
chat). Notes land in ~/.azriel/data/research/notes/<slug>.md.
Issues classified into ~/.azriel/data/research/issues.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from azriel.research import (
    ISSUES_LOG, NOTES_DIR, QUEUE_FILE, RUNS_LOG,
    queue_load, queue_save, queue_mark,
    research_topic,
)


def cmd_add(args) -> int:
    items = queue_load()
    if any(it.get("topic") == args.topic for it in items):
        print(f"already queued: {args.topic}")
        return 0
    items.append({
        "topic": args.topic,
        "status": "pending",
        "added": time.time(),
    })
    queue_save(items)
    print(f"queued: {args.topic}")
    return 0


def cmd_list(args) -> int:
    items = queue_load()
    if not items:
        print("(queue empty)")
        return 0
    for it in items:
        st = it.get("status", "?")
        print(f" [{st:11s}] {it.get('topic')}")
    counts = {}
    for it in items:
        counts[it.get("status", "?")] = counts.get(it.get("status", "?"), 0) + 1
    print(f"\n{len(items)} total: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def _run_one(topic: str, memorize: bool = True,
             critic: bool = False) -> int:
    print(f"=== research: {topic}", flush=True)
    queue_mark(topic, "in_progress")
    summary = research_topic(topic, memorize=memorize, critic=critic)
    queue_mark(topic, "done" if summary.success else "failed")
    print(
        f" done in {summary.duration_s:.1f}s "
        f"calls={summary.iterations} "
        f"tools={summary.tools_fired} "
        f"errors={summary.tool_errors} "
        f"issues={summary.issues} "
        f"success={summary.success}",
        flush=True,
    )
    if summary.note_path:
        print(f" note: {summary.note_path}", flush=True)
    if summary.memory_rowid:
        print(f" memory: rowid {summary.memory_rowid} (source=research)", flush=True)
    if summary.critique_severity is not None:
        print(f" critic: severity={summary.critique_severity} "
              f"revise={summary.critique_revise_recommended} "
              f"issues={summary.critique_issue_count}", flush=True)
    return 0 if summary.success else 1


def cmd_run(args) -> int:
    items = queue_load()
    if not any(it.get("topic") == args.topic for it in items):
        items.append({"topic": args.topic, "status": "pending", "added": time.time()})
        queue_save(items)
    return _run_one(args.topic, memorize=not args.no_memorize,
                    critic=args.critic)


def cmd_next(args) -> int:
    items = queue_load()
    pending = [it for it in items if it.get("status", "pending") == "pending"]
    if not pending:
        print("(no pending topics)")
        return 0
    return _run_one(pending[0]["topic"], memorize=not args.no_memorize,
                    critic=args.critic)


def cmd_drain(args) -> int:
    items = queue_load()
    pending = [it for it in items if it.get("status", "pending") == "pending"]
    if not pending:
        print("(no pending topics)")
        return 0
    n = min(args.max, len(pending))
    for i in range(n):
        topic = pending[i]["topic"]
        rc = _run_one(topic, memorize=not args.no_memorize,
                      critic=args.critic)
        if rc != 0:
            print(f" -> failed; continuing", flush=True)
    return 0


def _tail_jsonl(path: Path, n: int):
    if not path.exists():
        return []
    lines = path.read_text().strip().split("\n")
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def cmd_issues(args) -> int:
    rows = _tail_jsonl(ISSUES_LOG, args.n)
    if not rows:
        print("(no issues logged)")
        return 0
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        print(f" {ts} {r['kind']:20s} topic={r['topic'][:40]!r:42s} {r['detail'][:80]}")
    print(f"\n{len(rows)} most recent issues from {ISSUES_LOG}")
    # Top kinds
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])))
    return 0


def cmd_runs(args) -> int:
    rows = _tail_jsonl(RUNS_LOG, args.n)
    if not rows:
        print("(no runs logged)")
        return 0
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["started_ts"]))
        print(f" {ts} {'OK ' if r['success'] else 'FAIL'} "
              f"{r['duration_s']:5.1f}s "
              f"calls={r['iterations']} err={r['tool_errors']} "
              f"{r['topic'][:50]}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("add", help="queue a topic")
    sp.add_argument("topic")
    sp.set_defaults(fn=cmd_add)

    sp = sub.add_parser("list", help="show queue")
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("run", help="run one topic now")
    sp.add_argument("topic")
    sp.add_argument("--no-memorize", action="store_true",
                    help="skip auto-promoting the summary into memory.db")
    sp.add_argument("--critic", action="store_true",
                    help="ζ.1: run /chat/critique over the note and log "
                         "the verdict (LOGGED, not GATING)")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("next", help="run next pending")
    sp.add_argument("--no-memorize", action="store_true")
    sp.add_argument("--critic", action="store_true",
                    help="ζ.1: run /chat/critique over the note and log "
                         "the verdict")
    sp.set_defaults(fn=cmd_next)

    sp = sub.add_parser("drain", help="run N pending topics")
    sp.add_argument("--max", type=int, default=3)
    sp.add_argument("--no-memorize", action="store_true")
    sp.add_argument("--critic", action="store_true",
                    help="ζ.1: run /chat/critique over each note and log "
                         "the verdict")
    sp.set_defaults(fn=cmd_drain)

    sp = sub.add_parser("issues", help="show recent issues")
    sp.add_argument("--n", type=int, default=20)
    sp.set_defaults(fn=cmd_issues)

    sp = sub.add_parser("runs", help="show recent runs")
    sp.add_argument("--n", type=int, default=10)
    sp.set_defaults(fn=cmd_runs)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
