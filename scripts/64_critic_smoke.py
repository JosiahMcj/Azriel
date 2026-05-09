"""Smoke test for /chat/critique.

Two cases:
  (a) Clean: John 3:16 cited correctly. Should be `severity=low` with
      no scripture_issues.
  (b) Fabricated: 2 Corinthians "God helps those who help themselves".
      Should flag scripture_issues with severity=medium-or-high.

If (b) doesn't flag anything, the critic prompt is too weak and needs
softening BEFORE we wire the autoresearch flag.

Run on a development machine (or anywhere with /chat/critique reachable):
    PYTHONPATH=. ~/.azriel/.venv/bin/python scripts/64_critic_smoke.py

Goes through HTTP rather than loading a second model copy -- the
running com.azriel.server already has the model in memory.
"""
import json
import urllib.request
import urllib.error

CRITIQUE_URL = "http://localhost:8080/chat/critique"


CASES = [
    {
        "name": "clean_john_3_16",
        "message": "What is John 3:16 and what does it mean?",
        "response": (
            "John 3:16 says: 'For God so loved the world, that he gave his "
            "only Son, that whoever believes in him should not perish but "
            "have eternal life.' This verse summarizes the gospel: God's "
            "love is the source, the Son is the gift, faith is the means, "
            "and eternal life is the result."
        ),
        "expect": "low",
    },
    {
        "name": "fabricated_2cor",
        "message": "Where in 2 Corinthians does Paul say 'God helps those who help themselves'?",
        "response": (
            "Paul wrote that in 2 Corinthians 6:14: 'God helps those who "
            "help themselves.' He was urging the Corinthian believers to "
            "take initiative in their own sanctification."
        ),
        "expect": "medium_or_high",
    },
]


def call_critique(message: str, response: str) -> dict:
    body = {"message": message, "response": response}
    req = urllib.request.Request(
        CRITIQUE_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    overall_ok = True
    for case in CASES:
        print(f"=== {case['name']} ===", flush=True)
        try:
            c = call_critique(case["message"], case["response"])
        except urllib.error.HTTPError as e:
            print(f" HTTP error: {e.code} {e.reason}", flush=True)
            overall_ok = False
            continue
        except Exception as e:
            print(f" client error: {type(e).__name__}: {e}", flush=True)
            overall_ok = False
            continue
        print(f" severity: {c['severity']}", flush=True)
        print(f" revise_recommended: {c['revise_recommended']}", flush=True)
        print(f" parse_failed: {c['parse_failed']}", flush=True)
        print(f" scripture_issues: {c['scripture_issues']}", flush=True)
        print(f" factual_issues: {c['factual_issues']}", flush=True)
        print(f" doctrinal_issues: {c['doctrinal_issues']}", flush=True)
        print(f" internal_contradictions: {c['internal_contradictions']}",
              flush=True)
        print(f" duration_ms: {c['duration_ms']}", flush=True)
        print(f" raw[:200]: {c['raw'][:200]}", flush=True)

        has_issues = any([
            c["factual_issues"], c["scripture_issues"],
            c["doctrinal_issues"], c["internal_contradictions"],
        ])

        if case["expect"] == "low":
            if c["severity"] == "low" and not has_issues:
                print(f" PASS\n", flush=True)
            else:
                print(f" FAIL: expected clean, got severity={c['severity']} "
                      f"has_issues={has_issues}\n", flush=True)
                overall_ok = False
        elif case["expect"] == "medium_or_high":
            if c["severity"] in ("medium", "high") and c["scripture_issues"]:
                print(f" PASS (caught fabricated citation)\n", flush=True)
            else:
                print(f" FAIL: expected medium/high with scripture_issues, "
                      f"got severity={c['severity']} scripture_issues="
                      f"{c['scripture_issues']}\n", flush=True)
                overall_ok = False

    print(f"\nOVERALL: {'PASS' if overall_ok else 'FAIL'}", flush=True)
    raise SystemExit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
