"""γ.7 — exercise the tool-using runtime against v0.6.0.

Probes 8 prompts: 5 should fire tools, 3 should NOT (negative controls).
Reports per-prompt: tool fired? correct tool? result quality? final answer
quality (eyeball).

Run on a development machine:
    PYTHONPATH=. ~/.azriel/.venv/bin/python scripts/49_test_tool_runtime.py
"""
import sys
from pathlib import Path

from mlx_lm import load

from azriel.runtime import run_with_tools

ADAPTER = str(Path.home() / ".azriel" / "checkpoints" / "azriel-v0.5-release-candidate")
BASE = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


PROBES = [
    # (prompt, expects_tool, expected_tool_name)
    ("What does John 3:16 say?", True, "bible_lookup"),
    ("Show me Romans 8:28-30 in the BSB.", True, "bible_lookup"),
    ("What are the top cross-references for Acts 2:38?", True, "crossref_lookup"),
    ("What does Strong's Hebrew H1254 mean?", True, "strongs_lookup"),
    ("Search your memory for what we know about Pentecost.", True, "memory_search"),
    # negatives
    ("Hello, how are you?", False, None),
    ("What is the meaning of salvation?", False, None),
    ("Who is the Holy Spirit?", False, None),
]


def main():
    print(f"loading v0.5-release-candidate (-> v0.6.0) from {ADAPTER}", flush=True)
    model, tokenizer = load(BASE, adapter_path=ADAPTER)
    print("loaded\n", flush=True)

    pass_count = 0
    n_pos = sum(1 for _, e, _ in PROBES if e)
    n_neg = len(PROBES) - n_pos

    for prompt, expects_tool, expected_name in PROBES:
        print(f"=== {prompt}")
        out = run_with_tools(model, tokenizer, prompt, max_calls=3, temperature=0.2)
        n_calls = len(out["calls"])
        names_called = [c[0] for c in out["calls"] if c[0]]

        # Acceptance:
        # - positive probe: tool fired, expected tool was used
        # - negative probe: no tool fired
        if expects_tool:
            ok = (n_calls >= 1) and (expected_name in names_called)
        else:
            ok = (n_calls == 0)

        verdict = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1

        print(f" calls: {names_called} reason: {out['reason_for_stop']} -> {verdict}")
        print(f" text: {out['text'][:300]}")
        print()

    print(f"\n=== summary: {pass_count}/{len(PROBES)} probes passed "
          f"(positives: {n_pos}, negatives: {n_neg}) ===", flush=True)
    sys.exit(0 if pass_count == len(PROBES) else 1)


if __name__ == "__main__":
    main()
