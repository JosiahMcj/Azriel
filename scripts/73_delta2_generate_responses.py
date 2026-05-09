"""generate teacher responses for the distillation
seed set.

Reads ~/.azriel/data/delta2/seeds.jsonl (output of scripts/72), and
for each record asks the local Ollama teacher to produce the response
that matches `expected_pattern`. Filters out responses that don't pass
basic validation (empty, fake markup, off-pattern). Writes the survivors
to ~/.azriel/data/delta2/distilled.jsonl in mlx_lm chat format ready
for delta.2.5 LoRA training.

Default teacher: qwen2.5:32b via Ollama (already on a development machine, no API spend).
Override with --teacher gemma3:27b (or any Ollama-pulled tag).

Usage:
  python scripts/73_delta2_generate_responses.py \\
    --in ~/.azriel/data/delta2/seeds.jsonl \\
    --out ~/.azriel/data/delta2/distilled.jsonl \\
    --teacher qwen2.5:32b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"

# Constitution path (system prompt for the teacher's render). The
# teacher should produce Azriel-shaped responses, not generic responses.
CONSTITUTION_PATH = Path.home() / ".azriel" / "AZRIEL_CONSTITUTION_SYSTEM.txt"


def _load_constitution() -> str:
    try:
        return CONSTITUTION_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return "I am Azriel, a language-model tool grounded in Scripture."


def _ollama_generate(model: str, system: str, prompt: str,
                     timeout: int = 120) -> str:
    body = json.dumps({
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 800},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return f"__ERROR__: {e}"
    return (payload.get("response") or "").strip()


# ---------- Filter stack ----------

_FAKE_RESULT_RE = re.compile(r"<tool_result>", re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool>\s*(\w+)\s*\(.*?\)\s*</tool>", re.DOTALL | re.IGNORECASE)


def _passes_filters(record: dict, response: str) -> tuple[bool, str]:
    """Six-filter stack tailored to delta.2 distillation. Returns
    (ok, reason)."""
    if not response or len(response.strip()) < 3:
        return (False, "empty response")
    if response.startswith("__ERROR__"):
        return (False, "teacher error")
    if _FAKE_RESULT_RE.search(response):
        # The whole point is to teach the model NEVER to emit
        # <tool_result> -- if the teacher does it, that record poisons
        # the corpus.
        return (False, "teacher emitted <tool_result> markup")
    tier = record.get("tier")
    if tier == 1:
        # Tier 1 expects a single <tool>NAME(arg)</tool> line, name
        # matching the targeted tool.
        m = _TOOL_CALL_RE.search(response)
        if not m:
            return (False, "tier1: no <tool>...</tool> in response")
        target = (record.get("axis") or "").split(".", 1)[-1]
        if target and target != m.group(1):
            return (False, f"tier1: tool name mismatch (got {m.group(1)}, expected {target})")
        # Reject extra prose -- response should be just the tool call.
        stripped = re.sub(r"<tool>.*?</tool>", "", response, flags=re.DOTALL).strip()
        if len(stripped) > 80:
            return (False, "tier1: extra prose around tool call")
    elif tier == 2:
        # Tier 2 expects the planted fact text to appear (substring).
        expected = record.get("expected_pattern", "")
        # Pull a few content words from the planted fact and require
        # at least 2 of them in the response.
        tokens = [
            t for t in re.findall(r"[A-Za-z0-9]{4,}", expected)
            if t.lower() not in {"with", "from", "this", "that", "have", "name", "named"}
        ][:5]
        hits = sum(1 for t in tokens if t.lower() in response.lower())
        if hits < min(2, len(tokens)):
            return (False, f"tier2: response missed key fact tokens (hits={hits}/{len(tokens)})")
    elif tier == 3:
        # Tier 3 expects a <tool>document_create("...")</tool> shape
        # with the format token (docx/pptx/xlsx) preserved at the front.
        if "document_create" not in response or "<tool>" not in response:
            return (False, "tier3: no document_create tool call")
        if not re.search(r'document_create\(\s*"\s*(docx|pptx|xlsx)\|', response):
            return (False, "tier3: missing format|name|content shape")
    return (True, "ok")


def _to_mlxlm_record(record: dict, response: str) -> dict:
    """Convert (seed, response) to the messages-format record the LoRA
    trainer ingests. system is the constitution; the user turn is the
    seed prompt; the assistant turn is the filtered teacher response."""
    return {
        "messages": [
            {"role": "system", "content": _load_constitution()},
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True,
                    help="output dir; writes {out}/train.jsonl + {out}/valid.jsonl (mlx_lm format)")
    ap.add_argument("--teacher", default="qwen2.5:32b")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap prompts processed (0 = all)")
    ap.add_argument("--report", default=None,
                    help="write per-prompt status JSONL here (debug)")
    ap.add_argument("--valid-fraction", type=float, default=0.1,
                    help="fraction routed to valid.jsonl (default 0.1)")
    args = ap.parse_args()

    in_path = Path(args.in_path).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    report_path = Path(args.report).expanduser() if args.report else None

    with in_path.open() as f:
        seeds = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        seeds = seeds[: args.limit]
    print(f"loaded {len(seeds)} seeds, teacher={args.teacher}", flush=True)

    system = _load_constitution()

    kept = 0
    rejected = {}
    train_f = train_path.open("w")
    valid_f = valid_path.open("w")
    rep_f = report_path.open("w") if report_path else None
    t_start = time.time()
    import random as _random
    _random.seed(2026_05_05)

    for i, rec in enumerate(seeds, 1):
        prompt = (
            f"{rec['teacher_instruction']}\n\n"
            f"User message:\n{rec['prompt']}\n\n"
            f"Your response (single message, no prose framing):"
        )
        t0 = time.time()
        response = _ollama_generate(args.teacher, system, prompt)
        dt = time.time() - t0
        ok, reason = _passes_filters(rec, response)
        if rep_f:
            rep_f.write(json.dumps({
                "i": i, "axis": rec.get("axis"), "ok": ok,
                "reason": reason, "elapsed_s": round(dt, 2),
                "response_preview": response[:200],
            }) + "\n")
            rep_f.flush()
        if ok:
            line = json.dumps(_to_mlxlm_record(rec, response.strip()),
                              ensure_ascii=False) + "\n"
            target = valid_f if _random.random() < args.valid_fraction else train_f
            target.write(line)
            target.flush()
            kept += 1
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
        if i % 25 == 0:
            elapsed_total = time.time() - t_start
            eta = elapsed_total / i * (len(seeds) - i)
            print(f" [{i}/{len(seeds)}] kept={kept} "
                  f"reject={sum(rejected.values())} "
                  f"avg={elapsed_total/i:.1f}s/prompt "
                  f"ETA={eta/60:.1f}min", flush=True)

    train_f.close()
    valid_f.close()
    if rep_f:
        rep_f.close()

    total = len(seeds)
    pct = 100 * kept / max(1, total)
    print(f"\nDone. wrote {kept}/{total} ({pct:.1f}%) to {out_dir}/train.jsonl + valid.jsonl", flush=True)
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f}min", flush=True)
    if rejected:
        print("Rejected by reason:", flush=True)
        for r, n in sorted(rejected.items(), key=lambda x: -x[1]):
            print(f" {n:4d} {r}", flush=True)


if __name__ == "__main__":
    main()
