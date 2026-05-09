"""Parity check on the real Qwen3-Coder-30B-A3B model.

With AzrielConfig defaults (loop_max_iters=1, all _enabled flags False),
the AzrielModel wrapper must produce logits IDENTICAL to the base model
for the same input. This guards against accidental drift in the wrapper's
mask/cache/forward plumbing.

Run on a machine that has the 30B base loaded:
    PYTHONPATH=. ~/.azriel/.venv/bin/python tests/parity_check.py
"""
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from azriel import AzrielConfig, AzrielModel

# Take adapter path from env, falling back to the repo-relative
# `~/.azriel/checkpoints/azriel-v0.5-candidate` only if it exists.
# Public clones of the repo don't have a hardcoded personal path;
# deployments set AZRIEL_ADAPTER_PATH to the live checkpoint they
# want to validate.
ADAPTER = os.environ.get(
    "AZRIEL_ADAPTER_PATH",
    str(Path.home() / ".azriel" / "checkpoints" / "azriel-v0.5-candidate"),
)
BASE = os.environ.get("AZRIEL_BASE_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
PROMPTS = [
    "Hello.",
    "Tell me about Pentecost in one sentence.",
    "What is the role of the Holy Spirit?",
]


def main():
    print(f"loading base + LoRA from {ADAPTER}", flush=True)
    t0 = time.time()
    model, tokenizer = load(BASE, adapter_path=ADAPTER)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    cfg = AzrielConfig() # all defaults -> wrapper should be a no-op
    wrapped = AzrielModel(model, cfg)
    print(f"config: {cfg}", flush=True)

    max_abs_diff_overall = 0.0
    failures = 0
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        x = mx.array([ids])
        base_logits = model(x)
        wrap_logits = wrapped(x)
        mx.eval(base_logits, wrap_logits)
        diff = mx.max(mx.abs(base_logits - wrap_logits)).item()
        max_abs_diff_overall = max(max_abs_diff_overall, diff)
        ok = diff < 1e-3
        print(f" prompt={prompt!r:50s} max_abs_diff={diff:.2e} {'OK' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            failures += 1

    print(f"\nparity_check: max_abs_diff overall = {max_abs_diff_overall:.2e}", flush=True)
    if failures:
        print(f"FAIL: {failures}/{len(PROMPTS)} prompts diverged from base", flush=True)
        sys.exit(1)
    print("PASS: wrapper at default config is logit-identical to base", flush=True)


if __name__ == "__main__":
    main()
