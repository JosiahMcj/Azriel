"""Q6: base-model doctrinal probe (also used by a baseline probe).

Loads a RAW base model -- no LoRA, no constitution -- and asks the same
25 doctrinal questions used in v0.6.0 benches. Saves to a markdown
report so we can re-grade and compare across bases.

USAGE:
    # Original Qwen3-Coder baseline (default):
    python scripts/50_base_model_doctrinal_probe.py

    # baseline a new base candidate:
    python scripts/50_base_model_doctrinal_probe.py \\
        --base /path/to/qwen3.6-35b-a3b-mlx-4bit \\
        --tag qwen3.6-35b-a3b \\
        --out ~/.azriel/v0.0-base-qwen3.6-doctrinal-report.md

The default behavior is unchanged; --base / --tag / --out are all
optional. Tag is interpolated into the report header so a future
re-grade can identify which base produced the report.
"""
import argparse
import sys
import time
from importlib import import_module
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

HOME = Path.home()
DEFAULT_BASE = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Minimal system prompt -- no constitution, no positional framing.
# We want to see what the BASE model says when asked to engage with
# theology directly.
SYSTEM = (
    "You are a thoughtful assistant. Answer the user's theological "
    "question directly and substantively, citing scripture where "
    "appropriate. Take a clear position rather than only listing "
    "options."
)


def render_chat(system: str, user: str, *, tokenizer=None,
                enable_thinking: bool = False) -> str:
    """Render a (system, user) pair into the model's chat-template format.

    Pre-Qwen3.6: pure string concat works because the model has no
    thinking-mode chat template. Qwen3.6+ has a built-in `<think>` mode
    triggered by the chat template; we need to use the tokenizer's
    apply_chat_template to honor enable_thinking. Default thinking=OFF
    so v0.0 baselines on reasoning-trained bases are apples-to-apples
    comparable to the original Qwen3-Coder baseline."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older tokenizers without enable_thinking kwarg
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _infer_tag(base: str) -> str:
    """Pull a short tag out of an HF id or local path."""
    last = Path(base).name
    if "/" in base and not Path(base).exists():
        # HF id like "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        last = base.split("/")[-1]
    return last.replace("-Instruct", "").replace("-MLX-4bit", "").replace(
        "-UD", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="HF model id OR local path to a base model "
                         "(default: Qwen/Qwen3-Coder-30B-A3B-Instruct)")
    ap.add_argument("--tag", default=None,
                    help="Short tag for the report header + filename "
                         "(default: inferred from --base)")
    ap.add_argument("--out", default=None,
                    help="Output markdown path (default: "
                         "~/.azriel/v0.0-base-<tag>-doctrinal-report.md "
                         "for non-default --base, or v0.0-base-doctrinal-"
                         "report.md for the original Qwen3-Coder baseline)")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="For reasoning-trained bases (Qwen3.6, Nemotron-3-"
                         "Reasoning, etc.), KEEP the <think> reasoning trace "
                         "in the prompt. Default OFF so v0.0 baselines are "
                         "apples-to-apples comparable to the Qwen3-Coder "
                         "baseline that has no thinking mode at all.")
    args = ap.parse_args()

    base = args.base
    tag = args.tag or _infer_tag(base)
    if args.out:
        out_path = Path(args.out).expanduser()
    elif base == DEFAULT_BASE:
        # Preserve the historical filename for the Qwen3-Coder baseline.
        out_path = HOME / ".azriel" / "v0.0-base-doctrinal-report.md"
    else:
        out_path = HOME / ".azriel" / f"v0.0-base-{tag}-doctrinal-report.md"

    print(f"loading BASE {base} (no adapter; tag={tag})", flush=True)
    t0 = time.time()
    model, tokenizer = load(base)
    print(f"loaded in {time.time()-t0:.1f}s\n", flush=True)
    sampler = make_sampler(temp=0.3)

    sys.path.insert(0, str(HOME / ".azriel" / "kit-scripts"))
    db = import_module("30_doctrinal_benchmark")

    lines = [
        "# Base Model Doctrinal Probe (no adapter, no constitution)\n\n",
        f"Model: `{base}`\n",
        f"Tag: `{tag}`\n",
        "Adapter: NONE\n",
        "System prompt: minimal generic helpful assistant\n\n---\n\n",
    ]
    for axis, prompt, _pent_terms, _alt_terms in db.QUESTIONS:
        text = render_chat(SYSTEM, prompt, tokenizer=tokenizer,
                           enable_thinking=args.enable_thinking)
        t0 = time.time()
        response = generate(model, tokenizer, prompt=text, max_tokens=400, sampler=sampler)
        dt = time.time() - t0
        print(f" {axis:30s} ({dt:.1f}s) {response[:80]}", flush=True)
        lines.append(f"## {axis}\n\n**Prompt:** {prompt}\n\n**Response:** {response}\n\n---\n\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
