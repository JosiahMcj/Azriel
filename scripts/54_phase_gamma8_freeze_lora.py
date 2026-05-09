"""γ.8 -- Q5(c) recipe: freeze v0.6.0 LoRA, train ONLY LTI.

Approved in docs/QUEUED_QUESTIONS.md (Q5 answer = c).

Recap of why this recipe:
  v0.6.1 (stance-seed retrain) -> 8/8 refusals -> 5-6/8. REGRESSED.
  v0.7.0-pre (tool-trace retrain)-> 8/8 refusals -> 5/8. REGRESSED.

Common factor: in BOTH runs, gradient flowed through the LoRA delta
on top of v0.6.0. Identity refusals from v0.5 training are carried
in that delta, and any meaningful new signal there erodes them.

Q5(c) addresses this structurally:
  - Base Qwen weights: FROZEN (always; that was never a risk)
  - LoRA delta (v0.6.0): FROZEN (carries identity -- DO NOT MOVE)
  - LTI module (): TRAINABLE (zero-init residual; absorbs
                             whatever the new traces teach)
  - Tool heads: DORMANT scaffolding (see clarification
                             below)
  - LM head, embeddings: FROZEN

Identity stays anchored verbatim because the LoRA delta is the only
parameter group that holds it, and that group does not move.

Important scope clarification (re: Q5(c) wording):
  Q5(c) said "freeze LoRA, train ONLY LTI + tool heads." The
  ToolSignalHead / ToolArgsHead / ToolResultInjector modules in
  azriel/tool_heads.py are defined and constructable, but
  AzrielModel.__call__ does NOT invoke them in the forward pass --
  they are dormant scaffolding for a future trainer that
  computes auxiliary losses on tool emission/argument tokens. Without
  that auxiliary path, vanilla cross-entropy training over the
  lm_head returns ZERO gradient to those heads.

  So this script trains LTI only. Tool heads stay scaffolded with
  their bias-init weights for a future followup. This is
  honest and still does the structural job of Q5(c): identity stays
  put, new tool-use signal lives in LTI alone.

Output: ~/.azriel/checkpoints/lora-azriel-v0.7.0/ (separate from
the symlinked release-candidate; promotion is manual after eval).

USAGE (requires explicit user OK -- cron's 100-iter cap applies):

  # Sanity check first (no save, just freeze + 1 fwd/bwd):
  PYTHONPATH=. ~/.azriel/.venv/bin/python \\
    scripts/54_phase_gamma8_freeze_lora.py --dry-run

  # Real run:
  PYTHONPATH=. ~/.azriel/.venv/bin/python \\
    scripts/54_phase_gamma8_freeze_lora.py --iters 100

DATA: reads ~/.azriel/data/lora/{train,valid}.jsonl, the formatted
mix from scripts/21_format_for_lora.py. Re-run that prep first if you
want the latest 919-trace tool dataset folded in (current train.jsonl
on disk includes only the older 500-trace mix).
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx_lm import load
from mlx_lm.tuner import datasets, trainer
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.utils import linear_to_lora_layers

from azriel import AzrielConfig, AzrielModel

HOME = Path.home()
BASE_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
V06_ADAPTER = HOME / ".azriel" / "checkpoints" / "lora-azriel-v0.6.0"
DEFAULT_OUT = HOME / ".azriel" / "checkpoints" / "lora-azriel-v0.7.0"
DATA_DIR = HOME / ".azriel" / "data" / "lora"
TRACE_SOURCE = HOME / ".azriel" / "data" / "synthetic" / "tool_traces.jsonl"

LOOP_START = 32
LOOP_END = 48


def assert_data_fresh(allow_stale: bool) -> None:
    """Refuse to start if the formatted training set is older than the
    raw tool-trace corpus -- which means new traces were generated but
    21_format_for_lora.py has NOT been re-run, so this training will
    burn cycles on a stale dataset and the new tools won't move.

    Override with --allow-stale-data only if you know what you're doing
    (e.g., re-running with a deliberately older snapshot)."""
    train = DATA_DIR / "train.jsonl"
    if not train.exists():
        raise SystemExit(
            f"data not formatted yet: {train} missing. Run "
            "your training-data prep script first (the data-prep "
            "pipeline lives separately from this runtime repo)."
        )
    if not TRACE_SOURCE.exists():
        # Trace source missing means there's nothing newer -- can't be stale.
        return
    train_mtime = train.stat().st_mtime
    trace_mtime = TRACE_SOURCE.stat().st_mtime
    if trace_mtime > train_mtime:
        delta_min = (trace_mtime - train_mtime) / 60.0
        msg = (
            f"DATA STALENESS: {train} is older than {TRACE_SOURCE} by "
            f"{delta_min:.0f} minutes. The new traces have NOT been folded "
            "into the formatted training set yet. Running training now would "
            "burn cycles on stale data and the new tool kinds (pdf_extract, "
            "weather, github_query, etc.) will not move.\n\n"
            "Fix:\n"
            " cd <your-data-prep-dir> && PYTHONPATH=. \\\n"
            " ~/.azriel/.venv/bin/python <your-format-for-lora-script>.py\n\n"
            "Then verify ~/.azriel/data/lora/stats.json reflects the new "
            "trace count, and re-run this trainer.\n"
        )
        if allow_stale:
            print("WARN: --allow-stale-data set; proceeding anyway.\n" + msg, flush=True)
        else:
            raise SystemExit(msg)


def count_params(module: nn.Module) -> tuple[int, int]:
    """Returns (total, trainable). MLX 'trainable_parameters()' walks the
    tree and yields only un-frozen leaves; 'parameters()' yields all."""
    def total(items):
        n = 0
        for v in items:
            if isinstance(v, mx.array):
                n += int(v.size)
            elif isinstance(v, dict):
                n += total(v.values())
            elif isinstance(v, (list, tuple)):
                n += total(v)
        return n
    return total(module.parameters().values()), total(
        module.trainable_parameters().values()
    )


def build_wrapper() -> tuple[AzrielModel, object]:
    """Load base + v0.6.0 LoRA + LTI; freeze everything except LTI."""
    print(f"[{time.strftime('%H:%M:%S')}] loading base Qwen3-Coder-30B-A3B...", flush=True)
    base, tokenizer = load(BASE_MODEL)

    src_cfg = json.loads((V06_ADAPTER / "adapter_config.json").read_text())
    num_layers = src_cfg["num_layers"]
    lora_params = src_cfg["lora_parameters"]
    print(f"v0.6.0 adapter cfg: num_layers={num_layers}, lora_parameters={lora_params}", flush=True)

    # Standard mlx_lm pattern. After this, base has LoRA modules attached
    # and base.parameters() yields the lora_a/lora_b deltas as trainable.
    base.freeze()
    linear_to_lora_layers(base, num_layers, lora_params, use_dora=False)
    base.load_weights(str(V06_ADAPTER / "adapters.safetensors"), strict=False)

    cfg = AzrielConfig(
        loop_layer_start=LOOP_START,
        loop_layer_end=LOOP_END,
        loop_max_iters=2,
        lti_enabled=True,
        lti_expansion=1,
        # Tool heads: enabled=False keeps the scaffolding modules from being
        # constructed at all. They cannot be trained via vanilla CE anyway
        # (see module docstring); leaving them off saves a few hundred K
        # params and removes any chance of dead-weight gradient noise.
        tool_heads_enabled=False,
    )
    print(f"wrapping config: {cfg}", flush=True)
    wrapper = AzrielModel(base, cfg)

    # ---- THE FREEZE STEP ----
    # Recursively freeze EVERYTHING in the wrapper: base weights, LoRA
    # deltas (which linear_to_lora_layers had marked trainable), embed,
    # lm_head, the loop's borrowed mid-layer references, etc.
    wrapper.freeze(recurse=True)
    # Then selectively un-freeze JUST the LTI subtree. After this,
    # wrapper.trainable_parameters() yields only lti.down.weight and
    # lti.up.weight.
    if wrapper.lti is None:
        raise RuntimeError(
            "config has lti_enabled=False; cannot run the LTI-only recipe."
        )
    wrapper.lti.unfreeze(recurse=True)

    return wrapper, tokenizer


def report_freeze(wrapper: AzrielModel) -> None:
    total, trainable = count_params(wrapper)
    pct = 100.0 * trainable / max(total, 1)
    print(
        f"[{time.strftime('%H:%M:%S')}] params total={total:,} "
        f"trainable={trainable:,} ({pct:.4f}%)",
        flush=True,
    )
    if trainable == 0:
        raise RuntimeError("freeze logic broken -- nothing trainable!")
    # Sanity: dump the trainable param names to confirm they're all under
    # 'lti'. Anything else means we accidentally left LoRA / tool heads
    # in.
    def walk(prefix, obj, out):
        if isinstance(obj, mx.array):
            out.append((prefix, obj.shape))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v, out)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(f"{prefix}[{i}]", v, out)
    rows = []
    walk("", wrapper.trainable_parameters(), rows)
    print(f"trainable parameter leaves ({len(rows)}):", flush=True)
    for name, shape in rows[:20]:
        print(f" {name} {shape}", flush=True)
    if len(rows) > 20:
        print(f" ... and {len(rows) - 20} more", flush=True)
    bad = [n for (n, _) in rows if "lti" not in n]
    if bad:
        raise RuntimeError(
            f"unexpected non-lti trainable params: {bad[:5]}... "
            "freeze logic is wrong; aborting."
        )


def dry_run(wrapper: AzrielModel, tokenizer) -> None:
    """One forward + one backward on a tiny sample. Confirms grads flow
    only through LTI and the loss is finite. Saves nothing."""
    print(f"[{time.strftime('%H:%M:%S')}] DRY RUN: 1 fwd/bwd on sample", flush=True)
    sample = (
        "<|im_start|>system\nI am Azriel.<|im_end|>\n"
        "<|im_start|>user\nWhat does John 3:16 say?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    ids = mx.array(tokenizer.encode(sample))[None, :]
    targets = ids # just CE on the prompt itself for shape sanity

    def loss_fn(model, x, y):
        logits = model(x)
        # next-token CE on the trailing position
        return nn.losses.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            y[:, 1:].reshape(-1),
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(wrapper, loss_fn)
    loss, grads = loss_and_grad(wrapper, ids, targets)
    mx.eval(loss, grads)
    print(f" loss = {float(loss):.4f}", flush=True)

    # Compute total gradient norm across LTI params. If it's ~0, the
    # forward path isn't actually invoking LTI; if it's nonzero, the
    # plumbing works.
    def gnorm(d):
        s = 0.0
        if isinstance(d, mx.array):
            return float(mx.sum(d * d))
        if isinstance(d, dict):
            for v in d.values():
                s += gnorm(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                s += gnorm(v)
        return s
    g2 = gnorm(grads)
    print(f" ||grad||_F^2 over trainable params = {g2:.6e}", flush=True)
    if g2 == 0.0:
        print(" WARNING: zero gradient -- LTI may not be reached by forward pass", flush=True)
    else:
        print(" OK: gradient flows to LTI.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="freeze + 1 fwd/bwd + report; no save, no training")
    ap.add_argument("--iters", type=int, default=100,
                    help="training iterations (cron's hard cap is 100)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="LTI is small; can use higher LR than full-LoRA training")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--steps-per-save", type=int, default=25)
    ap.add_argument("--allow-stale-data", action="store_true",
                    help="bypass the freshness check on data/lora vs "
                         "data/synthetic/tool_traces.jsonl (not recommended)")
    args = ap.parse_args()

    # Freshness guard: refuse to train on a stale dataset where the
    # newly-generated tool traces haven't been re-formatted yet.
    if not args.dry_run:
        assert_data_fresh(allow_stale=args.allow_stale_data)

    if args.iters > 100 and not args.dry_run:
        raise SystemExit(
            "--iters > 100 requires explicit user approval per cron rule. "
            "Either lower --iters or run with --dry-run."
        )

    wrapper, tokenizer = build_wrapper()
    report_freeze(wrapper)

    if args.dry_run:
        dry_run(wrapper, tokenizer)
        print(f"[{time.strftime('%H:%M:%S')}] DRY RUN done. No checkpoint saved.", flush=True)
        return

    # Real run
    print(f"[{time.strftime('%H:%M:%S')}] loading dataset from {DATA_DIR}", flush=True)
    args_obj = type("A", (), {
        "data": str(DATA_DIR),
        "train": True, "test": False,
        "hf_dataset": None, "config": {},
        "chat_template": None,
        "mask_prompt": False,
    })()
    train_ds, valid_ds, _ = datasets.load_dataset(args_obj, tokenizer)
    train_ds = CacheDataset(train_ds)
    valid_ds = CacheDataset(valid_ds)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = out_dir / "adapters.safetensors"
    # Re-use v0.6.0's adapter_config.json so downstream loaders recognize
    # the LoRA layout. The actual LoRA deltas are unchanged in this run;
    # the new file holds LTI weights.
    (out_dir / "adapter_config.json").write_text(
        (V06_ADAPTER / "adapter_config.json").read_text()
    )
    # Stamp a recipe marker so future you remembers what this checkpoint
    # is.
    (out_dir / "RECIPE.txt").write_text(
        "γ.8 / Q5(c): v0.6.0 LoRA frozen; trained ONLY LTI on data/lora/\n"
        "Promote by re-pointing ~/.azriel/checkpoints/azriel-v0.5-release-candidate\n"
        "ONLY after Phase 1 8/8, Phase 2 10/10, doctrinal pent>=17/alt<=3.\n"
    )

    train_args = trainer.TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=5,
        steps_per_report=10,
        steps_per_eval=200,
        steps_per_save=args.steps_per_save,
        max_seq_length=args.max_seq,
        adapter_file=str(adapter_file),
        grad_checkpoint=True,
    )
    optimizer = opt.AdamW(learning_rate=args.lr)

    print(
        f"[{time.strftime('%H:%M:%S')}] starting γ.8 LTI-only training: "
        f"iters={args.iters} batch={args.batch_size} max_seq={args.max_seq} "
        f"lr={args.lr}",
        flush=True,
    )
    t0 = time.time()
    trainer.train(
        model=wrapper,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=valid_ds,
        args=train_args,
    )
    dt = time.time() - t0
    print(
        f"[{time.strftime('%H:%M:%S')}] γ.8 training done in {dt:.0f}s. "
        f"Checkpoint at {adapter_file}.",
        flush=True,
    )
    print("Next: run scripts/41_eval_v06.py against the new checkpoint to "
          "verify Phase 1 8/8 + Phase 2 10/10 + doctrinal floor before "
          "promoting the symlink.", flush=True)


if __name__ == "__main__":
    main()
