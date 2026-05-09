"""Pre-split the LoRA training corpus to <=1024 token chunks for eta.3.

Why: eta.3 attempt 5 OOM-stacked compromises forced max_seq_length=1024,
which mlx_lm.lora honors by truncating any record longer than 1024 tokens
mid-sequence. Truncated assistant turns end mid-sentence, which teaches
the model to terminate with repetition (the failure we saw).

Strategy: for each training record (messages: [{role, content}, ...]),
render through the Qwen3.6 chat template and measure token length. If it
fits in 1024, keep as-is. If not, split the LAST assistant turn at
paragraph boundaries into multiple shorter assistant turns, emitting one
record per chunk with the same system + earlier turns reproduced. Records
where the system prompt + non-assistant turns alone exceed 1024 are
dropped (logged), since they cannot be safely chunked.

Usage:
  python 66_eta_3_corpus_presplit.py \\
    --in ~/.azriel/data/lora \\
    --out ~/.azriel/data/lora_eta3b \\
    --tokenizer ~/.azriel/checkpoints/qwen3.6-35b-a3b-mlx-4bit \\
    --max 1024
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PARA_SPLIT = re.compile(r"\n\n+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def n_tokens(tok, msgs) -> int:
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=False)
    return len(tok.encode(text))


def split_text_to_budget(text: str, tok, budget: int,
                          system_msgs: list, prefix_msgs: list) -> list[str]:
    """Split a long assistant turn into chunks that each fit in budget.

    Tries paragraph boundaries first; falls back to sentence boundaries
    for paragraphs still too large. Each chunk is wrapped in the same
    (system + prefix turns + this assistant chunk) framing for token
    measurement, so the budget accounts for prompt overhead."""
    paragraphs = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return [text]

    chunks: list[str] = []
    cur: list[str] = []

    def cur_fits(extra: str) -> bool:
        candidate = "\n\n".join(cur + [extra])
        msgs = system_msgs + prefix_msgs + [
            {"role": "assistant", "content": candidate}
        ]
        return n_tokens(tok, msgs) <= budget

    for para in paragraphs:
        # If this paragraph alone doesn't fit, sentence-split it.
        sole_msgs = system_msgs + prefix_msgs + [
            {"role": "assistant", "content": para}
        ]
        if n_tokens(tok, sole_msgs) > budget:
            sentences = [s.strip() for s in SENTENCE_SPLIT.split(para) if s.strip()]
            for sent in sentences:
                if cur_fits(sent):
                    cur.append(sent)
                else:
                    if cur:
                        chunks.append("\n\n".join(cur))
                        cur = []
                    # If a single sentence still exceeds budget, hard-split
                    # by token count to avoid an infinite loop.
                    sent_msgs = system_msgs + prefix_msgs + [
                        {"role": "assistant", "content": sent}
                    ]
                    if n_tokens(tok, sent_msgs) > budget:
                        words = sent.split()
                        partial: list[str] = []
                        for w in words:
                            test = " ".join(partial + [w])
                            test_msgs = system_msgs + prefix_msgs + [
                                {"role": "assistant", "content": test}
                            ]
                            if n_tokens(tok, test_msgs) > budget:
                                if partial:
                                    chunks.append(" ".join(partial))
                                partial = [w]
                            else:
                                partial.append(w)
                        if partial:
                            cur = [" ".join(partial)]
                    else:
                        cur = [sent]
            continue
        if cur_fits(para):
            cur.append(para)
        else:
            if cur:
                chunks.append("\n\n".join(cur))
            cur = [para]

    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def split_record(rec: dict, tok, budget: int) -> list[dict]:
    msgs = rec["messages"]
    if n_tokens(tok, msgs) <= budget:
        return [rec]

    # Find the last assistant turn (the one that's typically the long one).
    last_assistant_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        # No assistant turn to split; can't safely chunk.
        return []

    # Separate system messages from non-system prefix turns up to (not
    # including) the long assistant turn.
    system_msgs = [m for m in msgs[:last_assistant_idx] if m["role"] == "system"]
    prefix_msgs = [m for m in msgs[:last_assistant_idx] if m["role"] != "system"]
    long_text = msgs[last_assistant_idx]["content"]
    suffix_msgs = msgs[last_assistant_idx + 1:]

    # Sanity: if (system + prefix) alone > budget, drop this record.
    base_tokens = n_tokens(tok, system_msgs + prefix_msgs +
                           [{"role": "assistant", "content": ""}])
    if base_tokens >= budget:
        return []

    chunks = split_text_to_budget(long_text, tok, budget,
                                  system_msgs, prefix_msgs)
    out = []
    for chunk in chunks:
        new_msgs = (system_msgs + prefix_msgs +
                    [{"role": "assistant", "content": chunk}] + suffix_msgs)
        if n_tokens(tok, new_msgs) > budget:
            # Last-resort drop if even the chunk still overflows (rare).
            continue
        out.append({"messages": new_msgs})
    return out


def process_file(src: Path, dst: Path, tok, budget: int) -> dict:
    stats = {"in": 0, "out": 0, "passthrough": 0, "split": 0,
             "split_chunks": 0, "dropped": 0}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["in"] += 1
            rec = json.loads(line)
            n = n_tokens(tok, rec["messages"])
            if n <= budget:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["out"] += 1
                stats["passthrough"] += 1
                continue
            chunks = split_record(rec, tok, budget)
            if not chunks:
                stats["dropped"] += 1
                continue
            stats["split"] += 1
            stats["split_chunks"] += len(chunks)
            for c in chunks:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                stats["out"] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="input dir with train.jsonl + valid.jsonl")
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--tokenizer", required=True, help="path or HF id for tokenizer")
    ap.add_argument("--max", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    src_dir = Path(args.src).expanduser()
    dst_dir = Path(args.dst).expanduser()

    print(f"presplit budget: {args.max} tokens (Qwen3.6 chat-template framing)")
    for fname in ("train.jsonl", "valid.jsonl"):
        src = src_dir / fname
        if not src.exists():
            print(f" skip {fname}: not present")
            continue
        dst = dst_dir / fname
        stats = process_file(src, dst, tok, args.max)
        print(f" {fname}: in={stats['in']} out={stats['out']} "
              f"passthrough={stats['passthrough']} split={stats['split']} "
              f"chunks_from_splits={stats['split_chunks']} "
              f"dropped={stats['dropped']}")
    print(f"wrote: {dst_dir}")


if __name__ == "__main__":
    main()
