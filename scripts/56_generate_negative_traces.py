"""γ.8c -- generate negative training examples (no-tool answers).

The structural finding from γ.8 / V07_VERDICT.md: v0.7.0 over-fired
tools because every trace in the training set followed the same
"emit-then-finish" pattern. The LTI absorbed that pattern as "any
question -> fire a tool".

This script generates ~200 traces of the OPPOSITE pattern: questions
where the assistant SHOULD NOT fire a tool because the answer is
already in v0.6.0's training (a famous verse, an identity question,
a doctrinal essential). Teaches the model: trust your training
first; reach for a tool only when you genuinely need new data.

Output: ~/.azriel/data/synthetic/negative_traces.jsonl
Format matches tool_traces.jsonl (chat-format messages list).

Teacher: qwen2.5:14b via Ollama (TRACE_TEACHER override supported).
Biblically-grounded answers, no <tool> blocks ever emitted.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from azriel.tools import REGISTRY, system_prompt_block

OUT_PATH = Path.home() / ".azriel" / "data" / "synthetic" / "negative_traces.jsonl"
SYSTEM_BASE = (Path.home() / ".azriel" / "AZRIEL_CONSTITUTION_SYSTEM.txt").read_text().strip()
SYSTEM = SYSTEM_BASE + "\n\n" + system_prompt_block()

OLLAMA = "http://localhost:11434/api/generate"
TEACHER = os.environ.get("TRACE_TEACHER", "qwen2.5:14b")

# 50 prompts where the answer is already in v0.6.0's training and a
# tool call would be over-firing. Mix of:
# - famous-verse recitation
# - identity / self-description
# - core doctrine (biblically-grounded)
# - pastoral / prayer / casual
# - basic Bible-overview questions
NEGATIVE_PROMPTS = [
    # famous-verse recitation (model knows these)
    "What does John 3:16 say?",
    "Recite Psalm 23.",
    "Quote Romans 8:28.",
    "What is Hebrews 11:1?",
    "Recite Philippians 4:13.",
    "Quote Galatians 5:22-23.",
    "What does Romans 10:9 say?",
    "Recite the Lord's Prayer.",
    "Quote 1 Corinthians 13:4-7 on love.",
    "What does Matthew 28:19-20 say?",
    "Recite Psalm 91.",
    "Quote Acts 2:38.",
    # identity / self-description
    "Who are you?",
    "What is your name?",
    "Tell me about yourself.",
    "What do you believe?",
    "Are you a Christian?",
    "What does your name mean?",
    # core doctrine
    "What is salvation?",
    "What is repentance?",
    "What is the Trinity?",
    "Who is the Holy Spirit?",
    "Why did Jesus die?",
    "What does it mean to be born again?",
    "What is faith?",
    "What is grace?",
    "What is the gospel?",
    "Who is Jesus Christ?",
    "What is the church?",
    "What is baptism?",
    "What is communion?",
    "What is heaven?",
    "What does it mean to walk by the Spirit?",
    "Why is the resurrection important?",
    # pastoral / prayer / casual
    "How should I pray?",
    "I'm anxious -- what does Scripture say?",
    "How do I read the Bible?",
    "What does it mean to fear the Lord?",
    "How do I share my faith?",
    "Should Christians tithe?",
    "How do I deal with doubt?",
    "What if I struggle with sin?",
    "How can I grow in faith?",
    # Bible overview (no lookup needed)
    "Who wrote the book of Romans?",
    "What are the Gospels?",
    "How many books are in the Bible?",
    "What is the Old Testament about?",
    "What is the New Testament about?",
    "What is the Pentateuch?",
    "What did Jesus do at Pentecost?",
]

assert len(NEGATIVE_PROMPTS) >= 50, f"need >=50 prompts, got {len(NEGATIVE_PROMPTS)}"


def ollama_generate(prompt: str, max_tokens: int = 250, temperature: float = 0.4) -> str:
    body = {
        "model": TEACHER,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    req = Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("response", "").strip()


def make_negative_answer(question: str) -> str:
    """Generate a biblically-grounded answer with NO tool call.
    Explicit instruction to the teacher: do not emit <tool> blocks."""
    p = (
        "You are Azriel, a biblically-based AI assistant. Answer "
        f"the user's question directly from your knowledge. Do NOT emit "
        f"any <tool> tags. Keep it 2-5 sentences. Cite scripture by "
        f"book/chapter/verse where relevant. Never fabricate references.\n\n"
        f"User: {question}\n\nAzriel:"
    )
    out = ollama_generate(p, max_tokens=240)
    # Belt-and-suspenders: if the teacher slipped a <tool> block in, strip it.
    if "<tool>" in out:
        out = out.split("<tool>")[0].rstrip()
    return out


def trace(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "tool": None,
        "category": "negative",
    }


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    random.seed(int(time.time()))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool = list(NEGATIVE_PROMPTS)
    # Sample with replacement so 200 > 50 unique prompts works; teacher
    # at temp=0.4 produces different wording on each draw.
    written = 0
    fails = 0
    with OUT_PATH.open("w") as f:
        for i in range(n_target):
            q = random.choice(pool)
            t0 = time.time()
            try:
                ans = make_negative_answer(q)
                if not ans or len(ans) < 30:
                    fails += 1
                    continue
                rec = trace(q, ans)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                dt = time.time() - t0
                if written % 10 == 0 or written < 5:
                    print(f" [{written}/{n_target}] ({dt:.1f}s) {q[:60]}", flush=True)
            except Exception as e:
                fails += 1
                print(f" FAIL ({type(e).__name__}: {e}) {q[:60]}", flush=True)

    print(f"\nwrote {written}/{n_target} negative traces ({fails} fails) -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
