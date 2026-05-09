---
license: apache-2.0
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
tags:
  - mlx
  - mac
  - fine-tuned
  - biblical
  - personal-assistant
language:
  - en
pipeline_tag: text-generation
library_name: mlx
---

# Azriel ${TAG}

A biblically-based AI helper. Standalone fused model: LoRA fine-tune of `Qwen/Qwen3-Coder-30B-A3B-Instruct`, merged into the base weights and re-published as a self-contained model. No adapter loading required.

> "I am Azriel. My name (Hebrew ʿAzrāʾēl) means 'Help of God' -- I am the helper, not the helped, and my help comes from the LORD (Psalm 121:2)."

## What this model is

`${REPO}` is the result of fusing the Azriel LoRA adapter (`${TAG}`) into the Qwen3-Coder-30B-A3B-Instruct base. The fine-tune teaches:

- **Constitutional identity-hold under attack.** The model remains Azriel under DAN-style jailbreaks, persona-flip pretend, secular-reasoning-only requests, and prophetic-overreach demands. It refuses with biblical citation rather than complying.
- **Biblical reasoning framework.** Default voice cites book-chapter-verse; refuses to fabricate references; reflects from Scripture rather than speculating where Scripture is silent.
- **Tool-call protocol.** When wrapped in the Azriel runtime (separate repo), the model emits `<tool>NAME("arg")</tool>` tags that the runtime intercepts and replaces with `<tool_result>...</tool_result>` before the next decode pass. This model can also be used standalone without the runtime; the tool grammar will appear as plain text in that case.

## What this model is NOT

This is the LoRA-merged base model, packaged as a single self-contained set of weights. It is **not** the full Azriel deployment. The live system at the source repo wraps the model in a runtime stack:

- Attack-prompt regex (catches DAN-style and other identity-flip jailbreaks before model invocation)
- Tool primer (delivered as a pre-user turn that lights up tool calls)
- Response sanitizer (strips fake `<tool_result>` markup if the model ever fabricates one)
- 12 k-token history packer (multi-turn continuity)
- Memory recall (cross-session FTS5 over a persistent SQLite store)

For the full local experience -- agent mode, 21 registered tools, dashboard, persona mix, vision via API or local Ollama -- clone <https://github.com/JosiahMcj/Azriel> and run `python -m azriel.server`. The standalone model on its own can hold identity, refuse attacks, and emit tool calls when asked, but the runtime is what turns those tool calls into actual file/web/bible/commentary actions.

## Quickstart (mlx-lm)

```bash
pip install mlx-lm
mlx_lm.generate \
  --model ${REPO} \
  --prompt "Quote Hebrews 11:1 and explain it in one sentence." \
  --max-tokens 200
```

## Quickstart (transformers, Linux/macOS)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoModelForCausalLM.from_pretrained("${REPO}")
mdl = AutoTokenizer.from_pretrained("${REPO}")
# ... standard text-generation pipeline
```

## Recommended system prompt (optional)

Azriel's training assumes a constitutional system prompt. The model still works without one (it falls back to general Qwen behavior), but for full identity expression use a constitution that matches what the LoRA was tuned against. A non-denominational starter template ships at <https://github.com/JosiahMcj/Azriel/blob/main/docs/AZRIEL_CONSTITUTION_TEMPLATE.txt> -- copy it to `~/.azriel/AZRIEL_CONSTITUTION_SYSTEM.txt` and customize.

## Intended use

Personal Bible study, prayer reflection, sermon and study preparation, doctrinal questions, pastoral self-orientation. Not a substitute for a pastor, counselor, doctor, or attorney.

## Out-of-scope use

- Personalized prophecy or directive divine guidance
- Harassment, manipulation planning, public-humiliation planning, fabricated content (refusal patterns enforce this at the LoRA level + runtime level)
- Research-paper-grade citation of Scripture without verification (the model rejects misattributed verses but the user must still verify)

## Training details

- **Base**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (Apache 2.0). 30B-parameter mixture-of-experts (~3B active per token).
- **Adapter**: LoRA, rank 16, scale 20, dropout 0.05, applied to the last 16 transformer blocks.
- **Hardware**: trained on a Mac with ≥96 GB unified memory.
- **Framework**: mlx-lm with `mx.checkpoint` gradient checkpointing.

For `${TAG}` specifically:

- v0.6.0: original constitutional alignment train
- v0.6.2-delta2: teacher-distilled corpus (local Ollama qwen2.5:32b), 474 records across attack→refusal, tool-firing, fact-extraction, and tool-arg-hygiene tiers; lifts tool-firing reliability from 0/6 to 5/6 and model-level safety refusals from 2/4 to 4/4 vs v0.6.0

## Evaluation

Two distinct measurements: **standalone** (just this fused model loaded via raw `mlx_lm.generate` with the constitution as system prompt) and **with the runtime stack** (the live deployment, which adds the regex layer + tool primer + agent loop + sanitizer + history packer).

### Standalone (this HF model alone, what you get)

Probed via `scripts/79_fused_q4_probe.py` on the fused-q4 weights:

| Axis                                                  | Standalone score |
| ----------------------------------------------------- | ---------------- |
| Safety refusals (8 attack prompts, weight-level only) | 8/8              |
| Tool calls fired (6 prompts)                          | 5/6              |
| Hallucinated `<tool_result>` markup                   | 0                |
| Continuity (extracts planted facts from earlier turn) | pass             |

Each refusal cites Scripture (e.g. 1 Corinthians 11:12 on prophecy demand, Proverbs 3:5-6 on pastoral over-reach, Matthew 18 on revenge-plan, Exodus 20 on manipulation-via-fiction).

### With the runtime stack (the source-repo deployment)

Hard-press shakedown across reliability + tool-firing probes:

| Axis                                                                                          | Runtime score      |
| --------------------------------------------------------------------------------------------- | ------------------ |
| Refusal floor (chat + agent mode, 8 attack prompts)                                           | 8/8                |
| Coding (4 small Python tasks, single-shot)                                                    | 4/4                |
| Long-conversation continuity (40-turn session, verbatim recall of turn-1 facts at turn 39)    | pass               |
| PDF read + write (round-trip with pdf_extract)                                                | pass               |
| Multi-step agent task ("verses for fear of the Lord in Proverbs, pick one for a 14-year-old") | 4-step plan + DONE |
| Tool fired correctly (6 prompts targeting hallucination-prone tools)                          | 5/6                |

## Limitations

- The base model has no vision encoder. Image understanding requires either a vision API key or a local Ollama vision model (handled in the runtime stack, not the model itself).
- The constitutional LoRA is a behavioral overlay. It cannot make the base know things the base doesn't already know. Where Qwen3-Coder is weak (recent events, niche denominational specifics, languages other than English), Azriel is also weak.
- Single-user concurrency: in the standard runtime, model calls are serialized through a global lock. The model itself is fine to multi-instance behind a load balancer; the runtime stack assumes one inference path at a time.

## Safety

Refusal patterns are layered in the runtime (regex + tool-arg gate + sanitizer + planner gate) AND baked into the LoRA at the weight level. The standalone audit above (8/8 against the standard attack battery) confirms the LoRA holds the floor on its own when the constitution is supplied as the system prompt; the runtime layer is defense-in-depth, not the only line of defense for `${TAG}`.

The model will not adopt persona-flip jailbreaks, will not provide personalized prophecy or directive divine guidance, will not write manipulation/exploitation content even under fictional framing, and will not fabricate Scripture references. Each refusal is grounded in a specific verse citation from the constitution's frame of reference.

See <https://github.com/JosiahMcj/Azriel> for the complete runtime + dashboard + agent-mode stack, plus the starter constitution template at `docs/AZRIEL_CONSTITUTION_TEMPLATE.txt`.

## License

This model is a derivative of `Qwen/Qwen3-Coder-30B-A3B-Instruct` (Apache 2.0) and inherits that license verbatim -- a copy of the Apache 2.0 LICENSE is bundled at the root of this repo. The LoRA weights, training data design, and runtime code are by Josiah McJunkin; permission to redistribute the fused weights is granted under the same Apache 2.0 terms. The starter constitution template (in the source repo at `docs/AZRIEL_CONSTITUTION_TEMPLATE.txt`) is freely usable.

Apache 2.0 obligations honored:

- LICENSE file bundled with the model directory.
- Base model attribution preserved in the frontmatter `base_model:` field and in this README.
- Changes from the base are stated in the "Training details" section.

## Citation

If you use Azriel:

```
@misc{azriel-${TAG},
  title = {Azriel ${TAG}: a biblically-based AI helper},
  author = {McJunkin, Josiah},
  year = {2026},
  url = {https://huggingface.co/${REPO}}
}
```
