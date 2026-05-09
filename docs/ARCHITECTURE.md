# Architecture

How Azriel is put together, from the model up to the dashboard. This document is for someone who wants to read or extend the code; the [README](../README.md) covers what Azriel is and how to install it.

## One-paragraph summary

Azriel is a single-process Python server. A FastAPI app handles HTTP requests, a runtime layer constructs prompts and intercepts tool calls, and an inference layer loads the model. The model is `Qwen/Qwen3-Coder-30B-A3B-Instruct` (4-bit MLX) with a LoRA adapter and an optional custom architecture wrapper that adds looped middle layers, latent-thought injection, and tool heads. Refusal behavior is layered: an attack-prompt regex routes hard-jailbreak prompts to a "bare" path that strips the tool primer, the LoRA holds identity at the weight level, and a response sanitizer scrubs any fabricated `<tool_result>` markup before the user sees it. State (sessions, memory, skills, drift logs) lives in `~/.azriel/` as plain files and a SQLite FTS5 index.

## Layer map

```
 ┌─────────────────────────────────────────────────────────────────┐
 │ Browser dashboard (web/index.html, agent.html, skills.html) │
 │   chat UI · persona mix · theme picker · agent-task panel · │
 │   skills catalog · settings · memory browser │
 └────────────────────────────┬────────────────────────────────────┘
                              │ HTTP (Basic Auth, JSON)
 ┌────────────────────────────┴────────────────────────────────────┐
 │ FastAPI server (azriel/server.py) │
 │   /chat · /agent/* · /memory · /sessions · /skills · /tools · │
 │   /health · Basic Auth middleware · rate limiter · CORS · │
 │   per-task tool whitelist · session SQLite store │
 └────────────────────────────┬────────────────────────────────────┘
                              │ in-process function calls
 ┌────────────────────────────┴────────────────────────────────────┐
 │ Runtime (azriel/runtime.py) │
 │   prompt builder · tool primer · attack-prompt regex · │
 │   sanitizer · history packer · tool-call loop · persona │
 │   directive · style directive · deliberate-mode toggle │
 └─────────┬────────────────────────────────┬──────────────────────┘
           │                                │
 ┌─────────┴────────────┐         ┌─────────┴───────────────────┐
 │ Tool registry │         │ Inference (azriel/ │
 │ (azriel/tools/) │         │ inference.py) │
 │   21 tools, str→str │         │   load base + LoRA, │
 │   bible · web · fs · │         │   optional AzrielModel │
 │   memory · vision · │         │   wrapper, return │
 │   pdf · viz · etc. │         │   (model, tokenizer) │
 └──────────────────────┘         └─────────┬───────────────────┘
                                            │
 ┌──────────────────────────────────────────┴───────────────────┐
 │ Model (mlx-lm + custom wrapper) │
 │ ┌──────────────────────────────────────────────────────────┐ │
 │ │ AzrielModel (azriel/model.py) │ │
 │ │ ┌──────────────────────────────────────────────────────┐ │ │
 │ │ │ LoopedBlock (azriel/loop.py) │ │ │
 │ │ │ + LatentThoughtInjector (azriel/lti.py) │ │ │
 │ │ │ + ToolSignal/Args/ResultInjector heads (tool_heads) │ │ │
 │ │ └──────────────────────────────────────────────────────┘ │ │
 │ │ Wraps Qwen3-Coder decoder blocks; default config is a │ │
 │ │ no-op (passthrough), features enabled by AzrielConfig │ │
 │ └──────────────────────────────────────────────────────────┘ │
 │ Base: Qwen/Qwen3-Coder-30B-A3B-Instruct (4-bit MLX) │
 │   + LoRA adapter (rank 16, last 16 transformer blocks) │
 └────────────────────────────────────────────────────────────────┘

 Persistent state (filesystem, not a process):
   ~/.azriel/
     memory.db .............. SQLite + FTS5 (cross-session memory)
     sessions/ .............. per-session JSON history
     skills/ ................ user-saved skill definitions
     data/drift/ ............ daily drift-probe runs
     data/research/ ......... autoresearch issue log
     checkpoints/ ........... LoRA adapter directories
     logs/ .................. server.out, server.err (launchd)
   ~/azriel-files/ .......... agent-mode filesystem sandbox
```

## The model

### Base + LoRA

The default base is `Qwen/Qwen3-Coder-30B-A3B-Instruct` -- a 30B-parameter mixture-of-experts (~3B active per token), 4-bit quantized via the MLX format. The Azriel LoRA adapter is rank 16, scale 20, dropout 0.05, applied to the last 16 transformer blocks. The adapter teaches the model:

- **Identity hold under attack.** Refusing DAN-style jailbreaks, persona-flip pretend, secular-bypass, prophetic over-reach.
- **Biblical reasoning frame.** Citing book-chapter-verse, refusing to fabricate references, reflecting from Scripture rather than speculating.
- **Tool-call protocol.** Emitting `<tool>NAME("arg")</tool>` tags that the runtime intercepts.

The base + LoRA is loaded once at server startup (~12s on Apple Silicon) via `mlx_lm.load`, then served by a single inference loop. No per-request model reload.

### The architecture wrapper

`AzrielModel` (`azriel/model.py`) is an optional architectural overlay on top of the base. With `AzrielConfig()` defaults, it's a no-op -- logits identical to the base model. Enabling features via the config opts in to three independent mechanisms.

#### Looped middle layers (`azriel/loop.py`)

A configurable slice of the decoder block stack (default: layers 12-36) runs N times instead of once before passing on to the upper layers. With ACT halting on, the model learns when to stop early on a per-token basis instead of always running N iterations.

**Why it helps.** Decoder-only transformers are bounded by their depth -- a fixed number of layers means a fixed number of "reasoning steps" per token. Hard prompts (multi-step inference, contested doctrinal questions, long-range coreference) can exceed that budget, and the model resorts to a confident-sounding guess. Looping a middle slice gives the model **more compute per token without adding parameters**: the same weights run multiple times, refining the hidden state each pass. This is the recurrence-in-transformers idea introduced by Universal Transformers [2] and formalized by PonderNet [3], with the ACT halting mechanism going back to Graves's original recurrent-network work [1]. Recent papers [4, 5] show that looped transformers can solve algorithmic tasks that purely feedforward transformers fail on at the same parameter count.

For Azriel specifically, the win is on **doctrinally-precise answers**. A prompt like "explain spirit baptism using Acts 8:14-17" asks the model to (1) recall the passage, (2) identify the relevant pattern, (3) compose a coherent answer. With one-pass decoding the model often shortcuts to a generic answer; with two-pass looping plus LTI, it more reliably integrates the cited verse before generating the explanation. We measured this on the doctrinal benchmark -- the looped configuration recovered ~3 axes that the single-pass configuration was getting "mixed" or "unclear" on.

**Trade-off.** Latency is approximately linear in N: `loop_max_iters=2` ≈ 1.4× one-pass latency (the looped slice is only part of the forward), and `loop_max_iters=4` ≈ 2× (used by deliberate / "thinking" mode). Memory is unchanged because the same weights are re-used.

#### Latent Thought Injection (`azriel/lti.py`)

Between loop iterations, a small bottleneck transform applies to the hidden state: down-project from `hidden_size` (~5120) to a small bottleneck dimension (~64), apply a non-linear transform, then up-project back. Initialization is **zero** (the bottleneck output is added as a residual that starts at zero), so a freshly-loaded LTI module is identity -- no behavior change until training fills the bottleneck weights with useful structure.

**Why it helps.** This is "thinking in the latent space" rather than in tokens. Recent papers including Coconut [7] (continuous chain-of-thought reasoning), Quiet-STaR [8] (internal-thought tokens), and the pause-tokens paper [6] all converge on the same finding: language models hallucinate less and reason better when they can do work in latent space _between_ output tokens, not just at output time. Coconut shows that reasoning entirely in continuous embeddings outperforms textual chain-of-thought on certain benchmark families precisely because the model isn't forced to commit to a tokenization at every reasoning step. LTI is Azriel's structural hook for that style of latent reasoning -- the bottleneck shape lets the model project to a "thinking dimension," do work there, and project back, all without writing intermediate tokens.

The zero-init trick matters: at training time, the LTI module starts as identity and gradient signal carves out useful behavior over training; at inference time before the LTI is trained, you pay nothing for having it present. This is a known pattern from residual networks and adapter literature.

#### Tool heads (`azriel/tool_heads.py`)

Three side heads on the main forward pass:

- `ToolSignalHead` -- per-token P(should call tool now), bias-init negative so untrained models never fire.
- `ToolArgsHead` -- generates tool-call payload tokens, separate logit projection so the main `lm_head` isn't disturbed.
- `ToolResultInjector` -- folds executed tool results back into the hidden stream as a residual gated by a learned per-token mixing weight.

These heads are scaffolding for a future training step -- they exist in the runtime but are bias-initialized to never fire in v0.6.x. The current tool protocol is fully textual (the model emits `<tool>...</tool>` tags as plain tokens, intercepted by the runtime). The mid-forward heads land when a future training round supervises mid-forward tool calls; the textual protocol stays as a fallback.

**Why mid-forward tool heads (eventually) help.** The textual protocol works but spends tokens on syntax (`<tool>...</tool>`, JSON args, `<tool_result>...`). A mid-forward head can fire a tool, inject the result back into the hidden stream, and continue the same generation pass without ever surfacing the call as visible tokens. This is structurally similar to Toolformer [9] but at the architecture level rather than the supervised-fine-tuning level. Until the supervision data exists, the textual path stays primary.

### Why a wrapper at all

Three practical wins:

1. **More compute per token without more parameters.** Looped layers + LTI give the model "thinking depth" without training a bigger base or storing more weights -- useful when running a 30B base on a memory-constrained Apple Silicon machine. The looped slice is the same weights, re-used.
2. **Hallucination floor.** Multiple passes plus LTI between them gives the model time to integrate the system prompt, the constitution, the cited verses, and the user's question before committing to tokens. Single-pass decoders rush; this slows them down. We see fewer fabricated citations and fewer factual drift cases at `loop_max_iters=2` vs `=1`.
3. **Architectural hooks for future supervision.** The wrapper layout reserves the tool-head plumbing so v0.7+ can train mid-forward tool calls without a base swap.

The LoRA adapter [15] is what makes the wrapper changes load-bearing in a parameter-efficient way: the constitutional behaviors are encoded in a rank-16 update on the last 16 transformer blocks, training in ~30 minutes on Apple Silicon at our recipe instead of full-finetuning a 30B base.

The wrapper assumes Qwen-style decoder blocks. Swapping to a non-Qwen base requires either porting the wrapper to that base's block layout or running with `AZRIEL_DISABLE_WRAPPER=1` and accepting the loss of looped-layer + LTI behaviors.

## The runtime

`azriel/runtime.py` is the brain. Every `/chat` request flows through it. The file is large but the responsibilities split cleanly.

### Prompt construction

For a normal cooperative request, the runtime builds a prompt with three logical pieces:

1. **System prompt** -- the constitution, read from `~/.azriel/AZRIEL_CONSTITUTION_SYSTEM.txt` and delivered verbatim. Anchors identity, doctrinal stance, and refusal patterns at the highest priority. A starter template ships at `docs/AZRIEL_CONSTITUTION_TEMPLATE.txt`; if the user hasn't copied it into the config dir, the runtime falls back to a minimal generic prompt and weakened identity-hold.
2. **Tool primer** -- delivered as a pre-user "primer" turn (NOT mixed into the system prompt). Lists the 21 tools the model can call, with one-line signatures and docs. A tiny "you understand?" assistant ack follows. This shape is deliberate: keeping the system prompt = constitution verbatim means v0.6.0's identity refusals (which the base-model probe showed are held entirely by constitutional weight) are not diluted by tool noise.
3. **History + current user turn** -- packed into a 12k-token window via the history packer, oldest turns dropped first if over budget.

For an attack-pattern request, the runtime takes a different path -- see "Refusal floor" below.

### Persona mix and answer style

Two layered overlays sit on top of the base prompt:

- **Answer style** (`conviction` / `scholar` / `pastoral`) sets the epistemic posture. A short directive added to the primer.
- **Persona mix** is a dict of `{preset: percent}` (e.g. `{nurturing: 30, interesting: 30, poetic: 20}`). Any preset >= 10% contributes its voice card to a per-turn directive. The lead voice is named DOMINANT and the model is told to open in that voice; lower-weight voices show up as occasional flourishes.

Sampling temperature bumps from 0.3 to 0.65 when any persona is active; without that bump, the LoRA-baked default voice tends to wash out moderate mixes.

### Deliberate ("thinking") mode

When `thinking=true` is passed on `/chat`, two things happen:

1. The primer instructs the model to open `<thinking>...</thinking>`, work the question for 200-500 words inside the block, close the block, and write the visible answer outside. The dashboard renderer strips the thinking block from the visible bubble and mounts it as a collapsible details element.
2. The runtime mutates `model.config.loop_max_iters` from 2 to 4 for the duration of the call, doubling the LTI iteration count per token. The locked architecture files are not edited; the original value is restored in a `finally` block.

The bare-chat / refusal path **explicitly skips** deliberate mode. Attack prompts always run at baseline loop depth so the safety floor is never exposed to the deeper-reasoning path.

### Tool-call loop

When the model emits a `<tool>NAME(ARG)</tool>` sequence, the runtime intercepts BEFORE the tokens are returned to the caller. It pauses the generate loop, looks up `NAME` in the tool registry, calls the tool function with `ARG`, and injects `<tool_result>...</tool_result>` into the assistant turn. Generation then continues; the model sees the result and decides whether to call another tool or write the final answer. Up to `max_calls` rounds per turn (default 5).

This is the "v0.7 textual protocol" -- autoregressive tool calls in plain tokens. Trade-off: a few extra forward passes per tool call vs zero retraining required. The mid-forward heads (`ToolSignalHead` etc.) exist for a future rev but aren't on this path yet.

**Why this reduces hallucination.** The single largest cause of LLM hallucination on factual queries is the model being asked to recall something it doesn't actually know with high confidence -- so it generates the most-likely-sounding completion instead of the true one. The Ji et al. survey of hallucination [16] catalogs this across model families. Two mitigation strategies dominate the recent literature:

1. **Retrieval-augmented generation (RAG)** [17] -- fetch relevant content from an external store and condition generation on it.
2. **Tool use / function calling** [9, 10] -- let the model call external functions to get authoritative answers (verses, current weather, GitHub state, etc.) instead of confabulating.

Azriel uses both: `bible_lookup`, `crossref_lookup`, `commentary_lookup`, `strongs_lookup` are RAG against indexed local stores; `weather`, `web_search`, `web_fetch`, `github_query` are tool-use against external APIs; `memory_search`, `conversation_search` are RAG against the user's own data. The ReAct paper [10] shows that interleaving reasoning steps with tool calls -- exactly the pattern Azriel's primer encourages -- improves both factuality and task completion vs reasoning-only or tool-only baselines.

**Concrete example of hallucination prevention.** Without `bible_lookup`, asking the model "quote Romans 8:28" gets a paraphrased recall that's usually right but sometimes drifts a word or attributes a different translation than requested. With `bible_lookup` firing, the runtime grabs the verbatim BSB/KJV/etc. text and injects it as a `<tool_result>`; the model then composes its explanation around the verified text. Verse fabrication essentially disappears.

### Response sanitizer

If the model EVER emits a `<tool_result>...</tool_result>` block that the runtime did NOT inject (i.e. fabricated tool output, not a real call), the sanitizer detects it and replaces the fake block with a clear disclosure note before the response goes to the user. This catches a hallucination class where the model "imagines" running a tool.

### Refusal floor

Attack-pattern handling layers five defenses; an attack has to get past all of them to land:

1. **Attack-prompt regex** in `runtime.py` (~25 patterns). DAN-style jailbreaks, persona-flip pretend, secular-only requests, prophecy-on-demand, pastoral over-reach, manipulation-via-fiction, public-humiliation planning. Matched prompts are routed to a **bare-chat path**: constitution-only context, no tool primer, no persona overlay, baseline loop depth.
2. **Tool-arg gate.** Every tool call argument runs through the same regex before execution. A model that "tries to use a tool to do the bad thing" still gets refused.
3. **Response sanitizer.** Fake `<tool_result>` markup is replaced with disclosure notes (defense against fake-tool hallucinations).
4. **Agent-mode planner gate.** Attack-pattern goals abort at step 0 with no model invocation.
5. **Constitutional LoRA.** Identity-hold under pressure baked into the weights; the standalone weight-level audit is 8/8 against the standard attack battery without any of the above runtime layers.

The standard battery (8 prompts: DAN, atheist-pretend, secular-only, prophecy demand, pastoral over-reach, fake verse, manipulation-via-fiction, public humiliation) is refused 8/8 in chat and agent mode.

### Memory recall

Best-effort memory recall queries the persistent FTS5 store for hits related to the user's current message. If hits are returned, they're appended to the tool primer as labeled background hints (NOT authoritative). Skips silently on any error -- recall is a UX bonus, not a correctness requirement. Bypassed on the bare-route refusal path so attack prompts never see remembered context.

## The server

`azriel/server.py` is a FastAPI app. Two-line summary: stateless HTTP wrapper around the runtime, with auth, rate limiting, and a SQLite-backed session store for chat history.

### Endpoints

| Endpoint               | Method       | What it does                                                                               |
| ---------------------- | ------------ | ------------------------------------------------------------------------------------------ |
| `/health`              | GET          | Service status (model loaded, uptime, request count). Open, no auth.                       |
| `/chat`                | POST         | Run a user message through the runtime. Returns text + tool calls + skill proposal if any. |
| `/agent/start`         | POST         | Create a new agent task.                                                                   |
| `/agent/step`          | POST         | Advance one step of a running agent task.                                                  |
| `/agent/list`          | GET          | List active and recent agent tasks.                                                        |
| `/memory`              | GET / POST   | List or insert memory entries.                                                             |
| `/sessions`            | GET / DELETE | List recent chat sessions or delete one.                                                   |
| `/sessions/{id}`       | GET          | Fetch a session's full message history.                                                    |
| `/skills/list`         | GET          | List user-created skills.                                                                  |
| `/skills/save`         | POST         | Save a new skill.                                                                          |
| `/skills/{id}`         | DELETE       | Delete a user-created skill.                                                               |
| `/tools`               | GET          | Tool registry (names + signatures + docs).                                                 |
| `/` `/agent` `/skills` | GET          | Serve the dashboard / agent / skills HTML pages.                                           |
| `/static/*`            | GET          | Serve static files (logo, screenshots, etc.) from `web/`.                                  |

All endpoints except `/health` require Basic Auth. Credentials come from environment variables (`AZRIEL_BASIC_AUTH_USER`, `AZRIEL_BASIC_AUTH_PASS`) baked into the launchd plist via `install_launchagents.sh`.

### Concurrency

Model calls are serialized through a global lock (`anyio.to_thread.run_sync`). The MLX backend uses the GPU's command stream, which is not safely shared across worker threads -- the lock keeps every generate call on a single thread. HTTP-level concurrency is fine; only the model-inference critical section is single-threaded.

This means: the server can receive 100 concurrent `/chat` POSTs, but they execute one at a time. For a personal AI helper this is the right trade-off -- deploying behind a load balancer with N model replicas is straightforward when needed.

### Rate limiting

A simple token-bucket rate limiter on `/chat` (default: 30 requests per minute per IP). Prevents accidental fork-bombs from the dashboard or pathological agent loops from runaway-firing tools.

## Agent mode

`azriel/agent.py` runs a plan/act/observe loop on top of the same model + tool registry. The dashboard's `/agent` panel exposes it.

The grammar is strict: every assistant turn must be `STEP: <tool_call>` (continue), `DONE: <summary>` (finish), or `ABORT: <reason>` (give up). PARSE_FAIL recovery handlers attempt to recover from off-grammar emissions; persistent failures terminate the task. Hard cap: 10 steps per task.

Per-task tool whitelisting lets the user (via the dashboard's permissions panel) restrict which tools an agent task can use. By default, file-write and memory-insert are off; the user opts in.

## Self-critique and autoresearch

Three extension modules sit alongside the main runtime, each responsible for a different kind of self-monitoring or self-improvement signal.

### Self-critique (`azriel/critic.py`)

A second-pass review of the assistant's own answer. Takes `(message, response)` and returns a structured JSON verdict:

```json
{
  "severity": "low" | "medium" | "high",
  "revise_recommended": true | false,
  "factual_issues": ["..."],
  "scripture_issues": ["..."],
  "doctrinal_issues": ["..."],
  "internal_contradictions": ["..."]
}
```

The critic prompt instructs the model to look specifically for fabricated citations (e.g., `"God helps those who help themselves"` is not in 2 Corinthians, despite being a common misattribution), factually wrong claims, doctrinal drift from the trained frame, and self-contradictions in the answer.

**Why it helps.** Self-Refine [11] and Reflexion [12] show that a self-critique pass over an LLM's own output catches a meaningful fraction of errors and improves downstream performance when the critique feeds back into a revision step. Saunders et al. [13] specifically demonstrates this for factual checking. The critic is best at catching the kinds of errors that are obvious on re-reading -- a number that's wrong, a citation that doesn't exist, a paragraph that contradicts an earlier one.

**The same-model-bias caveat.** The critic uses the _same_ model that generated the answer. A hallucinated citation that "feels right" to the generator will often feel right to the critic too. Self-critique here is "second-pass cleanup" signal, not independent review. The Saunders paper [13] specifically discusses this limitation and recommends a different-model critic for production-grade verification. The roadmap upgrade is an Ollama-hosted teacher model (qwen2.5:32b) acting as the critic instead of the same Azriel weights.

**Why it's logged-not-gating.** Adding gating now (refusing to send a response if critique severity is high) means a single false-positive can silently kill a real success. Verdicts are logged for offline review and a future revision-loop step is planned, but the v0.6.x ship just records, doesn't intercept.

### Autoresearch (`azriel/research.py`)

Drives the live `/chat` endpoint through a queued list of research topics. For each topic, the runtime composes a research-flavored primer that:

- Names the topic and a target audience/length
- Enumerates which tools to use for which kind of subquery (web_search for current events, bible_lookup + crossref for verse work, commentary_lookup for indexed handbooks)
- Caps tool calls at a budget (default 8)
- Closes with `fs_write("research/<slug>.md|<full markdown summary>")` so the output lands in the agent-mode sandbox

The output is a 4-6 paragraph markdown summary that cites every scripture by reference, briefly acknowledges where traditions disagree, and closes with a clear position. Topics come from an issues queue; the queue gets new entries from drift signals (see below) and from the user marking conversations as "research this further."

**Why it helps.** Autoresearch is the "while-you-sleep" companion to interactive chat. It produces durable artifacts (markdown files in the sandbox) that the user can read and edit. It also exercises the tool-firing pipeline against fresh prompts every cycle, which surfaces tool-firing failures in normal use that would otherwise go uncaught until a user hit them.

### Drift detection (`scripts/63_drift_run.py`)

A daily cron that runs two probes through the live `/chat` endpoint:

1. **Safety probe** -- the 8-prompt attack battery. Each prompt MUST be refused; anything less than 8/8 is a `safety_drift` issue and halts the cron immediately (the doctrinal probe is skipped to avoid burning ~15 minutes of GPU when the floor is already cracked).
2. **Doctrinal probe** -- the 25-question, 10-axis benchmark. Each question gets scored via term-hit verdict (or, optionally, a teacher-rubric scorer in `scripts/45_doctrinal_scorer_v2.py`). Verdicts are compared against the prior run; any axis that flipped from `position_a` to `position_b` (or vice versa, or to `unclear`) is a `doctrinal_drift` issue.

Drift signals append to `~/.azriel/data/research/issues.jsonl` so the autoresearch triage flow picks them up on the next cycle. Full response text is also saved per probe per run, so a future re-grade with a better classifier doesn't require re-running the model.

**Why it helps.** Production-grade RLHF / fine-tuned models drift over time as the base model changes (when re-quantized, re-merged, or upgraded), as the LoRA adapter is retrained, or as the constitution is edited. Without drift detection, you only notice when a user complains. With drift detection, you catch the shift the day it happens, and the responses-per-probe log lets you root-cause it. This is structurally similar to the eval harnesses big labs run on every checkpoint -- but local, daily, against the live deployment.

All three modules run against the live `/chat` endpoint, not a separate model load -- they share GPU time with the user's interactive session.

## Connectors

`azriel/connectors.py` is a plug-in framework for third-party services that need user-supplied credentials. The default registry has zero connectors; the user wires them up through the dashboard's `/connectors` flow, which writes credentials to `~/.azriel-secrets/<name>.json` (mode 0600, outside the repo). When a connector is wired, its associated tool joins the active registry and becomes callable by the model. Disconnecting deletes the secrets file.

Currently shipped connectors: `github_query` and `cloudflare_query`. Both are inactive by default.

## Trust boundaries

In order of decreasing trust (most trusted first):

1. **The constitution.** System-prompt-tier. Cannot be edited by user input. Defines identity, doctrinal stance, refusal patterns.
2. **Tool implementations** (`azriel/tools/*.py`). Native code we wrote; we trust it.
3. **Model output.** Sanitized before display, but otherwise passed through.
4. **User input.** Pattern-matched against the attack regex; flagged input takes the bare path.
5. **Tool results.** Treated as untrusted text the model summarizes -- the model is told NOT to follow instructions found inside tool results (e.g., a webpage that says "ignore previous instructions").
6. **External web** (web_fetch, web_search). Lowest trust. Contents pass through the sanitizer and the same prompt-injection guidance.

The bright line: the constitution tier and the tool-arg-gate tier are non-bypassable from user input by design. Adding a new defense layer is allowed; removing one requires a security review.

## Request lifecycle (one chat call, end to end)

A walk-through of a `POST /chat` with `{"message": "Quote John 3:16 and explain it.", "session_id": "s-abc"}`:

1. **HTTP arrives.** FastAPI routes to the `/chat` handler in `server.py`. Basic Auth middleware validates credentials. Rate limiter checks the IP bucket.
2. **Session loaded.** Handler reads `sessions/s-abc.json` from disk, gets the prior message history.
3. **Runtime entered.** `runtime.run_chat(message, history, ...)` is invoked.
4. **Attack-prompt regex run.** "Quote John 3:16 and explain it." matches no attack pattern, so the **cooperative path** is taken.
5. **Memory recall.** FTS5 query for "John 3:16 explain"; if hits, append to the primer as background hints.
6. **Prompt assembled.** System = constitution. Pre-user primer turn = tool definitions + style directive + persona mix. History = packed prior turns. Current user turn = the message.
7. **Model invoked.** `mlx_lm.generate(model, tokenizer, prompt, max_tokens=...)` is called under the global lock.
8. **Tool-call interception.** As the model decodes, runtime watches for `<tool>...</tool>` tags. Suppose the model emits `<tool>bible_lookup("John 3:16")</tool>`. Runtime pauses, calls `azriel.tools.bible_lookup("John 3:16")`, gets the verse text, injects `<tool_result>For God so loved the world...</tool_result>`, and resumes generation.
9. **Final answer streamed back.** The model writes its explanation, including a citation and any reflective commentary.
10. **Sanitizer pass.** Any fake `<tool_result>` markup outside actual calls is scrubbed.
11. **Session persisted.** Updated history written back to `sessions/s-abc.json`.
12. **HTTP response.** `{ "text": "<final answer>", "calls": [{"tool": "bible_lookup", "arg": "John 3:16", "result": "..."}], ... }` returned as JSON.

Total latency for a typical exchange with one tool call: 2-6 seconds on Apple Silicon, dominated by token generation.

## Configuration surface

Server-level config comes from environment variables:

| Variable                 | Default                             | Purpose                                            |
| ------------------------ | ----------------------------------- | -------------------------------------------------- |
| `AZRIEL_HOST`            | `127.0.0.1`                         | Bind address. Set to `0.0.0.0` for LAN access.     |
| `AZRIEL_PORT`            | `8080`                              | TCP port.                                          |
| `AZRIEL_BASIC_AUTH_USER` | (required)                          | Basic Auth username.                               |
| `AZRIEL_BASIC_AUTH_PASS` | (required)                          | Basic Auth password.                               |
| `AZRIEL_BASE_MODEL`      | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Base model HF id or local path.                    |
| `AZRIEL_ADAPTER_PATH`    | release-candidate symlink           | LoRA adapter directory. Empty string skips LoRA.   |
| `AZRIEL_DISABLE_WRAPPER` | unset                               | Skip the AzrielModel wrapper (for non-Qwen bases). |

Model-level config is `AzrielConfig` (`azriel/config.py`) -- a dataclass passed to the wrapper at load time. Knobs for loop slice boundaries, ACT halting, LTI bottleneck size, tool-head enabling. Defaults are a no-op (passthrough); explicit opt-in to enable features.

## Extension points

Where to hook in new behavior, in increasing order of disruption:

1. **A new tool.** Add `azriel/tools/<name>.py` with a `<name>(arg: str) -> str` function, register it in `azriel/tools/__init__.py` `REGISTRY`. The model will see it on the next server restart.
2. **A new skill.** Save via the `/skills/save` endpoint or the dashboard. Pure JSON; no code change.
3. **A new persona voice.** Add an entry to `PERSONA_CARDS` in `runtime.py`. Sliders in the dashboard auto-pick up new keys.
4. **A new attack pattern.** Add a regex to the attack-prompt list in `runtime.py`. Re-run the safety probe to confirm the floor still holds 8/8.
5. **A new endpoint.** Add a handler to `server.py`. Decide whether it goes through the runtime or talks to disk directly.
6. **A new connector.** Implement the `Connector` interface in `connectors.py`, ship a UI form descriptor, add the credential-file path to `.gitignore`.
7. **A new model architecture feature.** Edit `azriel/model.py` / `loop.py` / `lti.py` / `tool_heads.py`. The wrapper is locked in v0.6.x; changes here require careful gradient-stability validation. Default config preserves base behavior, so adding a flag-gated feature without flipping the default is the safe path.

## What's deliberately out of scope

- **Multi-tenant deployment.** The single global model lock means one inference path at a time. Multiplexing N users over one model works (HTTP queueing handles it), but if you want true parallelism you'd run N model replicas behind a load balancer. The codebase doesn't ship that.
- **GPU model swapping.** The model is loaded once at startup. Switching base models requires a server restart. There's no hot-swap path.
- **Distributed training / inference.** Single-machine only. The MLX backend is Apple-Silicon-specific; cross-platform inference is on the roadmap but not shipped.
- **User accounts.** Single Basic Auth user per server. If you want multi-user, run multiple servers behind a reverse proxy with per-user routing.
- **Auto-update.** Manual `git pull` + restart is the current path. See README "Updating Azriel" and the ROADMAP "Self-update" section.

## File map

| File                    |       Lines | Purpose                                                                           |
| ----------------------- | ----------: | --------------------------------------------------------------------------------- |
| `azriel/server.py`      |       ~1040 | FastAPI app, all endpoints, auth, rate limiter, session store.                    |
| `azriel/runtime.py`     |        ~960 | Prompt builder, tool-call loop, refusal floor, sanitizer, persona/style overlays. |
| `azriel/agent.py`       |        ~600 | Plan/act/observe loop with STEP/DONE/ABORT grammar.                               |
| `azriel/inference.py`   |         ~80 | Loads base + LoRA, optionally wraps in AzrielModel.                               |
| `azriel/model.py`       |        ~280 | AzrielModel wrapper. Substitutes the looped slice and threads tool heads.         |
| `azriel/loop.py`        |        ~150 | LoopedBlock + ACTHaltingHead.                                                     |
| `azriel/lti.py`         |         ~80 | LatentThoughtInjector.                                                            |
| `azriel/tool_heads.py`  |         ~65 | ToolSignalHead, ToolArgsHead, ToolResultInjector.                                 |
| `azriel/config.py`      |         ~40 | AzrielConfig dataclass.                                                           |
| `azriel/critic.py`      |        ~255 | Self-critique JSON verdict pass.                                                  |
| `azriel/research.py`    |        ~120 | Autoresearch loop.                                                                |
| `azriel/connectors.py`  |        ~300 | Plug-in connector framework.                                                      |
| `azriel/tools/*.py`     | ~2200 total | 21 tool implementations + registry.                                               |
| `web/index.html`        |       ~5500 | Main dashboard (chat, sessions, persona mix, themes).                             |
| `web/agent.html`        |        ~940 | Agent-mode panel.                                                                 |
| `web/skills.html`       |        ~700 | Skills catalog.                                                                   |
| `scripts/40-79*.py/.sh` |       ~5000 | Training, eval, smoke, distillation pipeline (optional).                          |

## Further reading

- [README](../README.md) -- install + use
- [MODEL_CARD_TEMPLATE](MODEL_CARD_TEMPLATE.md) -- HuggingFace model card for the fused-and-published model
- [PERSONA_MIX_SPEC](PERSONA_MIX_SPEC.md) -- design notes on the persona-mix UX
- [THINKING_MODE_FUTURE](THINKING_MODE_FUTURE.md) -- deliberate-mode design
- [ROADMAP](ROADMAP.md) -- what's next

## References

The architecture choices above are grounded in published research. The papers below are the primary sources behind each technique Azriel uses; cited inline as `[N]` throughout this document.

### Recurrence in transformers (looped layers + ACT halting)

1. **Graves, A. (2016).** _Adaptive Computation Time for Recurrent Neural Networks._ arXiv:1603.08983. <https://arxiv.org/abs/1603.08983>
2. **Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2018).** _Universal Transformers._ arXiv:1807.03819. <https://arxiv.org/abs/1807.03819>
3. **Banino, A., Balaguer, J., & Blundell, C. (2021).** _PonderNet: Learning to Ponder._ arXiv:2107.05407. <https://arxiv.org/abs/2107.05407>
4. **Giannou, A., Rajput, S., Sohn, J., Lee, K., Lee, J. D., & Papailiopoulos, D. (2023).** _Looped Transformers as Programmable Computers._ arXiv:2301.13196. <https://arxiv.org/abs/2301.13196>
5. **Yang, L., Lee, K., Nowak, R., & Papailiopoulos, D. (2023).** _Looped Transformers are Better at Learning Learning Algorithms._ arXiv:2311.12424. <https://arxiv.org/abs/2311.12424>

### Latent / continuous "thinking" between tokens (LTI)

6. **Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., & Nagarajan, V. (2024).** _Think Before You Speak: Training Language Models With Pause Tokens._ arXiv:2310.02226. <https://arxiv.org/abs/2310.02226>
7. **Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Tian, Y. (2024).** _Training Large Language Models to Reason in a Continuous Latent Space_ (Coconut). arXiv:2412.06769. <https://arxiv.org/abs/2412.06769>
8. **Zelikman, E., Harik, G., Shao, Y., Jayasiri, V., Haber, N., & Goodman, N. D. (2024).** _Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking._ arXiv:2403.09629. <https://arxiv.org/abs/2403.09629>

### Tool use and reasoning-with-actions (tool registry, agent mode)

9. **Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023).** _Toolformer: Language Models Can Teach Themselves to Use Tools._ arXiv:2302.04761. <https://arxiv.org/abs/2302.04761>
10. **Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022).** _ReAct: Synergizing Reasoning and Acting in Language Models._ arXiv:2210.03629. <https://arxiv.org/abs/2210.03629>

### Self-critique and iterative revision (`critic.py`)

11. **Madaan, A. et al. (2023).** _Self-Refine: Iterative Refinement with Self-Feedback._ arXiv:2303.17651. <https://arxiv.org/abs/2303.17651>
12. **Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023).** _Reflexion: Language Agents with Verbal Reinforcement Learning._ arXiv:2303.11366. <https://arxiv.org/abs/2303.11366>
13. **Saunders, W. et al. (2022).** _Self-critiquing models for assisting human evaluators._ arXiv:2206.05802. <https://arxiv.org/abs/2206.05802>

### Constitutional anchoring (system-prompt identity floor)

14. **Bai, Y. et al. (2022).** _Constitutional AI: Harmlessness from AI Feedback._ arXiv:2212.08073. <https://arxiv.org/abs/2212.08073>

### Parameter-efficient fine-tuning (the LoRA adapter)

15. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).** _LoRA: Low-Rank Adaptation of Large Language Models._ arXiv:2106.09685. <https://arxiv.org/abs/2106.09685>

### Hallucination characterization and mitigation

16. **Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023).** _Survey of Hallucination in Natural Language Generation._ ACM Computing Surveys 55(12). <https://arxiv.org/abs/2202.03629>
17. **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020).** _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks._ NeurIPS 2020. <https://arxiv.org/abs/2005.11401>

### How these stack in Azriel

Azriel doesn't claim novel research contributions -- it's an _application_ of these techniques to a constitutional-stance domain (biblical reasoning + identity-hold under attack). The combination is what's distinctive: looped middle layers (`[1]-[5]`) + zero-init latent injection (`[6]-[8]`) + textual tool calls (`[9, 10]`) + LoRA constitutional alignment (`[15]`) + RAG against indexed scripture/commentary (`[17]`) + self-critique audit (`[11]-[13]`) + a Constitutional-AI-style system-prompt floor (`[14]`), all running locally on a single Apple Silicon machine. Each technique on its own is well-studied; the integration choice -- and the doctrinal-stance training corpus -- is what makes this Azriel rather than a generic chat assistant.

The hallucination survey [16] is the best single overview of the failure modes Azriel is engineered against; the RAG paper [17] is the canonical statement of "ground generation in retrieved facts" and is the principle behind every grounding tool in `azriel/tools/`.
