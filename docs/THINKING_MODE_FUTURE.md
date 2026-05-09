# Thinking Mode

A **Mode** toggle in Profile → Settings.

## Instant (default)

Single-pass cooperative chat. The runtime loop runs at baseline `loop_max_iters=2` (one extra LTI iteration on top of the layer pass). Direct answer, fast.

## Deliberate

Two-axis thinking on top of the cooperative path:

1. **Chain-of-thought scaffold**: the runtime primer instructs the model to open `<thinking>...</thinking>`, work the question 200–500 words inside the block, close the block, then write a 4–6 paragraph visible answer. The dashboard renderer strips the thinking block from the visible bubble and mounts it as a collapsible *details* element below the answer.

2. **Recurrent loop deepened**: when `thinking=true` is passed on a `/chat` request, the runtime mutates `model.config.loop_max_iters` from 2 to 4 — doubling the LTI iteration count per token. The locked architecture files are not edited; the original value is restored in a `finally` block.

Cost: roughly 2× latency on the cooperative path. The per-segment token budget is bumped to ~3000 tokens so the model has room for both the scratchpad and the visible answer.

## Safety carveout

The bare-chat / refusal path **explicitly skips** deliberate mode. Attack prompts always run at baseline loop depth + the standard token budget so the safety floor is never exposed to the deeper-reasoning path.

## Knobs

The toggle persists in localStorage as `azriel.thinking_mode` (`"instant"` or `"deliberate"`). It's sent on every `/chat` POST as the `thinking` boolean.

## Future ideas

- **Adaptive thinking**: the model decides per-prompt whether to engage deliberate mode based on a short pre-classifier run.
- **Streaming thinking**: stream the `<thinking>` block as it's generated so the user sees progress.
- **Per-skill defaults**: skills declare their preferred thinking mode in their kickoff metadata so launching a skill auto-sets the toggle.
