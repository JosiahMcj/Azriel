# Roadmap

Forward-looking plans for Azriel beyond the current release.

## Base-model swap

A future quality lift will explore swapping the base from Qwen3-Coder-30B-A3B to a stronger reasoning-trained base. Candidates: Qwen3-Reasoning (when available), DeepSeek-R1-distill-32B, Kimi K2 reasoning models, Qwen3-32B dense as a fallback. Constraints: license must be permissive, tokenizer + chat template must accept the constitution as a system prompt, weights must fit in the same memory envelope.

The custom architecture wrapper (`AzrielModel` in `azriel/model.py`) currently assumes Qwen-style decoder blocks. Swapping to a non-Qwen base would require either porting the wrapper or running with `AZRIEL_DISABLE_WRAPPER=1` and accepting the loss of looped-layer + LTI behaviors.

## Teacher upgrade

The default training pipeline can use a remote frontier API as a teacher for synthetic data generation, rubric scoring, and corpus distillation. The default scorer uses a local Ollama model. To upgrade quality, swap in a frontier model via the env-driven hooks in `scripts/45_doctrinal_scorer_v2.py` and the related teacher routing in `scripts/47_generate_tool_traces.py`.

## Self-research and drift detection

Autoresearch (`azriel/research.py`) and the drift cron (`scripts/63_drift_run.py`) detect doctrinal drift across an 8-prompt safety probe + 25-axis doctrinal probe, daily. Drift signals append to a runs log; a future iteration could feed those signals back into a corpus distillation step that surfaces the drift cases as new training examples.

## Self-critique and corpus distillation

`azriel/critic.py` implements a self-critique pass that scores responses on doctrinal grounding, citation accuracy, and refusal completeness. A planned next step takes the lowest-scoring responses, distills them through a teacher model with the rubric, and feeds the distilled exemplars into the next training round.

## Agent mode improvements

Current agent mode supports plan / act / observe with a 21-tool registry, a strict STEP / DONE / ABORT grammar, and a 10-step cap. Future work:

- Per-task capability scoping via the permissions panel in the dashboard (already shipped as a skeleton; deeper toggles for new capabilities like terminal access and browser automation are gated behind their respective tools landing first).
- Better recovery from PARSE_FAIL via a richer goal-keyword router (already prototyped).
- Multi-task orchestration: spawning sub-agents for branches of a goal, merging results.

## New tools

Capabilities in the Permissions panel marked "coming soon" need landing tools:

- **Terminal**: shell execution inside a tighter sandbox than the current `~/azriel-files/` filesystem sandbox. Per-command confirmation UX.
- **Browser**: Playwright-driven browser automation with a per-action confirmation prompt.
- **Email**: outbound mail through a configurable SMTP / API provider, with a per-message confirmation step.

Each tool needs a security review pass before landing.

## Self-update

When a new release lands in the upstream repo, Azriel should be able to update himself on demand. The flow:

- The user asks Azriel to check for updates.
- Azriel diffs the local install against the configured upstream tag, surfaces the changeset, and asks for explicit confirmation.
- On approval, Azriel pulls the new code, re-runs `pip install -r requirements.txt`, runs the smoke suite, and restarts the server.
- A rollback hook keeps the previous install one command away.

Self-update never fires without a user request, and never auto-applies migrations that would touch user data without a confirmation step.

## Skills marketplace

The skills system supports user-created skills today. A future iteration could let users export a skill as a portable JSON, share skill bundles across deployments, and (optionally) discover skills from a curated catalog.

## Vision improvements

`image_describe` currently routes to local Ollama vision (free) or a configurable remote vision API. A future option: train a small vision adapter directly on the base model so vision lives in-process, eliminating the network round-trip.
