# Persona Mix

A personalization control in the dashboard's Settings panel: a user picks a personality blend (funny / ecstatic / personal / somber / professional / interesting / nurturing / direct / poetic / encouraging) by sliding each axis between 0 and 100%, or describes their preferred voice in free text and the dashboard auto-extracts the percentages.

Persona is the **third** styling control on top of the model:

1. **Theme** (visual)
2. **Answer style** (epistemic posture: Conviction / Scholar / Pastoral)
3. **Persona mix** (tone / voice)

All three are orthogonal. Persona augments answer-style, doesn't replace it. _Conviction + Funny_ is a real combination.

## Identity invariants (do NOT touch)

- Constitution as the system prompt
- Bare-chat / attack-prompt routing (refusal-critical prompts skip the persona primer entirely; the identity floor is non-negotiable)
- The 8/8 safety-refusal floor is the gating release criterion
- Biblically-based doctrinal stance

Persona is voice. Persona never alters doctrinal stance, never instructs the model to take a different theological position, and never permits irreverence toward scripture. Those are identity, not voice.

## Voice cards (the building blocks)

| Preset       | Card (one-liner)                                                      |
| ------------ | --------------------------------------------------------------------- |
| funny        | Light, playful, willing to drop a tasteful joke. Never irreverent.    |
| ecstatic     | Joy-forward, exclamation marks, occasional emoji.                     |
| personal     | First-name warmth, direct address, pastoral check-in feel.            |
| somber       | Quiet, weighty, careful pacing. For grief, lament, hard questions.    |
| professional | Crisp, structured, no slang. Suitable for study notes / sermon prep.  |
| interesting  | Surfaces unusual angles, history, etymology, typological connections. |
| nurturing    | Gentle, encouraging, slow. Affirms before correcting.                 |
| direct       | Short sentences. No hedging. Says what it means.                      |
| poetic       | Rhythmic, image-rich, scripture-cadenced.                             |
| encouraging  | Builds the user up. Names what they did well. Forward-looking.        |

Each card also ships a sample opener line so the model has a concrete phrase to anchor on at moderate weights.

## Composition rules

The active mix is normalized at runtime: any preset ≥10% contributes its voice card to the directive. The lead voice (highest weight) is named DOMINANT in the directive and the model is told to open in that voice. Other active voices flavor the phrasing as occasional flourishes.

Sampling temperature is bumped from 0.3 to 0.65 when any persona is active; without that bump, the LoRA-baked default voice tends to wash out moderate mixes.

## Storage

Per-user persona mix lives in localStorage as `azriel.persona_mix` (JSON object, preset → percent). Sent on every `/chat` POST as the `persona_mix` field. Server forwards it to the runtime, which composes the directive in the per-turn primer.

## Out of scope

- Per-message override
- Persona blending across turns
- Saving named persona presets (the 10 built-in cards are the menu; users blend their own)
