"""Mid-forward tool-use heads.

ToolSignalHead -- per-token P(should call tool). Bias-init negative so the
                     model never fires until trained.
ToolArgsHead -- generates tool-call payload tokens (separate logits proj
                     so we don't disturb the main lm_head).
ToolResultInjector-- folds a tool result back into the hidden stream as a
                     residual gated by a learned per-token mixing weight.

This module only adds the structural hooks. Actual tool-use supervision
lands in a follow-on training step where we collect tool-call traces and
fine-tune these heads.
"""
import mlx.core as mx
import mlx.nn as nn


class ToolSignalHead(nn.Module):
    def __init__(self, hidden_size: int, bias_init: float = -5.0):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        self.proj.bias = mx.full(self.proj.bias.shape, bias_init)

    def __call__(self, h: mx.array) -> mx.array:
        return mx.sigmoid(self.proj(h)).squeeze(-1)


class ToolArgsHead(nn.Module):
    """Separate logits projection for tool-arg tokens, kept distinct from the
    main lm_head so tool decoding cannot accidentally bleed into normal text
    generation. Tied to the same vocabulary."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        # zero-init: untrained tool-arg head produces uniform logits
        self.proj.weight = mx.zeros_like(self.proj.weight)

    def __call__(self, h: mx.array) -> mx.array:
        return self.proj(h)


class ToolResultInjector(nn.Module):
    """Inject embedded tool-result tokens back into the hidden stream.

    Inputs:
      h (B, T, H) -- current hidden state
      result_emb (B, T, H) -- embedded tool-result tokens (already passed
                                 through the base model's embed table by the
                                 caller), zero where no tool result applies.

    A learned scalar gate per token controls how much of the result is mixed
    in. Gate is zero-initialized so this is a no-op at init.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)
        self.gate.weight = mx.zeros_like(self.gate.weight)
        self.gate.bias = mx.zeros_like(self.gate.bias)

    def __call__(self, h: mx.array, result_emb: mx.array) -> mx.array:
        g = mx.sigmoid(self.gate(h)) # (B, T, 1)
        return h + g * result_emb
