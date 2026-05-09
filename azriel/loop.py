"""Looped middle-layer block with optional ACT halting.

Design: loop-as-delta.

The heavy transformer layers in the middle slice are run EXACTLY ONCE per
forward pass. Each transformer block was calibrated by base pretraining for
a single sequential application; reapplying the whole stack to its own
output blows up activation magnitudes (NaN logits within a few iterations
during training).

Instead, "more thinking" is delegated to the LatentThoughtInjector: a small
zero-initialized residual MLP that is iterated `loop_max_iters - 1` times on
top of the layers' output. At init each LTI step is the identity (down
projection is zero), so the wrapper is logit-identical to the base model.
Training pushes LTI's `down` away from zero and the iterations start
contributing real refinement of the hidden state.

ACT (Graves 2016) optionally gates how many LTI iterations each TOKEN gets:
each iteration the act_head produces a halting probability; iterations
continue until cumulative halt mass crosses `halt_threshold`. The effective
output is the halt-mass-weighted sum across iterations. The exposed
`last_ponder_cost` is `mean(n_iters + remainders)`, suitable to add to the
loss as `ponder_cost_lambda * cost`.

Loop semantics with this design:
  - layers run once -> position embeddings (RoPE) and KV cache are unaffected
    by the loop count, so the existing MLX-LM cache integration just works
  - LTI iterations are pure point-wise residual MLP updates -- no attention,
    no positional handling, cheap
  - default config (loop_max_iters=1) skips LTI entirely -> exact base
"""
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import AzrielConfig
from .lti import LatentThoughtInjector


class ACTHaltingHead(nn.Module):
    def __init__(self, hidden_size: int, bias_init: float = 5.0):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        self.proj.bias = mx.full(self.proj.bias.shape, bias_init)

    def __call__(self, h: mx.array) -> mx.array:
        return mx.sigmoid(self.proj(h)).squeeze(-1)


class LoopedBlock(nn.Module):
    def __init__(
        self,
        layers: List[nn.Module],
        config: AzrielConfig,
        lti: Optional[LatentThoughtInjector] = None,
        act_head: Optional[ACTHaltingHead] = None,
    ):
        super().__init__()
        self.layers = layers
        self.config = config
        self.lti = lti
        self.act_head = act_head
        self.last_ponder_cost: Optional[mx.array] = None

    def _run_layers(self, h, mask, cache_slice):
        for i, layer in enumerate(self.layers):
            cache = cache_slice[i] if cache_slice is not None else None
            h = layer(h, mask, cache)
        return h

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache_slice: Optional[List[Any]],
    ) -> mx.array:
        # Run the heavy layers exactly once.
        h = self._run_layers(h, mask, cache_slice)

        extra = max(0, self.config.loop_max_iters - 1)
        if extra == 0 or self.lti is None:
            # Base behavior preserved when loop_max_iters == 1, or when LTI is
            # disabled (loop_max_iters > 1 without LTI is degenerate; the
            # config validation in AzrielModel warns in that case).
            self.last_ponder_cost = None
            return h

        if not self.config.loop_use_act or self.act_head is None:
            for _ in range(extra):
                h = self.lti(h)
            self.last_ponder_cost = None
            return h

        # ACT-gated LTI iterations. Same Graves accumulator as before, but the
        # iterated body is now LTI (cheap, point-wise) rather than the full
        # transformer stack (heavy, per-layer attention).
        threshold = self.config.halt_threshold
        cum_halt = mx.zeros(h.shape[:-1])
        weighted_h = mx.zeros_like(h)
        n_iters = mx.zeros(h.shape[:-1])
        remainders = mx.zeros(h.shape[:-1])
        still_running = mx.ones(h.shape[:-1], dtype=mx.bool_)

        for k in range(extra):
            h = self.lti(h)
            p = self.act_head(h)
            new_cum = cum_halt + p

            last_iter = (new_cum >= threshold) | (k == extra - 1)
            still = still_running.astype(h.dtype)
            w_halt = (1.0 - cum_halt) * last_iter.astype(h.dtype)
            w_step = p * (~last_iter).astype(h.dtype)
            w = (w_halt + w_step) * still
            weighted_h = weighted_h + h * w[..., None]

            n_iters = n_iters + still
            remainders = mx.where(last_iter & still_running, 1.0 - cum_halt, remainders)
            cum_halt = new_cum
            still_running = still_running & ~last_iter
            if not bool(mx.any(still_running).item()):
                break

        self.last_ponder_cost = mx.mean(n_iters + remainders)
        return weighted_h
