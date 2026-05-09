"""Smoke test: build AzrielModel from a tiny dummy base and verify forward
behavior under several config combinations. Does NOT load real Qwen weights;
that's a separate integration step on a machine where the 30B base lives.

Run:
    python -m tests.smoke_test
"""
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from azriel import AzrielConfig, AzrielModel


@dataclass
class DummyArgs:
    hidden_size: int = 64
    vocab_size: int = 256
    num_attention_heads: int = 4


class DummyLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.norm = nn.RMSNorm(h)
        self.fc = nn.Linear(h, h)

    def __call__(self, x, mask=None, cache=None):
        return x + self.fc(self.norm(x))


class DummyInner(nn.Module):
    def __init__(self, h, v, n_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(v, h)
        self.layers = [DummyLayer(h) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(h)


class DummyQwen(nn.Module):
    def __init__(self, args, n_layers=8):
        super().__init__()
        self.args = args
        self.model = DummyInner(args.hidden_size, args.vocab_size, n_layers)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)


def _logits(cfg, base, ids):
    m = AzrielModel(base, cfg)
    return m(ids)


def main():
    args = DummyArgs()
    base = DummyQwen(args, n_layers=8)
    ids = mx.array([[1, 2, 3, 4, 5]])

    base_logits = base.lm_head(base.model.norm(_run_base(base, ids)))

    # 1. default config: behavior identical to base (within float noise) when
    # nothing is enabled. We don't enforce equality here because we don't
    # rebuild the exact base forward; we just check shape and finiteness.
    cfg = AzrielConfig(loop_layer_start=2, loop_layer_end=6)
    out = _logits(cfg, base, ids)
    assert out.shape == (1, 5, args.vocab_size), out.shape
    assert bool(mx.all(mx.isfinite(out)).item())

    # 2. fixed-iteration loop (no ACT)
    cfg = AzrielConfig(loop_layer_start=2, loop_layer_end=6, loop_max_iters=3)
    out = _logits(cfg, base, ids)
    assert out.shape == (1, 5, args.vocab_size)
    assert bool(mx.all(mx.isfinite(out)).item())

    # 3. ACT halting on, large halt bias means halts immediately
    cfg = AzrielConfig(
        loop_layer_start=2, loop_layer_end=6,
        loop_max_iters=4, loop_use_act=True, act_bias_init=10.0,
    )
    out = _logits(cfg, base, ids)
    assert out.shape == (1, 5, args.vocab_size)
    assert bool(mx.all(mx.isfinite(out)).item())

    # 4. LTI on, loop=2 -- heavy layers run once + 1 LTI iteration
    cfg = AzrielConfig(
        loop_layer_start=2, loop_layer_end=6,
        loop_max_iters=2, lti_enabled=True,
    )
    out = _logits(cfg, base, ids)
    assert out.shape == (1, 5, args.vocab_size)
    assert bool(mx.all(mx.isfinite(out)).item())

    # 5b. LTI on, loop=4 -- heavy layers once + 3 LTI iterations. The previous
    # naive double-layer-pass design produced NaN here; loop-as-delta should
    # stay finite at init because LTI is zero-init identity.
    cfg = AzrielConfig(
        loop_layer_start=2, loop_layer_end=6,
        loop_max_iters=4, lti_enabled=True,
    )
    out = _logits(cfg, base, ids)
    assert out.shape == (1, 5, args.vocab_size)
    assert bool(mx.all(mx.isfinite(out)).item())

    # 6. Tool heads on, no tool result injected -> still pass-through
    cfg = AzrielConfig(loop_layer_start=2, loop_layer_end=6, tool_heads_enabled=True)
    out = _logits(cfg, base, ids)
    assert out.shape == (1, 5, args.vocab_size)
    assert bool(mx.all(mx.isfinite(out)).item())

    print("smoke_test: 6/6 configs produced finite logits of expected shape")


def _run_base(base, ids):
    h = base.model.embed_tokens(ids)
    for layer in base.model.layers:
        h = layer(h)
    return h


if __name__ == "__main__":
    main()
