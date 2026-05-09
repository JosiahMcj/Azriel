"""Latent Thought Injection.

Bottleneck residual: down-project to a tiny inner dim, nonlinearity, up-project
back. The UP projection is zero-initialized so f(h) = 0 at init -> the residual
is identity and the wrapper produces logits identical to the base model.

Sized small (default bottleneck=64) so the activation memory and gradient
overhead is negligible compared to the wrapped 30B base model. SwiGLU shape
preserved as an option (for eventual transfer of weights from a Qwen MLP)
but disabled by default to keep activation memory in check.
"""
import mlx.core as mx
import mlx.nn as nn


class LatentThoughtInjector(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 1):
        super().__init__()
        # `expansion` is reused as a multiplier on a small base bottleneck of
        # 64. expansion=1 -> 64-dim bottleneck, expansion=2 -> 128, etc.
        bottleneck = 64 * max(1, int(expansion))
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        # zero-init UP so the residual is identity at init
        self.up.weight = mx.zeros_like(self.up.weight)

    def __call__(self, h: mx.array) -> mx.array:
        return h + self.up(nn.silu(self.down(h)))
