"""AzrielModel -- the architectural wrapper.

Composes a pretrained Qwen3-Coder MLX-LM model with:
  - a LoopedBlock over the configured middle slice
  - LatentThoughtInjector applied between iterations (optional)
  - ACT halting head (optional)
  - tool-use heads (optional)

Default config (loop_max_iters=1, every "_enabled" flag False) preserves base
Qwen behavior exactly so the v0.5-candidate identity LoRA is not disturbed
until features are explicitly enabled and trained.

Design note: we do NOT subclass the MLX-LM Model class. We compose. The base
model is held intact; we slice its layer list once at construction time and
reroute the forward pass through our LoopedBlock for the middle slice.
"""
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import AzrielConfig
from .loop import ACTHaltingHead, LoopedBlock
from .lti import LatentThoughtInjector
from .tool_heads import ToolArgsHead, ToolResultInjector, ToolSignalHead

try:
    from mlx_lm.models.base import create_attention_mask
except ImportError: # older mlx_lm layouts re-exported it from each model module
    try:
        from mlx_lm.models.qwen3_moe import create_attention_mask
    except ImportError:
        create_attention_mask = None


class AzrielModel(nn.Module):
    def __init__(self, base_qwen: nn.Module, config: AzrielConfig):
        super().__init__()
        self.base = base_qwen
        self.config = config

        # MLX-LM layout: base.model.embed_tokens, base.model.layers, base.model.norm
        # base.lm_head (or weight tying via base.model.embed_tokens.as_linear).
        # Validate up front so we fail fast on a layout we don't understand.
        for attr in ("model", "args"):
            if not hasattr(base_qwen, attr):
                raise ValueError(f"base_qwen missing .{attr}; expected mlx_lm.models.qwen Model layout")
        layers = base_qwen.model.layers
        n = len(layers)
        if not (0 <= config.loop_layer_start < config.loop_layer_end <= n):
            raise ValueError(
                f"loop slice [{config.loop_layer_start}:{config.loop_layer_end}] "
                f"out of range for {n}-layer base"
            )

        self._pre = layers[: config.loop_layer_start]
        self._mid = layers[config.loop_layer_start : config.loop_layer_end]
        self._post = layers[config.loop_layer_end :]

        h = base_qwen.args.hidden_size
        v = base_qwen.args.vocab_size

        self.lti = LatentThoughtInjector(h, config.lti_expansion) if config.lti_enabled else None
        self.act_head = ACTHaltingHead(h, config.act_bias_init) if config.loop_use_act else None
        self.loop = LoopedBlock(self._mid, config, self.lti, self.act_head)

        if config.tool_heads_enabled:
            self.tool_signal = ToolSignalHead(h, config.tool_signal_bias_init)
            self.tool_args = ToolArgsHead(h, v)
            self.tool_inject = ToolResultInjector(h)
        else:
            self.tool_signal = None
            self.tool_args = None
            self.tool_inject = None

    @property
    def last_ponder_cost(self) -> Optional[mx.array]:
        return self.loop.last_ponder_cost

    @property
    def layers(self):
        # mlx_lm.tuner.trainer.train accesses model.layers[0] to install
        # gradient checkpointing. Forward to the wrapped base's layer list so
        # standard MLX-LM training infra works without modification.
        return self.base.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
        mask: Optional[mx.array] = None,
        tool_result_emb: Optional[mx.array] = None,
    ) -> mx.array:
        m = self.base.model
        h = m.embed_tokens(input_ids)

        # Build the mask the way Qwen3MoeModel.__call__ does it -- that helper
        # is cache-aware (handles both prefill and incremental decode shapes).
        if mask is None:
            if create_attention_mask is not None:
                first_cache = cache[0] if cache is not None else None
                mask = create_attention_mask(h, first_cache)
            elif h.shape[1] > 1:
                mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
                mask = mask.astype(h.dtype)

        def _slice_cache(start: int, end: int):
            if cache is None:
                return None
            return cache[start:end]

        # Pre-loop layers
        for i, layer in enumerate(self._pre):
            h = layer(h, mask, _slice_cache(i, i + 1)[0] if cache is not None else None)

        # Looped middle slice
        cs = self.config.loop_layer_start
        ce = self.config.loop_layer_end
        h = self.loop(h, mask, _slice_cache(cs, ce))

        # Tool-result injection happens AFTER the middle loop and BEFORE the
        # post layers, so the post-loop stack can integrate the tool output
        # into the eventual logits.
        if self.tool_inject is not None and tool_result_emb is not None:
            h = self.tool_inject(h, tool_result_emb)

        # Post-loop layers
        for i, layer in enumerate(self._post):
            idx = ce + i
            h = layer(h, mask, _slice_cache(idx, idx + 1)[0] if cache is not None else None)

        h = m.norm(h)
        if hasattr(self.base, "lm_head") and self.base.lm_head is not None:
            logits = self.base.lm_head(h)
        else:
            # weight-tied head
            logits = m.embed_tokens.as_linear(h)
        return logits
