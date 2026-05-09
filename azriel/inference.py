"""Loader for Azriel inference.

Default path: load the configured base model + the v0.6.x adapter
(LoRA deltas + LTI weights), wrap in AzrielModel, return a model
object compatible with mlx_lm.generate.

Alternate paths supported ():
  - empty adapter_path -> raw base model, no adapter applied. Lets
    users run the runtime stack against any MLX-compatible local
    model they already have.
  - disable_wrapper=True -> skip the AzrielModel architecture wrap
    entirely. Useful when the base model isn't Qwen-block-shaped
    (the wrapper's looped-layer + LTI plumbing assumes Qwen's
    decoder block layout).

mlx_lm.generate calls model(tokens, cache=...) -- the wrapper's
__call__ matches that signature, so generate works without
modification either way.
"""
import json
from pathlib import Path

from mlx_lm import load as mlx_load
from mlx_lm.tuner.utils import linear_to_lora_layers

from .config import AzrielConfig
from .model import AzrielModel


def load_phase_beta(
    base_model: str,
    adapter_path: str = "",
    config: AzrielConfig | None = None,
    disable_wrapper: bool = False,
):
    """Returns (model, tokenizer). `model` is either an AzrielModel
    wrapper or the raw base model depending on disable_wrapper."""
    base, tokenizer = mlx_load(base_model)

    if adapter_path:
        adapter = Path(adapter_path)
        cfg_path = adapter / "adapter_config.json"
        if cfg_path.exists():
            src_cfg = json.loads(cfg_path.read_text())
            num_layers = src_cfg["num_layers"]
            lora_params = src_cfg["lora_parameters"]
            base.freeze()
            linear_to_lora_layers(base, num_layers, lora_params, use_dora=False)
            adapter_weights = adapter / "adapters.safetensors"
        else:
            adapter_weights = None
    else:
        adapter_weights = None

    if disable_wrapper:
        # Raw base model + (optional) LoRA. No custom architecture.
        if adapter_weights and adapter_weights.exists():
            base.load_weights(str(adapter_weights), strict=False)
        return base, tokenizer

    if config is None:
        # Default custom-architecture config (Qwen3-Coder layout).
        config = AzrielConfig(
            loop_layer_start=32,
            loop_layer_end=48,
            loop_max_iters=2,
            lti_enabled=True,
            lti_expansion=1,
        )

    wrapped = AzrielModel(base, config)
    if adapter_weights and adapter_weights.exists():
        # load_weights walks the wrapper tree: LoRA deltas land in the
        # right base.model.layers.X.lora_a paths, LTI weights land in
        # lti.up/down.
        wrapped.load_weights(str(adapter_weights), strict=False)
    return wrapped, tokenizer
