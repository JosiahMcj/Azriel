from dataclasses import dataclass


@dataclass
class AzrielConfig:
    # Looped middle slice. Defaults match the plan: a 24-layer window
    # in the middle of Qwen3-Coder's 64 layers. Set loop_max_iters=1 to disable
    # looping (pass-through, identical to base model).
    loop_layer_start: int = 12
    loop_layer_end: int = 36
    loop_max_iters: int = 1
    loop_use_act: bool = False

    # Latent Thought Injection. Zero-init residual MLP -- identity when disabled.
    lti_enabled: bool = False
    lti_expansion: int = 4

    # Tool heads. Off by default; biased to never fire when first enabled so
    # supervised tool-use training in can move the bias gradually.
    tool_heads_enabled: bool = False

    # ACT halting parameters (Graves 2016).
    halt_threshold: float = 0.99
    ponder_cost_lambda: float = 0.01

    # Bias init for ACT and ToolSignal: large positive bias on halt_head means
    # initial halt_p is near 1 (model halts after one iteration -- safe default).
    # Large negative bias on tool_signal means initial p(tool) is near 0.
    act_bias_init: float = 5.0
    tool_signal_bias_init: float = -5.0
