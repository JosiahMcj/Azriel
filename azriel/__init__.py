"""Azriel architectural wrapping over Qwen3-Coder-30B-A3B + v0.5 LoRA.

Scope:
  - Looped middle layers (configurable slice, default layers 12..36)
  - Latent Thought Injection (LTI) between loop iterations
  - ACT (Adaptive Computation Time) halting head
  - Tool-use heads: ToolSignalHead / ToolArgsHead / ToolResultInjector

Default config preserves base Qwen behavior so the v0.5-candidate identity
LoRA is unchanged until features are explicitly enabled.
"""
from .config import AzrielConfig
from .model import AzrielModel

__all__ = ["AzrielConfig", "AzrielModel"]
