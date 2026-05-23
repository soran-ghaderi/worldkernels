r"""Per-session world-model generation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["WorldConfig"]


@dataclass
class WorldConfig:
    r"""Per-session generation parameters.

    For bidirectional models, ``frames_per_step`` is the full chunk size.
    For causal models, ``context_window`` and ``chunk_overlap`` control
    streaming and KV cache behavior.
    """

    height: int = 480
    width: int = 848
    fps: int = 24

    num_inference_steps: int = 4
    guidance_scale: float = 1.0

    frames_per_step: int = 8
    chunk_overlap: int = 0

    context_window: int = 0
    attention_sink_tokens: int = 0

    initial_prompt: str | None = None
    initial_image: str | None = None

    max_vram_gb: float | None = None
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
