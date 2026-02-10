"""Configuration dataclasses for WorldKernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WorldConfig:
    r"""Per-session generation parameters.

    For bidirectional models, ``frames_per_step`` is the full chunk size.
    For causal models, ``context_window`` and ``chunk_overlap`` control
    streaming and KV cache behavior.
    """

    # Resolution / format
    height: int = 480
    width: int = 848
    fps: int = 24

    # Generation quality
    num_inference_steps: int = 4
    guidance_scale: float = 1.0

    # Chunking
    frames_per_step: int = 8
    chunk_overlap: int = 0

    # KV cache (causal models)
    context_window: int = 0
    attention_sink_tokens: int = 0

    # Conditioning
    initial_prompt: str | None = None
    initial_image: str | None = None

    # Compute budget
    max_vram_gb: float | None = None
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"


@dataclass
class ServerConfig:
    r"""Configuration for the HTTP/WebSocket server."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_sessions: int = 4
    api_key: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
