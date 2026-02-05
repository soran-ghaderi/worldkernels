"""Configuration dataclasses for WorldKernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WorldConfig:
    """Per-session generation parameters."""

    # Resolution / format
    height: int = 480
    width: int = 848
    fps: int = 24

    # Generation quality
    num_inference_steps: int = 4
    guidance_scale: float = 1.0

    # Chunking (how many frames per step call)
    frames_per_step: int = 8

    # Conditioning
    initial_prompt: str | None = None
    initial_image: str | None = None  # path or base64

    # Compute budget
    max_vram_gb: float | None = None
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"


@dataclass
class ServerConfig:
    """Configuration for the HTTP/WebSocket server."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_sessions: int = 4
    api_key: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
