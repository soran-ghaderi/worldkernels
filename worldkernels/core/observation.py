"""Observation types returned by world models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """
    Multi-modal observation output from a world model step.

    This is NOT RL-specific (no reward/done by default).
    """

    step_index: int
    """Which step this observation corresponds to."""

    generation_time_ms: float
    """Time taken to generate this observation."""

    frames: list[bytes] | None = None
    """RGB video frames (encoded or raw numpy bytes)."""

    latent: bytes | None = None
    """Optional: return latent state for chaining."""

    audio: bytes | None = None
    """Optional: audio track."""

    depth: bytes | None = None
    """Optional: depth map."""

    segmentation: bytes | None = None
    """Optional: semantic segmentation."""

    structured: dict[str, Any] | None = field(default_factory=dict)
    """Optional: structured state (positions, velocities, etc.)."""
