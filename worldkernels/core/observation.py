"""Observation types returned by world models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worldkernels.runtime.stages import StageTiming


@dataclass
class Observation:
    r"""Multi-modal observation output from a world model step."""

    step_index: int
    generation_time_ms: float
    frames: Any | None = None
    latent: Any | None = None
    audio: Any | None = None
    depth: Any | None = None
    segmentation: Any | None = None
    structured: dict[str, Any] | None = field(default_factory=dict)
    stage_timing: StageTiming | None = None
    decode_skipped: bool = False
