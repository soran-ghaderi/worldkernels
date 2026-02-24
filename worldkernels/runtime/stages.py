r"""Stage definitions for the world model execution pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageType(str, Enum):
    """Pipeline stage identity."""

    ENCODE = "encode"
    TRANSITION = "transition"
    DECODE = "decode"


class StageExecMode(str, Enum):
    r"""How a stage executes: single forward pass or multi-step loop."""

    SINGLE_SHOT = "single_shot"
    ITERATIVE = "iterative"


class TransitionMode(str, Enum):
    r"""Attention architecture of the transition stage.

    Determines KV caching, streaming capability, and scheduling strategy.
    """

    BIDIRECTIONAL = "bidirectional"
    CAUSAL = "causal"
    HYBRID = "hybrid"


@dataclass
class StageConfig:
    r"""Per-stage execution configuration.

    Args:
        stage_type: Which pipeline stage this configures.
        exec_mode: Single-shot or iterative execution.
        device: Device override (None inherits from engine).
        dtype: Dtype override (None inherits from engine).
        memory_fraction: Fraction of GPU memory budget for this stage.
        backend: Execution backend name ("eager", "compile", "tensorrt").
        enabled: If False, the stage is skipped entirely.
    """

    stage_type: StageType
    exec_mode: StageExecMode = StageExecMode.SINGLE_SHOT
    device: str | None = None
    dtype: str | None = None
    memory_fraction: float = 1.0
    backend: str = "eager"
    enabled: bool = True


@dataclass
class StageOutput:
    r"""Typed inter-stage data payload."""

    stage_type: StageType
    data: Any
    timing_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTiming:
    r"""Per-stage timing breakdown for profiling."""

    encode_action_ms: float = 0.0
    transition_ms: float = 0.0
    decode_observation_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.encode_action_ms + self.transition_ms + self.decode_observation_ms

    def as_dict(self) -> dict[str, float]:
        return {
            "encode_action_ms": self.encode_action_ms,
            "transition_ms": self.transition_ms,
            "decode_observation_ms": self.decode_observation_ms,
            "total_ms": self.total_ms,
        }


DEFAULT_PIPELINE_STAGES: tuple[StageConfig, ...] = (
    StageConfig(StageType.ENCODE, StageExecMode.SINGLE_SHOT),
    StageConfig(StageType.TRANSITION, StageExecMode.ITERATIVE),
    StageConfig(StageType.DECODE, StageExecMode.SINGLE_SHOT),
)
