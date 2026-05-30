r"""RuntimeConfig — the single source of truth for component toggles.

A flat "switch panel": one field per worldkernels component, so benchmarks,
ablations, and demos can flip any subsystem on or off. Structured (non-toggle)
parameters stay in the nested `ParallelConfig` / `CacheConfig` / `SchedulerConfig`.

Import-light (no torch) — safe on the CLI cold path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from worldkernels.config.cache_config import CacheConfig
from worldkernels.config.parallel_config import ParallelConfig
from worldkernels.config.scheduler_config import SchedulerConfig

__all__ = [
    "RuntimeConfig",
    "SessionOverrides",
    "TOGGLE_BOOL_FIELDS",
    "TOGGLE_ENUM_FIELDS",
    "ALL_TOGGLE_FIELDS",
    "SESSION_OVERRIDE_FIELDS",
]


@dataclass
class RuntimeConfig:
    r"""Process-wide engine configuration and component switch panel.

    Every optimization is a field here. Defaults preserve current behavior.
    Flags split into *bake-time* (device/dtype/quantization/torch_compile/
    cuda_graphs — fixed at init or warmup) and *runtime* (caching, batching,
    placement, attention — some flip per-session via `SessionOverrides`).
    """

    device: str = "cuda"
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    quantization: Literal["none", "int8", "int4"] = "none"

    torch_compile: bool = True
    cuda_graphs: bool = True

    continuous_batching: bool = True
    iteration_batching: bool = True

    teacache: bool = False
    trajectory_cache: bool = True
    kv_cache_paged: bool = True
    latent_pool: bool = True

    offload_idle: bool = True

    attention_backend: Literal["auto", "flash", "sdpa"] = "auto"

    isolation: Literal["auto", "shared", "isolated"] = "auto"

    max_sessions: int = 4

    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class SessionOverrides:
    r"""Per-session toggle overrides — only the subset safe to flip without a
    warmup/load rebake. ``None`` means "inherit the engine value"."""

    teacache: bool | None = None
    trajectory_cache: bool | None = None
    iteration_batching: bool | None = None
    offload_idle: bool | None = None
    attention_backend: Literal["auto", "flash", "sdpa"] | None = None


TOGGLE_BOOL_FIELDS: tuple[str, ...] = (
    "torch_compile",
    "cuda_graphs",
    "continuous_batching",
    "iteration_batching",
    "teacache",
    "trajectory_cache",
    "kv_cache_paged",
    "latent_pool",
    "offload_idle",
)

TOGGLE_ENUM_FIELDS: dict[str, tuple[str, ...]] = {
    "dtype": ("auto", "bf16", "fp16", "fp32"),
    "quantization": ("none", "int8", "int4"),
    "attention_backend": ("auto", "flash", "sdpa"),
    "isolation": ("auto", "shared", "isolated"),
}

ALL_TOGGLE_FIELDS: tuple[str, ...] = TOGGLE_BOOL_FIELDS + tuple(TOGGLE_ENUM_FIELDS)

SESSION_OVERRIDE_FIELDS: tuple[str, ...] = (
    "teacache",
    "trajectory_cache",
    "iteration_batching",
    "offload_idle",
    "attention_backend",
)
