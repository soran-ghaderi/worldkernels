r"""Top-level engine configuration composing the runtime sub-configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from worldkernels.config.cache_config import CacheConfig
from worldkernels.config.parallel_config import ParallelConfig
from worldkernels.config.scheduler_config import SchedulerConfig

__all__ = ["EngineConfig"]


@dataclass
class EngineConfig:
    r"""Process-wide engine configuration.

    Args:
        device: Compute device (``"cuda"``, ``"cpu"``, or ``"cuda:N"``).
        dtype: Compute precision; ``"auto"`` picks bf16 on Ampere+, else fp16.
        parallel: Multi-GPU parallelism degrees.
        cache: Block-memory and caching policy.
        scheduler: Continuous-batching scheduler policy.
    """

    device: str = "cuda"
    dtype: Literal["bf16", "fp16", "fp32", "auto"] = "auto"
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
