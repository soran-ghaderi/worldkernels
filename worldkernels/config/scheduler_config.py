r"""Scheduler configuration: batching, admission, preemption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["SchedulerConfig"]


@dataclass
class SchedulerConfig:
    r"""Continuous-batching scheduler policy.

    Args:
        max_batch_size: Maximum sessions in one batched forward pass.
        max_concurrent_sessions: Admission cap on live sessions.
        policy: Queue ordering — first-come or priority-weighted.
        preemption_mode: Reclaim VRAM by swapping state to host, or by
            dropping it and recomputing from a checkpoint.
        enable_iteration_batching: Let sessions join a batch mid-denoise via
            ``WorldModel.transition_iter()``.
        admission_headroom_mb: VRAM kept free above the profiled per-session
            cost before a new session is admitted.
    """

    max_batch_size: int = 8
    max_concurrent_sessions: int = 16
    policy: Literal["fcfs", "priority"] = "fcfs"
    preemption_mode: Literal["swap", "recompute"] = "swap"
    enable_iteration_batching: bool = True
    admission_headroom_mb: float = 512.0

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.max_concurrent_sessions < 1:
            raise ValueError(
                f"max_concurrent_sessions must be >= 1, got {self.max_concurrent_sessions}"
            )
