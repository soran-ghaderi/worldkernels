r"""Admission control.

Before a session is created the admission controller checks that its profiled
VRAM cost, plus a headroom margin, fits in free device memory and that the
concurrent-session cap is not exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldkernels.config import SchedulerConfig, WorldConfig
    from worldkernels.worlds.base import WorldModel

__all__ = ["AdmissionDecision", "AdmissionController"]


@dataclass
class AdmissionDecision:
    r"""Outcome of an admission check.

    Args:
        admitted: Whether the session may be created.
        reason: Human-readable explanation when refused.
        required_mb: Profiled VRAM the session needs.
        available_mb: Free device VRAM at decision time.
    """

    admitted: bool
    reason: str = ""
    required_mb: float = 0.0
    available_mb: float = 0.0


class AdmissionController:
    r"""Gates session creation on VRAM headroom and the concurrency cap.

    Args:
        config: Scheduler policy (concurrency cap, headroom margin).
    """

    def __init__(self, config: "SchedulerConfig") -> None:
        self.config = config

    def check(
        self,
        world: "WorldModel",
        world_config: "WorldConfig",
        *,
        live_sessions: int,
        free_vram_mb: float | None,
    ) -> AdmissionDecision:
        r"""Decide whether a new session for ``world`` may be admitted.

        Args:
            world: The world model the session will run.
            world_config: Per-session generation parameters.
            live_sessions: Current number of live sessions.
            free_vram_mb: Free device VRAM, or ``None`` on CPU (skips the check).
        """
        if live_sessions >= self.config.max_concurrent_sessions:
            return AdmissionDecision(
                admitted=False,
                reason=(
                    f"concurrency cap reached "
                    f"({live_sessions}/{self.config.max_concurrent_sessions})"
                ),
            )
        required = world.profile_vram(world_config)
        if free_vram_mb is None:
            return AdmissionDecision(admitted=True, required_mb=required)
        budget = free_vram_mb - self.config.admission_headroom_mb
        if required > budget:
            return AdmissionDecision(
                admitted=False,
                reason=(
                    f"insufficient VRAM: need {required:.0f} MB + "
                    f"{self.config.admission_headroom_mb:.0f} MB headroom, "
                    f"{free_vram_mb:.0f} MB free"
                ),
                required_mb=required,
                available_mb=free_vram_mb,
            )
        return AdmissionDecision(admitted=True, required_mb=required, available_mb=free_vram_mb)
