r"""Preemption policy.

Under VRAM pressure the scheduler reclaims a running session's state. Two
mechanisms: *swap* (copy state to host memory, restore later) and *recompute*
(drop state, replay the action sequence from a checkpoint). The policy picks a
victim and the mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["PreemptionCandidate", "PreemptionDecision", "PreemptionPolicy"]


@dataclass
class PreemptionCandidate:
    r"""A running session considered for preemption.

    Args:
        session_id: The session.
        last_active_step: Monotonic step index of its last activity.
        priority: Higher means more important; preempt low-priority first.
        state_bytes: Footprint of its latent state.
    """

    session_id: str
    last_active_step: int
    priority: int = 0
    state_bytes: int = 0


@dataclass
class PreemptionDecision:
    victim_id: str
    mode: Literal["swap", "recompute"]


class PreemptionPolicy:
    r"""Selects which session to preempt and how.

    Args:
        mode: Preferred reclamation mechanism.
    """

    def __init__(self, mode: Literal["swap", "recompute"] = "swap") -> None:
        self.mode = mode

    def select_victim(
        self,
        candidates: "Sequence[PreemptionCandidate]",
    ) -> PreemptionDecision | None:
        r"""Pick the lowest-priority, least-recently-active session to preempt.

        Returns ``None`` when there is nothing to preempt.
        """
        if not candidates:
            return None
        victim = min(candidates, key=lambda c: (c.priority, c.last_active_step))
        return PreemptionDecision(victim_id=victim.session_id, mode=self.mode)
