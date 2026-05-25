r"""A scheduled unit of execution work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldkernels.core.action import Action
    from worldkernels.core.session import LatentState
    from worldkernels.worlds.base import WorldModel

__all__ = ["StepRequest"]


@dataclass
class StepRequest:
    r"""One simulation step queued for the scheduler.

    Carries everything the worker needs to advance one session by a step.
    Transient: the persistent ``Session`` state lives in the engine's session
    registry; the request only references it.

    Args:
        session_id: Owning session (empty for engine-internal calls).
        world: The world model to run.
        state: Current latent state.
        action: Action to apply.
        modalities: Observation modalities to decode.
        step_index: Monotonic step counter.
        decode: If False, skip the decode stage.
    """

    session_id: str
    world: WorldModel
    state: LatentState
    action: Action
    modalities: list[str] = field(default_factory=lambda: ["frames"])
    step_index: int = 0
    decode: bool = True
