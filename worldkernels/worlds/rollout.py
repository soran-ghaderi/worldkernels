r"""Rollout policies for presenting a one-shot video generator as a world model.

A `GeneratorWorld` wraps a stateless video
generator. The rollout policy is the explicit, swappable rule that turns the
generator's just-produced clip into the conditioning for the next step. Making
this a named object — rather than a hidden ``last_frame`` hack — keeps the
"generator presented as a world" relationship honest and self-describing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch

__all__ = ["RolloutPolicy", "LastFrameRolloutPolicy"]


class RolloutPolicy(ABC):
    r"""Rule mapping a generated clip to the next step's image conditioning.

    Subclasses declare ``name`` and ``rollout_kind``; the latter is surfaced in
    world metadata so callers know the rollout is not a learned transition.
    """

    name: ClassVar[str]
    rollout_kind: ClassVar[str]

    @abstractmethod
    def next_image_cond(self, generated_video: "torch.Tensor") -> "torch.Tensor":
        r"""Return the next conditioning frame from a generated clip.

        Args:
            generated_video: Clip of shape ``(1, 3, frames, H, W)`` in ``[-1, 1]``.

        Returns:
            Conditioning frame of shape ``(1, 3, H, W)`` in ``[0, 1]``.
        """


class LastFrameRolloutPolicy(RolloutPolicy):
    r"""Condition the next step on the last decoded frame of the previous clip.

    This is open-loop resampling, not a learned state transition: the generator
    has no memory of the action history beyond the single carried frame.
    """

    name: ClassVar[str] = "last_frame"
    rollout_kind: ClassVar[str] = "open_loop_resample"

    def next_image_cond(self, generated_video: "torch.Tensor") -> "torch.Tensor":
        last = generated_video[:, :, -1]
        return ((last + 1.0) * 0.5).clamp(0.0, 1.0)
