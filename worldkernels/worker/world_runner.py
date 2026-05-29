r"""WorldRunner — the compute layer.

Holds the executor and runs batched step requests under a
`ForwardContext`. Equivalent to
vLLM-Omni's ModelRunner: all model compute happens here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from worldkernels.runtime.executor import Executor
from worldkernels.runtime.forward_context import ForwardContext, set_forward_context

if TYPE_CHECKING:
    import torch

    from worldkernels.config import RuntimeConfig
    from worldkernels.core.observation import Observation
    from worldkernels.core.request import StepRequest
    from worldkernels.core.session import LatentState

__all__ = ["WorldRunner"]


class WorldRunner:
    r"""Runs world-model steps on one device.

    Args:
        device: Target device string.
        dtype: Compute dtype.
        config: Runtime config (component toggles); used by the executor.
    """

    def __init__(
        self,
        device: str,
        dtype: "torch.dtype",
        config: "RuntimeConfig | None" = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.config = config
        self.executor = Executor(device=device, dtype=dtype, config=config)

    def run_batch(
        self,
        requests: list["StepRequest"],
    ) -> list[tuple["LatentState", "Observation"]]:
        r"""Execute a batch of step requests under a forward context."""
        with set_forward_context(ForwardContext()):
            return self.executor.execute_batched(requests)
