r"""Worker — the device-infrastructure layer.

One worker owns one device: it binds the CUDA context, initializes the
distributed parallel state, and holds a `WorldRunner`. The scheduler
dispatches step requests to it. Multi-rank workers and out-of-process
isolation are added by the distributed step of the restructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from worldkernels.worker.world_runner import WorldRunner

if TYPE_CHECKING:
    import torch

    from worldkernels.config import ParallelConfig
    from worldkernels.core.observation import Observation
    from worldkernels.core.request import StepRequest
    from worldkernels.core.session import LatentState

__all__ = ["Worker"]


class Worker:
    r"""Owns one device and the runner that executes on it.

    Args:
        device: Target device string.
        dtype: Compute dtype.
        parallel_config: Parallelism degrees; single-rank by default.
    """

    def __init__(
        self,
        device: str,
        dtype: "torch.dtype",
        parallel_config: "ParallelConfig | None" = None,
    ) -> None:
        from worldkernels.config import ParallelConfig
        from worldkernels.distributed import init_distributed, is_initialized

        self.device = device
        self.dtype = dtype
        self.parallel_config = parallel_config or ParallelConfig()
        if not is_initialized():
            init_distributed(self.parallel_config)
        self.runner = WorldRunner(device, dtype)

    def execute(
        self,
        requests: list["StepRequest"],
    ) -> list[tuple["LatentState", "Observation"]]:
        r"""Run a batch of step requests on this worker's device."""
        return self.runner.run_batch(requests)
