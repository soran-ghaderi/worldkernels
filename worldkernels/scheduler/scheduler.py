r"""Scheduler — continuous-batching dispatch layer.

Owns the transient `StepRequest` queue and
dispatches work to a `Worker`. It does *not*
own sessions — the engine's session registry does.

Two entry points:

- `step()` — synchronous, used by ``Session.step`` today: dispatch one
  request and return its result.
- `add_request()` / `run_scheduled()` — the queue path: many requests
  are enqueued, partitioned into `CompatibilityGroup` batches, and each
  group dispatched as one batched forward. The async engine drives this.

Admission and preemption policy live here as `admission` and
`preemption` for the engine to consult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from worldkernels.config import SchedulerConfig
from worldkernels.scheduler.admission import AdmissionController
from worldkernels.scheduler.batching import CompatibilityGroup, group_requests
from worldkernels.scheduler.preemption import PreemptionPolicy

if TYPE_CHECKING:
    from worldkernels.core.action import Action
    from worldkernels.core.observation import Observation
    from worldkernels.core.request import StepRequest
    from worldkernels.core.session import LatentState
    from worldkernels.worker.worker import Worker
    from worldkernels.worlds.base import WorldModel

__all__ = ["Scheduler"]


class Scheduler:
    r"""Dispatches step requests to a worker, batching compatible ones.

    Args:
        worker: The worker that executes requests.
        config: Batching / admission / preemption policy.
    """

    def __init__(self, worker: "Worker", config: SchedulerConfig | None = None) -> None:
        self.worker = worker
        self.config = config or SchedulerConfig()
        self.admission = AdmissionController(self.config)
        self.preemption = PreemptionPolicy(self.config.preemption_mode)
        self._queue: list[StepRequest] = []

    @property
    def pending(self) -> int:
        r"""Number of requests waiting in the queue."""
        return len(self._queue)

    def step(
        self,
        *,
        world: "WorldModel",
        state: "LatentState",
        action: "Action",
        modalities: list[str],
        step_index: int,
        decode: bool = True,
    ) -> tuple["LatentState", "Observation"]:
        r"""Dispatch one simulation step synchronously and return its result."""
        from worldkernels.core.request import StepRequest

        request = StepRequest(
            session_id="",
            world=world,
            state=state,
            action=action,
            modalities=list(modalities),
            step_index=step_index,
            decode=decode,
        )
        (result,) = self.worker.execute([request])
        return result

    def add_request(self, request: "StepRequest") -> None:
        r"""Enqueue a step request for the next scheduling round."""
        self._queue.append(request)

    def schedule(self, max_batch_size: int | None = None) -> list[CompatibilityGroup]:
        r"""Drain the queue into compatibility-group batches.

        Args:
            max_batch_size: Override the configured cap. ``1`` collapses batching
                (one request per group) — used when ``continuous_batching`` is off.
        """
        if not self._queue:
            return []
        cap = max_batch_size if max_batch_size is not None else self.config.max_batch_size
        groups = group_requests(self._queue, max_batch_size=cap)
        self._queue.clear()
        return groups

    def run_scheduled(
        self, max_batch_size: int | None = None
    ) -> dict[str, tuple["LatentState", "Observation"]]:
        r"""Schedule pending requests, dispatch each group batched, and collect results.

        Returns a mapping from ``session_id`` to ``(new_state, observation)``.
        """
        results: dict[str, tuple[LatentState, Observation]] = {}
        for group in self.schedule(max_batch_size=max_batch_size):
            outputs = self.worker.execute(group.requests)
            for request, output in zip(group.requests, outputs):
                results[request.session_id] = output
        return results
