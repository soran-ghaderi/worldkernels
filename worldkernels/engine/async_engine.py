r"""Async engine — the continuous-batching serving front-end.

Wraps a `WorldEngine` with an async API and a background drain loop. A
``step`` call enqueues a `StepRequest` and awaits a future; the drain
loop wakes every ``batch_window`` and dispatches all queued requests as
compatibility-group batches through the scheduler. Concurrent requests for the
same world thus share a batched forward pass — genuine continuous batching.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from worldkernels.core.errors import (
    SessionNotFoundError,
    SessionPausedError,
    SessionTerminatedError,
)
from worldkernels.runtime import metrics

if TYPE_CHECKING:
    from worldkernels.config import WorldConfig
    from worldkernels.core.action import Action
    from worldkernels.core.observation import Observation
    from worldkernels.core.session import Session
    from worldkernels.engine.world_engine import WorldEngine

__all__ = ["AsyncEngine"]


class AsyncEngine:
    r"""Async, continuous-batching wrapper around a `WorldEngine`.

    Args:
        engine: The synchronous engine to drive.
        batch_window: Seconds to accumulate requests before a drain; the
            micro-batching window.
    """

    def __init__(self, engine: "WorldEngine", *, batch_window: float = 0.005) -> None:
        self.engine = engine
        self.batch_window = batch_window
        _cfg = getattr(engine, "config", None)
        self._continuous_batching = getattr(_cfg, "continuous_batching", True)
        self._pending: dict[str, asyncio.Future] = {}
        self._drain_task: asyncio.Task | None = None
        self._closed = False

    async def load_model(self, model_id: str, **kwargs: Any) -> None:
        await asyncio.to_thread(self.engine.load_model, model_id, **kwargs)

    async def unload_model(self, name: str) -> None:
        await asyncio.to_thread(self.engine.unload_model, name)

    async def create_session(
        self,
        world: str,
        config: "WorldConfig | None" = None,
        seed: int | None = None,
    ) -> "Session":
        session = await asyncio.to_thread(self.engine.create_session, world, config, seed)
        metrics.set_active_sessions(len(self.engine.list_sessions()))
        return session

    async def close_session(self, session_id: str) -> None:
        await asyncio.to_thread(self.engine.close_session, session_id)
        metrics.set_active_sessions(len(self.engine.list_sessions()))

    async def step(
        self,
        session_id: str,
        action: "Action",
        modalities: list[str] | None = None,
        decode: bool = True,
    ) -> "Observation":
        r"""Execute one step, batched with other concurrent steps."""
        from worldkernels.core.request import StepRequest
        from worldkernels.core.session import SessionStatus

        session = self.engine.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        if session.status is SessionStatus.TERMINATED:
            raise SessionTerminatedError(session_id)
        if session.status is SessionStatus.PAUSED:
            raise SessionPausedError(session_id)
        if session_id in self._pending:
            raise RuntimeError(f"a step is already in flight for session {session_id!r}")

        self._ensure_drain_loop()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[session_id] = future
        self.engine._scheduler.add_request(
            StepRequest(
                session_id=session_id,
                world=session._world,  # type: ignore[arg-type]
                state=session.state,
                action=action,
                modalities=modalities or ["frames"],
                step_index=session.step_index,
                decode=decode,
            )
        )
        return await future

    async def shutdown(self) -> None:
        self._closed = True
        if self._drain_task is not None:
            self._drain_task.cancel()
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        await asyncio.to_thread(self.engine.shutdown)

    def _ensure_drain_loop(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain_loop())

    async def _drain_loop(self) -> None:
        scheduler = self.engine._scheduler
        wait = self.batch_window if self._continuous_batching else 0.0
        cap = None if self._continuous_batching else 1
        while not self._closed:
            await asyncio.sleep(wait)
            batch = scheduler.pending
            if not batch:
                continue
            try:
                results = await asyncio.to_thread(scheduler.run_scheduled, cap)
            except Exception as exc:  # noqa: BLE001 - surface compute errors to callers
                self._fail_pending(exc)
                continue
            metrics.observe_batch(batch)
            for session_id, (new_state, obs) in results.items():
                self._resolve(session_id, new_state, obs)

    def _fail_pending(self, exc: BaseException) -> None:
        r"""Fail every awaiting step with ``exc`` rather than hang on a dead batch."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    def _resolve(self, session_id: str, new_state: Any, obs: "Observation") -> None:
        session = self.engine.get_session(session_id)
        if session is not None:
            session.state = new_state
            session.step_index += 1
            session.last_active_at = datetime.now()
        frames = len(obs.frames) if obs.frames is not None else 0
        metrics.observe_step(obs.generation_time_ms / 1000.0, frames=frames)
        future = self._pending.pop(session_id, None)
        if future is not None and not future.done():
            future.set_result(obs)
