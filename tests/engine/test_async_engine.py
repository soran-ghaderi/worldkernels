r"""Tests for the AsyncEngine continuous-batching front-end (CPU-safe)."""

from __future__ import annotations

import asyncio

import pytest

from worldkernels.config import WorldConfig
from worldkernels.core.action import Action
from worldkernels.core.errors import SessionNotFoundError
from worldkernels.engine import AsyncEngine, WorldEngine


def _run(coro):
    return asyncio.run(coro)


def _config() -> WorldConfig:
    return WorldConfig(height=32, width=32, frames_per_step=1)


class TestAsyncEngine:
    def test_step_returns_observation(self):
        async def main():
            engine = AsyncEngine(WorldEngine(device="cpu", max_sessions=4))
            await engine.load_model("dummy")
            session = await engine.create_session("dummy", config=_config())
            obs = await engine.step(session.session_id, Action("null", {}))
            await engine.shutdown()
            return obs

        obs = _run(main())
        assert obs.frames is not None
        assert obs.stage_timing is not None

    def test_step_advances_session_index(self):
        async def main():
            engine = AsyncEngine(WorldEngine(device="cpu", max_sessions=4))
            await engine.load_model("dummy")
            session = await engine.create_session("dummy", config=_config())
            await engine.step(session.session_id, Action("null", {}))
            await engine.step(session.session_id, Action("null", {}))
            idx = session.step_index
            await engine.shutdown()
            return idx

        assert _run(main()) == 2

    def test_concurrent_steps_batch_together(self):
        async def main():
            engine = AsyncEngine(WorldEngine(device="cpu", max_sessions=8))
            await engine.load_model("dummy")
            sessions = [await engine.create_session("dummy", config=_config()) for _ in range(4)]
            results = await asyncio.gather(
                *(engine.step(s.session_id, Action("null", {})) for s in sessions)
            )
            await engine.shutdown()
            return results

        results = _run(main())
        assert len(results) == 4
        assert all(obs.frames is not None for obs in results)

    def test_step_unknown_session_raises(self):
        async def main():
            engine = AsyncEngine(WorldEngine(device="cpu"))
            try:
                with pytest.raises(SessionNotFoundError):
                    await engine.step("ghost", Action("null", {}))
            finally:
                await engine.shutdown()

        _run(main())
