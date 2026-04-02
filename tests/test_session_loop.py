"""Tests for the full session loop with DummyWorld."""

from __future__ import annotations

import pytest

from worldkernels import Action, WorldConfig, WorldKernel
from worldkernels.core.errors import (
    CheckpointNotFoundError,
    SessionLimitError,
    SessionTerminatedError,
    WorldNotFoundError,
)
from worldkernels.core.session import SessionStatus


@pytest.fixture
def engine():
    """Create a CPU-only WorldKernel with DummyWorld loaded."""
    wk = WorldKernel(device="cpu", max_sessions=4)
    wk.load_model("dummy")
    yield wk
    wk.shutdown()


@pytest.fixture
def session(engine):
    """Create a session with small resolution for fast tests."""
    config = WorldConfig(height=64, width=64, frames_per_step=1)
    return engine.create_session("dummy", config=config, seed=42)


class TestWorldKernel:
    def test_load_model(self, engine):
        assert "dummy" in engine.list_worlds()

    def test_load_unknown_model_raises(self):
        wk = WorldKernel(device="cpu")
        with pytest.raises(WorldNotFoundError):
            wk.load_model("nonexistent_model_xyz")
        wk.shutdown()

    def test_create_session(self, engine):
        config = WorldConfig(height=64, width=64)
        session = engine.create_session("dummy", config=config)
        assert session.world_id == "dummy"
        assert session.status == SessionStatus.ACTIVE
        assert session.step_index == 0

    def test_create_session_unknown_world_raises(self, engine):
        with pytest.raises(WorldNotFoundError):
            engine.create_session("nonexistent")

    def test_max_sessions(self):
        wk = WorldKernel(device="cpu", max_sessions=2)
        wk.load_model("dummy")
        config = WorldConfig(height=32, width=32)
        wk.create_session("dummy", config=config)
        wk.create_session("dummy", config=config)
        with pytest.raises(SessionLimitError):
            wk.create_session("dummy", config=config)
        wk.shutdown()


class TestSessionStep:
    def test_single_step(self, session):
        obs = session.step(Action("keyboard", {"keys": ["W"]}))
        assert obs.step_index == 0
        assert session.step_index == 1
        assert obs.generation_time_ms > 0
        assert obs.frames is not None
        assert len(obs.frames) == 1

    def test_multiple_steps(self, session):
        for i in range(10):
            obs = session.step(Action("keyboard", {"keys": ["W"]}))
            assert obs.step_index == i
        assert session.step_index == 10

    def test_modalities_frames_only(self, session):
        obs = session.step(Action("keyboard", {}), modalities=["frames"])
        assert obs.frames is not None
        assert obs.depth is None
        assert obs.audio is None

    def test_modalities_all(self, session):
        obs = session.step(
            Action("keyboard", {}),
            modalities=["frames", "depth", "audio"],
        )
        assert obs.frames is not None
        assert obs.depth is not None
        assert obs.audio is not None

    def test_step_after_close_raises(self, session):
        session.close()
        with pytest.raises(SessionTerminatedError):
            session.step(Action("keyboard", {}))


class TestCheckpointBranch:
    def test_checkpoint_and_restore(self, session):
        # Take a few steps
        for _ in range(3):
            session.step(Action("keyboard", {"keys": ["W"]}))

        ckpt_id = session.checkpoint()
        assert ckpt_id.startswith("ckpt_")

        # Take more steps
        for _ in range(5):
            session.step(Action("keyboard", {"keys": ["D"]}))
        assert session.step_index == 8

        # Restore
        session.restore(ckpt_id)
        # step_index is not reset by restore (it's a state restore, not time travel)
        # but the latent state should be back to the checkpoint

    def test_restore_unknown_checkpoint_raises(self, session):
        with pytest.raises(CheckpointNotFoundError):
            session.restore("ckpt_nonexistent")

    def test_branch(self, session):
        session.step(Action("keyboard", {"keys": ["W"]}))
        session.step(Action("keyboard", {"keys": ["W"]}))

        branched = session.branch()
        assert branched.session_id != session.session_id
        assert branched.parent_session_id == session.session_id
        assert branched.step_index == session.step_index
        assert branched.world_id == session.world_id

        # Steps on branched session don't affect original
        branched.step(Action("keyboard", {"keys": ["A"]}))
        assert branched.step_index == 3
        assert session.step_index == 2

    def test_close_clears_state(self, session):
        session.step(Action("keyboard", {}))
        session.checkpoint()
        session.close()
        assert session.status == SessionStatus.TERMINATED
        assert session.state.data is None


class TestDeterminism:
    def test_same_seed_same_initial_state(self, engine):
        config = WorldConfig(height=32, width=32)
        s1 = engine.create_session("dummy", config=config, seed=123)
        s2 = engine.create_session("dummy", config=config, seed=123)
        # Same seed should produce same initial latent
        import torch

        assert torch.equal(s1.state.data, s2.state.data)
