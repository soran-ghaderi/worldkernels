r"""Tests for worldkernels/core/session.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from tests._helpers.factories import make_world_config
from tests._helpers.mocks import FakeTensor, MockWorld
from worldkernels.core.action import Action
from worldkernels.core.errors import (
    CheckpointNotFoundError,
    SessionPausedError,
    SessionTerminatedError,
)
from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState, Session, SessionStatus


class TestSessionStatus:
    def test_values(self):
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.TERMINATED.value == "terminated"

    def test_str_enum(self):
        assert SessionStatus.ACTIVE == "active"


class TestLatentState:
    def test_default_construction(self):
        st = LatentState()
        assert st.data is None
        assert st.device == "cpu"
        assert st.nbytes == 0

    def test_nbytes_with_tensor(self):
        t = torch.zeros(4, 4, dtype=torch.float32)
        st = LatentState(data=t)
        assert st.nbytes == 4 * 4 * 4

    def test_nbytes_with_non_tensor_payload(self):
        st = LatentState(data=object())
        assert st.nbytes == 0

    def test_clone_none(self):
        st = LatentState(data=None, device="cuda:1")
        cloned = st.clone()
        assert cloned.data is None
        assert cloned.device == "cuda:1"
        assert cloned is not st

    def test_clone_tensor_independent(self):
        st = LatentState(data=torch.ones(3, 3))
        cloned = st.clone()
        cloned.data.fill_(0)
        assert torch.all(st.data == 1)

    def test_clone_falls_back_to_deepcopy(self):
        st = LatentState(data={"k": [1, 2, 3]})
        cloned = st.clone()
        cloned.data["k"].append(99)
        assert st.data["k"] == [1, 2, 3]

    def test_clone_uses_payload_clone_method(self):
        st = LatentState(data=FakeTensor(7))
        cloned = st.clone()
        assert isinstance(cloned.data, FakeTensor)
        assert cloned.data.cloned is True

    def test_to_none_data(self):
        st = LatentState(data=None, device="cpu")
        moved = st.to("cuda:0")
        assert moved.data is None
        assert moved.device == "cuda:0"

    def test_to_same_device_no_op(self):
        t = torch.ones(2, 2)
        st = LatentState(data=t, device="cpu")
        moved = st.to("cpu")
        assert moved.data is t

    def test_to_different_device_uses_payload_to(self):
        st = LatentState(data=FakeTensor(1), device="cpu")
        moved = st.to("cuda")
        assert moved.data.value == ("cuda", 1)
        assert moved.device == "cuda"

    def test_to_without_to_method(self):
        st = LatentState(data=42, device="cpu")
        moved = st.to("cuda")
        assert moved.data == 42
        assert moved.device == "cuda"


class TestSessionDefaults:
    def test_unique_session_ids(self):
        a, b = Session(), Session()
        assert a.session_id != b.session_id
        assert a.session_id.startswith("sess_")

    def test_id_alias(self):
        s = Session()
        assert s.id == s.session_id

    def test_default_state_active(self):
        s = Session()
        assert s.status == SessionStatus.ACTIVE
        assert s.step_index == 0
        assert s.parent_session_id is None


class TestSessionStepGuards:
    def _make(self, status: SessionStatus) -> Session:
        s = Session(_world=MockWorld(), _scheduler=MagicMock())
        s.status = status
        return s

    def test_step_after_terminated(self):
        s = self._make(SessionStatus.TERMINATED)
        with pytest.raises(SessionTerminatedError):
            s.step(Action("null"))

    def test_step_when_paused(self):
        s = self._make(SessionStatus.PAUSED)
        with pytest.raises(SessionPausedError):
            s.step(Action("null"))

    def test_step_without_scheduler_or_world_raises(self):
        s = Session()
        with pytest.raises(RuntimeError, match="not bound"):
            s.step(Action("null"))


class TestSessionStep:
    def test_default_modalities_frames(self, engine):
        sess = engine.create_session("dummy", config=make_world_config(), seed=1)
        obs = sess.step(Action("null"))
        assert isinstance(obs, Observation)
        assert obs.frames is not None

    def test_explicit_modalities(self, engine):
        sess = engine.create_session("dummy", config=make_world_config(), seed=1)
        obs = sess.step(Action("null"), modalities=["frames", "depth"])
        assert obs.frames is not None
        assert obs.depth is not None
        assert obs.audio is None

    def test_increments_step_index(self, engine):
        sess = engine.create_session("dummy", config=make_world_config(), seed=1)
        for _ in range(3):
            sess.step(Action("null"))
        assert sess.step_index == 3

    def test_updates_last_active(self, engine):
        sess = engine.create_session("dummy", config=make_world_config(), seed=1)
        before = sess.last_active_at
        sess.step(Action("null"))
        assert sess.last_active_at >= before

    def test_decode_false_skips_decode_stage(self, engine):
        sess = engine.create_session("dummy", config=make_world_config(), seed=1)
        obs = sess.step(Action("null"), decode=False)
        assert obs.decode_skipped is True
        assert obs.frames is None


class TestCheckpointBranchRestore:
    def test_checkpoint_returns_id(self, session):
        ckpt = session.checkpoint()
        assert ckpt.startswith("ckpt_")
        assert ckpt in session._checkpoints

    def test_checkpoint_stores_clone(self, session):
        original = session.state.data.clone()
        ckpt = session.checkpoint()
        session.step(Action("null"))
        assert torch.equal(session._checkpoints[ckpt].data, original)

    def test_restore_resets_state(self, session):
        original = session.state.data.clone()
        ckpt = session.checkpoint()
        session.step(Action("null"))
        session.restore(ckpt)
        assert torch.equal(session.state.data, original)

    def test_restore_unknown_raises(self, session):
        with pytest.raises(CheckpointNotFoundError):
            session.restore("not_a_real_ckpt")

    def test_branch_creates_independent_session(self, session):
        session.step(Action("null"))
        branched = session.branch()
        assert branched.session_id != session.session_id
        assert branched.parent_session_id == session.session_id
        assert branched.step_index == session.step_index
        assert branched.world_id == session.world_id
        branched.state.data.fill_(0)
        assert not torch.equal(branched.state.data, session.state.data)


class TestCloseAndLifecycle:
    def test_close_terminates_and_clears(self, session):
        session.checkpoint()
        session.close()
        assert session.status == SessionStatus.TERMINATED
        assert session.state.data is None
        assert session._checkpoints == {}

    def test_pause_then_resume(self, session):
        session.pause()
        assert session.status == SessionStatus.PAUSED
        session.resume()
        assert session.status == SessionStatus.ACTIVE

    def test_pause_after_terminate_raises(self, session):
        session.close()
        with pytest.raises(SessionTerminatedError):
            session.pause()

    def test_resume_after_terminate_raises(self, session):
        session.close()
        with pytest.raises(SessionTerminatedError):
            session.resume()
