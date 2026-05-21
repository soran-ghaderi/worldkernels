r"""Tests for worldkernels/core/errors.py."""

from __future__ import annotations

import pytest

from worldkernels.core.errors import (
    CheckpointNotFoundError,
    SessionLimitError,
    SessionNotFoundError,
    SessionPausedError,
    SessionTerminatedError,
    VRAMExhaustedError,
    WorldAlreadyLoadedError,
    WorldInitError,
    WorldKernelError,
    WorldNotFoundError,
)


@pytest.mark.parametrize(
    "exc_cls,args,attrs,fragment",
    [
        (WorldNotFoundError, ("foo",), {"world_id": "foo"}, "foo"),
        (SessionLimitError, (8,), {"max_sessions": 8}, "8"),
        (SessionNotFoundError, ("s_1",), {"session_id": "s_1"}, "s_1"),
        (SessionTerminatedError, ("s_1",), {"session_id": "s_1"}, "terminated"),
        (VRAMExhaustedError, (1024.0, 256.0), {"required_mb": 1024.0, "available_mb": 256.0}, "1024"),
        (WorldInitError, ("foo", "bar"), {"world_id": "foo", "reason": "bar"}, "bar"),
        (WorldAlreadyLoadedError, ("foo",), {"name": "foo"}, "already loaded"),
        (
            CheckpointNotFoundError,
            ("ck_1", "s_1"),
            {"checkpoint_id": "ck_1", "session_id": "s_1"},
            "ck_1",
        ),
        (SessionPausedError, ("s_1",), {"session_id": "s_1"}, "paused"),
    ],
)
def test_error_signatures(exc_cls, args, attrs, fragment):
    err = exc_cls(*args)
    assert isinstance(err, WorldKernelError)
    assert isinstance(err, Exception)
    for k, v in attrs.items():
        assert getattr(err, k) == v
    assert fragment.lower() in str(err).lower()


class TestHierarchy:
    def test_all_inherit_world_kernel_error(self):
        for cls in (
            WorldNotFoundError,
            SessionLimitError,
            SessionNotFoundError,
            SessionTerminatedError,
            VRAMExhaustedError,
            WorldInitError,
            WorldAlreadyLoadedError,
            CheckpointNotFoundError,
            SessionPausedError,
        ):
            assert issubclass(cls, WorldKernelError)

    def test_can_catch_via_base(self):
        with pytest.raises(WorldKernelError):
            raise SessionLimitError(2)
