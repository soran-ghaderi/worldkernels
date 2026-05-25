r"""Tests for worldkernels/__init__.py (lazy import surface)."""

from __future__ import annotations

import importlib

import pytest

import worldkernels


def test_version_present():
    assert isinstance(worldkernels.__version__, str)
    assert len(worldkernels.__version__) > 0


def test_all_exports_resolvable():
    for name in worldkernels.__all__:
        getattr(worldkernels, name)


def test_lazy_exports_match_targets():
    from worldkernels.core.action import Action
    from worldkernels.core.config import ServerConfig, WorldConfig
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
    from worldkernels.core.observation import Observation
    from worldkernels.core.session import LatentState, Session, SessionStatus
    from worldkernels.engine import WorldEngine
    from worldkernels.runtime.stages import (
        StageConfig,
        StageExecMode,
        StageOutput,
        StageTiming,
        StageType,
        TransitionMode,
    )

    expected = {
        "Action": Action,
        "WorldConfig": WorldConfig,
        "ServerConfig": ServerConfig,
        "WorldEngine": WorldEngine,
        "Observation": Observation,
        "Session": Session,
        "SessionStatus": SessionStatus,
        "LatentState": LatentState,
        "StageConfig": StageConfig,
        "StageExecMode": StageExecMode,
        "StageOutput": StageOutput,
        "StageTiming": StageTiming,
        "StageType": StageType,
        "TransitionMode": TransitionMode,
        "WorldKernelError": WorldKernelError,
        "WorldNotFoundError": WorldNotFoundError,
        "WorldAlreadyLoadedError": WorldAlreadyLoadedError,
        "WorldInitError": WorldInitError,
        "SessionLimitError": SessionLimitError,
        "SessionNotFoundError": SessionNotFoundError,
        "SessionTerminatedError": SessionTerminatedError,
        "SessionPausedError": SessionPausedError,
        "CheckpointNotFoundError": CheckpointNotFoundError,
        "VRAMExhaustedError": VRAMExhaustedError,
    }
    for name, cls in expected.items():
        assert getattr(worldkernels, name) is cls


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute"):
        worldkernels.does_not_exist_zzz


def test_lazy_value_cached():
    importlib.reload(worldkernels)
    first = worldkernels.Action
    second = worldkernels.Action
    assert first is second
