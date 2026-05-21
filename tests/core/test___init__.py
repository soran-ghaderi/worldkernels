r"""Tests for worldkernels/core/__init__.py — eager re-exports."""

from __future__ import annotations

import worldkernels.core as core


def test_all_exports_resolvable():
    for name in core.__all__:
        assert hasattr(core, name), f"missing export: {name}"


def test_exports_correct_classes():
    from worldkernels.core.action import Action
    from worldkernels.core.config import ServerConfig, WorldConfig
    from worldkernels.core.engine import WorldKernel
    from worldkernels.core.observation import Observation
    from worldkernels.core.session import LatentState, Session, SessionStatus

    assert core.Action is Action
    assert core.WorldConfig is WorldConfig
    assert core.ServerConfig is ServerConfig
    assert core.WorldKernel is WorldKernel
    assert core.Observation is Observation
    assert core.Session is Session
    assert core.LatentState is LatentState
    assert core.SessionStatus is SessionStatus
