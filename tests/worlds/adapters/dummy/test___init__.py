r"""Tests for worldkernels/worlds/adapters/dummy/__init__.py (lazy export)."""

from __future__ import annotations

import importlib

import pytest


def test_lazy_export():
    mod = importlib.import_module("worldkernels.worlds.adapters.dummy")
    from worldkernels.worlds.adapters.dummy.adapter import DummyWorld

    assert mod.DummyWorld is DummyWorld


def test_unknown_attribute_raises():
    mod = importlib.import_module("worldkernels.worlds.adapters.dummy")
    with pytest.raises(AttributeError):
        mod.nonexistent
