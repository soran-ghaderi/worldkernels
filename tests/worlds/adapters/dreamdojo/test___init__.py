r"""Tests for worldkernels/worlds/adapters/dreamdojo/__init__.py."""

from __future__ import annotations

import importlib

import pytest


def test_lazy_export():
    mod = importlib.import_module("worldkernels.worlds.adapters.dreamdojo")
    from worldkernels.worlds.adapters.dreamdojo.adapter import DreamDojoWorld

    assert mod.DreamDojoWorld is DreamDojoWorld


def test_unknown_attribute_raises():
    mod = importlib.import_module("worldkernels.worlds.adapters.dreamdojo")
    with pytest.raises(AttributeError):
        mod.does_not_exist
