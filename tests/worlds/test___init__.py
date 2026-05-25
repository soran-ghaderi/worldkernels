r"""Tests for worldkernels/worlds/__init__.py (lazy import surface)."""

from __future__ import annotations

import importlib

import pytest


def test_all_exports_resolvable():
    worlds = importlib.import_module("worldkernels.worlds")
    for name in worlds.__all__:
        assert getattr(worlds, name) is not None


def test_lazy_attributes_match_targets():
    worlds = importlib.import_module("worldkernels.worlds")
    from worldkernels.worlds.base import WorldModel
    from worldkernels.worlds.registry import get_world_class, list_worlds, register_world

    assert worlds.WorldModel is WorldModel
    assert worlds.get_world_class is get_world_class
    assert worlds.list_worlds is list_worlds
    assert worlds.register_world is register_world


def test_unknown_attribute_raises():
    worlds = importlib.import_module("worldkernels.worlds")
    with pytest.raises(AttributeError, match="has no attribute"):
        worlds.does_not_exist
