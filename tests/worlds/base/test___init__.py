r"""Tests for worldkernels/worlds/base/__init__.py."""

from __future__ import annotations


def test_exports_abstract_world():
    from worldkernels.worlds.base import AbstractWorld
    from worldkernels.worlds.base.world import AbstractWorld as W

    assert AbstractWorld is W
