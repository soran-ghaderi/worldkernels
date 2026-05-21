r"""Tests for worldkernels/worlds/pipelines/__init__.py."""

from __future__ import annotations


def test_module_importable():
    import worldkernels.worlds.pipelines as pipelines

    assert pipelines.__doc__ is not None
