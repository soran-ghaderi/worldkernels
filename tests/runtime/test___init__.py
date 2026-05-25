r"""Tests for worldkernels/runtime/__init__.py."""

from __future__ import annotations


def test_module_importable():
    import worldkernels.runtime as runtime

    assert runtime is not None
