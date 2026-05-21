r"""Tests for worldkernels/runtime/backends/__init__.py."""

from __future__ import annotations


def test_module_importable():
    import worldkernels.runtime.backends as backends

    assert backends is not None
