r"""Tests for worldkernels/runtime/memory.py (currently a stub)."""

from __future__ import annotations

from worldkernels.runtime.memory import LatentCacheManager


class TestLatentCacheManager:
    def test_importable_and_instantiable(self):
        mgr = LatentCacheManager()
        assert isinstance(mgr, LatentCacheManager)
