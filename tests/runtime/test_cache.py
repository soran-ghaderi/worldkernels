r"""Tests for the TeaCache denoise-step cache."""

from __future__ import annotations

import pytest
import torch

from worldkernels.runtime.cache import TeaCache


class TestTeaCache:
    def test_first_step_always_computes(self):
        cache = TeaCache()
        assert cache.should_compute(torch.zeros(4)) is True

    def test_skips_when_change_below_threshold(self):
        cache = TeaCache(rel_l1_threshold=0.5)
        base = torch.ones(4)
        cache.should_compute(base)
        cache.store(torch.ones(4))
        assert cache.should_compute(base * 1.01) is False
        assert cache.hits == 1

    def test_recomputes_when_change_exceeds_threshold(self):
        cache = TeaCache(rel_l1_threshold=0.05)
        cache.should_compute(torch.ones(4))
        cache.store(torch.ones(4))
        assert cache.should_compute(torch.ones(4) * 5.0) is True

    def test_hit_rate(self):
        cache = TeaCache(rel_l1_threshold=10.0)
        cache.should_compute(torch.ones(4))
        cache.store(torch.ones(4))
        for _ in range(3):
            cache.should_compute(torch.ones(4))
        assert cache.hit_rate == pytest.approx(0.75)

    def test_reset_clears_state(self):
        cache = TeaCache()
        cache.should_compute(torch.ones(4))
        cache.store(torch.ones(4))
        cache.reset()
        assert cache.cached_residual is None
        assert cache.hits == 0

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="rel_l1_threshold"):
            TeaCache(rel_l1_threshold=-1.0)
