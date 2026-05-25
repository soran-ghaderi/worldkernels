r"""Tests for the Prometheus metrics module."""

from __future__ import annotations

from worldkernels.runtime import metrics


def test_render_emits_prometheus_text():
    out = metrics.render()
    assert isinstance(out, bytes)
    assert b"wk_step_latency_seconds" in out


def test_observe_step_increments_counters():
    before = metrics.render()
    metrics.observe_step(0.05, frames=4)
    after = metrics.render()
    assert b"wk_steps_total" in after
    assert before != after


def test_observe_batch_records_size():
    metrics.observe_batch(4)
    assert b"wk_batch_size" in metrics.render()


def test_gauges_settable():
    metrics.set_active_sessions(3)
    metrics.set_vram_bytes(1024.0)
    metrics.set_cache_hit_ratio(0.5)
    out = metrics.render()
    assert b"wk_active_sessions 3.0" in out
    assert b"wk_cache_hit_ratio 0.5" in out
