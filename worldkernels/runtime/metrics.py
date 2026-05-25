r"""Prometheus metrics for the serving runtime.

Metrics live in a private `CollectorRegistry` so they never collide with
a host application's default registry. The serving layer exposes them at
``/metrics`` via `render()`.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest as _generate_latest,
)

__all__ = [
    "REGISTRY",
    "CONTENT_TYPE_LATEST",
    "render",
    "observe_step",
    "observe_batch",
    "set_active_sessions",
    "set_vram_bytes",
    "set_cache_hit_ratio",
]

REGISTRY = CollectorRegistry()

_STEP_LATENCY = Histogram(
    "wk_step_latency_seconds",
    "Wall-clock latency of one simulation step.",
    registry=REGISTRY,
)
_FRAMES_TOTAL = Counter(
    "wk_frames_generated_total",
    "Total frames generated across all sessions.",
    registry=REGISTRY,
)
_STEPS_TOTAL = Counter(
    "wk_steps_total",
    "Total simulation steps executed.",
    registry=REGISTRY,
)
_BATCH_SIZE = Histogram(
    "wk_batch_size",
    "Number of sessions per batched forward pass.",
    buckets=(1, 2, 4, 8, 16, 32),
    registry=REGISTRY,
)
_ACTIVE_SESSIONS = Gauge(
    "wk_active_sessions",
    "Number of live sessions.",
    registry=REGISTRY,
)
_VRAM_BYTES = Gauge(
    "wk_vram_usage_bytes",
    "Device memory currently allocated.",
    registry=REGISTRY,
)
_CACHE_HIT_RATIO = Gauge(
    "wk_cache_hit_ratio",
    "Denoise-step cache hit ratio in [0, 1].",
    registry=REGISTRY,
)


def observe_step(latency_seconds: float, frames: int = 0) -> None:
    r"""Record one completed simulation step."""
    _STEP_LATENCY.observe(latency_seconds)
    _STEPS_TOTAL.inc()
    if frames:
        _FRAMES_TOTAL.inc(frames)


def observe_batch(size: int) -> None:
    r"""Record the size of one batched forward pass."""
    _BATCH_SIZE.observe(size)


def set_active_sessions(count: int) -> None:
    _ACTIVE_SESSIONS.set(count)


def set_vram_bytes(value: float) -> None:
    _VRAM_BYTES.set(value)


def set_cache_hit_ratio(ratio: float) -> None:
    _CACHE_HIT_RATIO.set(ratio)


def render() -> bytes:
    r"""Render the metrics registry in Prometheus text exposition format."""
    return _generate_latest(REGISTRY)
