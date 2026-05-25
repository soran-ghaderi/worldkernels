r"""TeaCache — timestep-embedding-aware denoise-step caching.

Across a denoise loop the transformer's per-step modulation signal changes
slowly. TeaCache accumulates the relative L1 change of that signal; while the
accumulated change stays below a threshold it reports that the heavy
transformer compute can be skipped and the previous step's residual reused.

This holds only the skip decision and the cached residual; the transformer
calls `should_compute()` / `store()` around its block stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["TeaCache"]


class TeaCache:
    r"""Relative-L1 step-skipping cache for a denoise loop.

    Args:
        rel_l1_threshold: Accumulated relative-L1 change at which a full
            recompute is forced. Larger skips more steps (faster, lower quality).
    """

    def __init__(self, rel_l1_threshold: float = 0.15) -> None:
        if rel_l1_threshold < 0:
            raise ValueError(f"rel_l1_threshold must be >= 0, got {rel_l1_threshold}")
        self.rel_l1_threshold = rel_l1_threshold
        self._prev_modulation: "torch.Tensor | None" = None
        self._cached_residual: "torch.Tensor | None" = None
        self._accumulated = 0.0
        self.hits = 0
        self.misses = 0

    def should_compute(self, modulation: "torch.Tensor") -> bool:
        r"""Whether the transformer must run this step, or its residual is reused.

        Updates the accumulated relative-L1 distance from the previous step's
        modulation signal.
        """
        if self._prev_modulation is None or self._cached_residual is None:
            self._prev_modulation = modulation
            self.misses += 1
            return True

        prev = self._prev_modulation
        rel_change = ((modulation - prev).abs().mean() / (prev.abs().mean() + 1e-8)).item()
        self._accumulated += rel_change
        self._prev_modulation = modulation

        if self._accumulated < self.rel_l1_threshold:
            self.hits += 1
            return False
        self._accumulated = 0.0
        self.misses += 1
        return True

    def store(self, residual: "torch.Tensor") -> None:
        r"""Cache the transformer residual from a computed step."""
        self._cached_residual = residual

    @property
    def cached_residual(self) -> "torch.Tensor | None":
        return self._cached_residual

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset(self) -> None:
        r"""Clear all state for a fresh denoise loop."""
        self._prev_modulation = None
        self._cached_residual = None
        self._accumulated = 0.0
        self.hits = 0
        self.misses = 0
