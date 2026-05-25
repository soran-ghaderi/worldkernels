r"""Pre-allocated tensor buffer pool.

Hot-path code (the denoise loop, frame buffers) must never call
``torch.zeros`` / ``torch.empty``. The pool hands out reusable buffers keyed by
``(shape, dtype)`` and reclaims them on release.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["LatentPool"]


class LatentPool:
    r"""Buffer pool that recycles tensors by shape and dtype.

    Args:
        device: Device buffers are allocated on.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._free: dict[tuple, list[torch.Tensor]] = {}
        self._in_use = 0

    @property
    def num_in_use(self) -> int:
        return self._in_use

    @property
    def num_pooled(self) -> int:
        return sum(len(bufs) for bufs in self._free.values())

    def acquire(self, shape: "Sequence[int]", dtype: torch.dtype) -> torch.Tensor:
        r"""Return a buffer of ``(shape, dtype)``, reusing a pooled one if available."""
        key = (tuple(shape), dtype)
        pool = self._free.get(key)
        self._in_use += 1
        if pool:
            return pool.pop()
        return torch.empty(tuple(shape), device=self.device, dtype=dtype)

    def release(self, tensor: torch.Tensor) -> None:
        r"""Return ``tensor`` to the pool for reuse."""
        key = (tuple(tensor.shape), tensor.dtype)
        self._free.setdefault(key, []).append(tensor)
        self._in_use = max(0, self._in_use - 1)

    def clear(self) -> None:
        r"""Drop all pooled buffers."""
        self._free.clear()
