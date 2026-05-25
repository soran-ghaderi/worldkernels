r"""Three-tier session-state offloading.

Paused or cold session state can spill from VRAM to pinned host memory and,
later, to NVMe. Host transfers use pinned buffers and a dedicated copy stream
so they overlap compute. The NVMe tier is the declared seam; host transfers
are implemented.
"""

from __future__ import annotations

from enum import Enum

import torch

__all__ = ["MemoryTier", "Offloader"]


class MemoryTier(str, Enum):
    r"""Where session state currently resides."""

    GPU = "gpu"
    HOST = "host"
    NVME = "nvme"


class Offloader:
    r"""Moves tensors between VRAM and pinned host memory.

    Args:
        device: The GPU device tensors are restored to.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self._pinning = device.startswith("cuda") and torch.cuda.is_available()
        self._stream = torch.cuda.Stream() if self._pinning else None

    @staticmethod
    def tier_of(tensor: torch.Tensor) -> MemoryTier:
        return MemoryTier.GPU if tensor.device.type == "cuda" else MemoryTier.HOST

    def to_host(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Copy ``tensor`` to (pinned) host memory."""
        if tensor.device.type != "cuda":
            return tensor
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=self._pinning)
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                host.copy_(tensor, non_blocking=True)
            self._stream.synchronize()
        else:
            host.copy_(tensor)
        return host

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Restore ``tensor`` to the GPU device."""
        if tensor.device.type == "cuda":
            return tensor
        return tensor.to(self.device, non_blocking=self._pinning)

    def offload(self, tensor: torch.Tensor, tier: MemoryTier) -> torch.Tensor:
        r"""Move ``tensor`` to ``tier``."""
        if tier is MemoryTier.GPU:
            return self.to_device(tensor)
        if tier is MemoryTier.HOST:
            return self.to_host(tensor)
        raise NotImplementedError("NVMe offload tier is not implemented yet")
