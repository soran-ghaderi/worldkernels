r"""Hardware-platform abstraction.

A `Platform` answers device-specific questions — default precision,
default attention backend, device count, synchronization — so the rest of the
engine never branches on CUDA / ROCm / CPU directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch

__all__ = ["Platform"]


class Platform(ABC):
    r"""Device-family capabilities and defaults.

    Subclasses declare ``name`` / ``device_type`` and implement the probes.
    """

    name: ClassVar[str]
    device_type: ClassVar[str]

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        r"""Whether this platform is usable in the current process."""

    @abstractmethod
    def default_dtype(self) -> "torch.dtype":
        r"""Preferred compute precision for this device family."""

    @abstractmethod
    def default_attention_backend(self) -> str:
        r"""Name of the best available attention backend."""

    @abstractmethod
    def device_count(self) -> int:
        r"""Number of visible devices of this family."""

    @abstractmethod
    def synchronize(self) -> None:
        r"""Block until all queued device work completes."""
