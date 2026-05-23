r"""CPU fallback platform."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from worldkernels.platforms.interface import Platform

if TYPE_CHECKING:
    import torch

__all__ = ["CpuPlatform"]


class CpuPlatform(Platform):
    name: ClassVar[str] = "cpu"
    device_type: ClassVar[str] = "cpu"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def default_dtype(self) -> "torch.dtype":
        import torch

        return torch.float32

    def default_attention_backend(self) -> str:
        return "sdpa"

    def device_count(self) -> int:
        return 1

    def synchronize(self) -> None:
        return None
