r"""AMD ROCm platform."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from worldkernels.platforms.interface import Platform

if TYPE_CHECKING:
    import torch

__all__ = ["RocmPlatform"]


class RocmPlatform(Platform):
    name: ClassVar[str] = "rocm"
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available() and bool(getattr(torch.version, "hip", None))

    def default_dtype(self) -> "torch.dtype":
        import torch

        return torch.bfloat16

    def default_attention_backend(self) -> str:
        return "sdpa"

    def device_count(self) -> int:
        import torch

        return torch.cuda.device_count()

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()
