r"""NVIDIA CUDA platform."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from worldkernels.platforms.interface import Platform

if TYPE_CHECKING:
    import torch

__all__ = ["CudaPlatform"]


class CudaPlatform(Platform):
    name: ClassVar[str] = "cuda"
    device_type: ClassVar[str] = "cuda"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available() and not getattr(torch.version, "hip", None)

    def default_dtype(self) -> "torch.dtype":
        import torch

        major = torch.cuda.get_device_capability()[0]
        return torch.bfloat16 if major >= 8 else torch.float16

    def default_attention_backend(self) -> str:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            return "sdpa"
        return "flash"

    def device_count(self) -> int:
        import torch

        return torch.cuda.device_count()

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()
