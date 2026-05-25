r"""FlashAttention backend.

Fast, low-memory attention on Ampere+ GPUs. Requires the optional
``flash-attn`` package; absent it, the backend stays importable and raises an
actionable error only when used.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from worldkernels.runtime.attention.backend import AttentionBackend
from worldkernels.utils import optional_import

__all__ = ["FlashAttentionBackend"]

_flash_attn = optional_import("flash_attn", "flash-attn")


class FlashAttentionBackend(AttentionBackend):
    r"""Attention via ``flash_attn.flash_attn_func`` (Ampere+ GPUs)."""

    name: ClassVar[str] = "flash"

    @staticmethod
    def is_available() -> bool:
        from worldkernels.utils.import_utils import PlaceholderModule

        return not isinstance(_flash_attn, PlaceholderModule)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        return _flash_attn.flash_attn_func(query, key, value, causal=is_causal, softmax_scale=scale)
