r"""PyTorch scaled-dot-product-attention backend.

The universal fallback: works on CPU and every GPU, no extra dependency.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from worldkernels.runtime.attention.backend import AttentionBackend

__all__ = ["SDPABackend"]


class SDPABackend(AttentionBackend):
    r"""Attention via ``torch.nn.functional.scaled_dot_product_attention``."""

    name: ClassVar[str] = "sdpa"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
        return out.transpose(1, 2)
