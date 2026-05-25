r"""Pluggable attention backend interface.

A backend computes scaled dot-product attention for query/key/value tensors in
the ``(batch, seq, heads, head_dim)`` layout — the layout diffusers and
flash-attn use. Backends are selected per hardware by
`select_attention_backend()` and
read from the active `ForwardContext`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch

__all__ = ["AttentionMetadata", "AttentionBackend"]


@dataclass
class AttentionMetadata:
    r"""Per-forward attention state threaded via the forward context.

    Args:
        is_causal: Whether the attention mask is causal (streaming worlds).
        scale: Softmax scale; ``None`` uses \(1/\sqrt{d}\).
    """

    is_causal: bool = False
    scale: float | None = None


class AttentionBackend(ABC):
    r"""Computes attention over ``(batch, seq, heads, head_dim)`` tensors."""

    name: ClassVar[str]

    @abstractmethod
    def forward(
        self,
        query: "torch.Tensor",
        key: "torch.Tensor",
        value: "torch.Tensor",
        *,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> "torch.Tensor":
        r"""Return the attention output in ``(batch, seq, heads, head_dim)`` layout."""
