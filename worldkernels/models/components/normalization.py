r"""Normalization layers for native video-diffusion models."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    r"""Root-mean-square norm computed in float32, cast back to the input dtype.

    $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\overline{x^2} + \epsilon}} \cdot w$$
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
