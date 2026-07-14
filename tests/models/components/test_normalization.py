r"""CPU tests for RMSNorm float32-compute semantics."""

from __future__ import annotations

import torch

from worldkernels.models.components.normalization import RMSNorm


def test_matches_reference_formula():
    norm = RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.mul_(2.0)
    x = torch.randn(3, 5, 8, dtype=torch.float64).float()
    expected = (
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ) * norm.weight
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)


def test_bf16_computes_in_float32_then_casts():
    norm = RMSNorm(64)
    x = torch.randn(2, 64).bfloat16()
    out = norm(x)
    assert out.dtype == torch.float32
    inner = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, inner * norm.weight, rtol=0, atol=0)
