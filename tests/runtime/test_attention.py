r"""Tests for the pluggable attention backends (CPU-safe)."""

from __future__ import annotations

import torch

from worldkernels.runtime.attention import (
    AttentionMetadata,
    SDPABackend,
    select_attention_backend,
)


def _qkv(batch=2, seq=4, heads=3, dim=8):
    g = torch.Generator().manual_seed(0)
    shape = (batch, seq, heads, dim)
    return (
        torch.rand(shape, generator=g),
        torch.rand(shape, generator=g),
        torch.rand(shape, generator=g),
    )


class TestSDPABackend:
    def test_output_shape_preserved(self):
        q, k, v = _qkv()
        out = SDPABackend().forward(q, k, v)
        assert out.shape == q.shape

    def test_causal_matches_reference(self):
        q, k, v = _qkv(batch=1, heads=1)
        out = SDPABackend().forward(q, k, v, is_causal=True)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
        assert torch.allclose(out, ref, atol=1e-6)


class TestSelector:
    def test_sdpa_explicit(self):
        assert isinstance(select_attention_backend("sdpa"), SDPABackend)

    def test_flash_falls_back_to_sdpa_when_unavailable(self):
        backend = select_attention_backend("flash")
        assert backend.name in ("flash", "sdpa")

    def test_default_resolves(self):
        assert select_attention_backend() is not None


def test_attention_metadata_defaults():
    meta = AttentionMetadata()
    assert meta.is_causal is False
    assert meta.scale is None
