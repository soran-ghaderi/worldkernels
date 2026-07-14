r"""CPU tests for timestep, patch, and rotary embeddings."""

from __future__ import annotations

import math

import torch

from worldkernels.models.components.embeddings import (
    PatchEmbed,
    SinusoidalTimesteps,
    TimestepEmbedding,
    VideoRoPE3D,
    apply_rotary_emb,
)


def test_sinusoidal_cos_sin_order():
    emb = SinusoidalTimesteps(8)
    t = torch.tensor([[0.0]])
    out = emb(t)
    assert out.shape == (1, 1, 8)
    torch.testing.assert_close(out[0, 0, :4], torch.ones(4))
    torch.testing.assert_close(out[0, 0, 4:], torch.zeros(4))


def test_sinusoidal_frequency_ladder():
    emb = SinusoidalTimesteps(6)
    t = torch.tensor([[2.5]])
    out = emb(t)
    freqs = torch.exp(-math.log(10000) * torch.arange(3).float() / 3)
    torch.testing.assert_close(out[0, 0, :3], torch.cos(2.5 * freqs))
    torch.testing.assert_close(out[0, 0, 3:], torch.sin(2.5 * freqs))


def test_timestep_embedding_adaln_lora_contract():
    emb = TimestepEmbedding(16, 16, use_adaln_lora=True)
    assert emb.linear_1.bias is None
    sample = torch.randn(2, 3, 16)
    out, lora = emb(sample)
    assert out is sample
    assert lora is not None and lora.shape == (2, 3, 48)

    emb_plain = TimestepEmbedding(16, 16, use_adaln_lora=False)
    assert emb_plain.linear_1.bias is not None
    out, lora = emb_plain(sample)
    assert out.shape == (2, 3, 16) and lora is None


def test_patch_embed_shape_and_keys():
    pe = PatchEmbed(spatial_patch_size=2, temporal_patch_size=1, in_channels=18, out_channels=32)
    assert set(pe.state_dict()) == {"proj.1.weight"}
    x = torch.randn(1, 18, 3, 8, 12)
    out = pe(x)
    assert out.shape == (1, 3, 4, 6, 32)


def test_rope_table_layout():
    rope = VideoRoPE3D(
        head_dim=24,
        len_h=8,
        len_w=8,
        len_t=8,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
    )
    table = rope(2, 3, 4)
    assert table.shape == (2 * 3 * 4, 1, 1, 24)
    assert table.dtype == torch.float32
    torch.testing.assert_close(table[..., :12], table[..., 12:], rtol=0, atol=0)
    assert torch.all(table[0] == 0)


def test_rope_forward_frame_matches_full_grid():
    rope = VideoRoPE3D(head_dim=24, len_h=8, len_w=8, len_t=8)
    full = rope(5, 3, 4)
    for idx in (0, 2, 4):
        frame = rope.forward_frame(idx, 3, 4)
        torch.testing.assert_close(frame, full[idx * 12 : (idx + 1) * 12], rtol=0, atol=0)


def test_apply_rotary_emb_rotation():
    rope = VideoRoPE3D(head_dim=24, len_h=4, len_w=4, len_t=4)
    table = rope(2, 2, 2)
    x = torch.randn(1, 8, 2, 24)
    out = apply_rotary_emb(x, table)
    assert out.shape == x.shape
    x1, x2 = x[..., :12], x[..., 12:]
    o1, o2 = out[..., :12], out[..., 12:]
    torch.testing.assert_close(o1 * o1 + o2 * o2, x1 * x1 + x2 * x2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(apply_rotary_emb(x, torch.zeros_like(table)), x)
