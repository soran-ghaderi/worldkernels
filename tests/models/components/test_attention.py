r"""CPU tests for Attention and the frame-granular KV cache."""

from __future__ import annotations

import torch

from worldkernels.models.components.attention import Attention, FrameKVCache


def test_self_and_cross_shapes():
    self_attn = Attention(32, None, n_heads=4, head_dim=8)
    cross_attn = Attention(32, 16, n_heads=4, head_dim=8)
    x = torch.randn(2, 10, 32)
    ctx = torch.randn(2, 5, 16)
    assert self_attn(x).shape == (2, 10, 32)
    assert cross_attn(x, ctx).shape == (2, 10, 32)


def test_rope_only_in_self_attention():
    torch.manual_seed(0)
    cross_attn = Attention(32, 16, n_heads=4, head_dim=8)
    x = torch.randn(1, 4, 32)
    ctx = torch.randn(1, 4, 16)
    rope = torch.randn(4, 1, 1, 8)
    torch.testing.assert_close(cross_attn(x, ctx, rope_emb=rope), cross_attn(x, ctx))


def test_frame_kv_cache_rolling_window():
    cache = FrameKVCache.allocate(
        max_frames=6,
        batch_size=2,
        tokens_per_frame=3,
        num_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        window=2,
    )
    frames = [torch.randn(2, 3, 2, 4) for _ in range(5)]
    for idx, f in enumerate(frames):
        cache.store(idx, f, f + 1)

    cur_k = torch.randn(2, 3, 2, 4)
    cur_v = torch.randn(2, 3, 2, 4)

    k0, v0 = cache.gather(0, cur_k, cur_v)
    torch.testing.assert_close(k0, cur_k)
    torch.testing.assert_close(v0, cur_v)

    k4, v4 = cache.gather(4, cur_k, cur_v)
    expected_k = torch.cat([frames[2], frames[3], cur_k], dim=1)
    expected_v = torch.cat([frames[2] + 1, frames[3] + 1, cur_v], dim=1)
    torch.testing.assert_close(k4, expected_k)
    torch.testing.assert_close(v4, expected_v)

    cache.window = None
    k4, _ = cache.gather(4, cur_k, cur_v)
    assert k4.shape[1] == 4 * 3 + 3


def test_attention_with_cache_matches_explicit_concat():
    r"""Per-frame cached attention equals attention over explicitly stacked context."""
    torch.manual_seed(1)
    attn = Attention(16, None, n_heads=2, head_dim=8)
    frames = [torch.randn(1, 4, 16) for _ in range(3)]

    cache = FrameKVCache.allocate(
        max_frames=4,
        batch_size=1,
        tokens_per_frame=4,
        num_heads=2,
        head_dim=8,
        device="cpu",
        dtype=torch.float32,
    )
    outs = []
    for idx, f in enumerate(frames):
        outs.append(attn(f, kv_cache=cache, frame_idx=idx, store_kv=True))

    import torch.nn.functional as F
    from einops import rearrange

    def manual(query_frame: torch.Tensor, context_frames: list[torch.Tensor]) -> torch.Tensor:
        q = rearrange(
            attn.q_norm(rearrange(attn.q_proj(query_frame), "b s (h d) -> b s h d", h=2)),
            "b s h d -> b h s d",
        )
        ctx = torch.cat(context_frames, dim=1)
        k = rearrange(
            attn.k_norm(rearrange(attn.k_proj(ctx), "b s (h d) -> b s h d", h=2)),
            "b s h d -> b h s d",
        )
        v = rearrange(
            rearrange(attn.v_proj(ctx), "b s (h d) -> b s h d", h=2), "b s h d -> b h s d"
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return attn.output_proj(rearrange(out, "b h s d -> b s (h d)"))

    for idx in range(3):
        torch.testing.assert_close(outs[idx], manual(frames[idx], frames[: idx + 1]))
