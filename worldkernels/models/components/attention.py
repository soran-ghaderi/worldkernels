r"""Multi-head attention with QK RMSNorm, RoPE, and a frame-granular KV cache."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from worldkernels.models.components.embeddings import apply_rotary_emb
from worldkernels.models.components.normalization import RMSNorm

__all__ = ["Attention", "FrameKVCache"]


class FrameKVCache:
    r"""Rolling per-frame K/V cache over preallocated tensors.

    Entries are stored per latent frame with shape ``(B, HW, heads, head_dim)``
    inside ``(max_frames, B, HW, heads, head_dim)`` buffers. Reads gather a
    rolling window of at most ``window`` frames preceding the current index and
    concatenate the current frame's K/V.
    """

    def __init__(self, k: torch.Tensor, v: torch.Tensor, window: int | None = None) -> None:
        self.k = k
        self.v = v
        self.window = window

    @classmethod
    def allocate(
        cls,
        *,
        max_frames: int,
        batch_size: int,
        tokens_per_frame: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
        window: int | None = None,
    ) -> "FrameKVCache":
        shape = (max_frames, batch_size, tokens_per_frame, num_heads, head_dim)
        return cls(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
            window=window,
        )

    def store(self, frame_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k[frame_idx] = k
        self.v[frame_idx] = v

    def gather(
        self, frame_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Concatenate the rolling window of cached frames with the current K/V."""
        if frame_idx == 0:
            return k, v
        start = max(0, frame_idx - self.window) if self.window is not None else 0
        B = k.shape[0]
        cached_k = self.k[start:frame_idx].transpose(0, 1).reshape(B, -1, *k.shape[-2:])
        cached_v = self.v[start:frame_idx].transpose(0, 1).reshape(B, -1, *v.shape[-2:])
        return torch.cat([cached_k, k], dim=1), torch.cat([cached_v, v], dim=1)


class Attention(nn.Module):
    r"""Self- or cross-attention with per-head QK RMSNorm and optional RoPE.

    Self-attention mode (``context_dim=None``) applies rotary embeddings to
    Q/K, in float32 when ``use_wan_fp32_strategy`` is set, and supports the
    frame-granular KV cache for causal streaming. The attention op routes
    through the runtime attention-backend selector.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        n_heads: int = 8,
        head_dim: int = 64,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        kv_cache: FrameKVCache | None = None,
        frame_idx: int = 0,
        store_kv: bool = False,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        kv_input = x if context is None else context
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        q, k, v = (
            rearrange(t, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim) for t in (q, k, v)
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.is_selfattn and rope_emb is not None:
            original_dtype = q.dtype
            if self.use_wan_fp32_strategy:
                q = q.to(torch.float32)
                k = k.to(torch.float32)
            q = apply_rotary_emb(q, rope_emb)
            k = apply_rotary_emb(k, rope_emb)
            if self.use_wan_fp32_strategy:
                q = q.to(original_dtype)
                k = k.to(original_dtype)

        if kv_cache is not None:
            if store_kv:
                kv_cache.store(frame_idx, k, v)
            k, v = kv_cache.gather(frame_idx, k, v)

        from worldkernels.runtime.attention.selector import select_attention_backend

        out = select_attention_backend().forward(q, k, v)
        return self.output_proj(rearrange(out, "b s h d -> b s (h d)"))
