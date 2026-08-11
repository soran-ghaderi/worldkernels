r"""Latent action encoder: continuous proxy actions from consecutive frame pairs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

if TYPE_CHECKING:
    pass

__all__ = ["LatentActionEncoder", "LAM_INPUT_HEIGHT", "LAM_INPUT_WIDTH", "LATENT_ACTION_DIM"]

LAM_INPUT_HEIGHT = 240
LAM_INPUT_WIDTH = 320
LATENT_ACTION_DIM = 32
LATENT_ACTION_SLICE = slice(352, 384)


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


class _Rotary(nn.Module):
    r"""Interleaved-pair rotary embedding over the temporal token axis."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def rotate(self, t: torch.Tensor) -> torch.Tensor:
        seq = torch.arange(t.shape[-2], device=t.device, dtype=t.dtype)
        freqs = torch.einsum("n,f->nf", seq.to(self.freqs.dtype), self.freqs)
        freqs = freqs.repeat_interleave(2, dim=-1)
        return t * freqs.cos() + _rotate_interleaved(t) * freqs.sin()


class _SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, rot_emb: bool = False) -> None:
        super().__init__()
        head_dim = model_dim // num_heads
        self.scale = head_dim**-0.5
        self.heads = num_heads
        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Dropout(0.0))
        if rot_emb:
            self.rotary_embedding = _Rotary(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        if hasattr(self, "rotary_embedding"):
            q = self.rotary_embedding.rotate(q).contiguous()
            k = self.rotary_embedding.rotate(k).contiguous()
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        return self.to_out(out)


class _SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.spatial_attn = _SelfAttention(model_dim, num_heads)
        self.temporal_attn = _SelfAttention(model_dim, num_heads, rot_emb=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_len, s_len = x.shape[1:3]
        x = rearrange(x, "b t s e -> (b t) s e")
        x = x + self.spatial_attn(self.norm1(x))
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        x = rearrange(x, "b t s e -> (b s) t e")
        x = x + self.temporal_attn(self.norm2(x))
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        return x + self.ffn(self.norm3(x))


class _SpatioTemporalTransformer(nn.Module):
    pe: torch.Tensor

    def __init__(
        self, in_dim: int, model_dim: int, out_dim: int, num_blocks: int, num_heads: int
    ) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, model_dim), nn.LayerNorm(model_dim)
        )
        pe = torch.zeros(5000, model_dim)
        position = torch.arange(0, 5000).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
        self.transformer_blocks = nn.ModuleList(
            _SpatioTemporalBlock(model_dim, num_heads) for _ in range(num_blocks)
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x)
        x = x + self.pe[: x.shape[2]]
        for block in self.transformer_blocks:
            x = block(x)
        return self.out(x)


class LatentActionEncoder(nn.Module):
    r"""Encoder half of the DreamDojo latent action VAE (frame pair → 32-dim action).

    A learned action-prompt token is prepended to each frame's 16x16 patch
    tokens; a 24-block spatiotemporal transformer attends across the pair and
    the second frame's prompt token is projected to the posterior mean, the
    continuous latent action occupying slice ``[352:384)`` of the universal
    action vector.
    """

    def __init__(
        self,
        *,
        in_dim: int = 3,
        model_dim: int = 1024,
        latent_dim: int = LATENT_ACTION_DIM,
        patch_size: int = 16,
        enc_blocks: int = 24,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        patch_token_dim = in_dim * patch_size**2
        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        self.encoder = _SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str, device: "str | torch.device" = "cuda"
    ) -> "LatentActionEncoder":
        r"""Load the encoder subset of a LAM Lightning checkpoint (``lam.`` prefix)."""
        encoder = cls()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        own = encoder.state_dict()
        subset = {
            k.removeprefix("lam."): v for k, v in state.items() if k.removeprefix("lam.") in own
        }
        missing = set(own) - set(subset)
        assert not missing, f"LAM checkpoint missing encoder keys: {sorted(missing)[:5]}"
        encoder.load_state_dict(subset, strict=True)
        return encoder.to(device).eval().requires_grad_(False)

    def encode_pairs(self, frame_pairs: torch.Tensor) -> torch.Tensor:
        r"""Posterior means for ``(P, 2, H, W, 3)`` frame pairs in \([0, 1]\) → ``(P, 32)``."""
        P, T, H, W, _ = frame_pairs.shape
        assert T == 2
        size = self.patch_size
        frame_pairs = frame_pairs[:, :, : H - (H % size), : W - (W % size), :]
        patches = rearrange(
            frame_pairs, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size
        )
        padded = torch.cat([self.action_prompt.expand(P, T, -1, -1), patches], dim=2)
        z = self.encoder(padded)[:, 1:, 0].reshape(P, self.model_dim)
        z_mu, _ = torch.chunk(self.fc(z), 2, dim=1)
        return z_mu

    def encode_video(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        r"""Latent actions for consecutive pairs of ``(T, 3, H, W)`` uint8 frames → ``(T-1, 32)``.

        Frames are bilinearly resized to the LAM's native 240x320 as in the
        reference data pipeline.
        """
        import torch.nn.functional as F

        frames = frames_uint8.float() / 255.0
        if frames.shape[-2:] != (LAM_INPUT_HEIGHT, LAM_INPUT_WIDTH):
            frames = F.interpolate(
                frames, size=(LAM_INPUT_HEIGHT, LAM_INPUT_WIDTH), mode="bilinear"
            )
        frames = frames.permute(0, 2, 3, 1)
        pairs = torch.stack([frames[:-1], frames[1:]], dim=1)
        device = next(self.parameters()).device
        with torch.no_grad():
            return self.encode_pairs(pairs.to(device))
