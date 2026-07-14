r"""Timestep, patch, and rotary position embeddings for native video DiTs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = [
    "SinusoidalTimesteps",
    "TimestepEmbedding",
    "PatchEmbed",
    "VideoRoPE3D",
    "apply_rotary_emb",
]


class SinusoidalTimesteps(nn.Module):
    r"""Sinusoidal timestep features, ``[cos | sin]`` ordered, computed in float32."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"expected 2D input, got {timesteps_B_T.ndim}"
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        emb = timesteps[:, None] * torch.exp(exponent / half_dim)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(
            emb.to(dtype=in_dtype),
            "(b t) d -> b t d",
            b=timesteps_B_T.shape[0],
            t=timesteps_B_T.shape[1],
        )


class TimestepEmbedding(nn.Module):
    r"""Two-layer timestep MLP; in adaLN-LoRA mode the MLP output is the LoRA stream.

    Returns:
        ``(emb_B_T_D, adaln_lora_B_T_3D)``. With adaLN-LoRA the embedding
        passed to the blocks is the raw sinusoidal input and the MLP produces
        the shared 3D LoRA modulation; without it the MLP output is the
        embedding and the LoRA stream is ``None``.
    """

    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        out_dim = 3 * out_features if use_adaln_lora else out_features
        self.linear_2 = nn.Linear(out_features, out_dim, bias=False)

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        if self.use_adaln_lora:
            return sample, emb
        return emb, None


class PatchEmbed(nn.Module):
    r"""Space-time patchify followed by a linear projection to model width."""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                bias=False,
            ),
        )

    def forward(self, x_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        assert x_B_C_T_H_W.dim() == 5
        _, _, T, H, W = x_B_C_T_H_W.shape
        assert (
            H % self.spatial_patch_size == 0
            and W % self.spatial_patch_size == 0
            and T % self.temporal_patch_size == 0
        ), f"input {(T, H, W)} not divisible by patch size"
        return self.proj(x_B_C_T_H_W)


class VideoRoPE3D(nn.Module):
    r"""3D rotary position embedding over (t, h, w) with NTK-aware extrapolation.

    The head dimension is split ``dim_h = dim_w = 2 * (d // 6)`` and
    ``dim_t = d - 2 dim_h``; each axis gets its own frequency ladder with base
    \(\theta = 10^4 \cdot r^{d_a/(d_a-2)}\) for extrapolation ratio \(r\).
    Emits a duplicated-frequency table of shape ``(T*H*W, 1, 1, d)`` in float32
    for the half-rotation form of `apply_rotary_emb`.
    """

    seq: torch.Tensor
    dim_spatial_range: torch.Tensor
    dim_temporal_range: torch.Tensor

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t

        dim_h = head_dim // 6 * 2
        dim_w = dim_h
        dim_t = head_dim - 2 * dim_h
        assert head_dim == dim_h + dim_w + dim_t, (
            f"bad head_dim: {head_dim} != {dim_h} + {dim_w} + {dim_t}"
        )

        self.register_buffer(
            "seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float), persistent=True
        )
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def _freqs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor
        h_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
        w_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
        t_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())
        return t_freqs, h_freqs, w_freqs

    def forward(self, T: int, H: int, W: int) -> torch.Tensor:
        r"""Full-grid table for a ``(T, H, W)`` token volume, shape ``(T*H*W, 1, 1, d)``."""
        assert H <= self.max_h and W <= self.max_w, (
            f"({H}, {W}) exceeds maximum ({self.max_h}, {self.max_w})"
        )
        t_freqs, h_freqs, w_freqs = self._freqs()
        half_emb_t = torch.outer(self.seq[:T], t_freqs)
        half_emb_h = torch.outer(self.seq[:H], h_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_freqs)
        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    def forward_frame(self, frame_idx: int, H: int, W: int) -> torch.Tensor:
        r"""Table for the single latent frame at absolute index ``frame_idx``."""
        assert frame_idx < self.max_t, f"frame_idx {frame_idx} exceeds maximum {self.max_t}"
        t_freqs, h_freqs, w_freqs = self._freqs()
        half_emb_t = torch.outer(self.seq[frame_idx : frame_idx + 1], t_freqs)
        half_emb_h = torch.outer(self.seq[:H], h_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_freqs)
        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=1, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=1, h=H),
            ]
            * 2,
            dim=-1,
        )
        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


def apply_rotary_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    r"""Half-rotation rotary embedding on ``(B, S, H, D)`` tensors.

    ``freqs`` is a duplicated-frequency table ``(S, 1, 1, rot_dim)``; cos/sin
    are taken in the input dtype and the rotation uses the NeoX half split.
    """
    if freqs.ndim == 4 and freqs.shape[1] == 1:
        freqs = freqs.transpose(0, 1)
    cos_ = freqs.cos().to(t.dtype)
    sin_ = freqs.sin().to(t.dtype)
    rot_dim = cos_.shape[-1]
    half = rot_dim // 2
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    x1, x2 = t_rot[..., :half], t_rot[..., half:]
    cos_h = cos_[..., :half]
    sin_h = sin_[..., :half]
    o1 = x1 * cos_h - x2 * sin_h
    o2 = x2 * cos_h + x1 * sin_h
    return torch.cat([o1, o2, t_pass], dim=-1)
