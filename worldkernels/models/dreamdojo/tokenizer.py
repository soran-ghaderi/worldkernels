r"""Wan2.1 video tokenizer for DreamDojo, matching the reference bf16 numerics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["VideoTokenizer"]

WAN_VAE_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class VideoTokenizer:
    r"""Deterministic Wan2.1 VAE encode/decode with the standard latent normalization.

    Runs the VAE fully in bfloat16 (weights and activations) as the reference
    stack does. ``encode`` takes video in \([-1, 1]\), returns the posterior
    mean normalized as \((\mu - m) \cdot s^{-1}\) with the canonical Wan
    16-channel statistics; ``decode`` inverts the normalization and returns
    video in \([-1, 1]\). Input dtypes are preserved on output.
    """

    spatial_compression_factor = 8
    temporal_compression_factor = 4
    latent_channels = 16

    def __init__(self, repo: str = WAN_VAE_REPO) -> None:
        self.repo = repo
        self._vae = None
        self._mean: torch.Tensor | None = None
        self._inv_std: torch.Tensor | None = None

    def load(self, device: str | "torch.device") -> "VideoTokenizer":
        import torch
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(
            self.repo, subfolder="vae", torch_dtype=torch.bfloat16
        )
        vae = vae.to(device).eval().requires_grad_(False)
        self._vae = vae
        self._mean = torch.tensor(
            vae.config.latents_mean, dtype=torch.bfloat16, device=device
        ).view(1, self.latent_channels, 1, 1, 1)
        self._inv_std = 1.0 / torch.tensor(
            vae.config.latents_std, dtype=torch.bfloat16, device=device
        ).view(1, self.latent_channels, 1, 1, 1)
        return self

    def encode(self, video_B_C_T_H_W: "torch.Tensor") -> "torch.Tensor":
        import torch

        assert self._vae is not None, "call load() first"
        in_dtype = video_B_C_T_H_W.dtype
        with torch.no_grad():
            mu = self._vae.encode(video_B_C_T_H_W.to(torch.bfloat16)).latent_dist.mode()
            mu = (mu - self._mean) * self._inv_std
        return mu.to(in_dtype)

    def decode(self, latent_B_C_T_H_W: "torch.Tensor") -> "torch.Tensor":
        import torch

        assert self._vae is not None, "call load() first"
        in_dtype = latent_B_C_T_H_W.dtype
        with torch.no_grad():
            z = latent_B_C_T_H_W.to(torch.bfloat16) / self._inv_std + self._mean
            video = self._vae.decode(z).sample
        return video.to(in_dtype)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // self.temporal_compression_factor

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * self.temporal_compression_factor + 1
