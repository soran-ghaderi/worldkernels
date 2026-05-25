r"""Video-generator interface.

A `VideoGenerator` is a *pipeline*: it produces a video clip from
conditioning in one forward pass and holds no per-session state. It is the
substrate a `GeneratorWorld` wraps to present
a one-shot generator as an interactive world model.

Conditioning objects are generator-specific (e.g. ``WanLatent``); they are
opaque to the world layer, which only moves them between generator calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["GenerationResult", "VideoGenerator"]


@dataclass
class GenerationResult:
    r"""Output of one generator forward pass.

    Args:
        latent: Denoised latent in the model's latent space.
        video: Decoded clip of shape ``(1, 3, frames, H, W)`` in ``[-1, 1]``.
    """

    latent: "torch.Tensor"
    video: "torch.Tensor"


class VideoGenerator(ABC):
    r"""A stateless video-generation pipeline.

    Subclasses compose components (VAE, transformer, text encoder) and own a
    single forward pass. The conditioning type is the subclass's own; the
    world layer treats it opaquely.
    """

    @abstractmethod
    def load(self, device: str, dtype: "torch.dtype") -> None:
        r"""Load weights and move components to ``device``."""

    @abstractmethod
    def encode_prompt(self, prompt: str) -> "torch.Tensor":
        r"""Encode a text prompt into a conditioning embedding."""

    @abstractmethod
    def initial_conditioning(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        image: Any | None,
        height: int,
        width: int,
        frames_per_step: int,
        seed: int,
    ) -> Any:
        r"""Build the conditioning for the first generation step."""

    @abstractmethod
    def apply_prompt(self, conditioning: Any, prompt_embeds: "torch.Tensor") -> Any:
        r"""Return conditioning with its text embedding replaced (mid-rollout action)."""

    @abstractmethod
    def generate(
        self,
        conditioning: Any,
        *,
        num_steps: int,
        guidance: float,
        num_frames: int,
        seed: int,
    ) -> GenerationResult:
        r"""Run one generation forward pass."""

    @abstractmethod
    def advance(
        self,
        conditioning: Any,
        result: GenerationResult,
        next_image: "torch.Tensor",
    ) -> Any:
        r"""Build the next-step conditioning from a result and a rollout frame."""

    @abstractmethod
    def decode(self, latent: "torch.Tensor") -> "torch.Tensor":
        r"""VAE-decode a latent to a video tensor in ``[-1, 1]``."""

    @abstractmethod
    def profile_vram(self, *, height: int, width: int, num_frames: int) -> float:
        r"""Estimate per-session VRAM (MB)."""
