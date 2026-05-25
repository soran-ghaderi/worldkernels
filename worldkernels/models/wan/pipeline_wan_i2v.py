r"""Wan image-to-video generator (diffusers-backed).

A `VideoGenerator`: loads the diffusers Wan
components, swaps in the owned `FlowUniPCMultistepScheduler`, and
produces a video clip per forward pass. Composed by
`GeneratorWorld` to expose Wan as a world.

The VAE is held in float32 (Wan's VAE is numerically sensitive); the
transformer and text encoder use the requested dtype.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.models.base import GenerationResult, VideoGenerator

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


class WanLatent:
    r"""Conditioning for a Wan image-to-video rollout.

    Holds the conditioning frame and prompt embeddings; ``latent`` / ``video``
    carry the most recent generation output so no recompute is needed.
    """

    __slots__ = ("last_frame", "prompt_embeds", "negative_prompt_embeds", "latent", "video")

    def __init__(
        self,
        last_frame: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
    ) -> None:
        self.last_frame = last_frame
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.latent = latent
        self.video = video

    def clone(self) -> "WanLatent":
        return WanLatent(
            last_frame=self.last_frame.clone(),
            prompt_embeds=self.prompt_embeds.clone(),
            negative_prompt_embeds=(
                None
                if self.negative_prompt_embeds is None
                else self.negative_prompt_embeds.clone()
            ),
            latent=None if self.latent is None else self.latent.clone(),
            video=None if self.video is None else self.video.clone(),
        )

    def to(self, device: Any) -> "WanLatent":
        return WanLatent(
            last_frame=self.last_frame.to(device),
            prompt_embeds=self.prompt_embeds.to(device),
            negative_prompt_embeds=(
                None
                if self.negative_prompt_embeds is None
                else self.negative_prompt_embeds.to(device)
            ),
            latent=None if self.latent is None else self.latent.to(device),
            video=None if self.video is None else self.video.to(device),
        )

    @property
    def nelement(self) -> int:
        n = self.last_frame.nelement() + self.prompt_embeds.nelement()
        for t in (self.negative_prompt_embeds, self.latent, self.video):
            if t is not None:
                n += t.nelement()
        return n

    @property
    def element_size(self) -> int:
        return self.last_frame.element_size()


class WanI2VPipeline(VideoGenerator):
    r"""Wan image-to-video generator driving diffusers components.

    Args:
        repo: diffusers-format Wan repo (e.g. ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``).
        pipeline_class: diffusers pipeline class name to load.
        flow_shift: Flow-matching schedule shift for the scheduler.
    """

    def __init__(
        self,
        *,
        repo: str,
        pipeline_class: str = "WanImageToVideoPipeline",
        flow_shift: float = 5.0,
    ) -> None:
        self.repo = repo
        self.pipeline_class = pipeline_class
        self.flow_shift = flow_shift
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self._dp: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str, dtype: torch.dtype) -> None:
        import diffusers
        from diffusers import AutoencoderKLWan

        from worldkernels.models.schedulers import FlowUniPCMultistepScheduler

        self.device = device
        self.dtype = dtype

        log.info("Loading Wan components from %s", self.repo)
        vae = AutoencoderKLWan.from_pretrained(
            self.repo, subfolder="vae", torch_dtype=torch.float32
        )
        pipe_cls = getattr(diffusers, self.pipeline_class)
        dp = pipe_cls.from_pretrained(self.repo, vae=vae, torch_dtype=dtype)
        dp.scheduler = FlowUniPCMultistepScheduler.for_wan(flow_shift=self.flow_shift)
        dp.to(device)

        self._dp = dp
        self._loaded = True
        log.info(
            "WanI2VPipeline loaded on %s (%.1f GB VRAM)",
            device,
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        )

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        embeds, _ = self._encode_text(prompt, None)
        return embeds

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
    ) -> WanLatent:
        prompt_embeds, negative_prompt_embeds = self._encode_text(prompt, negative_prompt)
        if image is not None:
            last_frame = self._encode_image(image, height=height, width=width)
        else:
            last_frame = torch.zeros(1, 3, height, width, device=self.device, dtype=self.dtype)
        return WanLatent(last_frame, prompt_embeds, negative_prompt_embeds)

    def apply_prompt(self, conditioning: WanLatent, prompt_embeds: torch.Tensor) -> WanLatent:
        return WanLatent(
            conditioning.last_frame, prompt_embeds, conditioning.negative_prompt_embeds
        )

    def generate(
        self,
        conditioning: WanLatent,
        *,
        num_steps: int,
        guidance: float,
        num_frames: int,
        seed: int,
    ) -> GenerationResult:
        assert self._dp is not None, "Pipeline not loaded — call load() first"
        height, width = conditioning.last_frame.shape[-2], conditioning.last_frame.shape[-1]
        gen_device = "cpu" if self.device == "cpu" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        with torch.no_grad():
            result = self._dp(
                image=self._to_pil(conditioning.last_frame),
                prompt_embeds=conditioning.prompt_embeds,
                negative_prompt_embeds=conditioning.negative_prompt_embeds,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
                output_type="latent",
            )
        latent = result.frames if hasattr(result, "frames") else result[0]
        if not torch.is_tensor(latent):
            latent = latent[0]
        return GenerationResult(latent=latent, video=self.decode(latent))

    def advance(self, conditioning: WanLatent, next_image: torch.Tensor) -> WanLatent:
        return WanLatent(
            next_image.to(self.dtype),
            conditioning.prompt_embeds,
            conditioning.negative_prompt_embeds,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        r"""VAE-decode a normalized latent to a video tensor in ``[-1, 1]``."""
        assert self._dp is not None, "Pipeline not loaded — call load() first"
        vae = self._dp.vae
        latent = latent.to(vae.dtype)
        mean = torch.tensor(vae.config.latents_mean, device=latent.device, dtype=vae.dtype)
        std = torch.tensor(vae.config.latents_std, device=latent.device, dtype=vae.dtype)
        shape = (1, vae.config.z_dim, 1, 1, 1)
        latent = latent * std.view(shape) + mean.view(shape)
        with torch.no_grad():
            return vae.decode(latent, return_dict=False)[0]

    def profile_vram(self, *, height: int, width: int, num_frames: int) -> float:
        clip_bytes = 3 * height * width * num_frames * 4
        latent_bytes = 16 * ((num_frames + 3) // 4) * (height // 8) * (width // 8) * 4
        return (clip_bytes + latent_bytes) / (1024 * 1024) * 2.0 + 1024.0

    def _encode_text(
        self,
        prompt: str,
        negative_prompt: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self._dp is not None, "Pipeline not loaded — call load() first"
        prompt_embeds, negative_prompt_embeds = self._dp.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=negative_prompt is not None,
            device=self.device,
        )
        return prompt_embeds, negative_prompt_embeds

    def _encode_image(self, image: Any, *, height: int, width: int) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        if isinstance(image, str):
            from PIL import Image

            img = Image.open(image).convert("RGB")
            tensor = TF.to_tensor(img.resize((width, height)))
        else:
            tensor = torch.as_tensor(image, dtype=torch.float32)
            if tensor.ndim == 3 and tensor.shape[0] != 3:
                tensor = tensor.permute(2, 0, 1)
            if tensor.ndim == 4:
                tensor = tensor[0]
        return tensor.clamp(0, 1).to(device=self.device, dtype=self.dtype).unsqueeze(0)

    def _to_pil(self, last_frame: torch.Tensor) -> "Image":
        from PIL import Image

        tensor = last_frame[0] if last_frame.ndim == 4 else last_frame
        array = tensor.clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array)
