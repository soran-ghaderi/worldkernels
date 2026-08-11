r"""Native DreamDojo inference pipeline: action-conditioned rectified-flow video2world."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from worldkernels.models.dreamdojo.net import ActionVideoDiT
    from worldkernels.models.dreamdojo.tokenizer import VideoTokenizer

log = logging.getLogger(__name__)

__all__ = ["DreamDojoPipeline", "seeded_noise"]

FRAMES_PER_CLIP = 13
NATIVE_HEIGHT = 480
NATIVE_WIDTH = 640


def seeded_noise(shape: tuple[int, ...], seed: int, device: "str | torch.device") -> "torch.Tensor":
    r"""Architecture-invariant unit Gaussian, identical to the reference generator."""
    import numpy as np
    import torch

    array = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(array).to(device=device)


class DreamDojoPipeline:
    r"""One 13-frame clip per call: encode conditioning, denoise with UniPC, decode.

    The clip forward mirrors the reference exactly: frame 0 (or a longer pixel
    prefix) is the conditioning, the latent condition mask replaces both the
    solver input and the predicted velocity on conditioned frames, the fixed
    empty-prompt CR1 embedding is the cross-attention context, and sampling
    runs the owned flow-matching UniPC solver in float32 around a bf16 net.
    """

    def __init__(
        self,
        variant: str = "2b_gr1",
        *,
        num_steps: int = 35,
        guidance: float = 0.0,
        flow_shift: float = 5.0,
    ) -> None:
        self.variant = variant
        self.num_steps = num_steps
        self.guidance = guidance
        self.flow_shift = flow_shift
        self.net: ActionVideoDiT | None = None
        self.tokenizer: VideoTokenizer | None = None
        self.text_emb: torch.Tensor | None = None
        self.device: str | torch.device = "cuda"
        self.dtype: torch.dtype | None = None

    def load(
        self,
        device: "str | torch.device",
        dtype: "torch.dtype | None" = None,
        ckpt_path: str | None = None,
    ) -> "DreamDojoPipeline":
        import torch

        from worldkernels.models.dreamdojo.checkpoint import (
            download_text_embedding,
            load_net,
        )
        from worldkernels.models.dreamdojo.tokenizer import VideoTokenizer

        dtype = dtype or torch.bfloat16
        self.device = device
        self.dtype = dtype
        self.net = load_net(self.variant, ckpt_path=ckpt_path, device=device, dtype=dtype)
        self.tokenizer = VideoTokenizer().load(device)

        emb = torch.load(download_text_embedding(), map_location="cpu", weights_only=True)
        if isinstance(emb, (list, tuple)):
            emb = emb[0]
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        self.text_emb = emb.to(device=device, dtype=dtype)
        log.info(
            "DreamDojoPipeline(%s) loaded on %s (%.1f GB VRAM)",
            self.variant,
            device,
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        )
        return self

    @property
    def num_action_per_latent_frame(self) -> int:
        assert self.net is not None
        return self.net.num_action_per_latent_frame

    def encode_frames(self, frames_uint8: "torch.Tensor") -> "torch.Tensor":
        r"""Encode ``(B, 3, T, H, W)`` uint8 frames to normalized latents (bf16)."""

        assert self.tokenizer is not None and self.dtype is not None
        video = frames_uint8.to(device=self.device, dtype=self.dtype) / 127.5 - 1.0
        return self.tokenizer.encode(video).to(self.dtype)

    def decode_latent(self, latent: "torch.Tensor") -> "torch.Tensor":
        r"""Decode latents to video in \([-1, 1]\) float32."""
        assert self.tokenizer is not None
        return self.tokenizer.decode(latent.to(self.dtype)).float()

    @staticmethod
    def to_uint8(video: "torch.Tensor") -> "torch.Tensor":
        import torch

        return ((video.float() + 1.0) * 0.5).clamp(0.0, 1.0).mul(255.0).to(torch.uint8)

    def generate_clip(
        self,
        cond_frames_uint8: "torch.Tensor",
        action: "torch.Tensor",
        *,
        num_steps: int | None = None,
        guidance: float | None = None,
        seed: int = 1,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        r"""Generate one clip conditioned on a pixel prefix and a 12-action chunk.

        Args:
            cond_frames_uint8: ``(B, 3, T_c, H, W)`` uint8 conditioning prefix,
                ``T_c = 4k + 1``.
            action: ``(B, 12, action_dim)`` relative-action chunk.
            seed: Noise seed (architecture-invariant generator).

        Returns:
            ``(latent, video)``: final latents ``(B, 16, 4, H/8, W/8)`` float32
            and decoded video ``(B, 3, 13, H, W)`` in \([-1, 1]\) float32.
        """
        import torch

        from worldkernels.models.schedulers.flow_unipc import FlowUniPCMultistepScheduler

        assert self.net is not None and self.dtype is not None and self.text_emb is not None
        num_steps = num_steps if num_steps is not None else self.num_steps
        guidance = guidance if guidance is not None else self.guidance

        B, _, t_cond, H, W = cond_frames_uint8.shape
        video = torch.zeros(
            B, 3, FRAMES_PER_CLIP, H, W, dtype=torch.uint8, device=cond_frames_uint8.device
        )
        video[:, :, :t_cond] = cond_frames_uint8
        x0 = self.encode_frames(video)

        _, C, t_lat, h, w = x0.shape
        num_cond_latent = 1 + (t_cond - 1) // 4 if cond_frames_uint8.any() else 0
        mask = torch.zeros(B, 1, t_lat, h, w, dtype=self.dtype, device=self.device)
        mask[:, :, :num_cond_latent] = 1.0
        mask_c = mask.repeat(1, C, 1, 1, 1)
        padding_mask = torch.zeros(B, 1, h, w, dtype=self.dtype, device=self.device)
        action = action.to(device=self.device, dtype=self.dtype)
        text_emb = self.text_emb.expand(B, -1, -1)

        noise = seeded_noise((B, C, t_lat, h, w), seed, self.device)
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0)
        scheduler.set_timesteps(num_steps, device=self.device, shift=self.flow_shift)

        net = self.net
        x0_f = x0.float()
        mask_f = mask_c.float()
        latents = noise

        def denoise(xt: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
            xt_in = x0_f * mask_f + xt * (1 - mask_f)
            with torch.no_grad():
                v = net(
                    xt_in.to(self.dtype),
                    t.expand(B, 1),
                    ctx,
                    condition_video_input_mask_B_C_T_H_W=mask,
                    action=action,
                    padding_mask=padding_mask,
                ).float()
            return (noise - x0_f) * mask_f + v * (1 - mask_f)

        for t in scheduler.timesteps:
            t_b = t.reshape(1, 1)
            cond_v = denoise(latents, t_b, text_emb)
            if guidance > 0:
                uncond_v = denoise(latents, t_b, torch.zeros_like(text_emb))
                velocity = cond_v + guidance * (cond_v - uncond_v)
            else:
                velocity = cond_v
            latents = scheduler.step(velocity, t, latents, return_dict=False)[0]

        return latents, self.decode_latent(latents)

    def generate_stream_latents(
        self,
        context_latents: "torch.Tensor",
        window_actions: "torch.Tensor",
        noise: "torch.Tensor",
        *,
        student_steps: int = 4,
        window: int = 3,
    ) -> "torch.Tensor":
        r"""Causal few-step generation over one latent window (distilled student).

        Mirrors the reference Self-Forcing rollout: clean context frames are
        prefilled into per-block KV caches with a single \(t=0\) forward each,
        then every remaining frame runs ``student_steps`` TrigFlow denoise
        passes (KV read-only) against the rolling window followed by one clean
        forward that stores its K/V. Actions are absolute per latent
        transition: frame \(f\) consumes ``window_actions[(f-1)A:(f)A]``.

        Args:
            context_latents: ``(B, 16, S, h, w)`` clean context, ``S >= 1``.
            window_actions: ``(B, (T-1)*A, action_dim)`` transitions for the
                whole window.
            noise: ``(B, 16, T, h, w)`` unit noise; frames ``S..T-1`` are
                generated.
            window: Rolling KV window in latent frames.

        Returns:
            ``(B, 16, T, h, w)`` latents in the net dtype.
        """
        import torch

        from worldkernels.models.schedulers.trigflow import (
            TRIGFLOW_TIMES_4STEP,
            trigflow_scaling,
            trigflow_step,
        )

        assert self.net is not None and self.dtype is not None and self.text_emb is not None
        net = self.net
        A = net.num_action_per_latent_frame
        B, C, T, h, w = noise.shape
        S = context_latents.shape[2]
        assert 1 <= student_steps <= len(TRIGFLOW_TIMES_4STEP)
        assert window_actions.shape[1] == (T - 1) * A

        noise = noise.to(device=self.device, dtype=self.dtype)
        actions = window_actions.to(device=self.device, dtype=self.dtype)
        text_emb = self.text_emb.expand(B, -1, -1)
        zero_mask = torch.zeros(B, 1, 1, h, w, dtype=self.dtype, device=self.device)
        padding_mask = torch.zeros(B, 1, h, w, dtype=self.dtype, device=self.device)
        t_zero = torch.zeros(B, 1, dtype=self.dtype, device=self.device)

        output = torch.zeros(B, C, T, h, w, dtype=self.dtype, device=self.device)
        output[:, :, :S] = context_latents.to(device=self.device, dtype=self.dtype)

        caches = net.allocate_kv_caches(
            max_frames=T,
            batch_size=B,
            latent_h=h,
            latent_w=w,
            window=window,
            device=self.device,
            dtype=self.dtype,
        )

        def frame_action(f: int) -> torch.Tensor | None:
            return actions[:, (f - 1) * A : f * A] if f > 0 else None

        def store_frame(latent_frame: torch.Tensor, f: int) -> None:
            with torch.no_grad():
                net.forward_frame(
                    latent_frame,
                    f,
                    t_zero,
                    text_emb,
                    condition_video_input_mask_B_C_1_H_W=zero_mask,
                    action=frame_action(f),
                    kv_caches=caches,
                    store_kv=True,
                    padding_mask=padding_mask,
                )

        for f in range(S):
            store_frame(output[:, :, f : f + 1], f)

        t_steps = TRIGFLOW_TIMES_4STEP
        for f in range(S, T):
            frame_noise = noise[:, :, f : f + 1]
            frame_seq = frame_noise
            x0_pred = frame_seq
            for s_idx in range(student_steps):
                t = torch.full((B, 1), t_steps[s_idx], dtype=self.dtype, device=self.device)
                c_skip, c_out, c_in, c_noise = trigflow_scaling(t)
                c_bc = c_in.view(B, 1, 1, 1, 1)
                with torch.no_grad():
                    net_out = net.forward_frame(
                        (frame_seq * c_bc).to(self.dtype),
                        f,
                        c_noise,
                        text_emb,
                        condition_video_input_mask_B_C_1_H_W=zero_mask,
                        action=frame_action(f),
                        kv_caches=caches,
                        store_kv=False,
                        padding_mask=padding_mask,
                    )
                net_out = net_out.float().to(frame_seq.dtype)
                x0_pred = (
                    c_skip.view(B, 1, 1, 1, 1) * frame_seq + c_out.view(B, 1, 1, 1, 1) * net_out
                )
                if s_idx < student_steps - 1:
                    t_next = torch.tensor(t_steps[s_idx + 1], dtype=self.dtype, device=self.device)
                    frame_seq = trigflow_step(x0_pred, frame_noise, t_next)
            output[:, :, f : f + 1] = x0_pred
            if f < T - 1:
                store_frame(x0_pred, f)

        return output

    def stream_chunk(
        self,
        cond_frames_uint8: "torch.Tensor",
        window_actions: "torch.Tensor",
        *,
        student_steps: int = 4,
        window: int = 3,
        new_frames: int = 3,
        seed: int = 1,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        r"""One causal streaming chunk from a pixel-context prefix.

        Encodes the prefix (``T_c = 4k+1`` pixels, ``k+1`` clean latent
        frames), rolls the student forward ``new_frames`` latent frames, and
        decodes. ``window_actions`` covers every transition in the window,
        context and new: ``(B, (k + new_frames) * 4, action_dim)``.

        Returns:
            ``(latents, new_video)``: full-window latents and the newly
            generated pixels ``(B, 3, new_frames*4, H, W)`` in \([-1, 1]\).
        """
        import torch

        B, _, t_cond, H, W = cond_frames_uint8.shape
        start_idx = 1 + (t_cond - 1) // 4
        T = start_idx + new_frames
        total_px = (T - 1) * 4 + 1

        video = torch.zeros(
            B, 3, total_px, H, W, dtype=torch.uint8, device=cond_frames_uint8.device
        )
        video[:, :, :t_cond] = cond_frames_uint8
        x0 = self.encode_frames(video)
        context = x0[:, :, :start_idx]

        noise = seeded_noise((B, 16, T, H // 8, W // 8), seed, self.device)
        latents = self.generate_stream_latents(
            context, window_actions, noise, student_steps=student_steps, window=window
        )
        decoded = self.decode_latent(latents)
        return latents, decoded[:, :, (start_idx - 1) * 4 + 1 :]

    def profile_vram(self, *, height: int, width: int, num_frames: int) -> float:
        r"""Rough VRAM estimate in MB for a session at the given geometry."""
        del num_frames
        latent = 16 * 4 * (height // 8) * (width // 8) * 2
        weights = 4.3e9 if self.variant.startswith("2b") else 28e9
        vae = 0.3e9
        return (weights + vae + latent * 50) / 1e6
