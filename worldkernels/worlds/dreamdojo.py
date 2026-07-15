r"""DreamDojo — an action-conditioned interactive world model, natively owned."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.core.state import BlockHandle, ConditioningBundle, WorldState, WorldStateMeta
from worldkernels.models.dreamdojo.checkpoint import CKPT_DIRS
from worldkernels.models.dreamdojo.pipeline import (
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    DreamDojoPipeline,
)
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.base import InteractiveWorldModel

if TYPE_CHECKING:
    from worldkernels.config import WorldConfig
    from worldkernels.core.action import Action

log = logging.getLogger(__name__)

DEFAULT_VARIANT = "2b_pretrain"
ACTION_DIM = 384


class DreamDojoWorld(InteractiveWorldModel):
    r"""Action-conditioned video world model (DreamDojo 2B/14B, all released variants).

    Actions are relative robot joint chunks \(\in \mathbb{R}^{12 \times 384}\)
    in the universal DreamDojo action layout. Each `transition` generates one
    13-frame clip via the native `DreamDojoPipeline`
    and carries the last decoded frame as the next conditioning prefix.
    """

    name = "dreamdojo"
    stage_exec_modes = {
        StageType.ENCODE: StageExecMode.SINGLE_SHOT,
        StageType.TRANSITION: StageExecMode.ITERATIVE,
        StageType.DECODE: StageExecMode.SINGLE_SHOT,
    }
    transition_mode = TransitionMode.BIDIRECTIONAL
    supports_streaming = False
    supports_kv_cache = False

    def __init__(
        self,
        ckpt_path: str | None = None,
        variant: str = DEFAULT_VARIANT,
        action_dim: int = ACTION_DIM,
        chunk_size: int = 12,
        num_inference_steps: int = 35,
        guidance_scale: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if variant not in CKPT_DIRS:
            raise ValueError(f"unknown DreamDojo variant {variant!r}; known: {sorted(CKPT_DIRS)}")
        self.ckpt_path = ckpt_path
        self.variant = variant
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.pipeline: DreamDojoPipeline | None = None

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.pipeline = DreamDojoPipeline(
            self.variant,
            num_steps=self.num_inference_steps,
            guidance=self.guidance_scale,
        ).load(device, dtype, self.ckpt_path)

    def _geometry(self, config: WorldConfig) -> tuple[int, int, int]:
        r"""Snap the generic WorldConfig defaults to DreamDojo's native 480x640."""
        height = NATIVE_HEIGHT if config.height == 480 else config.height
        width = NATIVE_WIDTH if config.width == 848 else config.width
        frames = self.chunk_size if config.frames_per_step == 8 else config.frames_per_step
        return height, width, frames

    def warmup(self, config: WorldConfig) -> None:
        if self.pipeline is None:
            return
        height, width, _ = self._geometry(config)
        cond = torch.zeros(1, 3, 1, height, width, dtype=torch.uint8, device=self.device)
        null_action = torch.zeros(1, self.chunk_size, self.action_dim, device=self.device)
        self.pipeline.generate_clip(cond, null_action, num_steps=1, seed=0)

    def encode_action(self, action: Action) -> torch.Tensor:
        if action.action_type == "null":
            return torch.zeros(
                1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
            )
        if "latent_action" in action.payload or "frames" in action.payload:
            return self._encode_latent_action(action)

        joints = action.payload.get("joints", [0.0] * self.action_dim)
        joints_t = torch.tensor(joints, device=self.device, dtype=self.dtype)

        if joints_t.ndim == 1:
            joints_t = joints_t.unsqueeze(0).expand(self.chunk_size, -1)
        elif joints_t.ndim == 2 and joints_t.shape[0] != self.chunk_size:
            pad_len = self.chunk_size - joints_t.shape[0]
            if pad_len > 0:
                joints_t = torch.cat(
                    [
                        joints_t,
                        torch.zeros(pad_len, self.action_dim, device=self.device, dtype=self.dtype),
                    ],
                    dim=0,
                )
            else:
                joints_t = joints_t[: self.chunk_size]

        return joints_t.unsqueeze(0)

    def _encode_latent_action(self, action: Action) -> torch.Tensor:
        r"""Universal action chunk carrying a continuous latent action in ``[352:384)``.

        ``payload["latent_action"]`` supplies latent actions directly (``(32,)``
        broadcast across the chunk or ``(chunk_size, 32)`` per frame);
        ``payload["frames"]`` supplies ``(T, 3, H, W)`` uint8 source frames whose
        consecutive pairs the LAM encodes.
        """
        from worldkernels.models.dreamdojo.lam import LATENT_ACTION_SLICE

        if "latent_action" in action.payload:
            z = torch.as_tensor(
                action.payload["latent_action"], device=self.device, dtype=self.dtype
            )
        else:
            frames = torch.as_tensor(action.payload["frames"])
            z = self._lam_encoder().encode_video(frames).to(device=self.device, dtype=self.dtype)

        if z.ndim == 1:
            z = z.unsqueeze(0).expand(self.chunk_size, -1)
        elif z.shape[0] != self.chunk_size:
            pad = self.chunk_size - z.shape[0]
            z = torch.cat([z, z[-1:].expand(pad, -1)], dim=0) if pad > 0 else z[: self.chunk_size]

        chunk = torch.zeros(
            1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
        )
        chunk[0, :, LATENT_ACTION_SLICE] = z
        return chunk

    def _lam_encoder(self):
        if getattr(self, "_lam", None) is None:
            from worldkernels.models.dreamdojo.checkpoint import download_lam_checkpoint
            from worldkernels.models.dreamdojo.lam import LatentActionEncoder

            self._lam = LatentActionEncoder.from_checkpoint(
                download_lam_checkpoint(), device=self.device
            )
        return self._lam

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        assert self.pipeline is not None, "Pipeline not loaded — call initialize() first"
        ws: WorldState = state.data
        action = (
            action_encoded
            if action_encoded.numel() > 0
            else torch.zeros(
                1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
            )
        )
        cond_frame = ws.conditioning.image_cond
        assert cond_frame is not None
        seed = int(ws.extras["seed"].item()) + ws.step_index

        latent, video = self.pipeline.generate_clip(cond_frame.unsqueeze(2), action, seed=seed)
        video_uint8 = self.pipeline.to_uint8(video)
        new_state = WorldState(
            latent=BlockHandle(latent),
            conditioning=ConditioningBundle(image_cond=video_uint8[:, :, -1]),
            meta=ws.meta,
            step_index=ws.step_index + 1,
            extras={"seed": ws.extras["seed"], "video": video_uint8},
        )
        return LatentState(data=new_state, device=state.device)

    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        assert self.pipeline is not None, "Pipeline not loaded — call initialize() first"
        t0 = time.perf_counter()
        ws: WorldState = state.data
        frames = None
        latent_out = None

        if "frames" in modalities:
            video_uint8 = ws.extras.get("video")
            if video_uint8 is None:
                video_uint8 = self.pipeline.to_uint8(self.pipeline.decode_latent(ws.latent.tensor))
            frames = [
                video_uint8[0, :, t].permute(1, 2, 0).cpu().numpy().tobytes()
                for t in range(video_uint8.shape[2])
            ]
        if "latent" in modalities:
            latent_out = ws.latent.tensor

        return Observation(
            step_index=ws.step_index,
            generation_time_ms=(time.perf_counter() - t0) * 1000.0,
            frames=frames,
            latent=latent_out,
        )

    def create_initial_state(self, config: WorldConfig, seed: int) -> LatentState:
        assert self.pipeline is not None, "Pipeline not loaded — call initialize() first"
        height, width, frames = self._geometry(config)
        config.height, config.width, config.frames_per_step = height, width, frames

        if config.initial_image is not None:
            cond_frame = _load_image_uint8(config.initial_image, height, width, self.device)
        else:
            cond_frame = torch.zeros(1, 3, height, width, dtype=torch.uint8, device=self.device)

        h, w = height // 8, width // 8
        ws = WorldState(
            latent=BlockHandle(
                torch.zeros(1, 16, 4, h, w, device=self.device, dtype=torch.float32)
            ),
            conditioning=ConditioningBundle(image_cond=cond_frame),
            meta=WorldStateMeta(
                height=height,
                width=width,
                dtype=str(self.dtype),
                transition_mode=self.transition_mode.value,
                frames_per_step=frames,
            ),
            extras={"seed": torch.tensor(seed, dtype=torch.int64)},
        )
        return LatentState(data=ws, device=self.device)

    def profile_vram(self, config: WorldConfig) -> float:
        height, width, frames = self._geometry(config)
        pipeline = self.pipeline or DreamDojoPipeline(self.variant)
        return pipeline.profile_vram(height=height, width=width, num_frames=frames)


class DreamDojoStudentWorld(DreamDojoWorld):
    r"""Self-Forcing-distilled DreamDojo student: causal per-frame streaming.

    Each `transition` consumes a 12-action chunk and rolls the causal student
    three latent frames (12 pixels) forward against a rolling 3-frame KV
    window, carrying the last 9 decoded pixels and 8 action transitions as the
    next chunk's context — the reference chunked streaming scheme with
    self-consistent absolute action alignment. Requires a distilled checkpoint
    (``ckpt_path``); the architecture is weight-compatible with teacher
    checkpoints for smoke runs.
    """

    name = "dreamdojo_student"
    transition_mode = TransitionMode.CAUSAL
    supports_streaming = True

    NEW_LATENT_FRAMES = 3
    KV_WINDOW = 3

    def __init__(self, *args: Any, student_steps: int = 4, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.student_steps = student_steps

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        assert self.pipeline is not None, "Pipeline not loaded — call initialize() first"
        ws: WorldState = state.data
        fresh = (
            action_encoded
            if action_encoded.numel() > 0
            else torch.zeros(
                1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
            )
        )
        history = ws.extras.get("action_history")
        window_actions = fresh if history is None else torch.cat([history.to(fresh), fresh], dim=1)
        context_px = ws.extras.get("context_px")
        if context_px is None:
            cond_frame = ws.conditioning.image_cond
            assert cond_frame is not None
            context_px = cond_frame.unsqueeze(2)
        seed = int(ws.extras["seed"].item()) + ws.step_index

        latents, new_video = self.pipeline.stream_chunk(
            context_px,
            window_actions,
            student_steps=self.student_steps,
            window=self.KV_WINDOW,
            new_frames=self.NEW_LATENT_FRAMES,
            seed=seed,
        )
        new_uint8 = self.pipeline.to_uint8(new_video)
        context_len = (self.KV_WINDOW - 1) * 4 + 1
        keep_actions = (self.KV_WINDOW - 1) * 4
        new_state = WorldState(
            latent=BlockHandle(latents.float()),
            conditioning=ConditioningBundle(image_cond=new_uint8[:, :, -1]),
            meta=ws.meta,
            step_index=ws.step_index + 1,
            extras={
                "seed": ws.extras["seed"],
                "video": new_uint8,
                "context_px": new_uint8[:, :, -context_len:],
                "action_history": window_actions[:, -keep_actions:].detach(),
            },
        )
        return LatentState(data=new_state, device=state.device)


def _load_image_uint8(image: str | Any, height: int, width: int, device: str) -> torch.Tensor:
    r"""Load an initial image as a ``(1, 3, H, W)`` uint8 conditioning frame."""
    if isinstance(image, str):
        from PIL import Image

        img = Image.open(image).convert("RGB").resize((width, height))
        import numpy as np

        arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
        return arr.unsqueeze(0).to(device)

    img_t = torch.as_tensor(image)
    if img_t.ndim == 3 and img_t.shape[0] != 3:
        img_t = img_t.permute(2, 0, 1)
    if torch.is_floating_point(img_t):
        img_t = (img_t.clamp(0, 1) * 255.0).to(torch.uint8)
    if img_t.ndim == 3:
        img_t = img_t.unsqueeze(0)
    return img_t.to(device)
