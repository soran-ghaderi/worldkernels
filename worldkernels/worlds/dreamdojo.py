r"""DreamDojo — an action-conditioned interactive world model.

Wraps NVIDIA DreamDojo (2B/14B), an action-conditioned video diffusion model
built on Cosmos-Predict2.5. Actions are robot joint vectors. Composes
`CosmosPredict2Pipeline` and injects the action tensor as a per-step
``extras`` field. A genuine `InteractiveWorldModel`:
the transition is conditioned on the action, not open-loop resampling.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.models.cosmos_predict2 import CosmosPredict2Latent, CosmosPredict2Pipeline
from worldkernels.models.dreamdojo.checkpoint import CKPT_DIRS, download_dreamdojo_checkpoint
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.base import InteractiveWorldModel

if TYPE_CHECKING:
    from worldkernels.config import WorldConfig
    from worldkernels.core.action import Action

log = logging.getLogger(__name__)

CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
DEFAULT_EXPERIMENT = "dreamdojo_2b_480_640_pretrain"

EXPERIMENTS = {
    "2b_pretrain": "dreamdojo_2b_480_640_pretrain",
    "2b_gr1": "dreamdojo_2b_480_640_gr1",
    "2b_agibot": "dreamdojo_2b_480_640_agibot",
    "2b_g1": "dreamdojo_2b_480_640_g1",
    "2b_yam": "dreamdojo_2b_480_640_yam",
    "14b_pretrain": "dreamdojo_14b_480_640_pretrain",
    "14b_gr1": "dreamdojo_14b_480_640_gr1",
}

DEFAULT_VARIANT = "2b_pretrain"

# Per-variant native geometry + action space. All DreamDojo action-conditioned
# experiments train at 480x640; the pretrain "bridge" uses a 7-dim action, the
# GR00T post-trains use the 384-dim unified action space.
VARIANT_DEFAULTS: dict[str, dict[str, int]] = {
    "2b_pretrain": {"action_dim": 7, "height": 480, "width": 640},
    "2b_gr1": {"action_dim": 384, "height": 480, "width": 640},
    "2b_agibot": {"action_dim": 384, "height": 480, "width": 640},
    "2b_g1": {"action_dim": 384, "height": 480, "width": 640},
    "2b_yam": {"action_dim": 384, "height": 480, "width": 640},
    "14b_pretrain": {"action_dim": 7, "height": 480, "width": 640},
    "14b_gr1": {"action_dim": 384, "height": 480, "width": 640},
}


class DreamDojoWorld(InteractiveWorldModel):
    r"""Action-conditioned video world model (DreamDojo 2B/14B).

    Actions are robot joint vectors \(\in \mathbb{R}^{T \times D}\) where
    \(T\) = chunk_size frames and \(D\) = action_dim joints.
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
        experiment: str | None = None,
        variant: str = DEFAULT_VARIANT,
        action_dim: int | None = None,
        chunk_size: int = 12,
        num_inference_steps: int = 35,
        guidance_scale: float = 0.0,
        text_encoder: str = "auto",
        **kwargs: Any,
    ) -> None:
        # num_inference_steps/guidance_scale are load-time (constructor) settings — the
        # vendor recipe is 35 steps with guidance 0.0 (no CFG); per-session WorldConfig
        # values do not override them. action_dim and native resolution resolve per variant.
        self.ckpt_path = ckpt_path
        self.variant = variant
        defaults = VARIANT_DEFAULTS.get(variant, VARIANT_DEFAULTS[DEFAULT_VARIANT])
        self.action_dim = action_dim if action_dim is not None else defaults["action_dim"]
        self.native_height = defaults["height"]
        self.native_width = defaults["width"]
        self.chunk_size = chunk_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.text_encoder = text_encoder
        self._experiment_override = experiment
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.pipeline: CosmosPredict2Pipeline | None = None

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        experiment = self._experiment_override or EXPERIMENTS.get(self.variant, DEFAULT_EXPERIMENT)
        self.pipeline = CosmosPredict2Pipeline(
            experiment=experiment, config_file=CONFIG_FILE, text_encoder=self.text_encoder
        )
        self.pipeline.load(device, dtype, self._resolve_checkpoint())

    def _resolve_checkpoint(self) -> str:
        if self.ckpt_path is not None:
            return self.ckpt_path
        ckpt_dir = CKPT_DIRS.get(self.variant, "2B_pretrain")
        log.info("Downloading DreamDojo %s checkpoint...", self.variant)
        return download_dreamdojo_checkpoint(ckpt_dir)

    def _geometry(self, config: WorldConfig) -> tuple[int, int, int]:
        r"""Snap the generic WorldConfig defaults (480x848, 8 frames) to native geometry.

        DreamDojo trains at a fixed per-variant resolution; the shared WorldConfig
        defaults are Wan/Cosmos-oriented. An explicit non-default value is honored.
        """
        height = self.native_height if config.height == 480 else config.height
        width = self.native_width if config.width == 848 else config.width
        frames = self.chunk_size if config.frames_per_step == 8 else config.frames_per_step
        return height, width, frames

    def warmup(self, config: WorldConfig) -> None:
        if self.pipeline is None:
            return
        height, width, frames = self._geometry(config)
        null_action = torch.zeros(
            1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
        )
        self.pipeline.warmup(
            height=height,
            width=width,
            frames_per_step=frames,
            extras={"action": null_action},
        )

    def encode_action(self, action: Action) -> torch.Tensor:
        if action.action_type == "null":
            return torch.zeros(
                1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
            )

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

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        assert self.pipeline is not None, "Pipeline not loaded — call initialize() first"
        cs: CosmosPredict2Latent = state.data
        action = (
            action_encoded.to(device=self.device, dtype=self.dtype)
            if action_encoded.numel() > 0
            else torch.zeros(
                1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
            )
        )
        new_latent, video = self.pipeline.denoise(
            cs,
            num_steps=self.num_inference_steps,
            guidance=self.guidance_scale,
            seed=int(time.perf_counter() * 1000) % (2**31),
            extras={"action": action},
        )
        new_last_frame = ((video + 1.0) * 0.5)[0, :, -1].clamp(0, 1)
        return LatentState(
            data=CosmosPredict2Latent(new_latent, new_last_frame, cs.text_emb, cs.neg_text_emb),
            device=state.device,
        )

    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        assert self.pipeline is not None, "Pipeline not loaded — call initialize() first"
        t0 = time.perf_counter()
        cs: CosmosPredict2Latent = state.data
        frames = None
        latent_out = None

        if "frames" in modalities:
            video = self.pipeline.decode(cs.latent)
            video_uint8 = ((video + 1.0) * 0.5).clamp(0, 1).mul(255).to(torch.uint8)
            frames = [
                video_uint8[0, :, t].permute(1, 2, 0).cpu().numpy().tobytes()
                for t in range(video_uint8.shape[2])
            ]
        if "latent" in modalities:
            latent_out = cs.latent

        return Observation(
            step_index=0,
            generation_time_ms=(time.perf_counter() - t0) * 1000.0,
            frames=frames,
            latent=latent_out,
        )

    def create_initial_state(self, config: WorldConfig, seed: int) -> LatentState:
        assert self.pipeline is not None, "Pipeline not loaded — call initialize() first"
        height, width, frames = self._geometry(config)
        config.height, config.width, config.frames_per_step = height, width, frames
        cs = self.pipeline.create_initial_state(
            prompt=config.initial_prompt or "",
            initial_image=config.initial_image,
            height=height,
            width=width,
            frames_per_step=frames,
            seed=seed,
        )
        return LatentState(data=cs, device=self.device)

    def profile_vram(self, config: WorldConfig) -> float:
        height, width, frames = self._geometry(config)
        return CosmosPredict2Pipeline.estimate_latent_vram_mb(
            CosmosPredict2Pipeline(experiment="", config_file=""),
            height=height,
            width=width,
            frames_per_step=frames,
        )
