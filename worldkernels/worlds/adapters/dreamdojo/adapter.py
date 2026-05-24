r"""DreamDojo action-conditioned world model adapter.

Wraps NVIDIA DreamDojo (2B/14B) — an action-conditioned video diffusion model
built on Cosmos-Predict2.5. Actions are robot joint vectors. Composes
:class:`CosmosPredict2Pipeline` and injects the action tensor as a per-step
``extras`` field rather than overriding the pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.adapters.dreamdojo.checkpoint import download_dreamdojo_checkpoint
from worldkernels.worlds.base import AbstractWorld
from worldkernels.worlds.pipelines.cosmos_predict2 import (
    CosmosPredict2Latent,
    CosmosPredict2Pipeline,
)

if TYPE_CHECKING:
    from worldkernels.core.action import Action
    from worldkernels.core.config import WorldConfig

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

CKPT_DIRS = {
    "2b_pretrain": "2B_pretrain",
    "2b_gr1": "2B_GR1_post-train",
    "2b_agibot": "2B_AgiBot_post-train",
    "2b_g1": "2B_G1_post-train",
    "2b_yam": "2B_YAM_post-train",
    "14b_pretrain": "14B_pretrain",
    "14b_gr1": "14B_GR1_post-train",
}


class DreamDojoWorld(AbstractWorld):
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
        variant: str = "2b_pretrain",
        action_dim: int = 384,
        chunk_size: int = 12,
        num_inference_steps: int = 35,
        guidance_scale: float = 3.0,
        **kwargs: Any,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.variant = variant
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self._experiment_override = experiment
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.pipeline: CosmosPredict2Pipeline | None = None

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        experiment = self._experiment_override or EXPERIMENTS.get(self.variant, DEFAULT_EXPERIMENT)
        self.pipeline = CosmosPredict2Pipeline(experiment=experiment, config_file=CONFIG_FILE)
        ckpt = self._resolve_checkpoint()
        self.pipeline.load(device, dtype, ckpt)

    def _resolve_checkpoint(self) -> str:
        if self.ckpt_path is not None:
            return self.ckpt_path
        ckpt_dir = CKPT_DIRS.get(self.variant, "2B_pretrain")
        log.info("Downloading DreamDojo %s checkpoint...", self.variant)
        return download_dreamdojo_checkpoint(ckpt_dir)

    def warmup(self, config: WorldConfig) -> None:
        if self.pipeline is None:
            return
        null_action = torch.zeros(
            1, self.chunk_size, self.action_dim, device=self.device, dtype=self.dtype
        )
        self.pipeline.warmup(
            height=config.height,
            width=config.width,
            frames_per_step=config.frames_per_step,
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
        new_latent, new_last_frame = self.pipeline.denoise(
            cs,
            num_steps=self.num_inference_steps,
            guidance=self.guidance_scale,
            seed=int(time.perf_counter() * 1000) % (2**31),
            extras={"action": action},
        )
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
        cs = self.pipeline.create_initial_state(
            prompt=config.initial_prompt or "",
            initial_image=config.initial_image,
            height=config.height,
            width=config.width,
            frames_per_step=config.frames_per_step,
            seed=seed,
        )
        return LatentState(data=cs, device=self.device)

    def estimate_vram_mb(self, config: WorldConfig) -> float:
        return CosmosPredict2Pipeline.estimate_latent_vram_mb(
            CosmosPredict2Pipeline(experiment="", config_file=""),
            height=config.height,
            width=config.width,
            frames_per_step=config.frames_per_step,
        )
