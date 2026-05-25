r"""Cosmos-Predict2.5 video-to-world adapter.

Text-conditioned video generation via NVIDIA Cosmos-Predict2.5-2B.
Composes :class:`CosmosPredict2Pipeline` and maps text actions to the
pipeline's text conditioning.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.base import AbstractWorld
from worldkernels.worlds.pipelines.cosmos_predict2 import (
    CosmosPredict2Latent,
    CosmosPredict2Pipeline,
)

if TYPE_CHECKING:
    from worldkernels.core.action import Action
    from worldkernels.core.config import WorldConfig

log = logging.getLogger(__name__)

HF_REPO = "nvidia/Cosmos-Predict2.5-2B"
HF_CKPT_FILE = "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"
CONFIG_FILE = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
DEFAULT_EXPERIMENT = (
    "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16"
    "-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2"
)
FALLBACK_EXPERIMENT = "dreamdojo_2b_480_640_pretrain"
FALLBACK_CONFIG = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"


class CosmosPredict2World(AbstractWorld):
    r"""Cosmos-Predict2.5 video-to-world model (2B).

    Actions are text prompts. Each step generates a video chunk conditioned on
    the prompt and the previous frame.
    """

    name = "cosmos_predict2"
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
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        **kwargs: Any,
    ) -> None:
        self.ckpt_path = ckpt_path
        self._experiment_override = experiment
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.pipeline: CosmosPredict2Pipeline | None = None

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        ckpt, experiment, config_file = self._resolve_checkpoint_and_config()
        self.pipeline = CosmosPredict2Pipeline(experiment=experiment, config_file=config_file)
        self.pipeline.load(device, dtype, ckpt)

    def _resolve_checkpoint_and_config(self) -> tuple[str, str, str]:
        r"""Pick checkpoint path + experiment + config_file with fallback to DreamDojo pretrain."""
        experiment = self._experiment_override or DEFAULT_EXPERIMENT
        if self.ckpt_path is not None:
            return self.ckpt_path, experiment, CONFIG_FILE
        from worldkernels.worlds.pipelines.cosmos_predict2.pipeline import _download_hf_file

        try:
            log.info("Downloading Cosmos-Predict2.5-2B checkpoint...")
            ckpt = _download_hf_file(HF_REPO, HF_CKPT_FILE)
            return ckpt, experiment, CONFIG_FILE
        except Exception as e:
            log.info(
                "Cosmos-Predict2.5-2B unavailable (%s), falling back to DreamDojo pretrain",
                e.__class__.__name__,
            )
            from worldkernels.worlds.adapters.dreamdojo.checkpoint import (
                download_dreamdojo_checkpoint,
            )

            return download_dreamdojo_checkpoint(), FALLBACK_EXPERIMENT, FALLBACK_CONFIG

    def warmup(self, config: WorldConfig) -> None:
        if self.pipeline is None:
            return
        self.pipeline.warmup(
            height=config.height, width=config.width, frames_per_step=config.frames_per_step
        )

    def encode_action(self, action: Action) -> torch.Tensor:
        prompt = action.payload.get("prompt", "") if action.payload else ""
        if not prompt or self.pipeline is None:
            return torch.empty(0, device=self.device)
        return self.pipeline.encode_text(prompt)

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        assert self.pipeline is not None, "Pipeline not loaded — call initialize() first"
        cs: CosmosPredict2Latent = state.data
        text_emb = action_encoded if action_encoded.numel() > 0 else cs.text_emb
        next_state = CosmosPredict2Latent(cs.latent, cs.last_frame, text_emb, cs.neg_text_emb)
        new_latent, new_last_frame = self.pipeline.denoise(
            next_state,
            num_steps=self.num_inference_steps,
            guidance=self.guidance_scale,
            seed=int(time.perf_counter() * 1000) % (2**31),
        )
        return LatentState(
            data=CosmosPredict2Latent(new_latent, new_last_frame, text_emb, cs.neg_text_emb),
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
