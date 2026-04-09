r"""Cosmos-Predict2.5 video-to-world adapter.

Text-conditioned video generation via NVIDIA Cosmos-Predict2.5-2B.
Auto-downloads checkpoint from HuggingFace.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from worldkernels.worlds.adapters._cosmos_predict2 import CosmosBaseWorld, download_hf_file

if TYPE_CHECKING:
    from worldkernels.core.action import Action

log = logging.getLogger(__name__)

_HF_REPO = "nvidia/Cosmos-Predict2.5-2B"
_HF_CKPT_FILE = "base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt"

_DEFAULT_EXPERIMENT = (
    "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16"
    "-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2"
)


class CosmosPredict2World(CosmosBaseWorld):
    r"""Cosmos-Predict2.5 video-to-world model (2B).

    Actions are text prompts. Each step generates a video chunk conditioned on
    the prompt and the previous frame.
    """

    name = "cosmos_predict2"
    config_file = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
    default_experiment = _DEFAULT_EXPERIMENT
    hf_repo = _HF_REPO

    def _resolve_checkpoint(self) -> str:
        if self.ckpt_path is not None:
            return self.ckpt_path
        try:
            log.info("Downloading Cosmos-Predict2.5-2B checkpoint...")
            return download_hf_file(_HF_REPO, _HF_CKPT_FILE)
        except Exception as e:
            log.info(
                "Cosmos-Predict2.5-2B unavailable (%s), falling back to DreamDojo pretrain",
                e.__class__.__name__,
            )
            from worldkernels.worlds.adapters._cosmos_predict2 import (
                download_dreamdojo_checkpoint,
            )

            self.experiment = "dreamdojo_2b_480_640_pretrain"
            self.config_file = (
                "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
            )
            return download_dreamdojo_checkpoint()

    def encode_action(self, action: Action) -> torch.Tensor:
        prompt = action.payload.get("prompt", "") if action.payload else ""
        if not prompt:
            return torch.empty(0, device=self.device)
        return self._compute_text_embedding(prompt)
