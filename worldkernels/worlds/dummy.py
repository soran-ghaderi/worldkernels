r"""DummyWorld — a CPU-safe world model for development and testing.

Returns random noise with no real weights or compute. Reference template for
`InteractiveWorldModel` implementations.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.base import InteractiveWorldModel

if TYPE_CHECKING:
    from worldkernels.config import WorldConfig
    from worldkernels.core.action import Action

_LATENT_C = 4
_LATENT_FACTOR = 8


class DummyWorld(InteractiveWorldModel):
    r"""World model that returns random noise. No real weights, no compute."""

    name = "dummy"

    stage_exec_modes = {
        StageType.ENCODE: StageExecMode.SINGLE_SHOT,
        StageType.TRANSITION: StageExecMode.SINGLE_SHOT,
        StageType.DECODE: StageExecMode.SINGLE_SHOT,
    }

    transition_mode = TransitionMode.BIDIRECTIONAL
    supports_streaming = False
    supports_kv_cache = False

    def __init__(self) -> None:
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self._initialized = False

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self._initialized = True

    def warmup(self, config: WorldConfig) -> None:
        _ = self.create_initial_state(config, seed=0)

    def encode_action(self, action: Action) -> torch.Tensor:
        return torch.randn(1, _LATENT_C, device=self.device, dtype=self.dtype)

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        noise = torch.randn_like(state.data) * 0.1
        return LatentState(data=state.data + noise, device=state.device)

    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        t0 = time.perf_counter()
        frames = None
        depth = None
        audio = None

        h = state.data.shape[-2] * _LATENT_FACTOR
        w = state.data.shape[-1] * _LATENT_FACTOR
        if "frames" in modalities:
            frame_tensor = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, device="cpu")
            frames = [frame_tensor.numpy().tobytes()]
        if "depth" in modalities:
            depth = torch.rand(h, w, dtype=torch.float32, device="cpu").numpy().tobytes()
        if "audio" in modalities:
            audio = torch.zeros(8000, dtype=torch.float32, device="cpu").numpy().tobytes()

        return Observation(
            step_index=0,
            generation_time_ms=(time.perf_counter() - t0) * 1000.0,
            frames=frames,
            depth=depth,
            audio=audio,
        )

    def profile_vram(self, config: WorldConfig) -> float:
        lh = config.height // _LATENT_FACTOR
        lw = config.width // _LATENT_FACTOR
        return _LATENT_C * lh * lw * 4 / (1024 * 1024) + 1.0

    def create_initial_state(self, config: WorldConfig, seed: int) -> LatentState:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        lh = config.height // _LATENT_FACTOR
        lw = config.width // _LATENT_FACTOR
        data = torch.randn(1, _LATENT_C, lh, lw, generator=gen, dtype=self.dtype, device="cpu").to(
            self.device
        )
        return LatentState(data=data, device=self.device)
