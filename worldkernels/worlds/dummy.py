r"""DummyWorld — a CPU-safe world model for development and testing.

Returns random noise with no real weights or compute. Reference template for
`InteractiveWorldModel` implementations.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator

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
    supports_iteration_batching = True

    def __init__(self) -> None:
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self._initialized = False

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self._initialized = True
        # tiny "weights" the runtime quantization gate can attach to
        self._quant_target = torch.nn.Linear(8, 8).to(device=device)

    def warmup(self, config: WorldConfig) -> None:
        _ = self.create_initial_state(config, seed=0)

    def encode_action(self, action: Action) -> torch.Tensor:
        return torch.randn(1, _LATENT_C, device=self.device, dtype=self.dtype)

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        pool, cache = _runtime_features()

        if cache is not None:
            cache.should_compute(state.data)

        if pool is not None:
            noise = pool.acquire(state.data.shape, state.data.dtype)
            torch.randn(*state.data.shape, out=noise)
            noise.mul_(0.1)
            new_data = state.data + noise
            pool.release(noise)
        else:
            new_data = state.data + torch.randn_like(state.data) * 0.1

        return LatentState(data=new_data, device=state.device)

    def transition_iter(
        self, state: LatentState, action_encoded: torch.Tensor
    ) -> Iterator[LatentState]:
        r"""Yield 3 intermediate states — the iteration-batching seam.

        Uses the runtime pool + cache via the active `ForwardContext` so the
        same toggles apply on either dispatch path.
        """
        pool, cache = _runtime_features()
        cur = state.data
        for _ in range(3):
            if cache is not None:
                cache.should_compute(cur)
            if pool is not None:
                noise = pool.acquire(cur.shape, cur.dtype)
                torch.randn(*cur.shape, out=noise)
                noise.mul_(0.033)
                cur = cur + noise
                pool.release(noise)
            else:
                cur = cur + torch.randn_like(cur) * 0.033
            yield LatentState(data=cur, device=state.device)

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


def _runtime_features() -> tuple["object | None", "object | None"]:
    r"""Return (pool, cache) from the active ForwardContext, or (None, None)."""
    try:
        from worldkernels.runtime.forward_context import get_forward_context

        ctx = get_forward_context()
    except RuntimeError:
        return None, None
    return getattr(ctx, "pool", None), getattr(ctx, "cache_backend", None)
