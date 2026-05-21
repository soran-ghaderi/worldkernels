r"""Reusable mocks/stubs for world models and pipelines."""

from __future__ import annotations

import time
from typing import Any

import torch

from worldkernels.core.action import Action
from worldkernels.core.config import WorldConfig
from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.base import AbstractWorld


class MockWorld(AbstractWorld):
    r"""Minimal CPU AbstractWorld for runtime/engine/session tests.

    Records every method invocation so tests can verify the executor
    actually drove the stages in the right order.
    """

    name = "mock"

    stage_exec_modes = {
        StageType.ENCODE: StageExecMode.SINGLE_SHOT,
        StageType.TRANSITION: StageExecMode.SINGLE_SHOT,
        StageType.DECODE: StageExecMode.SINGLE_SHOT,
    }

    transition_mode = TransitionMode.BIDIRECTIONAL
    supports_streaming = False
    supports_kv_cache = False

    def __init__(self, **kwargs: Any) -> None:
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.calls: list[str] = []
        self.warmup_called: bool = False
        self.last_modalities: list[str] | None = None
        self.init_kwargs: dict[str, Any] = dict(kwargs)

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.calls.append("initialize")

    def warmup(self, config: WorldConfig) -> None:
        self.warmup_called = True
        self.calls.append("warmup")

    def encode_action(self, action: Action) -> torch.Tensor:
        self.calls.append(f"encode_action:{action.action_type}")
        return torch.tensor([1.0, 2.0, 3.0], dtype=self.dtype)

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        self.calls.append("transition")
        new_data = state.data + 1.0 if state.data is not None else torch.zeros(1)
        return LatentState(data=new_data, device=state.device)

    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        self.calls.append(f"decode:{','.join(modalities)}")
        self.last_modalities = list(modalities)
        return Observation(
            step_index=0,
            generation_time_ms=1.0,
            frames=[b"\x00\x00\x00"] if "frames" in modalities else None,
            latent=state.data if "latent" in modalities else None,
            depth=b"\x00" if "depth" in modalities else None,
            audio=b"\x00" if "audio" in modalities else None,
        )

    def estimate_vram_mb(self, config: WorldConfig) -> float:
        return float(config.height * config.width) / 1024.0

    def create_initial_state(self, config: WorldConfig, seed: int) -> LatentState:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        data = torch.randn(1, 4, max(1, config.height // 8), max(1, config.width // 8), generator=gen, dtype=self.dtype)
        return LatentState(data=data, device=self.device)


class SlowMockWorld(MockWorld):
    r"""Variant whose stages sleep so timing fields are populated nonzero."""

    name = "mock-slow"

    def encode_action(self, action: Action) -> torch.Tensor:
        time.sleep(0.001)
        return super().encode_action(action)

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        time.sleep(0.001)
        return super().transition(state, action_encoded)

    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        time.sleep(0.001)
        return super().decode_observation(state, modalities)


def register_mock_world(name: str = "mock", cls: type[AbstractWorld] = MockWorld) -> None:
    r"""Register a mock world into the global worlds registry for tests that
    need the full hub/registry resolution path."""
    from worldkernels.worlds import registry

    registry.register_world(name, cls)


def unregister_world(name: str) -> None:
    from worldkernels.worlds import registry

    registry._REGISTRY.pop(name, None)


class FakeTensor:
    r"""Minimal stand-in exercising LatentState.clone/.to fall-through paths."""

    def __init__(self, value: Any = 0) -> None:
        self.value = value
        self.cloned = False

    def clone(self) -> "FakeTensor":
        new = FakeTensor(self.value)
        new.cloned = True
        return new

    def to(self, device: str) -> "FakeTensor":
        new = FakeTensor(self.value)
        new.value = (device, self.value)
        return new
