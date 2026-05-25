r"""Factories for building common test inputs without per-test boilerplate."""

from __future__ import annotations

from typing import Any

import torch

from worldkernels.core.action import Action
from worldkernels.core.config import WorldConfig
from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState


def make_world_config(**overrides: Any) -> WorldConfig:
    defaults: dict[str, Any] = dict(height=64, width=64, frames_per_step=1)
    defaults.update(overrides)
    return WorldConfig(**defaults)


def make_action(action_type: str = "null", payload: dict[str, Any] | None = None) -> Action:
    return Action(action_type=action_type, payload=payload or {})


def make_observation(**overrides: Any) -> Observation:
    defaults: dict[str, Any] = dict(step_index=0, generation_time_ms=0.0)
    defaults.update(overrides)
    return Observation(**defaults)


def make_latent_state(
    shape: tuple[int, ...] = (1, 4, 8, 8),
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int | None = None,
) -> LatentState:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    data = torch.randn(*shape, generator=gen, dtype=dtype)
    return LatentState(data=data, device=device)
