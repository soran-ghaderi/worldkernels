r"""Abstract interface for world models.

Three-stage pipeline: encode_action -> transition -> decode_observation.
Each stage is independently schedulable, compiled, and batchable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode

if TYPE_CHECKING:
    from worldkernels.core.action import Action
    from worldkernels.core.config import WorldConfig
    from worldkernels.core.observation import Observation
    from worldkernels.core.session import LatentState


class AbstractWorld(ABC):
    r"""Stage-decomposed world model interface.

    Subclasses must implement the three stage methods plus
    ``estimate_vram_mb`` and ``create_initial_state``.

    Class-level metadata to declare:
      - ``stage_exec_modes``: single-shot vs iterative per stage
      - ``transition_mode``: bidirectional / causal / hybrid
      - ``supports_streaming``: incremental frame emission
      - ``supports_kv_cache``: KV caching in transition stage
    """

    name: str = ""
    default_config: WorldConfig | None = None

    stage_exec_modes: dict[StageType, StageExecMode] = {
        StageType.ENCODE: StageExecMode.SINGLE_SHOT,
        StageType.TRANSITION: StageExecMode.ITERATIVE,
        StageType.DECODE: StageExecMode.SINGLE_SHOT,
    }

    transition_mode: TransitionMode = TransitionMode.BIDIRECTIONAL
    supports_streaming: bool = False
    supports_kv_cache: bool = False

    # ---- lifecycle -------------------------------------------------------

    @abstractmethod
    def initialize(self, device: str, dtype: torch.dtype) -> None:
        r"""Load weights, allocate buffers, move to device."""

    def warmup(self, config: WorldConfig) -> None:
        r"""Optional dummy forward pass to warm caches and JIT compile."""

    # ---- stage 1 ---------------------------------------------------------

    @abstractmethod
    def encode_action(self, action: Action) -> torch.Tensor:
        r"""Convert a typed Action into a conditioning tensor."""

    # ---- stage 2 ---------------------------------------------------------

    @abstractmethod
    def transition(
        self,
        state: LatentState,
        action_encoded: torch.Tensor,
    ) -> LatentState:
        r"""Advance the world by one step. Core compute hot path."""

    # ---- stage 3 ---------------------------------------------------------

    @abstractmethod
    def decode_observation(
        self,
        state: LatentState,
        modalities: list[str],
    ) -> Observation:
        r"""Decode latent state into requested observation modalities."""

    # ---- resource estimation ---------------------------------------------

    @abstractmethod
    def estimate_vram_mb(self, config: WorldConfig) -> float:
        r"""Estimate total VRAM (MB) for one session with this config."""

    # ---- initial state ---------------------------------------------------

    @abstractmethod
    def create_initial_state(
        self,
        config: WorldConfig,
        seed: int,
    ) -> LatentState:
        r"""Create the initial latent state for a new session."""
