r"""World model executor: stage-disaggregated hot path."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from worldkernels.core.observation import Observation
from worldkernels.runtime.connectors import LocalConnector, StageConnector
from worldkernels.runtime.stages import (
    DEFAULT_PIPELINE_STAGES,
    StageConfig,
    StageOutput,
    StageTiming,
    StageType,
)

if TYPE_CHECKING:
    from worldkernels.config import RuntimeConfig
    from worldkernels.core.action import Action
    from worldkernels.core.request import StepRequest
    from worldkernels.core.session import LatentState
    from worldkernels.worlds.base import WorldModel


class Executor:
    r"""Runs world model stages independently or as a unified pipeline.

    Args:
        device: Target device string.
        dtype: Target dtype.
        stage_configs: Per-stage execution configs.
        connector: Inter-stage transport. Defaults to ``LocalConnector``.
        config: Runtime config (component toggles); gates cuda_graphs /
            iteration_batching / latent_pool / kv_cache (wired incrementally).
    """

    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        stage_configs: tuple[StageConfig, ...] | None = None,
        connector: StageConnector | None = None,
        config: "RuntimeConfig | None" = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.config = config
        self.stage_configs = {
            cfg.stage_type: cfg for cfg in (stage_configs or DEFAULT_PIPELINE_STAGES)
        }
        self.connector = connector or LocalConnector()

    def _stage_enabled(self, stage_type: StageType) -> bool:
        cfg = self.stage_configs.get(stage_type)
        return cfg is None or cfg.enabled

    # ---- per-stage execution ---------------------------------------------

    @torch.no_grad()
    def execute_encode(
        self,
        world: WorldModel,
        action: Action,
    ) -> StageOutput:
        r"""Stage 1: encode action to conditioning tensor."""
        t0 = time.perf_counter()
        action_tensor = world.encode_action(action)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return StageOutput(
            stage_type=StageType.ENCODE,
            data=action_tensor,
            timing_ms=elapsed,
        )

    @torch.no_grad()
    def execute_transition(
        self,
        world: WorldModel,
        state: LatentState,
        action_encoded: torch.Tensor,
    ) -> StageOutput:
        r"""Stage 2: advance latent state.

        Treats ``transition()`` as opaque; the world's ``transition_mode``
        determines internal behavior (bidirectional vs causal vs hybrid).
        """
        t0 = time.perf_counter()
        new_state = world.transition(state, action_encoded)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return StageOutput(
            stage_type=StageType.TRANSITION,
            data=new_state,
            timing_ms=elapsed,
            metadata={"transition_mode": world.transition_mode.value},
        )

    @torch.no_grad()
    def execute_decode(
        self,
        world: WorldModel,
        state: LatentState,
        modalities: list[str],
    ) -> StageOutput:
        r"""Stage 3: decode latent state to observation."""
        t0 = time.perf_counter()
        obs = world.decode_observation(state, modalities)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return StageOutput(
            stage_type=StageType.DECODE,
            data=obs,
            timing_ms=elapsed,
        )

    # ---- unified pipeline ------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        world: WorldModel,
        state: LatentState,
        action: Action,
        modalities: list[str],
        step_index: int,
        decode: bool = True,
    ) -> tuple[LatentState, Observation]:
        r"""Execute the full 3-stage pipeline for one simulation step.

        Args:
            world: The world model adapter.
            state: Current latent state.
            action: Action to apply.
            modalities: Observation modalities to decode.
            step_index: Monotonic step counter.
            decode: If False, skip stage 3 and return latent-only observation.

        Returns:
            Tuple of (new_state, observation) with ``stage_timing`` attached.
        """
        timing = StageTiming()
        t_wall = time.perf_counter()

        encode_out = self.execute_encode(world, action)
        timing.encode_action_ms = encode_out.timing_ms

        transition_out = self.execute_transition(
            world,
            state,
            encode_out.data,
        )
        timing.transition_ms = transition_out.timing_ms
        new_state = transition_out.data

        if decode and self._stage_enabled(StageType.DECODE):
            decode_out = self.execute_decode(world, new_state, modalities)
            timing.decode_observation_ms = decode_out.timing_ms
            obs = decode_out.data
        else:
            obs = Observation(step_index=step_index, generation_time_ms=0.0)
            obs.decode_skipped = True

        wall_ms = (time.perf_counter() - t_wall) * 1000.0
        obs.step_index = step_index
        obs.generation_time_ms = wall_ms
        obs.stage_timing = timing

        return new_state, obs

    @torch.no_grad()
    def execute_batched(
        self,
        requests: list[StepRequest],
    ) -> list[tuple[LatentState, Observation]]:
        r"""Execute a batch of step requests.

        Currently iterates one request at a time; true batched forward passes
        over a compatibility group land with the throughput step. The list
        signature is the seam the scheduler dispatches into.
        """
        return [
            self.step(
                world=req.world,
                state=req.state,
                action=req.action,
                modalities=req.modalities,
                step_index=req.step_index,
                decode=req.decode,
            )
            for req in requests
        ]
