r"""World-model taxonomy.

A `WorldModel` is the interactive, stateful, session-bound layer the
engine schedules: ``encode_action -> transition -> decode_observation``.

Two concrete kinds resolve the world-model vs video-generator distinction:

- `InteractiveWorldModel` — a *true* action-conditioned world model
  whose transition is a learned function of the action and causal history.
- `GeneratorWorld` — a one-shot `VideoGenerator`
  presented as a world model via an explicit, swappable
  `RolloutPolicy`. Its rollout is open-loop
  resampling, not a learned transition, and it says so in its metadata.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import torch

from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode

if TYPE_CHECKING:
    from worldkernels.config import WorldConfig
    from worldkernels.core.action import Action
    from worldkernels.core.observation import Observation
    from worldkernels.core.session import LatentState
    from worldkernels.models.base import GenerationResult, VideoGenerator
    from worldkernels.worlds.rollout import RolloutPolicy

__all__ = ["WorldModel", "InteractiveWorldModel", "GeneratorWorld", "AbstractWorld"]

log = logging.getLogger(__name__)


class WorldModel(ABC):
    r"""Interactive, stage-decomposed world model.

    Subclasses implement the three stage methods plus ``profile_vram`` and
    ``create_initial_state``.

    Class metadata:
      - ``transition_mode``: bidirectional / causal / hybrid attention regime.
      - ``supports_streaming``: incremental frame emission.
      - ``supports_kv_cache``: KV caching in the transition stage.
      - ``supports_iteration_batching``: exposes `transition_iter()` so the
        scheduler can join sessions into a batch mid-denoise.
      - ``rollout_kind``: ``"learned_transition"`` for a true world model,
        ``"open_loop_resample"`` for a wrapped generator.
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
    supports_iteration_batching: bool = False
    rollout_kind: str = "learned_transition"

    @abstractmethod
    def initialize(self, device: str, dtype: torch.dtype) -> None:
        r"""Load weights, allocate buffers, move to device."""

    def warmup(self, config: WorldConfig) -> None:
        r"""Optional dummy forward pass to warm caches and JIT-compile."""

    @abstractmethod
    def encode_action(self, action: Action) -> torch.Tensor:
        r"""Convert a typed action into a conditioning tensor."""

    @abstractmethod
    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        r"""Advance the world by one step. Core compute hot path."""

    def transition_iter(
        self,
        state: LatentState,
        action_encoded: torch.Tensor,
    ) -> Iterator[LatentState]:
        r"""Yield intermediate states across the transition's inner loop.

        Optional: implemented only by worlds that set
        ``supports_iteration_batching = True``. Lets the scheduler drive the
        denoise loop step-by-step so sessions can join a batch mid-flight.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support iteration-level batching"
        )

    @abstractmethod
    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        r"""Decode latent state into the requested observation modalities."""

    def profile_vram(self, config: WorldConfig) -> float:
        r"""Estimate per-session VRAM (MB). Subclasses override either this
        or the legacy ``estimate_vram_mb``."""
        return self.estimate_vram_mb(config)

    def estimate_vram_mb(self, config: WorldConfig) -> float:
        return self.profile_vram(config)

    @abstractmethod
    def create_initial_state(self, config: WorldConfig, seed: int) -> LatentState:
        r"""Create the initial latent state for a new session."""


class InteractiveWorldModel(WorldModel):
    r"""A true action-conditioned world model.

    The transition is a learned function of the action and causal history.
    Marker base for genuine simulators (e.g. DreamDojo), distinct from
    `GeneratorWorld`.
    """

    rollout_kind: str = "learned_transition"


@dataclass
class _GeneratorState:
    r"""Per-session state for a `GeneratorWorld`.

    Holds the generator's opaque conditioning plus the most recent latent and
    decoded clip, so ``decode_observation`` needs no recompute.
    """

    conditioning: Any
    latent: torch.Tensor | None = None
    video: torch.Tensor | None = None

    def clone(self) -> "_GeneratorState":
        return _GeneratorState(
            conditioning=self.conditioning.clone(),
            latent=None if self.latent is None else self.latent.clone(),
            video=None if self.video is None else self.video.clone(),
        )

    def to(self, device: Any) -> "_GeneratorState":
        return _GeneratorState(
            conditioning=self.conditioning.to(device),
            latent=None if self.latent is None else self.latent.to(device),
            video=None if self.video is None else self.video.to(device),
        )

    @property
    def nelement(self) -> int:
        n = getattr(self.conditioning, "nelement", 0)
        n = n if isinstance(n, int) else n
        for t in (self.latent, self.video):
            if t is not None:
                n += t.nelement()
        return n

    @property
    def element_size(self) -> int:
        return 2


class GeneratorWorld(WorldModel):
    r"""A one-shot video generator presented as an interactive world model.

    Wraps a `VideoGenerator` and advances it
    with an explicit `RolloutPolicy`. The
    rollout is open-loop resampling — the generator has no learned memory of
    the action history beyond the single carried frame — and the world's
    metadata says so. Actions are text prompts.

    Args:
        generator: Pipeline-registry key of the wrapped video generator.
        rollout: Rollout-policy name (currently ``"last_frame"``).
        num_inference_steps: Denoising steps per generation.
        guidance_scale: Classifier-free guidance scale.
        num_frames: Frames generated per step.
        negative_prompt: Negative prompt held for CFG.
        **generator_kwargs: Forwarded to the generator pipeline's constructor.
    """

    transition_mode = TransitionMode.BIDIRECTIONAL
    supports_streaming = False
    supports_kv_cache = False
    supports_iteration_batching = False
    rollout_kind = "open_loop_resample"

    _ROLLOUTS = {"last_frame": "LastFrameRolloutPolicy"}

    def __init__(
        self,
        *,
        generator: str,
        rollout: str = "last_frame",
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        num_frames: int = 81,
        negative_prompt: str = "",
        **generator_kwargs: Any,
    ) -> None:
        if rollout not in self._ROLLOUTS:
            raise ValueError(
                f"unknown rollout policy {rollout!r}; available: {sorted(self._ROLLOUTS)}"
            )
        self.name = f"generator:{generator}"
        self._generator_key = generator
        self._generator_kwargs = generator_kwargs
        self._rollout_name = rollout
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_frames = num_frames
        self.negative_prompt = negative_prompt
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32
        self.generator: VideoGenerator | None = None
        self.rollout: RolloutPolicy | None = None
        self._warned = False

    def initialize(self, device: str, dtype: torch.dtype) -> None:
        from worldkernels.models import get_pipeline_class
        from worldkernels.worlds import rollout as rollout_mod

        self.device = device
        self.dtype = dtype
        self.rollout = getattr(rollout_mod, self._ROLLOUTS[self._rollout_name])()
        generator_cls = get_pipeline_class(self._generator_key)
        self.generator = generator_cls(**self._generator_kwargs)
        self.generator.load(device, dtype)

    def _warn_once(self) -> None:
        if not self._warned:
            log.warning(
                "'%s' is a video generator presented as a world model via the '%s' "
                "rollout policy (rollout_kind=%s); each step is open-loop resampling, "
                "not a learned state transition.",
                self.name,
                self.rollout.name if self.rollout else self._rollout_name,
                self.rollout_kind,
            )
            self._warned = True

    def encode_action(self, action: Action) -> torch.Tensor:
        prompt = action.payload.get("prompt", "") if action.payload else ""
        if not prompt or self.generator is None:
            return torch.empty(0, device=self.device)
        return self.generator.encode_prompt(prompt)

    def transition(self, state: LatentState, action_encoded: torch.Tensor) -> LatentState:
        from worldkernels.core.session import LatentState as _LatentState

        assert self.generator is not None and self.rollout is not None
        gen_state: _GeneratorState = state.data
        conditioning = gen_state.conditioning
        if action_encoded.numel() > 0:
            conditioning = self.generator.apply_prompt(conditioning, action_encoded)

        import time

        result: GenerationResult = self.generator.generate(
            conditioning,
            num_steps=self.num_inference_steps,
            guidance=self.guidance_scale,
            num_frames=self.num_frames,
            seed=int(time.perf_counter() * 1000) % (2**31),
        )
        next_image = self.rollout.next_image_cond(result.video)
        next_conditioning = self.generator.advance(conditioning, next_image)
        return _LatentState(
            data=_GeneratorState(next_conditioning, result.latent, result.video),
            device=state.device,
        )

    def decode_observation(self, state: LatentState, modalities: list[str]) -> Observation:
        import time

        from worldkernels.core.observation import Observation

        t0 = time.perf_counter()
        gen_state: _GeneratorState = state.data
        frames = None
        latent_out = None

        if "frames" in modalities and gen_state.video is not None:
            video_uint8 = ((gen_state.video + 1.0) * 0.5).clamp(0, 1).mul(255).to(torch.uint8)
            frames = [
                video_uint8[0, :, t].permute(1, 2, 0).cpu().numpy().tobytes()
                for t in range(video_uint8.shape[2])
            ]
        if "latent" in modalities:
            latent_out = gen_state.latent

        return Observation(
            step_index=0,
            generation_time_ms=(time.perf_counter() - t0) * 1000.0,
            frames=frames,
            latent=latent_out,
        )

    def create_initial_state(self, config: WorldConfig, seed: int) -> LatentState:
        from worldkernels.core.session import LatentState as _LatentState

        assert self.generator is not None, "Generator not loaded — call initialize() first"
        self._warn_once()
        conditioning = self.generator.initial_conditioning(
            prompt=config.initial_prompt or "",
            negative_prompt=self.negative_prompt,
            image=config.initial_image,
            height=config.height,
            width=config.width,
            frames_per_step=config.frames_per_step,
            seed=seed,
        )
        return _LatentState(data=_GeneratorState(conditioning), device=self.device)

    def profile_vram(self, config: WorldConfig) -> float:
        if self.generator is None:
            return 1024.0
        return self.generator.profile_vram(
            height=config.height, width=config.width, num_frames=self.num_frames
        )


AbstractWorld = WorldModel
