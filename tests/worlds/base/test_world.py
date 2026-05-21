r"""Tests for worldkernels/worlds/base/world.py."""

from __future__ import annotations

import pytest
import torch

from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.base.world import AbstractWorld


class TestAbstractWorldContract:
    def test_cannot_instantiate_without_overrides(self):
        with pytest.raises(TypeError):
            AbstractWorld()  # type: ignore[abstract]

    def test_default_class_metadata(self):
        assert AbstractWorld.name == ""
        assert AbstractWorld.default_config is None
        assert AbstractWorld.transition_mode == TransitionMode.BIDIRECTIONAL
        assert AbstractWorld.supports_streaming is False
        assert AbstractWorld.supports_kv_cache is False
        assert AbstractWorld.stage_exec_modes[StageType.ENCODE] == StageExecMode.SINGLE_SHOT
        assert AbstractWorld.stage_exec_modes[StageType.TRANSITION] == StageExecMode.ITERATIVE
        assert AbstractWorld.stage_exec_modes[StageType.DECODE] == StageExecMode.SINGLE_SHOT

    def test_warmup_is_optional_noop(self):
        class W(AbstractWorld):
            def initialize(self, device, dtype): pass
            def encode_action(self, action): return torch.zeros(1)
            def transition(self, state, action_encoded): return state
            def decode_observation(self, state, modalities):
                from worldkernels.core.observation import Observation

                return Observation(step_index=0, generation_time_ms=0.0)
            def estimate_vram_mb(self, config): return 0.0
            def create_initial_state(self, config, seed):
                from worldkernels.core.session import LatentState

                return LatentState()

        w = W()
        from worldkernels.core.config import WorldConfig

        assert w.warmup(WorldConfig()) is None
