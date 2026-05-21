r"""Tests for worldkernels/worlds/adapters/dummy/adapter.py."""

from __future__ import annotations

import torch

from worldkernels.core.action import Action
from worldkernels.core.config import WorldConfig
from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.adapters.dummy.adapter import DummyWorld


class TestMetadata:
    def test_name(self):
        assert DummyWorld.name == "dummy"

    def test_stage_modes(self):
        assert DummyWorld.stage_exec_modes[StageType.ENCODE] == StageExecMode.SINGLE_SHOT
        assert DummyWorld.stage_exec_modes[StageType.TRANSITION] == StageExecMode.SINGLE_SHOT
        assert DummyWorld.stage_exec_modes[StageType.DECODE] == StageExecMode.SINGLE_SHOT

    def test_transition_mode(self):
        assert DummyWorld.transition_mode == TransitionMode.BIDIRECTIONAL

    def test_no_streaming(self):
        assert DummyWorld.supports_streaming is False
        assert DummyWorld.supports_kv_cache is False


class TestLifecycle:
    def test_initialize(self):
        w = DummyWorld()
        assert w._initialized is False
        w.initialize("cpu", torch.float32)
        assert w.device == "cpu"
        assert w.dtype == torch.float32
        assert w._initialized is True

    def test_warmup_invokes_create_initial_state(self):
        w = DummyWorld()
        w.initialize("cpu", torch.float32)
        w.warmup(WorldConfig(height=64, width=64))


class TestStages:
    def _world(self) -> DummyWorld:
        w = DummyWorld()
        w.initialize("cpu", torch.float32)
        return w

    def test_encode_action_shape(self):
        w = self._world()
        out = w.encode_action(Action("null"))
        assert out.shape == (1, 4)
        assert out.dtype == torch.float32

    def test_transition_advances_state(self):
        w = self._world()
        cfg = WorldConfig(height=64, width=64)
        state = w.create_initial_state(cfg, seed=0)
        action = w.encode_action(Action("null"))
        new_state = w.transition(state, action)
        assert new_state.device == state.device
        assert not torch.equal(new_state.data, state.data)
        assert new_state.data.shape == state.data.shape

    def test_decode_frames_only(self):
        w = self._world()
        cfg = WorldConfig(height=64, width=64)
        state = w.create_initial_state(cfg, seed=0)
        obs = w.decode_observation(state, ["frames"])
        assert isinstance(obs, Observation)
        assert obs.frames is not None
        assert obs.depth is None
        assert obs.audio is None

    def test_decode_all_modalities(self):
        w = self._world()
        cfg = WorldConfig(height=64, width=64)
        state = w.create_initial_state(cfg, seed=0)
        obs = w.decode_observation(state, ["frames", "depth", "audio"])
        assert obs.frames is not None
        assert obs.depth is not None
        assert obs.audio is not None
        assert obs.generation_time_ms >= 0

    def test_decode_no_modalities(self):
        w = self._world()
        cfg = WorldConfig(height=64, width=64)
        state = w.create_initial_state(cfg, seed=0)
        obs = w.decode_observation(state, [])
        assert obs.frames is None
        assert obs.depth is None
        assert obs.audio is None


class TestEstimateVram:
    def test_positive(self):
        w = DummyWorld()
        cfg = WorldConfig(height=64, width=64)
        v = w.estimate_vram_mb(cfg)
        assert v > 0

    def test_grows_with_resolution(self):
        w = DummyWorld()
        v1 = w.estimate_vram_mb(WorldConfig(height=64, width=64))
        v2 = w.estimate_vram_mb(WorldConfig(height=512, width=512))
        assert v2 > v1


class TestCreateInitialState:
    def test_determinism(self):
        w = DummyWorld()
        w.initialize("cpu", torch.float32)
        s1 = w.create_initial_state(WorldConfig(height=64, width=64), seed=42)
        s2 = w.create_initial_state(WorldConfig(height=64, width=64), seed=42)
        assert torch.equal(s1.data, s2.data)

    def test_different_seeds_diverge(self):
        w = DummyWorld()
        w.initialize("cpu", torch.float32)
        s1 = w.create_initial_state(WorldConfig(height=64, width=64), seed=1)
        s2 = w.create_initial_state(WorldConfig(height=64, width=64), seed=2)
        assert not torch.equal(s1.data, s2.data)

    def test_returns_latent_state(self):
        w = DummyWorld()
        w.initialize("cpu", torch.float32)
        s = w.create_initial_state(WorldConfig(height=64, width=64), seed=0)
        assert isinstance(s, LatentState)
        assert s.device == "cpu"
