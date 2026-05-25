r"""Tests for worldkernels/worlds/adapters/dreamdojo/adapter.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from worldkernels.core.action import Action
from worldkernels.core.config import WorldConfig
from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.models.cosmos_predict2 import (
    CosmosPredict2Latent,
    CosmosPredict2Pipeline,
)
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds import dreamdojo as dd_adapter
from worldkernels.worlds.dreamdojo import (
    CKPT_DIRS,
    EXPERIMENTS,
    DreamDojoWorld,
)

LATENT_CH = CosmosPredict2Pipeline.LATENT_CH
SF = CosmosPredict2Pipeline.SPATIAL_FACTOR


def _make_pipeline_mock(h=480, w=640, frames=5):
    fake_video = torch.zeros(1, 3, frames, h, w)
    p = MagicMock(spec=CosmosPredict2Pipeline)
    p.denoise.return_value = (
        torch.randn(1, LATENT_CH, frames, h // SF, w // SF),
        torch.zeros(3, h, w),
    )
    p.decode.return_value = fake_video
    return p


def _make_state(h=480, w=640, frames=5) -> LatentState:
    return LatentState(
        data=CosmosPredict2Latent(
            latent=torch.randn(1, LATENT_CH, frames, h // SF, w // SF),
            last_frame=torch.zeros(1, 3, h, w),
            text_emb=torch.randn(1, 32, 16),
            neg_text_emb=torch.randn(1, 32, 16),
        ),
        device="cpu",
    )


class TestMetadata:
    def test_name(self):
        assert DreamDojoWorld.name == "dreamdojo"

    def test_stage_modes(self):
        assert DreamDojoWorld.stage_exec_modes[StageType.ENCODE] == StageExecMode.SINGLE_SHOT
        assert DreamDojoWorld.stage_exec_modes[StageType.TRANSITION] == StageExecMode.ITERATIVE
        assert DreamDojoWorld.stage_exec_modes[StageType.DECODE] == StageExecMode.SINGLE_SHOT

    def test_transition_mode(self):
        assert DreamDojoWorld.transition_mode == TransitionMode.BIDIRECTIONAL

    def test_no_streaming(self):
        assert DreamDojoWorld.supports_streaming is False
        assert DreamDojoWorld.supports_kv_cache is False


class TestExperimentMap:
    def test_known_variants(self):
        assert "2b_pretrain" in EXPERIMENTS
        assert "14b_pretrain" in EXPERIMENTS
        assert "2b_gr1" in EXPERIMENTS

    def test_ckpt_dirs_aligned(self):
        for variant in EXPERIMENTS:
            assert variant in CKPT_DIRS


class TestInit:
    def test_defaults(self):
        w = DreamDojoWorld()
        assert w.variant == "2b_pretrain"
        assert w.action_dim == 384
        assert w.chunk_size == 12
        assert w.num_inference_steps == 35
        assert w.guidance_scale == 3.0
        assert w.ckpt_path is None
        assert w._experiment_override is None
        assert w.pipeline is None

    def test_overrides(self):
        w = DreamDojoWorld(
            ckpt_path="/p",
            experiment="custom_exp",
            variant="14b_gr1",
            action_dim=7,
            chunk_size=8,
            num_inference_steps=2,
            guidance_scale=1.5,
            unused_extra="ignored",
        )
        assert w.ckpt_path == "/p"
        assert w._experiment_override == "custom_exp"
        assert w.variant == "14b_gr1"
        assert w.action_dim == 7
        assert w.chunk_size == 8


class TestInitialize:
    def test_with_ckpt_path_skips_download(self, monkeypatch):
        last_kw = {}

        def fake_ctor(**kw):
            last_kw.update(kw)
            p = _make_pipeline_mock()
            p.load = MagicMock()
            return p

        monkeypatch.setattr(dd_adapter, "CosmosPredict2Pipeline", fake_ctor)
        w = DreamDojoWorld(ckpt_path="/fake/ck", variant="2b_pretrain")
        w.initialize("cpu", torch.float32)
        assert last_kw["experiment"] == EXPERIMENTS["2b_pretrain"]
        assert last_kw["config_file"] == dd_adapter.CONFIG_FILE
        w.pipeline.load.assert_called_once_with("cpu", torch.float32, "/fake/ck")

    def test_initialize_downloads_when_no_ckpt(self, monkeypatch):
        download = MagicMock(return_value="/dl/ck")
        monkeypatch.setattr(dd_adapter, "download_dreamdojo_checkpoint", download)

        def fake_ctor(**kw):
            p = _make_pipeline_mock()
            p.load = MagicMock()
            return p

        monkeypatch.setattr(dd_adapter, "CosmosPredict2Pipeline", fake_ctor)
        w = DreamDojoWorld(variant="2b_gr1")
        w.initialize("cpu", torch.float32)
        download.assert_called_once_with(CKPT_DIRS["2b_gr1"])

    def test_experiment_override_wins(self, monkeypatch):
        last_kw = {}

        def fake_ctor(**kw):
            last_kw.update(kw)
            p = _make_pipeline_mock()
            p.load = MagicMock()
            return p

        monkeypatch.setattr(dd_adapter, "CosmosPredict2Pipeline", fake_ctor)
        w = DreamDojoWorld(ckpt_path="/c", experiment="my_exp")
        w.initialize("cpu", torch.float32)
        assert last_kw["experiment"] == "my_exp"

    def test_unknown_variant_falls_back_to_default(self, monkeypatch):
        last_kw = {}

        def fake_ctor(**kw):
            last_kw.update(kw)
            p = _make_pipeline_mock()
            p.load = MagicMock()
            return p

        monkeypatch.setattr(dd_adapter, "CosmosPredict2Pipeline", fake_ctor)
        w = DreamDojoWorld(ckpt_path="/c", variant="not_a_real_variant")
        w.initialize("cpu", torch.float32)
        assert last_kw["experiment"] == dd_adapter.DEFAULT_EXPERIMENT


class TestWarmup:
    def test_no_pipeline_is_noop(self):
        w = DreamDojoWorld()
        w.warmup(WorldConfig())

    def test_delegates_with_null_action(self):
        w = DreamDojoWorld(action_dim=4, chunk_size=3)
        w.device = "cpu"
        w.dtype = torch.float32
        w.pipeline = _make_pipeline_mock()
        w.warmup(WorldConfig(height=64, width=64, frames_per_step=1))
        kw = w.pipeline.warmup.call_args.kwargs
        assert kw["height"] == 64
        assert kw["width"] == 64
        assert kw["frames_per_step"] == 1
        action = kw["extras"]["action"]
        assert action.shape == (1, 3, 4)
        assert action.sum().item() == 0.0


class TestEncodeAction:
    def _world(self, action_dim=7, chunk_size=4):
        w = DreamDojoWorld(action_dim=action_dim, chunk_size=chunk_size)
        w.device = "cpu"
        w.dtype = torch.float32
        return w

    def test_null_action_returns_zeros(self):
        w = self._world()
        a = w.encode_action(Action("null"))
        assert a.shape == (1, 4, 7)
        assert torch.all(a == 0)

    def test_single_step_joints_broadcast(self):
        w = self._world()
        a = w.encode_action(Action("continuous", {"joints": [0.1] * 7}))
        assert a.shape == (1, 4, 7)
        assert torch.allclose(a[0, 0], a[0, 3])

    def test_default_joints_when_payload_missing_key(self):
        w = self._world()
        a = w.encode_action(Action("continuous", {}))
        assert a.shape == (1, 4, 7)
        assert torch.all(a == 0)

    def test_multistep_joints_exact_chunk(self):
        w = self._world(chunk_size=4)
        joints = [[float(i)] * 7 for i in range(4)]
        a = w.encode_action(Action("continuous", {"joints": joints}))
        assert a.shape == (1, 4, 7)
        assert a[0, 0, 0].item() == 0.0
        assert a[0, 3, 0].item() == 3.0

    def test_multistep_joints_padded(self):
        w = self._world(chunk_size=5)
        joints = [[1.0] * 7 for _ in range(3)]
        a = w.encode_action(Action("continuous", {"joints": joints}))
        assert a.shape == (1, 5, 7)
        assert torch.all(a[0, 3:] == 0.0)

    def test_multistep_joints_truncated(self):
        w = self._world(chunk_size=2)
        joints = [[float(i)] * 7 for i in range(10)]
        a = w.encode_action(Action("continuous", {"joints": joints}))
        assert a.shape == (1, 2, 7)
        assert a[0, 1, 0].item() == 1.0


class TestTransition:
    def _world(self):
        w = DreamDojoWorld(action_dim=7, chunk_size=4)
        w.device = "cpu"
        w.dtype = torch.float32
        w.pipeline = _make_pipeline_mock()
        return w

    def test_transition_invokes_denoise_with_action(self):
        w = self._world()
        state = _make_state()
        action = torch.randn(1, 4, 7)
        new_state = w.transition(state, action)
        assert isinstance(new_state, LatentState)
        kw = w.pipeline.denoise.call_args.kwargs
        assert "extras" in kw and "action" in kw["extras"]
        assert torch.equal(kw["extras"]["action"], action)

    def test_transition_empty_action_uses_null_chunk(self):
        w = self._world()
        state = _make_state()
        w.transition(state, torch.empty(0))
        action = w.pipeline.denoise.call_args.kwargs["extras"]["action"]
        assert action.shape == (1, 4, 7)
        assert action.sum().item() == 0.0

    def test_without_pipeline_raises(self):
        w = DreamDojoWorld()
        with pytest.raises(AssertionError):
            w.transition(_make_state(), torch.empty(0))


class TestDecode:
    def _world(self):
        w = DreamDojoWorld()
        w.device = "cpu"
        w.dtype = torch.float32
        w.pipeline = _make_pipeline_mock()
        return w

    def test_decode_frames(self):
        w = self._world()
        w.pipeline.decode.return_value = torch.zeros(1, 3, 2, 64, 64)
        state = _make_state(h=64, w=64, frames=2)
        obs = w.decode_observation(state, ["frames"])
        assert isinstance(obs, Observation)
        assert obs.frames is not None
        assert len(obs.frames) == 2

    def test_decode_latent(self):
        w = self._world()
        state = _make_state(h=64, w=64, frames=2)
        obs = w.decode_observation(state, ["latent"])
        assert obs.latent is state.data.latent
        assert obs.frames is None

    def test_decode_nothing(self):
        w = self._world()
        state = _make_state(h=64, w=64, frames=2)
        obs = w.decode_observation(state, [])
        assert obs.frames is None
        assert obs.latent is None

    def test_without_pipeline_raises(self):
        w = DreamDojoWorld()
        with pytest.raises(AssertionError):
            w.decode_observation(_make_state(), ["latent"])


class TestCreateInitialState:
    def test_delegates_to_pipeline(self):
        w = DreamDojoWorld()
        w.device = "cpu"
        w.pipeline = _make_pipeline_mock()
        fake = CosmosPredict2Latent(
            latent=torch.zeros(1, LATENT_CH, 1, 8, 8),
            last_frame=torch.zeros(1, 3, 64, 64),
            text_emb=torch.zeros(1, 32, 16),
        )
        w.pipeline.create_initial_state.return_value = fake
        cfg = WorldConfig(height=64, width=64, frames_per_step=1, initial_prompt="p")
        state = w.create_initial_state(cfg, seed=11)
        assert state.data is fake
        w.pipeline.create_initial_state.assert_called_once_with(
            prompt="p", initial_image=None, height=64, width=64, frames_per_step=1, seed=11
        )

    def test_without_pipeline_raises(self):
        w = DreamDojoWorld()
        with pytest.raises(AssertionError):
            w.create_initial_state(WorldConfig(), seed=0)


class TestEstimateVram:
    def test_positive(self):
        w = DreamDojoWorld()
        v = w.profile_vram(WorldConfig(height=64, width=64, frames_per_step=1))
        assert v > 0

    def test_grows_with_resolution(self):
        w = DreamDojoWorld()
        s = w.profile_vram(WorldConfig(height=64, width=64, frames_per_step=1))
        l_ = w.profile_vram(WorldConfig(height=512, width=512, frames_per_step=1))
        assert l_ > s
