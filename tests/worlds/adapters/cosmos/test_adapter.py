r"""Tests for worldkernels/worlds/adapters/cosmos/adapter.py.

The pipeline is mocked throughout — we only verify the adapter wiring
(delegation to the pipeline, modality routing, error paths)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from worldkernels.core.action import Action
from worldkernels.core.config import WorldConfig
from worldkernels.core.observation import Observation
from worldkernels.core.session import LatentState
from worldkernels.runtime.stages import StageExecMode, StageType, TransitionMode
from worldkernels.worlds.adapters.cosmos import adapter as cosmos_adapter
from worldkernels.worlds.adapters.cosmos.adapter import CosmosPredict2World
from worldkernels.worlds.pipelines.cosmos_predict2 import (
    CosmosPredict2Latent,
    CosmosPredict2Pipeline,
)

LATENT_CH = CosmosPredict2Pipeline.LATENT_CH
SF = CosmosPredict2Pipeline.SPATIAL_FACTOR


def _make_state(h: int = 480, w: int = 640, frames: int = 5) -> LatentState:
    latent = torch.randn(1, LATENT_CH, frames, h // SF, w // SF)
    return LatentState(
        data=CosmosPredict2Latent(
            latent=latent,
            last_frame=torch.zeros(1, 3, h, w),
            text_emb=torch.randn(1, 32, 16),
            neg_text_emb=torch.randn(1, 32, 16),
        ),
        device="cpu",
    )


def _make_pipeline_mock(h=480, w=640, frames=5):
    fake_video = torch.zeros(1, 3, frames, h, w)
    pipeline = MagicMock(spec=CosmosPredict2Pipeline)
    pipeline.denoise.return_value = (
        torch.randn(1, LATENT_CH, frames, h // SF, w // SF),
        torch.zeros(3, h, w),
    )
    pipeline.decode.return_value = fake_video
    pipeline.encode_text.return_value = torch.randn(1, 32, 16)
    return pipeline


class TestMetadata:
    def test_name(self):
        assert CosmosPredict2World.name == "cosmos_predict2"

    def test_stage_modes(self):
        assert CosmosPredict2World.stage_exec_modes[StageType.ENCODE] == StageExecMode.SINGLE_SHOT
        assert CosmosPredict2World.stage_exec_modes[StageType.TRANSITION] == StageExecMode.ITERATIVE
        assert CosmosPredict2World.stage_exec_modes[StageType.DECODE] == StageExecMode.SINGLE_SHOT

    def test_attention_mode(self):
        assert CosmosPredict2World.transition_mode == TransitionMode.BIDIRECTIONAL
        assert CosmosPredict2World.supports_streaming is False
        assert CosmosPredict2World.supports_kv_cache is False


class TestInit:
    def test_defaults(self):
        w = CosmosPredict2World()
        assert w.ckpt_path is None
        assert w._experiment_override is None
        assert w.num_inference_steps == 35
        assert w.guidance_scale == 7.0
        assert w.pipeline is None

    def test_overrides(self):
        w = CosmosPredict2World(
            ckpt_path="/p",
            experiment="e",
            num_inference_steps=2,
            guidance_scale=1.5,
            extra="ignored",
        )
        assert w.ckpt_path == "/p"
        assert w._experiment_override == "e"
        assert w.num_inference_steps == 2
        assert w.guidance_scale == 1.5


class TestInitialize:
    def test_with_ckpt_path_skips_download(self, monkeypatch):
        constructed = {}

        def fake_pipeline_ctor(*, experiment, config_file):
            constructed["experiment"] = experiment
            constructed["config_file"] = config_file
            p = _make_pipeline_mock()
            p.load = MagicMock()
            return p

        monkeypatch.setattr(cosmos_adapter, "CosmosPredict2Pipeline", fake_pipeline_ctor)
        w = CosmosPredict2World(ckpt_path="/fake/ckpt", experiment="x")
        w.initialize("cpu", torch.float32)
        assert constructed == {"experiment": "x", "config_file": cosmos_adapter.CONFIG_FILE}
        assert w.device == "cpu"
        assert w.dtype == torch.float32
        w.pipeline.load.assert_called_once_with("cpu", torch.float32, "/fake/ckpt")

    def test_initialize_downloads_when_no_ckpt(self, monkeypatch):
        download = MagicMock(return_value="/downloaded/ckpt")
        monkeypatch.setattr(
            "worldkernels.worlds.pipelines.cosmos_predict2.pipeline._download_hf_file",
            download,
        )
        monkeypatch.setattr(
            cosmos_adapter,
            "CosmosPredict2Pipeline",
            lambda **kw: _attached_load(_make_pipeline_mock()),
        )
        w = CosmosPredict2World()
        w.initialize("cpu", torch.float32)
        download.assert_called_once_with(cosmos_adapter.HF_REPO, cosmos_adapter.HF_CKPT_FILE)

    def test_initialize_falls_back_to_dreamdojo_on_download_failure(self, monkeypatch):
        monkeypatch.setattr(
            "worldkernels.worlds.pipelines.cosmos_predict2.pipeline._download_hf_file",
            MagicMock(side_effect=RuntimeError("HF gate")),
        )
        fb_dl = MagicMock(return_value="/fallback/ckpt")
        monkeypatch.setattr(
            "worldkernels.worlds.adapters.dreamdojo.checkpoint.download_dreamdojo_checkpoint",
            fb_dl,
        )
        last_kw = {}

        def fake_ctor(**kw):
            last_kw.update(kw)
            return _attached_load(_make_pipeline_mock())

        monkeypatch.setattr(cosmos_adapter, "CosmosPredict2Pipeline", fake_ctor)
        w = CosmosPredict2World()
        w.initialize("cpu", torch.float32)
        fb_dl.assert_called_once()
        assert last_kw["experiment"] == cosmos_adapter.FALLBACK_EXPERIMENT
        assert last_kw["config_file"] == cosmos_adapter.FALLBACK_CONFIG


def _attached_load(pipeline):
    pipeline.load = MagicMock()
    return pipeline


class TestWarmup:
    def test_no_pipeline_is_noop(self):
        w = CosmosPredict2World()
        w.warmup(WorldConfig())

    def test_delegates_to_pipeline(self):
        w = CosmosPredict2World()
        w.pipeline = _make_pipeline_mock()
        w.warmup(WorldConfig(height=128, width=128, frames_per_step=2))
        w.pipeline.warmup.assert_called_once_with(height=128, width=128, frames_per_step=2)


class TestEncodeAction:
    def test_no_prompt_returns_empty(self):
        w = CosmosPredict2World()
        w.pipeline = _make_pipeline_mock()
        result = w.encode_action(Action("text", {}))
        assert result.numel() == 0

    def test_pipeline_none_returns_empty(self):
        w = CosmosPredict2World()
        result = w.encode_action(Action("text", {"prompt": "hi"}))
        assert result.numel() == 0

    def test_prompt_invokes_encode_text(self):
        w = CosmosPredict2World()
        w.pipeline = _make_pipeline_mock()
        out = w.encode_action(Action("text", {"prompt": "go"}))
        w.pipeline.encode_text.assert_called_once_with("go")
        assert out is w.pipeline.encode_text.return_value


class TestTransitionAndDecode:
    def _build(self):
        w = CosmosPredict2World()
        w.device = "cpu"
        w.dtype = torch.float32
        w.pipeline = _make_pipeline_mock()
        return w

    def test_transition_invokes_denoise(self):
        w = self._build()
        state = _make_state()
        action = torch.randn(1, 32, 16)
        new_state = w.transition(state, action)
        assert isinstance(new_state, LatentState)
        w.pipeline.denoise.assert_called_once()
        passed_state = w.pipeline.denoise.call_args.args[0]
        assert torch.equal(passed_state.text_emb, action)

    def test_transition_uses_existing_text_emb_when_action_empty(self):
        w = self._build()
        state = _make_state()
        empty_action = torch.empty(0)
        w.transition(state, empty_action)
        passed_state = w.pipeline.denoise.call_args.args[0]
        assert torch.equal(passed_state.text_emb, state.data.text_emb)

    def test_transition_without_pipeline_asserts(self):
        w = CosmosPredict2World()
        with pytest.raises(AssertionError):
            w.transition(_make_state(), torch.zeros(0))

    def test_decode_observation_frames(self):
        w = self._build()
        w.pipeline.decode.return_value = torch.zeros(1, 3, 2, 64, 64)
        state = _make_state(h=64, w=64, frames=2)
        obs = w.decode_observation(state, ["frames"])
        assert isinstance(obs, Observation)
        assert obs.frames is not None
        assert obs.latent is None
        assert len(obs.frames) == 2

    def test_decode_observation_latent(self):
        w = self._build()
        state = _make_state(h=64, w=64, frames=2)
        obs = w.decode_observation(state, ["latent"])
        assert obs.latent is state.data.latent
        assert obs.frames is None

    def test_decode_observation_no_modalities(self):
        w = self._build()
        state = _make_state(h=64, w=64, frames=1)
        obs = w.decode_observation(state, [])
        assert obs.frames is None
        assert obs.latent is None

    def test_decode_without_pipeline_asserts(self):
        w = CosmosPredict2World()
        with pytest.raises(AssertionError):
            w.decode_observation(_make_state(), ["latent"])


class TestCreateInitialState:
    def test_delegates_to_pipeline(self):
        w = CosmosPredict2World()
        w.device = "cpu"
        w.pipeline = _make_pipeline_mock()
        fake = CosmosPredict2Latent(
            latent=torch.zeros(1, LATENT_CH, 1, 8, 8),
            last_frame=torch.zeros(1, 3, 64, 64),
            text_emb=torch.zeros(1, 32, 16),
        )
        w.pipeline.create_initial_state.return_value = fake
        cfg = WorldConfig(height=64, width=64, frames_per_step=1, initial_prompt="hi")
        state = w.create_initial_state(cfg, seed=7)
        assert isinstance(state, LatentState)
        assert state.data is fake
        w.pipeline.create_initial_state.assert_called_once_with(
            prompt="hi",
            initial_image=None,
            height=64,
            width=64,
            frames_per_step=1,
            seed=7,
        )

    def test_create_initial_without_pipeline_asserts(self):
        w = CosmosPredict2World()
        with pytest.raises(AssertionError):
            w.create_initial_state(WorldConfig(), seed=0)

    def test_empty_prompt_passes_empty_string(self):
        w = CosmosPredict2World()
        w.device = "cpu"
        w.pipeline = _make_pipeline_mock()
        w.pipeline.create_initial_state.return_value = CosmosPredict2Latent(
            latent=torch.zeros(1, LATENT_CH, 1, 8, 8),
            last_frame=torch.zeros(1, 3, 64, 64),
            text_emb=torch.zeros(1, 32, 16),
        )
        w.create_initial_state(WorldConfig(height=64, width=64, frames_per_step=1), seed=0)
        assert w.pipeline.create_initial_state.call_args.kwargs["prompt"] == ""


class TestEstimateVram:
    def test_positive(self):
        w = CosmosPredict2World()
        v = w.estimate_vram_mb(WorldConfig(height=64, width=64, frames_per_step=1))
        assert v > 0

    def test_grows_with_resolution(self):
        w = CosmosPredict2World()
        v_small = w.estimate_vram_mb(WorldConfig(height=64, width=64, frames_per_step=1))
        v_large = w.estimate_vram_mb(WorldConfig(height=512, width=512, frames_per_step=1))
        assert v_large > v_small
