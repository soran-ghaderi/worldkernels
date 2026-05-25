r"""Tests for worldkernels/worlds/pipelines/cosmos_predict2/pipeline.py.

The cosmos_predict2 external package is not installed in CI. Tests cover
the pure-Python parts (CosmosPredict2Latent dataclass-like behaviour,
configuration constants, VRAM estimator, data-batch shape contract, decode
and warmup short-circuits) and mock the heavy paths."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from worldkernels.models.cosmos_predict2.pipeline import (
    CosmosPredict2Latent,
    CosmosPredict2Pipeline,
    _download_hf_file,
)

LATENT_CH = CosmosPredict2Pipeline.LATENT_CH
SF = CosmosPredict2Pipeline.SPATIAL_FACTOR


def _make_latent(neg: bool = True) -> CosmosPredict2Latent:
    return CosmosPredict2Latent(
        latent=torch.randn(1, LATENT_CH, 2, 8, 8),
        last_frame=torch.zeros(1, 3, 64, 64),
        text_emb=torch.randn(1, 16, 8),
        neg_text_emb=torch.randn(1, 16, 8) if neg else None,
    )


class TestCosmosPredict2Latent:
    def test_init(self):
        cs = _make_latent()
        assert cs.latent.shape == (1, LATENT_CH, 2, 8, 8)
        assert cs.last_frame.shape == (1, 3, 64, 64)
        assert cs.text_emb.shape == (1, 16, 8)
        assert cs.neg_text_emb is not None

    def test_init_without_neg(self):
        cs = _make_latent(neg=False)
        assert cs.neg_text_emb is None

    def test_clone_with_neg(self):
        cs = _make_latent()
        cloned = cs.clone()
        assert torch.equal(cloned.latent, cs.latent)
        assert torch.equal(cloned.text_emb, cs.text_emb)
        assert torch.equal(cloned.neg_text_emb, cs.neg_text_emb)
        cloned.latent.fill_(0)
        assert not torch.equal(cloned.latent, cs.latent)

    def test_clone_without_neg(self):
        cs = _make_latent(neg=False)
        assert cs.clone().neg_text_emb is None

    def test_to_with_neg(self):
        cs = _make_latent()
        m = cs.to("cpu")
        assert m.latent.device.type == "cpu"
        assert m.neg_text_emb.device.type == "cpu"

    def test_to_without_neg(self):
        cs = _make_latent(neg=False)
        assert cs.to("cpu").neg_text_emb is None

    def test_nelement_with_neg(self):
        cs = _make_latent()
        expected = (
            cs.latent.nelement()
            + cs.last_frame.nelement()
            + cs.text_emb.nelement()
            + cs.neg_text_emb.nelement()
        )
        assert cs.nelement == expected

    def test_nelement_without_neg(self):
        cs = _make_latent(neg=False)
        expected = cs.latent.nelement() + cs.last_frame.nelement() + cs.text_emb.nelement()
        assert cs.nelement == expected

    def test_element_size_matches_latent(self):
        cs = _make_latent()
        assert cs.element_size == cs.latent.element_size()


class TestPipelineConstruction:
    def test_init_state(self):
        p = CosmosPredict2Pipeline(experiment="exp", config_file="cfg")
        assert p.experiment == "exp"
        assert p.config_file == "cfg"
        assert p.device == "cpu"
        assert p.dtype == torch.float32
        assert p._model is None
        assert p._neg_text_emb is None
        assert p._loaded is False

    def test_is_loaded_property(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        assert p.is_loaded is False
        p._loaded = True
        assert p.is_loaded is True

    def test_neg_text_emb_property(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        assert p.neg_text_emb is None
        p._neg_text_emb = torch.zeros(1)
        assert p.neg_text_emb is not None

    def test_class_constants(self):
        assert CosmosPredict2Pipeline.LATENT_CH == 16
        assert CosmosPredict2Pipeline.SPATIAL_FACTOR == 8
        assert CosmosPredict2Pipeline.HF_TOKENIZER_REPO == "nvidia/Cosmos-Predict2.5-2B"
        assert "ugly" in CosmosPredict2Pipeline.NEGATIVE_PROMPT


class TestEstimateLatentVram:
    def test_positive(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        v = p.estimate_latent_vram_mb(height=64, width=64, frames_per_step=1)
        assert v > 0

    def test_scales_with_resolution(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        small = p.estimate_latent_vram_mb(height=64, width=64, frames_per_step=1)
        large = p.estimate_latent_vram_mb(height=512, width=512, frames_per_step=1)
        assert large > small

    def test_scales_with_frames(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        s = p.estimate_latent_vram_mb(height=64, width=64, frames_per_step=1)
        l_ = p.estimate_latent_vram_mb(height=64, width=64, frames_per_step=8)
        assert l_ > s


class TestDenoiseAndDecode:
    def test_decode_returns_model_output(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p._model = MagicMock()
        out = torch.ones(1, 3, 1, 8, 8)
        p._model.decode.return_value = out
        result = p.decode(torch.zeros(1, LATENT_CH, 1, 1, 1))
        assert torch.equal(result, out)
        p._model.decode.assert_called_once()

    def test_denoise_drives_model(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = MagicMock()
        p._model.tokenizer.get_pixel_num_frames.return_value = 5
        p._model.config.state_t = 5
        sample_latent = torch.randn(1, LATENT_CH, 2, 8, 8)
        sample_video = torch.zeros(1, 3, 5, 64, 64)
        p._model.generate_samples_from_batch.return_value = sample_latent
        p._model.decode.return_value = sample_video

        cs = _make_latent()
        new_latent, video = p.denoise(cs, num_steps=3, guidance=1.2, seed=99)

        assert new_latent is sample_latent
        assert video is sample_video
        kwargs = p._model.generate_samples_from_batch.call_args.kwargs
        assert kwargs["guidance"] == 1.2
        assert kwargs["seed"] == 99
        assert kwargs["num_steps"] == 3
        assert kwargs["is_negative_prompt"] is True

    def test_denoise_without_neg(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = MagicMock()
        p._model.tokenizer.get_pixel_num_frames.return_value = 5
        p._model.config.state_t = 5
        p._model.generate_samples_from_batch.return_value = torch.randn(1, LATENT_CH, 2, 8, 8)
        p._model.decode.return_value = torch.zeros(1, 3, 5, 64, 64)
        cs = _make_latent(neg=False)
        p.denoise(cs)
        assert p._model.generate_samples_from_batch.call_args.kwargs["is_negative_prompt"] is False

    def test_denoise_extras_merged_into_batch(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = MagicMock()
        p._model.tokenizer.get_pixel_num_frames.return_value = 5
        p._model.config.state_t = 5
        p._model.generate_samples_from_batch.return_value = torch.randn(1, LATENT_CH, 2, 8, 8)
        p._model.decode.return_value = torch.zeros(1, 3, 5, 64, 64)
        cs = _make_latent()
        my_action = torch.ones(1, 4, 7)
        p.denoise(cs, extras={"action": my_action})
        data_batch = p._model.generate_samples_from_batch.call_args.args[0]
        assert "action" in data_batch
        assert torch.equal(data_batch["action"], my_action)


class TestBuildDataBatch:
    def _pipeline(self, num_frames=5):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = MagicMock()
        p._model.tokenizer.get_pixel_num_frames.return_value = num_frames
        p._model.config.state_t = num_frames
        return p

    def test_basic_shapes(self):
        p = self._pipeline()
        cs = _make_latent()
        batch = p._build_data_batch(cs)
        assert batch["dataset_name"] == "video_data"
        assert batch["video"].dtype == torch.uint8
        assert batch["video"].shape == (1, 3, 5, 64, 64)
        assert batch["padding_mask"].shape == (1, 1, 64, 64)
        assert batch["num_conditional_frames"] == 0
        assert torch.equal(batch["t5_text_embeddings"], cs.text_emb)
        assert "neg_t5_text_embeddings" in batch

    def test_with_conditioning_frame(self):
        p = self._pipeline()
        cs = _make_latent()
        cs.last_frame[0, :, 0, 0] = 0.5
        batch = p._build_data_batch(cs)
        assert batch["num_conditional_frames"] == 1

    def test_3d_last_frame_unsqueeze(self):
        p = self._pipeline()
        cs = _make_latent()
        cs.last_frame = cs.last_frame[0]
        batch = p._build_data_batch(cs)
        assert batch["video"].shape == (1, 3, 5, 64, 64)

    def test_no_neg(self):
        p = self._pipeline()
        cs = _make_latent(neg=False)
        batch = p._build_data_batch(cs)
        assert "neg_t5_text_embeddings" not in batch

    def test_extras_added(self):
        p = self._pipeline()
        cs = _make_latent()
        batch = p._build_data_batch(cs, extras={"x": torch.ones(1)})
        assert torch.equal(batch["x"], torch.ones(1))


class TestWarmupShortCircuit:
    def test_unloaded_warmup_is_noop(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.warmup(height=64, width=64, frames_per_step=1)

    def test_loaded_warmup_runs_denoise(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._loaded = True
        p._model = MagicMock()
        p._model.tokenizer.get_latent_num_frames.return_value = 1
        p._model.tokenizer.get_pixel_num_frames.return_value = 5
        p._model.config.state_t = 5
        p._model.generate_samples_from_batch.return_value = torch.zeros(1, LATENT_CH, 1, 8, 8)
        p._model.decode.return_value = torch.zeros(1, 3, 5, 64, 64)
        p.encode_text = MagicMock(return_value=torch.zeros(1, 4, 4))
        p.warmup(height=64, width=64, frames_per_step=1)
        p._model.generate_samples_from_batch.assert_called_once()


class TestLoadShortCircuit:
    def test_load_invokes_ensure_and_load_model(self, monkeypatch):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        monkeypatch.setattr(
            "worldkernels.models.cosmos_predict2.deps.ensure_cosmos_predict2",
            lambda: None,
        )
        monkeypatch.setattr(p, "_load_model", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(p, "encode_text", lambda prompt: torch.zeros(1, 4, 4))
        monkeypatch.setattr(
            "torch.cuda.memory_allocated", lambda *a, **kw: 0, raising=False
        )
        p.load("cpu", torch.float32, "/fake/ckpt")
        assert p.is_loaded is True
        assert p._neg_text_emb is not None
        assert p.device == "cpu"
        assert p.dtype == torch.float32


class TestCreateInitialLatent:
    def test_shapes(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = MagicMock()
        p._model.tokenizer.get_latent_num_frames.return_value = 3
        latent, last_frame = p.create_initial_latent(height=64, width=64, frames_per_step=1, seed=7)
        assert latent.shape == (1, LATENT_CH, 3, 8, 8)
        assert last_frame.shape == (1, 3, 64, 64)


class TestCreateInitialState:
    def test_with_initial_image_uses_encode_image(self):
        pytest.importorskip("torchvision")
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = MagicMock()
        p._model.tokenizer.get_latent_num_frames.return_value = 1
        p._model.encode.return_value = torch.randn(1, LATENT_CH, 1, 8, 8)

        p.encode_text = MagicMock(return_value=torch.zeros(1, 4, 4))
        img = torch.zeros(3, 64, 64)
        state = p.create_initial_state(
            prompt="x", initial_image=img, height=64, width=64, frames_per_step=1, seed=0
        )
        assert isinstance(state, CosmosPredict2Latent)
        p.encode_text.assert_called_once_with("x")

    def test_without_initial_image_uses_random(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = MagicMock()
        p._model.tokenizer.get_latent_num_frames.return_value = 1
        p.encode_text = MagicMock(return_value=torch.zeros(1, 4, 4))

        state = p.create_initial_state(
            prompt="y", initial_image=None, height=64, width=64, frames_per_step=1, seed=0
        )
        assert isinstance(state, CosmosPredict2Latent)
        assert state.latent.shape == (1, LATENT_CH, 1, 8, 8)


@pytest.mark.skipif(
    pytest.importorskip.__module__ and __import__("importlib").util.find_spec("torchvision") is None,  # noqa: E501
    reason="torchvision required by encode_image",
)
class TestEncodeImage:
    def _pipeline(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = MagicMock()
        p._model.encode.return_value = torch.randn(1, LATENT_CH, 1, 8, 8)
        return p

    def test_with_tensor_chw(self):
        p = self._pipeline()
        img = torch.zeros(3, 64, 64)
        last, latent = p.encode_image(img, height=64, width=64, frames_per_step=1)
        assert last.shape == (1, 3, 64, 64)
        assert latent.shape == (1, LATENT_CH, 1, 8, 8)

    def test_with_tensor_hwc_permuted(self):
        p = self._pipeline()
        img = torch.zeros(64, 64, 3)
        last, _ = p.encode_image(img, height=64, width=64, frames_per_step=1)
        assert last.shape == (1, 3, 64, 64)

    def test_with_4d_tensor_kept(self):
        p = self._pipeline()
        img = torch.zeros(1, 3, 64, 64)
        last, _ = p.encode_image(img, height=64, width=64, frames_per_step=1)
        assert last.shape == (1, 3, 64, 64)

    def test_with_path_uses_pillow(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image

        p = self._pipeline()
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        path = tmp_path / "in.png"
        img.save(path)
        last, _ = p.encode_image(str(path), height=64, width=64, frames_per_step=1)
        assert last.shape == (1, 3, 64, 64)


class TestEncodeTextRouting:
    def test_uses_model_text_encoder_when_available(self):
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        text_encoder = MagicMock()
        text_encoder.compute_text_embeddings_online.return_value = torch.ones(1, 4, 8, dtype=torch.float32)  # noqa: E501
        p._model = MagicMock()
        p._model.text_encoder = text_encoder
        out = p.encode_text("hello")
        assert out.shape == (1, 4, 8)
        text_encoder.compute_text_embeddings_online.assert_called_once()

    def test_fallback_to_t5_helper(self, monkeypatch):
        r"""When the model has no text_encoder, fall back to the cosmos t5 helper."""
        import sys
        import types

        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._model = None

        cosmos = types.ModuleType("cosmos_predict2")
        src = types.ModuleType("cosmos_predict2._src")
        predict2 = types.ModuleType("cosmos_predict2._src.predict2")
        inference = types.ModuleType("cosmos_predict2._src.predict2.inference")
        get_t5_emb_mod = types.ModuleType(
            "cosmos_predict2._src.predict2.inference.get_t5_emb"
        )

        def fake_get_text_embedding(prompt):
            assert prompt == "hello"
            return torch.ones(1, 4, 8, dtype=torch.float32)

        get_t5_emb_mod.get_text_embedding = fake_get_text_embedding
        cosmos._src = src
        src.predict2 = predict2
        predict2.inference = inference
        inference.get_t5_emb = get_t5_emb_mod

        monkeypatch.setitem(sys.modules, "cosmos_predict2", cosmos)
        monkeypatch.setitem(sys.modules, "cosmos_predict2._src", src)
        monkeypatch.setitem(sys.modules, "cosmos_predict2._src.predict2", predict2)
        monkeypatch.setitem(
            sys.modules, "cosmos_predict2._src.predict2.inference", inference
        )
        monkeypatch.setitem(
            sys.modules,
            "cosmos_predict2._src.predict2.inference.get_t5_emb",
            get_t5_emb_mod,
        )

        out = p.encode_text("hello")
        assert out.shape == (1, 4, 8)


class TestStubTextEncoder:
    r"""Text-encoder-free capture path (WK_STUB_TEXT_ENCODER)."""

    def _pipeline(self, dim: int = 16) -> CosmosPredict2Pipeline:
        p = CosmosPredict2Pipeline(experiment="e", config_file="c")
        p.device = "cpu"
        p.dtype = torch.float32
        p._stub_text_emb_dim = dim
        return p

    def test_stub_emb_shape_and_dtype(self):
        emb = self._pipeline(dim=32)._stub_text_emb("a robot picks up a block")
        assert emb.shape == (1, 16, 32)
        assert emb.dtype == torch.float32

    def test_stub_emb_deterministic(self):
        p = self._pipeline()
        assert torch.equal(p._stub_text_emb("hello"), p._stub_text_emb("hello"))

    def test_stub_emb_prompt_sensitive(self):
        p = self._pipeline()
        assert not torch.equal(p._stub_text_emb("hello"), p._stub_text_emb("world"))

    def test_encode_text_routes_to_stub(self, monkeypatch):
        monkeypatch.setenv("WK_STUB_TEXT_ENCODER", "1")
        p = self._pipeline(dim=12)
        p._model = MagicMock()
        emb = p.encode_text("some prompt")
        assert emb.shape == (1, 16, 12)
        p._model.text_encoder.compute_text_embeddings_online.assert_not_called()

    def test_encode_text_skips_stub_when_unset(self, monkeypatch):
        monkeypatch.delenv("WK_STUB_TEXT_ENCODER", raising=False)
        p = self._pipeline()
        text_encoder = MagicMock()
        text_encoder.compute_text_embeddings_online.return_value = torch.ones(
            1, 4, 8, dtype=torch.float32
        )
        p._model = MagicMock()
        p._model.text_encoder = text_encoder
        out = p.encode_text("hello")
        assert out.shape == (1, 4, 8)


class TestDownloadHelper:
    def test_calls_hf_hub_download(self, monkeypatch):

        fake = MagicMock(return_value="/path")
        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake, raising=False)
        assert _download_hf_file("repo", "file.bin") == "/path"
        fake.assert_called_once_with(repo_id="repo", filename="file.bin", repo_type="model")
