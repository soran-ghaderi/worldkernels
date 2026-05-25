r"""Tests for worldkernels/core/config.py."""

from __future__ import annotations

from worldkernels.core.config import ServerConfig, WorldConfig


class TestWorldConfig:
    def test_defaults(self):
        cfg = WorldConfig()
        assert cfg.height == 480
        assert cfg.width == 848
        assert cfg.fps == 24
        assert cfg.num_inference_steps == 4
        assert cfg.guidance_scale == 1.0
        assert cfg.frames_per_step == 8
        assert cfg.chunk_overlap == 0
        assert cfg.context_window == 0
        assert cfg.attention_sink_tokens == 0
        assert cfg.initial_prompt is None
        assert cfg.initial_image is None
        assert cfg.max_vram_gb is None
        assert cfg.precision == "bf16"

    def test_overrides(self):
        cfg = WorldConfig(
            height=720,
            width=1280,
            initial_prompt="hello",
            precision="fp16",
            max_vram_gb=12.0,
        )
        assert cfg.height == 720
        assert cfg.width == 1280
        assert cfg.initial_prompt == "hello"
        assert cfg.precision == "fp16"
        assert cfg.max_vram_gb == 12.0


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.max_sessions == 4
        assert cfg.api_key is None
        assert cfg.cors_origins == ["*"]

    def test_cors_origins_independent_per_instance(self):
        a, b = ServerConfig(), ServerConfig()
        a.cors_origins.append("http://x")
        assert b.cors_origins == ["*"]

    def test_overrides(self):
        cfg = ServerConfig(host="127.0.0.1", port=9000, api_key="k", cors_origins=["x"])
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9000
        assert cfg.api_key == "k"
        assert cfg.cors_origins == ["x"]
