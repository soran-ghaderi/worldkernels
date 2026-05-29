r"""Profile bundles + resolution precedence (default < profile < env < cli)."""

from __future__ import annotations

import pytest

from worldkernels.config.profiles import PROFILES, profile_config, resolve_runtime_config


class TestProfiles:
    def test_baseline_turns_optimizations_off(self):
        cfg = profile_config("baseline")
        assert cfg.torch_compile is False
        assert cfg.cuda_graphs is False
        assert cfg.continuous_batching is False
        assert cfg.teacache is False
        assert cfg.attention_backend == "sdpa"

    def test_default_profile_is_defaults(self):
        cfg = profile_config("default")
        assert cfg.torch_compile is True
        assert cfg.continuous_batching is True

    def test_production_enables_quant_and_teacache(self):
        cfg = profile_config("production")
        assert cfg.teacache is True
        assert cfg.quantization == "int8"

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="unknown profile"):
            resolve_runtime_config(profile="nope")

    def test_all_profiles_resolve(self):
        for name in PROFILES:
            cfg, sources = resolve_runtime_config(profile=name)
            assert cfg is not None
            assert isinstance(sources, dict)


class TestPrecedence:
    def test_sources_default(self):
        _, sources = resolve_runtime_config(env={})
        assert sources["torch_compile"] == "default"

    def test_profile_source_attribution(self):
        _, sources = resolve_runtime_config(profile="baseline", env={})
        assert sources["torch_compile"] == "profile:baseline"

    def test_env_overrides_profile(self):
        cfg, sources = resolve_runtime_config(
            profile="baseline", env={"WK_TORCH_COMPILE": "1"}
        )
        assert cfg.torch_compile is True
        assert sources["torch_compile"] == "env:WK_TORCH_COMPILE"

    def test_wk_disable_csv(self):
        cfg, sources = resolve_runtime_config(env={"WK_DISABLE": "teacache,latent_pool"})
        assert cfg.teacache is False
        assert cfg.latent_pool is False
        assert sources["teacache"] == "env:WK_DISABLE"

    def test_wk_enable_csv(self):
        cfg, _ = resolve_runtime_config(env={"WK_ENABLE": "teacache"})
        assert cfg.teacache is True

    def test_enum_env_override(self):
        cfg, sources = resolve_runtime_config(env={"WK_ATTENTION_BACKEND": "sdpa"})
        assert cfg.attention_backend == "sdpa"
        assert sources["attention_backend"] == "env:WK_ATTENTION_BACKEND"

    def test_enum_env_invalid_ignored(self):
        cfg, sources = resolve_runtime_config(env={"WK_DTYPE": "garbage"})
        assert cfg.dtype == "auto"
        assert sources["dtype"] == "default"

    def test_cli_overrides_everything(self):
        cfg, sources = resolve_runtime_config(
            profile="baseline",
            env={"WK_TORCH_COMPILE": "1"},
            cli_overrides={"torch_compile": False},
        )
        assert cfg.torch_compile is False
        assert sources["torch_compile"] == "cli:--torch-compile"

    def test_cli_none_is_ignored(self):
        cfg, _ = resolve_runtime_config(cli_overrides={"teacache": None})
        assert cfg.teacache is False  # default, not overridden
