r"""Profile bundles + resolution precedence (default < profile < file < env < cli)."""

from __future__ import annotations

import pytest

from worldkernels.config.profiles import (
    PROFILES,
    profile_config,
    resolve_runtime_config,
    split_config_file,
)


@pytest.fixture
def yaml_file(tmp_path):
    def write(text: str):
        path = tmp_path / "wk.yaml"
        path.write_text(text)
        return path

    return write


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
        cfg, sources = resolve_runtime_config(profile="baseline", env={"WK_TORCH_COMPILE": "1"})
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

    def test_unknown_cli_field_raises(self):
        with pytest.raises(ValueError, match="unknown config field"):
            resolve_runtime_config(cli_overrides={"teacachee": True})

    def test_invalid_cli_enum_raises(self):
        with pytest.raises(ValueError, match="invalid value"):
            resolve_runtime_config(cli_overrides={"dtype": "garbage"})

    def test_nested_cli_override(self):
        cfg, sources = resolve_runtime_config(
            cli_overrides={"parallel.tensor_parallel_size": 2}, env={}
        )
        assert cfg.parallel.tensor_parallel_size == 2
        assert sources["parallel.tensor_parallel_size"] == "cli:--tensor-parallel-size"


class TestConfigFileLayer:
    def test_file_overrides_profile(self, yaml_file):
        path = yaml_file("teacache: true\n")
        cfg, sources = resolve_runtime_config(profile="baseline", config_file=path, env={})
        assert cfg.teacache is True
        assert sources["teacache"] == f"config:{path}"

    def test_env_overrides_file(self, yaml_file):
        path = yaml_file("teacache: true\n")
        cfg, sources = resolve_runtime_config(config_file=path, env={"WK_TEACACHE": "0"})
        assert cfg.teacache is False
        assert sources["teacache"] == "env:WK_TEACACHE"

    def test_cli_overrides_env_and_file(self, yaml_file):
        path = yaml_file("attention_backend: flash\n")
        cfg, sources = resolve_runtime_config(
            config_file=path,
            env={"WK_ATTENTION_BACKEND": "auto"},
            cli_overrides={"attention_backend": "sdpa"},
        )
        assert cfg.attention_backend == "sdpa"
        assert sources["attention_backend"] == "cli:--attention-backend"

    def test_full_chain(self, yaml_file):
        path = yaml_file("torch_compile: true\n")
        cfg, sources = resolve_runtime_config(
            profile="baseline",
            config_file=path,
            env={"WK_CUDA_GRAPHS": "1"},
            cli_overrides={"teacache": True},
        )
        assert cfg.torch_compile is True
        assert cfg.cuda_graphs is True
        assert cfg.teacache is True
        assert cfg.continuous_batching is False
        assert sources["torch_compile"] == f"config:{path}"
        assert sources["cuda_graphs"] == "env:WK_CUDA_GRAPHS"
        assert sources["teacache"] == "cli:--teacache"
        assert sources["continuous_batching"] == "profile:baseline"

    def test_nested_mapping_and_dotted_key(self, yaml_file):
        path = yaml_file("parallel:\n  tensor_parallel_size: 2\ncache.block_frames: 8\n")
        cfg, sources = resolve_runtime_config(config_file=path, env={})
        assert cfg.parallel.tensor_parallel_size == 2
        assert cfg.cache.block_frames == 8
        assert sources["parallel.tensor_parallel_size"] == f"config:{path}"
        assert sources["cache.block_frames"] == f"config:{path}"

    def test_frontend_keys_ignored_by_resolver(self, yaml_file):
        path = yaml_file("port: 9000\nteacache: true\n")
        cfg, _ = resolve_runtime_config(config_file=path, env={})
        assert cfg.teacache is True

    def test_invalid_enum_in_file_raises(self, yaml_file):
        path = yaml_file("dtype: garbage\n")
        with pytest.raises(ValueError, match="invalid value"):
            resolve_runtime_config(config_file=path, env={})

    def test_unknown_nested_field_raises(self, yaml_file):
        path = yaml_file("parallel:\n  bogus_degree: 2\n")
        with pytest.raises(ValueError, match="unknown config field"):
            resolve_runtime_config(config_file=path, env={})

    def test_nested_validation_reruns(self, yaml_file):
        path = yaml_file("cache:\n  gpu_memory_fraction: 5.0\n")
        with pytest.raises(ValueError, match="gpu_memory_fraction"):
            resolve_runtime_config(config_file=path, env={})

    def test_non_mapping_file_raises(self, yaml_file):
        path = yaml_file("- a\n- b\n")
        with pytest.raises(ValueError, match="must contain a mapping"):
            resolve_runtime_config(config_file=path, env={})


class TestSplitConfigFile:
    def test_classification(self, yaml_file):
        path = yaml_file(
            "host: 0.0.0.0\nport: 9000\ndevice: cpu\nmax_sessions: 2\n"
            "teacache: true\nparallel:\n  ring_degree: 2\nscheduler.policy: priority\n"
        )
        frontend, engine = split_config_file(path)
        assert frontend == {"host": "0.0.0.0", "port": 9000, "device": "cpu", "max_sessions": 2}
        assert engine == {
            "teacache": True,
            "parallel": {"ring_degree": 2},
            "scheduler.policy": "priority",
        }

    def test_dash_keys_normalized(self, yaml_file):
        path = yaml_file("attention-backend: sdpa\n")
        _, engine = split_config_file(path)
        assert engine == {"attention_backend": "sdpa"}
