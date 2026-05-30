r"""RuntimeConfig + SessionOverrides shape and defaults."""

from __future__ import annotations

from worldkernels.config import EngineConfig, RuntimeConfig, SessionOverrides
from worldkernels.config.runtime import (
    ALL_TOGGLE_FIELDS,
    SESSION_OVERRIDE_FIELDS,
    TOGGLE_BOOL_FIELDS,
    TOGGLE_ENUM_FIELDS,
)


class TestRuntimeConfigDefaults:
    def test_default_preserves_current_behavior(self):
        cfg = RuntimeConfig()
        assert cfg.device == "cuda"
        assert cfg.dtype == "auto"
        assert cfg.max_sessions == 4
        assert cfg.offload_idle is True
        assert cfg.torch_compile is True
        assert cfg.continuous_batching is True
        assert cfg.quantization == "none"
        assert cfg.attention_backend == "auto"
        assert cfg.isolation == "auto"

    def test_nested_subconfigs_present(self):
        cfg = RuntimeConfig()
        assert cfg.cache is not None
        assert cfg.scheduler is not None
        assert cfg.parallel is not None

    def test_engineconfig_is_runtimeconfig_alias(self):
        assert EngineConfig is RuntimeConfig
        ec = EngineConfig(device="cpu", dtype="fp32")
        assert isinstance(ec, RuntimeConfig)
        assert ec.device == "cpu"

    def test_toggle_field_registry_consistent(self):
        cfg = RuntimeConfig()
        for f in ALL_TOGGLE_FIELDS:
            assert hasattr(cfg, f), f
        for f in TOGGLE_BOOL_FIELDS:
            assert isinstance(getattr(cfg, f), bool)
        for f, allowed in TOGGLE_ENUM_FIELDS.items():
            assert getattr(cfg, f) in allowed


class TestSessionOverrides:
    def test_defaults_are_none(self):
        ov = SessionOverrides()
        for f in SESSION_OVERRIDE_FIELDS:
            assert getattr(ov, f) is None

    def test_override_subset_is_safe_only(self):
        # bake-time flags must NOT be expressible as session overrides
        for unsafe in ("torch_compile", "cuda_graphs", "dtype", "quantization", "kv_cache_paged"):
            assert unsafe not in SESSION_OVERRIDE_FIELDS
