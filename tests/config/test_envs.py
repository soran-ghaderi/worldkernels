r"""Tests for worldkernels/envs.py (central WK_* registry)."""

from __future__ import annotations

import pytest

from worldkernels import envs
from worldkernels.config.runtime import ALL_TOGGLE_FIELDS


class TestRegistry:
    def test_core_variables_registered(self):
        for name in ("WK_DISABLE", "WK_ENABLE", "WK_TARGET_DEVICE", "WORLDKERNELS_QUIET"):
            assert name in envs.environment_variables

    def test_per_toggle_variables_registered(self):
        for field in ALL_TOGGLE_FIELDS:
            assert f"WK_{field.upper()}" in envs.environment_variables

    def test_attribute_access_reads_env(self, monkeypatch):
        monkeypatch.setenv("WK_TEACACHE", "1")
        assert envs.WK_TEACACHE == "1"

    def test_attribute_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("WK_DISABLE", raising=False)
        assert envs.WK_DISABLE == ""

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            envs.WK_NOT_A_VAR

    def test_set_wk_vars(self, monkeypatch):
        monkeypatch.setenv("WK_TEACACHE", "1")
        monkeypatch.setenv("WK_DTYPE", "bf16")
        present = envs.set_wk_vars()
        assert present["WK_TEACACHE"] == "1"
        assert present["WK_DTYPE"] == "bf16"

    def test_registry_names_match_resolver(self):
        from worldkernels.config.profiles import _env_overrides

        env = {f"WK_{f.upper()}": "1" for f in ALL_TOGGLE_FIELDS}
        for name, _, var in _env_overrides(env):
            assert var in envs.environment_variables
