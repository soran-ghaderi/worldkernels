r"""Tests for worldkernels/cli/commands/model.py."""

from __future__ import annotations

import pytest

from worldkernels.cli.model import run_inspect


class TestRunInspect:
    def test_known_model(self, capsys):
        run_inspect("dummy", device="cpu")
        out = capsys.readouterr().out
        assert "Model: dummy" in out
        assert "Class: " in out
        assert "Transition mode" in out
        assert "Stage execution modes" in out

    def test_config_json_used(self, capsys):
        run_inspect("dummy", device="cpu", config_json='{"height": 64, "width": 64}')
        out = capsys.readouterr().out
        assert "Model: dummy" in out

    def test_unknown_model_fails_lookup(self):
        with pytest.raises(KeyError):
            run_inspect("xyz_unknown_model", device="cpu")

    def test_with_default_config_prints_config(self, monkeypatch, capsys):
        from tests._helpers.mocks import MockWorld
        from worldkernels.core.config import WorldConfig
        from worldkernels.worlds import registry as reg

        class _WithCfg(MockWorld):
            default_config = WorldConfig(height=16, width=16)

        reg.register_world("_pytest_with_cfg", _WithCfg)
        try:
            run_inspect("_pytest_with_cfg", device="cpu")
            out = capsys.readouterr().out
            assert "Default config:" in out
            assert "height: 16" in out
        finally:
            reg._REGISTRY.pop("_pytest_with_cfg", None)

    def test_vram_estimate_failure_swallowed(self, capsys):
        from tests._helpers.mocks import MockWorld
        from worldkernels.worlds import registry as reg

        class _BadInit(MockWorld):
            def initialize(self, device, dtype):
                raise RuntimeError("nope")

        reg.register_world("_pytest_badinit", _BadInit)
        try:
            run_inspect("_pytest_badinit", device="cpu")
            out = capsys.readouterr().out
            assert "Model: _pytest_badinit" in out
            assert "VRAM estimate" not in out
        finally:
            reg._REGISTRY.pop("_pytest_badinit", None)
