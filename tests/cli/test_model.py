r"""Tests for worldkernels/cli/model.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from worldkernels.cli.model import (
    run_download,
    run_export,
    run_inspect,
    run_list,
    run_remove,
)


class TestRunList:
    def test_default_lists_hub_entries(self, capsys):
        run_list()
        out = capsys.readouterr().out
        assert "dummy" in out
        assert "dreamdojo" in out

    def test_verbose_includes_adapters(self, capsys):
        run_list(verbose=True)
        out = capsys.readouterr().out
        assert "adapter=" in out
        assert "Adapters" in out


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


class TestRunDownload:
    def test_uses_card_repo(self, monkeypatch, capsys):
        sd = MagicMock(return_value="/path/to/snap")
        monkeypatch.setattr("huggingface_hub.snapshot_download", sd, raising=False)
        run_download("dummy")
        sd.assert_called_once()
        assert sd.call_args.args[0] == "dummy"

    def test_falls_back_to_model_id_when_no_card(self, monkeypatch):
        sd = MagicMock(return_value="/p")
        monkeypatch.setattr("huggingface_hub.snapshot_download", sd, raising=False)
        run_download("owner/no_card_repo", revision="r", cache_dir="/c")
        sd.assert_called_once_with("owner/no_card_repo", revision="r", cache_dir="/c")


class TestRunRemove:
    def test_removes_matching(self, monkeypatch, capsys):
        rev = MagicMock()
        rev.commit_hash = "abcdef123456"
        repo = MagicMock()
        repo.repo_id = "owner/repo"
        repo.revisions = [rev]
        cache = MagicMock()
        cache.repos = [repo]
        cache.delete_revisions.return_value = MagicMock()
        monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache, raising=False)
        run_remove("owner/repo")
        out = capsys.readouterr().out
        assert "Removed owner/repo" in out

    def test_not_in_cache_exits(self, monkeypatch):
        cache = MagicMock()
        cache.repos = []
        monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache, raising=False)
        with pytest.raises(SystemExit) as ei:
            run_remove("owner/never_cached")
        assert ei.value.code == 1


class TestRunExport:
    def test_tensorrt_requires_module(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "torch_tensorrt", None)
        with pytest.raises(SystemExit) as ei:
            run_export("dummy", fmt="tensorrt", device="cpu")
        assert ei.value.code == 1

    def test_onnx_stub_message(self, monkeypatch, capsys):
        run_export("dummy", fmt="onnx", device="cpu")
        out = capsys.readouterr().out
        assert "ONNX export not yet implemented" in out

    def test_unknown_fmt_exits(self):
        with pytest.raises(SystemExit) as ei:
            run_export("dummy", fmt="unsupported", device="cpu")
        assert ei.value.code == 1

    def test_tensorrt_message_when_available(self, monkeypatch, capsys):
        import sys
        import types

        fake = types.ModuleType("torch_tensorrt")
        monkeypatch.setitem(sys.modules, "torch_tensorrt", fake)
        run_export("dummy", fmt="tensorrt", device="cpu")
        out = capsys.readouterr().out
        assert "TensorRT export not yet implemented" in out
