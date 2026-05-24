r"""Tests for worldkernels/cli/serve.py."""

from __future__ import annotations

from unittest.mock import patch

from worldkernels.cli.serve import run_serve


class TestRunServe:
    def test_without_model(self, monkeypatch):
        captured = {}

        def fake_uvicorn_run(app, host, port):
            captured.update(host=host, port=port)

        monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
        run_serve("0.0.0.0", 8000, 2, None, "cpu")
        assert captured == {"host": "0.0.0.0", "port": 8000}

    def test_with_model_preloads(self, monkeypatch, capsys):
        called = {}
        monkeypatch.setattr("uvicorn.run", lambda *a, **kw: called.setdefault("uvicorn", True))

        with patch("worldkernels.core.engine.WorldKernel.load_model") as load_mock:
            run_serve("0.0.0.0", 8000, 2, None, "cpu", "dummy", {})
            load_mock.assert_called_once_with("dummy")

        out = capsys.readouterr().out
        assert "Pre-loading model: dummy" in out
        assert called["uvicorn"] is True

    def test_with_model_kwargs(self, monkeypatch):
        monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)
        with patch("worldkernels.core.engine.WorldKernel.load_model") as load_mock:
            run_serve("0.0.0.0", 8000, 1, "k", "cpu", "dummy", {"variant": "v"})
            load_mock.assert_called_once_with("dummy", variant="v")
