r"""Tests for worldkernels/cli/serve.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from worldkernels.cli.commands.serve import base_url, enumerate_routes, run_serve
from worldkernels.config import ServerConfig
from worldkernels.serving.server import create_app


@pytest.fixture
def uvicorn_calls(monkeypatch):
    calls: list[dict] = []

    def fake_run(app, host, port, **kwargs):
        calls.append({"app": app, "host": host, "port": port, **kwargs})

    monkeypatch.setattr("uvicorn.run", fake_run)
    return calls


class TestRunServe:
    def test_without_model(self, uvicorn_calls):
        run_serve("0.0.0.0", 8000, 2, None, "cpu")
        assert uvicorn_calls[0]["host"] == "0.0.0.0"
        assert uvicorn_calls[0]["port"] == 8000

    def test_with_model_preloads(self, uvicorn_calls):
        with patch("worldkernels.engine.WorldEngine.load_model") as load_mock:
            run_serve("0.0.0.0", 8000, 2, None, "cpu", model="dummy")
            load_mock.assert_called_once()
            assert load_mock.call_args.args == ("dummy",)
        assert len(uvicorn_calls) == 1

    def test_with_model_kwargs(self, uvicorn_calls):
        with patch("worldkernels.engine.WorldEngine.load_model") as load_mock:
            run_serve(
                "0.0.0.0",
                8000,
                1,
                "k",
                "cpu",
                model="dummy",
                model_kwargs={"num_inference_steps": 5, "guidance_scale": 3.0},
            )
            load_mock.assert_called_once()
            assert load_mock.call_args.args == ("dummy",)
            assert load_mock.call_args.kwargs.get("num_inference_steps") == 5
            assert load_mock.call_args.kwargs.get("guidance_scale") == 3.0


class TestFrontendPlumbing:
    def test_uvicorn_kwargs(self, uvicorn_calls):
        run_serve(
            "0.0.0.0",
            8000,
            1,
            None,
            "cpu",
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            uvicorn_log_level="warning",
            disable_access_log=True,
        )
        call = uvicorn_calls[0]
        assert call["ssl_keyfile"] == "key.pem"
        assert call["ssl_certfile"] == "cert.pem"
        assert call["log_level"] == "warning"
        assert call["access_log"] is False

    def test_uvicorn_defaults(self, uvicorn_calls):
        run_serve("0.0.0.0", 8000, 1, None, "cpu")
        call = uvicorn_calls[0]
        assert call["log_level"] == "info"
        assert call["access_log"] is True
        assert call["ssl_keyfile"] is None

    def test_server_config_fields(self, uvicorn_calls, monkeypatch):
        import worldkernels.serving.server as server_mod

        seen: dict = {}
        real = server_mod.create_app

        def spy(cfg, **kwargs):
            seen["cfg"] = cfg
            return real(cfg, **kwargs)

        monkeypatch.setattr(server_mod, "create_app", spy)
        run_serve(
            "0.0.0.0",
            8000,
            1,
            "tok",
            "cpu",
            allowed_origins=["http://a"],
            allowed_methods=["GET"],
            allowed_headers=["X-Custom"],
            allow_credentials=True,
            root_path="/api",
        )
        cfg = seen["cfg"]
        assert cfg.allowed_origins == ["http://a"]
        assert cfg.allowed_methods == ["GET"]
        assert cfg.allowed_headers == ["X-Custom"]
        assert cfg.allow_credentials is True
        assert cfg.root_path == "/api"
        assert cfg.api_key == "tok"


class TestStartupOutput:
    def test_plain_startup_sequence(self, uvicorn_calls, capsys):
        run_serve("127.0.0.1", 8001, 1, None, "cpu")
        out = capsys.readouterr().out
        assert "worldkernels v" in out
        assert "Available routes are:" in out
        assert "Route: /health" in out
        assert "/metrics" in out

    def test_shutdown_logs(self, uvicorn_calls, capsys):
        run_serve("127.0.0.1", 8001, 1, None, "cpu")
        out = capsys.readouterr().out
        assert "[shutdown] closing 0 session(s)" in out
        assert "[shutdown] engine released" in out

    def test_base_url_https_with_cert(self):
        cfg = ServerConfig(host="h", port=1234, ssl_certfile="c.pem", root_path="/api")
        assert base_url(cfg) == "https://h:1234/api"

    def test_base_url_http_default(self):
        assert base_url(ServerConfig()) == "http://0.0.0.0:8000"


class TestEnumerateRoutes:
    def test_core_routes_present(self):
        routes = dict(enumerate_routes(create_app(device="cpu")))
        assert "GET" in routes["/health"]
        assert "GET" in routes["/metrics"]

    def test_included_router_surface_is_enumerated(self):
        r"""Guards against FastAPI keeping included routers nested: the whole
        ``/v1`` surface must be listed, not just the app-level routes."""
        routes = enumerate_routes(create_app(device="cpu"))
        paths = {p for p, _ in routes}
        assert {"/v1/worlds", "/v1/sessions", "/v1/sessions/{session_id}/step"} <= paths
        assert ("/v1/sessions", ["POST"]) in routes

    def test_websocket_route_labeled(self):
        routes = enumerate_routes(create_app(device="cpu"))
        ws = [(p, m) for p, m in routes if m == ["WEBSOCKET"]]
        assert ws == [("/v1/sessions/{session_id}/stream", ["WEBSOCKET"])]

    def test_no_empty_paths(self):
        assert all(p for p, _ in enumerate_routes(create_app(device="cpu")))
